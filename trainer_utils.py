from __future__ import annotations

import os
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
# from datasets import load_metric
import evaluate
from sklearn.metrics import accuracy_score
from torch.optim import *
from transformers import BertForSequenceClassification, AutoConfig
from transformers import EvalPrediction
from transformers import ViTForImageClassification
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.training_args import TrainingArguments
from thop import profile
import nni
from models.modeling_mask2former import Mask2FormerForUniversalSegmentation
from paths import get_path
from utils import get_model_param_keys

model_dispatcher = {
    'bert-base-uncased': BertForSequenceClassification,
    'bert-large-uncased': BertForSequenceClassification,
    # 'vit-base': ViTForImageClassification,
    # 'vit-large': ViTForImageClassification,
    # 'm2f': Mask2FormerForUniversalSegmentation
}


def build_model(pretrained_model_name_or_path: str, task_name: str, data_name: str, **kwargs):

    if data_name == 'cifar100':
        num_labels = 100
    elif data_name == 'tinyimagenet':
        num_labels = 200
    elif data_name == 'cityscapes' or data_name == 'kitti':
        num_labels = 19
    elif data_name == 'stsb':
        num_labels = 1  # STSB是回归任务，应该有1个输出
    elif data_name == 'mnli':
        num_labels = 3  # MNLI是三分类任务
    else:
        num_labels = 2

    if task_name == 'img_class':
        if 'vit' in pretrained_model_name_or_path:
            if pretrained_model_name_or_path == 'vit-base':
                model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                                  id2label=kwargs['id2label'],
                                                                  label2id=kwargs['label2id'], cache_dir='cache')
            elif pretrained_model_name_or_path == 'vit-large':
                model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224',
                                                                  id2label=kwargs['id2label'],
                                                                  label2id=kwargs['label2id'],
                                                                  ignore_mismatched_sizes=True, cache_dir='cache')
            elif pretrained_model_name_or_path == 'vit-huge':
                # 使用ViT-Huge模型，注意这个模型更大，需要更多资源
                model = ViTForImageClassification.from_pretrained('google/vit-huge-patch14-224-in21k',
                                                                  id2label=kwargs['id2label'],
                                                                  label2id=kwargs['label2id'],
                                                                  ignore_mismatched_sizes=True, cache_dir='cache')
            else:
                raise NotImplementedError(f"ViT架构 '{pretrained_model_name_or_path}' 尚未支持")
        else:
            raise NotImplementedError
    elif task_name == 'img_seg':
        if 'm2f' in pretrained_model_name_or_path:
            model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-IN21k-cityscapes-semantic", cache_dir='cache')
        else:
            raise NotImplementedError
    else:
        # 对于STSB任务，需要将问题设置为回归类型
        if data_name == 'stsb':
            model = model_dispatcher[pretrained_model_name_or_path].from_pretrained(
                pretrained_model_name_or_path, 
                num_labels=num_labels, 
                problem_type="regression", 
                cache_dir='cache'
            )
        else:
            model = model_dispatcher[pretrained_model_name_or_path].from_pretrained(pretrained_model_name_or_path, num_labels=num_labels, cache_dir='cache')
    return model


def prepare_traced_trainer(model, args, data_content, training_params={}, for_train_flag=True,  for_eval_flag=True,
                           tag='default', device=None, send_tag='train'):

    if 'img' in args.task:
        save_strategy = 'no' if 'prune' in tag else 'epoch'
        evaluation_strategy = 'no' if 'prune' in tag else 'epoch'
    else:
        save_strategy = 'no' if 'prune' in tag else 'epoch'
        evaluation_strategy = 'no' if 'prune' in tag else 'epoch'

    def compute_metrics(p: EvalPrediction):
        if args.task == 'glue':
            metric = evaluate.load('glue', args.data)

            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            
            # 处理不同任务类型的预测
            if args.data == 'stsb':
                # STSB是回归任务，不需要argmax
                preds = preds.squeeze()
            else:
                # 分类任务需要argmax
                preds = np.argmax(preds, axis=1)
                
            result = metric.compute(predictions=preds, references=p.label_ids)
            
            # 处理MNLI任务的验证集
            if args.data == 'mnli' and hasattr(args, 'validation_file'):
                prefix = "matched_" if "validation_matched" in args.validation_file else "mismatched_"
                result = {f"{prefix}{k}": v for k, v in result.items()}

        elif args.task == 'img_class':
            predictions, labels = p.predictions, p.label_ids
            predictions = np.argmax(predictions, axis=1)
            result = dict(accuracy=accuracy_score(predictions, labels))

        elif args.task == 'img_seg':
            predictions, labels = p.predictions, p.label_ids
            predictions = predictions.sum(0)
            pos = predictions.sum(1)
            res = predictions.sum(0)
            tp = np.diag(predictions)
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            score = IoU_array[pos + res - tp != 0].mean()
            result = dict(accuracy=score)
        else:
            raise NotImplementedError

        return result

    if tag == 'default':
        logging_dir = None
    else:
        logging_dir = get_path(args, 'TRAINER_FOLDER_DIR') + '/runs/' + tag

    if device is None:
        device = args.device

    if device == 'cpu':
        no_cuda = True
    else:
        no_cuda = False

    if for_train_flag and for_eval_flag and args.task == 'img_seg':
        for_eval_flag = False

    num_steps = min(int(training_params.get('num_train_epochs', 3) * len(data_content['train']) / training_params.get('batch_size', 8)) + 5, 10000)
    training_args = TrainingArguments(output_dir=get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/{tag}',
                                      do_train=for_train_flag,
                                      do_eval=for_eval_flag,
                                      evaluation_strategy=evaluation_strategy,
                                      save_strategy=save_strategy,
                                      logging_strategy='epoch',
                                      logging_dir=logging_dir,
                                      logging_steps=500,
                                      per_device_train_batch_size=training_params.get('batch_size', 32),
                                      per_device_eval_batch_size=32,
                                      max_steps=num_steps,
                                      weight_decay=training_params.get('weight_decay', 1e-2),
                                      lr_scheduler_type='linear',
                                      dataloader_num_workers=1,
                                      learning_rate=training_params.get('learning_rate', 1e-4),
                                      save_total_limit=1,
                                      metric_for_best_model=args.metric_name,
                                      load_best_model_at_end=True,
                                      greater_is_better=True,
                                      disable_tqdm=True,
                                      optim='adamw_torch',
                                      seed=1024,
                                      use_mps_device=device == 'mps',
                                      no_cuda=no_cuda,
                                      remove_unused_columns=False)

    trainer = nni.trace(Trainer)(model=model,
                                 args=training_args,
                                 data_collator=data_content['collator'],
                                 train_dataset=data_content[send_tag],
                                 eval_dataset=data_content['val'],
                                 tokenizer=data_content['tokenizer'],
                                 compute_metrics=compute_metrics
                                                                                                                     
                                 )

    return trainer

def prepare_traced_trainer_prune(model, args, data_content, target_modules, callbacks, training_params={}, for_train_flag=True,  for_eval_flag=True,
                           tag='default', device=None, send_tag='train'):

    if 'img' in args.task:
        save_strategy = 'no' if 'prune' in tag else 'epoch'
        evaluation_strategy = 'no' if 'prune' in tag else 'epoch'
    else:
        save_strategy = 'no' if 'prune' in tag else 'epoch'
        evaluation_strategy = 'no' if 'prune' in tag else 'epoch'

    def compute_metrics(p: EvalPrediction):
        if args.task == 'glue':
            metric = evaluate.load('glue', args.data)
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            
            # 处理不同任务类型的预测
            if args.data == 'stsb':
                # STSB是回归任务，不需要argmax
                preds = preds.squeeze()
            else:
                # 分类任务需要argmax
                preds = np.argmax(preds, axis=1)
                
            result = metric.compute(predictions=preds, references=p.label_ids)
            
            # 处理MNLI任务的验证集
            if args.data == 'mnli' and hasattr(args, 'validation_file'):
                prefix = "matched_" if "validation_matched" in args.validation_file else "mismatched_"
                result = {f"{prefix}{k}": v for k, v in result.items()}

        elif args.task == 'img_class':
            predictions, labels = p.predictions, p.label_ids
            predictions = np.argmax(predictions, axis=1)
            result = dict(accuracy=accuracy_score(predictions, labels))

        elif args.task == 'img_seg':
            predictions, labels = p.predictions, p.label_ids
            predictions = predictions.sum(0)
            pos = predictions.sum(1)
            res = predictions.sum(0)
            tp = np.diag(predictions)
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            score = IoU_array[pos + res - tp != 0].mean()
            result = dict(accuracy=score)
        else:
            raise NotImplementedError

        return result

    if tag == 'default':
        logging_dir = None
    else:
        logging_dir = get_path(args, 'TRAINER_FOLDER_DIR') + '/runs/' + tag

    if device is None:
        device = args.device

    if device == 'cpu':
        no_cuda = True
    else:
        no_cuda = False

    if for_train_flag and for_eval_flag and args.task == 'img_seg':
        for_eval_flag = False

    num_steps = min(int(training_params.get('num_train_epochs', 3) * len(data_content['train']) / training_params.get('batch_size', 8)) + 5, 10000)
    training_args = TrainingArguments(output_dir=get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/{tag}',
                                      do_train=for_train_flag,
                                      do_eval=for_eval_flag,
                                      evaluation_strategy=evaluation_strategy,
                                      save_strategy=save_strategy,
                                      logging_strategy='epoch',
                                      logging_dir=logging_dir,
                                      logging_steps=500,
                                      per_device_train_batch_size=training_params.get('batch_size', 32),
                                      per_device_eval_batch_size=32,
                                      max_steps=num_steps,
                                      weight_decay=training_params.get('weight_decay', 1e-2),
                                      lr_scheduler_type='linear',
                                      dataloader_num_workers=1,
                                      learning_rate=training_params.get('learning_rate', 1e-4),
                                      save_total_limit=1,
                                      metric_for_best_model=args.metric_name,
                                      load_best_model_at_end=True,
                                      greater_is_better=True,
                                      disable_tqdm=True,
                                      optim='adamw_torch',
                                      seed=1024,
                                      use_mps_device=device == 'mps',
                                      no_cuda=no_cuda,
                                      remove_unused_columns=False)

    trainer = nni.trace(Trainer)(model=model,
                                 args=training_args,
                                 data_collator=data_content['collator'],
                                 train_dataset=data_content[send_tag],
                                 eval_dataset=data_content['val'],
                                 tokenizer=data_content['tokenizer'],
                                 compute_metrics=compute_metrics,
                                 callbacks = callbacks # 添加捕获输入的回调                                                             
                                 )

    return trainer

def predict(model_path, args, data_content, tag='default'):
    # if not Path(model_path).exists():
    #     print(f'Model does not exist at {model_path}, exiting...')
    #     return {}

    if args.task == 'img_class' and tag == 'test':
        send_tag = 'test'
    else:
        send_tag = 'val'

    # 如果是字符串路径，加载模型，否则直接使用模型对象
    if isinstance(model_path, str):
        # 针对MNLI任务，确保加载模型时设置正确的num_labels
        if args.data == 'mnli':
            config = AutoConfig.from_pretrained(args.arch, num_labels=3, problem_type="single_label_classification", cache_dir='cache')
            model = model_dispatcher[args.arch].from_pretrained(args.arch, config=config, cache_dir='cache')
            model.load_state_dict(torch.load(model_path).state_dict())
        # 针对STSB任务，确保加载模型时设置正确的num_labels
        elif args.data == 'stsb':
            config = AutoConfig.from_pretrained(args.arch, num_labels=1, problem_type="regression", cache_dir='cache')
            model = model_dispatcher[args.arch].from_pretrained(args.arch, config=config, cache_dir='cache')
            model.load_state_dict(torch.load(model_path).state_dict())
        else:
            model = torch.load(model_path)
    else:
        model = model_path.to(args.device)

    trainer = prepare_traced_trainer(model.to(args.device), args, data_content, {}, for_train_flag=False, tag=tag)

    output = trainer.predict(data_content[send_tag], metric_key_prefix=tag)

    print(f'Metric loss: {output.metrics}')
    return output


def prepare_masked_trainer(args, trainer, max_steps, decay_zero=True):
    trainer.create_optimizer_and_scheduler(num_training_steps=max_steps)

    if os.path.exists(get_path(args, 'INIT_MASKS_PATH')):
        masks = torch.load(get_path(args, 'INIT_MASKS_PATH'))
    else:
        masks = 1

    keys = get_model_param_keys(trainer.model)

    decay_parameters = get_parameter_names(trainer.model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    decay_val = 0 if decay_zero else trainer.args.weight_decay

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in trainer.model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": decay_val,
        },
        {
            "params": [
                p for n, p in trainer.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0,
        },
    ] 
    _, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(trainer.args)
    trainer.optimizer = CustomAdamW(keys, masks, optimizer_grouped_parameters, **optimizer_kwargs)


class CustomAdamW(AdamW):
    # 初始化函数，继承自AdamW类
    def __init__(self, keys, masks, args, **kwargs):
        super().__init__(args, **kwargs)
        # 保存keys和masks
        self.keys = keys
        self.masks = masks

    # 步进函数
    def step(self, closure=None):
        c = -1
        # 遍历param_groups
        for i in range(len(self.param_groups)):
            # 遍历params
            for j, param in enumerate(self.param_groups[i]['params']):
                c += 1
                # 获取key
                key = self.keys[i][j]

                # 获取key_和_key
                key_ = '.'.join(key.split('.')[:-1])
                _key = key.split('.')[-1]

                # 尝试获取mask
                try:
                    if isinstance(self.masks, dict):
                        mask = self.masks[key_][_key]
                    else:
                        continue
                except:
                    continue

                # 如果param.grad为空，则跳过
                if param.grad is None:
                    continue

                # 如果mask的形状与param.grad的形状不一致，则抛出异常
                if mask.shape != param.grad.shape:
                    print(key)
                    raise RuntimeError

                # 将param.grad乘以mask
                param.grad *= mask.to(param.device)

        # 调用父类的step函数
        super(CustomAdamW, self).step(closure)

def prepare_traced_self_trainer(model, args, data_content, training_params={}, for_train_flag=True, for_eval_flag=True,
                           tag='default', device=None, send_tag='train'):
   
    # 2. 自定义 compute_metrics 方法
    def compute_metrics(p: EvalPrediction):
        if args.task == 'glue':
            metric = evaluate.load('glue', args.data)
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            
            # 处理不同任务类型的预测
            if args.data == 'stsb':
                # STSB是回归任务，不需要argmax
                preds = preds.squeeze()
            else:
                # 分类任务需要argmax
                preds = np.argmax(preds, axis=1)
                
            result = metric.compute(predictions=preds, references=p.label_ids)
            
            # 处理MNLI任务的验证集
            if args.data == 'mnli' and hasattr(args, 'validation_file'):
                prefix = "matched_" if "validation_matched" in args.validation_file else "mismatched_"
                result = {f"{prefix}{k}": v for k, v in result.items()}

        elif args.task == 'img_class':
            predictions, labels = p.predictions, p.label_ids
            predictions = np.argmax(predictions, axis=1)
            result = dict(accuracy=accuracy_score(predictions, labels))

        elif args.task == 'img_seg':
            predictions, labels = p.predictions, p.label_ids
            predictions = predictions.sum(0)
            pos = predictions.sum(1)
            res = predictions.sum(0)
            tp = np.diag(predictions)
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            score = IoU_array[pos + res - tp != 0].mean()
            result = dict(accuracy=score)
        else:
            raise NotImplementedError

        return result

    # 3. 训练配置
    if tag == 'default':
        logging_dir = None
    else:
        logging_dir = get_path(args, 'TRAINER_FOLDER_DIR') + '/runs/' + tag

    if device is None:
        device = args.device

    if device == 'cpu':
        no_cuda = True
    else:
        no_cuda = False

    if for_train_flag and for_eval_flag and args.task == 'img_seg':
        for_eval_flag = False

    num_steps = min(
        int(training_params.get('num_train_epochs', 3) * len(data_content['train']) / training_params.get('batch_size', 8)) + 5,
        10000
    )

    # # 4. 创建自定义优化器
    # lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    # optimizer = torch.optim.AdamW(lora_params, lr=training_params.get('learning_rate', 1e-4))
    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/{tag}',
        do_train=for_train_flag,
        do_eval=for_eval_flag,
        evaluation_strategy='epoch' if for_eval_flag else 'no',
        save_strategy='epoch' if for_eval_flag else 'no',
        logging_strategy='epoch',
        logging_dir=logging_dir,
        logging_steps=500,
        per_device_train_batch_size=training_params.get('batch_size', 32),
        per_device_eval_batch_size=32,
        max_steps=num_steps,
        weight_decay=training_params.get('weight_decay', 1e-2),
        lr_scheduler_type='cosine',
        dataloader_num_workers=1,
        learning_rate=training_params.get('learning_rate', 1e-4),
        save_total_limit=1,
        metric_for_best_model=args.metric_name,
        load_best_model_at_end=True,
        greater_is_better=True,
        disable_tqdm=True,
        optim='adamw_torch',
        seed=1024,
        use_mps_device=device == 'mps',
        no_cuda=no_cuda,
        remove_unused_columns=False
    )

    # 6. 自定义 Trainer
    trainer = nni.trace(Trainer)(
        model=model,
        args=training_args,
        data_collator=data_content['collator'],
        train_dataset=data_content[send_tag],
        eval_dataset=data_content['val'],
        tokenizer=data_content['tokenizer'],
        compute_metrics=compute_metrics
    )

    return trainer

from transformers import TrainerCallback
import functools

class GradientSensitivityCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.sensitivities = {}
        self.final_gradients = {}
        # self.hook = InputCaptureHook()  # 初始化 Hook
        # self.target_modules = target_modules  # 目标模块名称列表
   
    def on_backward_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        print("on_backward_end")
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                grad_A = module.lora_A["default"].weight.grad
                grad_B = module.lora_B["default"].weight.grad
                if grad_A is not None and grad_B is not None:
                    self.final_gradients[name] = (grad_A.clone(), grad_B.clone())
                    print(f"Captured gradients for {name}")
    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        """
        在训练开始时注册钩子
        """
        pass
        # model = kwargs["model"]
        # self.hook.register_hooks(model, self.target_modules)
    def on_train_end(self, args, state, control, **kwargs):
        """
        在训练结束时计算敏感性分数。
        """
        model = kwargs["model"]
        print("Computing sensitivities...")

        for name, module in model.named_modules():
            if name in self.final_gradients:
                try:
                    A = module.lora_A["default"].weight
                    B = module.lora_B["default"].weight
                    print("has A and B",A)
                    grad_A, grad_B = self.final_gradients[name]
                    # 计算敏感性分数
                    sensitivity = grad_B @ A + B @ grad_A - grad_B @ grad_A
                    sensitivity = sensitivity.abs().sum().item()
                    self.sensitivities[name] = sensitivity
                    # 存储结果
                    name_without_prefix = name.replace("base_model.model.", "")
                    self.sensitivities[name_without_prefix] = sensitivity
                except Exception as e:
                    print(f"Error computing sensitivity for {name}: {e}")

        print("Final sensitivities:", self.sensitivities)

callback = GradientSensitivityCallback()
class InputCaptureHook:
    def __init__(self):
        self.inputs = {}  # 存储每个模块的输入张量

    def save_input(self, module, inputs, outputs, module_name):
        """
        捕获模块的输入张量并存储
        """
        self.inputs[module_name] = inputs[0]  # 默认保存第一个输入张量

    def register_hooks(self, model, target_modules):
        """
        为目标模块注册前向传播钩子
        Args:
            model: 要注册钩子的模型
            target_modules: 目标模块名称列表
        """
        for name, module in model.named_modules():
            if name in target_modules:  # 检查是否为目标模块
                module.register_forward_hook(
                    functools.partial(self.save_input, module_name=name)
                )




def get_trainer(model, args, data_content, training_params={}, for_train_flag=True, for_eval_flag=True,
                                tag='default', device=None, send_tag='train'):
    """
    为 LoRA 模型准备自定义 Trainer，支持动态稀疏化、LoRA 参数冻结和评估。
    """
    def compute_metrics(p: EvalPrediction):
        # 同原始代码
        if args.task == 'glue':
            metric = evaluate.load('glue', args.data)
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)            
            result = metric.compute(predictions=preds, references=p.label_ids)
            # save_results_to_file_append(preds, p.label_ids, result, tag="default")

        elif args.task == 'img_class':
            predictions, labels = p.predictions, p.label_ids
            predictions = np.argmax(predictions, axis=1)
            result = dict(accuracy=accuracy_score(predictions, labels))

        elif args.task == 'img_seg':
            predictions, labels = p.predictions, p.label_ids
            predictions = predictions.sum(0)
            pos = predictions.sum(1)
            res = predictions.sum(0)
            tp = np.diag(predictions)
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            score = IoU_array[pos + res - tp != 0].mean()
            result = dict(accuracy=score)
        else:
            raise NotImplementedError

        return result

    # 校验数据集键
    # required_keys = ['train', 'val', send_tag, 'collator', 'tokenizer']
    # for key in required_keys:
    #     if key not in data_content:
    #         raise ValueError(f"Missing required key in `data_content`: {key}")

    # 3. 训练配置
    if tag == 'default':
        logging_dir = None
    else:
        logging_dir = get_path(args, 'TRAINER_FOLDER_DIR') + '/runs/' + tag

    if device is None:
        device = args.device

    if device == 'cpu':
        no_cuda = True
    else:
        no_cuda = False

    if for_train_flag and for_eval_flag and args.task == 'img_seg':
        for_eval_flag = False

    num_steps = min(
        int(training_params.get('num_train_epochs', 3) * len(data_content['train']) / training_params.get('batch_size', 8)) + 5,
        10000
    )

    # 动态冻结非 LoRA 参数
    # for name, param in model.named_parameters():
    #     if "lora_" not in name:
    #         param.requires_grad = False

    # 自定义优化器
    # lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(lora_params, lr=training_params.get('learning_rate', 1e-4))

    # 自定义训练参数
    training_args = TrainingArguments(
        output_dir=get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/{tag}',
        do_train=for_train_flag,
        do_eval=for_eval_flag,
        evaluation_strategy='epoch' if for_eval_flag else 'no',
        save_strategy='epoch' if for_eval_flag else 'no',
        logging_strategy='epoch',
        logging_dir=logging_dir,
        logging_steps=500,
        max_steps=num_steps,
        per_device_train_batch_size=training_params.get('batch_size', 32),
        per_device_eval_batch_size=32,
        num_train_epochs=training_params.get('num_train_epochs', 3),
        weight_decay=training_params.get('weight_decay', 1e-2),
        learning_rate=training_params.get('learning_rate', 1e-4),
        save_total_limit=1,
        metric_for_best_model=args.metric_name,
        load_best_model_at_end=True,
        greater_is_better=True,
        disable_tqdm=True,
        optim='adamw_torch',
        seed=1024,
        use_mps_device=device == 'mps',
        no_cuda=no_cuda,
        remove_unused_columns=False
    )

    # 初始化自定义 Trainer
    trainer = nni.trace(SparseTrainer)(
        model=model,
        args=training_args,
        data_collator=data_content['collator'],
        train_dataset=data_content[send_tag],
        eval_dataset=data_content['val'],
        tokenizer=data_content['tokenizer'],
        compute_metrics=compute_metrics,
        # optimizers=(optimizer, None),
        l1_lambda=training_params.get('l1_lambda', 1e-4)  # 稀疏正则化系数
    )

    return trainer
class SparseTrainer(Trainer):
    def __init__(self, *args, l1_lambda=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1_lambda = l1_lambda  # L1 正则化系数

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        自定义损失函数，添加稀疏正则化（L1）项。
        """
        outputs = model(**inputs)
        loss = outputs.loss  # 基础损失（如交叉熵）
        
        # 稀疏正则化：对 LoRA 参数应用 L1 正则化
        sparse_loss = 0.0
        for name, param in model.named_parameters():
            if "lora_" in name:
                sparse_loss += torch.sum(torch.abs(param))    

        # 合并损失
        total_loss = loss + self.l1_lambda * sparse_loss
        # print(f"Total loss: {total_loss.item()}, Sparse loss: {sparse_loss.item()}")
        return (total_loss, outputs) if return_outputs else total_loss
class SensitivityTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.sensitivities = {}
        self.general_flops =0
    def training_step(self, model, inputs):
        # 前向传播
        model.train()
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        loss = outputs.loss

        # 反向传播
        loss.backward()

        # 累积 sensitivity
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                A = module.lora_A["default"].weight
                B = module.lora_B["default"].weight
                grad_A = module.lora_A["default"].weight.grad
                grad_B = module.lora_B["default"].weight.grad
                
                # 检查梯度是否存在
                if grad_A is None or grad_B is None:
                    continue
                
                # 计算 sensitivity
                sensitivity = grad_B @ A + B @ grad_A - grad_B @ grad_A
                sensitivity = sensitivity.abs().sum().item()
                name_without_prefix = name.replace("base_model.model.", "")
                
                self.sensitivities[name_without_prefix] = sensitivity

        return loss
def save_results_to_file_append(preds, labels, results, tag="default"):
    """
    追加保存验证结果到同一个文件
    :param preds: 预测值列表
    :param labels: 真实标签列表
    :param metrics: 计算的指标字典
    :param round_idx: 当前轮次
    :param tag: 文件名标识
    """
    output_dir = Path("./validation_results")
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / f"{tag}_validation_results.json"

    # 准备要追加的数据
    round_data = {
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
        "results": results
    }

    # 追加写入文件
    if file_path.exists():
        with open(file_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(round_data)

    with open(file_path, "w") as f:
        json.dump(all_results, f, indent=4,default=custom_serializer)

    # print(f"Validation results for round {round_idx} saved to {file_path}")


def custom_serializer(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()  # 尝试转换为字典
    return str(obj)  # 如果没有 to_dict 方法，则转换为字符串



