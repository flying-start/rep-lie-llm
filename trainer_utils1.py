from __future__ import annotations

import os
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import evaluate
# from datasets import load_metric
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
import FLOP 


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
    elif data_name =='mnli':
        num_labels = 3
    elif data_name == 'stsb':
        num_labels = 1  # STSB是回归任务，应该有1个输出
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
            else:
                raise NotImplementedError
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
                problem_type="regression"
            )
        else:
            model = model_dispatcher[pretrained_model_name_or_path].from_pretrained(pretrained_model_name_or_path, num_labels=num_labels) 
    print(f"[DEBUG] 模型类别数: {model.config.num_labels}")
    return model


def prepare_traced_trainer_with_flop(model, args, data_content, hc_modules,target_sparsity,training_params={}, for_train_flag=True,  for_eval_flag=True,
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
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)

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

    trainer = nni.trace(FLOPTrainer)(model=model,
                                 args=training_args,
                                 data_collator=data_content['collator'],
                                 train_dataset=data_content[send_tag],
                                 eval_dataset=data_content['val'],
                                 tokenizer=data_content['tokenizer'],
                                 compute_metrics=compute_metrics,
                                 hc_modules = hc_modules,
                                 l1_lambda=training_params.get('l1_lambda', 1e-4),
                                 target_sparsity=target_sparsity,
                                                                                                                     
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
            config = AutoConfig.from_pretrained(args.arch, num_labels=3, problem_type="single_label_classification")
            model = model_dispatcher[args.arch].from_pretrained(args.arch, config=config)
            model.load_state_dict(torch.load(model_path).state_dict())
        # 针对STSB任务，确保加载模型时设置正确的num_labels
        elif args.data == 'stsb':
            config = AutoConfig.from_pretrained(args.arch, num_labels=1, problem_type="regression")
            model = model_dispatcher[args.arch].from_pretrained(args.arch, config=config)
            model.load_state_dict(torch.load(model_path).state_dict())
        else:
            model = torch.load(model_path)
    else:
        model = model_path

    trainer = prepare_traced_trainer_with_flop(model.to(args.device), args, data_content, {}, for_train_flag=False, tag=tag)

    output = trainer.predict(data_content[send_tag], metric_key_prefix=tag)

    print(f'Metric loss: {output.metrics}')
    return output

from transformers import Trainer
import torch
from typing import Dict, Union, Any

class FLOPTrainer(Trainer):
    def __init__(self, *args, hc_modules=None, l1_lambda=1e-4, target_sparsity=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.hc_modules = hc_modules  # HardConcrete 模块
        self.l1_lambda = l1_lambda    # L1 正则化系数
        self.target_sparsity = target_sparsity
        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0).cuda())  # 拉格朗日乘子
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0).cuda())
        self.optimizer_lagrangian = torch.optim.Adam([self.lambda_1, self.lambda_2], lr=1e-3)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs with FLOP pruning logic.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)  # 基础损失，例如交叉熵损失

        # FLOP 剪枝逻辑
        l1_regularization = sum(m().sum() for m in self.hc_modules)  # 计算 L1 正则化项
        total_loss = loss + self.l1_lambda * l1_regularization

        # 计算稀疏度相关损失

        hc_linear_modules = FLOP.get_hardconcrete_linear_modules(model)
        num_prunable_params = sum(m.num_prunable_parameters() for m in hc_linear_modules)
        expected_size = sum(m.num_parameters(train=True) for m in hc_linear_modules)
        expected_sparsity = 1.0 - expected_size / num_prunable_params
        lagrangian_loss = self.lambda_1 * (expected_sparsity - self.target_sparsity) + \
                          self.lambda_2 * (expected_sparsity - self.target_sparsity) ** 2
        total_loss += lagrangian_loss

        # 反向传播
        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()  # 多 GPU 平均化损失

        if self.do_grad_scaling:
            self.scaler.scale(total_loss).backward()
        elif self.use_apex:
            with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(total_loss)

        # 更新拉格朗日乘子优化器
        self.optimizer_lagrangian.step()
        self.optimizer_lagrangian.zero_grad()

        return total_loss


# from FLOP.utils import get_hardconcrete_modules

# def prepare_traced_trainer_with_flop(model, args, data_content, training_params={}, target_sparsity=0.8):
#     # 获取 HardConcrete 模块
#     hc_modules = get_hardconcrete_modules(model)

#     # 训练参数
#     training_args = TrainingArguments(
#         output_dir='./results',
#         evaluation_strategy='epoch',
#         save_strategy='epoch',
#         logging_strategy='epoch',
#         per_device_train_batch_size=training_params.get('batch_size', 32),
#         per_device_eval_batch_size=32,
#         num_train_epochs=training_params.get('num_train_epochs', 3),
#         learning_rate=training_params.get('learning_rate', 1e-4),
#         weight_decay=training_params.get('weight_decay', 1e-2),
#     )

#     # 初始化自定义 Trainer
#     trainer = FLOPTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=data_content['train'],
#         eval_dataset=data_content['val'],
#         tokenizer=data_content['tokenizer'],
#         data_collator=data_content['collator'],
#         hc_modules=hc_modules,
#         l1_lambda=training_params.get('l1_lambda', 1e-4),
#         target_sparsity=target_sparsity,
#     )

#     return trainer
