import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置环境变量避免数据集索引问题
os.environ['HF_DATASETS_CACHE'] = './cache/datasets'
os.environ['HF_DATASETS_OFFLINE'] = '1' 

import gc
import json
import math
import shutil
import subprocess
import time
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import torch.optim
from torch.utils.data import ConcatDataset
import transformers
import evaluate
from transformers import BertTokenizerFast, DataCollatorWithPadding, \
    ViTImageProcessor, Mask2FormerImageProcessor

import compression.pruner1 as compress_p
from compression.speedup import *
from args import arg_parser, modify_args
from config import *
from data_utils import prepare_datasets,avg_seq_length
from trainer_utils1 import *
from utils import get_model_param_keys
from finetune import *
from finetune_peft import *
from test2 import *
import peft
from peft.tuners.lora import *
from sparse_ratio import *
# 导入两个训练器，提供选择
from loraprune.trainer_FLOPs import LoRAPruneTrainer
import loraprune.utils as utils
from dataset_wrapper import SafeDatasetWrapper, StableDatasetWrapper
from vit_trainer_fix import ViTLoRAPruneTrainer

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from fvcore.nn import FlopCountAnalysis

def measure_flops(model, args):
    model.eval()
    device = next(model.parameters()).device
    batch_size = 16  # 减小batch size以节省内存
    
    # 根据任务类型创建不同的输入
    if args.task == 'img_class':
        # 为图像分类任务创建图像输入
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
        input_dict = {"pixel_values": dummy_input}
    else:
        # 为NLP任务创建token输入
        seq_length = 128
        dummy_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_length), dtype=torch.long, device=device)
        input_dict = {"input_ids": dummy_input}
    
    with torch.no_grad():
        try:
            flops = FlopCountAnalysis(model, input_dict).total()
        except Exception as e:
            print(f"[WARNING] FLOPs计算失败: {e}")
            print("[INFO] 使用默认FLOPs估计")
            # 为ViT-Base提供一个粗略的FLOPs估计
            if args.task == 'img_class':
                flops = 17.6e9 * batch_size  # ViT-Base大约17.6 GFLOPs per image
            else:
                flops = 22.0e9 * batch_size  # BERT-Base大约22 GFLOPs per sequence
    return flops

torch.cuda.empty_cache()
args = arg_parser.parse_args()
args = modify_args(args)
torch.manual_seed(args.seed)

tokenizer_dispatcher = {
    'bert-base-uncased': BertTokenizerFast,
    'bert-large-uncased': BertTokenizerFast,
    'vit-base': ViTImageProcessor,
    'vit-large': ViTImageProcessor,
    'vit-huge': ViTImageProcessor,
    'm2f': Mask2FormerImageProcessor
}
def get_vit_processor_path(arch):
    """根据ViT架构返回对应的预训练模型路径"""
    vit_model_mapping = {
        'vit-base': 'google/vit-base-patch16-224-in21k',
        'vit-large': 'google/vit-large-patch16-224',
        'vit-huge': 'google/vit-huge-patch14-224-in21k'
    }
    return vit_model_mapping.get(arch, 'google/vit-base-patch16-224-in21k')

def get_vit_config_for_arch(arch):
    """根据ViT架构返回相应的配置参数"""
    vit_configs = {
        'vit-base': {
            'patch_size': 16,
            'image_size': 224,
            'seq_len': 197,  # 196 patches + 1 CLS token
            'batch_size': 4,
            'gradient_accumulation_steps': 16,
            'learning_rate': 2e-4
        },
        'vit-large': {
            'patch_size': 16,
            'image_size': 224, 
            'seq_len': 197,  # 196 patches + 1 CLS token
            'batch_size': 2,      # 更小的batch size
            'gradient_accumulation_steps': 32,  # 更大的梯度累积
            'learning_rate': 1e-4  # 更小的学习率
        },
        'vit-huge': {
            'patch_size': 14,
            'image_size': 224,
            'seq_len': 257,  # 256 patches + 1 CLS token (14x14 patches for 224x224 image)
            'batch_size': 1,      # 极小的batch size
            'gradient_accumulation_steps': 64,  # 极大的梯度累积
            'learning_rate': 5e-5  # 极小的学习率
        }
    }
    return vit_configs.get(arch, vit_configs['vit-base'])

def prepare_data(args, eval_key):
    try:
        if 'vit' in args.arch:
            processor_path = get_vit_processor_path(args.arch)
            tokenizer = ViTImageProcessor.from_pretrained(processor_path, cache_dir='cache')
            print(f"[INFO] 使用 {args.arch} 对应的处理器: {processor_path}")
        elif 'm2f' in args.arch:
            tokenizer = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-IN21k-cityscapes-semantic", cache_dir='cache')
        else:
            tokenizer = tokenizer_dispatcher[args.arch].from_pretrained(args.arch, cache_dir='cache')
        
        train_dataset, validation_datasets, test_dataset = prepare_datasets(args.arch, args.task, args.data, tokenizer,
                                                                            args.data_root, eval_key)
        
        # 在包装之前保存标签信息
        label_info = None
        if args.task == 'img_class' and hasattr(train_dataset, 'features'):
            if args.data == 'cifar100':
                label_info = {
                    'id2label': {id: label for id, label in enumerate(train_dataset.features['fine_label'].names)},
                    'num_labels': len(train_dataset.features['fine_label'].names)
                }
            elif args.data == 'tiny-imagenet':
                label_info = {
                    'id2label': {id: f"class_{id}" for id in range(200)},  # TinyImageNet有200个类别
                    'num_labels': 200
                }
            else:
                label_info = {
                    'id2label': {id: label for id, label in enumerate(train_dataset.features['label'].names)},
                    'num_labels': len(train_dataset.features['label'].names)
                }
        
        # 验证数据集完整性
        print(f"[INFO] 训练集大小: {len(train_dataset)}")
        if hasattr(validation_datasets, '__len__'):
            print(f"[INFO] 验证集大小: {len(validation_datasets)}")
        print(f"[INFO] 测试集大小: {len(test_dataset)}")
        
    except Exception as e:
        print(f"[ERROR] 数据加载失败: {e}")
        raise e

    dtype = torch.float32

    if args.task == 'img_class':
        def collate_fn_cls(examples):
            try:
                # 过滤掉无效的样本
                valid_examples = [ex for ex in examples if ex is not None and "pixel_values" in ex]
                if not valid_examples:
                    print("[WARNING] 批次中没有有效样本，跳过...")
                    return None
                
                pixel_values = torch.stack([example["pixel_values"] for example in valid_examples])
                # 根据数据集选择正确的标签字段
                if args.data == 'cifar100':
                    labels = torch.tensor([example["fine_label"] for example in valid_examples])
                else:
                    # 适用于TinyImageNet和其他数据集
                    labels = torch.tensor([example["label"] for example in valid_examples])

                # 确保返回的batch只包含ViT模型需要的参数
                batch = {
                    "pixel_values": pixel_values.to(dtype), 
                    "labels": labels.long()
                }
                
                return batch
            except Exception as e:
                print(f"[ERROR] 数据collate失败: {e}")
                # 返回默认批次
                batch = {
                    "pixel_values": torch.zeros(1, 3, 224, 224).to(dtype), 
                    "labels": torch.zeros(1, dtype=torch.long)
                }
                return batch

        data_collator = collate_fn_cls
    elif args.task == 'img_seg':
        def collate_fn_seg(examples):
            data = []
            for key in examples[0].keys():
                if key == 'class_labels':
                    key_ = 'labels'
                else:
                    key_ = key

                if 'labels' in key:
                    val = [torch.tensor(np.stack(e[key], 0))[0] for e in examples]
                else:
                    val = np.concatenate([np.stack(e[key], 0) for e in examples])
                    val = torch.tensor(val).to(dtype)
                data.append((key_, val))
            return dict(data)

        data_collator = collate_fn_seg
    else:
        # 对于MNLI任务，特殊处理两个验证集
        if args.data == 'mnli':
            # 保存用于评估的验证集名称
            if 'validation_matched' in validation_datasets:
                args.validation_file = 'validation_matched'
                validation_datasets = validation_datasets['validation_matched']
            elif 'validation_mismatched' in validation_datasets:
                args.validation_file = 'validation_mismatched'
                validation_datasets = validation_datasets['validation_mismatched']
        else:
            validation_datasets = ConcatDataset([d for d in validation_datasets.values()])
            
        data_collator = DataCollatorWithPadding(tokenizer)

    # 使用安全的数据集包装器
    print(f"[INFO] 使用安全数据集包装器...")
    if args.task == 'img_class':
        # 对于图像分类任务，使用预加载包装器（CIFAR100数据集不大）
        train_dataset = SafeDatasetWrapper(train_dataset, max_samples=45000)  # 限制训练样本数
        if hasattr(validation_datasets, '__len__'):
            validation_datasets = SafeDatasetWrapper(validation_datasets, max_samples=5000)
        test_dataset = SafeDatasetWrapper(test_dataset, max_samples=10000)
    else:
        # 对于其他任务，使用轻量包装器
        train_dataset = StableDatasetWrapper(train_dataset)
        validation_datasets = StableDatasetWrapper(validation_datasets)
        test_dataset = StableDatasetWrapper(test_dataset)
    
    return {'train': train_dataset, 'val': validation_datasets, 'test': test_dataset,
            'collator': data_collator, 'tokenizer': tokenizer, 'label_info': label_info}

def compute_metrics(p: EvalPrediction):
        try:
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
                    print(f"MNLI {args.validation_file} metrics: {result}")

            elif args.task == 'img_class':
                predictions, labels = p.predictions, p.label_ids
                if predictions is None or labels is None:
                    print("[WARNING] compute_metrics: predictions或labels为None")
                    return {"accuracy": 0.0}
                
                if len(predictions.shape) > 1:
                    predictions = np.argmax(predictions, axis=1)
                else:
                    print("[WARNING] compute_metrics: predictions形状异常")
                    return {"accuracy": 0.0}
                
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(predictions, labels)
                print(f"[INFO] 当前精度: {accuracy:.4f}")
                result = dict(accuracy=accuracy)

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
        except Exception as e:
            print(f"[ERROR] compute_metrics失败: {e}")
            return {"accuracy": 0.0}

# @profile
def execute_main(args):
    model_name = args.arch

    if os.path.exists(get_path(args, 'MAIN_FOLDER_DIR', temp=False)):
        shutil.rmtree(get_path(args, 'MAIN_FOLDER_DIR', temp=False))
    Path(get_path(args, 'TRAINER_FOLDER_DIR')).mkdir(exist_ok=True, parents=True)
    Path(get_path(args, 'MODEL_FOLDER_DIR')).mkdir(exist_ok=True, parents=True)

    with open(get_path(args, 'ARGS_PATH'), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    config = Config(args)
    data_content = prepare_data(args, 'val')

    if args.task == 'img_class':
        # 使用保存的标签信息或提供默认值
        if data_content['label_info'] is not None:
            id2label = data_content['label_info']['id2label']
            label2id = {label: id for id, label in id2label.items()}
        else:
            # 为不同数据集提供默认标签映射
            if args.data == 'cifar100':
                id2label = {id: f"class_{id}" for id in range(100)}
            elif args.data == 'tiny-imagenet':
                id2label = {id: f"class_{id}" for id in range(200)}  # TinyImageNet有200个类别
            else:
                id2label = {id: f"class_{id}" for id in range(10)}  # CIFAR10默认
            label2id = {label: id for id, label in id2label.items()}
        
        print(f"[INFO] 使用标签映射，类别数量: {len(id2label)}")
        model = build_model(model_name, args.task, args.data, id2label=id2label, label2id=label2id)
    else:
        model = build_model(model_name, args.task, args.data)

    original_flops = measure_flops(model, args)
    print(f"[INFO] 原始模型 FLOPs: {original_flops / 1e9:.2f} GFLOPs")

    # torch.save(model, get_path(args, 'INIT_MODEL_PATH'))
    print("Number of labels:", model.config.num_labels)
    total_num_steps = 0
    
    print(model)
    print('Initial prune starts...')
    model = model.to(args.device)
    
    training_params = deepcopy(config.get_init_training_params(args.arch, args.data))
    num_steps = min(int(training_params.get('num_train_epochs', 3) * len(data_content['train']) / training_params.get('batch_size', 8)) + 5, 10000)

    new_model = apply_lora_with_sparsity(model, r=args.rank)
   
    # 获取架构特定配置
    vit_config = get_vit_config_for_arch(args.arch) if 'vit' in args.arch else None
    
    # 根据架构调整训练参数
    if vit_config:
        batch_size = vit_config['batch_size']
        gradient_accumulation_steps = vit_config['gradient_accumulation_steps'] 
        learning_rate = vit_config['learning_rate']
        seq_len = vit_config['seq_len']
        print(f"[INFO] 使用 {args.arch} 架构特定配置:")
        print(f"       - Batch Size: {batch_size}")
        print(f"       - Gradient Accumulation: {gradient_accumulation_steps}")
        print(f"       - Learning Rate: {learning_rate}")
        print(f"       - Sequence Length: {seq_len}")
        print(f"       - 有效Batch Size: {batch_size * gradient_accumulation_steps}")
    else:
        # 非ViT架构使用原有配置
        batch_size = 4
        gradient_accumulation_steps = 16
        learning_rate = training_params.get('learning_rate', 2e-4)
        seq_len = avg_seq_length(args.data) if args.task=='glue' else 197

    # 使用修复后的ViT训练器
    trainer = ViTLoRAPruneTrainer(
        model=new_model,
        train_dataset=data_content['train'],
        eval_dataset=data_content['val'],
        compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,  # 使用架构特定的batch size
            per_device_eval_batch_size=batch_size,  
            gradient_accumulation_steps=gradient_accumulation_steps,  # 使用架构特定的梯度累积
            warmup_steps=0,
            max_steps=num_steps,  # 使用完整的训练步数
            logging_strategy="steps",
            logging_steps=20,
            learning_rate=learning_rate,  # 使用架构特定的学习率
            fp16=False, bf16=True,
            optim="adamw_torch",
            warmup_ratio=0.1,  # 添加warmup      
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,                                                                                
            output_dir=get_path(args, 'TRAINER_FOLDER_DIR')+ f'/runs/sb_prune',                          
            save_total_limit=1,
            metric_for_best_model=args.metric_name,
            load_best_model_at_end=True if args.method == 'one' else False,
            greater_is_better=True,
            disable_tqdm=False,  # 启用进度条便于调试
            group_by_length=False,
            dataloader_num_workers=0,  # 单线程数据加载
            dataloader_pin_memory=False,  
            dataloader_drop_last=True,  
            ignore_data_skip=True,  
            remove_unused_columns=False,  
            skip_memory_metrics=True,  # 跳过内存监控
            report_to=[],  # 禁用wandb等外部报告
            label_names=["labels"],  # 明确指定标签列名
            include_inputs_for_metrics=False,  # 不在metrics中包含输入
        ),
        data_collator=data_content['collator'],
        ratio=args.ratio,
        init_ratio=args.init_ratio,
        warmup_iters=args.warmup_iters,#0.1
        cooldown_iters=args.cooldown_iters,#0.1
        prune_freq=args.prune_freq,
        prune_metric=args.prune_metric,
        mac=args.mac,
        seq_len=seq_len,  # 使用架构特定的序列长度
        # 如果使用稳定性训练器，添加稳定性参数
        use_stability=getattr(args, 'use_stability', True),
        stability_components= getattr(args, 'stability_components', ['attention', 'ffn']),
        stability_weight=getattr(args, 'stability_weight', 0.4),
        stability_collection_ratio=getattr(args, 'stability_collection_ratio', 0.25),
    )
    
    # trainer.add_callback(LogToFileCallback(get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/prune/log.txt'))
    trainer.train()
    masks = trainer.masks
    torch.save(masks, get_path(args, 'TRAINER_FOLDER_DIR') + '/runs/sb_prune/masks.pth')
    utils.apply_masked_modules(trainer.model, masks)
    test_output = predict(trainer.model, args, data_content, tag=args.final_eval_split)
    final_model = trainer.model.merge_and_unload()
    final_out = predict(final_model, args, data_content, tag=args.final_eval_split)
    # pruned_model = speedup_bert_with_ffn_mask(args, trainer.model, masks)

    # pruned_flops = measure_flops(pruned_model, args)
    # print(f"[INFO] 剪枝后模型 FLOPs: {pruned_flops / 1e9:.2f} GFLOPs")
   
    # test_output = predict(trainer.model, args, data_content, tag=args.final_eval_split)
    # val_ouput = predict(pruned_model, args, data_content, tag=args.final_eval_split)
    # test_metric = test_output.metrics
    prediction_logger = PredictionLogCallback(get_path(args, 'TRAINER_FOLDER_DIR') + f'/runs/sb_prune/prediction_log.txt')
    prediction_logger.log_prediction('no_mask',final_out.metrics)
    prediction_logger.log_prediction('val_mask',test_output.metrics)
    # prediction_logger.log_prediction('pruned',val_ouput.metrics)

    # prediction_logger.log_prediction('pruned_flops',pruned_flops/1e9)
    prediction_logger.log_prediction('original_flops',original_flops/1e9)
    # prediction_logger.log_prediction('pruned_ratio',(original_flops-pruned_flops)/original_flops)

    print(f"[INFO] 原始模型 FLOPs: {original_flops / 1e9:.2f} GFLOPs")
    # print(f"[INFO] 剪枝后模型 FLOPs: {pruned_flops / 1e9:.2f} GFLOPs")
    # print(f"[INFO] 剪枝率: {(original_flops-pruned_flops)/original_flops:.2f}")
    
    
    # 生成稳定性报告（如果支持）
    if hasattr(trainer, 'generate_stability_report'):
        trainer.generate_stability_report()
    

from transformers import TrainerCallback

class LogToFileCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            with open(self.log_path, "a") as f:
                f.write(f"Step: {state.global_step}, Epoch: {state.epoch:.2f}, Metrics: {metrics}\n")


class PredictionLogCallback:
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log_prediction(self, model_name, metrics):
        with open(self.log_path, "a") as f:
            f.write(f"Model: {model_name}, Metrics: {metrics}\n")


def restore_original_forward(model):
    for name, module in model.named_modules():
        if hasattr(module, '_original_forward'):
            module.forward = module._original_forward  # 还原原始方法
            del module._original_forward               # 清理自定义属性


 


    
if __name__ == '__main__':

    run_mode = args.run_mode

    if run_mode == 'train':
        execute_main(args)
        # execute_main(args)
    elif run_mode == 'evaluate':
        model_path = args.evaluate_from
        data_content = prepare_data(args, args.final_eval_split)
        output_metric_dict = predict(model_path, args, data_content, tag='test')
    else:
        raise NotImplementedError
