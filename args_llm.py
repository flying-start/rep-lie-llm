"""
args_llm.py
===========
在原 args.py 的基础上扩展，新增对 LLM（LLaMA-2、Mistral 等）的支持：
  - 新增 model_names 条目
  - 新增 task = 'llm_causal' 分支
  - 新增 LLM 专属 CLI 参数（model_path / calibration_nsamples / max_seq_length 等）
  - 修复原始 modify_args 中 task=='llm_causal' 缺失的分支
"""

import argparse
import datetime
import os
from typing import List


def modify_args(args):
    # ------------------------------------------------------------------ #
    # GPU 可见性设置（原始逻辑有 bug：应赋字符串而非整数 0）
    # ------------------------------------------------------------------ #
    if args.device == 'gpu' and args.gpu_idx:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)

    args.datetime = format(str(datetime.datetime.now()))
    args.mask_finetune_flag = args.iter_sparse_ratio != 0

    # ------------------------------------------------------------------ #
    # 任务类型 → 评估指标映射
    # ------------------------------------------------------------------ #
    if args.task == 'glue':
        if args.data == "cola":
            args.metric_name = "matthews_correlation"
        elif args.data == "mnli":
            args.metric_name = "matched_accuracy"
        elif args.data == "stsb":
            args.metric_name = "spearmanr"
        else:
            args.metric_name = "accuracy"
    elif args.task in ('img_class', 'img_seg'):
        args.metric_name = 'accuracy'
    elif args.task == 'llm_causal':
        # LLM 校准/剪枝阶段以 perplexity (PPL) 为主要指标
        args.metric_name = 'perplexity'
    else:
        raise NotImplementedError(f"Unknown task: {args.task}")

    # ------------------------------------------------------------------ #
    # 最终评估集键名
    # ------------------------------------------------------------------ #
    if 'cifar' in args.data:
        args.final_eval_split = 'test'
    elif args.task == 'llm_causal':
        args.final_eval_split = 'test'  # WikiText-2 test split
    else:
        args.final_eval_split = 'val'

    return args


# ====================================================================== #
# 支持的模型列表（新增 LLM 条目）
# ====================================================================== #
model_names = [
    # --- 原有 ---
    'bert-base-uncased',
    'bert-large-uncased',
    'vit-base',
    'vit-large',
    'vit-huge',
    'm2f',
    # --- 新增：LLM ---
    'llama2-7b',
    'llama2-13b',
    'llama3-8b',
    'mistral-7b',
    'qwen2-7b',
]

# ====================================================================== #
# ArgumentParser 构建
# ====================================================================== #
arg_parser = argparse.ArgumentParser(description='Pruning main script (with LLM support)')

# ------------------------------------------------------------------ #
# 实验基础参数
# ------------------------------------------------------------------ #
exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save_path', default='output_sb', type=str, metavar='SAVE',
                       help='path to the experiment logging directory')
exp_group.add_argument('--evaluate_from',
                       default='output/glue/cola/bert-base-uncased/0.33_-0.875/models/compressed_model.pth',
                       type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--run_mode', default='train', type=str, choices=['train', 'evaluate'],
                       help='Script mode')
exp_group.add_argument('--seed', default=0, type=int, help='random seed')
exp_group.add_argument('--gpu_idx', default=None, type=str, help='Index of available GPU')
exp_group.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'],
                       help='Device type for finetuning')
exp_group.add_argument('--comp_device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'],
                       help='Device type for pruning/masking operations')

# ------------------------------------------------------------------ #
# 剪枝压缩参数
# ------------------------------------------------------------------ #
comp_group = arg_parser.add_argument_group('comp', 'compression setting')
comp_group.add_argument('--num_pruning_rounds', '-num_pr', default=10, type=int)
comp_group.add_argument('--core_res', '-res', default=64, type=float, help='Sparsity resolution')
comp_group.add_argument('--init_sparse_ratio', '-init_sparse', default=0.5, type=float,
                        help='Pruning sparsity')
comp_group.add_argument('--iter_sparse_ratio', '-iter_sparse', default=-0.75, type=float,
                        help='Finetuning sparsity')
comp_group.add_argument('--num_pruning_iters', '-num_pi', default=4, type=int,
                        help='Gradually prune in x iters')
comp_group.add_argument('--ratio', '-ratio', default=0.5, type=float, help='Pruning ratio')
comp_group.add_argument('--mac', '-mac', default=0.5, type=float, help='mac ratio')
comp_group.add_argument('--init_ratio', '-init_ratio', default=0.1, type=float,
                        help='Pruning init_ratio')
comp_group.add_argument('--warmup_iters', '-warmup_iters', default=0.1, type=float,
                        help='warmup_iters')
comp_group.add_argument('--cooldown_iters', '-cooldown_iters', default=0.1, type=float,
                        help='cooldown_iters')
comp_group.add_argument('--prune_freq', '-prune_freq', default=100, type=float,
                        help='Gradually prune in x step')
exp_group.add_argument('--prune_metric', default='lora', type=str,
                       choices=['lora', 'grad', 'magnitude'],
                       help='method type for pruning operations')
comp_group.add_argument('--rank', default=8, type=int, help='lora rank')
comp_group.add_argument('--method', default='gradual', type=str, help='pruning method')

# ------------------------------------------------------------------ #
# 数据集参数
# ------------------------------------------------------------------ #
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--task', metavar='D', default='glue',
                        # 新增 llm_causal
                        choices=['glue', 'qa', 'img_class', 'img_seg', 'llm_causal'],
                        help='task to work on')
data_group.add_argument('--data', metavar='D', default='cola',
                        help='data to work on (e.g. cola / cifar100 / c4 for llm)')
data_group.add_argument('--data_root', metavar='DIR', default='data',
                        help='path to dataset folder (default: data)')
data_group.add_argument('-j', '--workers', default=1, type=int,
                        help='number of data loading workers (default: 1)')

# ------------------------------------------------------------------ #
# 模型架构参数
# ------------------------------------------------------------------ #
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='bert-base-uncased',
                        type=str, choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: bert-base-uncased)')

# ------------------------------------------------------------------ #
# LLM 专属参数（新增）
# ------------------------------------------------------------------ #
llm_group = arg_parser.add_argument_group('llm', 'LLM-specific settings')
llm_group.add_argument('--model_path', default=None, type=str,
                       help='本地或 Hub 模型路径，例如 /data/llama2-7b 或 meta-llama/Llama-2-7b-hf')
llm_group.add_argument('--calibration_nsamples', default=256, type=int,
                       help='校准集样本数（默认 256 条 2048-token 序列）')
llm_group.add_argument('--max_seq_length', default=2048, type=int,
                       help='LLM 最大序列长度')
llm_group.add_argument('--calibration_dataset', default='c4', type=str,
                       choices=['c4', 'wikitext2', 'ptb'],
                       help='校准数据集（默认 c4）')
llm_group.add_argument('--eval_ppl_dataset', default='wikitext2', type=str,
                       choices=['wikitext2', 'ptb', 'c4'],
                       help='PPL 评估数据集（默认 wikitext2）')
llm_group.add_argument('--use_flash_attn', action='store_true',
                       help='是否启用 Flash Attention 2（节省显存）')
llm_group.add_argument('--load_in_4bit', action='store_true',
                       help='使用 bitsandbytes 4-bit 量化加载（QLoRA 模式）')
llm_group.add_argument('--load_in_8bit', action='store_true',
                       help='使用 bitsandbytes 8-bit 量化加载')
llm_group.add_argument('--lm_eval_tasks', default=None, type=str, nargs='+',
                       help='lm-evaluation-harness zero-shot 任务列表，例如 winogrande arc_easy')

# ------------------------------------------------------------------ #
# 知识蒸馏参数（原始已有，保留）
# ------------------------------------------------------------------ #
arg_parser.add_argument('--distill_lambda', type=float, default=0.5,
                        help='Knowledge distillation loss weight')
arg_parser.add_argument('--temperature', type=float, default=2.0,
                        help='Distillation temperature')
arg_parser.add_argument('--distill_warmup_steps', type=int, default=1000,
                        help='Distillation warmup steps')
arg_parser.add_argument('--sparsity_lambda', type=float, default=1e-4,
                        help='Sparsity regularization weight')
arg_parser.add_argument('--structure_lambda', type=float, default=0.1,
                        help='Structure similarity loss weight')

# ------------------------------------------------------------------ #
# 剪枝方法比较开关（原始已有，保留）
# ------------------------------------------------------------------ #
arg_parser.add_argument('--compare_pruning_methods', action='store_true',
                        help='是否比较三种剪枝方法的性能')

# ------------------------------------------------------------------ #
# 稳定性训练参数（原始已有，保留）
# ------------------------------------------------------------------ #
stability_group = arg_parser.add_argument_group('stability', '稳定性剪枝设置')
stability_group.add_argument('--use_stability_trainer', action='store_true',
                             help='使用稳定性剪枝训练器')
stability_group.add_argument('--use_stability', action='store_true', default=True,
                             help='是否使用稳定性分数')
stability_group.add_argument('--stability_components', type=str, nargs='+',
                             default=['attention'], choices=['attention', 'ffn'],
                             help='应用稳定性分数的组件列表')
stability_group.add_argument('--stability_weight', type=float, default=0.3,
                             help='稳定性分数的权重')
stability_group.add_argument('--stability_collection_ratio', type=float, default=0.25,
                             help='在剪枝周期中用于收集稳定性的比例')
