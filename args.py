import argparse
import datetime
import os
import time
from typing import List

def modify_args(args):
    if args.device == 'gpu' and args.gpu_idx:
        os.environ["CUDA_VISIBLE_DEVICES"] = 0

    args.datetime = format(str(datetime.datetime.now()))
    args.mask_finetune_flag = args.iter_sparse_ratio != 0

    if args.task == 'glue':
        if args.data == "cola":
            args.metric_name = "matthews_correlation"
        elif args.data == "mnli":
            args.metric_name = "matched_accuracy"
        elif args.data == "stsb":
            args.metric_name = "spearmanr"
        else:
            args.metric_name = "accuracy"
    elif args.task == 'img_class':
        args.metric_name = 'accuracy'
    elif args.task == 'img_seg':
        args.metric_name = 'accuracy'
    else:
        raise NotImplementedError

    if 'cifar' in args.data:
        args.final_eval_split = 'test'
    else:
        args.final_eval_split = 'val'

    return args


model_names = ['bert-base-uncased', 'bert-large-uncased', 'vit-base', 'vit-large', 'vit-huge', 'm2f']

arg_parser = argparse.ArgumentParser(description='Pruning main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save_path', default='output_sb', type=str, metavar='SAVE',
                       help='path to the experiment logging directory')
exp_group.add_argument('--evaluate_from', default='output/glue/cola/bert-base-uncased/0.33_-0.875/models/compressed_model.pth', type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--run_mode', default='train', type=str, choices=['train', 'evaluate'], help='Script mode')
exp_group.add_argument('--seed', default=0, type=int, help='random seed')
exp_group.add_argument('--gpu_idx', default=None, type=str, help='Index of available GPU')
exp_group.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'], help='Device type for finetuning')
exp_group.add_argument('--comp_device', default='cuda', type=str, choices=['cpu', 'cuda', 'mps'], help='Device type for pruning/masking operations')

# compression related
comp_group = arg_parser.add_argument_group('comp', 'compression setting')
comp_group.add_argument('--num_pruning_rounds', '-num_pr', default=10, type=int)
comp_group.add_argument('--core_res', '-res', default=64, type=float, help='Sparsity resolution')
comp_group.add_argument('--init_sparse_ratio', '-init_sparse', default=0.5, type=float, help='Pruning sparsity')
comp_group.add_argument('--iter_sparse_ratio', '-iter_sparse', default=-0.75, type=float, help='Finetuning sparsity')
comp_group.add_argument('--num_pruning_iters', '-num_pi', default=4, type=int, help='Gradually prune in x iters')
comp_group.add_argument('--ratio', '-ratio', default=0.5, type=float, help='Pruning ratio')
comp_group.add_argument('--mac', '-mac', default=0.5, type=float, help='mac ratio')
#mac constraint
comp_group.add_argument('--init_ratio', '-init_ratio', default=0.1, type=float, help='Pruning init_ratio')
comp_group.add_argument('--warmup_iters', '-warmup_iters', default=0.1, type=float, help='warmup_iters')
comp_group.add_argument('--cooldown_iters', '-cooldown_iters', default=0.1, type=float, help='cooldown_iters')
comp_group.add_argument('--prune_freq', '-prune_freq', default=100, type=float, help='Gradually prune in x step')
exp_group.add_argument('--prune_metric', default='lora', type=str, choices=['lora', 'grad', 'magnitude'], help='method type for pruning operations')
comp_group.add_argument('--rank', default=8, type=int, help='lora rank')
comp_group.add_argument('--method', default='gradual', type=str, help='lora rank')
# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--task', metavar='D', default='glue', choices=['glue', 'qa', 'img_class', 'img_seg'], help='task to work on')
data_group.add_argument('--data', metavar='D', default='cola', help='data to work on')
data_group.add_argument('--data_root', metavar='DIR', default='data', help='path to dataset folder (default: data)')
data_group.add_argument('-j', '--workers', default=1, type=int, help='number of data loading workers (default: 1)')

# 在args.py中
# 找到arg_parser定义的地方，添加以下参数

# 知识蒸馏相关参数
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
                      
# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='bert-base-uncased',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: bert-base-uncased)')
# 在args.py中添加
arg_parser.add_argument('--compare_pruning_methods', action='store_true',
                    help='是否比较三种剪枝方法的性能')

# 添加稳定性训练相关参数
stability_group = arg_parser.add_argument_group('stability', '稳定性剪枝设置')
stability_group.add_argument('--use_stability_trainer', action='store_true',
                       help='使用稳定性剪枝训练器')
stability_group.add_argument('--use_stability', action='store_true',default=True,
                       help='是否使用稳定性分数')
stability_group.add_argument('--stability_components', type=str, nargs='+', 
                       default=['attention'], choices=['attention', 'ffn'], 
                       help='应用稳定性分数的组件列表')
stability_group.add_argument('--stability_weight', type=float, default=0.3,
                       help='稳定性分数的权重')
stability_group.add_argument('--stability_collection_ratio', type=float, default=0.25,
                       help='在剪枝周期中用于收集稳定性的比例')