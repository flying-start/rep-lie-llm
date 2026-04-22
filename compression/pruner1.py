import logging
from copy import deepcopy

from compression.speedup import speedup
from config_helpers import *
from general_utils import LogLevel
from nni.contrib.compression.pruning import TaylorPruner,TaylorPruner_change,TaylorPrunerWithDelta,TaylorPrunerWithDelta1
from nni.contrib.compression.utils import TransformersEvaluator
from trainer_utils import *
from utils import get_model_param_keys
from test2 import adjust_delta_weights_with_masks
pruner_dispatcher = {'taylor': TaylorPruner,'taylor_delta': TaylorPrunerWithDelta1}


def update_full_model(model, args, config, trainer_state, total_num_steps):
    init_model = torch.load(get_path(args, 'INIT_MODEL_PATH'), map_location='cpu')
    init_masks = torch.load(get_path(args, 'INIT_MASKS_PATH'), map_location='cpu')
    opt_found_flag = False
    keys = get_model_param_keys(model)
    keys = keys[0] + keys[1]

    if os.path.exists(get_path(args, 'OPT_STATE_PATH')):
        opt_states = torch.load(get_path(args, 'OPT_STATE_PATH'), map_location='cpu')
        opt_found_flag = True
    else:
        opt_states = dict([(i, {'step': 0}) for i in range(len(trainer_state.opt_state))])
    model = model.to('cpu')

    if args.mask_finetune_flag:
        iter_masks = torch.load(get_path(args, 'ITER_MASKS_PATH'), map_location='cpu')
    else:
        iter_masks = None

    init_model_state_dict = init_model.state_dict()
    model_state_dict = model.state_dict()

    for key, val in model_state_dict.items():
        key_ = '.'.join(key.split('.')[:-1])
        _key = key.split('.')[-1]

        if key not in keys:
            continue

        opt_idx = keys.index(key)

        if 'embeddings.mask_token' in key:
            continue

        try:
            init_mask = init_masks[key_][_key]
        except:
            # print(f'Could not find init mask for {key}')
            init_mask = None

        if 'relative_position_bias_table' in key:
            init_mask = init_mask.repeat([model_state_dict[key].shape[0], 1])

        try:
            iter_mask = iter_masks[key_][_key]  # update these values
        except:
            # print(f'Could not find iter mask for {key}')
            iter_mask = torch.ones_like(model_state_dict[key]).bool()

        if init_mask is None:  # check this
            if init_model_state_dict[key].shape != model_state_dict[key].shape:
                # print(key)
                raise RuntimeError
            init_model_state_dict[key] = model_state_dict[key]

            if opt_found_flag:
                opt_states[opt_idx]['exp_avg'] = trainer_state.opt_state[opt_idx]['exp_avg'].to('cpu')
                opt_states[opt_idx]['exp_avg_sq'] = trainer_state.opt_state[opt_idx]['exp_avg_sq'].to('cpu')

        else:
            pad_idx = init_mask.flatten().nonzero().squeeze()[iter_mask.flatten() == 1]
            mask_padded = torch.zeros_like(init_mask).flatten()
            mask_padded[pad_idx] = 1
            mask_padded = mask_padded.reshape(init_mask.shape)

            try:
                init_model_state_dict[key][mask_padded] = model_state_dict[key][iter_mask].flatten()
            except:
                # print(f'Could not find update {key}')
                pass

            if opt_found_flag:
                try:
                    opt_states[opt_idx]['exp_avg'][mask_padded] = trainer_state.opt_state[opt_idx]['exp_avg'].to('cpu')[iter_mask].flatten()
                    opt_states[opt_idx]['exp_avg_sq'][mask_padded] = trainer_state.opt_state[opt_idx]['exp_avg_sq'].to('cpu')[iter_mask].flatten()
                    opt_states[opt_idx]['exp_avg'][~mask_padded] *= 0.9
                    opt_states[opt_idx]['exp_avg_sq'][~mask_padded] *= 0.999
                except:
                    print(key)

        opt_states[opt_idx]['step'] = int(trainer_state.opt_state[opt_idx]['step'].item() + opt_states[opt_idx]['step'])
        if not opt_found_flag:
            opt_states[opt_idx]['exp_avg'] = torch.zeros_like(init_model_state_dict[key])
            opt_states[opt_idx]['exp_avg_sq'] = torch.zeros_like(init_model_state_dict[key])

    init_model.load_state_dict(init_model_state_dict)
    torch.save(init_model, get_path(args, 'INIT_MODEL_PATH'))  # save the updated model
    torch.save(opt_states, get_path(args, 'OPT_STATE_PATH'))  # save the updated model

    return init_model


def init_pruning(model, args, config, data_content,tag='default', method=None, beta=-1):
    training_params = config.get_init_training_params(args.arch, args.data)
    pruning_params = config.get_init_pruning_params(args.arch, args.data)
    pruning_params['beta'] = beta

    full_masks = None
    cur_pruning_params = deepcopy(pruning_params)
    num_iters = pruning_params.get('num_iters', 1)# 1
    
    # print(f'num_iters:{num_iters}')
    for iter_idx in range(num_iters):
        cur_pruning_params['attn']['sparse_ratio'] = pruning_params['attn']['sparse_ratio'] \
                                                     / num_iters / (1 - iter_idx * pruning_params['attn']['sparse_ratio'] / num_iters)
        cur_pruning_params['ffn']['sparse_ratio'] = pruning_params['ffn']['sparse_ratio'] \
                                                    / num_iters / (1 - iter_idx * pruning_params['ffn']['sparse_ratio'] / num_iters)

        config_list = get_prune_config_for_attn(args, model, cur_pruning_params['attn']) \
                      + get_prune_config_for_ffn(args, model, cur_pruning_params['ffn'])
        # config_list = get_prune_config_for_qkv(args, model, cur_pruning_params['attn'])
        # print(f'cofig_list: {config_list}')

        for c in config_list:
            if not cur_pruning_params['global_flag']:
                del c['global_group_id']

        if method is None:
            method = 'taylor'

        if model.device != args.comp_device:
            # print(model.device, args.comp_device)
            model = model.to(args.comp_device)

        model, masks, _ = prune(model, args, data_content, training_params, cur_pruning_params,
                                config_list,method, tag=tag, device=args.comp_device)
        print('Pruning done')
        # print(f'mask:{masks}')
        if full_masks is None:
            full_masks = deepcopy(masks)
        else:
            for k in masks.keys():
                for k_ in masks[k].keys():
                    if full_masks[k][k_] is None:
                        continue
                    pad_idx = full_masks[k][k_].flatten().nonzero().squeeze()[masks[k][k_].flatten() == 1]
                    mask_padded = torch.zeros_like(full_masks[k][k_]).flatten()
                    mask_padded[pad_idx] = 1
                    mask_padded = mask_padded.reshape(full_masks[k][k_].shape)
                    full_masks[k][k_][mask_padded == 0] = False
        # print(f'full_masks:{full_masks}')
        if iter_idx == num_iters - 1:
            torch.save(model, get_path(args, 'COMPRESSED_MODEL_PATH'))
            torch.save(full_masks, get_path(args, 'INIT_MASKS_PATH'))

    return model

def iter_pruning(model, args, config, data_content, tag='default', method=None, sparsity_ratio_mul=0):
    training_params = config.get_iter_training_params(args.arch, args.data)
    pruning_params = config.get_iter_pruning_params(args.arch, args.data)
    init_pruning_params = config.get_init_pruning_params(args.arch, args.data)

    pruning_params['beta'] = 1
    cur_pruning_params = deepcopy(pruning_params)

    if sparsity_ratio_mul == 0:
        if cur_pruning_params['attn']['sparse_ratio'] < 0:
            cur_pruning_params['attn']['sparse_ratio'] *= -1
        if cur_pruning_params['ffn']['sparse_ratio'] < 0:
            cur_pruning_params['ffn']['sparse_ratio'] *= -1
    else:
        if cur_pruning_params['attn']['sparse_ratio'] < 0:
            cur_pruning_params['attn']['sparse_ratio'] *= -1
        else:
            cur_pruning_params['attn']['sparse_ratio'] += \
                (init_pruning_params['attn']['sparse_ratio'] -
                 pruning_params['attn']['sparse_ratio']) * sparsity_ratio_mul
        if cur_pruning_params['ffn']['sparse_ratio'] < 0:
            cur_pruning_params['ffn']['sparse_ratio'] *= -1
        else:
            cur_pruning_params['ffn']['sparse_ratio'] += \
                (init_pruning_params['ffn']['sparse_ratio'] -
                 pruning_params['ffn']['sparse_ratio']) * sparsity_ratio_mul

    config_list = get_prune_config_for_attn(args, model, cur_pruning_params['attn']) \
                  + get_prune_config_for_ffn(args, model, cur_pruning_params['ffn'])

    for c in config_list:
        if 'dependency_group_id' in c.keys():
            del c['dependency_group_id']
        if not cur_pruning_params['global_flag']:
            del c['global_group_id']

    if method is None:
        method = 'taylor'

    if model.device != args.comp_device:
        model = model.to(args.comp_device)

    model, masks, pruner = prune(model, args, data_content, training_params, cur_pruning_params,
                                 config_list, method, tag=tag, device=args.comp_device, speedup_flag=False)

    torch.save(masks, get_path(args, 'ITER_MASKS_PATH'))

    return model
def init_pruning_change(model, args, config, data_content,lora_data, tag='default', method=None, beta=-1):
    # 初始化训练和剪枝参数
    training_params = config.get_init_training_params(args.arch, args.data)
    pruning_params = config.get_init_pruning_params(args.arch, args.data)
    pruning_params['beta'] = beta

    total_steps = training_params.get('total_steps', 1000)  # 总训练步数
    warmup_steps = int(0.1 * total_steps)  # 预热阶段步数
    transition_steps = int(0.4 * total_steps)  # 过渡阶段步数
    initial_sparse_ratio = pruning_params['attn']['sparse_ratio']
    target_sparse_ratio = pruning_params['attn'].get('max_sparse_ratio', 1.0)
    prune_interval = pruning_params.get('prune_interval', 100)  # 剪枝间隔

    # 定义稀疏率调整函数
    def get_dynamic_sparse_ratio(step):
        if step < warmup_steps:
            return initial_sparse_ratio
        elif step < total_steps - transition_steps:
            return target_sparse_ratio + (initial_sparse_ratio - target_sparse_ratio) * \
                   ((total_steps - transition_steps - step) / (total_steps - transition_steps - warmup_steps)) ** 3
        else:
            return target_sparse_ratio

    # 初始剪枝配置
    full_masks = None
    print(f"Model mode: {model.training}")

    # 遍历总步数，动态剪枝
    for step in range(0, total_steps, prune_interval):
        # 动态计算稀疏率
        current_sparse_ratio = get_dynamic_sparse_ratio(step)
        cur_pruning_params = deepcopy(pruning_params)
        cur_pruning_params['attn']['sparse_ratio'] = current_sparse_ratio
        cur_pruning_params['ffn']['sparse_ratio'] = current_sparse_ratio

        # 动态生成剪枝配置
        config_list = get_prune_config_for_attn(args, model, cur_pruning_params['attn']) \
                      + get_prune_config_for_ffn(args, model, cur_pruning_params['ffn'])
        
        target_modules = extract_target_modules(config_list)
        for c in config_list:
            if not cur_pruning_params['global_flag']:
                del c['global_group_id']

        # 确定剪枝方法
        if method is None:
            method = 'taylor'

        # 确保模型在正确的设备上
        if model.device != args.comp_device:
            model = model.to(args.comp_device)

        # 执行剪枝
        model, masks, _ = prune_change(model, args, data_content, training_params, cur_pruning_params,lora_data,target_modules,
                                config_list, method, tag=tag, device=args.comp_device)
        if step % 200 == 0:
            print(f'Step {step}: Pruning done with sparse ratio {current_sparse_ratio:.4f}')

        # 合并剪枝掩码
        if full_masks is None:
            full_masks = deepcopy(masks)
        else:
            for k in masks.keys():
                for k_ in masks[k].keys():
                    if full_masks[k][k_] is None:
                        continue
                    pad_idx = full_masks[k][k_].flatten().nonzero().squeeze()[masks[k][k_].flatten() == 1]
                    mask_padded = torch.zeros_like(full_masks[k][k_]).flatten()
                    mask_padded[pad_idx] = 1
                    mask_padded = mask_padded.reshape(full_masks[k][k_].shape)
                    full_masks[k][k_][mask_padded == 0] = False

        # 保存最后一步的模型和掩码
        if step + prune_interval >= total_steps:
            torch.save(model, get_path(args, 'COMPRESSED_MODEL_PATH'))
            torch.save(full_masks, get_path(args, 'INIT_MASKS_PATH'))

    return model
def extract_target_modules(combined_config_list):
    """
    提取剪枝目标模块名称（target_modules）。

    Args:
        model: 待剪枝的模型实例。
        prune_params_dict: 剪枝参数字典，包含 sparse_ratio、granularity 等。
        args: 训练参数（包含模型架构信息）。

    Returns:
        target_modules (list): 剪枝目标模块的名称列表。
    """
    # 合并两种配置

    # 提取所有目标模块名称
    target_modules = set()
    for config in combined_config_list:
        # 提取 op_names
        if "op_names" in config:
            target_modules.update(config["op_names"])

        # 提取 op_names_re (正则表达式)
        if "op_names_re" in config:
            target_modules.update(config["op_names_re"])

    return list(target_modules)


def iter_pruning_change(model, args, config, data_content,lora_delta, tag='default', method=None, sparsity_ratio_mul=0):
    training_params = config.get_iter_training_params(args.arch, args.data)
    pruning_params = config.get_iter_pruning_params(args.arch, args.data)
    init_pruning_params = config.get_init_pruning_params(args.arch, args.data)

    pruning_params['beta'] = 1
    cur_pruning_params = deepcopy(pruning_params)

    if sparsity_ratio_mul == 0:
        if cur_pruning_params['attn']['sparse_ratio'] < 0:
            cur_pruning_params['attn']['sparse_ratio'] *= -1
        if cur_pruning_params['ffn']['sparse_ratio'] < 0:
            cur_pruning_params['ffn']['sparse_ratio'] *= -1
    else:
        if cur_pruning_params['attn']['sparse_ratio'] < 0:
            cur_pruning_params['attn']['sparse_ratio'] *= -1
        else:
            cur_pruning_params['attn']['sparse_ratio'] += \
                (init_pruning_params['attn']['sparse_ratio'] -
                 pruning_params['attn']['sparse_ratio']) * sparsity_ratio_mul
        if cur_pruning_params['ffn']['sparse_ratio'] < 0:
            cur_pruning_params['ffn']['sparse_ratio'] *= -1
        else:
            cur_pruning_params['ffn']['sparse_ratio'] += \
                (init_pruning_params['ffn']['sparse_ratio'] -
                 pruning_params['ffn']['sparse_ratio']) * sparsity_ratio_mul

    config_list = get_prune_config_for_attn(args, model, cur_pruning_params['attn']) \
                  + get_prune_config_for_ffn(args, model, cur_pruning_params['ffn'])
    # config_list = get_prune_config_for_qkv(args, model, cur_pruning_params['attn'])
    for c in config_list:
        if 'dependency_group_id' in c.keys():
            del c['dependency_group_id']
        if not cur_pruning_params['global_flag']:
            del c['global_group_id']

    if method is None:
        method = 'taylor_delta'

    if model.device != args.comp_device:
        model = model.to(args.comp_device)
    # if os.path.exists(get_path(args, 'ITER_MASKS_PATH')):
    #     iter_mask = torch.load(get_path(args, 'ITER_MASKS_PATH'))
    #     lora_delta = adjust_delta_weights_with_masks(lora_delta,iter_mask)
   
    init_mask = torch.load(get_path(args, 'INIT_MASKS_PATH'))
    # print(f'init_mask: {init_mask}')
    if init_mask is not None :
        lora_delta = adjust_delta_weights_with_masks(lora_delta,init_mask)
        # 遍历 lora_delta 的每个元素
        # for name, param in lora_delta.items():
        #     print(f"Layer: {name}, Shape: {param.shape}")
    model, masks, pruner = prune_change(model, args, data_content, training_params, cur_pruning_params,
                                 lora_delta,config_list, method, tag=tag, device=args.comp_device, speedup_flag=False)
    
    # torch.save(model, get_path(args, 'COMPRESSED_MODEL_PATH'))   
    torch.save(masks, get_path(args, 'ITER_MASKS_PATH'))

    return model


# @profile
def prune(model, args, data_content, training_params, pruning_params, config_list, pruner_method,
          tag='default', device='cpu', speedup_flag=True):
    training_params = deepcopy(training_params)
    training_params['learning_rate'] = 0
    trainer = prepare_traced_trainer(model, args, data_content, training_params, for_train_flag=False,
                                     for_eval_flag=False, tag=tag, device=device, send_tag='train')
    evaluator = TransformersEvaluator(trainer)
    pruner_init_kwargs = {}
    pruner_compress_kwargs = {}
    if pruner_method == 'movement':
        pruner_init_kwargs = {'warmup_step': pruning_params['warmup_step'],
                              'cooldown_begin_step': pruning_params['cooldown_begin_step']}
        pruner_compress_kwargs = {'max_steps': pruning_params['cooldown_begin_step'],
                                  'max_epochs': training_params.get('num_train_epochs', 3)}
    elif pruner_method == 'taylor':
        pruner_init_kwargs = {'training_steps': pruning_params['training_steps'],
                            # 'beta': pruning_params['beta'],
                            #   'global_flag': pruning_params['global_flag']
                              }
               
    with LogLevel(logging.ERROR):
        pruner = pruner_dispatcher[pruner_method](model, config_list, evaluator, **pruner_init_kwargs)       
        # 确保 evaluator 的 optimizer 不为 None
        pruner.compress(**pruner_compress_kwargs)
        # pruner.compress()
        pruner.unwrap_model()
        
    masks = pruner.get_masks()
    if speedup_flag:
        pruned_model = speedup(args, model.to('cpu'), masks)

    else:
        pruned_model = None

    return pruned_model, masks, pruner

def prune_change(model, args, data_content, training_params, pruning_params, lora_delta,target_modules,config_list, pruner_method,
          tag='default', device='cpu', speedup_flag=True):
    training_params = deepcopy(training_params)
    training_params['learning_rate'] = 0
    target_modules = get_target_modules(model)

    # 创建捕获输入的回调
    capture_callback = InputCaptureCallback(target_modules)

    trainer = prepare_traced_trainer_prune(model, args, data_content,training_params, for_train_flag=False, 
                                     for_eval_flag=False, tag=tag, device=device, send_tag='train', callbacks=[capture_callback])
    evaluator = TransformersEvaluator(trainer)
    pruner_init_kwargs = {}
    pruner_compress_kwargs = {}
    if pruner_method == 'movement':
        pruner_init_kwargs = {'warmup_step': pruning_params['warmup_step'],
                              'cooldown_begin_step': pruning_params['cooldown_begin_step']}
        pruner_compress_kwargs = {'max_steps': pruning_params['cooldown_begin_step'],
                                  'max_epochs': training_params.get('num_train_epochs', 3)}
    elif pruner_method == 'taylor':
        pruner_init_kwargs = {'training_steps': pruning_params['training_steps'],
                            # 'beta': pruning_params['beta'],
                            #   'global_flag': pruning_params['global_flag']
                              }
    elif pruner_method == 'taylor_delta':
        pruner_init_kwargs = {'training_steps': pruning_params['training_steps'],
                            # 'beta': pruning_params['beta'],
                            #   'global_flag': pruning_params['global_flag']
                              }
               
    with LogLevel(logging.ERROR):
        # pruner = TaylorPrunerWithDelta(model, config_list, evaluator, lora_delta, **pruner_init_kwargs)
        pruner = pruner_dispatcher['taylor_delta'](model, config_list, evaluator, lora_delta, **pruner_init_kwargs)       
        # 确保 evaluator 的 optimizer 不为 None
        pruner.compress(**pruner_compress_kwargs)
        # pruner.compress()
        pruner.unwrap_model()

        # save_dir = 'recap-main'
        # torch.save(pruner.importance_logs, os.path.join(save_dir, "importance_logs.pth"))
        
    masks = pruner.get_masks()
    if speedup_flag:
        pruned_model = speedup(args, model.to('cpu'), masks)

    else:
        pruned_model = None

    return pruned_model, masks, pruner

class InputCaptureCallback(TrainerCallback):
    def __init__(self, target_modules):
        super().__init__()
        self.hook = InputCaptureHook()
        self.target_modules = target_modules

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        self.hook.register_hooks(model, self.target_modules)

    def get_captured_inputs(self):
        return self.hook.inputs

import re

def get_target_modules(model):
    rules = [
        r"attention\.self\.(query|key|value)",  # Q/K/V
        r"attention\.output\.dense",           # 注意力输出
        r"intermediate\.dense",                # 前馈网络中间层
        r"output\.dense"                       # 前馈网络输出层
    ]
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(re.search(rule, name) for rule in rules):
                target_modules.append(name)
    return target_modules
