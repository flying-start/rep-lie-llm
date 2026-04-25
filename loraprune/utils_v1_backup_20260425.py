import numpy as np
import torch
import torch.nn as nn
# from .lora import Linear, Linear8bitLt
from peft.tuners.lora import Linear,MaskedModuleWrapper

pruning_groups = {'self_attn': ['query', 'key', 'value', 'output.dense'],
                  'mlp': ['output.dense', 'intermediate.dense']}

# LLaMA/Mistral: head_dim=128; BERT/ViT: head_dim=64. 通过函数参数动态传入，不再硬编码全局变量
# DIM = 64 

def _is_target_larer(module):
    return isinstance(module, Linear)  
def _is_target_layer(module):
    return isinstance(module, nn.Linear)  

def unfreeze(model):
    for name, module in model.named_modules():
        if _is_target_larer(module):
            module.weight.requires_grad = True

def freeze(model):
    layers = len(model.model.model.layers)
    freeze_layer = int(layers * 0.1)
    for name, module in model.named_modules():
        if _is_target_larer(module):
            layer = int(name.split('.')[4])
            if layer < freeze_layer or layer == layers-1:
                module.is_prune = False

def init_sensitivity_dict(model):
    """初始化敏感度字典，确保掩码维度与层输出维度匹配。

    支持 BERT/ViT 和 LLaMA/Mistral/Qwen 架构：
      - BERT/ViT: 注意力层名为 ...attention.query/key/value/output.dense
      - LLaMA等: 注意力层名为 ...self_attn.q_proj/k_proj/v_proj/o_proj
    """
    sensitivity_record = {}
    # 动态获取 head_dim，兼容 BERT(64) 和 LLaMA(128)
    hidden_size = model.config.hidden_size
    num_attention_heads = getattr(model.config, 'num_attention_heads', None)
    head_dim = hidden_size // num_attention_heads if num_attention_heads else hidden_size // 12

    for name, module in model.named_modules():
        if not _is_target_larer(module):
            continue

        out_features = module.out_features if hasattr(module, 'out_features') else module.weight.shape[0]
        last_part = name.split('.')[-1]

        # ---------- 判断层类型（兼容 BERT + LLaMA） ----------
        # 注意力层判断：BERT 用 query/key/value/output.dense；LLaMA 用 q_proj/k_proj/v_proj/o_proj
        is_attn = last_part in ('query', 'key', 'value', 'output.dense') \
                  or last_part in ('q_proj', 'k_proj', 'v_proj', 'o_proj')

        # FFN 层判断：BERT 用 intermediate.dense/output.dense（无 attention 前缀）；
        #             LLaMA 用 gate_proj/up_proj/down_proj
        is_ffn = ('intermediate' in name and 'attention' not in name) \
                 or last_part in ('gate_proj', 'up_proj', 'down_proj')

        # 根据层类型设置掩码维度
        if is_attn:
            # 注意力头掩码：[num_heads]
            num_heads = out_features // head_dim
            mask = torch.ones(num_heads, requires_grad=False)
        else:
            # FFN 神经元 / 其他层掩码：[out_features]
            mask = torch.ones(out_features, requires_grad=False)

        # ---------- 统一 key：注意力层去尾缀，FFN 层保留完整名 ----------
        if is_attn:
            name = ".".join(name.split('.')[:-2])  # q_proj/k_proj/v_proj → model.layers.N.self_attn
        # else: 保留完整名（gate_proj / up_proj / down_proj / output.dense）

        sensitivity_record[name] = mask

    return sensitivity_record

def apply_masked_modules(model, mask_dict):
    """
    为模型的每个Linear层应用掩码。
    
    Args:
        model: 需要应用掩码的模型
        mask_dict: 包含每层掩码的字典
    """
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            if not hasattr(module, '_original_forward'): 
                module._original_forward = module.forward
            layer_name = ".".join(name.split('.')[:-1])  # 获取层名称
            
             # ⭐️ 特别注意此处: 通过参数硬绑定到闭包中，避免循环覆盖
            def make_masked_forward(orig_forward, layer_name):
                # 使用 `orig_forward` 作为默认参数传递，解决变量捕获问题
                def masked_forward(module, *inputs, orig_forward=orig_forward, layer_name=layer_name):
                    output = orig_forward(*inputs)
                    if layer_name in mask_dict:
                        mask = mask_dict[layer_name].to(output.device)
                        # print("Applying mask to layer:",layer_name)
                        # print("Applying mask to layer:", mask.sum(), mask.shape)
                        # 根据层的类型调整掩码形状
                        #  处理LoRA参数
                        # if hasattr(module, 'lora_A'):
                        #     # 对于lora_A，我们需要调整mask的形状以匹配输入维度
                        #     if 'attention' in layer_name:
                        #         # 对于注意力层，mask的形状是[num_heads]
                        #         # lora_A的形状是[r, in_features]
                        #         # 我们需要将mask扩展到输入维度
                        #         head_dim = module.weight.shape[1] // mask.size(0)
                        #         expanded_mask = mask.repeat_interleave(head_dim)  # [hidden_size]
                        #         module.lora_A['default'].weight.data *= expanded_mask.view(1, -1)  # [r, in_features]
                        #     else:
                        #         # 对于其他层，mask直接作用于输入维度
                        #         module.lora_A['default'].weight.data *= mask.view(1, -1)  # [r, in_features]
                        if 'self_attn' in layer_name or 'attention' in layer_name:
                            # s_name = ".".join(layer_name.split('.')[:-2])
                            mask = mask_dict[layer_name].to(output.device)
                            # 对于注意力层，调整掩码形状以匹配输出
                            batch_size = output.size(0)
                            seq_len = output.size(1)
                            head_dim = output.size(-1) // mask.size(0)
                            # print(f'Applying mask to {layer_name} with shape {mask.shape} and output shape {output.shape}')

                            # 将掩码扩展到正确的维度 [num_heads] -> [batch_size, seq_len, num_heads, head_dim]
                            mask = mask.view(-1, 1).repeat(1, head_dim)  # [num_heads, head_dim]
                            mask = mask.view(1, 1, -1).expand(batch_size, seq_len, -1)

                        elif 'mlp' in layer_name or 'output' in layer_name or 'intermediate' in layer_name:
                            # 对于 FFN 层（BERT: output.dense/intermediate.dense；LLaMA: mlp.gate_proj/mlp.up_proj/mlp.down_proj）
                            # 掩码形状：[out_features] -> [1, 1, out_features]
                            mask = mask.view(1, 1, -1)  # 变成 [1, 1, hidden_size] 或 [1, 1, ffn_size]
                        # 使用广播机制应用掩码
                        # print(f'Applying mask to {name} with shape {mask.shape} and output shape {output.shape}')
                        ## 只在真正 optimizer 更新时才应用
                        # if self.state.global_step % args.gradient_accumulation_steps == 0:
                        output = output * mask
                        
                    return output               
                return masked_forward            
            # 为每个模块创建一个新的闭包
            module.forward = make_masked_forward(module._original_forward, name).__get__(module)

# 注意：确保在训练开始前调用 apply_masked_modules 函数

def update_sensitivity_dict(model, s_dict, prune_metric):
    """
    更新敏感分数字典。

    参数:
        model: 模型。
        s_dict: 当前的敏感分数字典。
        prune_metric: 剪枝类型（如 'lora', 'magnitude', 'grad'）。
        mask_dict: 掩码字典，格式为 {'module_name': {'weight': weight_mask_tensor, 'bias': bias_mask_tensor}}。

    返回:
        更新后的敏感分数字典。
    """
    # 初始化新的敏感分数字典
    device = next(model.parameters()).device
    new_s_dict = init_sensitivity_dict(model)
    # print("new_s_dict", new_s_dict)
    # 动态获取 head_dim（兼容 BERT=64 / LLaMA=128）
    hidden_size = model.config.hidden_size
    num_attention_heads = getattr(model.config, 'num_attention_heads', None)
    head_dim = hidden_size // num_attention_heads if num_attention_heads else hidden_size // 12

    for name, module in model.named_modules():
        if not _is_target_larer(module):
            continue
        last_part = name.split('.')[-1]
        # 判断模块类型（兼容 BERT + LLaMA）
        is_attn = 'attention' in name or last_part in ('q_proj', 'k_proj', 'v_proj', 'o_proj')
        is_output = last_part == 'output.dense'
        intermediate = 'intermediate' in name and 'attention' not in name
        sensitivity = compute_sensitivity(module, is_attn, is_output, intermediate,
                                         prune_metric, head_dim=head_dim)
        if is_attn:
            name = ".".join(name.split('.')[:-2])
            # else:
            #     name = ".".join(name.split('.')[:-1])
            # 存储敏感分数
            new_s_dict[name] += sensitivity.to(new_s_dict[name].device)

    # 检查敏感分数是否包含 NaN
    if any(torch.isnan(imp.sum()) for imp in new_s_dict.values()):
        return s_dict  # 如果存在 NaN，返回原始敏感分数字典

    # 使用指数移动平均更新敏感分数字典
    for name, current_sensitivity in s_dict.items():
        if name in new_s_dict:
            
            s_dict[name] = current_sensitivity * 0.9 + new_s_dict[name] * 0.1
        # 如果模块不存在于 new_s_dict 中，保持原值不变

    return s_dict


def compute_sensitivity(layer, is_attn, is_output, intermediate,
                       prune_metric='lora', transpose=False, norm=True, head_dim=64):
    # grad = 0
    if prune_metric == 'lora':
        a = layer.lora_A['default'].weight.data
        b = layer.lora_B['default'].weight.data
        grad_a = layer.lora_A['default'].weight.grad
        grad_b = layer.lora_B['default'].weight.grad
        # grad = b @ a * layer.scaling['default']
        grad = (grad_b @ a + b @ grad_a - grad_b @ grad_a)
        weight = layer.weight.data
        s = (grad * (b @ a * layer.scaling['default'] + weight)).abs()
    elif prune_metric == 'magnitude':
        grad = 1
        s = layer.weight.data.abs()
    elif prune_metric == 'grad':
        grad = layer.weight.grad
        if grad is None:
            print(f"[WARNING] {layer.__class__.__name__} 的梯度为None，使用权重幅度作为备选")
            s = layer.weight.data.abs()
        else:
            s =(layer.weight.data * grad * grad).abs()
    else:
        raise NotImplementedError
    if hasattr(layer, 'state'):
        print("if hasattr(layer, 'state')")#False
        weight = (layer.weight.data * layer.state.SCB.reshape(-1, 1)) / 127
    else:
        weight = layer.weight.data
    # print(layer.scaling)
    
    if transpose:
        s = s.t()
    if is_attn:
        s = s.reshape(s.shape[0] // head_dim, -1)
    s=s.sum(1)
    if norm:
        s = s / (torch.linalg.norm(s) + 1e-8)
    return s

def prune_fp16_module(module, mask, transpose):
    mask = mask.bool()
    module.train()
    if not transpose:
        module.weight.data = module.weight.data[mask]
        module.out_features = int(mask.sum())
        if module.bias:
            module.bias.data = module.bias.data[mask]
        module.lora_B.weight.data = module.lora_B.weight.data[mask]
        module.lora_B.out_features = int(mask.sum())
    else:
        module.weight.data = module.weight.data[:, mask]
        module.in_features = int(mask.sum())
        module.lora_A.weight.data = module.lora_A.weight.data[:, mask]
        module.lora_A.in_features = int(mask.sum())
    module.merge_weights = True
    module.train(False)

def prune_one_layer(layer):
    ## self_attn
    prune_fp16_module(layer.self_attn.q_proj, layer.self_attn.q_proj.lora_mask, False)
    prune_fp16_module(layer.self_attn.k_proj, layer.self_attn.k_proj.lora_mask, False)
    prune_fp16_module(layer.self_attn.v_proj, layer.self_attn.v_proj.lora_mask, False)
    prune_fp16_module(layer.self_attn.o_proj, layer.self_attn.q_proj.lora_mask, True)
    layer.self_attn.num_heads = int(layer.self_attn.q_proj.lora_mask.sum()) // DIM
    layer.self_attn.hidden_size = int(layer.self_attn.q_proj.lora_mask.sum())

    ## mlp
    prune_fp16_module(layer.mlp.gate_proj, layer.mlp.gate_proj.lora_mask, False)
    prune_fp16_module(layer.mlp.up_proj, layer.mlp.up_proj.lora_mask, False)
    prune_fp16_module(layer.mlp.down_proj, layer.mlp.gate_proj.lora_mask, True)

    ## reset mask
    del(layer.self_attn.q_proj.lora_mask)
    del(layer.self_attn.k_proj.lora_mask)
    del(layer.self_attn.v_proj.lora_mask)
    del(layer.mlp.gate_proj.lora_mask)
    del(layer.mlp.up_proj.lora_mask)

def prune(model):
    for layer_id, layer in enumerate(model.model.model.layers):
        print("pruning layer {}".format(layer_id))
        prune_one_layer(layer)

# def local_prune(model, s_dict, ratio, target_ratio):
#     original_param_num = 0
#     pruned_param_num = 0
#     for name, module in model.named_modules():
#         if _is_target_larer(module):
#             original_param_num += np.prod(module.weight.shape)
#             pruned_param_num += np.prod(module.weight.shape) * ratio
#             is_attn = name.split('.')[-1] in pruning_groups['self_attn']
#             if name.split('.')[-1] in pruning_groups['block']:
#                 continue
#             name = ".".join(name.split('.')[:-1])
#             if not hasattr(module, 'lora_mask'):
#                 continue
#             if (1-module.lora_mask.mean()).item() >= target_ratio:
#                 continue
#             total_num = module.lora_mask.numel()
#             c_mask = module.lora_mask.data
#             mask = torch.ones_like(c_mask)

#             if is_attn:
#                 mask = mask.reshape(-1, DIM)[:, 0]
#                 c_mask = c_mask.reshape(-1, DIM)[:, 0]
#                 total_num /= DIM
#             need_prune_num = int(total_num * ratio)
#             importance = s_dict[name] * c_mask
#             can_prune = torch.argsort(importance)[:need_prune_num]
#             mask[can_prune] = 0
#             if is_attn:
#                 mask = (mask.new_ones(module.lora_mask.shape).reshape(-1, DIM) * mask.unsqueeze(1)).reshape(-1)
#             module.lora_mask.data = mask
#         else:
#             if hasattr(module, 'weight'):
#                 original_param_num += np.prod(module.weight.shape)
#     print("pruned/original parameters number:{:3f}/{:3f}  ratio:{:3f}".format(pruned_param_num*1e-9,
#                                                                                original_param_num*1e-9,
#                                                                                pruned_param_num/original_param_num))

def local_prune(model, s_dict, ratio, target_ratio, mask_dict=None):
    """
    根据重要性分数和剪枝率生成每个模块的剪枝掩码，并更新掩码状态。

    参数:
        model: nn.Module
            模型对象。
        s_dict: dict
            每个模块的重要性分数字典。
        ratio: float
            本次新增剪枝比例。
        target_ratio: float
            每个模块目标稀疏率。
        mask_dict: dict or None
            保存每个模块剪枝掩码的字典，默认为 None。

    返回:
        mask_dict: dict
            更新后的剪枝掩码字典。
    """
    if mask_dict is None:
        mask_dict = {}

    original_param_num = 0
    pruned_param_num = 0

    for name, module in model.named_modules():
        if _is_target_larer(module):
            # 统计参数数量
            original_param_num += module.weight.numel()

            # 检查是否是注意力层
            is_attn = name.split('.')[-1] in pruning_groups['self_attn']  or 'attention.output.dense' in name          
            # 初始化掩码
            if name not in mask_dict:
                mask_dict[name] = torch.ones(module.weight.shape[0], dtype=torch.float32, device=module.weight.device)

            # 获取当前掩码
            current_mask = mask_dict[name]
            mask = torch.ones_like(current_mask)
            current_sparsity = 1 - current_mask.mean().item()
            print(f"Current sparsity of {name}: {current_sparsity:.4f}")

            # 如果已经达到目标稀疏率，跳过
            if current_sparsity >= target_ratio:
                continue

            # 计算需要剪枝的数量
            total_num = current_mask.numel()
            if is_attn:
                mask =mask.reshape(-1, DIM)[:, 0]
                current_mask = current_mask.reshape(-1, DIM)[:, 0]
                total_num //= DIM
            need_prune_num = int(total_num * ratio)

            # 根据当前掩码计算有效重要性
            if is_attn:
                s_name = ".".join(name.split('.')[:-2])
            else:
                s_name = ".".join(name.split('.')[:-1])
            effective_importance = s_dict[s_name] * current_mask.to(s_dict[s_name].device)
            can_prune = torch.argsort(effective_importance)[:need_prune_num]
            mask[can_prune] = 0

            if is_attn:
                mask = (mask.new_ones(module.weight.shape[0]).reshape(-1, DIM) * mask.unsqueeze(1)).reshape(-1)
            mask_dict[name] = mask
            # 更新剪枝统计
            pruned_param_num += mask.sum().item() 
            print(f"Current mask: {current_mask.numel()}")
            print(f"Mask after pruning: {mask.sum().item()}")

        else:
            if hasattr(module, 'weight'):
                original_param_num += module.weight.numel()

    # 打印剪枝统计信息
    print("Pruned/original parameters number: {:.3f}B/{:.3f}B  Ratio: {:.3f}".format(
        pruned_param_num * 1e-9,
        original_param_num * 1e-9,
        original_param_num-pruned_param_num / original_param_num
    ))

    return mask_dict

import torch
import itertools
import torch
import itertools


def global_prune(model, s_dict, ratio, target_ratio, mask_dict=None):
    """
    全局剪枝 Transformer 模型的注意力头和 FFN 参数，分别计算全局剪枝阈值。
    
    Args:
        model: nn.Module, 目标 Transformer 模型。
        s_dict: dict, 重要性分数字典 (需要提前标准化)。
        ratio: float, 当前轮次的剪枝比例。
        target_ratio: float, 目标剪枝比例。
        mask_dict: dict or None, 剪枝掩码字典。

    Returns:
        dict: 更新后的剪枝掩码字典。
    """
    if mask_dict is None:
        mask_dict = {}

    attn_scores = []  # 存储注意力层的重要性分数
    ffn_scores = []   # 存储 FFN 层的重要性分数
    attn_info = []    # 记录注意力层模块信息
    ffn_info = []     # 记录 FFN 层模块信息
    all_masks = []    # 记录所有剪枝掩码

    for name, module in model.named_modules():
        if not _is_target_larer(module):
            continue
        
        is_attn = name.split('.')[-1] in pruning_groups['self_attn'] or 'attention.output.dense' in name  

        if name not in mask_dict:
            mask_dict[name] = torch.ones(module.weight.shape[0], dtype=torch.float32, device=module.weight.device)

        current_mask = mask_dict[name]
        all_masks.append(current_mask)  # 收集所有 mask 计算全局稀疏率

        s_name = ".".join(name.split('.')[:-1])
        
        if s_name not in s_dict:
            print(f"Warning: {s_name} not found in s_dict, skipping pruning for {name}")
            continue

        score = 0.0

        if is_attn:
            head_importances = s_dict[s_name]

            if head_importances.numel() == 0:
                print(f"Warning: {name} has empty importance scores in s_dict, skipping")
                continue

            if current_mask.numel() % DIM == 0:
                valid_heads = current_mask.reshape(-1, DIM).any(dim=1)
            else:
                print(f"Warning: {name} mask shape mismatch, using default valid_heads")
                valid_heads = torch.ones_like(head_importances, dtype=torch.bool)

            if valid_heads.sum() == 0:
                print(f"Warning: {name} all valid_heads are pruned, setting default valid_heads")
                valid_heads = torch.ones_like(head_importances, dtype=torch.bool)

            score = head_importances * valid_heads.to(head_importances.device).float() + 1e-6
            attn_info.append((name, len(score), current_mask, 'head'))
            # attn_info.append((name, len(head_importances), current_mask, 'head'))
            attn_scores.append(score)

        else:
            param_scores = s_dict[s_name].flatten()

            if param_scores.numel() == 0:
                print(f"Warning: {name} has empty FFN scores in s_dict, skipping")
                continue

            valid_params = current_mask.flatten().bool()

            if valid_params.sum() == 0:
                print(f"Warning: {name} all valid_params are pruned, setting default valid_params")
                valid_params = torch.ones_like(param_scores, dtype=torch.bool)

            score = param_scores * valid_params.to(param_scores.device).float() + 1e-6
            ffn_info.append((name, len(score), current_mask, 'param'))
            # ffn_info.append((name, len(param_scores), current_mask, 'ffn'))
            ffn_scores.append(score)

# **计算全局稀疏度**
    total_params = sum(mask.numel() for mask in all_masks)
    remaining_params = sum(mask.sum().item() for mask in all_masks)
    global_sparsity = 1 - (remaining_params / total_params)
    total_params=0
    pruned_params = 0
    for name, tensor in mask_dict.items():
        total_params += tensor.numel()
        pruned_params += torch.sum(tensor == 0).item()
    compression_rate =  pruned_params/ total_params

    print(f"Current global sparsity: {global_sparsity:.4f}, Target: {target_ratio}")
    print(f"Compression rate: {compression_rate:.4f}")
    print(f'ratio: {ratio:.4f}')

    if global_sparsity >= target_ratio:
        print("Pruning stopped: global sparsity reached target.")
        return mask_dict
    global_threshold_attn = compute_threshold(attn_scores, ratio-global_sparsity)
    global_threshold_ffn = compute_threshold(ffn_scores, ratio-global_sparsity)

    # **更新剪枝掩码**
    for (name, num_elements, current_mask, prune_type), scores in itertools.chain(zip(attn_info, attn_scores), zip(ffn_info, ffn_scores)):
        mask = torch.ones_like(current_mask, dtype=torch.bool)

        if prune_type == 'head' and global_threshold_attn is not None:
            head_mask = scores <= global_threshold_attn.to(scores.device)
            mask = head_mask.repeat_interleave(DIM)

        elif prune_type == 'ffn' and global_threshold_ffn is not None:
            param_mask = scores <= global_threshold_ffn.to(scores.device)
            mask = param_mask.reshape_as(current_mask)

        mask_dict[name] = mask.bool().float()
        print(f"name mask: {name}")
        print(f"Current mask: {current_mask.numel()}")
        print(f"Mask after pruning: {mask.sum().item()}")

    return mask_dict
def compute_threshold(scores_list, ratio):
    if len(scores_list) == 0:
        return None
    flat_scores = list(itertools.chain.from_iterable([s.tolist() for s in scores_list if s.numel() > 0]))
    if len(flat_scores) == 0:
        return None
    flat_scores = torch.tensor(flat_scores, dtype=torch.float32)
    k = max(1, int(len(flat_scores) * ratio))
    return torch.kthvalue(flat_scores, k).values


import math

def local_prune_change(model, s_dict, ratio, target_ratio, mask_dict=None, DIM=64):
    mask_dict = mask_dict.copy() if mask_dict else {}
    total_keep = 0
    total_params = 0
    pruned_heads = 0
    
    for name, module in model.named_modules():
        if not _is_target_larer(module) or not hasattr(module, 'weight'):
            continue
            
        weight = module.weight
        n_neurons = weight.shape[0]
        device = weight.device
        
        # 初始化或获取当前掩码
        current_mask = mask_dict.get(name, torch.ones(n_neurons, device=device))
        current_sparsity = 1 - current_mask.float().mean().item()
        # # 改进的目标计算逻辑v1
        # progress_ratio = 1 - (1 - current_sparsity) / (1 - current_sparsity + ratio)
        # target_layer_ratio = min(target_ratio, current_sparsity + progress_ratio * (target_ratio - current_sparsity))
         # 计算目标稀疏率
        # 计算全局稀疏度
        global_sparsity = 1 - total_keep / total_params if total_params > 0 else 0
        force_prune_ratio = (target_ratio - global_sparsity) / (1 - global_sparsity)

        # 自适应调整剪枝比例
        # 限制初始剪枝速率，避免一次性剪太多
        # 自适应调整剪枝比例
        progressive_ratio = max(ratio*0.5, force_prune_ratio)  # 防止剪枝速率过低
        target_layer_ratio = min(target_ratio, current_sparsity + (target_ratio - current_sparsity) * progressive_ratio)
        target_layer_ratio = min(target_ratio, current_sparsity + (target_ratio - current_sparsity) * progressive_ratio)
        if current_sparsity >= target_layer_ratio:
            total_keep += current_mask.sum().item()
            total_params += current_mask.numel()
            continue
        s_name = ".".join(name.split('.')[:-1])
        old_mask_sum = current_mask.sum().item()
        # 注意力层特殊处理
        if 'attention' in name :
            heads_mask = current_mask.view(-1, DIM).any(1)  # [n_heads]
            # print(f"heads_mask: {heads_mask.size(0)}")
            n_heads = heads_mask.size(0)
            active_heads = heads_mask.sum().item()
            max_prune = max(1, min(int(n_heads * progressive_ratio*1.2 ), int(active_heads * 0.8)))
            step_prune = max(1, min(int(active_heads * progressive_ratio), max_prune))
            # 进阶比例剪枝
            # progressive_ratio = ratio * (1 + current_sparsity)
            # min_prune = max(1, int(active_heads * 0.05))  # 保底至少剪5%
            # step_prune = max(min_prune, min(int(active_heads * progressive_ratio), max_prune))
            # if step_prune ==0:
            #     continue
            # head_importance = s_dict[s_name].view(n_heads, -1).sum(1)
            # prune_heads = head_importance[heads_mask.to(head_importance.device)].argsort()[:step_prune]
            
            # new_head_mask = heads_mask.scatter(0, prune_heads.to(heads_mask.device), False)
            # pruned_heads += (heads_mask.sum() - new_head_mask.sum()).item()
            # mask_dict[name] = new_head_mask.repeat_interleave(DIM)
            if step_prune > 0:
                head_importance = s_dict[s_name].view(n_heads, -1).sum(1)
                prune_heads = head_importance[heads_mask.to(head_importance.device)].argsort()[:step_prune]
                new_head_mask = heads_mask.scatter(0, prune_heads.to(heads_mask.device), False)
                pruned_heads += (heads_mask.sum() - new_head_mask.sum()).item()
                mask_dict[name] = new_head_mask.repeat_interleave(DIM)
            # # 计算最大可剪头数
            # max_prune_heads = int((target_layer_ratio - current_sparsity) * n_heads)
            # prune_heads = min(max_prune_heads, int(ratio * n_heads))
            
            # # 按重要性剪枝头部
            
            # head_importance = s_dict[s_name].view(n_heads, -1).sum(1)
            # head_scores = head_importance * heads_mask.to(head_importance.device).float()  # 排除已剪枝头
            # # head_scores = head_importance
            # prune_ids = torch.topk(head_scores, k=prune_heads, largest=False).indices
            
            # # 更新头部掩码
            # new_heads_mask = heads_mask.scatter(0, prune_ids.to(heads_mask.device), False)
            # pruned_heads += (heads_mask.sum() - new_heads_mask.sum()).item()
            
            # # 转换为参数掩码
            # current_mask = new_heads_mask.unsqueeze(1).repeat(1, DIM).view(-1)
        else:
            valid_mask = current_mask.bool()
            remaining = valid_mask.sum().item()
            max_prune = max(1, min(int(n_neurons * progressive_ratio * 1.5), int(remaining * 0.8)))

            step_prune = max(1, min(math.ceil(remaining * progressive_ratio), max_prune))

            # step_prune = min(step_prune, remaining - 1) if remaining > 1 else 0
            # 进阶比例剪枝
            # progressive_ratio = ratio * (1 + current_sparsity)
            # min_prune = max(1, int(remaining * 0.05))  # 保底至少剪5%
            # step_prune = max(min_prune, min(math.ceil(remaining * progressive_ratio), max_prune))

            
            if step_prune >0:
                prune_scores = s_dict[s_name].squeeze() * valid_mask.to(s_dict[s_name].device).float()
                prune_idx = torch.topk(prune_scores, k=step_prune, largest=False).indices
                current_mask = current_mask.scatter(0, prune_idx.to(current_mask.device), False)
                mask_dict[name] = current_mask
            # 确保 mask 没有变得更密集
        # 如果剪枝未成功，强制剪掉 5%
        if mask_dict[name].sum().item() > old_mask_sum:
            print(f"Warning: {name} mask increased! Reverting update.")
            mask_dict[name] = mask_dict[name] & (old_mask_sum > 0)
            # valid_neuron_mask = current_mask.bool()
            # remaining = valid_neuron_mask.sum().item()
            # print(f"remaining: {remaining}")
            
            # # 计算本步可剪数量
            # max_to_prune = int((target_layer_ratio - current_sparsity) * n_neurons)
            # step_prune = min(int(ratio * remaining), max_to_prune)
            
            # # 保护最低保留数（至少保留1个神经）
            # step_prune = min(step_prune, remaining - 1)
            
            # # 按重要性剪枝
            # neuron_importance = s_dict[s_name].squeeze()
            # # prune_scores = neuron_importance * valid_neuron_mask.to(neuron_importance.device).float()
            # prune_scores = neuron_importance
            # prune_ids = torch.topk(prune_scores, k=step_prune, largest=False).indices
            
            # current_mask = current_mask.scatter(0, prune_ids.to(current_mask.device), False)
        
        # 更新统计数据
        # 更新统计
        total_keep += mask_dict[name].sum().item()
        total_params += mask_dict[name].numel()

    final_sparsity = 1 - total_keep / total_params
    print(f"Progress sparsity: {final_sparsity*100:.1f}% (+{final_sparsity*100 - current_sparsity*100:.1f}%) | Heads pruned: {pruned_heads}")
    return mask_dict
    #     current_keep = current_mask.sum().item()
    #     total_keep += current_keep
    #     total_params += current_mask.numel()
        
    #     # 保存更新后的掩码
    #     mask_dict[name] = current_mask

    # # 打印统计信息
    # final_sparsity = 1 - total_keep / total_params
    # print(f"Final sparsity: {final_sparsity*100:.1f}% | Pruned heads: {pruned_heads}")
    # return mask_dict
import torch
import math

def search_mac(
    model,
    importance_scores,
    seq_len,
    mac_constraint,
    mask_dict=None
):
    mask_dict = mask_dict.copy() if mask_dict else {}
    assert mac_constraint < 1

    num_hidden_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    attention_head_size = hidden_size // num_attention_heads
    model_type = model.config.model_type

    original_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
        model_type=model_type,
    )
    max_mac = mac_constraint * original_mac
    print(f"Original MAC: {original_mac :.4f} | MAC Constraint: {max_mac :.4f}")

    attention_masks = {}
    ffn_masks = {}
    attention_score={}
    ffn_score= {}
    def get_sorted_scores(score_dict):
        all_scores = [(v.item(), i, layer) for layer, tensor in score_dict.items() 
                    for i, v in enumerate(tensor.view(-1))]
        
        sorted_scores = sorted(all_scores, key=lambda x: x[0], reverse=True)
        sorted_importance = torch.tensor([x[0] for x in sorted_scores])
        sorted_indices = [(x[1], x[2]) for x in sorted_scores]
        
        return sorted_importance, sorted_indices

    # 分类 importance scores
    attention_score, ffn_score = {}, {}
    for name, score_matrix in importance_scores.items():
        (attention_score if 'attention' in name else ffn_score)[name] = score_matrix

    # 计算排序后的分数和索引
    sorted_head_importance, sorted_head_indicies = get_sorted_scores(attention_score)
    sorted_neuron_importance, sorted_neuron_indicies = get_sorted_scores(ffn_score)
    print(f"sorted_head_indicies:", sorted_head_indicies)
    max_importance = 0
    for num_heads in range(1, num_hidden_layers * num_attention_heads + 1):
        heads_mac = mac_per_head(seq_len, hidden_size, attention_head_size) * num_heads
        neurons_mac = max_mac - heads_mac
        num_neurons = int(neurons_mac / mac_per_neuron(seq_len, hidden_size))
        num_neurons = max(num_neurons, 0)

        total_importance = sorted_head_importance[:num_heads].sum() + sorted_neuron_importance[:num_neurons].sum()
        if total_importance > max_importance:
            max_importance = total_importance
            head_indicies = sorted_head_indicies[:num_heads]
            neuron_indicies = sorted_neuron_indicies[:num_neurons]
    
    attention_masks = torch.zeros(num_hidden_layers * num_attention_heads).cuda()
    print(f"head_indicies:{head_indicies}")
    head_indicies = list(map(int, head_indicies))
    attention_masks[head_indicies] = 1.0
    attention_masks = attention_masks.view(num_hidden_layers, num_attention_heads)

    ffn_masks = torch.zeros(num_hidden_layers * intermediate_size).cuda()
    neuron_indicies = list(map(int, neuron_indicies))   
    ffn_masks[neuron_indicies] = 1.0
    ffn_masks = ffn_masks.view(num_hidden_layers, intermediate_size)
    
    mask_dict = {**attention_masks, **ffn_masks}
    
    return mask_dict

import torch
import math

import torch
import math
import re
def search_mac_change(
    model,
    importance_scores,
    seq_len,
    mac_constraint,
    mask_dict=None
):
    mask_dict_key = {}
    num_hidden_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    attention_head_size = hidden_size // num_attention_heads
    # print(f"importance_scores:{importance_scores}")
    model_type = model.config.model_type
    attention_mac, ffn_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
        model_type=model_type,
    )
    assert mac_constraint < 1
    #mac
    # max_mac_all = mac_constraint * (attention_mac+ffn_mac)
    # max_mac_head = max_mac_all * 0.7
    # max_mac_neuron = max_mac_all * 0.3
    #sparsity
    max_mac_head =(1-mac_constraint ) * attention_mac 
    max_mac_neuron = (1-mac_constraint) * ffn_mac
    max_mac={
        'head':max_mac_head,
        'neuron':max_mac_neuron
    } 
    # 计算每个头和神经元的 FLOPS（动态获取 head_dim，传入 model_type 识别 SwiGLU）
    model_type = model.config.model_type
    head_dim = hidden_size // num_attention_heads
    head_flops = mac_per_head(seq_len, hidden_size, head_dim)
    neuron_flops = mac_per_neuron(seq_len, hidden_size, model_type)
    importance_list_heads = []
    importance_list_neurons = [] 
    for name, module in model.named_modules():
        # if 'attention' in name or 'intermediate' in name or 'output' in name:
        if _is_target_larer(module):
            # mask_dict_key[name]= torch.zeros(module.weight.shape[0], dtype=torch.float32, device=module.weight.device)
            if name not in mask_dict:
                mask_dict[name] = torch.ones(module.weight.shape[0], dtype=torch.float32, device=module.weight.device)                
            # if 'attention' in name:
            #     key = ".".join(name.split('.')[:-2])
            #     value = importance_scores[key]
            #     mask = mask_dict[name].reshape(-1, DIM)[:, 0]#变成头维度
            #     value =value * mask.to(value.device)
            #     mask_dict_key[key]= torch.zeros_like(mask)
            #     #计算有效重要性     
            #     print(f'value.shape:',{value.shape})   
            #     for i, imp in enumerate(value.view(-1)):  # 直接展平遍历
            #         importance_list_heads.append((imp.item(),head_flops, key, i))  # (重要性, FLOPS, 层名, 头索引)
            # else:
            #     key = ".".join(name.split('.')[:-1])
            #     value = importance_scores[key] * mask_dict[name].to(value.device)
            #     mask_dict_key[name]= torch.zeros_like(mask_dict[name])
            #     #计算有效重要性
            #     for i, imp in enumerate(value.view(-1)):  # 直接展平遍历
            #         importance_list_neurons.append((imp.item(), neuron_flops, name, i))
    
    # 解析重要性分数
    # print("importance_list_heads",len(importance_list_heads))
    # importance_list =[]
    # importance_shapes = {}  # 记录每个 key 的原始形状

    for key, value in importance_scores.items():
        # 记录形状
        # if 'attention' in key:
        #     importance_shapes[key] = hidden_size // DIM #768
        # elif 'intermediate' in key:
        #     importance_shapes[key] = intermediate_size #3072
        # else:
        #     importance_shapes[key] = hidden_size
        # 判断是注意力层还是 FFN 层（兼容 BERT/ViT/LLaMA/Mistral）
        if "attention" in key or "self_attn" in key:
            model_type = model.config.model_type
            if model_type == 'bert':
                key = key + '.self'
                q_key = key + '.query'
            elif model_type == 'vit':
                key = key + '.attention'
                q_key = key + '.query'
            elif model_type in ('llama', 'mistral', 'qwen', 'gpt2', 'gpt_neox'):
                # LLaMA/Mistral: attention key 已是 model.layers.N.self_attn
                q_key = key + '.q_proj'
            else:
                # 默认降级：尝试直接用 key（某些架构可能不需要后缀）
                q_key = key
            # 动态获取 head_dim（BERT=64，LLaMA=128）
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            mask = mask_dict[q_key].reshape(-1, head_dim)[:, 0]  # 变成头维度
            value =value * mask.to(value.device)
            mask_dict_key[key]= torch.zeros_like(mask)  # 注意力层，已经按头计算
            for i, imp in enumerate(value.view(-1)):  # 直接展平遍历
                importance_list_heads.append((imp.item(), head_flops, key, i))  # (重要性, FLOPS, 层名, 头索引)

        else:
            value = importance_scores[key] * mask_dict[key].to(value.device)
            mask_dict_key[key]= torch.zeros_like(mask_dict[key])
              # FFN 层
            for i, imp in enumerate(value.view(-1)):  # 按神经元展开
                importance_list_neurons.append((imp.item(), neuron_flops, key, i))  # (重要性, FLOPS, 层名, 神经元索引)
    # print("importance_list_heads",len(importance_list_heads))
    # print("importance_list_neurons",len(importance_list_neurons))
    # importance_list.sort(key=lambda x: x[0] / x[1], reverse=True)
    # 生成掩码字典
    # mask_dict_key = {key: torch.zeros(shape).cuda() for key, shape in importance_shapes.items()}
    mask_dict, flops =balanced_pruning(model,importance_list_heads, importance_list_neurons, max_mac,mask_dict_key)
    # mask_dict, flops = layerwise_pruning(model, importance_scores, max_mac, seq_len)
    mac_ratio = 1-flops/(attention_mac+ffn_mac)
    print(f'Reduced  flops: {mac_ratio:.4%}')

    return mask_dict,mac_ratio
def balanced_pruning(
    model,
    importance_list_heads,
    importance_list_neurons,
    max_mac,
    mask_dict_key=None,
):
    # BERT/ViT 专用 head_dim（LLaMA 使用 model.config 动态获取）
    DIM = 64
    # 阶段一：独立处理注意力头
    mask_dict = {}
    current_mac = 0
    total_head_mac = sum([h[1] for h in importance_list_heads])
    # print(f'max_mac:{max_mac}')
    # print(f"total_head_mac:{total_head_mac}")
    
    # 按照单位重要性排序头部(降序)
    sorted_heads = sorted(importance_list_heads, 
                         key=lambda x: x[0]/x[1], reverse=True)

    head_mac_budget = max_mac['head']
    i=0
    for imp,  flops, key, idx in sorted_heads:
        if current_mac + flops > head_mac_budget:
            # print("head_flops:",i)
            # print(f"head_mac_budget/head_flops:{head_mac_budget/flops}")           
            break
        mask_dict_key[key][idx] = 1.0
        i=i+1
        current_mac += flops
    for key, mask in mask_dict_key.items():
        if "attention" in key and model.config.model_type == 'bert':
            mask_dict[key+'.self.query'] = mask.repeat_interleave(DIM)
            mask_dict[key+'.self.key'] = mask.repeat_interleave(DIM)
            mask_dict[key+'.self.value'] = mask.repeat_interleave(DIM)
            mask_dict[key+'.output.dense'] = mask.repeat_interleave(DIM)
        if "attention" in key and model.config.model_type == 'vit':
            mask_dict[key+'.query'] = mask.repeat_interleave(DIM)
            mask_dict[key+'.key'] = mask.repeat_interleave(DIM)
            mask_dict[key+'.value'] = mask.repeat_interleave(DIM)
            mask_dict[key+'.output.dense'] = mask.repeat_interleave(DIM)
        # LLaMA/Mistral/Qwen 等架构：掩码展开到 q/k/v/o_proj
        if "self_attn" in key and model.config.model_type in ('llama', 'mistral', 'qwen', 'gpt2', 'gpt_neox'):
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            mask_dict[key+'.q_proj'] = mask.repeat_interleave(head_dim)
            mask_dict[key+'.k_proj'] = mask.repeat_interleave(head_dim)
            mask_dict[key+'.v_proj'] = mask.repeat_interleave(head_dim)
            mask_dict[key+'.o_proj'] = mask.repeat_interleave(head_dim)
        # print(mask_dict[key].shape)
        # s_name= ''
        # for name, module in model.named_modules():
        #     if _is_target_larer(module):
        #         if 'attention' in name:
        #             s_name = ".".join(name.split('.')[:-2])                    
        #         else:
        #             s_name = ".".join(name.split('.')[:-1])
        #         if  s_name == key:
        #             mask=None                   
        #             mask = (mask_dict_key[key].new_ones(module.weight.shape[0]).reshape(-1, DIM) * mask_dict_key[key].unsqueeze(1)).reshape(-1)
        #             # print(f"mask:{mask.shape}")
        #             mask_dict[name] = mask
        
    print(f"current_mac_head:{current_mac}")
    # 阶段二：处理FFN神经元，使用剩余预算
    # remaining_mac = max_mac - current_mac
    sorted_neurons = sorted(importance_list_neurons,
                           key=lambda x: x[0]/x[1], reverse=True)
    
    # neuron_mask = {}
    neuron_mac_used = 0
    
    max_neuron_mac = max_mac['neuron'] 
    # print(f"max_neuron_mac:{max_neuron_mac}")
    for imp, flops, key, idx in sorted_neurons:
        if neuron_mac_used + flops > max_neuron_mac:
            print(f"max_neuron_mac/flops:{max_neuron_mac/flops}")
            break
        mask_dict_key[key][idx] = 1.0
        neuron_mac_used +=flops
    for key in mask_dict_key.keys():
        if "attention" not in key:
            mask_dict[key] = mask_dict_key[key]
    # print(f"current_mac_neuron:{current_mac+neuron_mac_used}")
    # 合并结果
    return mask_dict, current_mac +neuron_mac_used

#计算各层的重要性总和
def get_importance_sum(importance_scores):
    layer_importance = {}
    total_importance = 0.0
    for key, tensor in importance_scores.items():
        layer_name = key.split('.')[-2]#获取层数
        # if layer_name not in layer_importance:
        #     layer_importance[layer_name] = tensor.sum()
        # else:
        #     layer_importance[layer_name] += tensor.sum()
        layer_importance[layer_name] += tensor.sum()
        total_importance += tensor.sum()
    return layer_importance,total_importance
#分层剪枝，搜索mac
def layerwise_pruning(
    model,
    importance_scores,
    max_mac,
    seq_len,
    budget_strategy="uniform"  # 预算分配策略
):
    # 注：省略参数校验等前置代码
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size=model.config.hidden_size
    ffn_size = model.config.intermediate_size
    layer_budgets = {}  # 每层的可用预算字典 {layer_idx: budget}
    max_mac_flop =max_mac['neuron']+ max_mac['head']
    ratio =max_mac['head']/max_mac_flop

    # --- 预算分配策略 ---
    if budget_strategy == "uniform":
        # 均分法：各层等分预算
        per_layer_mac =  max_mac_flop/ num_layers
        for layer in range(num_layers):
            layer_budgets[layer] = per_layer_mac
    elif budget_strategy == "depth_decay":
        # 衰减法：浅层获得更多预算 (系数按指数衰减)
        for layer in range(num_layers):
            depth_ratio = (layer + 1) / num_layers
            decay_factor = 0.9 ** layer  # 例如，每层衰减10%
            layer_budgets[layer] = max_mac_flop * decay_factor / sum(0.9**i for i in range(num_layers))
    elif budget_strategy == "importance_based":
        # 基于预计算层重要性的分配（需预先计算各层score）
        layer_importances,total_imp = get_importance_sum(importance_scores)
        # total_imp = sum(layer_importances)
        for layer in range(num_layers):
            layer_budgets[layer] = max_mac_flop * layer_importances[layer] / total_imp
    
    # --- 层循环：每层独立处理 ---
    total_used_mac = 0.0
    mask_dict = {}
    for layer_idx in range(num_layers):
        if(model.config.model_type == 'bert'):
            prefix= 'base_model.model.bert'
        elif(model.config.model_type == 'vit'):
            prefix= 'base_model.model.vit'
        else:
            raise ValueError(f"Unsupported model type: {model.config.model_type}")
        layer_key_attn = f"{prefix}.encoder.layer.{layer_idx}.attention"
        layer_key_inter = f"{prefix}.encoder.layer.{layer_idx}.intermediate"
        layer_key_output = f"{prefix}.encoder.layer.{layer_idx}.output"
        
        # Step 1: 处理该层的注意力头剪枝
        layer_budget = layer_budgets[layer_idx]
        print(f"Layer {layer_idx}: budget = {layer_budget:.2f} MACs")
        head_mac_used, head_mask = prune_layer_heads(
            layer_idx, layer_key_attn, 
            importance_scores, 
            layer_budget*ratio, seq_len,
            num_heads,hidden_size
        )
        mask_dict.update(head_mask)
        # print(f"mask_dict:{mask_dict}")
        
        # 更新总消耗与剩余预算（剩余层预算）
        total_used_mac += head_mac_used
        remaining_budget = layer_budget - head_mac_used
        print(f"remaining_budget:{remaining_budget}")
        print(f"remaining_ratio:{remaining_budget/layer_budget}")


        
        # Step 2: 若剩余预算允许，处理FFN神经元剪枝
        if remaining_budget > 0:
            neu_mac_used, neu_mask = prune_layer_neurons(
                layer_idx, layer_key_inter, layer_key_output,
                importance_scores, 
                remaining_budget, seq_len,
                hidden_size,
                ffn_size
            )
            mask_dict.update(neu_mask)
            total_used_mac += neu_mac_used
    
    return mask_dict, total_used_mac

def prune_layer_heads(layer_idx, layer_key, importance_scores, layer_budget, seq_len,num_heads,hidden_size):
    print("pruning layer", layer_key)
    if layer_key not in importance_scores:
        return 0.0, {}  # 无此层数据
    
    # 获取当前层的参数
    head_size = hidden_size // num_heads
    head_importance = importance_scores[layer_key] 
    # print(head_importance)
    
    # 构造头FLOPs列表
    head_flops = mac_per_head(seq_len, hidden_size, head_size)
    # print("layer_budget", layer_budget)
    can_prune = int(layer_budget // head_flops)
    # print("can prune", can_prune, "heads")
    _,topk_indices = torch.topk(head_importance, can_prune)
    layer_mask ={layer_key:torch.zeros_like(head_importance)}
    mask = torch.zeros_like(head_importance)
    mask[topk_indices] = 1
    print(f" attention head  sum:{mask}")
    mask = mask.repeat_interleave(head_size)
    layer_mask[layer_key]=mask
    return head_flops*can_prune, layer_mask

def prune_layer_neurons(layer_idx, layer_key_inter, layer_key_output, importance_scores, remaining_budget, seq_len, hidden_size, ffn_size):
    required_keys = {layer_key_inter, layer_key_output}
    if not all(key in importance_scores for key in required_keys):
        return 0.0, {}
    
    # 定义层参数配置
    layer_configs = [
        (layer_key_inter, ffn_size, importance_scores[layer_key_inter]),
        (layer_key_output, hidden_size, importance_scores[layer_key_output])
    ]
    
    # 预计算每个神经元的计算量
    neuron_flops = mac_per_neuron(seq_len, hidden_size)
    
    # 初始化掩码和预算跟踪
    layer_mask = {
        layer_key_inter: torch.zeros(ffn_size),
        layer_key_output: torch.zeros(hidden_size)
    }
    current_mac = 0.0
    
    # 统一处理所有层类型
    for layer_key, layer_size, importance in layer_configs:
        # 生成排序后的神经元列表
        sorted_neurons = sorted(
            [(importance[i].item(), neuron_flops, layer_key, i) for i in range(layer_size)],
            key=lambda x: x[0] / x[1], 
            reverse=True
        )
        
        # 应用预算约束选择神经元
        for imp, flops, key, idx in sorted_neurons:
            if current_mac + flops > remaining_budget:
                break
            layer_mask[key][idx] = 1.0
            current_mac += flops
    
    return current_mac, layer_mask

def mac_per_neuron_intermediate(seq_len, hidden_size):
    """中间层单个神经元的计算量 (2*input_dim)"""
    return 2 * seq_len * hidden_size  # MAC操作数：input_dim = hidden_size -> (hidden_size, 1) 的矩阵乘法

def mac_per_neuron_output(seq_len, ffn_size):
    """输出层单个神经元的计算量 (2*input_dim)"""
    return 2 * seq_len * ffn_size     # input_dim = ffn_size -> (ffn_size, 1) 的矩阵乘法

def prune_ffn_layers(layer_idx, layer_key_inter, layer_key_output, 
                     importance_scores, remaining_budget, seq_len, 
                     hidden_size, ffn_size):
    """
    基于预算约束的前馈层剪枝 (包含中间层和输出层)
    
    Args:
        layer_idx: 当前处理的层索引 (debug用)
        layer_key_inter: 中间层的标识符 (如 "ffn.mid")
        layer_key_output: 输出层的标识符 (如 "ffn.out")
        importance_scores: 包含各层重要性得分的字典 (张量)
        remaining_budget: 当前可用的MAC预算 (FLOPs)
        seq_len: 序列长度
        hidden_size: 隐藏层维度 (d_model)
        ffn_size: 中间层维度 (d_ff)
        
    Returns:
        total_cost: 实际使用的FLOPs
        mask_dict: 包含两层的二进制掩码 (1=保留, 0=裁减)
    """
    # 1. 初始检查
    required_keys = {layer_key_inter, layer_key_output}
    if missing_keys := required_keys - importance_scores.keys():
        print(f"[层 {layer_idx}错误] 缺失重要性分数：{missing_keys}")
        return 0.0, {} 

    # 2. 配置层级参数
    layer_configs = [
        {   # Intermediate Layer (hidden_size -> ffn_size)
            "key": layer_key_inter,
            "size": ffn_size,
            "flops_func": lambda: mac_per_neuron_intermediate(seq_len, hidden_size),
            "importance": importance_scores[layer_key_inter].cpu().detach()
        },
        {   # Output Layer (ffn_size -> hidden_size) 
            "key": layer_key_output,
            "size": hidden_size,
            "flops_func": lambda: mac_per_neuron_output(seq_len, ffn_size),
            "importance": importance_scores[layer_key_output].cpu().detach()
        }
    ]
    
    # 3. 初始化掩码和预算跟踪
    mask_dict = {cfg["key"]: torch.ones(cfg["size"], dtype=torch.bool) for cfg in layer_configs}
    total_cost = 0.0
    budget = remaining_budget
    
    # 4. 分层剪枝（按中间层->输出层顺序）
    for cfg in layer_configs:
        # 计算单位FLOPs
        flops_per_neuron = cfg["flops_func"]()
        max_keep = cfg["size"]  # 理论最大保留数量
        
        # 生成神经元排序（性价比=重要性/FLOPs)
        cost_efficiency = cfg["importance"] / flops_per_neuron
        ranked_indices = torch.argsort(cost_efficiency, descending=True)
        
        # 动态确定保留数量
        max_possible = min(max_keep, budget // flops_per_neuron)
        if max_possible <=0:
            mask_dict[cfg["key"]][:] = False  # 完全剪除当前层
            continue
            
        # 核心剪枝逻辑
        selected_indices = ranked_indices[:max_possible]
        new_mask = torch.zeros_like(mask_dict[cfg["key"]])
        new_mask[selected_indices] = True
        mask_dict[cfg["key"]] = new_mask
        
        # 更新预算
        actual_cost = len(selected_indices) * flops_per_neuron
        total_cost += actual_cost
        budget -= actual_cost
        
    # 5. 后处理检查
    assert torch.all(mask_dict[layer_key_inter].sum() > 0) or budget <=0, "中间层误删引发架构破坏!"
    return total_cost, mask_dict

def mac_per_head(
    seq_len,
    hidden_size,
    attention_head_size,
):
    #1、计算生成Q、K、V的MAC的次数;2、计算生成注意力矩阵的MAC次数；3、计算注意力权重矩阵

    per_head_qkv = lambda seq_len: 3 * seq_len * hidden_size * attention_head_size
    per_head_attn = lambda seq_len: 2 * seq_len * seq_len * attention_head_size
    per_head_output = lambda seq_len: seq_len * attention_head_size * hidden_size
    mac = per_head_qkv(seq_len) + per_head_attn(seq_len) + per_head_output(seq_len)
    return mac

#计算每个隐藏层的FLOPS
def mac_per_neuron(seq_len, hidden_size, model_type=None):
    """计算 FFN 单个神经元的 MAC。

    Args:
        seq_len: 序列长度
        hidden_size: 隐藏层维度
        model_type: 模型类型，默认从 model.config.model_type 推断。
                    LLaMA/Mistral/Qwen 使用 SwiGLU（3 个矩阵：gate+up+down），
                    而标准 BERT FFN 使用 2 个矩阵（intermediate+output）。
    """
    # 为保持向后兼容，model_type=None 时默认返回标准 MLP 的 MAC
    if model_type in ('llama', 'mistral', 'qwen', 'gpt2', 'gpt_neox'):
        # SwiGLU: gate_proj + up_proj + down_proj = 3 次矩阵乘
        return 3 * seq_len * hidden_size
    else:
        # 标准 2 层 MLP（up + down）
        return 2 * seq_len * hidden_size

#计算总的FLOPS
def compute_mac(
    num_heads_per_layer,#[12,12,...]
    num_neurons_per_layer,#[3072,3072....]
    seq_len,
    hidden_size,
    attention_head_size,
    model_type=None,
):
    """计算总 MAC，可选传入 model_type 以正确计算 SwiGLU 的 FFN 成本。"""
    attention_mac = 0.0
    ffn_mac = 0.0
    for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
        attention_mac += num_heads * mac_per_head(seq_len, hidden_size, attention_head_size)
        ffn_mac += num_neurons * mac_per_neuron(seq_len, hidden_size, model_type)
    return attention_mac, ffn_mac





def schedule_sparsity_ratio(
    step,
    total_step,
    initial_warmup,
    final_warmup,
    initial_sparsity,
    final_sparsity,
):
    if step <= initial_warmup * total_step:
        sparsity = initial_sparsity
    elif step > (total_step - final_warmup * total_step):
        sparsity = final_sparsity
    else:
        spars_warmup_steps = initial_warmup * total_step
        spars_schedu_steps = (final_warmup + initial_warmup) * total_step
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (mul_coeff ** 3)
    return sparsity

def prune_from_checkpoint(model): 
    prune(model)

def print_trainable_parameters(model):
    total_params = 0
    trainable_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        total_params += p.numel()
    print("total params:{}   trainable params:{}    ratio:{}".format(total_params * 1e-6, trainable_params * 1e-6, trainable_params / total_params))
def apply_weight_mask(model, mask_dict):
    """
    将 mask 应用到 Linear 层的 weight 参数（非 forward 输出）。
    与 LoRA 兼容，仅屏蔽原始权重部分。
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                # layer_name = ".".join(name.split('.')[:-1])
                if name in mask_dict:
                    mask = mask_dict[name].to(module.weight.device)
                    # 权重硬掩码应用
                    module.weight.data *= mask
                    a = module.lora_A['default'].weight.data
                    b = module.lora_B['default'].weight.data
