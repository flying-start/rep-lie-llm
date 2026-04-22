import numpy as np
import torch
# from .lora import Linear, Linear8bitLt
from peft.tuners.lora import Linear,MaskedModuleWrapper

pruning_groups = {'self_attn': ['query', 'key', 'value', 'output.dense'],
                  'mlp': ['output.dense', 'intermediate.dense']}

DIM = 64 

def _is_target_larer(module):
    return isinstance(module, Linear)  


def init_sensitivity_dict(model):
    """初始化敏感度字典，确保掩码维度与层输出维度匹配"""
    sensitivity_record = {}
    for name, module in model.named_modules():
        if _is_target_larer(module):
            # 获取模块的输出特征数
            out_features = module.out_features if hasattr(module, 'out_features') else module.weight.shape[0]
            
            # 判断层类型
            is_attn = name.split('.')[-1] in pruning_groups['self_attn'] or 'attention.output' in name
            is_output = 'output.dense' in name and 'attention' not in name
            intermediate = 'intermediate' in name and 'attention' not in name
            # 计算分组数量            
            # 根据不同层类型设置掩码维度
            if is_attn:
                # 注意力头的掩码
                num_heads = out_features // DIM
                mask = torch.ones(num_heads, requires_grad=False)
            elif is_output or intermediate:
                # 输出层和中间层的掩码
                mask = torch.ones(out_features, requires_grad=False)
            if is_attn:
                name = ".".join(name.split('.')[:-2])
            # else:
            #     name = ".".join(name.split('.')[:-1])
            if is_attn or is_output or intermediate:
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
        if isinstance(module, Linear)  and name in mask_dict:
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
                        if 'attention' in layer_name:
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
                                
                        elif 'output' in layer_name or 'intermediate' in layer_name:
                            # 对于输出层和中间层，调整掩码形状
                            mask = mask.view(1, 1, -1)  # 变成 [1, 1, 3072]
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
    for name, module in model.named_modules():
        if _is_target_larer(module):  # 判断是否是目标模块
            # 判断模块类型
            # print(name)
            is_attn = name.split('.')[-1] in pruning_groups['self_attn'] or 'attention.output.dense' in name
            is_output = 'output.dense' in name  and 'attention' not in name
            intermediate = 'intermediate' in name and 'attention' not in name            
            sensitivity = compute_sensitivity(module, is_attn,is_output,intermediate, prune_metric)
            if is_attn:
                name = ".".join(name.split('.')[:-2])
            # else:
            #     name = ".".join(name.split('.')[:-1])
            # 存储敏感分数
            if name in new_s_dict:
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


def compute_sensitivity(layer, is_attn,is_output,intermediate,  prune_metric='lora',transpose=False, norm=True):
    a = layer.lora_A['default'].weight.data
    b = layer.lora_B['default'].weight.data
    if prune_metric == 'lora':
        grad_a = layer.lora_A['default'].weight.grad
        grad_b = layer.lora_B['default'].weight.grad
        # grad = b @ a * layer.scaling['default']
        grad = (grad_b @ a + b @ grad_a - grad_b @ grad_a)
    elif prune_metric == 'magnitude':
        grad = 1
    elif prune_metric == 'grad':
        # 检查梯度是否存在，如果不存在则使用权重作为替代
        if hasattr(layer.weight, 'grad') and layer.weight.grad is not None:
            grad = layer.weight.grad
        else:
            # 如果梯度不存在，使用权重作为替代
            grad = layer.weight.data
    else:
        raise NotImplementedError
    if hasattr(layer, 'state'):
        weight = (layer.weight.data * layer.state.SCB.reshape(-1, 1)) / 127
    else:
        weight = layer.weight.data
    # print(layer.scaling)
    s = (grad * (b @ a * layer.scaling['default'] + weight)).abs()
    if transpose:
        s = s.t()
    if is_attn:
        s = s.reshape(s.shape[0] // DIM, -1)
    s=s.sum(1)
    if norm:
        s = s / (torch.linalg.norm(s) + 1e-8)
    return s

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

    original_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
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
    # 初始化一个空的掩码字典键
    mask_dict_key = {}
    # 获取模型配置中的隐藏层数量
    num_hidden_layers = model.config.num_hidden_layers
    # 获取模型配置中的注意力头数量
    num_attention_heads = model.config.num_attention_heads
    # 获取模型配置中的中间层大小
    intermediate_size = model.config.intermediate_size
    # 获取模型配置中的隐藏层大小
    hidden_size = model.config.hidden_size
    # 计算每个注意力头的大小
    attention_head_size = hidden_size // num_attention_heads
    # print(f"importance_scores:{importance_scores}")
    attention_mac ,ffn_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
    )
    assert mac_constraint < 1
    # max_mac = mac_constraint * (attention_mac+ffn_mac)
    max_mac_head =(1-mac_constraint) * attention_mac 
    max_mac_neuron = (1-mac_constraint) * ffn_mac
    max_mac={
        'head':max_mac_head,
        'neuron':max_mac_neuron
    } 
    # 计算每个头和神经元的 FLOPS
    head_flops = mac_per_head(seq_len, hidden_size, attention_head_size)
    # print(f'head_flops:',head_flops)
    neuron_flops = mac_per_neuron(seq_len, hidden_size)
    importance_list_heads = []
    importance_list_neurons = [] 
    for name, module in model.named_modules():
        if _is_target_larer(module):
            if name not in mask_dict:
                mask_dict[name] = torch.ones(module.weight.shape[0], dtype=torch.float32, device=module.weight.device)                


    for key, value in importance_scores.items():
        
        # print(f"key:{key}")
        if "attention" in key:
            mask = mask_dict[key+'.self.query'].reshape(-1, DIM)[:, 0]#变成头维度
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
    print("importance_list_heads",len(importance_list_heads))
    print("importance_list_neurons",len(importance_list_neurons))
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
            print("head_flops:",i)
            print(f"head_mac_budget/head_flops:{head_mac_budget/flops}")           
            break
        mask_dict_key[key][idx] = 1.0
        i=i+1
        current_mac += flops
    for key, mask in mask_dict_key.items():
        if "attention" in key:
            mask_dict[key+'.self.query'] = mask.repeat_interleave(DIM)
            mask_dict[key+'.self.key'] = mask.repeat_interleave(DIM)
            mask_dict[key+'.self.value'] = mask.repeat_interleave(DIM)
            mask_dict[key+'.output.dense'] = mask.repeat_interleave(DIM)     
    print(f"current_mac_head:{current_mac}")
    # 阶段二：处理FFN神经元，使用剩余预算
    # remaining_mac = max_mac - current_mac
    sorted_neurons = sorted(importance_list_neurons,
                           key=lambda x: x[0]/x[1], reverse=True)
    
    # neuron_mask = {}
    neuron_mac_used = 0
    
    max_neuron_mac = max_mac['neuron'] 
    print(f"max_neuron_mac:{max_neuron_mac}")
    for imp, flops, key, idx in sorted_neurons:
        if neuron_mac_used + flops > max_neuron_mac:
            print(f"max_neuron_mac/flops:{max_neuron_mac/flops}")
            break
        mask_dict_key[key][idx] = 1.0
        neuron_mac_used +=flops
    for key in mask_dict_key.keys():
        if "attention" not in key:
            mask_dict[key] = mask_dict_key[key]
    print(f"current_mac_neuron:{current_mac+neuron_mac_used}")
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
        prefix= 'base_model.model.bert'
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
def mac_per_neuron(seq_len, hidden_size):
    return 2 * seq_len * hidden_size

#计算总的FLOPS
def compute_mac(
    num_heads_per_layer,#[12,12,...]
    num_neurons_per_layer,#[3072,3072....]
    seq_len,
    hidden_size,
    attention_head_size,
):
    attention_mac = 0.0
    ffn_mac = 0.0
    for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
        attention_mac += num_heads * mac_per_head(seq_len, hidden_size, attention_head_size)
        ffn_mac += num_neurons * mac_per_neuron(seq_len, hidden_size)     
    return attention_mac,ffn_mac

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

def apply_model_mask(model, mask_dict):
    """
    将 mask 应用到 Linear 层的 weight 参数（非 forward 输出）。
    与 LoRA 兼容，仅屏蔽原始权重部分。
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, Linear) and name in mask_dict:
                # layer_name = ".".join(name.split('.')[:-1])
                mask = mask_dict[name].to(module.weight.device)
                # 权重硬掩码应用                   
                module.weight.data = apply_weight_mask(module.weight.data, mask)
                # 如果使用了 LoRA，mask 掉 LoRA 的部分
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    # 假设使用的是 'default' LORA adapter
                    a = module.lora_A["default"].weight  # 形状：[r, in_features]
                    b = module.lora_B["default"].weight  # 形状：[out_features, r]

                    if mask.shape[0] == module.weight.shape[0]:
                        # mask 输出维度（影响 B）
                        b.data *= mask.view(-1, 1)

                    elif mask.shape[0] == module.weight.shape[1]:
                        # mask 输入维度（影响 A）
                        a.data *= mask.view(1, -1)
                    a = module.lora_A['default'].weight.data
                    b = module.lora_B['default'].weight.data
def apply_weight_mask(weight, mask):
    if mask.shape[0] == weight.shape[0]:  # 输出维度
        return weight * mask.view(-1, 1)
    elif mask.shape[0] == weight.shape[1]:  # 输入维度
        return weight * mask.view(1, -1)
    else:
        raise ValueError(f"Mask shape {mask.shape} does not match weight {weight.shape}")
