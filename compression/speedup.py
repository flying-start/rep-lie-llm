import torch
from transformers.pytorch_utils import prune_linear_layer

from paths import get_path


def speedup(args, model, masks):
    if 'bert' in args.arch:
        return speedup_bert(args, model, masks)
    elif 'vit' in args.arch:
        return speedup_vit(args, model, masks)
    elif 'm2f' in args.arch:
        return speedup_swin_m2f(args, model, masks)
    else:
        raise NotImplementedError


def speedup_bert(args, model, masks):
    if isinstance(model, str):
        model = torch.load(model, map_location='cpu')

    if isinstance(masks, str):
        masks = torch.load(masks, map_location='cpu')

    def _prune_head_idxs(mask, num_heads):
        head_mask = (mask.reshape([num_heads, -1]).sum(-1) == 0.)
        return torch.arange(len(head_mask))[head_mask].long().tolist()

    # prune heads
    attention_modules = dict(
        [(name, module) for name, module in model.named_modules() if name.split('.')[-1] == 'attention'])
    for name, att_module in attention_modules.items():
        # print(masks)
        mask = masks[name+'.self.query']['weight'].to('cpu')
        num_heads = att_module.self.num_attention_heads
        prune_head_idxs = _prune_head_idxs(mask, num_heads)
        att_module.prune_heads(prune_head_idxs)
        att_module.pruned_heads = set()

    # prune ffns
    module_names = [name for name, _ in model.named_modules()]
    for name in module_names:
        if name not in masks.keys():
            continue
        if 'attention' not in name:
            module = model.get_submodule(name)
            if 'output' in name:
                # print('output:',masks[name].shape)
                module = prune_linear_layer(module, masks[name]['weight'].nonzero()[:, 0],dim=1)
            else:
                # print('intermediate:',masks[name].shape)
                module = prune_linear_layer(module, masks[name]['weight'].nonzero()[:, 0])
            setattr(model.get_submodule('.'.join(name.split('.')[:-1])), name.split('.')[-1], module)

    return model
import torch
import torch.nn as nn
# from transformers.utils import prune_linear_layer

def speedup_bert_with_ffn_mask(args, model, masks_path):
    """
    通过剪枝加速 BERT，并使用 intermediate 层的掩码替换 output.dense 层的掩码。

    Args:
        args: 训练参数（可选）。
        model: 需要剪枝的 BERT 模型（或路径）。
        masks_path: 掩码的存储路径（.pth 文件）。

    Returns:
        pruned_model: 剪枝后的模型。
    """

    # 加载模型
    if isinstance(model, str):
        model = torch.load(model, map_location='cpu')

    # 加载掩码
    if isinstance(masks_path, str):
        masks = torch.load(masks_path, map_location='cpu')
    else:
        masks = masks_path  # 直接传递掩码字典

    def _prune_head_idxs(mask, num_heads):
        """
        计算需要剪枝的注意力头索引。
        """
        head_mask = (mask.reshape([num_heads, -1]).sum(-1) == 0.)
        return torch.arange(len(head_mask))[head_mask].long().tolist()

    # 1️⃣ **剪枝注意力层（Prune Attention Heads）**
    # for name, module in model.named_modules():
    #     attention_modules = {name: module for name, module in model.named_modules() if name.endswith("attention")}
    #     if not hasattr(module, 'weight'):
    #         continue 
    #     if 'attention' in name:
    #         # print(f"Pruning with masks: ",masks)
    #         # name = ".".join(name.split('.')[:-1])
    #         mask = masks['base_model.model.'+name]
    #         print(f"Pruning {name} with mask: {mask is not None}")
            
    #         if mask is not None:
    #             mask = mask.to('cpu')
    #             num_heads = module.self.num_attention_heads
    #             prune_head_idxs = _prune_head_idxs(mask, num_heads)
    #             module.prune_heads(prune_head_idxs)
    #             module.pruned_heads = set()  # 记录已剪枝的头部
    #             print(f"Pruned {len(prune_head_idxs)} heads in {name}")
    #         # 剪枝注意力头
    #     else:
    #         if 'intermediate' in name:
    #         # **intermediate 层的掩码**
    #             name = ".".join(name.split('.')[:-1])
    #             ffn_mask = masks['base_model.model.'+name].to('cpu')
    #             pruned_module = prune_linear_layer(module, ffn_mask.nonzero()[:, 0])  # 默认 dim=0
    #             setattr(module, name.split('.')[-1], pruned_module)
    #             print(f"Pruned FFN layer in {name}")

    #         elif 'output' in name and 'attention' not in name:
    #             # **用 intermediate 的掩码剪枝 output**
    #             name = ".".join(name.split('.')[:-1])
    #             intermediate_name = name.replace("output", "intermediate")
    #             if intermediate_name in masks:
    #                 ffn_mask = masks[intermediate_name].to('cpu')
    #                 pruned_module = prune_linear_layer(module, ffn_mask.nonzero()[:, 0], dim=1)  # dim=1 确保输入剪枝
    #                 setattr(module, name.split('.')[-1], pruned_module)
    #                 print(f"Pruned output layer in {name}")

    attention_modules = {name: module for name, module in model.named_modules() if name.endswith("attention")}
    
    for name, att_module in attention_modules.items():
        # print(f"Pruning with masks: ",masks)
        # name = ".".join(name.split('.')[:-1])
        # print(f"Pruning {name}...")
        if model.config.model_type == 'bert':
            mask = masks[name+'.self.query']
        elif model.config.model_type == 'vit':
            mask = masks[name+'.attention.query']
        # mask =masks[name]
        # print(f"Pruning {name} with mask: {mask is not None}")
        
        if mask is not None:
            mask = mask.to('cpu')
            num_heads = att_module.self.num_attention_heads
            prune_head_idxs = _prune_head_idxs(mask, num_heads)
            att_module.prune_heads(prune_head_idxs)
            att_module.pruned_heads = set()  # 记录已剪枝的头部
            print(f"Pruned {len(prune_head_idxs)} heads in {name}")

    # 2️⃣ **剪枝 FFN 层（Prune FFN Layers）**
    module_names = list(model.named_modules())
    print(f"Pruning FFN layers...")
    total =0
    for name, module in module_names:
        if not hasattr(module, 'weight') or 'lora' in name:
            continue
        if '.' in name:
            parent_module = model.get_submodule('.'.join(name.split('.')[:-1]))
        else:
            parent_module = model  # 防止 name 为空
        if 'intermediate' in name and name.endswith("dense"):
            # **intermediate 层的掩码**
            # name = ".".join(name.split('.')[:-1])
            mask_key =  name
            
            if mask_key in masks:
                ffn_mask = masks[mask_key].to('cpu')
                # print(f"  - 原始维度: {module.weight.shape}")
                pruned_module = prune_linear_layer(module, ffn_mask.nonzero()[:, 0])  # dim=0
                setattr(parent_module, name.split('.')[-1], pruned_module)
                print(f"Pruned FFN layer in {name}")
                print(f"  - 剪枝后维度: {pruned_module.weight.shape}")
                total+=pruned_module.weight.shape[0]

        elif 'output' in name and 'attention' not in name and name.endswith("dense"):
            # **用 intermediate 的掩码剪枝 output**
            # name = ".".join(name.split('.')[:-1])
            intermediate_name = name.replace("output", "intermediate")
            mask_key =  intermediate_name
            if mask_key in masks:
                ffn_mask = masks[mask_key].to('cpu')
                # print(f"  - 原始维度: {module.weight.shape}")
                pruned_module = prune_linear_layer(module, ffn_mask.nonzero()[:, 0], dim=1)  # dim=1 确保输入剪枝
                setattr(parent_module, name.split('.')[-1], pruned_module)
                print(f"Pruned output layer in {name}")
                print(f"  - 剪枝后维度: {pruned_module.weight.shape}")
                total+=pruned_module.weight.shape[1]
    if args.arch =='bert-base-uncased':
        print(f"Total prune ratio: {1-((total+144*(1-args.mac)*64*4)/(3072*24+144*64*4))}")
    else:
        print(f"Total prune ratio: {1-((total+168*(1-args.mac)*64*4)/(4096*36+168*64*4))}")
    return model


def speedup_vit2(args, model, masks):
    if isinstance(model, str):
        model = torch.load(model, map_location='cpu')

    if isinstance(masks, str):
        masks = torch.load(masks, map_location='cpu')

    def _prune_head_idx(module, name, parent_module, mask):
        return model


def speedup_vit(args, model, masks):
    if isinstance(model, str):
        model = torch.load(model, map_location='cpu')

    if isinstance(masks, str):
        masks = torch.load(masks, map_location='cpu')

    def _prune_head_idxs(mask, num_heads):
        head_mask = (mask.reshape([num_heads, -1]).sum(-1) == 0.)
        return torch.arange(len(head_mask))[head_mask].long().tolist()

    # prune heads
    attention_modules = dict(
        [(name, module) for name, module in model.named_modules() if name.split('.')[-1] == 'attention' and name.split('.')[-2] != 'attention'])
    for name, att_module in attention_modules.items():
        mask = masks[name + '.attention.query']['weight'].to('cpu')
        num_heads = att_module.attention.num_attention_heads
        prune_head_idxs = _prune_head_idxs(mask, num_heads)
        att_module.prune_heads(prune_head_idxs)
        att_module.pruned_heads = set()

    # prune ffns
    module_names = [name for name, _ in model.named_modules()]
    for name in module_names:
        if name not in masks.keys():
            continue
        if 'attention' not in name:
            module = model.get_submodule(name)
            if 'output' in name:
                module = prune_linear_layer(module, masks[name]['weight'].sum(dim=0).nonzero()[:, 0], dim=1)
            else:
                module = prune_linear_layer(module, masks[name]['weight'].sum(dim=1).nonzero()[:, 0])
            setattr(model.get_submodule('.'.join(name.split('.')[:-1])), name.split('.')[-1], module)

    return model


def speedup_swin_m2f(args, model, masks):
    if isinstance(model, str):
        model = torch.load(model, map_location='cpu')

    if isinstance(masks, str):
        masks = torch.load(masks, map_location='cpu')

    def _prune_head_idxs(mask, num_heads):
        head_mask = (mask.reshape([num_heads, -1]).sum(-1) == 0.)
        return torch.arange(len(head_mask))[head_mask].long().tolist()

    # prune heads
    # attention_modules = dict(
    #     [(name, module) for name, module in model.named_modules() if name.split('.')[-1] == 'attention'])

    attention_modules = dict(
        [(name, module) for name, module in model.named_modules() if name.split('.')[-1] == 'attention' and name.split('.')[-2] != 'attention'])
    for name, att_module in attention_modules.items():
        mask = masks[name + '.self.query']['weight'].to('cpu')
        num_heads = att_module.self.num_attention_heads
        prune_head_idxs = _prune_head_idxs(mask, num_heads)
        att_module.prune_heads(prune_head_idxs)
        rem_heads = [i for i in range(att_module.self.relative_position_bias_table.shape[-1]) if i not in prune_head_idxs]
        att_module.self.relative_position_bias_table = torch.nn.Parameter(att_module.self.relative_position_bias_table[:, rem_heads])
        att_module.pruned_heads = set()

        mask = torch.zeros(num_heads, dtype=bool)
        mask[rem_heads] = True
        masks[name +'.self'] = {'relative_position_bias_table': mask}

    # prune ffns
    module_names = [name for name, _ in model.named_modules()]
    for name in module_names:
        if name not in masks.keys():
            continue
        if 'attention' not in name:
            module = model.get_submodule(name)
            if 'output' in name:
                module = prune_linear_layer(module, masks[name]['weight'].sum(dim=0).nonzero()[:, 0], dim=1)
            else:
                module = prune_linear_layer(module, masks[name]['weight'].sum(dim=1).nonzero()[:, 0])
            setattr(model.get_submodule('.'.join(name.split('.')[:-1])), name.split('.')[-1], module)

    torch.save(masks, get_path(args, 'INIT_MASKS_PATH'))

    return model