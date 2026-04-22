import loralib as lora
import torch
import os
from trainer_utils1 import *
import math
import FLOP
from FLOP.utils import *
import sys

def finetune(model, args, data_content, training_params, for_eval_flag=True, tag="default"):

    model = make_hard_concrete(model, in_place=True, init_mean=0.5, init_std=0.01)
    model.cuda()

    hc_modules = get_hardconcrete_modules(model)
    trainer = prepare_traced_trainer_with_flop(model, args, data_content,hc_modules,0.8,training_params,for_eval_flag=for_eval_flag, tag=f"{tag}")
    trainer.train() 
    

    pruning_masks = get_pruning_masks(model)


    # 打印剪枝掩码
    for name, mask in pruning_masks.items():
        print(f"Module: {name}, Mask Shape: {mask.shape}")
    # 应用剪枝掩码
    pruned_model = apply_pruning_masks(model, pruning_masks)
    # 保存剪枝后的模型
    torch.save(pruned_model, "pruned_bert_model.pth")

    return model,trainer.state,pruning_masks

def get_pruning_masks(model):
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, (FLOP.HardConcreteLinear, FLOP.HardConcreteProjectedLinear)):
            # 获取 HardConcrete 掩码
            mask = module.mask()  # 调用 HardConcrete 的 mask() 方法
            masks[name] = mask
    return masks

def get_hardconcrete_gate_parameters(model):
    gate_params = {}
    for name, module in model.named_modules():
        if isinstance(module, (FLOP.HardConcreteLinear, FLOP.HardConcreteProjectedLinear)):
            gate_params[name] = module.mask.log_alpha  # 获取门控参数
    return gate_params

def apply_pruning_masks(model, masks):
    for name, module in model.named_modules():
        if name in masks:
            mask = masks[name]
            # 应用剪枝掩码
            if isinstance(module, FLOP.HardConcreteLinear):
                module.weight.data *= mask.view(-1, 1)
            elif isinstance(module, FLOP.HardConcreteProjectedLinear):
                module.weight.data *= mask.view(-1, 1)
                module.weight_proj.data *= mask.view(1, -1)
    return model


    