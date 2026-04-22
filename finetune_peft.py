from peft import LoraConfig, get_peft_model
import torch
import os
from trainer_utils import *
import math
from transformers import AdamW
import loralib as lora

def apply_lora_with_mask(model, lora_mask):
    """
    根据剪枝生成的 mask 动态应用 LoRA。
    """
    lora_config = LoraConfig(
    task_type="SEQ_CLS",       # 任务类型，例如文本分类 (SEQ_CLS)
    inference_mode=False,      # 训练模式（False 表示训练，True 表示推理）
    r=8,                       # LoRA 的秩
    lora_alpha=16,             # LoRA 放大因子
    lora_dropout=0.1           # LoRA Dropout 比例
    )
    peft_model = get_peft_model(model, lora_config)
    # 打印模型结构，检查 LoRA 层是否被正确添加
    print(peft_model)
    return peft_model


def finetune_with_lora_and_mask(model, args, data_content, training_params, model_path=None, for_eval_flag=True, tag='default'):
    # 从指定路径加载优化器状态和初始化掩码
    if os.path.exists(get_path(args, 'OPT_STATE_PATH')):
        opt_states = torch.load(get_path(args, 'OPT_STATE_PATH'))
    init_masks = torch.load(get_path(args, 'INIT_MASKS_PATH'))

    # 根据 mask 动态应用 LoRA
    model = apply_lora_with_mask(model, init_masks)

    # # 冻结被剪枝的参数，仅训练未剪枝权重中的 LoRA 参数
    # for name, param in model.named_parameters():
    #     if "lora" in name and name in init_masks and init_masks[name]:
    #         param.requires_grad = True  # 解冻 LoRA 参数
    #     else:
    #         param.requires_grad = False  # 冻结被剪枝的权重
    # for name, param in model.named_parameters():
    #     param.requires_grad = True  # 确保梯度计算流通


    # 检查可训练参数
    # trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    # print(f"Trainable parameters: {trainable_params}")

    # 准备训练器
    trainer = prepare_traced_self_trainer(model, args, data_content, training_params, for_eval_flag=for_eval_flag, tag=tag)
    
    # 设置优化器
    # 仅优化包含 "lora" 的参数  
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=training_params.get('learning_rate', 1e-4))
    trainer.create_optimizer_and_scheduler(num_training_steps=math.ceil(training_params['num_train_epochs'] * len(data_content['train'])))
    
    # 执行训练
    trainer.train()

    # 保存训练后的模型
    trainer_state = trainer.state
    trainer_state.opt_state = trainer.optimizer.state_dict()['state']
    print("Completed fine-tuning with LoRA and mask")  
    if model_path:
        torch.save(model, model_path)
        print(f"Model saved to {model_path}")
    
    del trainer
    return model, trainer_state
