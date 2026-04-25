from peft import LoraConfig ,get_peft_model
# from loraprune.peft_model import get_peft_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import math
from transformers import AdamW
from trainer_utils import *
from paths import *
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.tune import TuneConfig, Tuner
from ray.air.config import RunConfig
from ray.tune.search.sample import choice
from ray.tune import with_parameters
import optuna
from optuna.samplers import TPESampler


def apply_lora_with_sparsity(model,r,lora_alpha=32, lora_dropout=0.1):
    """
    动态应用 LoRA，并添加稀疏正则化。
    参数:
        model: 原始 Transformer 模型
        lora_mask: 剪枝掩码
        r: LoRA 的秩
        lora_alpha: LoRA 的放大因子
        lora_dropout: LoRA 的 Dropout 比例
        sparsity_weight: 稀疏正则化权重（L1 范数系数）
    返回:
        配置了 LoRA 的模型
    """
    # 对于ViT图像分类任务，使用正确的task_type
    lora_config = LoraConfig(
        # task_type="SEQ_CLS"#序列分类任务-bert
        task_type="IMAGE_CLASSIFICATION",  # 修改为图像分类任务-vit
        target_modules=["query","value","key","intermediate.dense","output.dense"],
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, lora_config)


    return model

def finetune_with_lora_and_sparsity(
    model, args, data_content, training_params, model_path=None, lora_save_path=None, sparsity_weight=1e-4, for_eval_flag=True, tag="default", round=1
):
    rank =4*round
    new_model = apply_lora_with_sparsity(model, r=rank)
    trainer = prepare_traced_trainer(new_model, args, data_content, training_params, for_eval_flag=for_eval_flag, tag=f"{tag}_rank_{rank}")
    max_steps = math.ceil(training_params['num_train_epochs'] * len(data_content['train']))
    prepare_masked_trainer(args, trainer, max_steps)
    trainer.train()
    normalized_deltas = trainer.sensitivities
    # normalized_deltas=trainer.sensitivities.items()/trainer.state.global_step
    # normalized_deltas = {
    # name: value / trainer.state.global_step
    # for name, value in trainer.sensitivities.items()
    # }
    # print("FLOP:",trainer.general_flops)


    # normalized_deltas = compute_layerwise_normalized_deltas(new_model, 'default')

    new_model = new_model.merge_and_unload()
    # 保存最终模型
    if round == 1:
        torch.save(new_model, get_path(args, "INIT_FINETUNED_MODEL_PATH"))
    else:
        torch.save(new_model, get_path(args, "ITER_FINETUNED_MODEL_PATH"))

    return new_model,trainer.state,normalized_deltas



def finetune_with_lora_and_sparsity_tune(
    model, args, data_content, training_params, model_path=None, lora_save_path=None, sparsity_weight=1e-4, for_eval_flag=True, tag="default", round=1
):
    TASK_METRICS = {
    "cola": "matthews_correlation",  # CoLA 任务使用 Matthews Correlation
    "sst2": "accuracy",             # SST-2 任务使用 Accuracy
    "mrpc": "f1",                   # MRPC 任务使用 F1 分数
    "sts-b": "pearson",             # STS-B 任务使用 Pearson Correlation
    "qqp": "accuracy",                    # QQP 任务使用 F1 分数
    "mnli": "accuracy",             # MNLI 任务使用 Accuracy
    "qnli": "accuracy",             # QNLI 任务使用 Accuracy
    "rte": "accuracy",              # RTE 任务使用 Accuracy
}

    """
    使用 LoRA 和稀疏正则化对模型进行微调，同时保存微调后的稀疏化参数。
    """
    metric_name = TASK_METRICS[args.data]
    def objective(trial):
        """
        定义 Optuna 调优目标函数。
        """
        # 从 Optuna 试验中获取超参数
        rank = trial.suggest_categorical("rank", [1,2,4,8,16])  # Rank 搜索空间

        # 迁移到 GPU
        local_model = model.to("cuda")
        local_data_content = data_content  # 假设你已经将数据传递给目标函数

        # 应用 LoRA 和稀疏化
        new_model = apply_lora_with_sparsity(local_model, r=rank)

        # 准备训练器
        trainer = prepare_traced_trainer(new_model, args, local_data_content, training_params, for_eval_flag=for_eval_flag, tag=f"{tag}_rank_{rank}")
        trainer.train()

        # 验证集性能
        eval_results = trainer.evaluate()
        accuracy = eval_results.get(f"eval_{metric_name}", 0.0)

        # 返回目标指标
        return accuracy  # Optuna 会自动优化该目标

    # 创建 Optuna study 和调优
    study = optuna.create_study(direction="maximize", sampler=TPESampler())  # 采用贝叶斯优化（TPE）
    study.optimize(objective, n_trials=5)  # 进行 5 次超参数调优

    # 输出最佳超参数
    best_rank = study.best_params["rank"]
    print(f"Best rank: {best_rank}")

    # 调优 rank 参数
    # best_rank = tune_rank(args, model, data_content, training_params)

    # 使用最佳 rank 参数重新微调
    final_model = apply_lora_with_sparsity(model, r=best_rank)
    trainer = prepare_traced_trainer(final_model, args, data_content, training_params, for_eval_flag=for_eval_flag, tag=f"{tag}_{best_rank}_best_rank")
    trainer.train()

    # normalized_deltas = compute_layerwise_normalized_deltas(model, 'default', normalization="l2")
    normalized_deltas = compute_global_normalized_deltas(final_model, 'default', normalization="l2")
    new_model = final_model.merge_and_unload()
    # 保存最终模型
    if round == 1:
        torch.save(new_model, get_path(args, "INIT_FINETUNED_MODEL_PATH"))
    else:
        torch.save(new_model, get_path(args, "ITER_FINETUNED_MODEL_PATH"))

    return new_model,trainer.state,normalized_deltas
def compute_module_delta_weights(model, adapter_name):
        """
        计算模型中所有 LoRA 包裹模块的增量权重。

        Args:
        model: PEFT LoRA 模型实例。
        adapter_name (str): 指定的 LoRA adapter 名称。

        Returns:
        dict: 模块名与对应增量权重的映射。
        """
        delta_weights = {}
        for name, module in model.named_modules():
            if hasattr(module, "get_delta_weight"):  # 检查模块是否支持获取增量
                try:
                    delta_weight = module.get_delta_weight(adapter_name) 
                     # 获取增量权重
                    name_without_prefix = name.replace("base_model.model.", "")  
                    delta_weights[name_without_prefix] = delta_weight                    
                except Exception as e:
                    print(f"Error computing delta weight for {name}: {e}")
        
        return delta_weights
def compute_module_sensitivity(model):
    """
    计算模型中所有 LoRA 包裹模块的敏感性分数。

    Args:
    model: PEFT LoRA 模型实例。
    adapter_name (str): 指定的 LoRA adapter 名称。

    Returns:
    dict: 模块名与对应敏感性分数的映射。
    """
    sensitivities = {}

    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            try:
                # 获取 LoRA 权重和梯度
                A = module.lora_A["default"].weight
                B = module.lora_B["default"].weight
                grad_A = module.lora_A["default"].weight.grad
                grad_B = module.lora_B["default"].weight.grad

                # 检查梯度是否存在
                if grad_A is None or grad_B is None:
                    print(f"Gradients not available for {name}, skipping.")
                    continue

                # 计算敏感性分数公式: \nabla B \cdot A + B \cdot \nabla A - \nabla B \cdot \nabla A
                sensitivity = grad_B @ A + B @ grad_A - grad_B @ grad_A

                # 归一化敏感性分数
                sensitivity = sensitivity.abs().sum().item()  # 转化为标量

                # 存储结果
                name_without_prefix = name.replace("base_model.model.", "")
                sensitivities[name_without_prefix] = sensitivity

            except Exception as e:
                print(f"Error computing sensitivity for {name}: {e}")

    return sensitivities
def compute_layerwise_normalized_deltas(model, adapter_name, normalization="max"):
    """
    计算并对模型中所有 LoRA 包裹模块的增量权重进行层级归一化。

    Args:
    model: PEFT LoRA 模型实例。
    adapter_name (str): 指定的 LoRA adapter 名称。
    normalization (str): 归一化方法，可选 "max" 或 "l2"。

    Returns:
    dict: 模块名与归一化后增量权重的映射。
    """
    delta_weights = compute_module_delta_weights(model, adapter_name)
    normalized_delta_weights = {}

    for name, delta in delta_weights.items():
        # 确保增量权重是张量
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(delta)
        
        if normalization == "max":
            # 按最大值归一化
            norm_factor = delta.abs().max().item()
        elif normalization == "l2":
            # 按 L2 范数归一化
            norm_factor = delta.norm(p=2).item()
        else:
            raise ValueError(f"Unsupported normalization method: {normalization}")

        # 防止归一化因子为零
        if norm_factor > 0:
            normalized_delta = delta / norm_factor
        else:
            normalized_delta = delta

        normalized_delta_weights[name] = normalized_delta

    return normalized_delta_weights
def compute_global_normalized_deltas(model, adapter_name, normalization="max"):
    """
    对模型中所有 LoRA 包裹模块的增量权重进行全局归一化。

    Args:
    model: PEFT LoRA 模型实例。
    adapter_name (str): 指定的 LoRA adapter 名称。
    normalization (str): 归一化方法，可选 "max" 或 "l2"。

    Returns:
    dict: 模块名与全局归一化后增量权重的映射。
    """
    delta_weights = compute_module_sensitivity(model)
    all_deltas = []

    # 收集所有增量权重
    for delta in delta_weights.values():
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(delta)
        all_deltas.append(delta.flatten())  # 展平增量权重张量
    
    # 拼接所有增量到一个大张量
    global_delta_tensor = torch.cat(all_deltas)

    # 计算全局归一化因子
    if normalization == "max":
        norm_factor = global_delta_tensor.abs().max().item()
    elif normalization == "l2":
        norm_factor = global_delta_tensor.norm(p=2).item()
    else:
        raise ValueError(f"Unsupported normalization method: {normalization}")

    # 防止归一化因子为零
    if norm_factor == 0:
        raise ValueError("Normalization factor is zero, possibly due to all deltas being zero.")

    # 对每一层的增量权重进行归一化
    normalized_delta_weights = {}
    for name, delta in delta_weights.items():
        normalized_delta_weights[name] = delta / norm_factor

    return normalized_delta_weights
      
def save_lora_parameters(model, save_path):
    """
    保存微调后的 LoRA 参数。
    参数:
        model: 配置了 LoRA 的模型
        save_path: 保存路径
    """
    lora_params = {name: param for name, param in model.named_parameters() if "lora" in name}
    torch.save(lora_params, save_path)
    print(f"LoRA parameters saved to {save_path}")


def adjust_delta_weights_with_masks(delta_weights, masks):
    """
    根据剪枝掩码调整 delta_weights 的大小，仅保留 mask 中为 1 的部分。
    支持 delta_weights 和 masks 的嵌套字典结构。

    Args:
        delta_weights (dict): 模块名与增量权重的嵌套字典。
        masks (dict): 模块名与掩码的嵌套字典。

    Returns:
        dict: 调整后的增量权重嵌套字典。
    """
    adjusted_delta_weights = {}

    for name, delta_weight in delta_weights.items():
        if name not in masks:
            raise KeyError(f"Mask for module {name} not found in masks dictionary!")

        mask = masks[name]["weight"]  # 获取对应模块的掩码
        # print(mask.shape)
        if isinstance(delta_weight, dict) and isinstance(mask, dict):
            # 如果 delta_weight 和 mask 都是字典，递归处理
            adjusted_delta_weights[name] = adjust_delta_weights_with_masks(delta_weight, mask)
        elif isinstance(delta_weight, torch.Tensor) and isinstance(mask, torch.Tensor):
            # 如果是张量，直接逐元素相乘
            if delta_weight.shape != mask.shape:
                raise ValueError(
                    f"Shape mismatch for module {name}: "
                    f"delta_weight shape {delta_weight.shape}, mask shape {mask.shape}"
                )
            adjusted_delta_weights[name] = delta_weight * mask
            
        else:
            print(mask)
            raise TypeError(f"Unsupported types for masking: {type(delta_weight)} and {type(mask)}")
    for name, param in adjusted_delta_weights.items():
        if "output.dense" in name and param is not None and isinstance(param, torch.Tensor):
        # 找出非全零列的索引
            non_zero_columns = torch.any(param != 0, dim=0)  # 按列检查
        # 压缩全零列
            compressed_param = param[:, non_zero_columns]
        # 更新 adjusted_delta_weights
            adjusted_delta_weights[name] = compressed_param
            # print(f"Layer: {name}, Original Shape: {param.shape}, Compressed Shape: {compressed_param.shape}")

        elif param is not None and isinstance(param, torch.Tensor):
        # 找出非全零行的索引
            non_zero_rows = torch.any(param != 0, dim=1)
        # 压缩全零行
            compressed_param = param[non_zero_rows]
        # 更新 adjusted_delta_weights
            adjusted_delta_weights[name] = compressed_param
            # print(f"Layer: {name}, Original Shape: {param.shape}, Compressed Shape: {compressed_param.shape}")
        else:
            print(f"Layer: {name} is not a valid tensor.")

    return adjusted_delta_weights
