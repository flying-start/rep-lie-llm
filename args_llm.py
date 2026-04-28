"""
args_llm.py
===========
提供 LLM 剪枝所需的全部参数 dataclass，与 HfArgumentParser 兼容。

注意：ModelArguments / DataTrainingArguments 并非 transformers 内置类，
      必须在此自行定义（参考 HuggingFace 官方 run_clm.py 示例）。
"""

from dataclasses import dataclass, field
from typing import Optional


# ================================================================
# ModelArguments：模型加载相关参数
# ================================================================
@dataclass
class ModelArguments:
    """模型路径、缓存、鉴权等加载参数"""
    model_name_or_path: str = field(
        metadata={"help": "预训练模型路径或 HuggingFace Hub 模型名称"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer 路径（不填则与 model_name_or_path 相同）"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "模型/tokenizer 缓存目录"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用 fast tokenizer（Rust 实现）"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "模型 git 版本（branch/tag/commit hash）"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "是否使用 HuggingFace Hub token（访问私有/受限模型，如 LLaMA）"},
    )


# ================================================================
# DataTrainingArguments：数据集与序列长度相关参数
# ================================================================
@dataclass
class DataTrainingArguments:
    """数据集路径、序列长度、采样数等数据处理参数"""
    task_name: Optional[str] = field(
        default="c4",
        metadata={"help": "数据集名称，目前支持: c4"},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace datasets 数据集名称（task_name 的别名）"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace datasets 数据集配置名称"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大输入序列长度（token 数）"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "限制训练样本数（调试用）"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "限制评估样本数（调试用）"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "是否覆盖数据集缓存"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "数据预处理并行进程数"},
    )


# ================================================================
# 自定义参数：PruneArguments（剪枝超参）
# ================================================================
@dataclass
class PruneArguments:
    """REP-LIE 剪枝超参数"""
    ratio: float = field(
        default=0.5,
        metadata={"help": "目标稀疏率（0.5 = 剪掉 50%）"},
    )
    init_ratio: float = field(
        default=0.1,
        metadata={"help": "剪枝初始比例"},
    )
    warmup_iters: float = field(
        default=0.1,
        metadata={"help": "预热阶段比例（0.1 = 前 10% 步不剪枝）"},
    )
    cooldown_iters: float = field(
        default=0.1,
        metadata={"help": "冷却阶段比例（0.1 = 后 10% 步不剪枝）"},
    )
    prune_freq: float = field(
        default=100,
        metadata={"help": "每 N 步剪枝一次"},
    )
    prune_metric: str = field(
        default="lora",
        metadata={
            "help": "剪枝指标: lora / grad / magnitude",
            "choices": ["lora", "grad", "magnitude"],
        },
    )
    mac: float = field(
        default=0.5,
        metadata={"help": "MAC 约束上限"},
    )
    # LoRA 参数（也通过 PruneArguments 传入）
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA 秩"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA 缩放因子"},
    )
    # 稳定性参数
    use_stability: bool = field(
        default=True,
        metadata={"help": "是否使用稳定性分数"},
    )
    stability_components: str = field(
        default="attention",
        metadata={"help": "稳定性作用的组件: attention / ffn / attention,ffn"},
    )
    stability_weight: float = field(
        default=0.3,
        metadata={"help": "稳定性分数权重"},
    )
    stability_collection_ratio: float = field(
        default=0.25,
        metadata={"help": "用于收集稳定性分数的训练步数比例"},
    )
    # 任务名（兼容 main9.py 格式）
    task: str = field(
        default="llm_causal",
        metadata={"help": "任务类型: llm_causal"},
    )


# ================================================================
# 自定义参数：LLMArguments（LLM 特有超参）
# ================================================================
@dataclass
class LLMArguments:
    """LLM 特有参数（HuggingFace 标准参数中没有的）"""
    calibration_nsamples: int = field(
        default=512,
        metadata={"help": "C4 校准集采样数量"},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "是否以 4-bit 量化加载模型（bitsandbytes）"},
    )
    lm_eval_tasks: Optional[str] = field(
        default=None,
        metadata={"help": "lm-evaluation-harness zero-shot 任务，逗号分隔"},
    )
