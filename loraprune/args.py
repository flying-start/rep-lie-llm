#!/usr/bin/env python
# coding=utf-8

from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class ModelArguments:
    """
    用于定义模型相关的参数
    """
    model_name_or_path: str = field(
        metadata={"help": "预训练模型的路径或名称"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "要使用的特定模型版本(分支名、标签名或提交ID)"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练模型配置名称或路径，如果与model_name_or_path不同"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练分词器名称或路径，如果与model_name_or_path不同"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "存储下载模型和特征的目录"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用快速tokenizer (由 Rust 实现)"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "使用 Hugging Face token 访问私有模型"},
    )


@dataclass
class DataTrainingArguments:
    """
    用于定义数据集和训练过程的参数
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "任务名称，用于GLUE基准测试任务"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "训练数据集名称"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "数据集的特定配置"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "输入序列的最大长度"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的预处理特征"}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "是否将所有样本填充到`max_seq_length`。"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "最大训练样本数，用于调试"
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "最大评估样本数，用于调试"
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "最大预测样本数，用于调试"
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "训练数据文件"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "验证数据文件"}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "测试数据文件"}
    )
    is_regression: bool = field(
        default=False,
        metadata={"help": "是否为回归任务"}
    )


@dataclass
class PruneArguments:
    """
    用于定义剪枝和LoRA参数的类
    """
    # LoRA参数
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA适配器的秩"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha参数，通常为r的2倍"}
    )
    
    # 剪枝参数
    ratio: float = field(
        default=0.5,
        metadata={"help": "剪枝目标比例"}
    )
    mac: float = field(
        default=0.5,
        metadata={"help": "MAC约束比例"}
    )
    init_ratio: float = field(
        default=0.1,
        metadata={"help": "初始剪枝比例"}
    )
    warmup_iters: float = field(
        default=0.1,
        metadata={"help": "预热迭代次数比例"}
    )
    cooldown_iters: float = field(
        default=0.1,
        metadata={"help": "冷却迭代次数比例"}
    )
    prune_freq: int = field(
        default=200,
        metadata={"help": "剪枝频率，每隔多少步进行一次剪枝"}
    )
    prune_metric: str = field(
        default="lora",
        metadata={"help": "重要性度量方法: 'lora', 'grad', 或 'magnitude'"}
    )
    
    # 稳定性参数
    use_stability: bool = field(
        default=True,
        metadata={"help": "是否使用稳定性分数进行剪枝"}
    )
    stability_components: List[str] = field(
        default_factory=lambda: ["attention", "ffn"],
        metadata={"help": "应用稳定性计算的组件: 'attention', 'ffn', 或两者"}
    )
    stability_weight: float = field(
        default=0.3,
        metadata={"help": "稳定性分数的权重"}
    )
    stability_collection_ratio: float = field(
        default=0.25,
        metadata={"help": "在剪枝周期中用于收集稳定性的比例"}
    ) 