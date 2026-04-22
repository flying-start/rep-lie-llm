#!/usr/bin/env python
# coding=utf-8

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset
import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# 添加项目路径并导入自定义模块
sys.path.append("../")
# import loraprune

# 导入稳定性剪枝训练器
from trainer_sb import StabilityLoRAPruneTrainer
import utils as utils
from args import ModelArguments, DataTrainingArguments, PruneArguments

logger = logging.getLogger(__name__)

def main():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PruneArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果提供了JSON文件作为配置，使用它
        model_args, data_args, training_args, prune_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, prune_args = parser.parse_args_into_dataclasses()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 打印参数信息
    logger.warning(
        f"进程排名: {training_args.local_rank}，设备: {training_args.device}，n_gpu: {training_args.n_gpu}，"
        + f"分布式训练: {bool(training_args.local_rank != -1)}，16位训练: {training_args.fp16}"
    )
    logger.info(f"训练/评估参数 {training_args}")
    logger.info(f"剪枝参数 {prune_args}")

    # 设置随机种子
    set_seed(training_args.seed)

    # 加载数据集
    if data_args.task_name is not None:
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1] if data_args.train_file else "json"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    # 根据任务获取标签
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        if data_args.is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    # 加载预训练模型和分词器
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # 应用LoRA初始化
    if prune_args.lora_r > 0:
        print("应用LoRA初始化，r =", prune_args.lora_r)
        utils.add_lora_to_model(model, prune_args.lora_r, prune_args.lora_alpha)

    # 预处理数据集
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        sentence1_key, sentence2_key = "sentence1", "sentence2"

    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        return result

    # 应用预处理
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="运行分词器进行数据预处理",
    )

    # 准备训练和验证数据集
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # 准备性能指标
    metric = evaluate.load("glue", data_args.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # 数据整理器
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # 打印稳定性配置
    print("\n" + "="*50)
    print("稳定性剪枝配置:")
    print(f"- 使用稳定性分数: {prune_args.use_stability}")
    print(f"- 稳定性组件列表: {prune_args.stability_components}")
    print(f"- 稳定性权重: {prune_args.stability_weight}")
    print(f"- 稳定性收集比例: {prune_args.stability_collection_ratio}")
    print("="*50 + "\n")

    # 初始化稳定性剪枝训练器
    trainer = StabilityLoRAPruneTrainer(
        model=model,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        args=training_args,
        ratio=prune_args.ratio,
        init_ratio=prune_args.init_ratio,
        warmup_iters=prune_args.warmup_iters,
        cooldown_iters=prune_args.cooldown_iters,
        prune_freq=prune_args.prune_freq,
        prune_metric=prune_args.prune_metric,
        mac=prune_args.mac,
        seq_len=max_seq_length,
        task_name=data_args.task_name,
        # 稳定性配置参数
        use_stability=prune_args.use_stability,
        stability_components=prune_args.stability_components,
        stability_weight=prune_args.stability_weight,
        stability_collection_ratio=prune_args.stability_collection_ratio
    )

    # 训练
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # 保存稳定性分析数据
        if prune_args.use_stability:
            trainer.save_stability_distribution()
            trainer.generate_stability_report()

    # 评估
    if training_args.do_eval:
        logger.info("*** 评估 ***")
        metrics = trainer.evaluate()

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 生成完整训练报告
    trainer.generate_full_training_report()

# 任务到键的映射
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

if __name__ == "__main__":
    main() 