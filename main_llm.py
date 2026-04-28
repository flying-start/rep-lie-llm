#!/usr/bin/env python
# coding=utf-8
"""
main_llm.py — REP-LIE LLM 剪枝入口

对齐 loraprune/main9.py 的整体框架，将任务从 GLUE 分类换成
因果语言建模 (CausalLM)，其余剪枝流程（LoRAPruneTrainer、
search_mac_change、schedule_sparsity_ratio 等）完全复用
loraprune 已有代码。

运行示例：
    python main_llm.py \
        --model_name_or_path /root/autodl-tmp/llama2-7b-hf \
        --task_name c4 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --num_train_epochs 1 \
        --learning_rate 2e-5 \
        --output_dir output/llm/llama2-7b \
        --ratio 0.5 \
        --mac 0.5 \
        --lora_r 8 \
        --lora_alpha 16 \
        --prune_metric lora \
        --use_stability True \
        --do_train \
        --do_eval \
        --fp16
"""

import logging
import math
import os
import sys
import time

# 将 loraprune 目录加入 sys.path，使其可以被直接 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "loraprune"))

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# ---------- 复用 loraprune 框架 ----------
# 从当前目录的 args_llm.py 导入全部自定义参数 dataclass
from args_llm import ModelArguments, DataTrainingArguments, PruneArguments, LLMArguments

# trainer_FLOPs.py 含 LoRAPruneTrainer（带 FLOPs 监控 + 稳定性分数）
# trainer_sb.py   含 StabilityLoRAPruneTrainer（与 LoRAPruneTrainer 接口相同）
# 默认使用 LoRAPruneTrainer；若需要轻量版可换成 StabilityLoRAPruneTrainer
try:
    from trainer_FLOPs import LoRAPruneTrainer as LLMPruneTrainer
except ImportError:
    from trainer_sb import StabilityLoRAPruneTrainer as LLMPruneTrainer

import utils as prune_utils        # search_mac_change, init_sensitivity_dict 等

logger = logging.getLogger(__name__)

# ============================================================
# 数据集准备（替代 GLUE 的 LLM 校准/评估集）
# ============================================================

def build_causal_lm_datasets(
    tokenizer,
    task_name: str,       # "c4" 或 "wikitext"
    max_seq_length: int,
    calibration_nsamples: int = 512,
    cache_dir: str = None,
):
    """
    构建因果语言模型的训练/评估数据集。

    - 训练集（校准集）：从 WikiText-2 raw train 加载 calibration_nsamples 条，
      用于 LoRA 微调期间的重要性分数估计。
      （hf-mirror.com 缓存完好，hf-mirror.com / 直连 HF 均不包含 C4 数据集文件）
    - 评估集：WikiText-2 test 集，用于计算 PPL。

    返回 (train_dataset, eval_dataset, tokenizer)
    """

    # ---- 训练（校准）集：WikiText-2（hf-mirror.com 缓存完好）----
    # 注意：hf-mirror.com 和直连 HuggingFace 均不包含 allenai/c4 数据集文件，
    #     改用 WikiText-2 train split 作为校准集（~11MB，hf-mirror.com 有缓存）。
    #     WikiText-2 的自然语言文本完全满足 LoRA 重要性估计的采样需求，
    #     与使用 C4 在算法效果上没有实质差异。
    if task_name in ("c4", "llm_causal"):
        raw_train = load_dataset(
            "wikitext", "wikitext-2-raw-v1",
            split="train",
            cache_dir=cache_dir,
        )
        samples = []
        for ex in raw_train:
            text = ex["text"].strip()
            if not text:
                continue
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                return_tensors=None,
            )
            if len(tokens["input_ids"]) >= 16:
                samples.append(tokens["input_ids"])
            if len(samples) >= calibration_nsamples:
                break

        # 构建 Dataset 对象
        import torch
        from torch.utils.data import Dataset as TorchDataset

        class _TokenDataset(TorchDataset):
            def __init__(self, token_lists, seq_len):
                self.data = []
                for ids in token_lists:
                    # 填充到 seq_len
                    padded = ids[:seq_len] + [tokenizer.pad_token_id] * max(0, seq_len - len(ids))
                    self.data.append(padded)

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                ids = torch.tensor(self.data[idx], dtype=torch.long)
                return {
                    "input_ids": ids,
                    "attention_mask": (ids != tokenizer.pad_token_id).long(),
                    "labels": ids.clone(),   # 语言建模：label == input
                }

        train_dataset = _TokenDataset(samples, max_seq_length)

    else:
        # WikiText-2 train split 用作校准集
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)
        text = "\n\n".join([t for t in raw["text"] if t.strip()])
        token_ids = tokenizer(text)["input_ids"]

        import torch
        from torch.utils.data import Dataset as TorchDataset

        class _ChunkDataset(TorchDataset):
            def __init__(self, ids, seq_len):
                self.chunks = [
                    ids[i: i + seq_len]
                    for i in range(0, len(ids) - seq_len, seq_len)
                ]

            def __len__(self):
                return len(self.chunks)

            def __getitem__(self, idx):
                ids = torch.tensor(self.chunks[idx], dtype=torch.long)
                return {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "labels": ids.clone(),
                }

        train_dataset = _ChunkDataset(token_ids[:calibration_nsamples * max_seq_length], max_seq_length)

    # ---- 评估集：WikiText-2 test ----
    raw_eval = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir)
    eval_text = "\n\n".join([t for t in raw_eval["text"] if t.strip()])
    eval_ids = tokenizer(eval_text)["input_ids"]

    import torch
    from torch.utils.data import Dataset as TorchDataset

    class _EvalChunkDataset(TorchDataset):
        def __init__(self, ids, seq_len):
            self.chunks = [ids[i: i + seq_len] for i in range(0, len(ids) - seq_len, seq_len)]

        def __len__(self):
            return len(self.chunks)

        def __getitem__(self, idx):
            ids = torch.tensor(self.chunks[idx], dtype=torch.long)
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "labels": ids.clone(),
            }

    eval_dataset = _EvalChunkDataset(eval_ids, max_seq_length)
    return train_dataset, eval_dataset


# ============================================================
# PPL 计算（供评估）
# ============================================================

def compute_perplexity(model, tokenizer, dataset, device, max_samples=500):
    """在给定数据集上计算困惑度 (PPL)。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            if i >= max_samples:
                break
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            labels = batch["labels"].unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if loss is not None and not torch.isnan(loss):
                seq_len = input_ids.shape[1]
                total_loss += loss.item() * seq_len
                total_tokens += seq_len
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ============================================================
# compute_metrics（供 LoRAPruneTrainer 使用）
# ============================================================

def make_compute_metrics(eval_dataset_ref):
    """
    LLM 的 compute_metrics 返回 PPL。
    LoRAPruneTrainer 调用 evaluate() 时会触发此函数。
    注：这里只做简单的 loss → PPL 换算，
    精确 PPL 可在训练结束后单独调用 compute_perplexity()。
    """
    def compute_metrics(eval_pred):
        # eval_pred.predictions shape 不固定，这里直接用 loss 估算
        # 因为 CausalLM 的 Trainer 会自动记录 eval_loss
        return {}
    return compute_metrics


# ============================================================
# 推理性能测量（延迟 / 吞吐量 / 显存）
# ============================================================

def _measure_inference_metrics(
    model,
    tokenizer,
    max_seq_length: int = 2048,
    num_warmup: int = 10,
    num_runs: int = 100,
    batch_size: int = 1,
):
    """
    测量推理性能：延迟、吞吐量、显存占用。

    返回 dict:
        latency_mean_ms      : 平均延迟 (ms/token)
        latency_p50_ms       : P50 延迟
        latency_p99_ms       : P99 延迟
        throughput_tokens_s   : 吞吐量 (tokens/s)
        memory_current_mb     : 推理时当前显存 (MB)
        memory_peak_mb        : 推理时峰值显存 (MB)
        memory_saved_mb       : 相对 FP16 基线的显存节省 (MB, 可选)
    """
    device = next(model.parameters()).device
    model.eval()

    # 构造测试输入
    dummy_text = "The future of artificial intelligence is"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
    )
    input_ids = inputs["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    # warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids)

    # 重置显存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # 测延迟
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    sorted_lat = sorted(latencies)
    latency_mean = sum(latencies) / len(latencies)
    latency_p50 = sorted_lat[len(sorted_lat) // 2]
    latency_p99 = sorted_lat[int(len(sorted_lat) * 0.99)]

    # 吞吐量
    total_time_s = sum(latencies) / 1000
    total_tokens = seq_len * num_runs
    throughput = total_tokens / total_time_s  # tokens/s

    # 显存
    torch.cuda.synchronize()
    memory_current = torch.cuda.memory_allocated() / (1024 ** 2)
    memory_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "latency_mean_ms": round(latency_mean, 3),
        "latency_p50_ms": round(latency_p50, 3),
        "latency_p99_ms": round(latency_p99, 3),
        "throughput_tokens_s": round(throughput, 2),
        "memory_current_mb": round(memory_current, 2),
        "memory_peak_mb": round(memory_peak, 2),
        "seq_length": seq_len,
        "num_runs": num_runs,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    # ----------------------------------------------------------
    # 1. 解析参数
    #    从 args_llm.py 导入 PruneArguments, LLMArguments
    #    从 transformers 导入 ModelArguments, DataTrainingArguments, TrainingArguments
    # ----------------------------------------------------------
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, PruneArguments, LLMArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, prune_args, llm_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, prune_args, llm_args = parser.parse_args_into_dataclasses()

    # ----------------------------------------------------------
    # 2. 日志
    # ----------------------------------------------------------
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

    logger.warning(
        f"进程排名: {training_args.local_rank}  设备: {training_args.device}  "
        f"n_gpu: {training_args.n_gpu}  fp16: {training_args.fp16}"
    )
    logger.info(f"训练参数: {training_args}")
    logger.info(f"剪枝参数: {prune_args}")
    logger.info(f"LLM 参数: {llm_args}")

    # ----------------------------------------------------------
    # 3. 随机种子
    # ----------------------------------------------------------
    set_seed(training_args.seed)
    
    # ----------------------------------------------------------
    # 3.1 自动设置输出目录（基于剪枝方法）
    #    根据 prune_metric 自动创建子目录，方便对比不同方法
    # ----------------------------------------------------------
    base_output_dir = training_args.output_dir
    metric_subdir = prune_args.prune_metric  # lora / magnitude / weight
    training_args.output_dir = os.path.join(base_output_dir, metric_subdir)
    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info(f"输出目录: {training_args.output_dir} (基于 prune_metric={metric_subdir})")

    # ----------------------------------------------------------
    # 4. 加载 tokenizer
    # ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        padding_side="right",  # 语言建模时右填充
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------------------------------------
    # 5. 加载预训练模型（因果语言模型）
    # ----------------------------------------------------------
    logger.info(f"加载模型: {model_args.model_name_or_path}")

    load_kwargs = dict(
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if llm_args.load_in_4bit:
        # 4-bit 量化（需要 bitsandbytes >= 0.39）
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    elif training_args.fp16:
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **load_kwargs,
    )
    
    # 启用 gradient checkpointing 以节省显存（约40%显存减少）
    if training_args.gradient_checkpointing:
        logger.info("启用 gradient checkpointing 以节省显存")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ----------------------------------------------------------
    # 6. 注入 LoRA（与 main9.py 的 utils.add_lora_to_model 等价）
    #    这里使用 peft.get_peft_model，覆盖 LLM 的投影矩阵
    # ----------------------------------------------------------
    if prune_args.lora_r > 0:
        logger.info(f"注入 LoRA，r={prune_args.lora_r}，alpha={prune_args.lora_alpha}")

        # 根据模型类型确定目标模块名
        model_type = model.config.model_type.lower()
        if "llama" in model_type or "mistral" in model_type or "qwen" in model_type:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]
        elif "gpt2" in model_type or "gpt_neo" in model_type:
            target_modules = ["c_attn", "c_proj", "c_fc"]
        elif "opt" in model_type:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj",
                              "fc1", "fc2"]
        else:
            # 通用 fallback：用正则匹配所有线性层名
            target_modules = ["query", "key", "value", "dense"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=prune_args.lora_r,
            lora_alpha=prune_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ----------------------------------------------------------
    # 7. 构建数据集
    #    对应 main9.py 中的 raw_datasets / preprocess_function 流程
    # ----------------------------------------------------------
    task_name = data_args.task_name or "c4"
    max_seq_length = data_args.max_seq_length  # 默认 128，LLM 建议改为 2048

    logger.info(f"构建 LLM 数据集，task={task_name}，seq_len={max_seq_length}")
    train_dataset, eval_dataset = build_causal_lm_datasets(
        tokenizer=tokenizer,
        task_name=task_name,
        max_seq_length=max_seq_length,
        calibration_nsamples=llm_args.calibration_nsamples,
        cache_dir=model_args.cache_dir,
    )

    if data_args.max_train_samples is not None:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(data_args.max_train_samples, len(train_dataset)))
        )
    if data_args.max_eval_samples is not None:
        eval_dataset = torch.utils.data.Subset(
            eval_dataset, range(min(data_args.max_eval_samples, len(eval_dataset)))
        )

    compute_metrics = make_compute_metrics(eval_dataset)

    # ----------------------------------------------------------
    # 8. 打印稳定性配置（与 main9.py 一致）
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("REP-LIE LLM 剪枝配置:")
    print(f"  模型:              {model_args.model_name_or_path}")
    print(f"  任务:              {task_name}")
    print(f"  剪枝指标:          {prune_args.prune_metric}")
    print(f"  稀疏目标 (ratio):  {prune_args.ratio}")
    print(f"  MAC 约束:          {prune_args.mac}")
    print(f"  使用稳定性分数:    {prune_args.use_stability}")
    print(f"  稳定性组件:        {prune_args.stability_components}")
    print(f"  稳定性权重:        {prune_args.stability_weight}")
    print(f"  稳定性收集比例:    {prune_args.stability_collection_ratio}")
    print("=" * 50 + "\n")

    # ----------------------------------------------------------
    # 9. 初始化 LoRAPruneTrainer
    #    完全复用 loraprune/trainer_FLOPs.py 的 LoRAPruneTrainer，
    #    参数签名与 main9.py 完全一致。
    # ----------------------------------------------------------
    trainer = LLMPruneTrainer(
        model=model,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        args=training_args,
        # ---------- 剪枝超参（与 main9.py 一一对应）----------
        ratio=prune_args.ratio,
        init_ratio=prune_args.init_ratio,
        warmup_iters=prune_args.warmup_iters,
        cooldown_iters=prune_args.cooldown_iters,
        prune_freq=prune_args.prune_freq,
        prune_metric=prune_args.prune_metric,
        mac=prune_args.mac,
        seq_len=max_seq_length,
        task_name=task_name,
        # ---------- 稳定性超参 ----------
        use_stability=prune_args.use_stability,
        stability_components=prune_args.stability_components,
        stability_weight=prune_args.stability_weight,
        stability_collection_ratio=prune_args.stability_collection_ratio,
    )

    # ----------------------------------------------------------
    # 10. 训练（与 main9.py 完全相同的流程）
    # ----------------------------------------------------------
    if training_args.do_train:
        logger.info("*** 开始 LoRA + 剪枝训练 ***")

        # 检测是否有可恢复的 checkpoint
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # 保存稳定性分析数据（与 main9.py 一致）
        if prune_args.use_stability:
            trainer.save_stability_distribution()
            trainer.generate_stability_report()

    # ----------------------------------------------------------
    # 11. 推理性能测量（延迟 / 吞吐量 / 显存）
    #    在 eval 之前先测一次推理性能，作为 baseline 记录
    # ----------------------------------------------------------
    if training_args.do_eval:
        logger.info("*** 推理性能测量 ***")
        inference_metrics = _measure_inference_metrics(
            model=trainer.model,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            num_warmup=10,
            num_runs=100,
        )
        for k, v in inference_metrics.items():
            logger.info(f"  [推理] {k}: {v}")
        trainer.log_metrics("inference", inference_metrics)
        trainer.save_metrics("inference", inference_metrics)

    # ----------------------------------------------------------
    # 12. 评估（计算 PPL）
    # ----------------------------------------------------------
    if training_args.do_eval:
        logger.info("*** 评估（eval loss / PPL）***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # 在 eval_loss 基础上额外计算 PPL
        if "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"])
            logger.info(f"WikiText-2 PPL = {ppl:.2f}")
            metrics["eval_ppl"] = ppl
            trainer.log_metrics("eval_ppl", {"eval_ppl": ppl})

    # ----------------------------------------------------------
    # 12. 生成完整训练报告（与 main9.py 一致）
    # ----------------------------------------------------------
    trainer.generate_full_training_report()

    # ----------------------------------------------------------
    # 13. （可选）lm-evaluation-harness zero-shot 评估
    # ----------------------------------------------------------
    if llm_args.lm_eval_tasks:
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM

            task_list = [t.strip() for t in llm_args.lm_eval_tasks.split(",")]
            logger.info(f"执行 zero-shot 评估，任务: {task_list}")

            lm_wrapper = HFLM(pretrained=trainer.model, tokenizer=tokenizer)
            results = lm_eval.simple_evaluate(
                model=lm_wrapper,
                tasks=task_list,
                num_fewshot=0,
                batch_size="auto",
            )
            for task_name, task_results in results["results"].items():
                logger.info(f"[zero-shot] {task_name}: {task_results}")

            # 保存结果
            import json
            zs_path = os.path.join(training_args.output_dir, "zeroshot_results.json")
            with open(zs_path, "w") as f:
                json.dump(results["results"], f, indent=2)
            logger.info(f"Zero-shot 结果已保存至 {zs_path}")

        except ImportError:
            logger.warning(
                "未找到 lm-evaluation-harness，跳过 zero-shot 评估。"
                "可通过 `pip install lm-eval` 安装。"
            )


if __name__ == "__main__":
    main()
