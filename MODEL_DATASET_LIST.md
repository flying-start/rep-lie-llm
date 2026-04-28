# REP-LIE 模型与数据集清单

## 📁 已下载模型

| 模型 | 本地路径 | 来源 | 用途 |
|------|----------|------|------|
| Mistral-7B-v0.1 | `/root/autodl-tmp/models/AI-ModelScope/Mistral-7B-v0.1` | ModelScope | 主要实验模型 |
| Llama-2-7b-hf | `/root/autodl-tmp/models/shakechen/Llama-2-7b-hf` | 本地共享 | 对比实验 |

## 📥 模型下载命令

### Mistral-7B-v0.1（已完成）
```bash
modelscope download \
    --model AI-ModelScope/Mistral-7B-v0.1 \
    --local_dir /root/autodl-tmp/models/AI-ModelScope/Mistral-7B-v0.1
```

### Llama-2-7b-hf（已完成）
```bash
modelscope download \
    --model shakechen/Llama-2-7b-hf \
    --local_dir /root/autodl-tmp/models/shakechen/Llama-2-7b-hf
```

### Chinese-LLaMA-2-13B（待下载，用于大模型对比）
```bash
modelscope download \
    --model AI-ModelScope/Chinese-LLaMA-2-13B \
    --local_dir /root/autodl-tmp/models/Chinese-LLaMA-2-13B
```

### Qwen-13B（待下载，阿里开源）
```bash
modelscope download \
    --model Qwen/Qwen-13B \
    --local_dir /root/autodl-tmp/models/Qwen-13B
```

---

## 📂 数据集

| 数据集 | 说明 | 来源 |
|--------|------|------|
| c4 | 校准/训练数据集 | HuggingFace 自动下载 |
| hellaswag | Zero-shot 评测 | lm-evaluation-harness |
| arc_challenge | Zero-shot 评测 | lm-evaluation-harness |
| arc_easy | Zero-shot 评测 | lm-evaluation-harness |
| piqa | Zero-shot 评测 | lm-evaluation-harness |
| boolq | Zero-shot 评测 | lm-evaluation-harness |
| winogrande | Zero-shot 评测 | lm-evaluation-harness |
| openbookqa | Zero-shot 评测 | lm-evaluation-harness |
|wikitext2 | 训练数据集 | HuggingFace 自动下载 |
**说明**：c4 数据集会根据 `HF_ENDPOINT` 环境变量自动从镜像源下载，无需手动下载。

---

## 🔧 环境变量（AutoDL 已配置）

```bash
cat >> ~/.bashrc << 'EOF'
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
export HF_ENDPOINT=https://hf-mirror.com
# export HF_TOKEN=hf_xxxxxxxxxxxxx  # 如果用 LLaMA
EOF
```

---

## 🚀 快速运行命令

### Mistral-7B + Lora剪枝
```bash
python main_llm.py \
    --model_name_or_path /root/autodl-tmp/models/AI-ModelScope/Mistral-7B-v0.1 \
    --output_dir output/llm/mistral-7b \
    --prune_metric lora \
    --task_name c4 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --do_train \
    --do_eval \
    --fp16 \
    --lm_eval_tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
    --prune_freq 32 \
    --use_stability True \
    --stability_collection_ratio 0.125
```

### Llama-2-7B + Lora剪枝
```bash
python main_llm.py \
    --model_name_or_path /root/autodl-tmp/models/shakechen/Llama-2-7b-hf \
    --output_dir output/llm/llama2-7b \
    --prune_metric lora \
    --task_name c4 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --do_train \
    --do_eval \
    --fp16 \
    --lm_eval_tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa \
    --prune_freq 32 \
    --use_stability True \
    --stability_collection_ratio 0.125
```
