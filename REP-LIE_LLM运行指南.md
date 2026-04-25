# REP-LIE LLM 剪枝运行指南

**版本**：v1.0  
**适用模型**：LLaMA-2/7B、Mistral-7B、Qwen2-7B、GPT-2 等 CausalLM 架构  
**代码版本**：D:/rep-lie，基于 2026-04-24 修复后的代码

---

## 一、运行命令示例

### 1.1 标准运行命令（LLaMA-2-7B，50% 稀疏率）

```bash
cd D:/rep-lie

python main_llm.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --task_name c4 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --output_dir output/llm/llama2-7b-s50 \
    --ratio 0.5 \
    --mac 0.5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --prune_metric lora \
    --use_stability True \
    --stability_weight 0.1 \
    --stability_components attention,ffn \
    --stability_collection_ratio 0.1 \
    --do_train \
    --do_eval \
    --fp16
```

### 1.2 量化运行（节省显存，推荐 16GB GPU）

```bash
python main_llm.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --task_name c4 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --output_dir output/llm/llama2-7b-s50-4bit \
    --ratio 0.5 \
    --mac 0.5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --prune_metric lora \
    --use_stability True \
    --load_in_4bit True \
    --do_train \
    --do_eval
```

### 1.3 Mistral-7B 运行

```bash
python main_llm.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --task_name c4 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --output_dir output/llm/mistral-7b-s50 \
    --ratio 0.5 \
    --mac 0.5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --prune_metric lora \
    --use_stability True \
    --fp16
```

### 1.4 WikiText 校准集（替代 C4，离线可用）

```bash
python main_llm.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --task_name wikitext \
    --calibration_nsamples 1024 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --output_dir output/llm/llama2-7b-wt \
    --ratio 0.5 \
    --mac 0.5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --prune_metric lora \
    --use_stability True \
    --fp16
```

### 1.5 从 checkpoint 恢复训练

```bash
python main_llm.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --task_name c4 \
    --max_seq_length 2048 \
    --output_dir output/llm/llama2-7b-s50 \
    --ratio 0.5 \
    --mac 0.5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --prune_metric lora \
    --resume_from_checkpoint output/llm/llama2-7b-s50/checkpoint-100 \
    --do_train \
    --fp16
```

---

## 二、模型加载注意事项

### 2.1 模型来源与认证

| 模型 | HuggingFace 路径 | 是否需要 Token |
|------|-----------------|---------------|
| LLaMA-2-7B | `meta-llama/Llama-2-7b-hf` | ✅ 需要（申请 Meta 后在 HF 设置） |
| LLaMA-3-8B | `meta-llama/Meta-Llama-3-8B` | ✅ 需要 |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | ❌ 不需要 |
| Qwen2-7B | `Qwen/Qwen2-7B` | ❌ 不需要 |
| GPT-2 | `openai-community/gpt2` | ❌ 不需要 |

**Token 配置方式（任选其一）：**

```bash
# 方式 1：环境变量（推荐）
export HF_TOKEN=your_huggingface_token_here

# 方式 2：命令行参数（不安全，不推荐）
# main_llm.py 目前不支持 --use_auth_token 参数
# 需要手动修改代码或通过环境变量传递
```

### 2.2 显存估算

| 模型 | 参数量 | FP16 显存 | 4-bit 量化显存 | 推荐配置 |
|------|--------|---------|---------------|---------|
| LLaMA-2-7B | 6.7B | ~14GB | ~4GB | 16GB GPU 或量化 |
| LLaMA-3-8B | 8B | ~16GB | ~5GB | 24GB GPU 或量化 |
| Mistral-7B | 7B | ~14GB | ~4GB | 16GB GPU 或量化 |
| Qwen2-7B | 7B | ~14GB | ~4GB | 16GB GPU 或量化 |

**显存组成估算：**
```
总显存 ≈ 模型权重 + 优化器状态 + 梯度 + 激活值 + LoRA参数 + 掩码缓存

FP16 训练（LLaMA-2-7B，batch=1, seq_len=2048）：
  ≈ 14GB（权重）+ 14GB（Adam优化器）+ 7GB（梯度）+ 1GB（激活）+ 0.1GB（LoRA）
  ≈ 36GB → 需要 A100 40GB 或 A800

4-bit 量化 + LoRA（LLaMA-2-7B）：
  ≈ 4GB（量化权重）+ 0.5GB（量化优化器）+ 7GB（梯度）+ 1GB（激活）
  ≈ 12.5GB → 可在 16GB GPU（如 4090、A5000）运行
```

### 2.3 dtype 配置优先级

```python
# main_llm.py 中的加载逻辑（第 336-350 行）：
if llm_args.load_in_4bit:
    # 4-bit 量化加载（最省显存）
    # 需要：pip install bitsandbytes>=0.39
    torch_dtype = bfloat16（bnb 内部处理）

elif training_args.fp16:
    # FP16 加载（中等显存）
    torch_dtype = float16

else:
    # BF16 加载（最高精度，需要 Ampere+ GPU）
    torch_dtype = bfloat16
```

**推荐配置：**
- A100/H100/A800：使用 `--fp16 False`（默认 BF16）
- 3090/4090/A5000：使用 `--fp16` 或 `--load_in_4bit True`
- V100：仅支持 FP16，不支持 BF16

### 2.4 可能出现的问题

#### ❌ 问题 1：HuggingFace Token 缺失

```
OSError: Access token does not exist or is invalid.
```

**解决：**
```bash
# 在 Linux/Mac
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 在 Windows PowerShell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 或登录 HuggingFace CLI
huggingface-cli login
```

#### ❌ 问题 2：模型下载失败（网络问题）

```
ConnectionError: Could not reach the HuggingFace Hub
```

**解决：**
```bash
# 设置镜像（国内）
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后指定本地路径
--model_name_or_path /path/to/local/llama2-7b-hf
```

#### ❌ 问题 3：Tokenizer 缺失

```
ValueError: Could not find the tokenizer file
```

**原因：** 指定了 `--tokenizer_name` 但 tokenizer 文件不存在  
**解决：** 确保 tokenizer 与模型在同一目录，或不指定 `--tokenizer_name`

#### ❌ 问题 4：量化依赖缺失

```
ImportError: cannot import name 'BitsAndBytesConfig' from 'transformers'
```

**解决：**
```bash
pip install bitsandbytes>=0.39
pip install --upgrade transformers accelerate
```

---

## 三、数据集注意事项

### 3.1 数据集选择

| 数据集 | 说明 | 适用场景 |
|--------|------|---------|
| `c4`（默认） | C4 英文数据集，streaming 模式，512-1024 样本 | 通用校准，推荐 |
| `wikitext` | WikiText-2，离线可用 | 网络受限环境、复现对比 |
| 自定义 | 需实现类似 `_TokenDataset` 接口 | 特定领域适配 |

### 3.2 关键参数

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--calibration_nsamples` | 512 | 256-1024 | 校准样本越多，重要性估计越准确，但训练越慢 |
| `--max_seq_length` | 128（BERT） | **2048** | LLM 必须设为 2048 |
| `--task_name` | None | `c4` | 指定数据集来源 |

### 3.3 数据集加载流程

```
build_causal_lm_datasets()
    │
    ├── task_name == "c4" or "llm_causal"
    │       │
    │       └── load_dataset("allenai/c4", "en", streaming=True)
    │               └── 采样 calibration_nsamples 条
    │               └── 过滤：input_ids >= 16
    │
    └── task_name == "wikitext"
            │
            └── load_dataset("wikitext", "wikitext-2-raw-v1")
                    └── 按 max_seq_length 分块
```

### 3.4 可能出现的问题

#### ❌ 问题 1：C4 数据集下载失败

```
DatasetDownloadError: Could not download https://huggingface.co/datasets/allenai/c4
```

**解决：**
```bash
# 方式 1：设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方式 2：切换到 WikiText（离线可用）
--task_name wikitext

# 方式 3：手动下载后缓存
# 设置 cache_dir 指向已下载目录
--cache_dir /path/to/cache
```

#### ❌ 问题 2：序列长度不匹配警告

```
UserWarning: Token indices sequence length is longer than the specified maximum sequence length
```

**原因：** max_seq_length 设置过小，tokenizer 输出被截断  
**解决：** 确保 `--max_seq_length 2048`（LLM 标准）

#### ❌ 问题 3：校准集为空

```
ValueError: train_dataset must have at least 1 sample
```

**原因：** C4 采样未成功，或所有样本都被过滤（input_ids < 16）  
**解决：**
1. 检查网络连接
2. 增加 `--calibration_nsamples`
3. 切换到 wikitext：`--task_name wikitext`

#### ❌ 问题 4：Streaming 模式数据量不足

```
# C4 streaming 可能采不到足够样本
# 原因：C4 的 en 子集可能不包含足够高质量样本
```

**解决：**
```python
# 临时修改 main_llm.py 第 106 行
# 将 input_ids >= 16 改为 >= 8
if len(tokens["input_ids"]) >= 8:
```

---

## 四、剪枝参数注意事项

### 4.1 核心剪枝参数

| 参数 | 必选 | 推荐值 | 说明 |
|------|------|--------|------|
| `--ratio` | ✅ | 0.3-0.7 | 目标稀疏率（0.5 = 剪掉 50% 的注意力头/FFN神经元） |
| `--mac` | ✅ | 与 ratio 接近 | MAC 约束上限，建议设为与 ratio 相同 |
| `--lora_r` | ✅ | 8-16 | LoRA 秩，越大越强但显存越高 |
| `--lora_alpha` | ✅ | 2×lora_r | LoRA 缩放因子 |
| `--prune_metric` | ✅ | `lora`/`magnitude`/`grad` | 剪枝指标：LoRA梯度/权重幅值/梯度幅值 |

### 4.2 稀疏率调度

| 阶段 | 参数 | 说明 |
|------|------|------|
| 预热期 | `--warmup_iters` | 不剪枝，LoRA 预热，默认 100 |
| 渐进期 | `--prune_freq` | 每 N 步剪枝一次，默认 10 |
| 冷却期 | `--cooldown_iters` | 训练末期不再剪枝，默认 100 |

**示例（总步数 1000）：**
```bash
--warmup_iters 100 \
--prune_freq 10 \
--cooldown_iters 100 \
--num_train_epochs 1
# → 实际剪枝步数 ≈ (1000 - 100 - 100) / 10 = 80 次
```

### 4.3 稳定性参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_stability` | True | 是否使用稳定性分数加权 |
| `--stability_components` | `attention,ffn` | 稳定性作用于哪些组件 |
| `--stability_weight` | 0.1 | 稳定性权重（越大越保守剪枝） |
| `--stability_collection_ratio` | 0.1 | 前 10% 步收集稳定性分数 |

### 4.4 可能出现的问题

#### ❌ 问题 1：ratio 与 mac 不一致导致约束失效

```
WARNING: MAC constraint may not be satisfied. ratio=0.5, mac=0.8
```

**原因：** ratio（稀疏率）和 mac（MAC 约束）是两个独立约束  
**解决：** 通常设为相同值：`--ratio 0.5 --mac 0.5`

#### ❌ 问题 2：所有模块被剪掉

```
ValueError: Cannot prune all neurons. pruning_dict is empty.
```

**原因：** ratio 过大（>0.9）或稀疏调度过快  
**解决：**
1. 降低 ratio：`--ratio 0.7`
2. 调整调度：增加 warmup，减少 prune_freq

#### ❌ 问题 3：mac_per_neuron 计算错误（SwiGLU）

```
RuntimeError: Expected mac_per_neuron for SwiGLU to be 3x, got 2x
```

**检查：** utils.py 第 1243-1259 行的 `mac_per_neuron` 函数  
**原因：** 未传入 `model_type` 参数  
**状态：** 代码已修复，main_llm.py 会透传 `model.config.model_type`

#### ❌ 问题 4：head_dim 不匹配

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x128 and 64x64)
```

**原因：** BERT 的 head_dim=64，但 LLaMA 的 head_dim=128  
**状态：** 代码已修复，使用 `model.config.hidden_size // model.config.num_attention_heads` 动态获取

---

## 五、训练过程中的问题

### 5.1 显存不足（OOM）

#### 诊断步骤：
```python
# 检查当前显存占用
torch.cuda.memory_allocated() / 1024**3  # GB
torch.cuda.max_memory_allocated() / 1024**3  # 峰值
```

#### 解决方案（按优先级）：

| 方案 | 效果 | 命令 |
|------|------|------|
| 启用 4-bit 量化 | 显存 -60% | `--load_in_4bit True` |
| 减少 batch_size | 显存 -30%/步 | `--per_device_train_batch_size 1` |
| 增加梯度累积 | 显存不变，训练慢 | `--gradient_accumulation_steps 32` |
| 使用 gradient checkpointing | 显存 -40% | `--gradient_checkpointing True` |
| 减少 max_seq_length | 显存 -50% | `--max_seq_length 1024` |

### 5.2 训练loss不收敛

#### 检查清单：
1. **学习率**：LLM 的 LoRA 通常需要更高的学习率
   - BERT：2e-5
   - LLaMA/Mistral：1e-4 到 3e-4
   
2. **warmup_ratio**：
   ```bash
   --warmup_ratio 0.1  # 前 10% 步预热
   ```

3. **稳定性分数未收集**：
   ```bash
   --use_stability True
   --stability_collection_ratio 0.1
   ```

### 5.3 PPL 不降反升

**可能原因：**
1. 稀疏率过高（>0.7）
2. LoRA rank 过低（<4）
3. 训练步数不足

**诊断：**
```bash
# 查看训练日志中的 eval_loss 变化
# 正常情况：eval_loss 逐渐下降
# 异常情况：eval_loss 波动或上升
```

---

## 六、评测相关

### 6.1 训练后 PPL 评测

训练完成后，查看日志中的 PPL：
```
INFO:wikitext2 ppl = 12.34
```

或查看输出目录：
```bash
cat output/llm/llama2-7b-s50/eval_results.json
```

### 6.2 使用 lm-evaluation-harness（推荐）

训练完成后，用 lm-evaluation-harness 进行标准评测：

```bash
pip install lm-eval

python -m lm_eval \
    --model hf \
    --model_args pretrained=output/llm/llama2-7b-s50,tokenizer=meta-llama/Llama-2-7b-hf,dtype=bfloat16 \
    --tasks wikitext,ptb,pile \
    --batch_size 8 \
    --limit 1000
```

### 6.3 零样本评测（已在 main_llm.py 实现）

```bash
python main_llm.py \
    --model_name_or_path output/llm/llama2-7b-s50 \
    --lm_eval_tasks hellaswag,arc_easy,winogrande \
    --do_eval
```

---

## 七、快速检查清单

运行前请确认：

```
✅ 模型路径正确（HuggingFace ID 或本地路径）
✅ 已设置 HF_TOKEN（私有模型）
✅ 显存足够（16GB+ 推荐 4bit 量化）
✅ 网络可访问 HuggingFace（加载 C4 数据集）
✅ --max_seq_length 设为 2048
✅ --task_name 为 c4 或 wikitext
✅ --ratio 和 --mac 设为相同值
✅ --lora_r >= 8
```

---

## 八、环境依赖

```bash
# 基础依赖
pip install torch transformers datasets peft accelerate

# 量化支持（可选）
pip install bitsandbytes>=0.39

# 评测支持（可选）
pip install lm-eval

# 代码检查
pip install ruff  # 代码风格检查
```

---

*文档版本：v1.0*
*生成时间：2026-04-24*
*基于代码：D:/rep-lie/main_llm.py*
