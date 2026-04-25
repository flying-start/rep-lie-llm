# REP-LIE LLM 适配性深度代码审查报告

**版本**：v1.0  
**审查时间**：2026-04-25  
**审查范围**：`config_helpers_llm.py` / `config_llm.py` / `data_utils_llm.py` / `main_llm.py`，对照 `loraprune/utils.py` 和 `loraprune/trainer_FLOPs.py` 剪枝逻辑进行一致性校验  
**审查结论**：发现 **3 处真实 Bug 和 4 处设计隐患**，需要修改代码。

---

## 一、调用关系梳理

### 1.1 main8.py 与 LLM 四个文件的关系

**关键发现：`main8.py` 完全不调用这四个文件。**

```
main8.py
    ├── from config import *         ← config.py（BERT/ViT 原版）
    ├── from data_utils import ...   ← data_utils.py（原版）
    └── 直接调用 LoRAPruneTrainer    ← 不走 config_llm.py 的 pruning_params
```

`config_llm.py`、`config_helpers_llm.py`、`data_utils_llm.py` 这三个文件是为 `main_llm.py` 准备的，但 **`main_llm.py` 目前也没有 import 它们**：

```python
# main_llm.py 中的实际 imports（无 config_llm 相关）
from args import ModelArguments, DataTrainingArguments, PruneArguments
from trainer_FLOPs import LoRAPruneTrainer as LLMPruneTrainer
import utils as prune_utils
```

### 1.2 剪枝逻辑的实际数据流

**实际调用链（main_llm.py 真实路径）：**

```
main_llm.py
    │
    ├── 加载模型（AutoModelForCausalLM）
    ├── 注入 LoRA（peft.get_peft_model）  ← 直接指定 target_modules
    ├── build_causal_lm_datasets()          ← 内嵌的数据集逻辑，不调用 data_utils_llm.py
    │
    └── LLMPruneTrainer（from trainer_FLOPs.py）
            │
            ├── utils.init_sensitivity_dict(model)    ← 实际剪枝逻辑入口
            ├── utils.update_sensitivity_dict(...)    ← 重要性分数计算
            └── utils.search_mac_change(...)          ← MAC 约束搜索
                    └── utils.balanced_pruning(...)   ← 掩码生成
```

**结论：`config_llm.py` 的 `pruning_params`（granularity / gqa_kv_groups / coupled_proj）对剪枝核心逻辑完全未生效。**

---

## 二、核心问题分析

### 🔴 Bug 1：`config_llm.py` 的 granularity 对剪枝逻辑无任何影响

**现象：**

`config_llm.py` 为 llama2-7b 精心设置了：
```python
'attn': {
    'granularity': [512, 4096],  # GQA：4 heads/group * 128 = 512
    'gqa_kv_groups': 8,
},
'ffn': {
    'granularity': [1, 4096],
    'coupled_proj': True,        # gate/up 必须耦合剪枝
}
```

**但 `utils.py` 的 `search_mac_change()` 和 `balanced_pruning()` 根本不接受 granularity 参数：**

```python
# utils.py search_mac_change 函数签名（第 777-783 行）
def search_mac_change(
    model,
    importance_scores,
    seq_len,
    mac_constraint,
    mask_dict=None
):
    # 内部直接读 model.config，从不接受 granularity 参数
    head_dim = hidden_size // num_attention_heads  # 固定按 1 head 为粒度
```

**影响：**
- GQA 的 `gqa_kv_groups=8` 参数被完全忽略
- `coupled_proj=True`（gate/up 耦合剪枝）标志被完全忽略
- `granularity=[512, 4096]` 设定完全无效

---

### 🔴 Bug 2：GQA 架构下 K/V 掩码维度与 Q 不一致，导致形状错误

**背景：**  
LLaMA-2-7B 采用 GQA（Grouped Query Attention）：
- Q: 32 heads，hidden_size=4096，head_dim=128
- K/V: 8 heads，hidden_size=1024（KV head dim = 128）

**当前 `balanced_pruning()` 展开掩码的逻辑（第 938-943 行）：**

```python
# 当前代码：Q/K/V/O 使用相同的掩码展开方式
if "self_attn" in key and model.config.model_type in ('llama', 'mistral', 'qwen', ...):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    # head_dim = 4096 // 32 = 128
    mask_dict[key+'.q_proj'] = mask.repeat_interleave(head_dim)  # Q: [32] → [4096] ✅
    mask_dict[key+'.k_proj'] = mask.repeat_interleave(head_dim)  # K: [32] → [4096] ❌ 应为 [1024]
    mask_dict[key+'.v_proj'] = mask.repeat_interleave(head_dim)  # V: [32] → [4096] ❌ 应为 [1024]
    mask_dict[key+'.o_proj'] = mask.repeat_interleave(head_dim)  # O: [32] → [4096] ✅（输入维度）
```

**问题：**  
对于 LLaMA-2-7B GQA：
- `k_proj.weight.shape = [1024, 4096]`（8 KV heads × head_dim 128 = 1024）
- `v_proj.weight.shape = [1024, 4096]`
- 当前代码生成的 K/V 掩码是 `[4096]`，而实际 K/V 输出维度是 `[1024]`
- 调用 `apply_masked_modules` 时：`output * mask` → **shape mismatch → RuntimeError**

**严重性：⭐⭐⭐⭐（会直接导致运行崩溃）**

---

### 🔴 Bug 3：`update_sensitivity_dict` 的 FFN 分支从未写入 `new_s_dict`

**当前代码（第 169-184 行）：**

```python
for name, module in model.named_modules():
    if not _is_target_larer(module):
        continue
    last_part = name.split('.')[-1]
    
    is_attn = 'attention' in name or last_part in ('q_proj', 'k_proj', 'v_proj', 'o_proj')
    is_output = last_part == 'output.dense'
    intermediate = 'intermediate' in name and 'attention' not in name
    
    sensitivity = compute_sensitivity(module, is_attn, is_output, intermediate, ...)
    
    if is_attn:
        name = ".".join(name.split('.')[:-2])
        new_s_dict[name] += sensitivity.to(new_s_dict[name].device)
    # ← FFN 分支（is_ffn）没有写入 new_s_dict！
```

**缺失的 FFN 写入逻辑：**  
当 `is_attn == False`（即 FFN 层 gate_proj / up_proj / down_proj）时，`new_s_dict` 不会被更新。  
虽然 `init_sensitivity_dict` 对 FFN 层初始化了 `sensitivity_record[name] = mask`，但 `update_sensitivity_dict` 的循环不会给这些 FFN 条目赋值，最终所有 FFN 重要性分数永远保持初始的全1掩码。

**影响：FFN 层重要性分数永远不更新 → 所有 FFN 神经元重要性相同 → FFN 剪枝变成随机剪枝。**

---

### ⚠️ 隐患 1：`prune_one_layer()` 中 DIM 硬编码为 64 未修复

```python
# utils.py 第 264 行
def prune_one_layer(layer):
    ...
    layer.self_attn.num_heads = int(layer.self_attn.q_proj.lora_mask.sum()) // DIM
    # DIM 在此函数作用域内未定义（全局 DIM=64 已被删除注释掉）
    # LLaMA 应为 head_dim=128，此处会产生 NameError 或引用错误值
```

**注意：** 这个 `DIM` 是 `balanced_pruning` 函数开头的局部变量，但 `prune_one_layer` 是独立函数，它引用的 `DIM` 不存在（全局 `DIM=64` 已被注释掉），会在运行时抛出 `NameError`。

---

### ⚠️ 隐患 2：`config_helpers_llm.py` 与 `compression/pruner1.py` 完全断开

`config_helpers_llm.py` 定义了精心设计的 `get_prune_config_for_attn()`，但：

```python
# compression/pruner1.py 第 5 行
from config_helpers import *   # ← 原版，不是 config_helpers_llm
```

`compression/speedup.py` 使用的是原版 `config_helpers.py`，所有 LLM 的 granularity 配置被绕过。  
这说明这套配置是为 NNI-style 结构化剪枝准备的（pruner1.py），而当前的软掩码剪枝（utils.py）绕过了它。

---

### ⚠️ 隐患 3：`data_utils_llm.py` 与 `main_llm.py` 各自独立实现相同逻辑

`main_llm.py` 内嵌了 `build_causal_lm_datasets()` 函数（第 78-196 行），而 `data_utils_llm.py` 单独实现了 `prepare_datasets_llm_causal()`，两者功能高度重复但实现细节不同：

| 差异点 | main_llm.py 内嵌版 | data_utils_llm.py 版 |
|--------|-------------------|---------------------|
| 校准样本策略 | 过滤 input_ids >= 16 | 拼接 token 后切分，更准确 |
| 评估集 | 单独读取 wikitext test | 支持 wikitext2/ptb/c4 |
| 数据 labels | ids.clone() | ids.clone() |
| 错误处理 | 无 | 有警告 + 重复填充 |
| PTB 支持 | ❌ | ✅ |

**推荐：`main_llm.py` 应 import `data_utils_llm.py`，消除重复代码。**

---

### ⚠️ 隐患 4：`config_llm.py` 的 training_params 对 main_llm.py 完全未使用

`config_llm.py` 定义了 LLM 的学习率、batch_size 等配置：
```python
'llama2-7b': {
    'learning_rate': 2e-5,
    'batch_size': 1,
    'gradient_accumulation': 16,
}
```

但 `main_llm.py` 中这些参数通过命令行 `TrainingArguments` 传入，完全绕过了 `Config` 类。

---

## 三、剪枝粒度对齐性总结

### 3.1 注意力头剪枝粒度

| 组件 | config_llm.py 设置 | utils.py 实际行为 | 是否匹配 |
|------|-------------------|-----------------|---------|
| Q 粒度 | [512, 4096]（4heads/group） | head_dim=128，按 1 head 剪 | ❌ 不匹配 |
| K/V 粒度 | GQA 感知（8 KV heads） | 与 Q 相同展开 → shape error | ❌ Bug |
| O 粒度 | 反转 Q 粒度 | 与 Q 相同展开 | 部分正确 |

### 3.2 FFN 剪枝粒度

| 组件 | config_llm.py 设置 | utils.py 实际行为 | 是否匹配 |
|------|-------------------|-----------------|---------|
| gate_proj / up_proj 耦合 | coupled_proj=True | 独立剪枝，互不影响 | ❌ Bug |
| down_proj 反向依赖 | granularity 反转 | 独立剪枝 | ❌ Bug |
| FFN 重要性分数更新 | - | 从未写入 | ❌ Bug |

### 3.3 MAC 计算

| 项目 | 状态 | 说明 |
|------|------|------|
| SwiGLU 3× | ✅ | mac_per_neuron 已修复 |
| intermediate_size（LLaMA） | ⚠️ | LLaMA 的 intermediate_size 是 ffn_dim（11008），与 hidden_size(4096) 不同，当前正确读取 |
| GQA KV heads 计算 | ❌ | search_mac_change 用 num_attention_heads（32），应用 num_key_value_heads（8） |

---

## 四、需要修改的代码清单

### P0（必须修复，否则运行崩溃）

#### Fix 1：`balanced_pruning` 中 K/V 掩码 shape 修复

```python
# 位置：loraprune/utils.py，balanced_pruning()，第 938 行附近
# 修改前（Q/K/V 使用相同 head_dim）：
if "self_attn" in key and model.config.model_type in ('llama', 'mistral', 'qwen', ...):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    mask_dict[key+'.k_proj'] = mask.repeat_interleave(head_dim)  # ❌ GQA 时 shape 错误
    mask_dict[key+'.v_proj'] = mask.repeat_interleave(head_dim)  # ❌

# 修改后（区分 Q 和 KV 的 head_dim）：
if "self_attn" in key and model.config.model_type in ('llama', 'mistral', 'qwen', 'gpt2', 'gpt_neox'):
    q_head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
    kv_head_dim = model.config.hidden_size // model.config.num_attention_heads
    # GQA: K/V mask 需要压缩到 KV head 数
    num_q_heads = model.config.num_attention_heads
    group_size = num_q_heads // num_kv_heads  # e.g., 32//8=4
    kv_mask = mask.view(-1, group_size)[:, 0]  # 按 group 取第一个 Q head → [num_kv_heads]
    
    mask_dict[key+'.q_proj'] = mask.repeat_interleave(q_head_dim)
    mask_dict[key+'.k_proj'] = kv_mask.repeat_interleave(kv_head_dim)
    mask_dict[key+'.v_proj'] = kv_mask.repeat_interleave(kv_head_dim)
    mask_dict[key+'.o_proj'] = mask.repeat_interleave(q_head_dim)
```

#### Fix 2：`update_sensitivity_dict` 增加 FFN 分支写入

```python
# 位置：loraprune/utils.py，update_sensitivity_dict()，第 179 行附近
# 修改前（只有 is_attn 分支写入）：
if is_attn:
    name = ".".join(name.split('.')[:-2])
    new_s_dict[name] += sensitivity.to(new_s_dict[name].device)
# FFN 分支完全缺失 ❌

# 修改后（增加 FFN 分支）：
if is_attn:
    name = ".".join(name.split('.')[:-2])
    if name in new_s_dict:
        new_s_dict[name] += sensitivity.to(new_s_dict[name].device)
else:
    # FFN 层：gate_proj/up_proj/down_proj（LLaMA），intermediate/output（BERT）
    if name in new_s_dict:
        new_s_dict[name] = new_s_dict[name] + sensitivity.to(new_s_dict[name].device)
```

#### Fix 3：`prune_one_layer` 中 DIM 变量 NameError

```python
# 位置：loraprune/utils.py，prune_one_layer()，第 264 行
# 修改前（DIM 未定义，会 NameError）：
layer.self_attn.num_heads = int(layer.self_attn.q_proj.lora_mask.sum()) // DIM

# 修改后（动态从 config 获取）：
head_dim = layer.self_attn.q_proj.weight.shape[0] // layer.self_attn.num_heads
# 或者直接从 layer.config 拿（需要传入 model.config）
# 暂时用权重形状反推：
head_dim = 128  # LLaMA 固定值，或从上下文传入
layer.self_attn.num_heads = int(layer.self_attn.q_proj.lora_mask.sum()) // head_dim
```

### P1（建议修复，影响正确性）

#### Fix 4：`search_mac_change` 中 GQA 的 KV heads 计数修正

```python
# 位置：loraprune/utils.py，search_mac_change()，第 786 行附近
# 修改前：
num_attention_heads = model.config.num_attention_heads  # 32（Q heads）
# 这导致 FFN MAC 计算基于 Q heads 而非 KV heads

# 修改后：
num_attention_heads = model.config.num_attention_heads  # 32（Q heads，用于掩码生成）
num_kv_heads = getattr(model.config, 'num_key_value_heads', num_attention_heads)  # 8（KV heads）
# 在 compute_mac 计算时，使用 num_kv_heads 计算 KV 部分的 FLOPs
```

---

## 五、文件定位与职责梳理

### 当前状态（2026-04-25）

```
main8.py                 ← BERT/ViT 入口，调用 config.py、data_utils.py
main_llm.py              ← LLM 入口，调用 loraprune/utils.py（剪枝核心）
                            但不调用 config_llm.py / config_helpers_llm.py / data_utils_llm.py

config_llm.py            ← 定义了精细的 GQA/SwiGLU 剪枝粒度配置
                            ⚠️ 当前无代码调用，配置无效

config_helpers_llm.py    ← 为 NNI 风格剪枝准备的 config_list 生成函数
                            ⚠️ 当前无代码调用

data_utils_llm.py        ← LLM 数据集工具
                            ⚠️ main_llm.py 有内嵌版本，两套代码独立运行

loraprune/utils.py       ← 实际的剪枝核心（软掩码）
                            有 3 个 Bug，需要修复
```

### 建议的正确调用关系

```
main_llm.py
    │
    ├── from config_llm import Config           ← 读取 LLM 训练参数
    ├── from data_utils_llm import prepare_datasets  ← 统一数据加载
    │
    └── loraprune/utils.py（剪枝核心修复后）
            ├── GQA-aware balanced_pruning
            ├── SwiGLU-coupled FFN pruning
            └── FFN 重要性分数正确写入
```

---

## 六、优先级总结

| 编号 | 问题 | 严重性 | 文件 | 行号 |
|------|------|--------|------|------|
| F1 | GQA K/V 掩码 shape 错误 → RuntimeError | P0 | utils.py | ~938 |
| F2 | FFN 重要性分数从不更新 → 随机剪枝 | P0 | utils.py | ~179 |
| F3 | prune_one_layer 中 DIM NameError | P0 | utils.py | 264 |
| F4 | search_mac_change GQA FLOPs 计算偏差 | P1 | utils.py | ~786 |
| F5 | config_llm.py 的 granularity 对剪枝无效 | P1 | main_llm.py | 导入缺失 |
| F6 | data_utils_llm.py 与内嵌版本重复 | P2 | main_llm.py | 78-196 |

---

*报告生成时间：2026-04-25*  
*基于代码的完整静态分析，所有结论均有代码行号支撑*
