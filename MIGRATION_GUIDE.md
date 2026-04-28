# REP-LIE LLM 剪枝项目 - 迁移包

## 项目结构

```
rep-lie-llm/
├── main_llm.py                 # 主入口（修改：自动创建子目录、添加time导入、gradient_checkpointing）
├── args_llm.py                 # 参数定义（已支持llama/mistral/qwen）
├── requirements.txt           # 依赖列表
├── setup.sh                    # 一键安装脚本
├── loraprune/
│   ├── __init__.py
│   ├── trainer_FLOPs.py        # 核心训练器（修复：TypeError、除零错误、perf_stats、grad_norm等）
│   ├── trainer_sb.py           # 稳定性训练器
│   ├── utils.py                # 工具函数（已支持LLM：动态head_dim、LLaMA层名识别）
│   └── utils1.py               # 额外工具
└── configs/                    # 可选：配置文件存放目录
```

## 修改记录汇总

### main_llm.py
| 行数 | 修改内容 |
|------|----------|
| 37 | 添加 `import time` |
| 388-391 | 添加 gradient_checkpointing 支持 |
| 385-393 | 自动基于 `prune_metric` 创建子目录 |

### trainer_FLOPs.py
| 行数 | 修改内容 |
|------|----------|
| 977 | 修复 `_maybe_log_save_evaluate` 缺少 `grad_norm` 参数 |
| 1232 | 修复 `perf_stats` → `self.perf_stats` |
| 1223 | 修复浮点赋值错误（时间/内存字段混淆） |
| 1302 | 添加除零检查 |
| 新增 | 添加模型性能指标（eval_loss, eval_ppl）到报告 |

### args_llm.py
| 修改内容 |
|----------|
| 支持 llama2-7b/13b, llama3-8b, mistral-7b, qwen2-7b |
| 新增 `load_in_4bit`, `lm_eval_tasks` 参数 |

### utils.py
| 修改内容 |
|----------|
| 动态 head_dim（移除硬编码 DIM=64） |
| LLaMA 层名识别（q_proj/k_proj/v_proj/o_proj + gate_proj/up_proj/down_proj） |
| 兼容 BERT（attention）+ LLaMA（self_attn）注意力层 |
| 兼容 BERT（intermediate）+ LLaMA（mlp）FFN 层 |

## 快速开始

```bash
# 1. 安装依赖
bash setup.sh

# 2. 运行 LLaMA-2-7B 剪枝（lora方法）
python main_llm.py \
    --model_name_or_path /path/to/Llama-2-7b-hf \
    --output_dir output/llm/llama-2-7b \
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
    --load_in_4bit \
    --prune_freq 32 \
    --use_stability True \
    --stability_collection_ratio 0.125

# 3. 运行 Mistral-7B 剪枝
python main_llm.py \
    --model_name_or_path /path/to/Mistral-7B-v0.1 \
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

## 支持的模型

| 模型 | model_name_or_path | 备注 |
|------|-------------------|------|
| LLaMA-2-7B | /path/to/Llama-2-7b-hf | 需要 HF 授权或本地文件 |
| LLaMA-3-8B | meta-llama/Meta-Llama-3-8B | 需要 HF 授权 |
| Mistral-7B | AI-ModelScope/Mistral-7B-v0.1 | ✅ 可用 |
| Qwen2-7B | Qwen/Qwen2-7B | ✅ 可用 |

## 支持的剪枝方法

```bash
--prune_metric lora      # LoRA重要性（推荐）
--prune_metric magnitude # 权重幅值
--prune_metric weight    # 权重本身
```

## 支持的评测任务

```
boolq, piqa, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa
```

## 实验对比指标

根据论文 Table 1，需要记录：
- Prune% (剪枝比例: 0%, 20%, 50%)
- #Params (参数量)
- PPL (困惑度)
- 7个零样本任务分数
- Avg. (平均分)
- Runtime / Memory (可选)
