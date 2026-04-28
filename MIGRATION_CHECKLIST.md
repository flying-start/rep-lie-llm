# REP-LIE LLM 迁移文件清单
# 在新服务器上，将这些文件复制到对应目录即可

## 必需文件（已修改）

### 根目录
D:/rep-lie/main_llm.py          # 主入口
D:/rep-lie/args_llm.py          # 参数定义
D:/rep-lie/requirements.txt     # 依赖列表
D:/rep-lie/setup.sh             # 安装脚本
D:/rep-lie/MIGRATION_GUIDE.md   # 迁移指南

### loraprune/ 目录
D:/rep-lie/loraprune/trainer_FLOPs.py   # 核心训练器
D:/rep-lie/loraprune/utils.py           # 工具函数（LLM兼容）
D:/rep-lie/loraprune/utils1.py          # 额外工具
D:/rep-lie/loraprune/__init__.py        # 包初始化

## 迁移步骤

1. 复制整个 D:/rep-lie/ 目录到新服务器
创建虚拟环境
conda create -n rep-lie python=3.11
激活虚拟环境
conda activate rep-lie
修改环境变量
cat >> ~/.bashrc << 'EOF'
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
export HF_ENDPOINT=https://hf-mirror.com
# export HF_TOKEN=hf_xxxxxxxxxxxxx  # 如果用 LLaMA
EOF
source ~/.bashrc

2. 安装依赖：
   - 方式A: pip install -r requirements.txt
   - 方式B: bash setup.sh
3. 配置模型路径（下载模型或修改路径）
4. 运行实验

## 本地模型路径映射

| 模型 | 原路径 | 新服务器路径 |
|------|--------|-------------|
| Mistral-7B | /root/autodl-tmp/models/AI-ModelScope/Mistral-7B-v0.1 | ??? |
| LLaMA-2-7B | /root/autodl-tmp/models/shakechen/Llama-2-7b-hf | ??? |

## 已验证的修改

### main_llm.py
- ✅ import time
- ✅ gradient_checkpointing 支持
- ✅ 自动创建 prune_metric 子目录

### trainer_FLOPs.py
- ✅ _maybe_log_save_evaluate grad_norm 参数
- ✅ perf_stats → self.perf_stats
- ✅ 除零错误保护
- ✅ 模型性能指标记录

### utils.py
- ✅ 动态 head_dim
- ✅ LLaMA 层名识别
- ✅ BERT/LLaMA 兼容

### args_llm.py
- ✅ LLM 模型支持
- ✅ load_in_4bit 参数
- ✅ lm_eval_tasks 参数
