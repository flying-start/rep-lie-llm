#!/bin/bash
# ============================================================
# REP-LIE 环境一键恢复脚本
# 使用方法：bash setup_env.sh
# 适用平台：AutoDL vGPU-32GB/CUDA 12.x / Python 3.10
# ============================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo " REP-LIE 环境一键安装脚本"
echo "=========================================="

# ----------------------------------------------------------
# Step 0: 设置数据盘环境变量
# ----------------------------------------------------------
echo ""
echo "[Step 0] 设置数据盘环境变量..."

cat >> ~/.bashrc << 'EOF'
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
export TORCH_HOME=/root/autodl-tmp/torch_cache
export HF_ENDPOINT=https://hf-mirror.com
EOF

source ~/.bashrc
echo "✅ 环境变量已写入 ~/.bashrc"

# ----------------------------------------------------------
# Step 1: 创建 conda 虚拟环境
# ----------------------------------------------------------
echo ""
echo "[Step 1] 创建 conda 虚拟环境 rep-lie..."

conda create -n rep-lie python=3.10 -y
echo "✅ 虚拟环境 rep-lie 创建完成"

# ----------------------------------------------------------
# Step 2: 激活环境
# ----------------------------------------------------------
echo ""
echo "[Step 2] 激活 rep-lie 环境..."
source activate rep-lie || conda activate rep-lie
echo "✅ 虚拟环境已激活"

# ----------------------------------------------------------
# Step 3: 安装 PyTorch（必须先装，CUDA 12.1 专用 wheel）
# ----------------------------------------------------------
echo ""
echo "[Step 3] 安装 PyTorch 2.2.2 (CUDA 12.1)..."

pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

echo "✅ PyTorch 安装完成"

# ----------------------------------------------------------
# Step 4: 安装项目依赖
# ----------------------------------------------------------
echo ""
echo "[Step 4] 安装项目依赖（requirements_autodl.txt）..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements_autodl.txt" --ignore-requires-python

echo "✅ 依赖安装完成"

# ----------------------------------------------------------
# Step 5: 验证安装
# ----------------------------------------------------------
echo ""
echo "[Step 5] 验证关键包..."

python -c "
import torch
import transformers
import peft
import bitsandbytes
import datasets

print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU 型号: {torch.cuda.get_device_name(0)}')
    print(f'  显存: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT: {peft.__version__}')
print(f'  BitsAndBytes: {bitsandbytes.__version__}')
print(f'  Datasets: {datasets.__version__}')
print()
print('✅ 全部验证通过！')
"

# ----------------------------------------------------------
# 完成
# ----------------------------------------------------------
echo ""
echo "=========================================="
echo " 环境安装完成！"
echo ""
echo " 下次登录使用方法："
echo "   conda activate rep-lie"
echo "   cd /root/autodl-tmp/rep-lie-llm"
echo ""
echo " 训练命令："
echo "   python main_llm.py \\"
echo "     --model_name_or_path /root/autodl-tmp/models/AI-ModelScope/Mistral-7B-v0.1 \\"
echo "     --task_name c4 \\"
echo "     --max_seq_length 2048 \\"
echo "     --per_device_train_batch_size 1 \\"
echo "     --gradient_accumulation_steps 16 \\"
echo "     --output_dir /root/autodl-tmp/output/mistral-7b-s50 \\"
echo "     --ratio 0.5 --mac 0.5 \\"
echo "     --lora_r 8 --lora_alpha 16 \\"
echo "     --prune_metric lora --use_stability True \\"
echo "     --fp16 --do_train --do_eval"
echo "=========================================="
