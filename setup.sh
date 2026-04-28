#!/bin/bash
# REP-LIE LLM - 环境安装脚本
# 使用方法: bash setup.sh

set -e

echo "=========================================="
echo "REP-LIE LLM 环境安装"
echo "=========================================="

# 检查 Python 版本
PYTHON_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python 版本: $PYTHON_VERSION"

# 创建 conda 环境（可选）
read -p "是否创建新的 conda 环境 rep-lie? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    conda create -n rep-lie python=3.10 -y
    conda activate rep-lie
    echo "已激活 conda 环境 rep-lie"
fi

# 升级 pip
pip install --upgrade pip

# 安装 PyTorch (根据 CUDA 版本选择)
echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装核心依赖
echo "安装核心依赖..."
pip install \
    transformers>=4.36.0 \
    peft>=0.7.0 \
    datasets>=2.14.0 \
    huggingface_hub>=0.19.0 \
    accelerate>=0.25.0

# 安装 NNI (剪枝框架)
echo "安装 NNI..."
pip install nni>=3.0

# 安装 FLOPs 统计
echo "安装 fvcore..."
pip install fvcore

# 安装 bitsandbytes (4bit 量化)
echo "安装 bitsandbytes..."
pip install bitsandbytes

# 安装 lm-evaluation-harness (零样本评测)
echo "安装 lm-evaluation-harness..."
pip install lm-eval

# 安装其他依赖
echo "安装其他依赖..."
pip install \
    numpy scipy scikit-learn \
    tqdm pandas pyyaml tensorboard

# 验证安装
echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import nni; print(f'NNI: {nni.__version__}')"
python -c "import fvcore; print('fvcore: OK')"
python -c "import lm_eval; print('lm-eval: OK')"

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 下载模型 (参考 MIGRATION_GUIDE.md)"
echo "2. 运行示例: python main_llm.py --help"
