#!/bin/bash
# Complete Environment Setup Script for EMBED Recall Reduction
# Recommended Python version: 3.9

echo "========================================"
echo "EMBED Recall Reduction - Environment Setup"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo ""
echo "[1/5] Creating conda environment with Python 3.9..."
conda create -n embed python=3.9 -y

echo ""
echo "[2/5] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate embed

# Verify Python version
echo ""
echo "[3/5] Verifying Python version..."
python --version

echo ""
echo "[4/5] Installing PyTorch (CUDA 11.8)..."
# For CUDA 11.8 (adjust based on your GPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Alternative: CPU-only version
# conda install pytorch torchvision cpuonly -c pytorch -y

# Alternative: Different CUDA versions
# CUDA 12.1: conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
# CUDA 11.7: conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -y

echo ""
echo "[5/5] Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate embed"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""
echo "Next steps:"
echo "  1. Prepare your EMBED data"
echo "  2. Run: python scripts/train.py --config configs/fairness_model.yaml --output_dir ./experiments/fairness"
echo ""
