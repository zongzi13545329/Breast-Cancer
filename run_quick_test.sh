#!/bin/bash
# Quick Test Training - Minimal configuration for fast validation
# Expected runtime: ~5-10 minutes

# 设置数据路径（请修改为你的实际路径）
CLINICAL_CSV="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled.csv"
METADATA_CSV="/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv"
IMAGE_ROOT="/projects/standard/lin01231/public/datasets/embed/"

# 创建输出目录
OUTPUT_DIR="./experiments/quick_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/checkpoints

echo "=================================="
echo "Quick Test Training"
echo "=================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# 运行最小化训练
python train.py \
    --clinical_csv $CLINICAL_CSV \
    --metadata_csv $METADATA_CSV \
    --image_root $IMAGE_ROOT \
    --backbone efficientnet_b0 \
    --pretrained \
    --no-temporal-attention \
    --no-fairness-adapter \
    --batch_size 32 \
    --num_epochs 1 \
    --image_size 512 256 \
    --num_workers 2 \
    --lr 1e-4 \
    --recall_weight 1.0 \
    --birads_weight 0.5 \
    --density_weight 0.3 \
    --train_split 0.7 \
    --val_split 0.15 \
    --random_seed 42 \
    2>&1 | tee $OUTPUT_DIR/logs/train.log

echo ""
echo "Training completed!"
echo "Check logs at: $OUTPUT_DIR/logs/train.log"