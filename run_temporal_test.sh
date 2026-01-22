#!/bin/bash
# Medium Training - With temporal analysis (Layer 1 innovation)
# Expected runtime: ~1-2 hours on GPU

# 设置数据路径（请修改为你的实际路径）
CLINICAL_CSV="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled.csv"
METADATA_CSV="/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv"
IMAGE_ROOT="/projects/standard/lin01231/public/datasets/embed/"

# 创建输出目录
OUTPUT_DIR="./experiments/temporal_model_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/checkpoints

echo "=================================="
echo "Medium Training - Temporal Model"
echo "=================================="
echo "Output directory: $OUTPUT_DIR"
echo "Features enabled:"
echo "  ✓ Layer 1: Temporal Siamese Network"
echo "  ✗ Layer 2: Fairness Adapters (disabled)"
echo "  ✓ Layer 3: Multi-task Learning"
echo ""

# 运行时序模型训练
python train.py \
    --clinical_csv $CLINICAL_CSV \
    --metadata_csv $METADATA_CSV \
    --image_root $IMAGE_ROOT \
    --backbone efficientnet_b0 \
    --pretrained \
    --use_temporal_attention \
    --no-fairness-adapter \
    --batch_size 6 \
    --num_epochs 20 \
    --image_size 512 256 \
    --num_workers 4 \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --dropout 0.3 \
    --recall_weight 1.0 \
    --birads_weight 0.5 \
    --density_weight 0.3 \
    --use_augmentation \
    --train_split 0.7 \
    --val_split 0.15 \
    --random_seed 42 \
    2>&1 | tee $OUTPUT_DIR/logs/train.log

echo ""
echo "Training completed!"
echo "Check logs at: $OUTPUT_DIR/logs/train.log"
echo "View tensorboard: tensorboard --logdir=$OUTPUT_DIR/logs"