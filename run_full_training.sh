#!/bin/bash
# Full Training - All three layers of innovation
# Expected runtime: Several hours to days on GPU

# 设置数据路径（请修改为你的实际路径）
CLINICAL_CSV="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled.csv"
METADATA_CSV="/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv"
IMAGE_ROOT="/projects/standard/lin01231/public/datasets/embed/"

# 创建输出目录
OUTPUT_DIR="./experiments/full_model_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/checkpoints

echo "=========================================="
echo "Full Training - Complete Fairness Model"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Features enabled:"
echo "  ✓ Layer 1: Temporal Siamese Network"
echo "  ✓ Layer 2: Race-Conditional Fairness Adapters"
echo "  ✓ Layer 3: Multi-task Learning"
echo ""
echo "This is the complete model with ALL innovations!"
echo ""

# 运行完整模型训练
python train.py \
    --clinical_csv $CLINICAL_CSV \
    --metadata_csv $METADATA_CSV \
    --image_root $IMAGE_ROOT \
    --backbone efficientnet_b0 \
    --pretrained \
    --use_temporal_attention \
    --use_fairness_adapter \
    --batch_size 32 \
    --num_epochs 40 \
    --image_size 512 256 \
    --num_workers 16 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --dropout 0.3 \
    --recall_weight 1.0 \
    --birads_weight 0.5 \
    --density_weight 0.3 \
    --fairness_lambda 0.1 \
    --fairness_type demographic_parity \
    --use_augmentation \
    --train_split 0.7 \
    --val_split 0.15 \
    --random_seed 42 \
    --resume /projects/standard/lin01231/song0760/embed_recall_reduction/scripts/outputs/longitudinal_fair_multitask_20260120_163454/checkpoint_epoch_15.pth \
    2>&1 | tee $OUTPUT_DIR/logs/train.log

echo ""
echo "Training completed!"
echo "=================================="
echo "Results:"
echo "  Logs: $OUTPUT_DIR/logs/train.log"
echo "  Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  TensorBoard: tensorboard --logdir=$OUTPUT_DIR/logs"
echo "=================================="