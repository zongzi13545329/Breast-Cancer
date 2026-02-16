#!/bin/bash

#SBATCH --job-name=embed_train          # 任务名称
#SBATCH --partition=msigpu              # 使用 msigpu 分区
#SBATCH --gres=gpu:1                    # 申请 1 个 GPU
#SBATCH --time=24:00:00                 # 申请时长 40 小时
#SBATCH --nodes=1                       # 使用 1 个节点
#SBATCH --ntasks=1                      # 运行 1 个任务
#SBATCH --cpus-per-task=16               # 根据需要调整 CPU 核心数 (通常 1 个 GPU 配 4-8 个 CPU)
#SBATCH --mem=32G                       # 根据需要调整内存大小
#SBATCH --output=log_%j.out             # 标准输出日志 (%j 会自动替换为作业 ID)
#SBATCH --error=log_%j.err              # 错误日志

# 1. 切换到工作目录
cd /projects/standard/lin01231/song0760/embed_recall_reduction/scripts

# 3. 激活环境
conda activate embed

# 4. 执行训练脚本
python train_pretrained.py