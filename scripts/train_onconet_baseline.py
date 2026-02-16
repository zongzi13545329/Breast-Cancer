"""
Training Script for OncoNet Baseline on EMBED Dataset
=====================================================

This script trains OncoNet (ImageOnly or Hybrid) as a baseline,
reusing your existing:
  - EMBEDRecallDataset + create_data_loaders (data/dataset.py)
  - BIRADSAwareLoss (train.py)
  - MCDropoutWrapper + evaluate_with_uncertainty (train.py)

Usage:
  # ImageOnly baseline (no race info)
  python train_onconet_baseline.py \
      --model_variant imageonly \
      --use_prior \
      --batch_size 4 \
      --num_epochs 50

  # Hybrid baseline (with race embedding, closest to your model)
  python train_onconet_baseline.py \
      --model_variant hybrid \
      --use_prior \
      --batch_size 4 \
      --num_epochs 50

  # Quick debug run
  python train_onconet_baseline.py \
      --model_variant hybrid \
      --sample_fraction 0.01 \
      --num_epochs 3

Author: Yiran
Date: 2025
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# Import your existing modules
# ============================================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from data.dataset import create_data_loaders
from onconet_baseline import OncoNetImageOnly, OncoNetHybrid

# Import loss and eval from your train.py
from train import (
    BIRADSAwareLoss,
    MCDropoutWrapper,
    evaluate_with_uncertainty,
    train_epoch,
    validate_epoch,
)


def get_args():
    parser = argparse.ArgumentParser(description="OncoNet Baseline Training")
    
    # =====================================================================
    # Model selection
    # =====================================================================
    parser.add_argument('--model_variant', type=str, default='hybrid',
                        choices=['imageonly', 'hybrid'],
                        help='OncoNet variant: imageonly (no risk factors) or hybrid (with race)')
    parser.add_argument('--pretrained_imagenet', action='store_true', default=True,
                        help='Initialize encoder with ImageNet weights')
    parser.add_argument('--freeze_mode', type=str, default='none',
                        choices=['full', 'partial', 'none'],
                        help='Encoder freezing strategy')
    
    # =====================================================================
    # Data (same as your train.py)
    # =====================================================================
    parser.add_argument('--clinical_csv', type=str,
                        default="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled.csv")
    parser.add_argument('--metadata_csv', type=str,
                        default="/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv")
    parser.add_argument('--image_size', type=int, nargs=2, default=[2944, 1920])
    parser.add_argument('--apply_nyu_preprocessing', action='store_true', default=True)
    parser.add_argument('--train_split', type=float, default=0.6)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--sample_fraction', type=float, default=1.0)
    parser.add_argument('--use_cache', action='store_true', default=True)
    
    # =====================================================================
    # Training
    # =====================================================================
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='OncoNet default: 1e-4 with Adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='OncoNet default: 5e-5')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--use_prior', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    
    # =====================================================================
    # Loss (same BIRADSAwareLoss as your model for fair comparison)
    # =====================================================================
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--fn_cost_ratio', type=float, default=5.0)
    parser.add_argument('--uncertain_birads_penalty', type=float, default=0.1)
    parser.add_argument('--fairness_lambda', type=float, default=0.05)
    
    # =====================================================================
    # Evaluation
    # =====================================================================
    parser.add_argument('--mc_samples', type=int, default=10)
    parser.add_argument('--target_sensitivity', type=float, default=0.95)
    
    # =====================================================================
    # Sampling
    # =====================================================================
    parser.add_argument('--use_balanced_batch', action='store_true', default=True)
    parser.add_argument('--positive_ratio', type=float, default=0.3)
    
    # =====================================================================
    # Optimization
    # =====================================================================
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # =====================================================================
    # Checkpointing
    # =====================================================================
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--exp_name', type=str, default='onconet_baseline')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    
    # =====================================================================
    # Resume
    # =====================================================================
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_optimizer', action='store_true', default=True)
    parser.add_argument('--resume_scheduler', action='store_true', default=True)
    
    # Device
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    return args


def create_baseline_model(args, device):
    """Create OncoNet baseline model based on variant selection."""
    
    print(f"\n{'='*70}")
    print(f"CREATING ONCONET BASELINE MODEL")
    print(f"  Variant: {args.model_variant.upper()}")
    print(f"{'='*70}")
    
    if args.model_variant == 'imageonly':
        model = OncoNetImageOnly(
            input_channels=1,
            pretrained_on_imagenet=args.pretrained_imagenet,
            use_prior=args.use_prior,
            dropout=args.dropout,
            freeze_mode=args.freeze_mode,
        )
    elif args.model_variant == 'hybrid':
        model = OncoNetHybrid(
            input_channels=1,
            pretrained_on_imagenet=args.pretrained_imagenet,
            use_prior=args.use_prior,
            dropout=args.dropout,
            freeze_mode=args.freeze_mode,
            num_races=4,
        )
    else:
        raise ValueError(f"Unknown model variant: {args.model_variant}")
    
    model = model.to(device)
    model.print_trainable_status()
    
    return model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = get_args()
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # =========================================================================
    # Experiment directory
    # =========================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.exp_name}_{args.model_variant}_{timestamp}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    
    # =========================================================================
    # Load data (EXACT same splits as your model)
    # =========================================================================
    print(f"\n{'='*70}")
    print("LOADING DATA (same pipeline as your model)")
    print(f"{'='*70}")
    
    train_loader, val_loader, test_loader, datasets = create_data_loaders(args)
    
    # =========================================================================
    # Create model
    # =========================================================================
    model = create_baseline_model(args, device)
    
    # =========================================================================
    # Loss (EXACT same as your model for fair comparison)
    # =========================================================================
    criterion = BIRADSAwareLoss(
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        fn_cost_ratio=args.fn_cost_ratio,
        uncertain_birads_penalty=args.uncertain_birads_penalty,
        fairness_lambda=args.fairness_lambda,
    )
    
    # =========================================================================
    # Optimizer (OncoNet uses Adam with lr=1e-4, wd=5e-5)
    # =========================================================================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    if args.scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
        
        def warmup_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    else:
        warmup_scheduler = None
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
    scaler = GradScaler() if args.use_amp else None
    
    # =========================================================================
    # Resume if needed
    # =========================================================================
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume and os.path.exists(args.resume):
        from train import load_checkpoint
        resume_info = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer if args.resume_optimizer else None,
            scheduler=main_scheduler if args.resume_scheduler else None,
            resume_optimizer=args.resume_optimizer,
            resume_scheduler=args.resume_scheduler,
            device=device,
        )
        start_epoch = resume_info['start_epoch']
        best_metric = resume_info['best_metric']
    
    # =========================================================================
    # Training loop (reuses your train_epoch and validate_epoch)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"TRAINING ONCONET {args.model_variant.upper()} BASELINE")
    print(f"{'='*70}")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{args.num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            args.use_amp, scaler, args.accumulation_steps,
        )
        
        # Validate
        val_losses, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer,
        )
        
        # Summary
        print(f"\n  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val Loss:   {val_losses['total']:.4f}")
        print(f"  Exam Primary (Spec@95%Sens): {val_metrics.get('exam_primary', 0):.4f}")
        
        # TensorBoard
        writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
        writer.add_scalar('Loss/val_total', val_losses['total'], epoch)
        writer.add_scalar('Metrics/exam_primary', val_metrics.get('exam_primary', 0), epoch)
        writer.add_scalar('Metrics/exam_auroc', val_metrics.get('exam_auroc', 0), epoch)
        
        # Scheduler
        if warmup_scheduler and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        elif args.scheduler == 'plateau':
            main_scheduler.step(val_losses['total'])
        else:
            main_scheduler.step()
        
        # Save best
        current_metric = val_metrics.get('exam_primary', 0)
        
        if current_metric > best_metric:
            print(f"\n  ✅ NEW BEST: {current_metric:.4f}")
            best_metric = current_metric
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': current_metric,
                'args': vars(args),
                'model_variant': args.model_variant,
            }
            if main_scheduler is not None:
                checkpoint['scheduler_state_dict'] = main_scheduler.state_dict()
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, os.path.join(exp_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{args.patience}")
        
        if patience_counter >= args.patience:
            print(f"\n  EARLY STOPPING at epoch {epoch+1}")
            break
        
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }
            if main_scheduler is not None:
                checkpoint['scheduler_state_dict'] = main_scheduler.state_dict()
            torch.save(checkpoint, os.path.join(exp_dir, f'checkpoint_epoch{epoch+1}.pth'))
    
    writer.close()
    
    # =========================================================================
    # Post-training: MC Dropout Evaluation (EXACT same as your model)
    # =========================================================================
    print(f"\n{'='*70}")
    print("POST-TRAINING: UNCERTAINTY-AWARE EVALUATION")
    print(f"{'='*70}")
    
    # Load best model
    best_ckpt_path = os.path.join(exp_dir, 'best_model.pth')
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print(f"  ✓ Loaded best model (epoch {best_ckpt['epoch']+1})")
    
    # Wrap with MC Dropout
    mc_model = MCDropoutWrapper(model, n_samples=args.mc_samples)
    mc_model.to(device)
    
    # Evaluate on test set
    test_results = evaluate_with_uncertainty(
        model=mc_model,
        data_loader=test_loader,
        device=device,
        target_sensitivity=args.target_sensitivity,
        n_mc_samples=args.mc_samples,
        output_dir=os.path.join(exp_dir, 'uncertainty_eval'),
    )
    
    # Also evaluate on val set
    val_results = evaluate_with_uncertainty(
        model=mc_model,
        data_loader=val_loader,
        device=device,
        target_sensitivity=args.target_sensitivity,
        n_mc_samples=args.mc_samples,
        output_dir=os.path.join(exp_dir, 'uncertainty_eval_val'),
    )
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"ONCONET {args.model_variant.upper()} BASELINE - COMPLETE")
    print(f"{'='*80}")
    print(f"  Best Val Metric (Spec@95%Sens): {best_metric:.4f}")
    print(f"  Test Exam AUROC:                {test_results.get('exam_auroc', 'N/A')}")
    print(f"  Test SRR:                       {test_results.get('SRR', 'N/A')}")
    print(f"  Test CMR:                       {test_results.get('CMR', 'N/A')}")
    print(f"  Sensitivity Gap:                {test_results.get('sensitivity_gap', 'N/A')}")
    print(f"  AUROC Gap:                      {test_results.get('auroc_gap', 'N/A')}")
    print(f"  Experiment: {exp_dir}")
    print(f"{'='*80}\n")