"""
Training Script: NYU Baseline on EMBED Recall Prediction
=========================================================
A clean baseline that reproduces NYU's four-view breast cancer classifier
on the EMBED dataset for recall prediction.

Differences from your custom model:
- NO temporal fusion (prior exams ignored)
- NO race conditioning
- NO attention aggregation (simple mean pooling per breast)
- Standard weighted BCE / Focal loss
- Faithful to NYU ResNet-22 + global avg pool + FC

This serves as a baseline to compare against your more advanced model.

Author: Yiran
Date: 2025
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import warnings
import shutil
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, roc_curve

# ============================================================================
# Path setup - adjust these to match your project structure
# ============================================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import from your existing codebase
from data.dataset import create_data_loaders

# Import baseline model
from models.nyu_baseline_model import NYUFourViewModel, NYUFourViewSeparateHeads


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance (same as your implementation)."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        probs = torch.sigmoid(predictions)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        return (focal_weight * bce_loss).mean()


class BaselineLoss(nn.Module):
    """
    Simple loss for NYU baseline.
    
    Options:
    - Focal Loss (default, handles imbalance)
    - Weighted BCE
    """
    
    def __init__(self, use_focal=True, focal_alpha=0.25, focal_gamma=2.0,
                 pos_weight=10.0):
        super().__init__()
        self.use_focal = use_focal
        
        if use_focal:
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.pos_weight_val = pos_weight
    
    def forward(self, predictions, labels, metadata=None):
        """
        Args:
            predictions: dict with 'left_recall', 'right_recall'
            labels: dict with 'left_malignant', 'right_malignant'
            metadata: ignored for baseline
        """
        left_pred = predictions['left_recall'].squeeze(-1)
        right_pred = predictions['right_recall'].squeeze(-1)
        left_label = labels['left_malignant'].squeeze(-1)
        right_label = labels['right_malignant'].squeeze(-1)
        
        if self.use_focal:
            left_loss = self.loss_fn(left_pred, left_label)
            right_loss = self.loss_fn(right_pred, right_label)
        else:
            pos_weight = torch.tensor([self.pos_weight_val], device=left_pred.device)
            left_loss = F.binary_cross_entropy_with_logits(
                left_pred, left_label,
                pos_weight=pos_weight
            )
            right_loss = F.binary_cross_entropy_with_logits(
                right_pred, right_label,
                pos_weight=pos_weight
            )
        
        total = (left_loss + right_loss) / 2.0
        
        return {
            'total': total,
            'recall': total,
            'left': left_loss,
            'right': right_loss,
            'fairness': torch.tensor(0.0, device=total.device),
        }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch,
                writer, use_amp, scaler, accumulation_steps):
    """Training epoch."""
    model.train()
    losses = defaultdict(float)
    
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
    nan_count = 0
    valid_batches = 0
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        current_views = {k: v.to(device) for k, v in batch['current_views'].items()}
        prior_views = {k: v.to(device) for k, v in batch['prior_views'].items()}
        current_mask = {k: v.to(device) for k, v in batch['current_mask'].items()}
        prior_mask = {k: v.to(device) for k, v in batch['prior_mask'].items()}
        race = batch['metadata']['race'].to(device)
        
        left_label = batch['labels']['left_malignant'].to(device)
        right_label = batch['labels']['right_malignant'].to(device)
        
        try:
            if use_amp:
                with autocast():
                    predictions = model(
                        current_views=current_views,
                        prior_views=prior_views,
                        current_mask=current_mask,
                        prior_mask=prior_mask,
                        race=race
                    )
                    loss_dict = criterion(
                        predictions=predictions,
                        labels={'left_malignant': left_label, 'right_malignant': right_label}
                    )
                    loss = loss_dict['total'] / accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                predictions = model(
                    current_views=current_views,
                    prior_views=prior_views,
                    current_mask=current_mask,
                    prior_mask=prior_mask,
                    race=race
                )
                
                if torch.isnan(predictions['left_recall']).any() or \
                   torch.isnan(predictions['right_recall']).any():
                    nan_count += 1
                    optimizer.zero_grad()
                    continue
                
                loss_dict = criterion(
                    predictions=predictions,
                    labels={'left_malignant': left_label, 'right_malignant': right_label}
                )
                
                if torch.isnan(loss_dict['total']):
                    nan_count += 1
                    optimizer.zero_grad()
                    continue
                
                loss = loss_dict['total'] / accumulation_steps
                loss.backward()
            
            for k, v in loss_dict.items():
                losses[k] += v.item()
            valid_batches += 1
            
            # Gradient update
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                        scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                        optimizer.step()
                
                optimizer.zero_grad()
            
            pbar.set_postfix({
                'loss': loss.item() * accumulation_steps,
                'nan': nan_count
            })
        
        except Exception as e:
            print(f"\n❌ Batch {batch_idx} error: {e}")
            nan_count += 1
            optimizer.zero_grad()
            continue
    
    if valid_batches > 0:
        for k in losses:
            losses[k] /= valid_batches
    
    if nan_count > 0:
        print(f"  ⚠️  {nan_count} NaN/error batches")
    
    return dict(losses)


def validate_epoch(model, val_loader, criterion, device, epoch, writer):
    """Validation with recall-reduction metrics."""
    model.eval()
    losses = defaultdict(float)
    
    all_left_preds = []
    all_left_labels = []
    all_right_preds = []
    all_right_labels = []
    all_race = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
            current_views = {k: v.to(device) for k, v in batch['current_views'].items()}
            prior_views = {k: v.to(device) for k, v in batch['prior_views'].items()}
            current_mask = {k: v.to(device) for k, v in batch['current_mask'].items()}
            prior_mask = {k: v.to(device) for k, v in batch['prior_mask'].items()}
            race = batch['metadata']['race'].to(device)
            
            left_label = batch['labels']['left_malignant'].to(device)
            right_label = batch['labels']['right_malignant'].to(device)
            
            predictions = model(
                current_views=current_views,
                prior_views=prior_views,
                current_mask=current_mask,
                prior_mask=prior_mask,
                race=race
            )
            
            loss_dict = criterion(
                predictions=predictions,
                labels={'left_malignant': left_label, 'right_malignant': right_label}
            )
            
            for k, v in loss_dict.items():
                losses[k] += v.item()
            
            all_left_preds.append(torch.sigmoid(predictions['left_recall']).cpu())
            all_left_labels.append(left_label.cpu())
            all_right_preds.append(torch.sigmoid(predictions['right_recall']).cpu())
            all_right_labels.append(right_label.cpu())
            all_race.append(race.cpu())
    
    for k in losses:
        losses[k] /= len(val_loader)
    
    # Compute metrics
    left_preds = torch.cat(all_left_preds).squeeze().numpy()
    left_labels = torch.cat(all_left_labels).squeeze().numpy()
    right_preds = torch.cat(all_right_preds).squeeze().numpy()
    right_labels = torch.cat(all_right_labels).squeeze().numpy()
    race_arr = torch.cat(all_race).numpy()
    
    metrics = {}
    
    print(f"\n{'='*80}")
    print(f"[BASELINE] VALIDATION METRICS - EPOCH {epoch+1}")
    print(f"{'='*80}")
    
    # Per-breast metrics
    for side, preds, labels in [
        ('Left', left_preds, left_labels),
        ('Right', right_preds, right_labels)
    ]:
        if len(np.unique(labels)) > 1:
            auroc = roc_auc_score(labels, preds)
            metrics[f'{side.lower()}_auroc'] = auroc
            print(f"\n  {side} Breast AUROC: {auroc:.4f}")
            
            fpr, tpr, thresholds = roc_curve(labels, preds)
            
            for target_sens in [0.90, 0.95, 0.98]:
                idx = np.where(tpr >= target_sens)[0]
                if len(idx) > 0:
                    best_idx = idx[np.argmin(fpr[idx])]
                    spec = 1 - fpr[best_idx]
                    metrics[f'{side.lower()}_spec_at_{int(target_sens*100)}_sens'] = spec
                    
                    print(f"    Spec@{target_sens*100:.0f}%Sens: {spec:.4f} "
                          f"(recall reduction: {spec*100:.1f}%)")
                    
                    if target_sens == 0.95:
                        metrics[f'{side.lower()}_primary'] = spec
        else:
            print(f"\n  {side} Breast: Only one class present, skipping AUROC")
    
    # Exam-level metrics
    exam_preds = np.maximum(left_preds, right_preds)
    exam_labels = np.maximum(left_labels, right_labels)
    
    if len(np.unique(exam_labels)) > 1:
        exam_auroc = roc_auc_score(exam_labels, exam_preds)
        metrics['exam_auroc'] = exam_auroc
        print(f"\n  Exam-level AUROC: {exam_auroc:.4f}")
        
        fpr, tpr, thresholds = roc_curve(exam_labels, exam_preds)
        for target_sens in [0.95]:
            idx = np.where(tpr >= target_sens)[0]
            if len(idx) > 0:
                best_idx = idx[np.argmin(fpr[idx])]
                spec = 1 - fpr[best_idx]
                metrics['exam_primary'] = spec
                print(f"    Exam Spec@95%Sens: {spec:.4f}")
    
    # Fairness metrics (for reference, no optimization)
    print(f"\n  {'─'*40}")
    print(f"  Fairness Analysis (no optimization):")
    race_names = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Other'}
    
    for r_idx, r_name in race_names.items():
        mask = race_arr == r_idx
        if mask.sum() > 10:
            r_preds = exam_preds[mask]
            r_labels = exam_labels[mask]
            if len(np.unique(r_labels)) > 1:
                r_auroc = roc_auc_score(r_labels, r_preds)
                metrics[f'race_{r_name.lower()}_auroc'] = r_auroc
                print(f"    {r_name}: AUROC={r_auroc:.4f} (n={mask.sum()})")
    
    print(f"{'='*80}\n")
    
    return dict(losses), metrics


# ============================================================================
# Test on Test Set
# ============================================================================

def evaluate_test_set(model, test_loader, criterion, device, output_dir):
    """
    Final evaluation on test set — paper-ready metrics for baseline comparison.
    
    Covers:
      - Table 1: AUROC, Spec@95%, Spec@98%, CMR, Sens Gap (SRR = "---" for baseline)
      - Table 3: Per-race AUROC, Sensitivity, Specificity
      - BI-RADS stratified FN analysis (if available)
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    
    print(f"\n{'='*80}")
    print("[BASELINE] FINAL TEST SET EVALUATION")
    print(f"{'='*80}")
    
    model.eval()
    
    all_left_preds = []
    all_left_labels = []
    all_right_preds = []
    all_right_labels = []
    all_race = []
    all_birads = []
    all_exam_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            current_views = {k: v.to(device) for k, v in batch['current_views'].items()}
            prior_views = {k: v.to(device) for k, v in batch['prior_views'].items()}
            current_mask = {k: v.to(device) for k, v in batch['current_mask'].items()}
            prior_mask = {k: v.to(device) for k, v in batch['prior_mask'].items()}
            race = batch['metadata']['race'].to(device)
            
            left_label = batch['labels']['left_malignant'].to(device)
            right_label = batch['labels']['right_malignant'].to(device)
            
            predictions = model(
                current_views=current_views,
                prior_views=prior_views,
                current_mask=current_mask,
                prior_mask=prior_mask,
                race=race
            )
            
            all_left_preds.append(torch.sigmoid(predictions['left_recall']).cpu())
            all_left_labels.append(left_label.cpu())
            all_right_preds.append(torch.sigmoid(predictions['right_recall']).cpu())
            all_right_labels.append(right_label.cpu())
            all_race.append(race.cpu())
            
            # Collect BI-RADS if available
            if 'birads' in batch['labels']:
                all_birads.append(batch['labels']['birads'].cpu())
            
            if 'exam_info' in batch:
                all_exam_ids.extend([e['exam_id'] for e in batch['exam_info']])
    
    left_preds = torch.cat(all_left_preds).squeeze().numpy()
    left_labels = torch.cat(all_left_labels).squeeze().numpy()
    right_preds = torch.cat(all_right_preds).squeeze().numpy()
    right_labels = torch.cat(all_right_labels).squeeze().numpy()
    race_arr = torch.cat(all_race).numpy()
    
    has_birads = len(all_birads) > 0
    if has_birads:
        birads_arr = torch.cat(all_birads).squeeze().numpy()
    
    exam_preds = np.maximum(left_preds, right_preds)
    exam_labels = np.maximum(left_labels, right_labels)
    N = len(exam_labels)
    n_pos = (exam_labels == 1).sum()
    n_neg = (exam_labels == 0).sum()
    
    test_metrics = {}
    
    # =====================================================================
    # 1. DISCRIMINATION METRICS (Side-level + Exam-level)
    # =====================================================================
    print(f"\n{'─'*40}")
    print("1. DISCRIMINATION METRICS")
    print(f"{'─'*40}")
    
    for side, preds, labels in [
        ('Left', left_preds, left_labels),
        ('Right', right_preds, right_labels)
    ]:
        if len(np.unique(labels)) > 1:
            auroc = roc_auc_score(labels, preds)
            test_metrics[f'{side.lower()}_auroc'] = auroc
            print(f"  {side} Breast AUROC: {auroc:.4f}")
    
    if len(np.unique(exam_labels)) > 1:
        exam_auroc = roc_auc_score(exam_labels, exam_preds)
        test_metrics['exam_auroc'] = exam_auroc
        print(f"  Exam AUROC: {exam_auroc:.4f}")
    
    # =====================================================================
    # 2. SPECIFICITY AT FIXED SENSITIVITY + CMR (Table 1)
    # =====================================================================
    print(f"\n{'─'*40}")
    print("2. SPECIFICITY @ FIXED SENSITIVITY & CMR (Table 1)")
    print(f"{'─'*40}")
    
    if len(np.unique(exam_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(exam_labels, exam_preds)
        
        for target_sens in [0.90, 0.95, 0.98, 0.99]:
            idx = np.where(tpr >= target_sens)[0]
            if len(idx) > 0:
                best_idx = idx[np.argmin(fpr[idx])]
                spec = 1 - fpr[best_idx]
                thresh = thresholds[best_idx]
                
                test_metrics[f'exam_spec_at_{int(target_sens*100)}'] = spec
                test_metrics[f'exam_thresh_at_{int(target_sens*100)}'] = float(thresh)
                
                # ---- CMR at this operating point ----
                decisions = (exam_preds >= thresh).astype(int)
                TP = ((decisions == 1) & (exam_labels == 1)).sum()
                FN = ((decisions == 0) & (exam_labels == 1)).sum()
                TN = ((decisions == 0) & (exam_labels == 0)).sum()
                
                cmr = float(FN / max(1, n_pos))
                sensitivity = float(TP / max(1, n_pos))
                specificity = float(TN / max(1, n_neg))
                
                test_metrics[f'exam_cmr_at_{int(target_sens*100)}'] = cmr
                test_metrics[f'exam_actual_sens_at_{int(target_sens*100)}'] = sensitivity
                
                print(f"\n  At {target_sens*100:.0f}% Target Sensitivity:")
                print(f"    Threshold:    {thresh:.4f}")
                print(f"    Specificity:  {spec:.4f}")
                print(f"    Actual Sens:  {sensitivity:.4f}")
                print(f"    CMR:          {cmr:.4f}  (FN={FN}, TP={TP})")
                
                # Store the 95% operating point decisions for fairness analysis
                if target_sens == 0.95:
                    decisions_at_95 = decisions
                    thresh_95 = thresh
    
    # =====================================================================
    # 3. PER-RACE FAIRNESS METRICS (Table 3)
    # =====================================================================
    print(f"\n{'─'*40}")
    print("3. FAIRNESS METRICS BY RACE (Table 3)")
    print(f"{'─'*40}")
    
    race_names = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Other'}
    race_aurocs = {}
    race_sensitivities = {}
    race_specificities = {}
    
    print(f"\n  {'Group':<10} {'N':>6} {'AUROC':>8} {'Sens':>8} {'Spec':>8}")
    print(f"  {'-'*45}")
    
    for r_idx, r_name in race_names.items():
        mask = race_arr == r_idx
        if mask.sum() < 10:
            continue
        
        r_exam_preds = exam_preds[mask]
        r_exam_labels = exam_labels[mask]
        r_n = mask.sum()
        
        if len(np.unique(r_exam_labels)) > 1:
            # AUROC
            r_auroc = roc_auc_score(r_exam_labels, r_exam_preds)
            race_aurocs[r_name] = r_auroc
            test_metrics[f'race_{r_name.lower()}_auroc'] = r_auroc
            
            # Sensitivity & Specificity at the global 95% threshold
            if 'decisions_at_95' in dir():
                r_decisions = decisions_at_95[mask]
                r_pos = (r_exam_labels == 1).sum()
                r_neg = (r_exam_labels == 0).sum()
                r_TP = ((r_decisions == 1) & (r_exam_labels == 1)).sum()
                r_TN = ((r_decisions == 0) & (r_exam_labels == 0)).sum()
                
                r_sens = float(r_TP / max(1, r_pos))
                r_spec = float(r_TN / max(1, r_neg))
                
                race_sensitivities[r_name] = r_sens
                race_specificities[r_name] = r_spec
                test_metrics[f'race_{r_name.lower()}_sensitivity'] = r_sens
                test_metrics[f'race_{r_name.lower()}_specificity'] = r_spec
                
                print(f"  {r_name:<10} {r_n:>6} {r_auroc:>8.4f} {r_sens:>7.1%} {r_spec:>7.1%}")
            else:
                print(f"  {r_name:<10} {r_n:>6} {r_auroc:>8.4f} {'N/A':>8} {'N/A':>8}")
        else:
            print(f"  {r_name:<10} {r_n:>6} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    
    # ---- Compute gaps ----
    print(f"\n  Cross-Group Gaps:")
    
    if len(race_aurocs) >= 2:
        auroc_gap = max(race_aurocs.values()) - min(race_aurocs.values())
        test_metrics['auroc_gap'] = auroc_gap
        print(f"    AUROC gap:       {auroc_gap:.4f}")
    
    if len(race_sensitivities) >= 2:
        sens_gap = max(race_sensitivities.values()) - min(race_sensitivities.values())
        test_metrics['sensitivity_gap'] = sens_gap
        print(f"    Sensitivity gap: {sens_gap:.4f}")
    
    if len(race_specificities) >= 2:
        spec_gap = max(race_specificities.values()) - min(race_specificities.values())
        test_metrics['specificity_gap'] = spec_gap
        print(f"    Specificity gap: {spec_gap:.4f}")
    
    # =====================================================================
    # 4. BI-RADS STRATIFIED FN ANALYSIS
    # =====================================================================
    if has_birads and 'decisions_at_95' in dir():
        print(f"\n{'─'*40}")
        print("4. BI-RADS STRATIFIED PERFORMANCE (at 95% Sens threshold)")
        print(f"{'─'*40}")
        
        print(f"\n  {'BI-RADS':<14} {'N':>7} {'Pos%':>7} {'FN':>5} {'FN Rate':>9}")
        print(f"  {'-'*45}")
        
        br_labels_map = {
            0: 'Incomplete', 1: 'Negative', 2: 'Benign',
            3: 'Prob Benign', 4: 'Suspicious'
        }
        
        for br in sorted(np.unique(birads_arr)):
            br_mask = birads_arr == int(br)
            n_br = br_mask.sum()
            if n_br == 0:
                continue
            
            br_labels = exam_labels[br_mask]
            br_decisions = decisions_at_95[br_mask]
            
            pos_rate = (br_labels == 1).mean()
            fn_count = ((br_decisions == 0) & (br_labels == 1)).sum()
            br_pos = (br_labels == 1).sum()
            fn_rate = float(fn_count / max(1, br_pos)) if br_pos > 0 else 0.0
            
            test_metrics[f'birads{int(br)}_n'] = int(n_br)
            test_metrics[f'birads{int(br)}_fn_count'] = int(fn_count)
            test_metrics[f'birads{int(br)}_fn_rate'] = fn_rate
            
            br_name = br_labels_map.get(int(br), f'BR{int(br)}')
            print(f"  {br_name:<14} {n_br:>7} {pos_rate:>6.1%} {fn_count:>5} {fn_rate:>8.1%}")
    
    # =====================================================================
    # 5. SAVE RESULTS
    # =====================================================================
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 5a. Per-sample predictions
        results_df = pd.DataFrame({
            'left_pred': left_preds,
            'left_label': left_labels,
            'right_pred': right_preds,
            'right_label': right_labels,
            'exam_pred': exam_preds,
            'exam_label': exam_labels,
            'race': race_arr,
        })
        if has_birads:
            results_df['birads'] = birads_arr
        if 'decisions_at_95' in dir():
            results_df['decision_at_95'] = decisions_at_95
        if len(all_exam_ids) == N:
            results_df['exam_id'] = all_exam_ids
        
        results_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
        
        # 5b. All metrics
        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4, default=str)
        
        # 5c. Paper-ready summary
        summary_lines = [
            "=" * 60,
            "BASELINE PAPER-READY SUMMARY",
            "=" * 60,
            f"AUROC:                {test_metrics.get('exam_auroc', 'N/A')}",
            f"Spec@95% Sens:        {test_metrics.get('exam_spec_at_95', 'N/A')}",
            f"Spec@98% Sens:        {test_metrics.get('exam_spec_at_98', 'N/A')}",
            f"SRR:                  --- (no uncertainty for baseline)",
            f"CMR (at 95% Sens):    {test_metrics.get('exam_cmr_at_95', 'N/A')}",
            f"Sensitivity Gap:      {test_metrics.get('sensitivity_gap', 'N/A')}",
            f"AUROC Gap:            {test_metrics.get('auroc_gap', 'N/A')}",
            f"Specificity Gap:      {test_metrics.get('specificity_gap', 'N/A')}",
            "=" * 60,
        ]
        with open(os.path.join(output_dir, 'paper_summary.txt'), 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"\n  Saved to: {output_dir}")
        print(f"    - test_predictions.csv")
        print(f"    - test_metrics.json")
        print(f"    - paper_summary.txt")
    
    print(f"\n{'='*80}\n")
    
    return test_metrics


# ============================================================================
# Main
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description='NYU Baseline on EMBED')
    
    # NYU pretrained weights
    parser.add_argument('--nyu_checkpoint', type=str,
                        default="/projects/standard/lin01231/song0760/embed_recall_reduction/pretrained/sample_image_model.p")
    parser.add_argument('--freeze_mode', type=str, default='partial',
                        choices=['full', 'partial', 'none'])
    
    # Model variant
    parser.add_argument('--separate_heads', action='store_true', default=False,
                        help='Use separate classification heads for left/right breast')
    
    # Data
    parser.add_argument('--clinical_csv', type=str,
                        default="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled.csv")
    parser.add_argument('--metadata_csv', type=str,
                        default="/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv")
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    
    # Loss
    parser.add_argument('--use_focal', action='store_true', default=True)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--pos_weight', type=float, default=10.0)
    
    # Data splits
    parser.add_argument('--image_size', type=int, nargs=2, default=[2944, 1920])
    parser.add_argument('--apply_nyu_preprocessing', action='store_true', default=True)
    parser.add_argument('--train_split', type=float, default=0.6)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--sample_fraction', type=float, default=1.0)
    
    # Sampling
    parser.add_argument('--use_balanced_batch', action='store_true', default=True)
    parser.add_argument('--positive_ratio', type=float, default=0.3)
    parser.add_argument('--use_cache', action='store_true', default=True)
    
    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau'])
    parser.add_argument('--warmup_epochs', type=int, default=3)
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--exp_name', type=str, default='nyu_baseline')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    
    # Resume
    parser.add_argument('--resume', type=str, default="/projects/standard/lin01231/song0760/embed_recall_reduction/scripts/outputs/nyu_baseline_partial_shared_head_20260212_163013/best_model.pth")
    
    # Device
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser.parse_args()


def main():
    args = get_args()
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    variant = 'sep_heads' if args.separate_heads else 'shared_head'
    exp_name = f"{args.exp_name}_{args.freeze_mode}_{variant}_{timestamp}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"NYU BASELINE EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Experiment: {exp_name}")
    print(f"  Output dir: {exp_dir}")
    print(f"  Freeze mode: {args.freeze_mode}")
    print(f"  Separate heads: {args.separate_heads}")
    print(f"  Learning rate: {args.lr}")
    print(f"{'='*70}")
    
    # Save config
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    
    # =========================================================================
    # Data Loading (reuse your existing pipeline)
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    train_loader, val_loader, test_loader, datasets = create_data_loaders(args)
    
    # =========================================================================
    # Model Creation
    # =========================================================================
    print("\n" + "="*70)
    print("CREATING NYU BASELINE MODEL")
    print("="*70)
    
    ModelClass = NYUFourViewSeparateHeads if args.separate_heads else NYUFourViewModel
    
    model = ModelClass(
        input_channels=1,
        nyu_checkpoint_path=args.nyu_checkpoint,
        freeze_mode=args.freeze_mode,
        num_classes=1,
    ).to(device)
    
    model.print_trainable_status()
    
    # =========================================================================
    # Loss, Optimizer, Scheduler
    # =========================================================================
    criterion = BaselineLoss(
        use_focal=args.use_focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        pos_weight=args.pos_weight,
    )
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n  Optimizing {len(trainable_params)} parameter groups")
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    if args.scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
        
        def warmup_fn(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    else:
        warmup_scheduler = None
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    
    scaler = GradScaler() if args.use_amp else None
    
    # Resume
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_metric = ckpt.get('metric', 0.0)
        print(f"\n  Resumed from epoch {start_epoch}, best={best_metric:.4f}")
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f"\n{'='*70}")
    print("STARTING BASELINE TRAINING")
    print(f"{'='*70}")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{args.num_epochs} (lr={optimizer.param_groups[0]['lr']:.2e})")
        print(f"{'='*80}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            writer, args.use_amp, scaler, args.accumulation_steps
        )
        
        # Validate
        val_losses, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Summary
        print(f"\n  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val Loss:   {val_losses['total']:.4f}")
        
        current_metric = val_metrics.get('exam_primary', 
                          val_metrics.get('exam_auroc', 0))
        print(f"  Primary Metric: {current_metric:.4f}")
        
        # Logging
        writer.add_scalar('Loss/train', train_losses['total'], epoch)
        writer.add_scalar('Loss/val', val_losses['total'], epoch)
        writer.add_scalar('Metrics/exam_primary', current_metric, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Metrics/{k}', v, epoch)
        
        # Scheduler step
        if warmup_scheduler and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        elif args.scheduler == 'plateau':
            main_scheduler.step(val_losses['total'])
        else:
            main_scheduler.step()
        
        # Save best
        if current_metric > best_metric:
            print(f"\n  ✅ NEW BEST: {current_metric:.4f} (prev: {best_metric:.4f})")
            best_metric = current_metric
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': current_metric,
                'metrics': val_metrics,
                'args': vars(args),
            }, os.path.join(exp_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  EARLY STOPPING at epoch {epoch+1}")
            break
        
        # Periodic save
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': current_metric,
                'args': vars(args),
            }, os.path.join(exp_dir, f'checkpoint_epoch{epoch+1}.pth'))
    
    writer.close()
    
    # =========================================================================
    # Test Set Evaluation
    # =========================================================================
    print("\nLoading best model for test evaluation...")
    best_ckpt = torch.load(os.path.join(exp_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    test_metrics = evaluate_test_set(model, test_loader, criterion, device, exp_dir)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n{'='*80}")
    print("NYU BASELINE - TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"  Best val metric: {best_metric:.4f}")
    print(f"  Test AUROC (exam): {test_metrics.get('exam_auroc', 'N/A')}")
    print(f"  Test Spec@95%Sens: {test_metrics.get('exam_spec_at_95_sens', 'N/A')}")
    print(f"  Experiment: {exp_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()