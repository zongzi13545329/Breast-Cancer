"""
Training Script for Single-View Longitudinal Fair Multi-task Model
===================================================================

✅ OPTIMIZED FOR CLASS IMBALANCE:
1. Focal Loss instead of BCE Loss
2. WeightedRandomSampler for balanced training
3. Dynamic threshold search for optimal F1
4. Enhanced metrics (AUPR, Precision-Recall curves)
5. Adaptive pos_weight (no upper limit)

Modified for single-view dataset (each sample = 2 images, not 8)

Key Changes:
- Batch structure: current/prior are [B, 1, H, W] instead of dict of 4 views
- Model expects single concatenated input instead of multi-view fusion
- Simplified preprocessing

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
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import single-view dataset
from data.dataset import (
    EMBEDSingleViewLongitudinalDataset,
    create_data_loaders,
    collate_fn
)

# Metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve
)


# ============================================================================
# Configuration
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description='Train Single-View Longitudinal Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument('--clinical_csv', type=str,
                        default="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled_new.csv")
    parser.add_argument('--metadata_csv', type=str,
                        default="/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv")
    
    # Model configuration
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b2'],
                        help='Backbone architecture')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='freeze_backbone')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # ✅ Loss configuration
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                        help='Use Focal Loss instead of BCE Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal Loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss gamma parameter')
    parser.add_argument('--max_pos_weight', type=float, default=None,
                        help='Maximum pos_weight for BCE Loss (None = no limit)')
    
    # Loss weights
    parser.add_argument('--recall_weight', type=float, default=1.0,
                        help='Weight for recall task')
    parser.add_argument('--birads_weight', type=float, default=0.1,
                        help='Weight for BI-RADS task')
    parser.add_argument('--density_weight', type=float, default=0.1,
                        help='Weight for density task')
    parser.add_argument('--fairness_lambda', type=float, default=0.1,
                        help='Weight for fairness regularization')
    
    # Image size
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 256],
                        help='Image size (H W)')
    
    # Data split
    parser.add_argument('--train_split', type=float, default=0.6,
                        help='Training set fraction')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation set fraction')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--sample_fraction', type=float, default=0.1,
                        help='Fraction of patients to use (for testing)')
    
    # ✅ Sampling strategy
    parser.add_argument('--use_balanced_sampling', action='store_true', default=True,
                        help='Use WeightedRandomSampler for balanced training')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--exp_name', type=str, default='single_view_longitudinal',
                        help='Experiment name')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Resume training
    parser.add_argument('--resume', type=str, default="./outputs/single_view_longitudinal_focal_balanced_20260122_011742/checkpoint_epoch_20.pth",
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    # Logging
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Log frequency (iterations)')
    
    args = parser.parse_args()
    return args


# ============================================================================
# ✅ NEW: Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (0-1)
        gamma: Focusing parameter (0 = CE loss, higher = more focus on hard examples)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 1] logits
            targets: [B, 1] binary labels (0 or 1)
        """
        # Ensure correct shapes
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # Compute BCE loss
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Compute p_t
        pt = torch.exp(-BCE_loss)
        
        # Compute focal loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# ============================================================================
# Simple Single-View Model
# ============================================================================

class SingleViewLongitudinalModel(nn.Module):
    """
    Simplified model for single-view longitudinal pairs.
    
    Input: Concatenated [prior | current] images → [B, 2, H, W]
    Output: Multi-task predictions
    """
    
    def __init__(
        self,
        backbone='resnet50',
        pretrained=True,
        freeze_backbone=True,
        num_races=4,
        dropout=0.3
    ):
        super().__init__()
        
        # Backbone (expects 3 channels)
        if backbone == 'resnet50':
            import torchvision.models as models
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            import torchvision.models as models
            base_model = models.efficientnet_b0(pretrained=pretrained)
            self.feature_extractor = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone_name = backbone
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("Backbone frozen!")   
        
        # Input adapter: 2 channels → 3 channels
        self.input_adapter = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        
        # Race-conditional adapter (simple)
        self.race_embeddings = nn.Embedding(num_races, 64)
        
        # Task heads
        self.recall_head = nn.Sequential(
            nn.Linear(feature_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)  # Binary classification
        )
        
        self.birads_head = nn.Sequential(
            nn.Linear(feature_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 5)  # 5 classes
        )
        
        self.density_head = nn.Sequential(
            nn.Linear(feature_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4)  # 4 classes
        )
    
    def forward(self, current, prior, race):
        """
        Args:
            current: [B, 1, H, W]
            prior: [B, 1, H, W]
            race: [B] - race labels (0-3)
        
        Returns:
            Dict with 'recall', 'birads', 'density' predictions
        """
        # Concatenate temporal dimension
        x = torch.cat([prior, current], dim=1)  # [B, 2, H, W]
        
        # Adapt to 3 channels
        x = self.input_adapter(x)  # [B, 3, H, W]
        
        # Extract features
        features = self.feature_extractor(x)  # [B, C, 1, 1]
        
        # Pool if needed
        if self.backbone_name.startswith('efficientnet'):
            features = self.pool(features)
        
        features = features.view(features.size(0), -1)  # [B, C]
        
        # Race conditioning
        race_emb = self.race_embeddings(race)  # [B, 64]
        features = torch.cat([features, race_emb], dim=1)  # [B, C+64]
        
        # Task predictions
        recall_logits = self.recall_head(features)  # [B, 1]
        birads_logits = self.birads_head(features)  # [B, 5]
        density_logits = self.density_head(features)  # [B, 4]
        
        return {
            'recall': recall_logits,
            'birads': birads_logits,
            'density': density_logits
        }


# ============================================================================
# ✅ OPTIMIZED: Multi-Task Loss with Focal Loss Support
# ============================================================================

class MultiTaskLoss(nn.Module):
    """Multi-task loss with fairness regularization and Focal Loss support."""
    
    def __init__(
        self,
        recall_weight=1.0,
        birads_weight=0.5,
        density_weight=0.3,
        pos_weight_recall=None,
        fairness_lambda=0.1,
        use_focal_loss=True,  # ✅ NEW
        focal_alpha=0.25,
        focal_gamma=2.0
    ):
        super().__init__()
        
        self.recall_weight = recall_weight
        self.birads_weight = birads_weight
        self.density_weight = density_weight
        self.fairness_lambda = fairness_lambda
        self.use_focal_loss = use_focal_loss
        
        # Loss functions
        if use_focal_loss:
            self.recall_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            print(f"✓ Using Focal Loss (α={focal_alpha}, γ={focal_gamma})")
        else:
            self.recall_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_recall)
            print(f"✓ Using BCE Loss (pos_weight={pos_weight_recall.item() if pos_weight_recall is not None else 'None'})")
        
        self.birads_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.density_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, predictions, labels, race_labels=None):
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dict with 'recall', 'birads', 'density'
            labels: Dict with 'recall', 'birads', 'density'
            race_labels: [B] race labels (for fairness)
        
        Returns:
            Dict with individual losses and total loss
        """
        total_task_loss = 0.0
        loss_dict = {}
        
        # Recall loss (binary)
        if self.recall_weight > 0:
            recall_loss = self.recall_loss_fn(
                predictions['recall'],
                labels['recall']
            )
            total_task_loss += self.recall_weight * recall_loss
            loss_dict['recall'] = recall_loss
        else:
            loss_dict['recall'] = torch.tensor(0.0, device=predictions['recall'].device)
        
        # BI-RADS loss (multi-class)
        if self.birads_weight > 0:
            birads_loss = self.birads_loss_fn(
                predictions['birads'],
                labels['birads']
            )
            total_task_loss += self.birads_weight * birads_loss
            loss_dict['birads'] = birads_loss
        else:
            loss_dict['birads'] = torch.tensor(0.0, device=predictions['recall'].device)
        
        # Density loss (multi-class)
        if self.density_weight > 0:
            density_loss = self.density_loss_fn(
                predictions['density'],
                labels['density']
            )
            total_task_loss += self.density_weight * density_loss
            loss_dict['density'] = density_loss
        else:
            loss_dict['density'] = torch.tensor(0.0, device=predictions['recall'].device)
        
        # Fairness regularization (simple demographic parity)
        fairness_loss = torch.tensor(0.0, device=predictions['recall'].device)
        if race_labels is not None and self.fairness_lambda > 0:
            recall_logits = predictions['recall']  # [B, 1] or [B]
            
            # Ensure consistent shape [B]
            if recall_logits.dim() > 1:
                recall_logits = recall_logits.squeeze(-1)  # [B, 1] -> [B]
            
            # ✅ CRITICAL FIX: Prevent 0D tensor when batch_size=1
            if recall_logits.dim() == 0:
                recall_logits = recall_logits.unsqueeze(0)  # [] -> [1]
            
            recall_probs = torch.sigmoid(recall_logits)  # [B]
            
            # Ensure race_labels is also [B]
            if race_labels.dim() > 1:
                race_labels = race_labels.squeeze(-1)
            if race_labels.dim() == 0:
                race_labels = race_labels.unsqueeze(0)
            
            # Compute mean prediction per race group
            race_means = []
            for race_id in range(4):  # 4 race groups
                race_mask = (race_labels == race_id)
                if race_mask.sum() > 0:
                    race_mean = recall_probs[race_mask].mean()
                    race_means.append(race_mean)
            
            # Penalize variance across groups
            if len(race_means) > 1:
                race_means = torch.stack(race_means)
                fairness_loss = race_means.var()
        
        loss_dict['fairness'] = fairness_loss
        
        # Total loss
        total_loss = total_task_loss + self.fairness_lambda * fairness_loss
        loss_dict['total'] = total_loss
        
        return loss_dict


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, log_freq=10):
    """Train for one epoch."""
    model.train()
    
    running_losses = defaultdict(float)
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # ✅ 单视图数据：直接获取 [B, 1, H, W]
            current = batch['current'].to(device)  # [B, 1, H, W]
            prior = batch['prior'].to(device)      # [B, 1, H, W]
            
            labels = {
                'recall': batch['labels']['recall'].to(device),
                'birads': batch['labels']['birads'].to(device),
                'density': batch['labels']['density'].to(device)
            }
            race_labels = batch['metadata']['race'].to(device)
            
            # Forward pass
            predictions = model(current, prior, race_labels)
            
            # Compute loss
            loss_dict = criterion(predictions, labels, race_labels)
            loss = loss_dict['total']
            
            if torch.isnan(loss):
                print(f"\n❌ NaN detected at batch {batch_idx}!")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                running_losses[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'recall': f"{loss_dict['recall'].item():.4f}"
            })
            
            # Log to TensorBoard
            if batch_idx % log_freq == 0:
                global_step = epoch * num_batches + batch_idx
                for key, value in loss_dict.items():
                    writer.add_scalar(f'Train_Batch/Loss_{key}', value.item(), global_step)
        
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Average losses
    avg_losses = {
        key: value / num_batches
        for key, value in running_losses.items()
    }
    
    # Log to TensorBoard
    for key, value in avg_losses.items():
        writer.add_scalar(f'Train/Loss_{key}', value, epoch)
    
    return avg_losses


def validate_epoch(model, val_loader, criterion, device, epoch, writer):
    """Validate for one epoch."""
    model.eval()
    
    running_losses = defaultdict(float)
    num_batches = len(val_loader)
    
    # Collect predictions and labels
    all_predictions = defaultdict(list)
    all_labels = defaultdict(list)
    all_races = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                # ✅ 单视图数据
                current = batch['current'].to(device)
                prior = batch['prior'].to(device)
                
                labels = {
                    'recall': batch['labels']['recall'].to(device),
                    'birads': batch['labels']['birads'].to(device),
                    'density': batch['labels']['density'].to(device)
                }
                race_labels = batch['metadata']['race'].to(device)
                
                # Forward pass
                predictions = model(current, prior, race_labels)
                
                # Compute loss
                loss_dict = criterion(predictions, labels, race_labels)
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    running_losses[key] += value.item()
                
                # Store predictions and labels
                for task in ['recall', 'birads', 'density']:
                    all_predictions[task].append(predictions[task].cpu())
                    all_labels[task].append(labels[task].cpu())
                
                all_races.append(race_labels.cpu())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}"
                })
            
            except Exception as e:
                print(f"\nError in validation batch {batch_idx}: {str(e)}")
                continue
    
    # Average losses
    avg_losses = {
        key: value / num_batches
        for key, value in running_losses.items()
    }
    
    # Concatenate predictions
    for task in ['recall', 'birads', 'density']:
        all_predictions[task] = torch.cat(all_predictions[task], dim=0)
        all_labels[task] = torch.cat(all_labels[task], dim=0)
    
    all_races = torch.cat(all_races, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels, all_races)
    
    # Log to TensorBoard
    for key, value in avg_losses.items():
        writer.add_scalar(f'Val/Loss_{key}', value, epoch)
    
    for task in ['recall', 'birads', 'density']:
        if task in metrics:
            for metric_name, metric_value in metrics[task].items():
                writer.add_scalar(f'Val/{task}_{metric_name}', metric_value, epoch)
    
    # Log fairness metrics
    if 'fairness' in metrics:
        for metric_name, metric_value in metrics['fairness'].items():
            writer.add_scalar(f'Val/Fairness_{metric_name}', metric_value, epoch)
    
    return avg_losses, metrics


# ============================================================================
# ✅ OPTIMIZED: Enhanced Metrics with Threshold Search
# ============================================================================

def compute_metrics(predictions, labels, races):
    """
    Compute comprehensive metrics with dynamic threshold search.
    
    Includes:
    - AUROC, AUPR
    - Optimal threshold search for F1
    - Precision, Recall, TPR, FPR at optimal threshold
    - Fairness metrics across racial groups
    """
    metrics = defaultdict(dict)
    
    # Recall task (binary)
    recall_logits = predictions['recall']
    recall_labels = labels['recall'].squeeze().numpy().astype(int)
    recall_probs = torch.sigmoid(recall_logits).squeeze().numpy()
    
    # Filter valid labels
    valid_mask = (recall_labels >= 0)
    if valid_mask.sum() == 0:
        print("⚠️  Warning: No valid samples for recall metrics")
        return metrics
    
    recall_probs = recall_probs[valid_mask]
    recall_labels = recall_labels[valid_mask]
    races_valid = races.numpy()[valid_mask]
    # =========================================================================
    # 1. AUROC and AUPR
    # =========================================================================
    try:
        metrics['recall']['auroc'] = roc_auc_score(recall_labels, recall_probs)
    except:
        metrics['recall']['auroc'] = 0.0
    
    try:
        metrics['recall']['aupr'] = average_precision_score(recall_labels, recall_probs)
    except:
        metrics['recall']['aupr'] = 0.0
    
    # =========================================================================
    # ✅ 2. OPTIMAL THRESHOLD SEARCH
    # =========================================================================
    try:
        precisions, recalls, thresholds = precision_recall_curve(recall_labels, recall_probs)
        
        # Compute F1 scores for all thresholds
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        metrics['recall']['best_threshold'] = float(best_threshold)
        metrics['recall']['best_f1'] = float(best_f1)
        
        # Compute predictions at optimal threshold
        recall_pred_optimal = (recall_probs >= best_threshold).astype(int)
        
        # Metrics at optimal threshold
        metrics['recall']['f1_optimal'] = f1_score(recall_labels, recall_pred_optimal, zero_division=0)
        metrics['recall']['precision_optimal'] = precision_score(recall_labels, recall_pred_optimal, zero_division=0)
        metrics['recall']['recall_optimal'] = recall_score(recall_labels, recall_pred_optimal, zero_division=0)
        
    except Exception as e:
        print(f"⚠️  Warning: Could not compute optimal threshold: {e}")
        best_threshold = 0.5
        recall_pred_optimal = (recall_probs >= 0.5).astype(int)
    
    # =========================================================================
    # 3. Standard Metrics at Default Threshold (0.5)
    # =========================================================================
    recall_pred = (recall_probs > 0.5).astype(int)
    
    # Accuracy metrics
    metrics['recall']['accuracy'] = accuracy_score(recall_labels, recall_pred)
    metrics['recall']['balanced_accuracy'] = balanced_accuracy_score(recall_labels, recall_pred)
    
    # F1, Precision, Recall
    metrics['recall']['f1'] = f1_score(recall_labels, recall_pred, zero_division=0)
    metrics['recall']['precision'] = precision_score(recall_labels, recall_pred, zero_division=0)
    metrics['recall']['tpr'] = recall_score(recall_labels, recall_pred, zero_division=0)
    
    # =========================================================================
    # 4. Confusion Matrix Metrics
    # =========================================================================
    try:
        tn, fp, fn, tp = confusion_matrix(recall_labels, recall_pred, labels=[0, 1]).ravel()
        
        # FPR (False Positive Rate)
        metrics['recall']['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # TNR (True Negative Rate / Specificity)
        metrics['recall']['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['recall']['specificity'] = metrics['recall']['tnr']
        
        # PPV (Positive Predictive Value)
        metrics['recall']['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # NPV (Negative Predictive Value)
        metrics['recall']['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Store raw confusion matrix values
        metrics['recall']['tn'] = int(tn)
        metrics['recall']['fp'] = int(fp)
        metrics['recall']['fn'] = int(fn)
        metrics['recall']['tp'] = int(tp)
        
    except Exception as e:
        print(f"⚠️  Error computing confusion matrix metrics: {e}")
        metrics['recall']['fpr'] = 0.0
        metrics['recall']['tnr'] = 0.0
        metrics['recall']['specificity'] = 0.0
        metrics['recall']['ppv'] = 0.0
        metrics['recall']['npv'] = 0.0
    
    # =========================================================================
    # ✅ 5. Metrics at Different Thresholds
    # =========================================================================
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        recall_pred_t = (recall_probs >= threshold).astype(int)
        metrics['recall'][f'f1_at_{threshold}'] = f1_score(
            recall_labels, recall_pred_t, zero_division=0
        )
        metrics['recall'][f'precision_at_{threshold}'] = precision_score(
            recall_labels, recall_pred_t, zero_division=0
        )
        metrics['recall'][f'recall_at_{threshold}'] = recall_score(
            recall_labels, recall_pred_t, zero_division=0
        )
    
    # =========================================================================
    # 6. Auxiliary Tasks (BI-RADS and Density)
    # =========================================================================
    for task in ['birads', 'density']:
        if task in predictions:
            pred = predictions[task].argmax(dim=1).numpy()
            true = labels[task].numpy()
            
            # Filter valid labels
            valid_mask = true >= 0
            if valid_mask.sum() > 0:
                pred = pred[valid_mask]
                true = true[valid_mask]
                
                metrics[task]['accuracy'] = accuracy_score(true, pred)
                try:
                    metrics[task]['f1_macro'] = f1_score(
                        true, 
                        pred, 
                        average='macro',
                        zero_division=0
                    )
                except:
                    metrics[task]['f1_macro'] = 0.0
    
    # =========================================================================
    # 7. Fairness Metrics
    # =========================================================================
    metrics['fairness'] = compute_fairness_metrics(
        recall_pred,
        recall_labels,
        races_valid
    )
    
    return metrics


def compute_fairness_metrics(predictions, labels, races):
    """Compute fairness metrics across racial groups."""
    fairness_metrics = {}
    
    # Ensure inputs are numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(races, torch.Tensor):
        races = races.cpu().numpy()
    
    # White vs Black comparison
    race_0_mask = (races == 0)
    race_1_mask = (races == 1)
    
    if race_0_mask.sum() == 0 or race_1_mask.sum() == 0:
        return {
            'demographic_parity_diff': 0.0,
            'accuracy_diff': 0.0
        }
    
    # Demographic Parity
    white_pos_rate = (predictions[race_0_mask] > 0).mean()
    black_pos_rate = (predictions[race_1_mask] > 0).mean()
    fairness_metrics['demographic_parity_diff'] = abs(white_pos_rate - black_pos_rate)
    
    # Accuracy per race
    for race_id, race_name in enumerate(['white', 'black', 'asian', 'other']):
        race_mask = (races == race_id)
        if race_mask.sum() > 0:
            race_acc = (predictions[race_mask] == labels[race_mask]).mean()
            fairness_metrics[f'{race_name}_accuracy'] = race_acc
    
    # Accuracy difference
    fairness_metrics['accuracy_diff'] = abs(
        fairness_metrics.get('white_accuracy', 0) - 
        fairness_metrics.get('black_accuracy', 0)
    )
    
    return fairness_metrics


# ============================================================================
# ✅ OPTIMIZED: Adaptive Class Weight Computation
# ============================================================================

def compute_pos_weight_from_csv(clinical_csv, device, max_pos_weight=None):
    """
    Compute pos_weight for BCEWithLogitsLoss from clinical CSV.
    
    Args:
        clinical_csv: Path to clinical CSV file
        device: torch device
        max_pos_weight: Maximum allowed pos_weight (None = no limit)
    
    Returns:
        pos_weight: torch.Tensor of shape [1]
    """
    print("\n" + "="*70)
    print("Computing pos_weight for Binary Classification")
    print("="*70)
    
    try:
        # Load full dataset
        df = pd.read_csv(clinical_csv, low_memory=False)
        labels = df['new_label'].values
        
        # Filter valid labels
        valid_mask = (labels >= 0) & (~pd.isna(labels))
        valid_labels = labels[valid_mask].astype(int)
        
        if len(valid_labels) == 0:
            print("❌ No valid labels found")
            return None
        
        # Convert to binary (0 vs 1+2)
        binary_labels = np.where(valid_labels >= 1, 1, 0)
        
        # Count
        counts = np.bincount(binary_labels)
        
        print(f"\nClass distribution:")
        print(f"  Class 0 (No Recall): {counts[0]:,} ({counts[0]/len(binary_labels)*100:.1f}%)")
        print(f"  Class 1 (Need Recall): {counts[1]:,} ({counts[1]/len(binary_labels)*100:.1f}%)")
        
        # Compute pos_weight
        pos_weight = float(counts[0]) / float(counts[1])
        
        print(f"\n✓ Raw pos_weight: {pos_weight:.3f}")
        
        # Clip if needed
        if max_pos_weight is not None and pos_weight > max_pos_weight:
            print(f"  Clipping pos_weight from {pos_weight:.3f} to {max_pos_weight:.3f}")
            pos_weight = max_pos_weight
        else:
            print(f"  No clipping applied (max_pos_weight={'None' if max_pos_weight is None else max_pos_weight})")
        
        pos_weight_tensor = torch.FloatTensor([pos_weight]).to(device)
        
        print(f"\n✓ Final pos_weight: {pos_weight_tensor.item():.3f}")
        print("="*70 + "\n")
        
        return pos_weight_tensor
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    args = get_args()
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name_full = f"{args.exp_name}_{'focal' if args.use_focal_loss else 'bce'}_balanced_{timestamp}"
    exp_dir = os.path.join(args.output_dir, exp_name_full)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}")
    
    # Save configuration
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    
    # ✅ Create data loaders with balanced sampling
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        clinical_csv=args.clinical_csv,
        metadata_csv=args.metadata_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
        random_seed=args.random_seed,
        sample_fraction=args.sample_fraction,
        image_size=tuple(args.image_size),
        handle_missing_prior='mixed',
        prior_required=False,
        use_cache=False,
        use_balanced_sampling=args.use_balanced_sampling,  # ✅ Enable balanced sampling
        verbose=True
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = SingleViewLongitudinalModel(
        backbone=args.backbone,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        num_races=4,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ✅ Compute class weights (only if not using Focal Loss)
    pos_weight = None
    if not args.use_focal_loss:
        pos_weight = compute_pos_weight_from_csv(
            clinical_csv=args.clinical_csv,
            device=device,
            max_pos_weight=args.max_pos_weight  # None = no limit
        )
    
    # Loss function
    criterion = MultiTaskLoss(
        recall_weight=args.recall_weight,
        birads_weight=args.birads_weight,
        density_weight=args.density_weight,
        pos_weight_recall=pos_weight,
        fairness_lambda=args.fairness_lambda,
        use_focal_loss=args.use_focal_loss,  # ✅ Enable Focal Loss
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_f1 = 0.0  # ✅ Track best F1
    patience_counter = 0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
    
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 70)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, args.log_freq
        )
        print(f"Train Loss: {train_losses['total']:.4f} | "
              f"Recall: {train_losses['recall']:.4f}")
        
        # Validate
        val_losses, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        print(f"Val Loss: {val_losses['total']:.4f} | "
              f"Recall: {val_losses['recall']:.4f}")
        
        if 'recall' in val_metrics:
            print(f"Val Recall Metrics:")
            print(f"  AUROC: {val_metrics['recall'].get('auroc', 0):.4f} | " +
                  f"AUPR: {val_metrics['recall'].get('aupr', 0):.4f}")
            print(f"  Best Threshold: {val_metrics['recall'].get('best_threshold', 0.5):.3f} | " +
                  f"Best F1: {val_metrics['recall'].get('best_f1', 0):.4f}")
            print(f"  F1 (0.5): {val_metrics['recall'].get('f1', 0):.4f} | " +
                  f"Precision: {val_metrics['recall'].get('precision', 0):.4f}")
            print(f"  TPR: {val_metrics['recall'].get('tpr', 0):.4f} | " +
                  f"FPR: {val_metrics['recall'].get('fpr', 0):.4f}")
        
        if 'fairness' in val_metrics:
            print(f"Fairness Metrics:")
            print(f"  DP Diff: {val_metrics['fairness'].get('demographic_parity_diff', 0):.4f} | " +
                  f"Acc Diff (W-B): {val_metrics['fairness'].get('accuracy_diff', 0):.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_losses['total'])
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
        
        # ✅ Save checkpoint based on BOTH loss and F1
        current_f1 = val_metrics['recall'].get('best_f1', 0)
        is_best = (val_losses['total'] < best_val_loss) or (current_f1 > best_val_f1)
        
        if is_best:
            if current_f1 > best_val_f1:
                print(f"✓ New best F1: {current_f1:.4f} (previous: {best_val_f1:.4f})")
                best_val_f1 = current_f1
            if val_losses['total'] < best_val_loss:
                print(f"✓ New best loss: {val_losses['total']:.4f} (previous: {best_val_loss:.4f})")
                best_val_loss = val_losses['total']
            
            patience_counter = 0
            
            # Save best model
            best_path = os.path.join(exp_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'best_val_f1': best_val_f1,
                'args': vars(args)
            }, best_path)
            print(f"✓ Saved best model to {best_path}")
        else:
            patience_counter += 1
        
        # Regular checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(exp_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'best_val_f1': best_val_f1,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs without improvement")
            break
    
    writer.close()
    print(f"\n✓ Training completed. Results saved to {exp_dir}")


if __name__ == '__main__':
    main()