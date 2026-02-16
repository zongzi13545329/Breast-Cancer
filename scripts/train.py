"""
Robust Training Script - Optimized for Imbalanced Data (9.9% positive)
======================================================================

Key optimizations for extreme class imbalance:
1. ✅ More aggressive Focal Loss (α=0.75, γ=3.0)
2. ✅ Gradient accumulation for stable training
3. ✅ Separate monitoring for pos/neg samples
4. ✅ Enhanced metrics (AUROC/AUPR priority)
5. ✅ Cost-sensitive evaluation
6. ✅ Per-race fairness tracking

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
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import dataset
from data.dataset import (
    EMBEDExamSideLongitudinalDataset,
    create_data_loaders,
    collate_fn
)

# Metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve
)


# ============================================================================
# ✅ ROBUST: Enhanced Multi-View Aggregator with NaN Prevention
# ============================================================================

class RobustMultiViewAggregator(nn.Module):
    """
    Robust multi-view aggregator with comprehensive NaN prevention.
    
    Key improvements:
    1. Proper attention masking (prevents attending to padding)
    2. Safe normalization (adds epsilon to prevent division by zero)
    3. Numerical stability in all operations
    4. Fallback mechanisms for edge cases
    """
    
    def __init__(self, feature_dim, num_heads=4, num_queries=1, dropout=0.1, eps=1e-6):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.eps = eps
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        # Position encoding
        self.position_embeddings = nn.Parameter(torch.randn(1, 4, feature_dim))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim, eps=self.eps)
        self.norm2 = nn.LayerNorm(feature_dim, eps=self.eps)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        
        # Final projection
        if num_queries > 1:
            self.output_proj = nn.Sequential(
                nn.Linear(feature_dim * num_queries, feature_dim),
                nn.LayerNorm(feature_dim, eps=self.eps)
            )
    
    def forward(self, features, mask=None):
        """
        Args:
            features: [B, N_views, feature_dim]
            mask: [B, N_views] where 1=valid, 0=padding
        
        Returns:
            aggregated: [B, feature_dim]
        """
        B, N, D = features.shape
        
        # Check for NaN in input
        if torch.isnan(features).any():
            print("⚠️  Warning: NaN detected in input features!")
            features = torch.nan_to_num(features, nan=0.0)
        
        # Safe position encoding
        if N <= 4:
            pos_enc = self.position_embeddings[:, :N, :]
        else:
            pos_enc = self.position_embeddings[:, :4, :].repeat(1, (N + 3) // 4, 1)[:, :N, :]
        
        features = features + pos_enc
        
        # Expand queries
        queries = self.query_tokens.expand(B, -1, -1)
        
        # Prepare attention mask
        if mask is not None:
            key_padding_mask = (mask == 0)
            all_padding = (mask.sum(dim=1) == 0)
            if all_padding.any():
                print(f"⚠️  Warning: {all_padding.sum().item()} samples have all padding!")
                mask[all_padding, 0] = 1
                key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None
        
        # Cross-attention with proper masking
        try:
            attn_output, attn_weights = self.cross_attn(
                queries, features, features,
                key_padding_mask=key_padding_mask,
                need_weights=True
            )
            
            if torch.isnan(attn_output).any():
                print("⚠️  Warning: NaN in attention output!")
                attn_output = torch.nan_to_num(attn_output, nan=0.0)
            
        except Exception as e:
            print(f"⚠️  Attention error: {e}")
            attn_output = queries
        
        # Residual + Norm
        queries = self.norm1(queries + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(queries)
        
        if torch.isnan(ffn_output).any():
            print("⚠️  Warning: NaN in FFN output!")
            ffn_output = torch.nan_to_num(ffn_output, nan=0.0)
        
        output = self.norm2(queries + ffn_output)
        
        # Aggregate queries
        if self.num_queries > 1:
            output = output.view(B, -1)
            output = self.output_proj(output)
        else:
            output = output.squeeze(1)
        
        # Final NaN check
        if torch.isnan(output).any():
            print("⚠️  Warning: NaN in final output! Replacing with zeros.")
            output = torch.nan_to_num(output, nan=0.0)
        
        return output


class RobustGatedTemporalFusion(nn.Module):
    """Gated temporal fusion with numerical stability."""
    
    def __init__(self, feature_dim, dropout=0.1, eps=1e-6):
        super().__init__()
        
        self.eps = eps
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim, eps=eps),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # Feature transformation
        self.prior_proj = nn.Linear(feature_dim, feature_dim)
        self.current_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim, eps=eps),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, prior_features, current_features):
        """Fuse with NaN checking."""
        if torch.isnan(prior_features).any():
            print("⚠️  NaN in prior_features!")
            prior_features = torch.nan_to_num(prior_features, nan=0.0)
        
        if torch.isnan(current_features).any():
            print("⚠️  NaN in current_features!")
            current_features = torch.nan_to_num(current_features, nan=0.0)
        
        concat = torch.cat([prior_features, current_features], dim=1)
        gate = self.gate(concat)
        gate = torch.clamp(gate, min=self.eps, max=1.0 - self.eps)
        
        prior_transformed = self.prior_proj(prior_features)
        current_transformed = self.current_proj(current_features)
        
        fused = gate * prior_transformed + (1 - gate) * current_transformed
        output = self.output_proj(fused)
        
        if torch.isnan(output).any():
            print("⚠️  NaN in fusion output!")
            output = torch.nan_to_num(output, nan=0.0)
        
        return output


class RobustTaskHead(nn.Module):
    """Task head with numerical stability."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, eps=1e-6):
        super().__init__()
        
        self.eps = eps
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=eps),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.hidden_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=eps),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.hidden_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=eps),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if torch.isnan(x).any():
            print("⚠️  NaN in task head input!")
            x = torch.nan_to_num(x, nan=0.0)
        
        h = self.input_proj(x)
        h = h + self.hidden_block1(h)
        h = h + self.hidden_block2(h)
        output = self.output_proj(h)
        
        if torch.isnan(output).any():
            print("⚠️  NaN in task head output!")
            output = torch.nan_to_num(output, nan=0.0)
        
        return output


class RobustExamSideModel(nn.Module):
    """Robust exam-side model with comprehensive NaN prevention."""
    
    def __init__(
        self,
        backbone='resnet50',
        pretrained=True,
        num_races=4,
        dropout=0.3,
        num_attention_queries=2,
        use_gated_fusion=True,
        task_hidden_dim=512,
        feature_reduction_dim=512,  # ✅ 降维目标维度
        eps=1e-6
    ):
        super().__init__()
        
        self.use_gated_fusion = use_gated_fusion
        self.eps = eps
        
        # Frozen Backbone
        if backbone == 'resnet50':
            import torchvision.models as models
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            backbone_feature_dim = 2048
        elif backbone == 'resnet101':
            import torchvision.models as models
            base_model = models.resnet101(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            backbone_feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            import torchvision.models as models
            base_model = models.efficientnet_b0(pretrained=pretrained)
            self.feature_extractor = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            backbone_feature_dim = 1280
        elif backbone == 'efficientnet_b2':
            import torchvision.models as models
            base_model = models.efficientnet_b2(pretrained=pretrained)
            self.feature_extractor = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            backbone_feature_dim = 1408
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone_name = backbone
        
        # ✅ 特征降维层
        self.feature_reducer = nn.Sequential(
            nn.Linear(backbone_feature_dim, feature_reduction_dim),
            nn.LayerNorm(feature_reduction_dim, eps=eps),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Freeze backbone
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        print(f"✓ Backbone ({backbone}) is FROZEN")
        print(f"✓ Feature reduction: {backbone_feature_dim} → {feature_reduction_dim}")
        
        # ✅ 使用降维后的维度
        reduced_dim = feature_reduction_dim
        
        # Trainable Components
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.input_adapter.weight, mode='fan_out', nonlinearity='relu')
        
        self.current_aggregator = RobustMultiViewAggregator(
            feature_dim=reduced_dim,  # ✅ 使用 reduced_dim
            num_heads=4,
            num_queries=num_attention_queries,
            dropout=dropout,
            eps=eps
        )
        
        self.prior_aggregator = RobustMultiViewAggregator(
            feature_dim=reduced_dim,  # ✅ 使用 reduced_dim
            num_heads=4,
            num_queries=num_attention_queries,
            dropout=dropout,
            eps=eps
        )
        
        if use_gated_fusion:
            self.temporal_fusion = RobustGatedTemporalFusion(
                feature_dim=reduced_dim,  # ✅ 使用 reduced_dim
                dropout=dropout,
                eps=eps
            )
            print("✓ Using Robust Gated Temporal Fusion")
        else:
            self.temporal_fusion = nn.Sequential(
                nn.Linear(reduced_dim * 2, reduced_dim),  # ✅ 修复：使用 reduced_dim
                nn.LayerNorm(reduced_dim, eps=eps),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            print("✓ Using Robust Simple Temporal Fusion")
        
        # ✅ Race conditioning - 减小 embedding 维度
        race_embed_dim = 64  # ✅ 从 128 减少到 64
        self.race_embeddings = nn.Embedding(num_races, race_embed_dim)
        nn.init.normal_(self.race_embeddings.weight, mean=0.0, std=0.02)
        
        self.race_fusion = nn.Sequential(
            nn.Linear(reduced_dim + race_embed_dim, reduced_dim),  # ✅ 修复：使用 reduced_dim
            nn.LayerNorm(reduced_dim, eps=eps),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ✅ Task heads - 使用 reduced_dim
        self.recall_head = RobustTaskHead(
            input_dim=reduced_dim,  # ✅ 修复：使用 reduced_dim
            hidden_dim=task_hidden_dim,
            output_dim=1,
            dropout=dropout,
            eps=eps
        )
        
        self.birads_head = RobustTaskHead(
            input_dim=reduced_dim,  # ✅ 修复：使用 reduced_dim
            hidden_dim=task_hidden_dim,
            output_dim=5,
            dropout=dropout,
            eps=eps
        )
        
        self._initialize_trainable_weights()
        self._register_gradient_hooks()
    
    def _initialize_trainable_weights(self):
        """Safe weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m not in self.feature_extractor.modules():
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m not in self.feature_extractor.modules():
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)
    
    def _register_gradient_hooks(self):
        """Register hooks to monitor gradients."""
        def gradient_hook(grad):
            if torch.isnan(grad).any():
                print("⚠️  NaN detected in gradients!")
                return torch.nan_to_num(grad, nan=0.0)
            if torch.isinf(grad).any():
                print("⚠️  Inf detected in gradients!")
                return torch.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
            return grad
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.register_hook(gradient_hook)
    
    def extract_view_features(self, views, mask):
        """Extract features with NaN prevention."""
        B, N_views, C, H, W = views.shape
        
        if torch.isnan(views).any():
            print("⚠️  NaN in input views!")
            views = torch.nan_to_num(views, nan=0.0)
        
        views_flat = views.view(B * N_views, C, H, W)
        views_flat = torch.clamp(views_flat, min=0.0, max=1.0)
        views_flat = self.input_adapter(views_flat)
        
        with torch.no_grad():
            self.feature_extractor.eval()
            features_flat = self.feature_extractor(views_flat)
            
            if self.backbone_name.startswith('efficientnet'):
                features_flat = self.pool(features_flat)
        
        if torch.isnan(features_flat).any():
            print("⚠️  NaN in backbone features!")
            features_flat = torch.nan_to_num(features_flat, nan=0.0)
        
        features_flat = features_flat.view(B * N_views, -1)
        
        # ✅ 应用特征降维
        features_flat = self.feature_reducer(features_flat)
        
        feature_dim = features_flat.size(-1)
        features = features_flat.view(B, N_views, feature_dim)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            features = features * mask_expanded
        
        return features
    
    def forward(self, current_views, prior_views, current_mask, prior_mask, race):
        """Forward pass with comprehensive NaN checking."""
        
        current_features = self.extract_view_features(current_views, current_mask)
        prior_features = self.extract_view_features(prior_views, prior_mask)
        
        current_aggregated = self.current_aggregator(current_features, current_mask)
        prior_aggregated = self.prior_aggregator(prior_features, prior_mask)
        
        if self.use_gated_fusion:
            temporal_features = self.temporal_fusion(prior_aggregated, current_aggregated)
        else:
            temporal_concat = torch.cat([prior_aggregated, current_aggregated], dim=1)
            temporal_features = self.temporal_fusion(temporal_concat)
        
        race_emb = self.race_embeddings(race)
        
        if torch.isnan(race_emb).any():
            print("⚠️  NaN in race embeddings!")
            race_emb = torch.nan_to_num(race_emb, nan=0.0)
        
        race_conditioned = torch.cat([temporal_features, race_emb], dim=1)
        final_features = self.race_fusion(race_conditioned)
        
        recall_logits = self.recall_head(final_features)
        birads_logits = self.birads_head(final_features)
        
        return {
            'recall': recall_logits,
            'birads': birads_logits
        }
    
    def get_trainable_parameters(self):
        """Get trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def print_trainable_status(self):
        """Print parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "="*70)
        print("MODEL PARAMETER STATISTICS")
        print("="*70)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params*100:.1f}%")
        print("="*70 + "\n")


# ============================================================================
# ✅ OPTIMIZED: Focal Loss for 9.9% Imbalance
# ============================================================================

class RobustFocalLoss(nn.Module):
    """Focal Loss optimized for extreme imbalance (9.9% positive)."""
    
    def __init__(self, alpha=0.75, gamma=3.0, reduction='mean', eps=1e-7):
        """
        Args:
            alpha: Weight for positive class (0.75 for 9.9% imbalance)
            gamma: Focusing parameter (3.0 for hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, inputs, targets):
        if inputs.dim() > 1:
            inputs = inputs.squeeze()
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("⚠️  NaN/Inf in loss inputs!")
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        inputs = torch.clamp(inputs, min=-10.0, max=10.0)
        
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        BCE_loss = torch.clamp(BCE_loss, min=self.eps, max=100.0)
        
        pt = torch.exp(-BCE_loss)
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if torch.isnan(F_loss).any():
            print("⚠️  NaN in focal loss output!")
            F_loss = torch.nan_to_num(F_loss, nan=0.0)
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss optimized for imbalanced data."""
    
    def __init__(
        self,
        recall_weight=1.0,
        birads_weight=0.1,
        pos_weight_recall=None,
        fairness_lambda=0.1,
        use_focal_loss=True,
        focal_alpha=0.75,  # ✅ Increased from 0.25
        focal_gamma=3.0,   # ✅ Increased from 2.0
        eps=1e-7
    ):
        super().__init__()
        
        self.recall_weight = recall_weight
        self.birads_weight = birads_weight
        self.fairness_lambda = fairness_lambda
        self.use_focal_loss = use_focal_loss
        self.eps = eps
        
        if use_focal_loss:
            self.recall_loss_fn = RobustFocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                eps=eps
            )
            print(f"✓ Using Robust Focal Loss (α={focal_alpha}, γ={focal_gamma}) for 9.9% imbalance")
        else:
            self.recall_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_recall)
            print(f"✓ Using BCE Loss")
        
        self.birads_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, predictions, labels, race_labels=None):
        loss_dict = {}
        total_task_loss = 0.0
        
        # Recall loss
        if self.recall_weight > 0:
            try:
                recall_loss = self.recall_loss_fn(
                    predictions['recall'],
                    labels['recall']
                )
                
                if torch.isnan(recall_loss):
                    print("⚠️  NaN in recall loss! Setting to 0.")
                    recall_loss = torch.tensor(0.0, device=predictions['recall'].device)
                
                total_task_loss += self.recall_weight * recall_loss
                loss_dict['recall'] = recall_loss
            except Exception as e:
                print(f"⚠️  Error computing recall loss: {e}")
                loss_dict['recall'] = torch.tensor(0.0, device=predictions['recall'].device)
        else:
            loss_dict['recall'] = torch.tensor(0.0, device=predictions['recall'].device)
        
        # BI-RADS loss
        if self.birads_weight > 0:
            try:
                birads_loss = self.birads_loss_fn(
                    predictions['birads'],
                    labels['birads']
                )
                
                if torch.isnan(birads_loss):
                    print("⚠️  NaN in birads loss! Setting to 0.")
                    birads_loss = torch.tensor(0.0, device=predictions['recall'].device)
                
                total_task_loss += self.birads_weight * birads_loss
                loss_dict['birads'] = birads_loss
            except Exception as e:
                print(f"⚠️  Error computing birads loss: {e}")
                loss_dict['birads'] = torch.tensor(0.0, device=predictions['recall'].device)
        else:
            loss_dict['birads'] = torch.tensor(0.0, device=predictions['recall'].device)
        
        # Fairness regularization
        fairness_loss = torch.tensor(0.0, device=predictions['recall'].device)
        if race_labels is not None and self.fairness_lambda > 0:
            try:
                recall_logits = predictions['recall']
                
                if recall_logits.dim() > 1:
                    recall_logits = recall_logits.squeeze(-1)
                if recall_logits.dim() == 0:
                    recall_logits = recall_logits.unsqueeze(0)
                
                recall_logits = torch.clamp(recall_logits, min=-10.0, max=10.0)
                recall_probs = torch.sigmoid(recall_logits)
                
                if race_labels.dim() > 1:
                    race_labels = race_labels.squeeze(-1)
                if race_labels.dim() == 0:
                    race_labels = race_labels.unsqueeze(0)
                
                race_means = []
                for race_id in range(4):
                    race_mask = (race_labels == race_id)
                    if race_mask.sum() > 0:
                        race_mean = recall_probs[race_mask].mean()
                        if not torch.isnan(race_mean):
                            race_means.append(race_mean)
                
                if len(race_means) > 1:
                    race_means = torch.stack(race_means)
                    fairness_loss = race_means.var()
                    
                    if torch.isnan(fairness_loss):
                        fairness_loss = torch.tensor(0.0, device=predictions['recall'].device)
            
            except Exception as e:
                print(f"⚠️  Error computing fairness loss: {e}")
                fairness_loss = torch.tensor(0.0, device=predictions['recall'].device)
        
        loss_dict['fairness'] = fairness_loss
        
        total_loss = total_task_loss + self.fairness_lambda * fairness_loss
        
        if torch.isnan(total_loss):
            print("⚠️  NaN in total loss! Using task loss only.")
            total_loss = total_task_loss
        
        loss_dict['total'] = total_loss
        
        return loss_dict

# ============================================================================
# ✅ OPTIMIZED: Training Loop with Pos/Neg Monitoring
# ============================================================================

def train_epoch(
    model, train_loader, criterion, optimizer, device, epoch, writer,
    log_freq=10, use_amp=False, scaler=None, accumulation_steps=1
):
    """
    Train for one epoch with:
    - ✅ Gradient accumulation
    - ✅ Separate pos/neg loss tracking
    - ✅ NaN detection
    """
    model.train()
    model.feature_extractor.eval()
    
    running_losses = defaultdict(float)
    pos_losses = []  # ✅ Track positive sample losses
    neg_losses = []  # ✅ Track negative sample losses
    num_batches = len(train_loader)
    nan_count = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Load data
            current_views = batch['current_views'].to(device)
            prior_views = batch['prior_views'].to(device)
            current_mask = batch['current_mask'].to(device)
            prior_mask = batch['prior_mask'].to(device)
            
            # Filter invalid samples
            valid_samples = (current_mask.sum(dim=1) > 0) & (prior_mask.sum(dim=1) > 0)
            
            if not valid_samples.all():
                invalid_count = (~valid_samples).sum().item()
                current_views = current_views[valid_samples]
                prior_views = prior_views[valid_samples]
                current_mask = current_mask[valid_samples]
                prior_mask = prior_mask[valid_samples]
                
                if len(current_views) == 0:
                    continue
                
                labels = {
                    'recall': batch['labels']['recall'][valid_samples.cpu()].to(device),
                    'birads': batch['labels']['birads'][valid_samples.cpu()].to(device),
                }
                race_labels = batch['metadata']['race'][valid_samples.cpu()].to(device)
            else:
                labels = {
                    'recall': batch['labels']['recall'].to(device),
                    'birads': batch['labels']['birads'].to(device),
                }
                race_labels = batch['metadata']['race'].to(device)
            
            # Forward pass
            if use_amp:
                with autocast():
                    predictions = model(
                        current_views, prior_views,
                        current_mask, prior_mask,
                        race_labels
                    )
                    loss_dict = criterion(predictions, labels, race_labels)
                    loss = loss_dict['total'] / accumulation_steps  # ✅ Scale for accumulation
            else:
                predictions = model(
                    current_views, prior_views,
                    current_mask, prior_mask,
                    race_labels
                )
                loss_dict = criterion(predictions, labels, race_labels)
                loss = loss_dict['total'] / accumulation_steps
            
            # ✅ Track pos/neg losses separately
            pos_mask = (labels['recall'] >= 0.5).squeeze()
            neg_mask = (labels['recall'] < 0.5).squeeze()
            
            if pos_mask.any():
                with torch.no_grad():
                    pos_loss_val = criterion.recall_loss_fn(
                        predictions['recall'][pos_mask],
                        labels['recall'][pos_mask]
                    ).item()
                    pos_losses.append(pos_loss_val)
            
            if neg_mask.any():
                with torch.no_grad():
                    neg_loss_val = criterion.recall_loss_fn(
                        predictions['recall'][neg_mask],
                        labels['recall'][neg_mask]
                    ).item()
                    neg_losses.append(neg_loss_val)
            
            # Check for NaN
            if torch.isnan(loss):
                nan_count += 1
                print(f"\n❌ NaN detected at batch {batch_idx}! (Total NaN: {nan_count})")
                if nan_count > 10:
                    raise ValueError("Training unstable - too many NaN losses")
                continue
            
            if loss.item() > 1000:
                print(f"\n⚠️  Extreme loss value: {loss.item():.2f} at batch {batch_idx}")
                continue
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # ✅ Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(grad_norm):
                        print(f"\n⚠️  NaN gradient norm at batch {batch_idx}")
                        optimizer.zero_grad()
                        continue
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(grad_norm):
                        print(f"\n⚠️  NaN gradient norm at batch {batch_idx}")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                running_losses[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'pos': f"{np.mean(pos_losses[-10:]) if pos_losses else 0:.3f}",
                'neg': f"{np.mean(neg_losses[-10:]) if neg_losses else 0:.3f}",
                'nan': nan_count
            })
            
            # Log to TensorBoard
            if batch_idx % log_freq == 0:
                global_step = epoch * num_batches + batch_idx
                for key, value in loss_dict.items():
                    writer.add_scalar(f'Train_Batch/Loss_{key}', value.item(), global_step)
                
                # ✅ Log pos/neg losses
                if pos_losses:
                    writer.add_scalar('Train_Batch/Pos_Loss', np.mean(pos_losses[-10:]), global_step)
                if neg_losses:
                    writer.add_scalar('Train_Batch/Neg_Loss', np.mean(neg_losses[-10:]), global_step)
        
        except Exception as e:
            print(f"\n❌ Error in batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Average losses
    valid_batches = max(num_batches - nan_count, 1)
    avg_losses = {
        key: value / valid_batches
        for key, value in running_losses.items()
    }
    
    # Log to TensorBoard
    for key, value in avg_losses.items():
        writer.add_scalar(f'Train/Loss_{key}', value, epoch)
    
    # ✅ Log pos/neg statistics
    if pos_losses:
        writer.add_scalar('Train/Avg_Pos_Loss', np.mean(pos_losses), epoch)
    if neg_losses:
        writer.add_scalar('Train/Avg_Neg_Loss', np.mean(neg_losses), epoch)
    
    if nan_count > 0:
        print(f"\n⚠️  Epoch {epoch}: {nan_count} batches with NaN")
    
    return avg_losses


def validate_epoch(model, val_loader, criterion, device, epoch, writer):
    """Validate for one epoch."""
    model.eval()
    
    running_losses = defaultdict(float)
    num_batches = len(val_loader)
    
    all_predictions = defaultdict(list)
    all_labels = defaultdict(list)
    all_races = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                current_views = batch['current_views'].to(device)
                prior_views = batch['prior_views'].to(device)
                current_mask = batch['current_mask'].to(device)
                prior_mask = batch['prior_mask'].to(device)
                
                valid_samples = (current_mask.sum(dim=1) > 0) & (prior_mask.sum(dim=1) > 0)
                
                if not valid_samples.all():
                    current_views = current_views[valid_samples]
                    prior_views = prior_views[valid_samples]
                    current_mask = current_mask[valid_samples]
                    prior_mask = prior_mask[valid_samples]
                    
                    if len(current_views) == 0:
                        continue
                    
                    labels = {
                        'recall': batch['labels']['recall'][valid_samples.cpu()].to(device),
                        'birads': batch['labels']['birads'][valid_samples.cpu()].to(device),
                    }
                    race_labels = batch['metadata']['race'][valid_samples.cpu()].to(device)
                else:
                    labels = {
                        'recall': batch['labels']['recall'].to(device),
                        'birads': batch['labels']['birads'].to(device),
                    }
                    race_labels = batch['metadata']['race'].to(device)
                
                predictions = model(
                    current_views, prior_views,
                    current_mask, prior_mask,
                    race_labels
                )
                
                loss_dict = criterion(predictions, labels, race_labels)
                
                for key, value in loss_dict.items():
                    running_losses[key] += value.item()
                
                for task in ['recall', 'birads']:
                    all_predictions[task].append(predictions[task].cpu())
                    all_labels[task].append(labels[task].cpu())
                
                all_races.append(race_labels.cpu())
                
                pbar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}"
                })
            
            except Exception as e:
                print(f"\n❌ Error in validation batch {batch_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Average losses
    avg_losses = {
        key: value / max(num_batches, 1)
        for key, value in running_losses.items()
    }
    
    # Concatenate predictions
    if len(all_predictions['recall']) > 0:
        for task in ['recall', 'birads']:
            all_predictions[task] = torch.cat(all_predictions[task], dim=0)
            all_labels[task] = torch.cat(all_labels[task], dim=0)
        
        all_races = torch.cat(all_races, dim=0)
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_labels, all_races)
    else:
        print("⚠️  No valid predictions in validation!")
        metrics = defaultdict(dict)
    
    # Log to TensorBoard
    for key, value in avg_losses.items():
        writer.add_scalar(f'Val/Loss_{key}', value, epoch)
    
    for task in ['recall', 'birads']:
        if task in metrics:
            for metric_name, metric_value in metrics[task].items():
                writer.add_scalar(f'Val/{task}_{metric_name}', metric_value, epoch)
    
    if 'fairness' in metrics:
        for metric_name, metric_value in metrics['fairness'].items():
            writer.add_scalar(f'Val/Fairness_{metric_name}', metric_value, epoch)
    
    return avg_losses, metrics


# ============================================================================
# ✅ OPTIMIZED: Enhanced Metrics for Imbalanced Data
# ============================================================================

def compute_metrics(predictions, labels, races):
    """
    Compute comprehensive metrics for imbalanced data.
    
    Priorities for 9.9% positive:
    1. AUROC/AUPR (threshold-independent)
    2. Precision at high recall (clinically useful)
    3. Cost-sensitive metrics
    4. Per-race fairness
    """
    metrics = defaultdict(dict)
    
    recall_logits = predictions['recall']
    recall_labels = labels['recall'].squeeze().numpy().astype(int)
    recall_probs = torch.sigmoid(recall_logits).squeeze().numpy()
    
    valid_mask = (recall_labels >= 0)
    if valid_mask.sum() == 0:
        return metrics
    
    recall_probs = recall_probs[valid_mask]
    recall_labels = recall_labels[valid_mask]
    races_valid = races.numpy()[valid_mask]
    
    # ✅ Priority 1: AUROC and AUPR
    try:
        metrics['recall']['auroc'] = roc_auc_score(recall_labels, recall_probs)
    except:
        metrics['recall']['auroc'] = 0.0
    
    try:
        metrics['recall']['aupr'] = average_precision_score(recall_labels, recall_probs)
    except:
        metrics['recall']['aupr'] = 0.0
    
    # ✅ Priority 2: Optimal threshold and metrics
    try:
        precisions, recalls, thresholds = precision_recall_curve(recall_labels, recall_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        metrics['recall']['best_threshold'] = float(best_threshold)
        metrics['recall']['best_f1'] = float(best_f1)
        
        recall_pred_optimal = (recall_probs >= best_threshold).astype(int)
        metrics['recall']['f1_optimal'] = f1_score(recall_labels, recall_pred_optimal, zero_division=0)
        metrics['recall']['precision_optimal'] = precision_score(recall_labels, recall_pred_optimal, zero_division=0)
        metrics['recall']['recall_optimal'] = recall_score(recall_labels, recall_pred_optimal, zero_division=0)
        
        # ✅ Precision at different recall levels
        for target_recall in [0.8, 0.9, 0.95]:
            recall_mask = recalls >= target_recall
            if recall_mask.sum() > 0:
                best_precision = precisions[recall_mask].max()
                metrics['recall'][f'precision_at_recall{int(target_recall*100)}'] = float(best_precision)
    
    except:
        best_threshold = 0.5
    
    # Standard metrics at 0.5 threshold
    recall_pred = (recall_probs > 0.5).astype(int)
    metrics['recall']['accuracy'] = accuracy_score(recall_labels, recall_pred)
    metrics['recall']['balanced_accuracy'] = balanced_accuracy_score(recall_labels, recall_pred)
    metrics['recall']['f1'] = f1_score(recall_labels, recall_pred, zero_division=0)
    metrics['recall']['precision'] = precision_score(recall_labels, recall_pred, zero_division=0)
    metrics['recall']['tpr'] = recall_score(recall_labels, recall_pred, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(recall_labels, recall_pred, labels=[0, 1]).ravel()
        metrics['recall']['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['recall']['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['recall']['specificity'] = metrics['recall']['tnr']
        
        # ✅ Cost-sensitive metric (FN cost = 10x FP cost)
        cost = fp * 1.0 + fn * 10.0
        metrics['recall']['total_cost'] = float(cost)
        metrics['recall']['avg_cost_per_sample'] = float(cost / len(recall_labels))
    
    except:
        metrics['recall']['fpr'] = 0.0
        metrics['recall']['tnr'] = 0.0
    
    # ✅ Per-race fairness metrics
    metrics['fairness'] = compute_fairness_metrics(recall_pred, recall_labels, races_valid)
    
    return metrics


def compute_fairness_metrics(predictions, labels, races):
    """Compute detailed fairness metrics per race."""
    fairness_metrics = {}
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(races, torch.Tensor):
        races = races.cpu().numpy()
    
    race_0_mask = (races == 0)
    race_1_mask = (races == 1)
    
    if race_0_mask.sum() == 0 or race_1_mask.sum() == 0:
        return {'demographic_parity_diff': 0.0, 'accuracy_diff': 0.0}
    
    white_pos_rate = (predictions[race_0_mask] > 0).mean()
    black_pos_rate = (predictions[race_1_mask] > 0).mean()
    fairness_metrics['demographic_parity_diff'] = abs(white_pos_rate - black_pos_rate)
    
    # ✅ Per-race performance
    race_names = {0: 'white', 1: 'black', 2: 'asian', 3: 'other'}
    for race_id, race_name in race_names.items():
        race_mask = (races == race_id)
        if race_mask.sum() > 0:
            race_acc = (predictions[race_mask] == labels[race_mask]).mean()
            fairness_metrics[f'{race_name}_accuracy'] = race_acc
            
            # ✅ Per-race TPR and FPR
            race_labels = labels[race_mask]
            race_preds = predictions[race_mask]
            
            if race_labels.sum() > 0:  # Has positive samples
                race_tpr = recall_score(race_labels, race_preds, zero_division=0)
                fairness_metrics[f'{race_name}_tpr'] = race_tpr
            
            if (1 - race_labels).sum() > 0:  # Has negative samples
                try:
                    tn, fp, fn, tp = confusion_matrix(race_labels, race_preds, labels=[0, 1]).ravel()
                    race_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fairness_metrics[f'{race_name}_fpr'] = race_fpr
                except:
                    pass
    
    fairness_metrics['accuracy_diff'] = abs(
        fairness_metrics.get('white_accuracy', 0) - 
        fairness_metrics.get('black_accuracy', 0)
    )
    
    return fairness_metrics


def compute_pos_weight_from_csv(clinical_csv, device, max_pos_weight=None):
    """Compute pos_weight for BCE loss."""
    print("\n" + "="*70)
    print("Computing pos_weight for Binary Classification")
    print("="*70)
    
    try:
        df = pd.read_csv(clinical_csv, low_memory=False)
        labels = df['new_label'].values
        
        valid_mask = (labels >= 0) & (~pd.isna(labels))
        valid_labels = labels[valid_mask].astype(int)
        
        if len(valid_labels) == 0:
            return None
        
        binary_labels = np.where(valid_labels >= 1, 1, 0)
        counts = np.bincount(binary_labels)
        
        print(f"\nClass distribution:")
        print(f"  Class 0: {counts[0]:,} ({counts[0]/len(binary_labels)*100:.1f}%)")
        print(f"  Class 1: {counts[1]:,} ({counts[1]/len(binary_labels)*100:.1f}%)")
        
        pos_weight = float(counts[0]) / float(counts[1])
        print(f"\n✓ Raw pos_weight: {pos_weight:.3f}")
        
        if max_pos_weight is not None and pos_weight > max_pos_weight:
            print(f"  Clipping to {max_pos_weight:.3f}")
            pos_weight = max_pos_weight
        
        return torch.FloatTensor([pos_weight]).to(device)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def get_args():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--clinical_csv', type=str,
                        default="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_deduplicated.csv")
    parser.add_argument('--metadata_csv', type=str,
                        default="/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv")
    
    # Model
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_attention_queries', type=int, default=2)
    parser.add_argument('--use_gated_fusion', action='store_true', default=True)
    parser.add_argument('--task_hidden_dim', type=int, default=256)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)  # ✅ Increased from 16
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)  # ✅ Decreased from 5e-4
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=2)  # ✅ NEW: gradient accumulation
    
    # Loss (✅ Optimized for 9.9% imbalance)
    parser.add_argument('--use_focal_loss', action='store_true', default=True)
    parser.add_argument('--focal_alpha', type=float, default=0.75)  # ✅ Increased
    parser.add_argument('--focal_gamma', type=float, default=3.0)   # ✅ Increased
    parser.add_argument('--recall_weight', type=float, default=1.0)
    parser.add_argument('--birads_weight', type=float, default=0.1)
    parser.add_argument('--fairness_lambda', type=float, default=0.1,
                        help='Weight for fairness regularization')
# Data
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 256])
    parser.add_argument('--train_split', type=float, default=0.6)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--sample_fraction', type=float, default=0.1)

    # ✅ Sampling strategies
    parser.add_argument('--use_balanced_sampling', action='store_true', default=True)
    parser.add_argument('--use_balanced_batch', action='store_true', default=True)  # NEW
    parser.add_argument('--positive_ratio', type=float, default=0.3)  # NEW: 30% positive per batch
    parser.add_argument('--use_label_aware_aug', action='store_true', default=True)  # NEW

    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=5)  # ✅ NEW

    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--exp_name', type=str, default='robust_exam_side_optimized')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)  # ✅ Increased
    parser.add_argument('--resume', type=str, default="/projects/standard/lin01231/song0760/embed_recall_reduction/scripts/outputs/robust_exam_side_optimized_focal0.75g3.0_20260124_162530/checkpoint_epoch_5.pth")

    # Device
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log_freq', type=int, default=10)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name_full = f"{args.exp_name}_focal{args.focal_alpha:.2f}g{args.focal_gamma:.1f}_{timestamp}"
    exp_dir = os.path.join(args.output_dir, exp_name_full)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))

    # ✅ Data loaders with optimizations
    print("\nCreating data loaders with imbalance optimizations...")
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
        max_views_per_side=4,
        use_cache=True,
        use_balanced_sampling=args.use_balanced_sampling,
        use_balanced_batch=args.use_balanced_batch,
        positive_ratio=args.positive_ratio,
        use_label_aware_aug=args.use_label_aware_aug,
        verbose=True
    )

    print(f"✓ Train: {len(train_loader)} batches")
    print(f"✓ Val: {len(val_loader)} batches")
    print(f"✓ Test: {len(test_loader)} batches")

    # Model
    print("\nCreating robust model...")
    model = RobustExamSideModel(
        backbone=args.backbone,
        pretrained=args.pretrained,
        num_races=4,
        dropout=args.dropout,
        num_attention_queries=args.num_attention_queries,
        use_gated_fusion=args.use_gated_fusion,
        task_hidden_dim=args.task_hidden_dim
    ).to(device)

    model.print_trainable_status()

    # Loss
    pos_weight = None
    if not args.use_focal_loss:
        pos_weight = compute_pos_weight_from_csv(
            args.clinical_csv, device, None
        )

    criterion = MultiTaskLoss(
        recall_weight=args.recall_weight,
        birads_weight=args.birads_weight,
        pos_weight_recall=pos_weight,
        fairness_lambda=args.fairness_lambda,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )

    # Optimizer
    trainable_params = model.get_trainable_parameters()
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # ✅ Scheduler with warmup
    if args.scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
        
        # Warmup scheduler
        def warmup_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        print(f"✓ Using CosineAnnealing with {args.warmup_epochs} warmup epochs")
    elif args.scheduler == 'plateau':
        warmup_scheduler = None
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
    else:
        warmup_scheduler = None
        main_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # AMP
    scaler = GradScaler() if args.use_amp else None

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_auroc = 0.0
    patience_counter = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_auroc = checkpoint.get('best_val_auroc', 0.0)

    # Training loop
    print("\n" + "="*70)
    print("STARTING OPTIMIZED TRAINING FOR IMBALANCED DATA")
    print("="*70)
    print(f"  Focal Loss: α={args.focal_alpha}, γ={args.focal_gamma}")
    print(f"  Gradient Accumulation: {args.accumulation_steps} steps")
    print(f"  Label-aware Augmentation: {args.use_label_aware_aug}")
    print(f"  Balanced Batch: {args.use_balanced_batch} (pos ratio: {args.positive_ratio})")
    print("="*70 + "\n")

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 70)
        
        try:
            train_losses = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, writer,
                args.log_freq, args.use_amp, scaler, args.accumulation_steps
            )
            print(f"Train Loss: {train_losses['total']:.4f} | Recall: {train_losses['recall']:.4f}")
            
            val_losses, val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch, writer
            )
            print(f"Val Loss: {val_losses['total']:.4f} | Recall: {val_losses['recall']:.4f}")
            
            if 'recall' in val_metrics:
                print(f"Val Metrics:")
                print(f"  AUROC: {val_metrics['recall'].get('auroc', 0):.4f} | " +
                    f"AUPR: {val_metrics['recall'].get('aupr', 0):.4f}")
                print(f"  Best F1: {val_metrics['recall'].get('best_f1', 0):.4f} @ threshold={val_metrics['recall'].get('best_threshold', 0.5):.3f}")
                
                # ✅ Show precision at high recall
                for recall_level in [80, 90, 95]:
                    key = f'precision_at_recall{recall_level}'
                    if key in val_metrics['recall']:
                        print(f"  Precision @ Recall {recall_level}%: {val_metrics['recall'][key]:.4f}")
            
            # Scheduler
            if warmup_scheduler is not None and epoch < args.warmup_epochs:
                warmup_scheduler.step()
            elif args.scheduler == 'plateau':
                main_scheduler.step(val_losses['total'])
            else:
                main_scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
            print(f"LR: {current_lr:.2e}")
            
            # ✅ Save best based on AUROC (more important for imbalanced data)
            current_auroc = val_metrics['recall'].get('auroc', 0)
            is_best = current_auroc > best_val_auroc
            
            if is_best:
                print(f"✓ New best AUROC: {current_auroc:.4f} (previous: {best_val_auroc:.4f})")
                best_val_auroc = current_auroc
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_losses['total'],
                    'val_metrics': val_metrics,
                    'best_val_loss': best_val_loss,
                    'best_val_auroc': best_val_auroc,
                    'args': vars(args)
                }, os.path.join(exp_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            
            # Regular checkpoint
            if (epoch + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': main_scheduler.state_dict(),
                    'val_loss': val_losses['total'],
                    'val_metrics': val_metrics,
                    'best_val_loss': best_val_loss,
                    'best_val_auroc': best_val_auroc,
                    'args': vars(args)
                }, os.path.join(exp_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        except Exception as e:
            print(f"\n❌ Error in epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Test
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    best_checkpoint = torch.load(
        os.path.join(exp_dir, 'best_model.pth'),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_losses, test_metrics = validate_epoch(
        model, test_loader, criterion, device, args.num_epochs, writer
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_losses['total']:.4f}")
    if 'recall' in test_metrics:
        print(f"  AUROC: {test_metrics['recall'].get('auroc', 0):.4f}")
        print(f"  AUPR: {test_metrics['recall'].get('aupr', 0):.4f}")
        print(f"  Best F1: {test_metrics['recall'].get('best_f1', 0):.4f}")
        
        # ✅ Show cost metrics
        if 'total_cost' in test_metrics['recall']:
            print(f"  Total Cost: {test_metrics['recall']['total_cost']:.0f}")
            print(f"  Avg Cost/Sample: {test_metrics['recall']['avg_cost_per_sample']:.2f}")

    # ✅ Save detailed results
    with open(os.path.join(exp_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'test_losses': {k: float(v) for k, v in test_losses.items()},
            'test_metrics': {
                k: {mk: float(mv) for mk, mv in v.items()}
                for k, v in test_metrics.items()
            }
        }, f, indent=4)

    writer.close()
    print(f"\n✓ Training completed: {exp_dir}")