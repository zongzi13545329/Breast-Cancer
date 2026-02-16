"""
Training Script for EMBED Recall Reduction - NYU-Compatible
===========================================================
Key Features:
1. ✅ Uses NYU's pretrained ResNet-22 with flexible freezing
2. ✅ Breast-specific predictions (left/right separate)
3. ✅ Optimized for recall reduction task
4. ✅ Compatible with new dataset structure

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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from data.dataset import create_data_loaders
from models.recall_model import NYURecallModel

# ============================================================================
# ✅ Step 1: Import NYU's ResNet-22 Architecture (Unchanged)
# ============================================================================

class BasicBlockV2(nn.Module):
    """Basic residual block (from NYU code)."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class ViewResNetV2(nn.Module):
    """NYU's ResNet-22 architecture."""
    
    def __init__(
        self,
        input_channels=1,
        num_filters=16,
        first_layer_kernel_size=7,
        first_layer_conv_stride=2,
        blocks_per_layer_list=[2, 2, 2, 2, 2],
        block_strides_list=[1, 2, 2, 2, 2],
        first_layer_padding=0,
        first_pool_size=3,
        first_pool_stride=2,
        first_pool_padding=0,
        growth_factor=2
    ):
        super(ViewResNetV2, self).__init__()
        
        self.first_conv = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )
        
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )

        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters
        
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            self.layer_list.append(self._make_layer(
                block=BasicBlockV2,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        
        self.final_bn = nn.BatchNorm2d(
            current_num_filters // growth_factor * BasicBlockV2.expansion
        )
        self.relu = nn.ReLU()
        self.output_dim = current_num_filters // growth_factor * BasicBlockV2.expansion

    def forward(self, x):
        h = self.first_conv(x)
        h = self.first_pool(h)
        for layer in self.layer_list:
            h = layer(h)
        h = self.final_bn(h)
        h = self.relu(h)
        return h

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
        )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


# ============================================================================
# ✅ Step 2: NYU Feature Extractor with Flexible Freezing
# ============================================================================

class NYUFeatureExtractor(nn.Module):
    """NYU's feature extractor with flexible freezing options."""
    
    def __init__(self, input_channels=1, freeze_backbone=True, freeze_mode='partial'):
        """
        Args:
            freeze_mode: 
                - 'full': Freeze all layers
                - 'partial': Freeze first 3 layers, train last 2 layers
                - 'last_only': Freeze first 4 layers, train last 1 layer
                - 'none': Train all layers
        """
        super().__init__()
        
        self.resnet = ViewResNetV2(
            input_channels=input_channels,
            num_filters=16,
            first_layer_kernel_size=7,
            first_layer_conv_stride=2,
            blocks_per_layer_list=[2, 2, 2, 2, 2],
            block_strides_list=[1, 2, 2, 2, 2],
            first_layer_padding=0,
            first_pool_size=3,
            first_pool_stride=2,
            first_pool_padding=0,
            growth_factor=2
        )
        
        self.output_dim = 256
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.freeze_backbone = freeze_backbone
        self.freeze_mode = freeze_mode
        
        if freeze_backbone:
            self._apply_freezing(freeze_mode)
    
    def _apply_freezing(self, mode):
        """Apply different freezing strategies."""
        if mode == 'full':
            for param in self.resnet.parameters():
                param.requires_grad = False
            print("✓ NYU ResNet-22: FULLY FROZEN")
            
        elif mode == 'partial':
            for param in self.resnet.first_conv.parameters():
                param.requires_grad = False
            
            for i, layer in enumerate(self.resnet.layer_list):
                if i < 3:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            for param in self.resnet.final_bn.parameters():
                param.requires_grad = True
            
            trainable = sum(p.numel() for p in self.resnet.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.resnet.parameters())
            print(f"✓ NYU ResNet-22: PARTIALLY FROZEN")
            print(f"  - First 3 layers: FROZEN")
            print(f"  - Last 2 layers: TRAINABLE")
            print(f"  - Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
            
        elif mode == 'last_only':
            for param in self.resnet.first_conv.parameters():
                param.requires_grad = False
            
            for i, layer in enumerate(self.resnet.layer_list):
                if i < 4:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            for param in self.resnet.final_bn.parameters():
                param.requires_grad = True
            
            trainable = sum(p.numel() for p in self.resnet.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.resnet.parameters())
            print(f"✓ NYU ResNet-22: LAST LAYER ONLY TRAINABLE")
            print(f"  - First 4 layers: FROZEN")
            print(f"  - Last 1 layer: TRAINABLE")
            print(f"  - Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
            
        elif mode == 'none':
            for param in self.resnet.parameters():
                param.requires_grad = True
            print("✓ NYU ResNet-22: FULLY TRAINABLE")
        else:
            raise ValueError(f"Unknown freeze_mode: {mode}")
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W]
        Returns:
            features: [B, 256]
        """
        features = self.resnet(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        return features
    
    def load_nyu_weights(self, checkpoint_path, view_position='CC'):
        """Load weights from NYU's pretrained model."""
        print(f"\n{'='*70}")
        print(f"Loading NYU Pretrained Weights")
        print(f"{'='*70}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  View position: {view_position}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        view_key = view_position.lower()
        prefix = f'four_view_resnet.{view_key}.'
        
        print(f"\n  Looking for keys with prefix: '{prefix}'")
        
        filtered_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = 'resnet.' + k[len(prefix):]
                filtered_dict[new_key] = v
        
        if len(filtered_dict) == 0:
            print(f"  ⚠️  No keys found with prefix '{prefix}'")
            print(f"  Trying alternative: looking for 'resnet.' prefix...")
            
            for k, v in state_dict.items():
                if k.startswith('resnet.'):
                    filtered_dict[k] = v
            
            if len(filtered_dict) == 0:
                raise ValueError(
                    f"❌ No matching keys found!\n"
                    f"  Expected format: '{prefix}xxx' or 'resnet.xxx'\n"
                    f"  Found keys like: {list(state_dict.keys())[:5]}"
                )
        
        print(f"  ✓ Found {len(filtered_dict)} parameters")
        
        missing_keys, unexpected_keys = self.resnet.load_state_dict(
            filtered_dict, strict=False
        )
        
        print(f"\n  ✓ Loaded {len(filtered_dict)} parameters into ResNet")
        if missing_keys:
            print(f"  ⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
        
        print(f"{'='*70}\n")


# ============================================================================
# ✅ Step 3: Multi-View Aggregation (per breast)
# ============================================================================

class BreastViewAggregator(nn.Module):
    """Aggregate CC and MLO views for one breast using attention."""
    
    def __init__(self, feature_dim=256, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Attention over 2 views (CC, MLO)
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(feature_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, cc_feat, mlo_feat, cc_mask, mlo_mask):
        """
        Args:
            cc_feat: [B, 256] - CC view features
            mlo_feat: [B, 256] - MLO view features
            cc_mask: [B] - 1 if CC exists, 0 otherwise
            mlo_mask: [B] - 1 if MLO exists, 0 otherwise
        Returns:
            aggregated: [B, 256] - breast-level features
        """
        B = cc_feat.size(0)
        
        # Stack views: [B, 2, 256]
        views = torch.stack([cc_feat, mlo_feat], dim=1)
        
        # Create mask: [B, 2]
        mask = torch.stack([cc_mask, mlo_mask], dim=1)  # [B, 2]
        
        # Attention query (learnable)
        query = views.mean(dim=1, keepdim=True)  # [B, 1, 256]
        
        # Apply attention
        attended, _ = self.attention(
            query=query,
            key=views,
            value=views,
            key_padding_mask=(mask == 0)  # True for invalid positions
        )
        
        attended = self.norm(attended.squeeze(1))  # [B, 256]
        aggregated = self.fusion(attended)
        
        return aggregated


# ============================================================================
# ✅ Step 4: Temporal Fusion (prior vs current)
# ============================================================================

class TemporalFusion(nn.Module):
    """Gated fusion for temporal information."""
    
    def __init__(self, feature_dim=256, dropout=0.3):
        super().__init__()
        
        self.prior_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.current_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, prior_features, current_features):
        prior_processed = self.prior_proj(prior_features)
        current_processed = self.current_proj(current_features)
        
        concat = torch.cat([prior_processed, current_processed], dim=1)
        gate_weight = self.gate(concat)
        
        fused = gate_weight * prior_processed + (1 - gate_weight) * current_processed
        fused = self.fusion(fused)
        
        return fused


# ============================================================================
# ✅ Step 5: Task Head
# ============================================================================

class RecallPredictionHead(nn.Module):
    """Binary prediction head for recall decision."""
    
    def __init__(self, input_dim=256, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Binary output
        )
    
    def forward(self, x):
        return self.head(x)


# ============================================================================
# ✅ Step 6: Main Model (Breast-Specific)
# ============================================================================

class EMBEDRecallModel(nn.Module):
    """
    EMBED Recall Reduction Model with NYU Pretrained Weights.
    
    Architecture:
    1. NYU ResNet-22 feature extractors (CC and MLO specific)
    2. Per-breast view aggregation (left and right separately)
    3. Temporal fusion (prior vs current)
    4. Race conditioning
    5. Binary recall prediction per breast
    """
    
    def __init__(
        self,
        nyu_checkpoint_path=None,
        freeze_nyu_backbone=True,
        freeze_mode='partial',
        feature_dim=256,
        num_races=4,
        dropout=0.3,
        task_hidden_dim=256
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # ✅ View-specific feature extractors (CC and MLO)
        self.cc_extractor = NYUFeatureExtractor(
            input_channels=1,
            freeze_backbone=freeze_nyu_backbone,
            freeze_mode=freeze_mode
        )
        
        self.mlo_extractor = NYUFeatureExtractor(
            input_channels=1,
            freeze_backbone=freeze_nyu_backbone,
            freeze_mode=freeze_mode
        )
        
        # Load NYU pretrained weights
        if nyu_checkpoint_path and os.path.exists(nyu_checkpoint_path):
            self.cc_extractor.load_nyu_weights(nyu_checkpoint_path, view_position='CC')
            self.mlo_extractor.load_nyu_weights(nyu_checkpoint_path, view_position='MLO')
        
        # ✅ Per-breast aggregators
        self.left_aggregator = BreastViewAggregator(feature_dim, dropout)
        self.right_aggregator = BreastViewAggregator(feature_dim, dropout)
        
        # ✅ Temporal fusion (per breast)
        self.left_temporal = TemporalFusion(feature_dim, dropout)
        self.right_temporal = TemporalFusion(feature_dim, dropout)
        
        # ✅ Race embedding
        race_embed_dim = 64
        self.race_embeddings = nn.Embedding(num_races, race_embed_dim)
        nn.init.normal_(self.race_embeddings.weight, mean=0.0, std=0.02)
        
        # ✅ Race fusion (per breast)
        self.left_race_fusion = nn.Sequential(
            nn.Linear(feature_dim + race_embed_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.right_race_fusion = nn.Sequential(
            nn.Linear(feature_dim + race_embed_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ✅ Recall prediction heads (per breast)
        self.left_recall_head = RecallPredictionHead(feature_dim, task_hidden_dim, dropout)
        self.right_recall_head = RecallPredictionHead(feature_dim, task_hidden_dim, dropout)
    
    def extract_view_features(self, views_dict, mask_dict):
        """
        Extract features from all 4 views.
        
        Args:
            views_dict: {'L-CC': [B,1,H,W], 'L-MLO': [B,1,H,W], ...}
            mask_dict: {'L-CC': [B], 'L-MLO': [B], ...}
        
        Returns:
            features_dict: {'L-CC': [B,256], 'L-MLO': [B,256], ...}
        """
        features_dict = {}
        
        # ✅ CC views - 直接前向传播，不要手动控制梯度
        for view_key in ['L-CC', 'R-CC']:
            view_img = views_dict[view_key]
            features = self.cc_extractor(view_img)
            features_dict[view_key] = features
        
        # ✅ MLO views
        for view_key in ['L-MLO', 'R-MLO']:
            view_img = views_dict[view_key]
            features = self.mlo_extractor(view_img)
            features_dict[view_key] = features
        
        return features_dict
    
    def forward(self, current_views, prior_views, current_mask, prior_mask, race):
        """
        Args:
            current_views: dict with keys 'L-CC', 'L-MLO', 'R-CC', 'R-MLO', each [B,1,H,W]
            prior_views: same structure
            current_mask: dict with same keys, each [B]
            prior_mask: same structure
            race: [B] - race indices
        
        Returns:
            dict with:
                'left_recall': [B, 1] - left breast recall logits
                'right_recall': [B, 1] - right breast recall logits
        """
        # ✅ Extract features from all views
        current_feats = self.extract_view_features(current_views, current_mask)
        prior_feats = self.extract_view_features(prior_views, prior_mask)
        
        # ✅ Aggregate per breast (current)
        left_current = self.left_aggregator(
            current_feats['L-CC'],
            current_feats['L-MLO'],
            current_mask['L-CC'],
            current_mask['L-MLO']
        )
        
        right_current = self.right_aggregator(
            current_feats['R-CC'],
            current_feats['R-MLO'],
            current_mask['R-CC'],
            current_mask['R-MLO']
        )
        
        # ✅ Aggregate per breast (prior)
        left_prior = self.left_aggregator(
            prior_feats['L-CC'],
            prior_feats['L-MLO'],
            prior_mask['L-CC'],
            prior_mask['L-MLO']
        )
        
        right_prior = self.right_aggregator(
            prior_feats['R-CC'],
            prior_feats['R-MLO'],
            prior_mask['R-CC'],
            prior_mask['R-MLO']
        )
        
        # ✅ Temporal fusion
        left_temporal = self.left_temporal(left_prior, left_current)
        right_temporal = self.right_temporal(right_prior, right_current)
        
        # ✅ Race conditioning
        race_emb = self.race_embeddings(race)
        
        left_with_race = torch.cat([left_temporal, race_emb], dim=1)
        left_final = self.left_race_fusion(left_with_race)
        
        right_with_race = torch.cat([right_temporal, race_emb], dim=1)
        right_final = self.right_race_fusion(right_with_race)
        
        # ✅ Predictions
        left_recall = self.left_recall_head(left_final)
        right_recall = self.right_recall_head(right_final)
        
        return {
            'left_recall': left_recall,
            'right_recall': right_recall
        }
    
    def print_trainable_status(self):
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
# ✅ Step 7: Loss Function with Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B] - logits
            targets: [B] - binary labels
        """
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


class RecallReductionLoss(nn.Module):
    """
    Loss function for recall reduction task.
    
    Key principles:
    1. High penalty for False Negatives (missing actual recalls)
    2. Moderate penalty for False Positives (unnecessary recalls)
    3. Support Focal Loss for extreme imbalance
    """
    
    def __init__(
        self,
        use_focal=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        fn_weight=10.0,  # False negative weight
        fp_weight=1.0,   # False positive weight
        fairness_lambda=0.05
    ):
        super().__init__()
        
        self.use_focal = use_focal
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.fairness_lambda = fairness_lambda
        
        if use_focal:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.focal_loss = None
    
    def forward(self, predictions, labels, metadata):
        """
        Args:
            predictions: dict with 'left_recall', 'right_recall'
            labels: dict with 'left_malignant', 'right_malignant'
            metadata: dict with 'race', etc.
        
        Returns:
            dict with loss components
        """
        left_pred = predictions['left_recall'].squeeze(-1)
        right_pred = predictions['right_recall'].squeeze(-1)
        
        left_label = labels['left_malignant'].squeeze(-1)
        right_label = labels['right_malignant'].squeeze(-1)
        
        # ✅ Compute loss per breast
        if self.use_focal:
            left_loss = self.focal_loss(left_pred, left_label)
            right_loss = self.focal_loss(right_pred, right_label)
        else:
            # Weighted BCE
            left_weights = torch.where(
                left_label == 1,
                torch.tensor(self.fn_weight, device=left_label.device),
                torch.tensor(self.fp_weight, device=left_label.device)
            )
            
            right_weights = torch.where(
                right_label == 1,
                torch.tensor(self.fn_weight, device=right_label.device),
                torch.tensor(self.fp_weight, device=right_label.device)
            )
            
            left_loss_raw = F.binary_cross_entropy_with_logits(
                left_pred, left_label, reduction='none'
            )
            right_loss_raw = F.binary_cross_entropy_with_logits(
                right_pred, right_label, reduction='none'
            )
            
            left_loss = (left_loss_raw * left_weights).mean()
            right_loss = (right_loss_raw * right_weights).mean()
        
        # ✅ Total recall loss
        recall_loss = (left_loss + right_loss) / 2
        
        # ✅ Fairness constraint (optional)
        if self.fairness_lambda > 0 and 'race' in metadata:
            fairness_loss = self._compute_fairness(
                predictions, labels, metadata['race']
            )
            total_loss = recall_loss + self.fairness_lambda * fairness_loss
        else:
            fairness_loss = torch.tensor(0.0, device=recall_loss.device)
            total_loss = recall_loss
        
        return {
            'total': total_loss,
            'recall': recall_loss,
            'left': left_loss,
            'right': right_loss,
            'fairness': fairness_loss
        }
    
    def _compute_fairness(self, predictions, labels, race):
        """Compute fairness constraint across racial groups."""
        left_probs = torch.sigmoid(predictions['left_recall'].squeeze(-1))
        right_probs = torch.sigmoid(predictions['right_recall'].squeeze(-1))
        
        left_labels = labels['left_malignant'].squeeze(-1)
        right_labels = labels['right_malignant'].squeeze(-1)
        
        # Combine left and right
        all_probs = torch.cat([left_probs, right_probs])
        all_labels = torch.cat([left_labels, right_labels])
        race_expanded = race.repeat(2)
        
        positive_mask = (all_labels == 1)
        
        if positive_mask.sum() < 2:
            return torch.tensor(0.0, device=race.device)
        
        unique_races = race_expanded.unique()
        
        if len(unique_races) < 2:
            return torch.tensor(0.0, device=race.device)
        
        group_rates = []
        for r in unique_races:
            race_positive_mask = positive_mask & (race_expanded == r)
            if race_positive_mask.sum() > 0:
                group_rate = all_probs[race_positive_mask].mean()
                group_rates.append(group_rate)
        
        if len(group_rates) < 2:
            return torch.tensor(0.0, device=race.device)
        
        group_rates_tensor = torch.stack(group_rates)
        fairness_loss = group_rates_tensor.var()
        
        return fairness_loss


# ============================================================================
# ✅ Step 8: Training & Validation Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer,
                use_amp, scaler, accumulation_steps):
    """Training epoch with NaN detection."""
    model.train()  # ✅ 确保模型在训练模式
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
        
        # Forward pass (模型已经在 train mode)
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
                        labels={
                            'left_malignant': left_label,
                            'right_malignant': right_label
                        },
                        metadata={'race': race}
                    )
                    
                    loss = loss_dict['total'] / accumulation_steps
            else:
                predictions = model(
                    current_views=current_views,
                    prior_views=prior_views,
                    current_mask=current_mask,
                    prior_mask=prior_mask,
                    race=race
                )
                
                # ✅ Check predictions for NaN
                if torch.isnan(predictions['left_recall']).any() or torch.isnan(predictions['right_recall']).any():
                    print(f"\n⚠️  NaN in predictions at batch {batch_idx}")
                    nan_count += 1
                    optimizer.zero_grad()
                    continue
                
                loss_dict = criterion(
                    predictions=predictions,
                    labels={
                        'left_malignant': left_label,
                        'right_malignant': right_label
                    },
                    metadata={'race': race}
                )
                
                # ✅ Check loss for NaN
                if torch.isnan(loss_dict['total']):
                    print(f"\n⚠️  NaN in loss at batch {batch_idx}")
                    nan_count += 1
                    optimizer.zero_grad()
                    continue
                
                loss = loss_dict['total'] / accumulation_steps
            
            # Backward (模型保持在 train mode)
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate losses
            for k, v in loss_dict.items():
                losses[k] += v.item()
            
            valid_batches += 1
            
            # Update
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    # ✅ Check gradients for NaN before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\n⚠️  Invalid gradient norm at batch {batch_idx}: {grad_norm}")
                        optimizer.zero_grad()
                        nan_count += 1
                        continue
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\n⚠️  Invalid gradient norm at batch {batch_idx}: {grad_norm}")
                        optimizer.zero_grad()
                        nan_count += 1
                        continue
                    
                    optimizer.step()
                
                optimizer.zero_grad()
            
            pbar.set_postfix({
                'loss': loss.item() * accumulation_steps,
                'nan_batches': nan_count
            })
        
        except Exception as e:
            print(f"\n❌ Error in batch {batch_idx}: {e}")
            nan_count += 1
            optimizer.zero_grad()
            continue
    
    # Average losses
    if valid_batches > 0:
        for k in losses:
            losses[k] /= valid_batches
    else:
        print("\n❌ ERROR: No valid batches in training epoch!")
        for k in losses:
            losses[k] = float('inf')
    
    if nan_count > 0:
        print(f"\n⚠️  Warning: Encountered {nan_count} batches with NaN/errors during training")
    
    return dict(losses)

def validate_epoch(model, val_loader, criterion, device, epoch, writer):
    """Validation epoch with correct metrics for recall reduction."""
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
                labels={
                    'left_malignant': left_label,
                    'right_malignant': right_label
                },
                metadata={'race': race}
            )
            
            for k, v in loss_dict.items():
                losses[k] += v.item()
            
            # Collect predictions
            all_left_preds.append(torch.sigmoid(predictions['left_recall']).cpu())
            all_left_labels.append(left_label.cpu())
            all_right_preds.append(torch.sigmoid(predictions['right_recall']).cpu())
            all_right_labels.append(right_label.cpu())
            all_race.append(race.cpu())
    
    # Average losses
    for k in losses:
        losses[k] /= len(val_loader)
    
    # Compute metrics
    left_preds = torch.cat(all_left_preds).squeeze().numpy()
    left_labels = torch.cat(all_left_labels).squeeze().numpy()
    right_preds = torch.cat(all_right_preds).squeeze().numpy()
    right_labels = torch.cat(all_right_labels).squeeze().numpy()
    race = torch.cat(all_race).numpy()
    
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    
    metrics = {}
    
    print(f"\n{'='*80}")
    print(f"VALIDATION METRICS - EPOCH {epoch+1}")
    print(f"{'='*80}")
    
    # ✅ Per-breast metrics
    for side, preds, labels in [
        ('Left', left_preds, left_labels),
        ('Right', right_preds, right_labels)
    ]:
        print(f"\n{'='*80}")
        print(f"{side.upper()} BREAST PERFORMANCE")
        print(f"{'='*80}")
        
        if len(np.unique(labels)) > 1:
            auroc = roc_auc_score(labels, preds)
            metrics[f'{side.lower()}_auroc'] = auroc
            
            print(f"\n🎯 Discrimination:")
            print(f"  AUROC: {auroc:.4f}")
            
            # ✅ KEY METRIC: Specificity at high sensitivity
            fpr, tpr, thresholds = roc_curve(labels, preds)
            
            for target_sens in [0.90, 0.95, 0.98, 0.99]:
                idx = np.where(tpr >= target_sens)[0]
                if len(idx) > 0:
                    best_idx = idx[np.argmin(fpr[idx])]
                    spec = 1 - fpr[best_idx]
                    threshold = thresholds[best_idx]
                    
                    metrics[f'{side.lower()}_spec_at_{int(target_sens*100)}_sens'] = spec
                    
                    print(f"\n  At {target_sens*100:.0f}% Sensitivity:")
                    print(f"    Specificity: {spec:.4f}")
                    print(f"    → Can avoid {spec*100:.1f}% of unnecessary recalls")
                    print(f"    Threshold: {threshold:.4f}")
                    
                    if target_sens == 0.95:
                        metrics[f'{side.lower()}_primary'] = spec
    
    # ✅ Overall exam-level metrics
    print(f"\n{'='*80}")
    print("EXAM-LEVEL PERFORMANCE")
    print(f"{'='*80}")
    
    # Any side needs recall
    exam_preds = np.maximum(left_preds, right_preds)
    exam_labels = np.maximum(left_labels, right_labels)
    
    if len(np.unique(exam_labels)) > 1:
        exam_auroc = roc_auc_score(exam_labels, exam_preds)
        metrics['exam_auroc'] = exam_auroc
        
        print(f"\n🎯 Exam-level AUROC: {exam_auroc:.4f}")
        
        fpr, tpr, thresholds = roc_curve(exam_labels, exam_preds)
        
        for target_sens in [0.95]:
            idx = np.where(tpr >= target_sens)[0]
            if len(idx) > 0:
                best_idx = idx[np.argmin(fpr[idx])]
                spec = 1 - fpr[best_idx]
                
                print(f"\n  At {target_sens*100:.0f}% Sensitivity:")
                print(f"    Specificity: {spec:.4f}")
                print(f"    → Exam-level recall reduction: {spec*100:.1f}%")
                
                metrics['exam_primary'] = spec
    
    print(f"\n{'='*80}\n")
    
    return dict(losses), metrics


# ============================================================================
# ✅ Step 9: Main Training Script
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser()
    
    # NYU weights
    parser.add_argument('--nyu_checkpoint', type=str,
                        default="/projects/standard/lin01231/song0760/embed_recall_reduction/pretrained/sample_image_model.p")
    parser.add_argument('--freeze_mode', type=str, default='partial',
                        choices=['full', 'partial', 'last_only', 'none'])
    
    # Data
    parser.add_argument('--clinical_csv', type=str,
                        default="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled.csv")
    parser.add_argument('--metadata_csv', type=str,
                        default="/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv")
    
    # Model
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--use_prior', action='store_true', default=True)
    parser.add_argument('--task_hidden_dim', type=int, default=256)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    
    # Loss
    parser.add_argument('--use_focal', action='store_true', default=True)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--fn_weight', type=float, default=10.0)
    parser.add_argument('--fp_weight', type=float, default=1.0)
    parser.add_argument('--fairness_lambda', type=float, default=0.05)
    
    # Data splits
    parser.add_argument('--image_size', type=int, nargs=2, default=[2944, 1920])
    parser.add_argument('--apply_nyu_preprocessing', action='store_true', default=True)
    parser.add_argument('--train_split', type=float, default=0.6)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--sample_fraction', type=float, default=1)
    
    # Sampling
    parser.add_argument('--use_balanced_batch', action='store_true', default=True)
    parser.add_argument('--positive_ratio', type=float, default=0.3)
    parser.add_argument('--use_cache', action='store_true', default=True)
    
    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--exp_name', type=str, default='embed_recall_nyu')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    
    # ✅ Resume training
    parser.add_argument('--resume', type=str, default="/projects/standard/lin01231/song0760/embed_recall_reduction/scripts/outputs/embed_recall_nyu_partial_20260201_180539/best_model.pth",
                        help='Path to checkpoint to resume from (e.g., outputs/xxx/best_model.pth)')
    parser.add_argument('--resume_optimizer', action='store_true', default=True,
                        help='Resume optimizer state (default: True)')
    parser.add_argument('--resume_scheduler', action='store_true', default=True,
                        help='Resume scheduler state (default: True)')
    
    # Device
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    return args


# ============================================================================
# ✅ New: Load Checkpoint Function
# ============================================================================

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, 
                   resume_optimizer=True, resume_scheduler=True, device='cpu'):
    """
    Load checkpoint and resume training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to restore state (optional)
        scheduler: Scheduler to restore state (optional)
        resume_optimizer: Whether to resume optimizer state
        resume_scheduler: Whether to resume scheduler state
        device: Device to load checkpoint to
    
    Returns:
        dict with:
            - start_epoch: Epoch to resume from
            - best_metric: Best metric so far
            - args: Original training arguments
    """
    print(f"\n{'='*70}")
    print("RESUMING FROM CHECKPOINT")
    print(f"{'='*70}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded model weights")
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_metric = checkpoint.get('metric', 0.0)
    
    print(f"  ✓ Resume from epoch: {start_epoch}")
    print(f"  ✓ Best metric so far: {best_metric:.4f}")
    
    # Load optimizer state
    if optimizer is not None and resume_optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  ✓ Loaded optimizer state")
        except Exception as e:
            print(f"  ⚠️  Failed to load optimizer state: {e}")
            print(f"  → Will use fresh optimizer state")
    
    # Load scheduler state
    if scheduler is not None and resume_scheduler and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  ✓ Loaded scheduler state")
        except Exception as e:
            print(f"  ⚠️  Failed to load scheduler state: {e}")
            print(f"  → Will use fresh scheduler state")
    
    # Load scaler state (for AMP)
    scaler_state = None
    if 'scaler_state_dict' in checkpoint:
        scaler_state = checkpoint['scaler_state_dict']
        print(f"  ✓ Found AMP scaler state")
    
    print(f"{'='*70}\n")
    
    return {
        'start_epoch': start_epoch,
        'best_metric': best_metric,
        'args': checkpoint.get('args', {}),
        'scaler_state': scaler_state
    }


if __name__ == "__main__":
    args = get_args()
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ✅ Handle resume: use existing exp_dir or create new one
    if args.resume:
        # Extract experiment directory from checkpoint path
        checkpoint_path = args.resume
        if 'best_model.pth' in checkpoint_path or 'checkpoint_epoch' in checkpoint_path:
            exp_dir = os.path.dirname(checkpoint_path)
            exp_name_full = os.path.basename(exp_dir)
        else:
            raise ValueError(
                f"Invalid checkpoint path format: {checkpoint_path}\n"
                f"Expected: .../exp_name_timestamp/best_model.pth or checkpoint_epochX.pth"
            )
        print(f"\n✅ RESUME MODE")
        print(f"  Using existing experiment: {exp_dir}")
    else:
        # Create new experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name_full = f"{args.exp_name}_{args.freeze_mode}_{timestamp}"
        exp_dir = os.path.join(args.output_dir, exp_name_full)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\n✅ NEW TRAINING")
        print(f"  Experiment directory: {exp_dir}")
    
    # Save config
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path) or not args.resume:
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    
    # ✅ Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    train_loader, val_loader, test_loader, datasets = create_data_loaders(args)
    
    # ✅ Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    model = NYURecallModel(
        input_channels=1,
        nyu_checkpoint_path=args.nyu_checkpoint if not args.resume else None,  # Don't reload NYU weights if resuming
        freeze_mode=args.freeze_mode,
        num_races=4,
        dropout=args.dropout,
        use_prior=args.use_prior
    ).to(device)
    
    model.print_trainable_status()
    
    # ✅ Loss, Optimizer, Scheduler
    criterion = RecallReductionLoss(
        use_focal=args.use_focal,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        fn_weight=args.fn_weight,
        fp_weight=args.fp_weight,
        fairness_lambda=args.fairness_lambda
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
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
        
        print(f"\n✅ Scheduler: Cosine with {args.warmup_epochs} warmup epochs")
    else:
        warmup_scheduler = None
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        print(f"\n✅ Scheduler: ReduceLROnPlateau")
    
    scaler = GradScaler() if args.use_amp else None
    
    # ✅ Load checkpoint if resuming
    start_epoch = 0
    best_metric = 0.0
    
    if args.resume:
        resume_info = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer if args.resume_optimizer else None,
            scheduler=main_scheduler if args.resume_scheduler else None,
            resume_optimizer=args.resume_optimizer,
            resume_scheduler=args.resume_scheduler,
            device=device
        )
        
        start_epoch = resume_info['start_epoch']
        best_metric = resume_info['best_metric']
        
        # Restore scaler state if using AMP
        if scaler is not None and resume_info['scaler_state'] is not None:
            scaler.load_state_dict(resume_info['scaler_state'])
            print(f"  ✓ Restored AMP scaler state")
    
    # ✅ Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    if args.resume:
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best metric so far: {best_metric:.4f}")
    print("="*70)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{args.num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            args.use_amp, scaler, args.accumulation_steps
        )
        
        # Validate
        val_losses, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Summary
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1} SUMMARY")
        print(f"{'='*80}")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val Loss: {val_losses['total']:.4f}")
        print(f"  Exam Primary Metric: {val_metrics.get('exam_primary', 0):.4f}")
        
        # Log
        writer.add_scalar('Loss/train', train_losses['total'], epoch)
        writer.add_scalar('Loss/val', val_losses['total'], epoch)
        writer.add_scalar('Metrics/exam_primary', val_metrics.get('exam_primary', 0), epoch)
        
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
            
            # ✅ Save complete checkpoint with all states
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': current_metric,
                'args': vars(args)
            }
            
            # Save scheduler state
            if main_scheduler is not None:
                checkpoint['scheduler_state_dict'] = main_scheduler.state_dict()
            
            # Save scaler state if using AMP
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, os.path.join(exp_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n{'='*80}")
            print("EARLY STOPPING")
            print(f"{'='*80}\n")
            break
        
        # Periodic save
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)
            }
            
            if main_scheduler is not None:
                checkpoint['scheduler_state_dict'] = main_scheduler.state_dict()
            
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, os.path.join(exp_dir, f'checkpoint_epoch{epoch+1}.pth'))
    
    writer.close()
    
    # Cleanup
    if 'temp_dir' in datasets:
        shutil.rmtree(datasets['temp_dir'])
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"  Best metric: {best_metric:.4f}")
    print(f"  Experiment: {exp_dir}")
    print(f"{'='*80}\n")