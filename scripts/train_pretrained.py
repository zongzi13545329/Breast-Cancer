"""
Training Script for EMBED Recall Reduction - NYU-Compatible
===========================================================
Key Features:
1. ✅ Uses NYU's pretrained ResNet-22 with flexible freezing
2. ✅ Breast-specific predictions (left/right separate)
3. ✅ Optimized for recall reduction task
4. ✅ Compatible with new dataset structure
5. ✅ BIRADSAwareLoss (replaces RecallReductionLoss)
6. ✅ MC Dropout uncertainty estimation at inference
7. ✅ Uncertainty-aware evaluation pipeline

Author: Yiran
Date: 2025
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
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
            print(f"  - Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
        elif mode == 'none':
            for param in self.resnet.parameters():
                param.requires_grad = True
            print("✓ NYU ResNet-22: FULLY TRAINABLE")
        else:
            raise ValueError(f"Unknown freeze_mode: {mode}")
    
    def forward(self, x):
        features = self.resnet(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        return features
    
    def load_nyu_weights(self, checkpoint_path, view_position='CC'):
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
        
        filtered_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = 'resnet.' + k[len(prefix):]
                filtered_dict[new_key] = v
        
        if len(filtered_dict) == 0:
            for k, v in state_dict.items():
                if k.startswith('resnet.'):
                    filtered_dict[k] = v
            if len(filtered_dict) == 0:
                raise ValueError(
                    f"❌ No matching keys found!\n"
                    f"  Found keys like: {list(state_dict.keys())[:5]}"
                )
        
        print(f"  ✓ Found {len(filtered_dict)} parameters")
        
        missing_keys, unexpected_keys = self.resnet.load_state_dict(
            filtered_dict, strict=False
        )
        
        print(f"  ✓ Loaded {len(filtered_dict)} parameters into ResNet")
        if missing_keys:
            print(f"  ⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
        print(f"{'='*70}\n")


# ============================================================================
# ✅ Step 3: Multi-View Aggregation (per breast)
# ============================================================================

class BreastViewAggregator(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, cc_feat, mlo_feat, cc_mask, mlo_mask):
        B = cc_feat.size(0)
        views = torch.stack([cc_feat, mlo_feat], dim=1)
        mask = torch.stack([cc_mask, mlo_mask], dim=1)
        query = views.mean(dim=1, keepdim=True)
        attended, _ = self.attention(
            query=query, key=views, value=views,
            key_padding_mask=(mask == 0)
        )
        attended = self.norm(attended.squeeze(1))
        aggregated = self.fusion(attended)
        return aggregated


# ============================================================================
# ✅ Step 4: Temporal Fusion (prior vs current)
# ============================================================================

class TemporalFusion(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.3):
        super().__init__()
        self.prior_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.current_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim),
            nn.ReLU(), nn.Dropout(dropout)
        )
    
    def forward(self, prior_features, current_features):
        prior_processed = self.prior_proj(prior_features)
        current_processed = self.current_proj(current_features)
        concat = torch.cat([prior_processed, current_processed], dim=1)
        gate_weight = self.gate(concat)
        fused = gate_weight * prior_processed + (1 - gate_weight) * current_processed
        return self.fusion(fused)


# ============================================================================
# ✅ Step 5: Task Head
# ============================================================================

class RecallPredictionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.head(x)


# ============================================================================
# ✅ Step 6: Main Model (Breast-Specific)
# ============================================================================

class EMBEDRecallModel(nn.Module):
    def __init__(
        self, nyu_checkpoint_path=None, freeze_nyu_backbone=True,
        freeze_mode='partial', feature_dim=256, num_races=4,
        dropout=0.3, task_hidden_dim=256
    ):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.cc_extractor = NYUFeatureExtractor(1, freeze_nyu_backbone, freeze_mode)
        self.mlo_extractor = NYUFeatureExtractor(1, freeze_nyu_backbone, freeze_mode)
        
        if nyu_checkpoint_path and os.path.exists(nyu_checkpoint_path):
            self.cc_extractor.load_nyu_weights(nyu_checkpoint_path, view_position='CC')
            self.mlo_extractor.load_nyu_weights(nyu_checkpoint_path, view_position='MLO')
        
        self.left_aggregator = BreastViewAggregator(feature_dim, dropout)
        self.right_aggregator = BreastViewAggregator(feature_dim, dropout)
        self.left_temporal = TemporalFusion(feature_dim, dropout)
        self.right_temporal = TemporalFusion(feature_dim, dropout)
        
        race_embed_dim = 64
        self.race_embeddings = nn.Embedding(num_races, race_embed_dim)
        nn.init.normal_(self.race_embeddings.weight, mean=0.0, std=0.02)
        
        self.left_race_fusion = nn.Sequential(
            nn.Linear(feature_dim + race_embed_dim, feature_dim),
            nn.LayerNorm(feature_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.right_race_fusion = nn.Sequential(
            nn.Linear(feature_dim + race_embed_dim, feature_dim),
            nn.LayerNorm(feature_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        
        self.left_recall_head = RecallPredictionHead(feature_dim, task_hidden_dim, dropout)
        self.right_recall_head = RecallPredictionHead(feature_dim, task_hidden_dim, dropout)
    
    def extract_view_features(self, views_dict, mask_dict):
        features_dict = {}
        for view_key in ['L-CC', 'R-CC']:
            features_dict[view_key] = self.cc_extractor(views_dict[view_key])
        for view_key in ['L-MLO', 'R-MLO']:
            features_dict[view_key] = self.mlo_extractor(views_dict[view_key])
        return features_dict
    
    def forward(self, current_views, prior_views, current_mask, prior_mask, race):
        current_feats = self.extract_view_features(current_views, current_mask)
        prior_feats = self.extract_view_features(prior_views, prior_mask)
        
        left_current = self.left_aggregator(
            current_feats['L-CC'], current_feats['L-MLO'],
            current_mask['L-CC'], current_mask['L-MLO']
        )
        right_current = self.right_aggregator(
            current_feats['R-CC'], current_feats['R-MLO'],
            current_mask['R-CC'], current_mask['R-MLO']
        )
        left_prior = self.left_aggregator(
            prior_feats['L-CC'], prior_feats['L-MLO'],
            prior_mask['L-CC'], prior_mask['L-MLO']
        )
        right_prior = self.right_aggregator(
            prior_feats['R-CC'], prior_feats['R-MLO'],
            prior_mask['R-CC'], prior_mask['R-MLO']
        )
        
        left_temporal = self.left_temporal(left_prior, left_current)
        right_temporal = self.right_temporal(right_prior, right_current)
        
        race_emb = self.race_embeddings(race)
        left_final = self.left_race_fusion(torch.cat([left_temporal, race_emb], dim=1))
        right_final = self.right_race_fusion(torch.cat([right_temporal, race_emb], dim=1))
        
        return {
            'left_recall': self.left_recall_head(left_final),
            'right_recall': self.right_recall_head(right_final)
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
# ✅ Step 7: BIRADSAwareLoss (REPLACES RecallReductionLoss)
# ============================================================================

class BIRADSAwareLoss(nn.Module):
    """
    BIRADS-aware loss that uses radiologist assessment as prior knowledge.
    
    Loss = Focal_Loss × BIRADS_Weight × Asymmetric_Weight + Confidence_Penalty + Fairness
    
    BIRADS Encoding (from dataset):
        0: Assessment incomplete (A/X)
        1: Negative (N)
        2: Benign (B)
        3: Probably benign (P) - UNCERTAIN, special handling
        4: Suspicious/Malignant (S/M/K) - must not miss
    """
    
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        birads_weights: Dict[int, float] = None,
        fn_cost_ratio: float = 5.0,
        uncertain_birads_penalty: float = 0.1,
        fairness_lambda: float = 0.05,
    ):
        super().__init__()
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.fn_cost_ratio = fn_cost_ratio
        self.uncertain_birads_penalty = uncertain_birads_penalty
        self.fairness_lambda = fairness_lambda
        
        if birads_weights is None:
            self.birads_weights = {
                0: 1.0,   # Incomplete
                1: 0.8,   # Negative
                2: 0.8,   # Benign
                3: 1.5,   # Probably benign → UNCERTAIN
                4: 3.0,   # Suspicious/Malignant → MUST NOT MISS
            }
        else:
            self.birads_weights = birads_weights
        
        print(f"\n✓ BIRADSAwareLoss initialized:")
        print(f"  Focal: α={focal_alpha}, γ={focal_gamma}")
        print(f"  FN cost ratio: {fn_cost_ratio}x")
        print(f"  BIRADS weights: {self.birads_weights}")
        print(f"  Uncertain BIRADS penalty: {uncertain_birads_penalty}")
        print(f"  Fairness lambda: {fairness_lambda}")
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        metadata: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: {'left_recall': [B,1], 'right_recall': [B,1]}
            labels: {'left_malignant': [B], 'right_malignant': [B]}
            metadata: {'race': [B], 'birads': [B] (optional)}
        """
        left_pred = predictions['left_recall'].squeeze(-1)
        right_pred = predictions['right_recall'].squeeze(-1)
        left_label = labels['left_malignant'].squeeze(-1)
        right_label = labels['right_malignant'].squeeze(-1)
        
        device = left_pred.device
        B = left_pred.size(0)
        
        birads = metadata.get('birads', None)
        
        # Component 1: Focal Loss (base)
        left_focal = self._focal_loss(left_pred, left_label)
        right_focal = self._focal_loss(right_pred, right_label)
        
        # Component 2: BIRADS-based sample weighting
        if birads is not None:
            birads_weight = torch.ones(B, device=device)
            for birads_val, weight in self.birads_weights.items():
                mask = (birads == birads_val)
                birads_weight[mask] = weight
        else:
            birads_weight = torch.ones(B, device=device)
        
        # Component 3: Asymmetric FN/FP weighting
        left_asym = self._asymmetric_weight(left_pred, left_label)
        right_asym = self._asymmetric_weight(right_pred, right_label)
        
        # Component 4: Confidence calibration penalty on uncertain cases
        if birads is not None and self.uncertain_birads_penalty > 0:
            left_conf_penalty = self._confidence_penalty(left_pred, birads, uncertain_birads=[3])
            right_conf_penalty = self._confidence_penalty(right_pred, birads, uncertain_birads=[3])
        else:
            left_conf_penalty = torch.tensor(0.0, device=device)
            right_conf_penalty = torch.tensor(0.0, device=device)
        
        # Combine: weighted loss per sample
        left_loss = (left_focal * birads_weight * left_asym).mean()
        right_loss = (right_focal * birads_weight * right_asym).mean()
        
        recall_loss = (left_loss + right_loss) / 2.0
        conf_penalty = (left_conf_penalty + right_conf_penalty) / 2.0
        
        # Component 5: Fairness constraint
        if self.fairness_lambda > 0 and 'race' in metadata:
            fairness_loss = self._compute_fairness(predictions, labels, metadata['race'])
        else:
            fairness_loss = torch.tensor(0.0, device=device)
        
        total_loss = (
            recall_loss 
            + self.uncertain_birads_penalty * conf_penalty
            + self.fairness_lambda * fairness_loss
        )
        
        return {
            'total': total_loss,
            'recall': recall_loss,
            'left': left_loss,
            'right': right_loss,
            'confidence_penalty': conf_penalty,
            'fairness': fairness_loss,
        }
    
    def _focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        probs = torch.sigmoid(pred)
        p_t = probs * target + (1 - probs) * (1 - target)
        focal_weight = (1 - p_t) ** self.focal_gamma
        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
            focal_weight = alpha_t * focal_weight
        return focal_weight * bce
    
    def _asymmetric_weight(self, pred, target):
        return torch.where(
            target == 1,
            torch.tensor(self.fn_cost_ratio, device=pred.device),
            torch.tensor(1.0, device=pred.device)
        )
    
    def _confidence_penalty(self, pred, birads, uncertain_birads=[3]):
        mask = torch.zeros_like(pred, dtype=torch.bool)
        for b in uncertain_birads:
            mask = mask | (birads == b)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        probs = torch.sigmoid(pred[mask])
        return ((probs - 0.5) ** 2).mean()
    
    def _compute_fairness(self, predictions, labels, race):
        left_probs = torch.sigmoid(predictions['left_recall'].squeeze(-1))
        right_probs = torch.sigmoid(predictions['right_recall'].squeeze(-1))
        left_labels = labels['left_malignant'].squeeze(-1)
        right_labels = labels['right_malignant'].squeeze(-1)
        
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
            race_pos_mask = positive_mask & (race_expanded == r)
            if race_pos_mask.sum() > 0:
                group_rates.append(all_probs[race_pos_mask].mean())
        
        if len(group_rates) < 2:
            return torch.tensor(0.0, device=race.device)
        
        return torch.stack(group_rates).var()


# ============================================================================
# ✅ Step 7b: MC Dropout Wrapper (for uncertainty estimation at inference)
# ============================================================================

class MCDropoutWrapper(nn.Module):
    """
    Model-agnostic MC Dropout wrapper for uncertainty estimation.
    
    During training: standard forward pass (same as base model).
    During inference: T forward passes with dropout ON → mean + variance.
    """
    
    def __init__(self, base_model: nn.Module, n_samples: int = 20):
        super().__init__()
        self.base_model = base_model
        self.n_samples = n_samples
    
    def forward(self, current_views, prior_views, current_mask, prior_mask, race=None):
        """Standard forward pass (used during training)."""
        return self.base_model(current_views, prior_views, current_mask, prior_mask, race)
    
    def print_trainable_status(self):
        if hasattr(self.base_model, 'print_trainable_status'):
            self.base_model.print_trainable_status()
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self, current_views, prior_views, current_mask, prior_mask, race=None,
        n_samples: int = None,
    ) -> Dict[str, torch.Tensor]:
        """MC Dropout inference: T forward passes with dropout enabled."""
        T = n_samples or self.n_samples
        self._enable_dropout()
        
        left_samples, right_samples = [], []
        
        for _ in range(T):
            output = self.base_model(
                current_views, prior_views, current_mask, prior_mask, race
            )
            left_samples.append(torch.sigmoid(output['left_recall'].squeeze(-1)))
            right_samples.append(torch.sigmoid(output['right_recall'].squeeze(-1)))
        
        self.base_model.eval()
        
        left_samples = torch.stack(left_samples, dim=1)   # [B, T]
        right_samples = torch.stack(right_samples, dim=1)  # [B, T]
        
        left_mean = left_samples.mean(dim=1)
        left_var = left_samples.var(dim=1)
        right_mean = right_samples.mean(dim=1)
        right_var = right_samples.var(dim=1)
        
        left_entropy = self._binary_entropy(left_mean)
        right_entropy = self._binary_entropy(right_mean)
        
        return {
            'left_recall': left_mean.unsqueeze(-1),
            'right_recall': right_mean.unsqueeze(-1),
            'left_recall_mean': left_mean,
            'left_recall_var': left_var,
            'left_recall_entropy': left_entropy,
            'right_recall_mean': right_mean,
            'right_recall_var': right_var,
            'right_recall_entropy': right_entropy,
            'left_recall_samples': left_samples,
            'right_recall_samples': right_samples,
        }
    
    def _enable_dropout(self):
        self.base_model.eval()
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @staticmethod
    def _binary_entropy(p, eps=1e-8):
        p = p.clamp(eps, 1 - eps)
        return -p * torch.log(p) - (1 - p) * torch.log(1 - p)


# ============================================================================
# ✅ Step 7c: Uncertainty-Aware Decision Framework
# ============================================================================

class UncertaintyAwareDecision:
    """
    Three-zone decision:
        Zone 1: pred < thresh AND low uncertainty  → SAFE NO-RECALL
        Zone 2: pred < thresh BUT high uncertainty → RECALL (conservative)
        Zone 3: pred >= thresh                     → RECALL
    """
    
    def __init__(self, recall_threshold=0.5, uncertainty_threshold=0.1,
                 use_entropy=False, entropy_threshold=0.5):
        self.recall_threshold = recall_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.use_entropy = use_entropy
        self.entropy_threshold = entropy_threshold
    
    def make_decision(self, predictions):
        results = {}
        for side in ['left', 'right']:
            mean_pred = predictions[f'{side}_recall_mean']
            variance = predictions[f'{side}_recall_var']
            
            if self.use_entropy:
                uncertainty = predictions[f'{side}_recall_entropy']
                unc_thresh = self.entropy_threshold
            else:
                uncertainty = variance
                unc_thresh = self.uncertainty_threshold
            
            n = len(mean_pred)
            decisions = np.ones(n, dtype=np.int32)
            zones = np.full(n, 3, dtype=np.int32)
            
            positive_mask = mean_pred >= self.recall_threshold
            zones[positive_mask] = 3
            decisions[positive_mask] = 1
            
            negative_mask = ~positive_mask
            confident_negative = negative_mask & (uncertainty < unc_thresh)
            zones[confident_negative] = 1
            decisions[confident_negative] = 0
            
            uncertain_negative = negative_mask & (uncertainty >= unc_thresh)
            zones[uncertain_negative] = 2
            decisions[uncertain_negative] = 1
            
            max_unc = uncertainty.max() if uncertainty.max() > 0 else 1.0
            confidence = 1.0 - (uncertainty / max_unc)
            
            results[f'{side}_decision'] = decisions
            results[f'{side}_zone'] = zones
            results[f'{side}_confidence'] = confidence
        
        return results
    
    def evaluate_safety(self, decisions, labels, predictions):
        metrics = {}
        for side in ['left', 'right']:
            decision = decisions[f'{side}_decision']
            label = labels[f'{side}_label']
            zone = decisions[f'{side}_zone']
            
            TP = ((decision == 1) & (label == 1)).sum()
            FN = ((decision == 0) & (label == 1)).sum()
            FP = ((decision == 1) & (label == 0)).sum()
            TN = ((decision == 0) & (label == 0)).sum()
            n_positive = (label == 1).sum()
            n_negative = (label == 0).sum()
            
            metrics[f'{side}_cancer_miss_rate'] = float(FN / max(1, n_positive))
            metrics[f'{side}_sensitivity'] = float(TP / max(1, n_positive))
            metrics[f'{side}_safe_recall_reduction'] = float(TN / max(1, n_negative))
            metrics[f'{side}_overall_recall_rate'] = float((decision == 1).sum() / len(decision))
            metrics[f'{side}_recall_reduction_pct'] = 1.0 - metrics[f'{side}_overall_recall_rate']
            metrics[f'{side}_specificity'] = float(TN / max(1, n_negative))
            
            for z in [1, 2, 3]:
                metrics[f'{side}_zone{z}_pct'] = float((zone == z).sum() / len(zone))
            
            if TP + FP > 0:
                metrics[f'{side}_ppv'] = float(TP / (TP + FP))
            if TN + FN > 0:
                metrics[f'{side}_npv'] = float(TN / (TN + FN))
        
        # Exam-level
        exam_decision = np.maximum(decisions['left_decision'], decisions['right_decision'])
        exam_label = np.maximum(labels['left_label'], labels['right_label'])
        
        exam_TP = ((exam_decision == 1) & (exam_label == 1)).sum()
        exam_FN = ((exam_decision == 0) & (exam_label == 1)).sum()
        n_pos = (exam_label == 1).sum()
        n_neg = (exam_label == 0).sum()
        exam_TN = ((exam_decision == 0) & (exam_label == 0)).sum()
        
        metrics['exam_cancer_miss_rate'] = float(exam_FN / max(1, n_pos))
        metrics['exam_sensitivity'] = float(exam_TP / max(1, n_pos))
        metrics['exam_safe_recall_reduction'] = float(exam_TN / max(1, n_neg))
        metrics['exam_recall_rate'] = float((exam_decision == 1).sum() / len(exam_decision))
        metrics['exam_recall_reduction_pct'] = 1.0 - metrics['exam_recall_rate']
        
        return metrics
    
    def find_optimal_thresholds(self, predictions, labels,
                                target_sensitivity=0.95, n_uncertainty_thresholds=50):
        from sklearn.metrics import roc_curve
        
        best_result = {
            'recall_threshold': 0.5, 'uncertainty_threshold': 0.1,
            'exam_recall_reduction': 0.0, 'exam_sensitivity': 0.0,
        }
        
        exam_preds = np.maximum(predictions['left_recall_mean'], predictions['right_recall_mean'])
        exam_labels = np.maximum(labels['left_label'], labels['right_label'])
        
        if len(np.unique(exam_labels)) < 2:
            return best_result
        
        fpr, tpr, thresholds = roc_curve(exam_labels, exam_preds)
        idx = np.where(tpr >= target_sensitivity)[0]
        if len(idx) == 0:
            return best_result
        
        best_idx = idx[np.argmin(fpr[idx])]
        recall_thresh = thresholds[best_idx]
        
        exam_var = np.maximum(predictions['left_recall_var'], predictions['right_recall_var'])
        unc_thresholds = np.linspace(0.001, np.percentile(exam_var, 95), n_uncertainty_thresholds)
        
        best_reduction = 0.0
        
        for unc_thresh in unc_thresholds:
            self.recall_threshold = recall_thresh
            self.uncertainty_threshold = unc_thresh
            
            decisions = self.make_decision(predictions)
            safety_metrics = self.evaluate_safety(decisions, labels, predictions)
            
            if safety_metrics['exam_sensitivity'] >= target_sensitivity:
                if safety_metrics['exam_safe_recall_reduction'] > best_reduction:
                    best_reduction = safety_metrics['exam_safe_recall_reduction']
                    best_result = {
                        'recall_threshold': float(recall_thresh),
                        'uncertainty_threshold': float(unc_thresh),
                        'exam_recall_reduction': float(best_reduction),
                        'exam_sensitivity': float(safety_metrics['exam_sensitivity']),
                        'exam_cancer_miss_rate': float(safety_metrics['exam_cancer_miss_rate']),
                        **{k: float(v) for k, v in safety_metrics.items()},
                    }
        
        self.recall_threshold = best_result['recall_threshold']
        self.uncertainty_threshold = best_result['uncertainty_threshold']
        
        return best_result


# ============================================================================
# ✅ Step 7d: Full Uncertainty-Aware Evaluation Pipeline
# ============================================================================

def evaluate_with_uncertainty(
    model: MCDropoutWrapper,
    data_loader,
    device: torch.device,
    target_sensitivity: float = 0.95,
    n_mc_samples: int = 10,
    output_dir: str = None,
) -> Dict:
    """
    Complete evaluation pipeline with MC Dropout uncertainty + decision zones.
    
    Metrics covered:
      - Table 1: AUROC, Spec@95%, Spec@98%, SRR, CMR, Sens Gap
      - Table 2: Zone distribution × BI-RADS, Zone 2 cancer rescue rate
      - Table 3: Per-race AUROC, Sensitivity, Specificity, Mean Variance
      - Threshold-only vs Three-Zone comparison
    
    ⚠️ IMPORTANT: `model` must be an MCDropoutWrapper instance.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    
    model.eval()
    
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)
    all_meta = defaultdict(list)
    
    print(f"\nRunning MC Dropout inference (T={n_mc_samples})...")
    
    for batch in tqdm(data_loader, desc="MC Dropout Eval"):
        current_views = {k: v.to(device) for k, v in batch['current_views'].items()}
        prior_views = {k: v.to(device) for k, v in batch['prior_views'].items()}
        current_mask = {k: v.to(device) for k, v in batch['current_mask'].items()}
        prior_mask = {k: v.to(device) for k, v in batch['prior_mask'].items()}
        race = batch['metadata']['race'].to(device)
        
        mc_output = model.predict_with_uncertainty(
            current_views, prior_views, current_mask, prior_mask, race,
            n_samples=n_mc_samples
        )
        
        for side in ['left', 'right']:
            all_preds[f'{side}_recall_mean'].append(mc_output[f'{side}_recall_mean'].cpu().numpy())
            all_preds[f'{side}_recall_var'].append(mc_output[f'{side}_recall_var'].cpu().numpy())
            all_preds[f'{side}_recall_entropy'].append(mc_output[f'{side}_recall_entropy'].cpu().numpy())
        
        all_labels['left_label'].append(batch['labels']['left_malignant'].squeeze().numpy())
        all_labels['right_label'].append(batch['labels']['right_malignant'].squeeze().numpy())
        all_meta['race'].append(batch['metadata']['race'].numpy())
        if 'birads' in batch['labels']:
            all_meta['birads'].append(batch['labels']['birads'].numpy())
        # =====================================================================
        # NEW: Collect per-side BI-RADS if available (for side-level zone analysis)
        # =====================================================================
        if 'left_birads' in batch['labels']:
            all_meta['left_birads'].append(batch['labels']['left_birads'].numpy())
            all_meta['right_birads'].append(batch['labels']['right_birads'].numpy())
    
    predictions = {k: np.concatenate(v) for k, v in all_preds.items()}
    labels = {k: np.concatenate(v) for k, v in all_labels.items()}
    meta = {k: np.concatenate(v) for k, v in all_meta.items()}
    
    results = {}
    N = len(labels['left_label'])
    
    print(f"\n{'='*80}")
    print("UNCERTAINTY-AWARE EVALUATION")
    print(f"{'='*80}")
    print(f"  Total samples: {N}")
    
    # =====================================================================
    # 1. STANDARD DISCRIMINATION METRICS (Side-level + Exam-level)
    # =====================================================================
    print(f"\n{'─'*40}")
    print("1. DISCRIMINATION METRICS")
    print(f"{'─'*40}")
    
    for side in ['left', 'right']:
        preds = predictions[f'{side}_recall_mean']
        labs = labels[f'{side}_label']
        if len(np.unique(labs)) > 1:
            auroc = roc_auc_score(labs, preds)
            results[f'{side}_auroc'] = auroc
            print(f"  {side.upper()} Breast AUROC: {auroc:.4f}")
    
    exam_preds = np.maximum(predictions['left_recall_mean'], predictions['right_recall_mean'])
    exam_labels = np.maximum(labels['left_label'], labels['right_label'])
    exam_var = np.maximum(predictions['left_recall_var'], predictions['right_recall_var'])
    
    if len(np.unique(exam_labels)) > 1:
        results['exam_auroc'] = roc_auc_score(exam_labels, exam_preds)
        print(f"  Exam AUROC: {results['exam_auroc']:.4f}")
    
    # =====================================================================
    # 2. SPECIFICITY AT FIXED SENSITIVITY (Table 1: Spec@95%, Spec@98%)
    # =====================================================================
    print(f"\n{'─'*40}")
    print("2. SPECIFICITY AT FIXED SENSITIVITY (Threshold-Only Baseline)")
    print(f"{'─'*40}")
    
    if len(np.unique(exam_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(exam_labels, exam_preds)
        
        for target_sens in [0.90, 0.95, 0.98, 0.99]:
            idx = np.where(tpr >= target_sens)[0]
            if len(idx) > 0:
                best_idx = idx[np.argmin(fpr[idx])]
                spec = 1 - fpr[best_idx]
                thresh = thresholds[best_idx]
                results[f'exam_spec_at_{int(target_sens*100)}'] = spec
                results[f'exam_thresh_at_{int(target_sens*100)}'] = float(thresh)
                print(f"  Spec@{int(target_sens*100)}% Sens: {spec:.4f}  (threshold={thresh:.4f})")
        
        # =====================================================================
        # 2b. THRESHOLD-ONLY BASELINE CMR (for comparison with Three-Zone)
        # =====================================================================
        thresh_95 = results.get('exam_thresh_at_95', 0.5)
        threshold_only_decisions = (exam_preds >= thresh_95).astype(int)
        threshold_only_TP = ((threshold_only_decisions == 1) & (exam_labels == 1)).sum()
        threshold_only_FN = ((threshold_only_decisions == 0) & (exam_labels == 1)).sum()
        threshold_only_TN = ((threshold_only_decisions == 0) & (exam_labels == 0)).sum()
        n_pos = (exam_labels == 1).sum()
        n_neg = (exam_labels == 0).sum()
        
        results['threshold_only_cmr'] = float(threshold_only_FN / max(1, n_pos))
        results['threshold_only_sensitivity'] = float(threshold_only_TP / max(1, n_pos))
        results['threshold_only_specificity'] = float(threshold_only_TN / max(1, n_neg))
        
        print(f"\n  Threshold-Only Baseline (at 95% target):")
        print(f"    CMR:          {results['threshold_only_cmr']:.4f}")
        print(f"    Sensitivity:  {results['threshold_only_sensitivity']:.4f}")
        print(f"    Specificity:  {results['threshold_only_specificity']:.4f}")
        
        # Cancers missed by threshold-only (for Zone 2 rescue comparison)
        threshold_only_missed_cancers = (threshold_only_decisions == 0) & (exam_labels == 1)
    
    # =====================================================================
    # 3. THREE-ZONE DECISION FRAMEWORK (Table 1: SRR, CMR)
    # =====================================================================
    print(f"\n{'─'*40}")
    print("3. THREE-ZONE DECISION FRAMEWORK")
    print(f"{'─'*40}")
    
    decision_maker = UncertaintyAwareDecision()
    optimal = decision_maker.find_optimal_thresholds(
        predictions, labels, target_sensitivity=target_sensitivity
    )
    
    print(f"\n  Optimal thresholds:")
    print(f"    Recall threshold:      {optimal['recall_threshold']:.4f}")
    print(f"    Uncertainty threshold:  {optimal['uncertainty_threshold']:.6f}")
    
    # Apply optimal thresholds
    decisions = decision_maker.make_decision(predictions)
    safety_metrics = decision_maker.evaluate_safety(decisions, labels, predictions)
    
    # ---- Table 1 core metrics ----
    results['SRR'] = safety_metrics['exam_safe_recall_reduction']
    results['CMR'] = safety_metrics['exam_cancer_miss_rate']
    results['three_zone_sensitivity'] = safety_metrics['exam_sensitivity']
    results['three_zone_specificity'] = results['SRR']  # SRR = specificity in three-zone
    results['three_zone_recall_reduction_pct'] = safety_metrics['exam_recall_reduction_pct']
    
    print(f"\n  Three-Zone Results:")
    print(f"    SRR (Safe Recall Reduction):  {results['SRR']:.4f}")
    print(f"    CMR (Cancer Miss Rate):       {results['CMR']:.4f}")
    print(f"    Sensitivity:                  {results['three_zone_sensitivity']:.4f}")
    print(f"    Recall Reduction %:           {results['three_zone_recall_reduction_pct']:.4f}")
    
    # ---- CMR reduction vs threshold-only ----
    if 'threshold_only_cmr' in results and results['threshold_only_cmr'] > 0:
        cmr_relative_reduction = (results['threshold_only_cmr'] - results['CMR']) / results['threshold_only_cmr']
        results['cmr_relative_reduction'] = cmr_relative_reduction
        print(f"    CMR Relative Reduction vs Threshold-Only: {cmr_relative_reduction*100:.1f}%")
    
    results.update({f'optimal_{k}': v for k, v in optimal.items()})
    
    # =====================================================================
    # 4. ZONE DISTRIBUTION (Table 2: Zone × BI-RADS alignment)
    # =====================================================================
    print(f"\n{'─'*40}")
    print("4. ZONE DISTRIBUTION & BI-RADS ALIGNMENT (Table 2)")
    print(f"{'─'*40}")
    
    # Exam-level zones: take the "worst" zone (highest zone number = most concerning)
    exam_zones = np.maximum(decisions['left_zone'], decisions['right_zone'])
    exam_decisions = np.maximum(decisions['left_decision'], decisions['right_decision'])
    
    for side in ['left', 'right']:
        zones = decisions[f'{side}_zone']
        total = len(zones)
        print(f"\n  {side.upper()} Breast:")
        for z in [1, 2, 3]:
            count = (zones == z).sum()
            pct = count / total * 100
            results[f'{side}_zone{z}_pct'] = pct / 100
            print(f"    Zone {z}: {count:,} ({pct:.1f}%)")
    
    print(f"\n  Exam-Level:")
    for z in [1, 2, 3]:
        count = (exam_zones == z).sum()
        pct = count / N * 100
        results[f'exam_zone{z}_pct'] = pct / 100
        print(f"    Zone {z}: {count:,} ({pct:.1f}%)")
    
    # ---- Zone × BI-RADS cross-tabulation (Table 2) ----
    has_birads = 'birads' in meta
    if has_birads:
        exam_birads = meta['birads']  # exam-level BI-RADS
        
        print(f"\n  Zone × BI-RADS Cross-Tabulation:")
        print(f"  {'Zone':<25} {'BR 1-2':>8} {'BR 3':>8} {'BR 4-5':>8} {'CMR':>8}")
        print(f"  {'-'*60}")
        
        for z, z_name in [(1, 'Zone 1 (Confident)'), (2, 'Zone 2 (Uncertain)'), (3, 'Zone 3 (Positive)')]:
            zone_mask = exam_zones == z
            n_zone = zone_mask.sum()
            
            if n_zone > 0:
                br_in_zone = exam_birads[zone_mask]
                
                br12_pct = ((br_in_zone <= 2)).sum() / n_zone
                br3_pct = (br_in_zone == 3).sum() / n_zone
                br45_pct = (br_in_zone >= 4).sum() / n_zone
                
                results[f'zone{z}_birads_12_pct'] = float(br12_pct)
                results[f'zone{z}_birads_3_pct'] = float(br3_pct)
                results[f'zone{z}_birads_45_pct'] = float(br45_pct)
                
                # Zone-specific CMR (only meaningful for Zone 1 where decision=no-recall)
                if z == 1:
                    zone_labels = exam_labels[zone_mask]
                    zone_fn = (zone_labels == 1).sum()  # cancers in zone 1 = missed
                    zone_cmr = zone_fn / max(1, (exam_labels == 1).sum())
                    results['zone1_cmr'] = float(zone_cmr)
                    cmr_str = f"{zone_cmr:.4f}"
                elif z == 2:
                    cmr_str = "(Saved)"
                else:
                    cmr_str = "---"
                
                print(f"  {z_name:<25} {br12_pct:>7.1%} {br3_pct:>7.1%} {br45_pct:>7.1%} {cmr_str:>8}")
        
        # ---- Zone 2 cancer rescue rate ----
        zone2_mask = exam_zones == 2
        zone2_cancers = ((exam_labels == 1) & zone2_mask).sum()
        total_cancers = (exam_labels == 1).sum()
        
        results['zone2_cancer_count'] = int(zone2_cancers)
        results['zone2_cancer_rescue_rate'] = float(zone2_cancers / max(1, total_cancers))
        
        print(f"\n  Zone 2 Cancer Rescue:")
        print(f"    Cancers in Zone 2 (rescued by uncertainty routing): {zone2_cancers}")
        print(f"    Rescue rate (of all cancers): {results['zone2_cancer_rescue_rate']:.4f}")
        
        # ---- Cancers that threshold-only would miss but Three-Zone saves ----
        if 'threshold_only_missed_cancers' in dir():
            # threshold-only missed AND three-zone caught (i.e., in Zone 2)
            rescued_from_threshold = (threshold_only_missed_cancers & zone2_mask & (exam_labels == 1)).sum()
            total_threshold_missed = threshold_only_missed_cancers.sum()
            
            results['zone2_rescued_from_threshold'] = int(rescued_from_threshold)
            results['zone2_rescue_of_threshold_missed'] = float(
                rescued_from_threshold / max(1, total_threshold_missed)
            )
            
            print(f"    Cancers missed by threshold-only but saved by Zone 2: {rescued_from_threshold}")
            if total_threshold_missed > 0:
                print(f"    Rescue % of threshold-missed cancers: "
                      f"{results['zone2_rescue_of_threshold_missed']*100:.1f}%")
    
    # =====================================================================
    # 5. PER-RACE FAIRNESS METRICS (Table 3)
    # =====================================================================
    print(f"\n{'─'*40}")
    print("5. FAIRNESS METRICS BY RACE (Table 3)")
    print(f"{'─'*40}")
    
    race_names = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Other'}
    race_aurocs = {}
    race_sensitivities = {}
    race_specificities = {}
    race_mean_vars = {}
    
    print(f"\n  {'Group':<10} {'AUROC':>8} {'Sens':>8} {'Spec':>8} {'MeanVar(×1e-3)':>15}")
    print(f"  {'-'*55}")
    
    for r_idx, r_name in race_names.items():
        mask = meta['race'] == r_idx
        if mask.sum() < 10:
            continue
        
        r_exam_preds = exam_preds[mask]
        r_exam_labels = exam_labels[mask]
        r_exam_var = exam_var[mask]
        r_exam_decisions = exam_decisions[mask]
        
        # Mean uncertainty variance
        mean_var = float(r_exam_var.mean())
        race_mean_vars[r_name] = mean_var
        results[f'race_{r_name.lower()}_mean_var'] = mean_var
        
        if len(np.unique(r_exam_labels)) > 1:
            # AUROC
            r_auroc = roc_auc_score(r_exam_labels, r_exam_preds)
            race_aurocs[r_name] = r_auroc
            results[f'race_{r_name.lower()}_auroc'] = r_auroc
            
            # Sensitivity & Specificity under Three-Zone decisions
            r_pos = (r_exam_labels == 1).sum()
            r_neg = (r_exam_labels == 0).sum()
            r_TP = ((r_exam_decisions == 1) & (r_exam_labels == 1)).sum()
            r_TN = ((r_exam_decisions == 0) & (r_exam_labels == 0)).sum()
            
            r_sens = float(r_TP / max(1, r_pos))
            r_spec = float(r_TN / max(1, r_neg))
            
            race_sensitivities[r_name] = r_sens
            race_specificities[r_name] = r_spec
            results[f'race_{r_name.lower()}_sensitivity'] = r_sens
            results[f'race_{r_name.lower()}_specificity'] = r_spec
            
            print(f"  {r_name:<10} {r_auroc:>8.4f} {r_sens:>7.1%} {r_spec:>7.1%} {mean_var*1000:>13.2f}")
        else:
            print(f"  {r_name:<10} {'N/A':>8} {'N/A':>8} {'N/A':>8} {mean_var*1000:>13.2f}")
    
    # ---- Compute gaps ----
    print(f"\n  Cross-Group Gaps:")
    
    if len(race_aurocs) >= 2:
        auroc_gap = max(race_aurocs.values()) - min(race_aurocs.values())
        results['auroc_gap'] = auroc_gap
        print(f"    AUROC gap:       {auroc_gap:.4f}  (target < 0.03)")
    
    if len(race_sensitivities) >= 2:
        sens_gap = max(race_sensitivities.values()) - min(race_sensitivities.values())
        results['sensitivity_gap'] = sens_gap
        print(f"    Sensitivity gap: {sens_gap:.4f}  (target < 0.05)")
    
    if len(race_specificities) >= 2:
        spec_gap = max(race_specificities.values()) - min(race_specificities.values())
        results['specificity_gap'] = spec_gap
        print(f"    Specificity gap: {spec_gap:.4f}")
    
    if len(race_mean_vars) >= 2:
        var_gap = max(race_mean_vars.values()) - min(race_mean_vars.values())
        results['mean_var_gap'] = var_gap
        print(f"    Mean Var gap:    {var_gap:.6f}")
    
    # =====================================================================
    # 6. BI-RADS-AWARE LOSS IMPACT ANALYSIS
    # =====================================================================
    if has_birads:
        print(f"\n{'─'*40}")
        print("6. BI-RADS STRATIFIED PERFORMANCE")
        print(f"{'─'*40}")
        
        print(f"\n  {'BI-RADS':<12} {'N':>8} {'Pos%':>8} {'MeanPred':>10} {'MeanVar':>10} {'FN_count':>10}")
        print(f"  {'-'*60}")
        
        for br in sorted(np.unique(exam_birads)):
            br_mask = exam_birads == br
            n_br = br_mask.sum()
            if n_br == 0:
                continue
            
            br_labels = exam_labels[br_mask]
            br_preds = exam_preds[br_mask]
            br_vars = exam_var[br_mask]
            br_decisions = exam_decisions[br_mask]
            
            pos_rate = (br_labels == 1).mean()
            mean_pred = br_preds.mean()
            mean_var = br_vars.mean()
            fn_count = ((br_decisions == 0) & (br_labels == 1)).sum()
            
            results[f'birads{int(br)}_n'] = int(n_br)
            results[f'birads{int(br)}_pos_rate'] = float(pos_rate)
            results[f'birads{int(br)}_fn_count'] = int(fn_count)
            results[f'birads{int(br)}_mean_var'] = float(mean_var)
            
            br_label = {0: 'Incomplete', 1: 'Negative', 2: 'Benign', 
                        3: 'Prob Benign', 4: 'Suspicious'}.get(int(br), f'BR{int(br)}')
            
            print(f"  {br_label:<12} {n_br:>8} {pos_rate:>7.1%} {mean_pred:>10.4f} {mean_var:>10.6f} {fn_count:>10}")
    
    # =====================================================================
    # 7. SAVE ALL RESULTS
    # =====================================================================
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 7a. All metrics as JSON
        with open(os.path.join(output_dir, 'uncertainty_metrics.json'), 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # 7b. Per-sample predictions CSV
        pred_df = pd.DataFrame({
            'left_pred': predictions['left_recall_mean'],
            'left_var': predictions['left_recall_var'],
            'left_entropy': predictions['left_recall_entropy'],
            'left_label': labels['left_label'],
            'left_decision': decisions['left_decision'],
            'left_zone': decisions['left_zone'],
            'right_pred': predictions['right_recall_mean'],
            'right_var': predictions['right_recall_var'],
            'right_entropy': predictions['right_recall_entropy'],
            'right_label': labels['right_label'],
            'right_decision': decisions['right_decision'],
            'right_zone': decisions['right_zone'],
            'exam_pred': exam_preds,
            'exam_label': exam_labels,
            'exam_var': exam_var,
            'exam_zone': exam_zones,
            'exam_decision': exam_decisions,
            'race': meta['race'],
        })
        if has_birads:
            pred_df['birads'] = meta['birads']
        pred_df.to_csv(os.path.join(output_dir, 'uncertainty_predictions.csv'), index=False)
        
        # 7c. Summary table for paper (easy copy-paste)
        summary_lines = [
            "=" * 60,
            "PAPER-READY SUMMARY",
            "=" * 60,
            f"AUROC:                    {results.get('exam_auroc', 'N/A')}",
            f"Spec@95% Sens:            {results.get('exam_spec_at_95', 'N/A')}",
            f"Spec@98% Sens:            {results.get('exam_spec_at_98', 'N/A')}",
            f"SRR (Three-Zone):         {results.get('SRR', 'N/A')}",
            f"CMR (Three-Zone):         {results.get('CMR', 'N/A')}",
            f"CMR (Threshold-Only):     {results.get('threshold_only_cmr', 'N/A')}",
            f"CMR Relative Reduction:   {results.get('cmr_relative_reduction', 'N/A')}",
            f"Zone 2 Cancer Rescue %:   {results.get('zone2_cancer_rescue_rate', 'N/A')}",
            f"Sensitivity Gap:          {results.get('sensitivity_gap', 'N/A')}",
            f"AUROC Gap:                {results.get('auroc_gap', 'N/A')}",
            f"Mean Var Gap:             {results.get('mean_var_gap', 'N/A')}",
            "=" * 60,
        ]
        with open(os.path.join(output_dir, 'paper_summary.txt'), 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"\n  Saved to: {output_dir}")
        print(f"    - uncertainty_metrics.json")
        print(f"    - uncertainty_predictions.csv")
        print(f"    - paper_summary.txt")
    
    print(f"\n{'='*80}\n")
    return results


# ============================================================================
# ✅ Step 8: Training & Validation Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer,
                use_amp, scaler, accumulation_steps):
    """Training epoch with BIRADSAwareLoss + NaN detection."""
    model.train()
    losses = defaultdict(float)
    
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
    
    nan_count = 0
    valid_batches = 0
    
    for batch_idx, batch in enumerate(pbar):
        current_views = {k: v.to(device) for k, v in batch['current_views'].items()}
        prior_views = {k: v.to(device) for k, v in batch['prior_views'].items()}
        current_mask = {k: v.to(device) for k, v in batch['current_mask'].items()}
        prior_mask = {k: v.to(device) for k, v in batch['prior_mask'].items()}
        race = batch['metadata']['race'].to(device)
        
        left_label = batch['labels']['left_malignant'].to(device)
        right_label = batch['labels']['right_malignant'].to(device)
        
        # =====================================================================
        # 🔧 CHANGE 1: Extract birads from batch for BIRADSAwareLoss
        # =====================================================================
        birads = batch['labels'].get('birads', None)
        if birads is not None:
            birads = birads.to(device)
        
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
                    
                    # 🔧 Pass birads in metadata
                    loss_dict = criterion(
                        predictions=predictions,
                        labels={
                            'left_malignant': left_label,
                            'right_malignant': right_label
                        },
                        metadata={'race': race, 'birads': birads}
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
                
                if torch.isnan(predictions['left_recall']).any() or torch.isnan(predictions['right_recall']).any():
                    print(f"\n⚠️  NaN in predictions at batch {batch_idx}")
                    nan_count += 1
                    optimizer.zero_grad()
                    continue
                
                # 🔧 Pass birads in metadata
                loss_dict = criterion(
                    predictions=predictions,
                    labels={
                        'left_malignant': left_label,
                        'right_malignant': right_label
                    },
                    metadata={'race': race, 'birads': birads}
                )
                
                if torch.isnan(loss_dict['total']):
                    print(f"\n⚠️  NaN in loss at batch {batch_idx}")
                    nan_count += 1
                    optimizer.zero_grad()
                    continue
                
                loss = loss_dict['total'] / accumulation_steps
            
            # Backward
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            for k, v in loss_dict.items():
                losses[k] += v.item()
            
            valid_batches += 1
            
            # Update
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
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
                'conf_pen': loss_dict['confidence_penalty'].item(),
                'fair': loss_dict['fairness'].item(),
                'nan': nan_count
            })
        
        except Exception as e:
            print(f"\n❌ Error in batch {batch_idx}: {e}")
            nan_count += 1
            optimizer.zero_grad()
            continue
    
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
    """Validation epoch with BIRADSAwareLoss."""
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
            
            # 🔧 Extract birads for validation loss too
            birads = batch['labels'].get('birads', None)
            if birads is not None:
                birads = birads.to(device)
            
            predictions = model(
                current_views=current_views,
                prior_views=prior_views,
                current_mask=current_mask,
                prior_mask=prior_mask,
                race=race
            )
            
            # 🔧 Pass birads in metadata
            loss_dict = criterion(
                predictions=predictions,
                labels={
                    'left_malignant': left_label,
                    'right_malignant': right_label
                },
                metadata={'race': race, 'birads': birads}
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
    
    left_preds = torch.cat(all_left_preds).squeeze().numpy()
    left_labels = torch.cat(all_left_labels).squeeze().numpy()
    right_preds = torch.cat(all_right_preds).squeeze().numpy()
    right_labels = torch.cat(all_right_labels).squeeze().numpy()
    race = torch.cat(all_race).numpy()
    
    from sklearn.metrics import roc_auc_score, roc_curve
    
    metrics = {}
    
    print(f"\n{'='*80}")
    print(f"VALIDATION METRICS - EPOCH {epoch+1}")
    print(f"{'='*80}")
    
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
    
    # Exam-level
    print(f"\n{'='*80}")
    print("EXAM-LEVEL PERFORMANCE")
    print(f"{'='*80}")
    
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
    
    # 🔧 NEW: Log BIRADSAwareLoss components
    print(f"\n{'─'*40}")
    print("Loss Components:")
    print(f"  Total:              {losses['total']:.4f}")
    print(f"  Recall:             {losses['recall']:.4f}")
    print(f"  Confidence Penalty: {losses['confidence_penalty']:.4f}")
    print(f"  Fairness:           {losses['fairness']:.4f}")
    
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
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    
    # =====================================================================
    # 🔧 CHANGE 1: BIRADSAwareLoss parameters (replaces RecallReductionLoss)
    # =====================================================================
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--fn_cost_ratio', type=float, default=5.0,
                        help='FN penalty relative to FP (higher = penalize missed cancer more)')
    parser.add_argument('--uncertain_birads_penalty', type=float, default=0.1,
                        help='Penalty weight for overconfident predictions on BIRADS 3')
    parser.add_argument('--fairness_lambda', type=float, default=0.05)
    
    # MC Dropout (for evaluation)
    parser.add_argument('--mc_samples', type=int, default=10,
                        help='Number of MC Dropout forward passes for uncertainty estimation')
    parser.add_argument('--target_sensitivity', type=float, default=0.95,
                        help='Target sensitivity for optimal threshold search')
    
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
    
    # Resume
    parser.add_argument('--resume', type=str, default="/projects/standard/lin01231/song0760/embed_recall_reduction/scripts/outputs/embed_recall_nyu_partial_20260212_161956/best_model.pth",
                        help='Path to checkpoint to resume from')
    parser.add_argument('--resume_optimizer', action='store_true', default=True)
    parser.add_argument('--resume_scheduler', action='store_true', default=True)
    
    # Device
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    return args


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, 
                   resume_optimizer=True, resume_scheduler=True, device='cpu'):
    print(f"\n{'='*70}")
    print("RESUMING FROM CHECKPOINT")
    print(f"{'='*70}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded model weights")
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_metric = checkpoint.get('metric', 0.0)
    
    print(f"  ✓ Resume from epoch: {start_epoch}")
    print(f"  ✓ Best metric so far: {best_metric:.4f}")
    
    if optimizer is not None and resume_optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  ✓ Loaded optimizer state")
        except Exception as e:
            print(f"  ⚠️  Failed to load optimizer state: {e}")
    
    if scheduler is not None and resume_scheduler and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  ✓ Loaded scheduler state")
        except Exception as e:
            print(f"  ⚠️  Failed to load scheduler state: {e}")
    
    scaler_state = checkpoint.get('scaler_state_dict', None)
    if scaler_state:
        print(f"  ✓ Found AMP scaler state")
    
    print(f"{'='*70}\n")
    
    return {
        'start_epoch': start_epoch,
        'best_metric': best_metric,
        'args': checkpoint.get('args', {}),
        'scaler_state': scaler_state
    }


# ============================================================================
# ✅ MAIN
# ============================================================================

if __name__ == "__main__":
    args = get_args()
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Handle resume vs new experiment
    if args.resume:
        checkpoint_path = args.resume
        if 'best_model.pth' in checkpoint_path or 'checkpoint_epoch' in checkpoint_path:
            exp_dir = os.path.dirname(checkpoint_path)
        else:
            raise ValueError(f"Invalid checkpoint path format: {checkpoint_path}")
        print(f"\n✅ RESUME MODE: {exp_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name_full = f"{args.exp_name}_{args.freeze_mode}_{timestamp}"
        exp_dir = os.path.join(args.output_dir, exp_name_full)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\n✅ NEW TRAINING: {exp_dir}")
    
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path) or not args.resume:
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    
    # =========================================================================
    # Load data
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    train_loader, val_loader, test_loader, datasets = create_data_loaders(args)
    
    # =========================================================================
    # Create model
    # =========================================================================
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    model = NYURecallModel(
        input_channels=1,
        nyu_checkpoint_path=args.nyu_checkpoint if not args.resume else None,
        freeze_mode=args.freeze_mode,
        num_races=4,
        dropout=args.dropout,
        use_prior=args.use_prior
    ).to(device)
    
    model.print_trainable_status()
    
    # =========================================================================
    # 🔧 CHANGE 1: Use BIRADSAwareLoss instead of RecallReductionLoss
    # =========================================================================
    criterion = BIRADSAwareLoss(
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        fn_cost_ratio=args.fn_cost_ratio,
        uncertain_birads_penalty=args.uncertain_birads_penalty,
        fairness_lambda=args.fairness_lambda,
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
    
    # Load checkpoint if resuming
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
        
        if scaler is not None and resume_info['scaler_state'] is not None:
            scaler.load_state_dict(resume_info['scaler_state'])
    
    # =========================================================================
    # Training loop
    # =========================================================================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    if args.resume:
        print(f"  Resuming from epoch {start_epoch}, best metric: {best_metric:.4f}")
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
        print(f"  Val Loss:   {val_losses['total']:.4f}")
        print(f"  Exam Primary Metric: {val_metrics.get('exam_primary', 0):.4f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
        writer.add_scalar('Loss/val_total', val_losses['total'], epoch)
        writer.add_scalar('Loss/val_recall', val_losses.get('recall', 0), epoch)
        writer.add_scalar('Loss/val_confidence_penalty', val_losses.get('confidence_penalty', 0), epoch)
        writer.add_scalar('Loss/val_fairness', val_losses.get('fairness', 0), epoch)
        writer.add_scalar('Metrics/exam_primary', val_metrics.get('exam_primary', 0), epoch)
        writer.add_scalar('Metrics/exam_auroc', val_metrics.get('exam_auroc', 0), epoch)
        
        # Scheduler step
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
                'args': vars(args)
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
            print(f"\n{'='*80}")
            print("EARLY STOPPING")
            print(f"{'='*80}\n")
            break
        
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
    
    # =========================================================================
    # 🔧 CHANGE 2 & 3: MC Dropout evaluation on test set after training
    # =========================================================================
    print("\n" + "="*70)
    print("POST-TRAINING: UNCERTAINTY-AWARE EVALUATION ON TEST SET")
    print("="*70)
    
    # Load best model
    best_ckpt_path = os.path.join(exp_dir, 'best_model.pth')
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print(f"  ✓ Loaded best model (epoch {best_ckpt['epoch']+1}, metric={best_ckpt['metric']:.4f})")
    
    # 🔧 CHANGE 2: Wrap model with MCDropoutWrapper
    mc_model = MCDropoutWrapper(model, n_samples=args.mc_samples)
    mc_model.to(device)
    
    # 🔧 CHANGE 3: Run uncertainty-aware evaluation
    uncertainty_output_dir = os.path.join(exp_dir, 'uncertainty_eval')
    
    test_results = evaluate_with_uncertainty(
        model=mc_model,
        data_loader=test_loader,
        device=device,
        target_sensitivity=args.target_sensitivity,
        n_mc_samples=args.mc_samples,
        output_dir=uncertainty_output_dir,
    )
    
    # Also evaluate on validation set for completeness
    val_uncertainty_dir = os.path.join(exp_dir, 'uncertainty_eval_val')
    
    val_results = evaluate_with_uncertainty(
        model=mc_model,
        data_loader=val_loader,
        device=device,
        target_sensitivity=args.target_sensitivity,
        n_mc_samples=args.mc_samples,
        output_dir=val_uncertainty_dir,
    )
    
    # =========================================================================
    # Final summary
    # =========================================================================
    # Cleanup
    if 'temp_dir' in datasets:
        shutil.rmtree(datasets['temp_dir'])
    
    print(f"\n{'='*80}")
    print("TRAINING + EVALUATION COMPLETED")
    print(f"{'='*80}")
    print(f"  Best training metric (Spec@95%Sens): {best_metric:.4f}")
    print(f"  Test Exam AUROC:                     {test_results.get('exam_auroc', 'N/A')}")
    print(f"  Test Recall Reduction:               {test_results.get('optimal_exam_recall_reduction', 'N/A')}")
    print(f"  Test Cancer Miss Rate:               {test_results.get('optimal_exam_cancer_miss_rate', 'N/A')}")
    print(f"  Experiment: {exp_dir}")
    print(f"  Uncertainty results: {uncertainty_output_dir}")
    print(f"{'='*80}\n")