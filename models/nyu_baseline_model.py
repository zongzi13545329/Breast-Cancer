"""
NYU Breast Cancer Classifier Baseline - Faithful Reproduction
=============================================================
Reproduces the NYU four-view breast cancer classification model
(Wu et al., IEEE TMI 2019) as a baseline for EMBED recall prediction.

Architecture (from the paper):
1. Four separate ResNet-22 backbones (one per view: L-CC, R-CC, L-MLO, R-MLO)
2. Global average pooling → 256-dim feature per view
3. Per-breast aggregation: mean(CC, MLO) features
4. Shared FC head → P(benign), P(malignant) per breast

Key differences from custom model:
- NO temporal fusion (no prior exams)
- NO race conditioning
- NO attention-based aggregation (simple mean pooling)
- Faithful to original NYU architecture

Author: Yiran
Date: 2025
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# NYU ResNet-22 Architecture (Unchanged from original)
# ============================================================================

class BasicBlockV2(nn.Module):
    """Pre-activation residual block (from NYU code)."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
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
    """
    NYU's ResNet-22 architecture (exact reproduction).
    
    5 residual layers with [2,2,2,2,2] blocks.
    Growth factor 2: 16 → 32 → 64 → 128 → 256 channels.
    Output: 256-dim feature map.
    """
    
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
        super().__init__()
        
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
        
        # Output dimension: 256 (16 * 2^4)
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
# NYU Four-View Model (Faithful Reproduction)
# ============================================================================

class NYUFourViewModel(nn.Module):
    """
    NYU's four-view breast cancer classifier (Wu et al., IEEE TMI 2019).
    
    This is the IMAGE-ONLY model from the paper:
    - 4 view-specific ResNet-22 backbones
    - Global average pooling per view
    - Per-breast feature averaging (CC + MLO) / 2
    - Shared classification head per breast
    
    Output: left_benign, left_malignant, right_benign, right_malignant
    
    Adapted for EMBED recall prediction:
    - Binary task: predict recall (malignant=1) vs no recall (malignant=0)
    - Use pretrained NYU weights for initialization
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        nyu_checkpoint_path: str = None,
        freeze_mode: str = 'partial',
        num_classes: int = 1,  # Binary: recall or not
    ):
        super().__init__()
        
        self.feature_dim = 256  # NYU ResNet-22 output dim
        
        # =====================================================================
        # 4 View-Specific ResNet-22 Backbones (same as NYU)
        # NYU uses separate weights per view in their pretrained model:
        #   four_view_resnet.cc, four_view_resnet.mlo
        # But for CC views (L-CC, R-CC) they share one backbone,
        # and for MLO views (L-MLO, R-MLO) they share another.
        # =====================================================================
        self.cc_backbone = ViewResNetV2(input_channels=input_channels)
        self.mlo_backbone = ViewResNetV2(input_channels=input_channels)
        
        # Global Average Pooling (same as NYU)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # =====================================================================
        # Classification Head (NYU style)
        # NYU uses: features → FC → output
        # For baseline: shared head for left and right breast
        # =====================================================================
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Load pretrained weights
        if nyu_checkpoint_path and os.path.exists(nyu_checkpoint_path):
            self._load_nyu_weights(nyu_checkpoint_path)
        
        # Apply freezing strategy
        self._apply_freeze(freeze_mode)
    
    def _load_nyu_weights(self, checkpoint_path: str):
        """
        Load NYU pretrained weights.
        
        NYU checkpoint key structure:
            four_view_resnet.cc.first_conv.weight
            four_view_resnet.cc.layer_list.0.xxx
            four_view_resnet.mlo.first_conv.weight
            four_view_resnet.mlo.layer_list.0.xxx
        """
        print(f"\n{'='*70}")
        print(f"Loading NYU Pretrained Weights (Baseline)")
        print(f"{'='*70}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Print available key prefixes for debugging
        prefixes = set()
        for k in state_dict.keys():
            parts = k.split('.')
            if len(parts) >= 2:
                prefixes.add('.'.join(parts[:2]))
        print(f"  Available key prefixes: {sorted(prefixes)}")
        
        # Load CC backbone weights
        cc_loaded = self._load_view_weights(
            state_dict, self.cc_backbone, 
            prefix='four_view_resnet.cc.'
        )
        print(f"  CC backbone: loaded {cc_loaded} parameters")
        
        # Load MLO backbone weights
        mlo_loaded = self._load_view_weights(
            state_dict, self.mlo_backbone,
            prefix='four_view_resnet.mlo.'
        )
        print(f"  MLO backbone: loaded {mlo_loaded} parameters")
        
        # Try to load classifier weights if available
        cls_dict = {}
        for k, v in state_dict.items():
            # NYU may use 'fc1', 'fc2' or similar naming
            if 'fc' in k.lower() or 'classifier' in k.lower():
                cls_dict[k] = v
        
        if cls_dict:
            print(f"  Found {len(cls_dict)} classifier keys (skipping - task mismatch)")
        
        print(f"{'='*70}\n")
    
    def _load_view_weights(self, state_dict, backbone, prefix):
        """Load weights for a specific view backbone."""
        filtered = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                filtered[new_key] = v
        
        if len(filtered) == 0:
            # Try alternative prefixes
            for alt_prefix in ['resnet.', 'backbone.', 'encoder.']:
                for k, v in state_dict.items():
                    if k.startswith(alt_prefix):
                        new_key = k[len(alt_prefix):]
                        filtered[new_key] = v
                if filtered:
                    break
        
        if filtered:
            missing, unexpected = backbone.load_state_dict(filtered, strict=False)
            if missing:
                print(f"    Missing: {len(missing)} keys")
            if unexpected:
                print(f"    Unexpected: {len(unexpected)} keys")
            return len(filtered) - len(unexpected)
        
        return 0
    
    def _apply_freeze(self, freeze_mode: str):
        """Apply freezing strategy to backbones."""
        if freeze_mode == 'full':
            # Freeze all backbone parameters
            for param in self.cc_backbone.parameters():
                param.requires_grad = False
            for param in self.mlo_backbone.parameters():
                param.requires_grad = False
            print("✓ Backbones: FULLY FROZEN")
            
        elif freeze_mode == 'partial':
            # Freeze first 3 layers, train last 2 + final_bn
            for backbone_name, backbone in [('CC', self.cc_backbone), ('MLO', self.mlo_backbone)]:
                for param in backbone.first_conv.parameters():
                    param.requires_grad = False
                for i, layer in enumerate(backbone.layer_list):
                    if i < 3:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        for param in layer.parameters():
                            param.requires_grad = True
                for param in backbone.final_bn.parameters():
                    param.requires_grad = True
            
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"✓ Backbones: PARTIALLY FROZEN (layers 0-2 frozen, 3-4 trainable)")
            print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
            
        elif freeze_mode == 'none':
            for param in self.parameters():
                param.requires_grad = True
            print("✓ Backbones: FULLY TRAINABLE")
            
        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")
    
    def extract_view_features(self, views_dict, mask_dict):
        """
        Extract 256-dim features from each view using the appropriate backbone.
        
        Args:
            views_dict: {'L-CC': [B,1,H,W], 'L-MLO': ..., 'R-CC': ..., 'R-MLO': ...}
            mask_dict:  {'L-CC': [B], ...}  1.0 if view exists, 0.0 otherwise
        
        Returns:
            features: {'L-CC': [B,256], 'L-MLO': [B,256], 'R-CC': [B,256], 'R-MLO': [B,256]}
        """
        features = {}
        
        # CC views share cc_backbone
        for view_key in ['L-CC', 'R-CC']:
            x = views_dict[view_key]  # [B, 1, H, W]
            feat_map = self.cc_backbone(x)  # [B, 256, h, w]
            feat = self.global_avg_pool(feat_map)  # [B, 256, 1, 1]
            feat = feat.view(feat.size(0), -1)  # [B, 256]
            features[view_key] = feat
        
        # MLO views share mlo_backbone
        for view_key in ['L-MLO', 'R-MLO']:
            x = views_dict[view_key]
            feat_map = self.mlo_backbone(x)
            feat = self.global_avg_pool(feat_map)
            feat = feat.view(feat.size(0), -1)
            features[view_key] = feat
        
        return features
    
    def aggregate_breast_features(self, features, mask):
        """
        NYU-style per-breast aggregation: simple mean of CC and MLO features.
        
        Handles missing views by using only the available view.
        
        Args:
            features: dict of view features
            mask: dict of view masks
        
        Returns:
            left_feat:  [B, 256]
            right_feat: [B, 256]
        """
        B = features['L-CC'].size(0)
        device = features['L-CC'].device
        
        # Left breast: mean of L-CC and L-MLO
        l_cc_mask = mask['L-CC'].unsqueeze(1)   # [B, 1]
        l_mlo_mask = mask['L-MLO'].unsqueeze(1)  # [B, 1]
        
        l_sum = features['L-CC'] * l_cc_mask + features['L-MLO'] * l_mlo_mask
        l_count = l_cc_mask + l_mlo_mask
        l_count = l_count.clamp(min=1.0)  # Avoid division by zero
        left_feat = l_sum / l_count
        
        # Right breast: mean of R-CC and R-MLO
        r_cc_mask = mask['R-CC'].unsqueeze(1)
        r_mlo_mask = mask['R-MLO'].unsqueeze(1)
        
        r_sum = features['R-CC'] * r_cc_mask + features['R-MLO'] * r_mlo_mask
        r_count = r_cc_mask + r_mlo_mask
        r_count = r_count.clamp(min=1.0)
        right_feat = r_sum / r_count
        
        return left_feat, right_feat
    
    def forward(self, current_views, prior_views, current_mask, prior_mask, race=None):
        """
        Forward pass (NYU baseline - current views only, no prior/race).
        
        Interface matches your existing training script for drop-in replacement.
        
        Args:
            current_views: dict of 4 views, each [B, 1, H, W]
            prior_views:   ignored (baseline has no temporal component)
            current_mask:  dict of 4 masks, each [B]
            prior_mask:    ignored
            race:          ignored (baseline has no race conditioning)
        
        Returns:
            dict with 'left_recall' [B, 1] and 'right_recall' [B, 1]
        """
        # Step 1: Extract per-view features
        features = self.extract_view_features(current_views, current_mask)
        
        # Step 2: Per-breast aggregation (NYU-style mean)
        left_feat, right_feat = self.aggregate_breast_features(features, current_mask)
        
        # Step 3: Classification
        left_logit = self.classifier(left_feat)    # [B, 1]
        right_logit = self.classifier(right_feat)  # [B, 1]
        
        return {
            'left_recall': left_logit,
            'right_recall': right_logit,
        }
    
    def print_trainable_status(self):
        """Print parameter statistics."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print(f"\n{'='*70}")
        print(f"NYU BASELINE MODEL - PARAMETER STATISTICS")
        print(f"{'='*70}")
        print(f"  CC backbone params:  {sum(p.numel() for p in self.cc_backbone.parameters()):,}")
        print(f"  MLO backbone params: {sum(p.numel() for p in self.mlo_backbone.parameters()):,}")
        print(f"  Classifier params:   {sum(p.numel() for p in self.classifier.parameters()):,}")
        print(f"  {'─'*50}")
        print(f"  Total:     {total:,}")
        print(f"  Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
        print(f"  Frozen:    {frozen:,} ({frozen/total*100:.1f}%)")
        print(f"{'='*70}\n")


# ============================================================================
# NYU Baseline with Separate Heads (Alternative)
# ============================================================================

class NYUFourViewSeparateHeads(NYUFourViewModel):
    """
    Variant with separate classification heads for left and right breast.
    
    This more closely matches NYU's original design where the left and right
    breast can learn different decision boundaries.
    """
    
    def __init__(self, input_channels=1, nyu_checkpoint_path=None,
                 freeze_mode='partial', num_classes=1):
        # Initialize parent but we'll override the classifier
        super().__init__(
            input_channels=input_channels,
            nyu_checkpoint_path=nyu_checkpoint_path,
            freeze_mode=freeze_mode,
            num_classes=num_classes,
        )
        
        # Override: separate heads for left and right
        self.left_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        self.right_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # Remove shared classifier
        del self.classifier
    
    def forward(self, current_views, prior_views, current_mask, prior_mask, race=None):
        features = self.extract_view_features(current_views, current_mask)
        left_feat, right_feat = self.aggregate_breast_features(features, current_mask)
        
        left_logit = self.left_classifier(left_feat)
        right_logit = self.right_classifier(right_feat)
        
        return {
            'left_recall': left_logit,
            'right_recall': right_logit,
        }


# ============================================================================
# Quick test
# ============================================================================

if __name__ == '__main__':
    print("Testing NYU Baseline Model...")
    
    model = NYUFourViewModel(
        input_channels=1,
        nyu_checkpoint_path=None,
        freeze_mode='none',
    )
    model.print_trainable_status()
    
    # Simulate a batch
    B = 2
    H, W = 2944, 1920
    
    # Use small images for testing
    H_test, W_test = 128, 128
    
    current_views = {
        'L-CC': torch.randn(B, 1, H_test, W_test),
        'L-MLO': torch.randn(B, 1, H_test, W_test),
        'R-CC': torch.randn(B, 1, H_test, W_test),
        'R-MLO': torch.randn(B, 1, H_test, W_test),
    }
    current_mask = {
        'L-CC': torch.ones(B),
        'L-MLO': torch.ones(B),
        'R-CC': torch.ones(B),
        'R-MLO': torch.tensor([1.0, 0.0]),  # Second sample missing R-CC
    }
    
    output = model(
        current_views=current_views,
        prior_views=current_views,  # ignored
        current_mask=current_mask,
        prior_mask=current_mask,    # ignored
        race=torch.zeros(B, dtype=torch.long),  # ignored
    )
    
    print(f"Left recall logits:  {output['left_recall'].shape}")
    print(f"Right recall logits: {output['right_recall'].shape}")
    print(f"Left values:  {torch.sigmoid(output['left_recall']).detach()}")
    print(f"Right values: {torch.sigmoid(output['right_recall']).detach()}")
    print("\n✓ Model test passed!")