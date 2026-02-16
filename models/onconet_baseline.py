"""
OncoNet Baseline Model - Compatible with EMBED Recall Prediction Pipeline
==========================================================================

Re-implements OncoNet's architecture (Yala et al., Radiology 2019) as a 
standalone PyTorch module that plugs directly into:
  - EMBEDRecallDataset (your dataset.py)
  - BIRADSAwareLoss (your train.py)
  - evaluate_with_uncertainty / MCDropoutWrapper (your eval pipeline)

OncoNet Architecture Summary:
  - Per-image: custom ResNet (BasicBlock×2 per layer, 4 layers, GlobalMaxPool)
  - Exam-level: cluster 4 views → aggregate → classify per breast
  - Original: ImageNet-pretrained, 3-channel input, cross-entropy loss

What we adapt for fair comparison:
  - Same 4-view input structure (L-CC, L-MLO, R-CC, R-MLO)
  - Same breast-level binary recall labels
  - Same evaluation (AUROC, Spec@Sens, MC Dropout uncertainty, fairness)
  - Input: 1-channel (grayscale mammogram) instead of 3-channel

Two variants provided:
  1. OncoNetImageOnly  - Image features only (no risk factors)
  2. OncoNetHybrid     - Image features + race embedding (like HybridDL)

Author: Yiran
Date: 2025
References:
  - Yala et al., "A Deep Learning Model to Triage Screening Mammograms", Radiology 2019
  - Yala et al., "A Deep Learning Mammography-Based Model for Improved Breast Cancer 
    Risk Prediction", Radiology 2019
  - https://github.com/yala/OncoNet_Public
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ============================================================================
# 1. OncoNet's BasicBlock (standard ResNet BasicBlock, pre-activation variant)
# ============================================================================

class OncoNetBasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock used in OncoNet.
    OncoNet uses the standard (not pre-activation) variant with:
      conv3x3 → BN → ReLU → conv3x3 → BN → (+residual) → ReLU
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# ============================================================================
# 2. OncoNet's Custom ResNet (per-view encoder)
# ============================================================================

class OncoNetResNet(nn.Module):
    """
    OncoNet's custom ResNet architecture.
    
    Default config from the paper's training command:
      --block_layout BasicBlock,2 BasicBlock,2 BasicBlock,2 BasicBlock,2
      --pool_name GlobalMaxPool
      --num_chan 3  (ImageNet pretrained)
      --img_size 1664 2048
    
    Architecture:
      conv7x7(stride=2) → BN → ReLU → MaxPool3x3(stride=2) →
      Layer1(64, ×2) → Layer2(128, ×2, stride=2) → 
      Layer3(256, ×2, stride=2) → Layer4(512, ×2, stride=2) →
      GlobalMaxPool → feature_dim=512
    
    For our 1-channel mammogram input, we modify the first conv layer.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_filters: int = 64,
        block_layout: list = None,  # [(num_blocks, stride), ...]
        pretrained_on_imagenet: bool = False,
    ):
        super().__init__()
        
        if block_layout is None:
            # OncoNet default: 4 layers, each with 2 BasicBlocks
            block_layout = [(2, 1), (2, 2), (2, 2), (2, 2)]
        
        self.inplanes = num_filters  # 64
        
        # Stem
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        channels = [num_filters, num_filters * 2, num_filters * 4, num_filters * 8]
        self.layers = nn.ModuleList()
        
        for i, ((num_blocks, stride), out_channels) in enumerate(
            zip(block_layout, channels)
        ):
            layer = self._make_layer(
                OncoNetBasicBlock, out_channels, num_blocks, stride=stride
            )
            self.layers.append(layer)
        
        # Global Max Pooling (OncoNet's default: GlobalMaxPool)
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        
        # Output feature dimension
        self.feature_dim = channels[-1]  # 512
        
        # Initialize weights
        self._initialize_weights()
        
        # Optionally load ImageNet weights and adapt first conv
        if pretrained_on_imagenet and input_channels != 3:
            self._adapt_from_imagenet(input_channels)
    
    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _adapt_from_imagenet(self, input_channels):
        """
        Load ImageNet-pretrained ResNet18 weights and adapt first conv layer
        from 3 channels to input_channels (e.g., 1 for grayscale).
        """
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            pretrained = resnet18(weights=ResNet18_Weights.DEFAULT)
            
            # Copy matching layers (skip first conv if channel mismatch)
            pretrained_dict = pretrained.state_dict()
            model_dict = self.state_dict()
            
            # Map pretrained keys to our keys
            key_mapping = {
                'conv1.weight': 'conv1.weight',
                'bn1.weight': 'bn1.weight',
                'bn1.bias': 'bn1.bias',
                'bn1.running_mean': 'bn1.running_mean',
                'bn1.running_var': 'bn1.running_var',
            }
            
            # Map layer1-4 to self.layers.0-3
            for layer_idx in range(4):
                src_prefix = f'layer{layer_idx + 1}'
                dst_prefix = f'layers.{layer_idx}'
                for k, v in pretrained_dict.items():
                    if k.startswith(src_prefix):
                        new_key = k.replace(src_prefix, dst_prefix)
                        if new_key in model_dict and v.shape == model_dict[new_key].shape:
                            key_mapping[k] = new_key
            
            # Load compatible weights
            loaded = 0
            for src_key, dst_key in key_mapping.items():
                if src_key in pretrained_dict and dst_key in model_dict:
                    src_tensor = pretrained_dict[src_key]
                    dst_tensor = model_dict[dst_key]
                    
                    if src_tensor.shape == dst_tensor.shape:
                        model_dict[dst_key] = src_tensor
                        loaded += 1
            
            # Handle first conv: average 3-channel weights → 1-channel
            if input_channels == 1:
                conv1_weight = pretrained_dict['conv1.weight']  # [64, 3, 7, 7]
                model_dict['conv1.weight'] = conv1_weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
                loaded += 1
            
            self.load_state_dict(model_dict)
            print(f"  ✓ Loaded {loaded} ImageNet-pretrained parameters (adapted to {input_channels}ch)")
            
        except Exception as e:
            print(f"  ⚠️  Could not load ImageNet weights: {e}")
            print(f"  → Using random initialization")
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - single mammogram view
        Returns:
            features: [B, 512] - global pooled features
        """
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)
        
        for layer in self.layers:
            h = layer(h)
        
        h = self.global_pool(h)  # [B, 512, 1, 1]
        h = h.view(h.size(0), -1)  # [B, 512]
        
        return h


# ============================================================================
# 3. OncoNet ImageOnly Baseline (for EMBED recall prediction)
# ============================================================================

class OncoNetImageOnly(nn.Module):
    """
    OncoNet ImageOnly baseline adapted for breast-level recall prediction.
    
    Architecture:
      1. Shared ResNet encoder processes each view independently
      2. Per-breast aggregation: max-pool over available views
      3. Per-breast classification head: FC → sigmoid
    
    Key differences from your NYURecallModel:
      - Single shared encoder (not CC/MLO split like NYU)
      - GlobalMaxPool instead of GlobalAvgPool
      - Simpler aggregation (max-pool, no attention)
      - No temporal fusion (no prior exam handling in original OncoNet)
      - Standard ResNet BasicBlock (not pre-activation like NYU)
    
    For fair comparison, we add:
      - Prior exam support (simple concatenation, not gated fusion)
      - Race embedding option (OncoNetHybrid variant)
    
    Forward signature matches your train.py expectations:
      model(current_views, prior_views, current_mask, prior_mask, race)
      → {'left_recall': [B,1], 'right_recall': [B,1]}
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        pretrained_on_imagenet: bool = True,
        use_prior: bool = True,
        dropout: float = 0.3,
        freeze_mode: str = 'none',  # OncoNet trains from scratch or ImageNet
    ):
        super().__init__()
        
        self.use_prior = use_prior
        
        # ✅ Shared encoder for all views (OncoNet's design)
        self.encoder = OncoNetResNet(
            input_channels=input_channels,
            pretrained_on_imagenet=pretrained_on_imagenet,
        )
        
        feature_dim = self.encoder.feature_dim  # 512
        
        # Apply freezing if needed
        self._apply_freezing(freeze_mode)
        
        # ✅ Per-breast classification heads
        # OncoNet: FC → dropout → output
        # We keep it simple to match the original design
        
        if use_prior:
            head_input_dim = feature_dim * 2  # current + prior
        else:
            head_input_dim = feature_dim
        
        self.left_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        
        self.right_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        
        print(f"\n✓ OncoNetImageOnly initialized:")
        print(f"  Encoder: ResNet-18 style ({feature_dim}D features)")
        print(f"  Pooling: GlobalMaxPool (OncoNet default)")
        print(f"  Use prior: {use_prior}")
        print(f"  Freeze mode: {freeze_mode}")
    
    def _apply_freezing(self, mode):
        if mode == 'full':
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("  ✓ Encoder: FULLY FROZEN")
        elif mode == 'partial':
            # Freeze stem + first 2 layers
            for param in self.encoder.conv1.parameters():
                param.requires_grad = False
            for param in self.encoder.bn1.parameters():
                param.requires_grad = False
            for i in range(min(2, len(self.encoder.layers))):
                for param in self.encoder.layers[i].parameters():
                    param.requires_grad = False
            trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.encoder.parameters())
            print(f"  ✓ Encoder: PARTIALLY FROZEN ({trainable:,}/{total:,} trainable)")
        elif mode == 'none':
            print("  ✓ Encoder: FULLY TRAINABLE")
    
    def _encode_views(self, views_dict, mask_dict):
        """
        Encode 4 views → per-breast features via max-pooling.
        
        OncoNet processes each image independently through the shared encoder,
        then aggregates at the exam/breast level.
        
        Returns:
            left_feat: [B, 512]
            right_feat: [B, 512]
        """
        B = views_dict['L-CC'].size(0)
        device = views_dict['L-CC'].device
        feature_dim = self.encoder.feature_dim
        
        # Encode each view
        encoded = {}
        for view_key in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
            encoded[view_key] = self.encoder(views_dict[view_key])  # [B, 512]
        
        # Aggregate per breast using max-pooling (OncoNet's approach)
        # Left breast: max(L-CC, L-MLO) with masking
        left_features = []
        for view_key in ['L-CC', 'L-MLO']:
            feat = encoded[view_key]  # [B, 512]
            mask = mask_dict[view_key].unsqueeze(-1)  # [B, 1]
            # Zero out invalid views so they don't affect max
            masked_feat = feat * mask + (-1e9) * (1 - mask)
            left_features.append(masked_feat)
        
        left_stacked = torch.stack(left_features, dim=1)  # [B, 2, 512]
        left_feat = left_stacked.max(dim=1)[0]  # [B, 512]
        
        # If both views are missing, use zeros instead of -1e9
        left_any_valid = (mask_dict['L-CC'] + mask_dict['L-MLO']).clamp(max=1).unsqueeze(-1)
        left_feat = left_feat * left_any_valid
        
        # Right breast: max(R-CC, R-MLO) with masking
        right_features = []
        for view_key in ['R-CC', 'R-MLO']:
            feat = encoded[view_key]
            mask = mask_dict[view_key].unsqueeze(-1)
            masked_feat = feat * mask + (-1e9) * (1 - mask)
            right_features.append(masked_feat)
        
        right_stacked = torch.stack(right_features, dim=1)
        right_feat = right_stacked.max(dim=1)[0]
        
        right_any_valid = (mask_dict['R-CC'] + mask_dict['R-MLO']).clamp(max=1).unsqueeze(-1)
        right_feat = right_feat * right_any_valid
        
        return left_feat, right_feat
    
    def forward(self, current_views, prior_views, current_mask, prior_mask, race=None):
        """
        Forward pass matching your train.py's calling convention.
        
        Args:
            current_views: dict {'L-CC': [B,1,H,W], 'L-MLO': ..., 'R-CC': ..., 'R-MLO': ...}
            prior_views: same structure
            current_mask: dict {'L-CC': [B], ...}
            prior_mask: dict
            race: [B] (ignored in ImageOnly, used in Hybrid)
        
        Returns:
            {'left_recall': [B,1], 'right_recall': [B,1]}
        """
        # Encode current exam
        left_current, right_current = self._encode_views(current_views, current_mask)
        
        if self.use_prior:
            # Encode prior exam
            left_prior, right_prior = self._encode_views(prior_views, prior_mask)
            
            # Simple concatenation (OncoNet doesn't have temporal fusion)
            left_feat = torch.cat([left_current, left_prior], dim=1)   # [B, 1024]
            right_feat = torch.cat([right_current, right_prior], dim=1)
        else:
            left_feat = left_current   # [B, 512]
            right_feat = right_current
        
        # Per-breast prediction
        left_recall = self.left_head(left_feat)    # [B, 1]
        right_recall = self.right_head(right_feat)  # [B, 1]
        
        return {
            'left_recall': left_recall,
            'right_recall': right_recall,
        }
    
    def print_trainable_status(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*70}")
        print(f"OncoNet ImageOnly - PARAMETER STATISTICS")
        print(f"{'='*70}")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters:    {total_params - trainable_params:,}")
        print(f"  Trainable ratio:      {trainable_params/total_params*100:.1f}%")
        print(f"{'='*70}\n")


# ============================================================================
# 4. OncoNet Hybrid Baseline (Image + Risk Factors)
# ============================================================================

class OncoNetHybrid(OncoNetImageOnly):
    """
    OncoNet HybridDL variant: adds risk factor (race) embedding.
    
    In the original paper, HybridDL uses a wide range of risk factors
    (density, family history, biopsy, age, menarche, etc.).
    
    For fair comparison with your model, we use the same race embedding
    that your EMBEDRecallModel uses.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        pretrained_on_imagenet: bool = True,
        use_prior: bool = True,
        dropout: float = 0.3,
        freeze_mode: str = 'none',
        num_races: int = 4,
    ):
        # Initialize parent (creates encoder + heads)
        # We'll override the heads to include race
        super().__init__(
            input_channels=input_channels,
            pretrained_on_imagenet=pretrained_on_imagenet,
            use_prior=use_prior,
            dropout=dropout,
            freeze_mode=freeze_mode,
        )
        
        feature_dim = self.encoder.feature_dim  # 512
        race_embed_dim = 64
        
        # Race embedding
        self.race_embeddings = nn.Embedding(num_races, race_embed_dim)
        nn.init.normal_(self.race_embeddings.weight, mean=0.0, std=0.02)
        
        # Rebuild heads with race input
        if use_prior:
            head_input_dim = feature_dim * 2 + race_embed_dim
        else:
            head_input_dim = feature_dim + race_embed_dim
        
        self.left_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        
        self.right_head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        
        print(f"  ✓ OncoNetHybrid: added race embedding ({num_races} → {race_embed_dim}D)")
    
    def forward(self, current_views, prior_views, current_mask, prior_mask, race=None):
        """Forward with race embedding."""
        left_current, right_current = self._encode_views(current_views, current_mask)
        
        if self.use_prior:
            left_prior, right_prior = self._encode_views(prior_views, prior_mask)
            left_feat = torch.cat([left_current, left_prior], dim=1)
            right_feat = torch.cat([right_current, right_prior], dim=1)
        else:
            left_feat = left_current
            right_feat = right_current
        
        # Add race embedding
        if race is not None:
            race_emb = self.race_embeddings(race)  # [B, 64]
            left_feat = torch.cat([left_feat, race_emb], dim=1)
            right_feat = torch.cat([right_feat, race_emb], dim=1)
        
        left_recall = self.left_head(left_feat)
        right_recall = self.right_head(right_feat)
        
        return {
            'left_recall': left_recall,
            'right_recall': right_recall,
        }
    
    def print_trainable_status(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*70}")
        print(f"OncoNet Hybrid - PARAMETER STATISTICS")
        print(f"{'='*70}")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters:    {total_params - trainable_params:,}")
        print(f"  Trainable ratio:      {trainable_params/total_params*100:.1f}%")
        print(f"{'='*70}\n")


# ============================================================================
# 5. Quick Sanity Check
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("OncoNet Baseline - Sanity Check")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2
    H, W = 2944, 1920
    
    # Create dummy input matching your dataset's output format
    current_views = {
        'L-CC':  torch.randn(B, 1, H, W).to(device),
        'L-MLO': torch.randn(B, 1, H, W).to(device),
        'R-CC':  torch.randn(B, 1, H, W).to(device),
        'R-MLO': torch.randn(B, 1, H, W).to(device),
    }
    prior_views = {k: v.clone() for k, v in current_views.items()}
    current_mask = {k: torch.ones(B).to(device) for k in current_views}
    prior_mask = {k: torch.ones(B).to(device) for k in current_views}
    race = torch.tensor([0, 1]).to(device)
    
    # Test ImageOnly
    print("\n--- OncoNetImageOnly ---")
    model_img = OncoNetImageOnly(
        input_channels=1, 
        pretrained_on_imagenet=False,
        use_prior=True,
        dropout=0.3,
    ).to(device)
    model_img.print_trainable_status()
    
    output = model_img(current_views, prior_views, current_mask, prior_mask, race)
    print(f"  left_recall shape: {output['left_recall'].shape}")
    print(f"  right_recall shape: {output['right_recall'].shape}")
    
    # Test Hybrid
    print("\n--- OncoNetHybrid ---")
    model_hyb = OncoNetHybrid(
        input_channels=1,
        pretrained_on_imagenet=False,
        use_prior=True,
        dropout=0.3,
        num_races=4,
    ).to(device)
    model_hyb.print_trainable_status()
    
    output = model_hyb(current_views, prior_views, current_mask, prior_mask, race)
    print(f"  left_recall shape: {output['left_recall'].shape}")
    print(f"  right_recall shape: {output['right_recall'].shape}")
    
    print("\n✓ All checks passed!")