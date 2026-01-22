"""
Temporal Siamese Network for Longitudinal Mammogram Comparison

This module implements a Siamese network that compares current and prior mammograms
to detect meaningful changes and reduce false-positive recalls.

Key Components:
1. Shared encoder for feature extraction
2. Spatial Transformer Network for alignment
3. Cross-attention for temporal correspondence
4. Change detection module
5. Fusion and decision layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import timm


class DepthwiseSeparableConv(nn.Module):
    """轻量级深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class SpatialTransformerNetwork(nn.Module):
    def __init__(self, in_channels: int = 2048):
        super().__init__()
        self.dim_reducer = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.in_channels = in_channels
        
        self.localization = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 512, 3, 2, 1),
            DepthwiseSeparableConv(512, 256, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 6)
        )
        
        # Initialize to identity transformation
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
    
    def forward(self, current_feat: torch.Tensor, prior_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_feat: Current exam features [B, C, H, W]
            prior_feat: Prior exam features [B, C, H, W]
        
        Returns:
            Aligned prior features [B, C, H, W]
        """
        # Concatenate for alignment estimation
        combined = torch.cat([current_feat, prior_feat], dim=1)  # [B, 2C, H, W]
        combined = self.dim_reducer(combined) # [B, 1280, H, W]
        
        # Estimate transformation
        theta = self.localization(combined)  # [B, 6]
        theta = theta.view(-1, 2, 3)  # [B, 2, 3]
        
        # Create sampling grid
        grid = F.affine_grid(theta, prior_feat.size(), align_corners=False)
        
        # Apply transformation
        aligned_prior = F.grid_sample(prior_feat, grid, align_corners=False)
        
        return aligned_prior


class CrossAttentionModule(nn.Module):
    """
    Cross-attention between current and prior features.
    Helps the model focus on relevant changes.
    """
    def __init__(self, dim: int = 2048, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, current: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current: Current features [B, N, C]
            prior: Prior features [B, N, C]
        
        Returns:
            Attended current features [B, N, C]
        """
        B, N, C = current.shape
        
        # Project to Q, K, V
        q = self.q_proj(current).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(prior).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(prior).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out


class ChangeDetectionModule(nn.Module):
    """
    Detect and quantify changes between current and aligned prior mammograms.
    Outputs change heatmap and change features.
    """
    def __init__(self, in_channels: int = 2048):
        super().__init__()
        self.dim_reducer = nn.Conv2d(in_channels * 2, in_channels, 1)
        # Change detection network
        self.change_net = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 512, 3, 1, 1),
            DepthwiseSeparableConv(512, 256, 3, 1, 1),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # Change feature extraction
        self.change_feat = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(
        self, 
        current_feat: torch.Tensor, 
        aligned_prior_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            current_feat: Current features [B, C, H, W]
            aligned_prior_feat: Aligned prior features [B, C, H, W]
        
        Returns:
            change_map: Change heatmap [B, 1, H, W]
            change_feat: Change features [B, 512]
        """
        # Concatenate features
        combined = torch.cat([current_feat, aligned_prior_feat], dim=1)
        combined = self.dim_reducer(combined)
        
        # Generate change map
        change_map = self.change_net(combined)  # [B, 1, H, W]
        change_map = torch.sigmoid(change_map)
        
        # Extract change features
        change_feat = self.change_feat(combined)  # [B, 512]
        
        return change_map, change_feat


class TemporalSiameseNetwork(nn.Module):
    """
    Main temporal model for recall prediction with multi-view fusion.
    
    Architecture:
    1. Multi-view attention fusion (4 views → fused representation)
    2. Shared encoder extracts features from current and prior
    3. STN aligns prior to current
    4. Cross-attention models temporal correspondence
    5. Change detection identifies significant changes
    6. Fusion layer combines all information for final prediction
    """
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        
        # Shared encoder
        self.encoder = self._create_encoder(backbone, pretrained)
        encoder_dim = self._get_encoder_dim(backbone)
        if freeze_backbone:
            print(f"Freezing backbone: {backbone}")
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # 统计冻结的参数
            frozen_params = sum(p.numel() for p in self.encoder.parameters())
            print(f"  Frozen parameters: {frozen_params:,}")        
        # ← NEW: Multi-view attention fusion
        self.view_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=False  # [seq_len, batch, embed_dim]
        )
        
        # Spatial Transformer Network
        
        self.stn = SpatialTransformerNetwork(encoder_dim)
        
        # Cross-attention (temporal)
        self.cross_attention = CrossAttentionModule(encoder_dim)
        
        # Change detection
        self.change_detector = ChangeDetectionModule(encoder_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim * 2 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
    def _create_encoder(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create feature encoder from timm models."""
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            in_chans=1
        )
        return model
    
    def _get_encoder_dim(self, backbone: str) -> int:
        """Get encoder output dimension."""
        dim_map = {
            'efficientnet_b0': 2048,
            'resnet101': 2048,
            'efficientnet_b0': 1280, 
            'efficientnet_b1': 1280,  
            'efficientnet_b2': 1408,  
            'efficientnet_b3': 1536,  
            'efficientnet_b4': 1792,
            'vit_base_patch16_224': 768,
        }
        return dim_map.get(backbone, 2048)
    
    def encode_multi_view(
        self, 
        views_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode 4 views with attention fusion.
        
        Args:
            views_dict: Dictionary with keys ['lcc', 'rcc', 'lmlo', 'rmlo']
                       Each value is [B, 1, H, W]
        
        Returns:
            fused_feat: [B, C, H', W'] - Fused spatial features
            attn_weights: [B, num_heads, 4, 4] - Attention weights between views
        """
        view_names = ['lcc', 'rcc', 'lmlo', 'rmlo']
        
        # Step 1: Encode each view independently
        view_feats = []
        spatial_shapes = None
        
        for view_name in view_names:
            img = views_dict[view_name]  # [B, 1, H, W]
            feat = self.encoder(img)      # [B, C, H', W']
            
            if spatial_shapes is None:
                B, C, H, W = feat.shape
                spatial_shapes = (H, W)
            
            # Global average pooling for attention
            # [B, C, H', W'] → [B, C]
            feat_pooled = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
            view_feats.append(feat_pooled)
        
        # Step 2: Stack views as sequence [4, B, C]
        view_seq = torch.stack(view_feats, dim=0)  # [4, B, C]
        
        # Step 3: Self-attention across views
        attended_seq, attn_weights = self.view_attention(
            view_seq,  # query
            view_seq,  # key
            view_seq   # value
        )  # attended_seq: [4, B, C], attn_weights: [B, 4, 4]
        
        # Step 4: Aggregate attended views
        # Simple average (you can also use weighted sum based on attention)
        fused_pooled = attended_seq.mean(dim=0)  # [B, C]
        
        # Step 5: Broadcast back to spatial dimensions for downstream processing
        # [B, C] → [B, C, H', W']
        fused_feat = fused_pooled.unsqueeze(-1).unsqueeze(-1).expand(
            B, C, spatial_shapes[0], spatial_shapes[1]
        )
        
        return fused_feat, attn_weights
    
    def forward(
        self,
        current_views: Dict[str, torch.Tensor],
        prior_views: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            current_views: Dict with 4 views ['lcc', 'rcc', 'lmlo', 'rmlo']
                          Each: [B, 1, H, W]
            prior_views: Dict with 4 views (can be None)
            return_attention: Whether to return attention maps
        
        Returns:
            Dictionary containing:
                - logits: [B, num_classes]
                - change_map: [B, 1, H', W'] (if prior exists)
                - view_attention_*: Attention weights (if return_attention=True)
        """
        # Multi-view encoding with attention
        current_feat, current_view_attn = self.encode_multi_view(current_views)
        
        # If no prior, baseline mode
        if prior_views is None:
            current_pooled = F.adaptive_avg_pool2d(current_feat, 1).flatten(1)
            fusion_feat = self.fusion(
                torch.cat([
                    current_pooled,
                    current_pooled,
                    torch.zeros_like(current_pooled[:, :512], device=current_pooled.device)
                ], dim=1)
            )
            logits = self.classifier(fusion_feat)
            return {'logits': logits}
        
        # Encode prior with multi-view fusion
        prior_feat, prior_view_attn = self.encode_multi_view(prior_views)
        
        # Align prior to current
        aligned_prior_feat = self.stn(current_feat, prior_feat)
        
        # Detect changes
        change_map, change_feat = self.change_detector(current_feat, aligned_prior_feat)
        
        # Cross-attention (temporal)
        B, C, H, W = current_feat.shape
        current_seq = current_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        prior_seq = aligned_prior_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        attended_current = self.cross_attention(current_seq, prior_seq)  # [B, H*W, C]
        attended_current = attended_current.transpose(1, 2).reshape(B, C, H, W)
        
        # Global pooling
        current_pooled = F.adaptive_avg_pool2d(attended_current, 1).flatten(1)
        prior_pooled = F.adaptive_avg_pool2d(aligned_prior_feat, 1).flatten(1)
        
        # Fusion
        fusion_input = torch.cat([current_pooled, prior_pooled, change_feat], dim=1)
        fusion_feat = self.fusion(fusion_input)
        
        # Classification
        logits = self.classifier(fusion_feat)
        
        output = {
            'logits': logits,
            'fusion_feat': fusion_feat,
            'change_map': change_map,
        }
        
        if return_attention:
            output['view_attention_current'] = current_view_attn
            output['view_attention_prior'] = prior_view_attn
            output['current_features'] = current_feat
            output['aligned_prior_features'] = aligned_prior_feat
        
        return output


def create_temporal_model(config: Dict) -> TemporalSiameseNetwork:
    """
    Factory function to create temporal model from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Initialized TemporalSiameseNetwork
    """
    model = TemporalSiameseNetwork(
        backbone=config.get('backbone', 'efficientnet_b0'),
        pretrained=config.get('pretrained', True),
        num_classes=config.get('num_classes', 1),
        dropout=config.get('dropout', 0.3)
    )
    return model


if __name__ == "__main__":
    # Test model with multi-view input
    model = TemporalSiameseNetwork(backbone='efficientnet_b0')
    
    # ← 修改1: 使用更小的图像尺寸
    H, W = 512, 256  # 或 (256, 128) 用于快速测试
    batch_size = 2
    
    current_views = {
        'lcc': torch.randn(batch_size, 1, H, W),   # ← 改小
        'rcc': torch.randn(batch_size, 1, H, W),
        'lmlo': torch.randn(batch_size, 1, H, W),
        'rmlo': torch.randn(batch_size, 1, H, W)
    }
    
    prior_views = {
        'lcc': torch.randn(batch_size, 1, H, W),
        'rcc': torch.randn(batch_size, 1, H, W),
        'lmlo': torch.randn(batch_size, 1, H, W),
        'rmlo': torch.randn(batch_size, 1, H, W)
    }
    
    print(f"Testing with image size: {H}×{W}, batch size: {batch_size}")
    
    # Test with temporal input
    output = model(current_views, prior_views, return_attention=True)
    
    print("\nOutput shapes:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test without prior
    output_baseline = model(current_views, None)
    print(f"\nBaseline mode output shape: {output_baseline['logits'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ← 修改2: 添加内存使用统计
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"\nMemory usage:")
        print(f"  RSS: {mem_info.rss / (1024**3):.2f} GB")
        print(f"  VMS: {mem_info.vms / (1024**3):.2f} GB")
    except:
        pass
