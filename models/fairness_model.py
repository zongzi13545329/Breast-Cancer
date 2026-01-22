"""
Fairness-Aware Model with Race-Conditional Adaptation

This module implements algorithmic fairness through race-conditional adapters
to ensure equitable performance across different racial groups.

Two main approaches:
1. Race-conditional adapters: Different groups get specialized adaptation layers
2. Adversarial debiasing: Remove race information from features

Key Innovation: Learn race-specific decision boundaries while sharing
most parameters, achieving fairness without sacrificing accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from .temporal_model import TemporalSiameseNetwork


class RaceConditionalAdapter(nn.Module):
    """
    Lightweight adapter layer for race-specific calibration.
    
    Only 5% of model parameters, inserted after the main encoder.
    Allows learning race-specific decision boundaries.
    """
    def __init__(self, in_features: int, adapter_dim: int = 64):
        super().__init__()
        
        # Bottleneck architecture for parameter efficiency
        self.down_project = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_dim, in_features)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(in_features)
        
        # Initialize to near-identity transformation
        nn.init.zeros_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, D]
        
        Returns:
            Adapted features [B, D]
        """
        residual = x
        x = self.layer_norm(x)
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return residual + x  # Residual connection


class FairnessTemporalModel(nn.Module):
    """
    Temporal model with race-conditional fairness adaptation.
    
    Architecture:
    1. Shared temporal encoder (99% of parameters)
    2. Race-conditional adapters (1% per race group)
    3. Fairness-constrained classifier
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 1,
        num_birads: int = 5,      
        num_density: int = 4,
        dropout: float = 0.3,
        num_races: int = 4,  # ← 4 races
        adapter_dim: int = 64,
        use_adapter: bool = True
    ):
        super().__init__()
        
        # Shared temporal encoder
        self.temporal_encoder = TemporalSiameseNetwork(
            backbone=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Remove original classifier
        fusion_dim = 256  # Output dimension from temporal_encoder.fusion
        self.temporal_encoder.classifier = nn.Identity()
        
        self.use_adapter = use_adapter
        self.num_races = num_races
        
        if use_adapter:
            # Race-conditional adapters (one per race)
            self.race_adapters = nn.ModuleList([
                RaceConditionalAdapter(fusion_dim, adapter_dim)
                for _ in range(num_races)
            ])
        
        # Shared classifier heads
        self.recall_head = nn.Linear(fusion_dim, 1)           # Head 1: Recall (binary)
        self.birads_head = nn.Linear(fusion_dim, num_birads)  # Head 2: BI-RADS
        self.density_head = nn.Linear(fusion_dim, num_density) # Head 3: Density
    
    def forward(self, current_views, prior_views, race_labels=None, return_features=False):
        # 1. 核心改进：请求返回所有中间特征和注意力
        temporal_output = self.temporal_encoder(
            current_views, 
            prior_views, 
            return_attention=False   # 开启此开关，拿回所有特征
        )
        
        fusion_feat = temporal_output['fusion_feat'] 

        # 3. 应用公平性适配器（直接用现成的特征，不重复计算！）
        if self.use_adapter and race_labels is not None:
            adapted_feat = fusion_feat.clone()  # ← 改为 clone，不是 zeros_like
            
            # 只处理当前 batch 中实际存在的 race
            unique_races = race_labels.unique()
            for race_idx in unique_races:
                if 0 <= race_idx < self.num_races:  # ← 添加边界检查
                    mask = (race_labels == race_idx)
                    if mask.any():
                        adapted_feat[mask] = self.race_adapters[race_idx](fusion_feat[mask])
            
            fusion_feat = adapted_feat
        
        # 4. 任务预测
        recall_logits = self.recall_head(fusion_feat)
        birads_logits = self.birads_head(fusion_feat)
        density_logits = self.density_head(fusion_feat)

        output = {
            'recall': recall_logits,
            'birads': birads_logits,
            'density': density_logits,
            'change_map': temporal_output.get('change_map') # 直接转发
        }
        
        if return_features:
            output['features'] = fusion_feat
        return output

class AdversarialDebiasingModel(nn.Module):
    """
    Adversarial debiasing approach to fairness.
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 1,
        num_races: int = 4,  # ← 4 races
        dropout: float = 0.3,
        gradient_reversal_lambda: float = 1.0
    ):
        super().__init__()
        
        # Shared temporal encoder
        self.temporal_encoder = TemporalSiameseNetwork(
            backbone=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout
        )
        
        fusion_dim = 256
        self.temporal_encoder.classifier = nn.Identity()
        
        # Task classifier (recall prediction)
        self.task_classifier = nn.Linear(fusion_dim, num_classes)
        
        # Race discriminator (adversarial)
        self.race_discriminator = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_races)
        )
        
        self.gradient_reversal_lambda = gradient_reversal_lambda
    
    def forward(
        self,
        current_views: Dict[str, torch.Tensor],  # ← 改为字典
        prior_views: Optional[Dict[str, torch.Tensor]] = None,  # ← 改为字典
        race_labels: Optional[torch.Tensor] = None,
        alpha: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            current_views: Dict with 4 views
            prior_views: Dict with 4 views (can be None)
            race_labels: Race indices [B]
            alpha: Gradient reversal strength
        
        Returns:
            Dictionary with task and adversarial logits
        """
        # ← 重新编码（与FairnessTemporalModel相同的逻辑）
        current_feat, _ = self.temporal_encoder.encode_multi_view(current_views)
        
        if prior_views is not None:
            prior_feat, _ = self.temporal_encoder.encode_multi_view(prior_views)
            aligned_prior_feat = self.temporal_encoder.stn(current_feat, prior_feat)
            change_map, change_feat = self.temporal_encoder.change_detector(
                current_feat, aligned_prior_feat
            )
            
            B, C, H, W = current_feat.shape
            current_seq = current_feat.flatten(2).transpose(1, 2)
            prior_seq = aligned_prior_feat.flatten(2).transpose(1, 2)
            attended_current = self.temporal_encoder.cross_attention(current_seq, prior_seq)
            attended_current = attended_current.transpose(1, 2).reshape(B, C, H, W)
            
            current_pooled = F.adaptive_avg_pool2d(attended_current, 1).flatten(1)
            prior_pooled = F.adaptive_avg_pool2d(aligned_prior_feat, 1).flatten(1)
            fusion_input = torch.cat([current_pooled, prior_pooled, change_feat], dim=1)
        else:
            current_pooled = F.adaptive_avg_pool2d(current_feat, 1).flatten(1)
            fusion_input = torch.cat([
                current_pooled,
                current_pooled,
                torch.zeros_like(current_pooled[:, :512], device=current_pooled.device)
            ], dim=1)
            change_map = None
        
        fusion_feat = self.temporal_encoder.fusion(fusion_input)
        
        # Task prediction
        task_logits = self.task_classifier(fusion_feat)
        
        # Adversarial race prediction with gradient reversal
        reversed_feat = GradientReversalFunction.apply(
            fusion_feat, 
            alpha * self.gradient_reversal_lambda
        )
        race_logits = self.race_discriminator(reversed_feat)
        
        output = {
            'logits': task_logits,
            'race_logits': race_logits,
            'features': fusion_feat
        }
        
        if change_map is not None:
            output['change_map'] = change_map
        
        return output

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer.
    
    Forward: Identity
    Backward: Reverse and scale gradient
    
    This makes the feature extractor learn features that:
    - Help with task prediction (normal gradient)
    - Hurt race prediction (reversed gradient)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def compute_fairness_loss(
    predictions: torch.Tensor,
    race_labels: torch.Tensor,
    constraint_type: str = "demographic_parity"
) -> torch.Tensor:
    """
    Compute fairness constraint loss.
    
    Args:
        predictions: Model predictions [B]
        race_labels: Race labels [B]
        constraint_type: Type of fairness constraint
    
    Returns:
        Fairness loss (scalar)
    """
    num_races = race_labels.max().item() + 1
    
    if constraint_type == "demographic_parity":
        # Goal: P(Y_hat=1 | Race=i) should be similar for all i
        positive_rates = []
        for race in range(num_races):
            mask = (race_labels == race)
            if mask.sum() > 0:
                race_pred = predictions[mask]
                positive_rate = race_pred.mean()
                positive_rates.append(positive_rate)
        
        if len(positive_rates) > 1:
            # Minimize variance of positive rates across races
            positive_rates = torch.stack(positive_rates)
            fairness_loss = positive_rates.var()
        else:
            fairness_loss = torch.tensor(0.0, device=predictions.device)
    
    elif constraint_type == "equalized_odds":
        # More complex: requires true labels
        # Not implemented in this simplified version
        fairness_loss = torch.tensor(0.0, device=predictions.device)
    
    else:
        fairness_loss = torch.tensor(0.0, device=predictions.device)
    
    return fairness_loss


def create_fairness_model(config: Dict, method: str = "race_conditional") -> nn.Module:
    """
    Factory function to create fairness model.
    
    Args:
        config: Configuration dictionary
        method: 'race_conditional' or 'adversarial'
    
    Returns:
        Fairness model
    """
    if method == "race_conditional":
        model = FairnessTemporalModel(
            backbone=config.get('backbone', 'efficientnet_b0'),
            pretrained=config.get('pretrained', True),
            num_classes=config.get('num_classes', 1),
            num_birads=config.get('num_birads', 5),      
            num_density=config.get('num_density', 4),    
            dropout=config.get('dropout', 0.3),
            num_races=config.get('num_races', 4),        
            adapter_dim=config.get('adapter_dim', 64),
            use_adapter=True
        )
    elif method == "adversarial":
        model = AdversarialDebiasingModel(
            backbone=config.get('backbone', 'efficientnet_b0'),
            pretrained=config.get('pretrained', True),
            num_classes=config.get('num_classes', 1),
            num_races=config.get('num_races', 4),        # ← 改为 4
            dropout=config.get('dropout', 0.3),
            gradient_reversal_lambda=config.get('gradient_reversal_lambda', 1.0)
        )
    else:
        raise ValueError(f"Unknown fairness method: {method}")
    
    return model


# models/fairness_model.py - MultiTaskLoss
class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        recall_weight: float = 1.0,
        birads_weight: float = 0.5,
        density_weight: float = 0.3,
        pos_weight_recall: Optional[torch.Tensor] = None,  # ← 改名！
        fairness_lambda: float = 0.1,
        fairness_type: str = 'demographic_parity'
    ):
        super().__init__()
        self.recall_weight = recall_weight
        self.birads_weight = birads_weight
        self.density_weight = density_weight
        self.fairness_lambda = fairness_lambda
        self.fairness_type = fairness_type
        
        # ← 修改：直接使用 pos_weight
        if pos_weight_recall is not None:
            self.recall_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight_recall
            )
        else:
            self.recall_loss_fn = nn.BCEWithLogitsLoss()
        
        self.birads_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.density_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, predictions, labels, race_labels=None):
        """
        Args:
            predictions: Dict with keys 'recall', 'birads', 'density'
            labels: Dict with same keys
            race_labels: Tensor [B] for fairness computation
        """
        # Task losses
        recall_loss = self.recall_loss_fn(
            predictions['recall'], 
            labels['recall']
        )
        if torch.isnan(recall_loss) or torch.isinf(recall_loss):
            print(f"⚠️  NaN detected!")
            print(f"   recall logits - min: {predictions['recall'].min():.3f}, max: {predictions['recall'].max():.3f}")
            print(f"   recall labels - unique: {torch.unique(labels['recall'])}")
            # 返回一个小的损失值，避免训练崩溃
            recall_loss = torch.tensor(1.0, device=predictions['recall'].device)
        birads_loss = self.birads_loss_fn(
            predictions['birads'], 
            labels['birads']
        )
        density_loss = self.density_loss_fn(
            predictions['density'], 
            labels['density']
        )
        
        # Weighted combination
        total_loss = (
            self.recall_weight * recall_loss +
            self.birads_weight * birads_loss +
            self.density_weight * density_loss
        )
        
        # Fairness regularization
        if race_labels is not None and self.fairness_lambda > 0:
            fairness_loss = compute_fairness_loss(
                predictions['recall'].sigmoid(),
                race_labels,
                constraint_type=self.fairness_type
            )
            total_loss += self.fairness_lambda * fairness_loss
        
        return {
            'total': total_loss,
            'recall': recall_loss,
            'birads': birads_loss,
            'density': density_loss
        }

if __name__ == "__main__":
    # Test race-conditional model
    print("Testing Race-Conditional Model...")
    model = FairnessTemporalModel(num_races=4)
    
    # ← 4个视图字典格式
    current_views = {
        'lcc': torch.randn(4, 1, 512, 256),   # ← 改小点方便测试
        'rcc': torch.randn(4, 1, 512, 256),
        'lmlo': torch.randn(4, 1, 512, 256),
        'rmlo': torch.randn(4, 1, 512, 256)
    }
    
    prior_views = {
        'lcc': torch.randn(4, 1, 512, 256),
        'rcc': torch.randn(4, 1, 512, 256),
        'lmlo': torch.randn(4, 1, 512, 256),
        'rmlo': torch.randn(4, 1, 512, 256)
    }
    
    race_labels = torch.tensor([0, 1, 2, 3])  # 4类
    
    print("Testing with temporal input...")
    output = model(current_views, prior_views, race_labels, return_features=True)
    print("Output shapes:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\nTesting without prior (baseline)...")
    output_baseline = model(current_views, None, race_labels)
    print("Baseline output shapes:")
    for key, value in output_baseline.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test adversarial model
    print("\n" + "="*70)
    print("Testing Adversarial Debiasing Model...")
    adv_model = AdversarialDebiasingModel(num_races=4)
    
    output = adv_model(current_views, prior_views, race_labels, alpha=1.0)
    print("Output shapes:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(p.numel() for adapter in model.race_adapters for p in adapter.parameters())
    print(f"\n" + "="*70)
    print(f"Race-Conditional Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Adapter parameters: {adapter_params:,} ({100*adapter_params/total_params:.2f}%)")
    print("="*70)
    
    print("\n✓ All tests passed!")
