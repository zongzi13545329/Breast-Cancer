"""
NYU-Compatible Recall Prediction Model
使用NYU的FourViewResNet架构，替换输出头为recall prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ============================================================================
# 从NYU代码复制的基础组件
# ============================================================================

class BasicBlockV2(nn.Module):
    """NYU's BasicBlock"""
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
    """NYU's ResNet-22 architecture (完全相同)"""
    
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
        
        block_fn = BasicBlockV2
        
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
                block=block_fn,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        
        self.final_bn = nn.BatchNorm2d(
            current_num_filters // growth_factor * block_fn.expansion
        )
        self.relu = nn.ReLU()
        
        self.num_filters = num_filters
        self.growth_factor = growth_factor

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
        layers_ = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))
        return nn.Sequential(*layers_)


class AllViewsAvgPool(nn.Module):
    """NYU's pooling layer"""
    def __init__(self):
        super(AllViewsAvgPool, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x_dict):
        """
        Args:
            x_dict: dict with keys 'L-CC', 'L-MLO', 'R-CC', 'R-MLO'
        Returns:
            dict with same keys, pooled features [B, 256]
        """
        return {
            view: self.pool(x).view(x.size(0), -1)
            for view, x in x_dict.items()
        }


# ============================================================================
# NYU的FourViewResNet (关键：CC和MLO共享权重)
# ============================================================================

class FourViewResNet(nn.Module):
    """
    NYU's four-view architecture with weight sharing.
    - CC views (L-CC, R-CC) share one ResNet
    - MLO views (L-MLO, R-MLO) share one ResNet
    """
    
    def __init__(self, input_channels=1):
        super(FourViewResNet, self).__init__()
        
        # ✅ 创建两个ResNet：一个用于CC，一个用于MLO
        self.cc = ViewResNetV2(input_channels=input_channels)
        self.mlo = ViewResNetV2(input_channels=input_channels)
        
        # ✅ 创建view映射（NYU的关键设计）
        self.model_dict = {}
        self.model_dict['L-CC'] = self.cc   # 左CC使用cc
        self.model_dict['R-CC'] = self.cc   # 右CC使用cc（共享）
        self.model_dict['L-MLO'] = self.mlo # 左MLO使用mlo
        self.model_dict['R-MLO'] = self.mlo # 右MLO使用mlo（共享）
        
        # ✅ 为了兼容NYU的checkpoint，创建别名
        self.l_cc = self.cc
        self.r_cc = self.cc
        self.l_mlo = self.mlo
        self.r_mlo = self.mlo

    def forward(self, x):
        """
        Args:
            x: dict with keys 'L-CC', 'L-MLO', 'R-CC', 'R-MLO'
               each value: [B, 1, H, W]
        
        Returns:
            dict with same keys, features: [B, 256, h, w]
        """
        h_dict = {}
        for view in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
            h_dict[view] = self.model_dict[view](x[view])
        
        return h_dict


# ============================================================================
# Recall Prediction Model (基于NYU架构)
# ============================================================================

class NYURecallModel(nn.Module):
    """
    使用NYU完整架构的Recall Prediction模型
    
    架构：
    1. FourViewResNet (CC和MLO共享权重) - 从NYU加载
    2. AllViewsAvgPool - 全局池化
    3. 替换NYU的输出头为recall prediction
    """
    
    def __init__(
        self,
        input_channels=1,
        nyu_checkpoint_path=None,
        freeze_mode='partial',
        num_races=4,
        dropout=0.3,
        use_prior=True
    ):
        super(NYURecallModel, self).__init__()
        
        self.use_prior = use_prior
        
        # ✅ 1. NYU的feature extractor
        self.four_view_resnet = FourViewResNet(input_channels=input_channels)
        self.avg_pool = AllViewsAvgPool()
        
        # ✅ 2. 加载NYU预训练权重
        if nyu_checkpoint_path:
            self._load_nyu_weights(nyu_checkpoint_path)
        
        # ✅ 3. 应用冻结策略
        self._apply_freezing(freeze_mode)
        
        # ✅ 4. Recall prediction头
        # NYU原始：4个view各自有fc + output_layer
        # 我们：将4个view的features融合后预测recall
        
        feature_dim = 256  # 每个view pooling后的维度
        
        # Per-view FC layers (类似NYU)
        self.fc1_lcc = nn.Linear(feature_dim, feature_dim)
        self.fc1_rcc = nn.Linear(feature_dim, feature_dim)
        self.fc1_lmlo = nn.Linear(feature_dim, feature_dim)
        self.fc1_rmlo = nn.Linear(feature_dim, feature_dim)
        
        # 融合层
        if use_prior:
            # 如果使用prior，每个breast有current + prior features
            breast_feature_dim = feature_dim * 2 * 2  # (CC + MLO) * (current + prior)
        else:
            breast_feature_dim = feature_dim * 2  # CC + MLO
        
        # Race embedding
        self.race_embeddings = nn.Embedding(num_races, 64)
        
        # Per-breast prediction heads
        self.left_head = nn.Sequential(
            nn.Linear(breast_feature_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # binary: recall or not
        )
        
        self.right_head = nn.Sequential(
            nn.Linear(breast_feature_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def _load_nyu_weights(self, checkpoint_path):
        """加载NYU预训练权重"""
        print(f"\n{'='*70}")
        print(f"Loading NYU Pretrained Weights")
        print(f"{'='*70}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # ✅ 提取FourViewResNet的权重
        # NYU的key格式: 'four_view_resnet.cc.first_conv.weight'
        
        resnet_state_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith('four_view_resnet.'):
                # 去掉前缀 'four_view_resnet.'
                new_key = k.replace('four_view_resnet.', '')
                resnet_state_dict[new_key] = v
        
        if len(resnet_state_dict) == 0:
            print("  ⚠️  Warning: No 'four_view_resnet' keys found in checkpoint")
            print(f"  Available keys: {list(state_dict.keys())[:5]}...")
            return
        
        # 加载权重
        missing, unexpected = self.four_view_resnet.load_state_dict(
            resnet_state_dict, strict=False
        )
        
        print(f"  ✓ Loaded {len(resnet_state_dict)} parameters")
        if missing:
            print(f"  ⚠️  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  ⚠️  Unexpected keys: {len(unexpected)}")
        
        print(f"{'='*70}\n")
    
    def _apply_freezing(self, mode):
        """冻结策略"""
        if mode == 'full':
            # 完全冻结FourViewResNet
            for param in self.four_view_resnet.parameters():
                param.requires_grad = False
            print("✓ FourViewResNet: FULLY FROZEN")
        
        elif mode == 'partial':
            # 冻结前3层，训练后2层
            for name, module in self.four_view_resnet.named_modules():
                if isinstance(module, ViewResNetV2):
                    # 冻结first_conv和first_pool
                    for param in module.first_conv.parameters():
                        param.requires_grad = False
                    
                    # 冻结前3个layer
                    for i in range(min(3, len(module.layer_list))):
                        for param in module.layer_list[i].parameters():
                            param.requires_grad = False
                    
                    # 后2层和final_bn保持可训练
                    for i in range(3, len(module.layer_list)):
                        for param in module.layer_list[i].parameters():
                            param.requires_grad = True
                    
                    for param in module.final_bn.parameters():
                        param.requires_grad = True
            
            trainable = sum(p.numel() for p in self.four_view_resnet.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.four_view_resnet.parameters())
            print(f"✓ FourViewResNet: PARTIALLY FROZEN")
            print(f"  - Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
        
        elif mode == 'none':
            # 全部可训练
            for param in self.four_view_resnet.parameters():
                param.requires_grad = True
            print("✓ FourViewResNet: FULLY TRAINABLE")
        
        else:
            raise ValueError(f"Unknown freeze_mode: {mode}")
    
    def forward(self, current_views, prior_views, current_mask, prior_mask, race):
        """
        Args:
            current_views: dict {'L-CC': [B,1,H,W], ...}
            prior_views: dict (same structure)
            current_mask: dict {'L-CC': [B], ...} - 1 if valid, 0 if missing
            prior_mask: dict (same)
            race: [B] - race indices
        
        Returns:
            {
                'left_recall': [B, 1],
                'right_recall': [B, 1]
            }
        """
        B = race.size(0)
        
        # ✅ 1. Extract features using NYU's FourViewResNet
        current_features_raw = self.four_view_resnet(current_views)  # dict of [B, 256, h, w]
        current_features = self.avg_pool(current_features_raw)       # dict of [B, 256]
        
        # ✅ 2. Apply per-view FC (类似NYU的设计)
        h_lcc = F.relu(self.fc1_lcc(current_features['L-CC']))    # [B, 256]
        h_rcc = F.relu(self.fc1_rcc(current_features['R-CC']))
        h_lmlo = F.relu(self.fc1_lmlo(current_features['L-MLO']))
        h_rmlo = F.relu(self.fc1_rmlo(current_features['R-MLO']))
        
        # ✅ 3. 处理prior（如果使用）
        if self.use_prior and prior_views is not None:
            prior_features_raw = self.four_view_resnet(prior_views)
            prior_features = self.avg_pool(prior_features_raw)
            
            h_lcc_prior = F.relu(self.fc1_lcc(prior_features['L-CC']))
            h_rcc_prior = F.relu(self.fc1_rcc(prior_features['R-CC']))
            h_lmlo_prior = F.relu(self.fc1_lmlo(prior_features['L-MLO']))
            h_rmlo_prior = F.relu(self.fc1_rmlo(prior_features['R-MLO']))
            
            # 拼接current和prior
            h_lcc = torch.cat([h_lcc, h_lcc_prior], dim=1)      # [B, 512]
            h_rcc = torch.cat([h_rcc, h_rcc_prior], dim=1)
            h_lmlo = torch.cat([h_lmlo, h_lmlo_prior], dim=1)
            h_rmlo = torch.cat([h_rmlo, h_rmlo_prior], dim=1)
        
        # ✅ 4. 按breast聚合 (CC + MLO)
        left_features = torch.cat([h_lcc, h_lmlo], dim=1)   # [B, 512 or 1024]
        right_features = torch.cat([h_rcc, h_rmlo], dim=1)
        
        # ✅ 5. 添加race信息
        race_emb = self.race_embeddings(race)  # [B, 64]
        
        left_with_race = torch.cat([left_features, race_emb], dim=1)
        right_with_race = torch.cat([right_features, race_emb], dim=1)
        
        # ✅ 6. Recall prediction
        left_recall = self.left_head(left_with_race)   # [B, 1]
        right_recall = self.right_head(right_with_race)
        
        return {
            'left_recall': left_recall,
            'right_recall': right_recall
        }
    
    def print_trainable_status(self):
        """打印参数统计"""
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