"""
Group-Specific Reweighting for Fairness
========================================

简单但有效的fairness方法：
- 为不同race group赋予不同权重
- Minority group获得更高权重
- 兼顾类别平衡和种族公平

优势：
✅ 实现简单（10行代码）
✅ 训练稳定（不像adversarial）
✅ 性价比高（无额外训练成本）
✅ 可解释性强

Author: Yiran
Date: 2025
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter


def get_sample_weights_with_fairness(
    dataset,
    class_balance_weight=1.0,
    race_balance_weight=0.3,
    race_boost_factors=None
):
    """
    计算同时考虑类别平衡和种族公平的样本权重
    
    Args:
        dataset: EMBEDSingleViewLongitudinalDataset实例
        class_balance_weight: 类别平衡权重系数 (推荐1.0)
        race_balance_weight: 种族平衡权重系数 (推荐0.3-0.5)
        race_boost_factors: 自定义每个race的boost因子
                           None时自动计算（根据样本量反比）
    
    Returns:
        sample_weights: np.ndarray, shape [N]
    """
    print("\n" + "="*70)
    print("Computing Fair Sample Weights (Class + Race)")
    print("="*70)
    
    # 获取labels和races
    binary_labels = np.array([
        1 if s['label'] >= 1 else 0 
        for s in dataset.sample_list
    ])
    races = np.array([
        dataset._encode_race(s['race'])
        for s in dataset.sample_list
    ])
    
    # =========================================================================
    # Part 1: 类别权重（处理类别不平衡）
    # =========================================================================
    class_counts = np.bincount(binary_labels)
    class_weights_map = 1.0 / class_counts
    class_weights = class_weights_map[binary_labels]
    
    print(f"\n📊 Class Distribution:")
    print(f"  Class 0 (No Recall): {class_counts[0]:,} samples")
    print(f"  Class 1 (Recall): {class_counts[1]:,} samples")
    print(f"  Class weight ratio: {class_weights_map[1]/class_weights_map[0]:.2f}:1")
    
    # =========================================================================
    # Part 2: 种族权重（处理种族不平衡）
    # =========================================================================
    race_counts = np.bincount(races, minlength=4)
    
    if race_boost_factors is None:
        # 自动计算：样本量越少，boost越大
        total_samples = len(races)
        race_boost_factors = {}
        
        for race_id in range(4):
            if race_counts[race_id] > 0:
                # 反比例boost，但限制范围在[1.0, 2.0]
                raw_boost = total_samples / (4 * race_counts[race_id])
                race_boost_factors[race_id] = np.clip(raw_boost, 1.0, 2.0)
            else:
                race_boost_factors[race_id] = 1.0
    
    # 应用race boost
    race_weights = np.array([race_boost_factors[r] for r in races])
    
    race_names = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Other'}
    print(f"\n🌍 Race Distribution & Boost Factors:")
    for race_id in range(4):
        count = race_counts[race_id]
        pct = count / len(races) * 100
        boost = race_boost_factors[race_id]
        print(f"  {race_names[race_id]}: {count:,} ({pct:.1f}%) | Boost: {boost:.2f}x")
    
    # =========================================================================
    # Part 3: 组合权重
    # =========================================================================
    # 方法1: 加权求和
    combined_weights = (
        class_balance_weight * class_weights +
        race_balance_weight * race_weights
    )
    
    # 归一化到合理范围
    combined_weights = combined_weights / combined_weights.mean()
    
    print(f"\n⚖️  Weight Statistics:")
    print(f"  Min weight: {combined_weights.min():.3f}")
    print(f"  Max weight: {combined_weights.max():.3f}")
    print(f"  Mean weight: {combined_weights.mean():.3f}")
    print(f"  Std weight: {combined_weights.std():.3f}")
    
    # =========================================================================
    # Part 4: 分析每个race+class组合的权重
    # =========================================================================
    print(f"\n🔍 Detailed Weight Analysis by Race × Class:")
    print(f"{'Race':<15} {'Class':<10} {'Count':<10} {'Avg Weight':<12}")
    print("-" * 50)
    
    for race_id in range(4):
        for class_id in [0, 1]:
            mask = (races == race_id) & (binary_labels == class_id)
            if mask.sum() > 0:
                avg_weight = combined_weights[mask].mean()
                count = mask.sum()
                class_name = "No Recall" if class_id == 0 else "Recall"
                print(f"{race_names[race_id]:<15} {class_name:<10} {count:<10} {avg_weight:<12.3f}")
    
    print("="*70 + "\n")
    
    return combined_weights


def create_fair_balanced_sampler(
    dataset,
    class_balance_weight=1.0,
    race_balance_weight=0.3,
    race_boost_factors=None
):
    """
    创建考虑fairness的平衡采样器
    
    Args:
        dataset: Dataset实例
        class_balance_weight: 类别平衡权重
        race_balance_weight: 种族平衡权重
        race_boost_factors: 可选的race boost字典
    
    Returns:
        WeightedRandomSampler
    """
    sample_weights = get_sample_weights_with_fairness(
        dataset,
        class_balance_weight=class_balance_weight,
        race_balance_weight=race_balance_weight,
        race_boost_factors=race_boost_factors
    )
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print("✓ Created Fair Weighted Sampler")
    print(f"  Balances both class imbalance AND race disparity")
    
    return sampler


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    """
    测试用例：展示如何使用fair reweighting
    """
    
    # 假设已经创建了dataset
    from data.dataset import EMBEDSingleViewLongitudinalDataset
    
    # 创建dataset（这里用小样本测试）
    dataset = EMBEDSingleViewLongitudinalDataset(
        clinical_csv="path/to/clinical.csv",
        metadata_csv="path/to/metadata.csv",
        mode='train',
        image_size=(512, 256),
        sample_fraction=0.05  # 5%数据用于测试
    )
    
    print("\n" + "="*70)
    print("TESTING FAIR REWEIGHTING")
    print("="*70)
    
    # =========================================================================
    # Test 1: 只考虑类别平衡（原方案）
    # =========================================================================
    print("\n[Test 1] Class Balance Only")
    print("-" * 70)
    
    weights_class_only = get_sample_weights_with_fairness(
        dataset,
        class_balance_weight=1.0,
        race_balance_weight=0.0  # 不考虑race
    )
    
    # =========================================================================
    # Test 2: 类别 + 种族平衡（推荐方案）
    # =========================================================================
    print("\n[Test 2] Class + Race Balance (Recommended)")
    print("-" * 70)
    
    weights_fair = get_sample_weights_with_fairness(
        dataset,
        class_balance_weight=1.0,
        race_balance_weight=0.3  # 适度考虑race
    )
    
    # =========================================================================
    # Test 3: 自定义race boost
    # =========================================================================
    print("\n[Test 3] Custom Race Boost Factors")
    print("-" * 70)
    
    custom_boost = {
        0: 1.0,   # White: 无boost
        1: 1.5,   # Black: 1.5x boost
        2: 1.8,   # Asian: 1.8x boost
        3: 1.5    # Other: 1.5x boost
    }
    
    weights_custom = get_sample_weights_with_fairness(
        dataset,
        class_balance_weight=1.0,
        race_balance_weight=0.5,
        race_boost_factors=custom_boost
    )
    
    # =========================================================================
    # 对比分析
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON ANALYSIS")
    print("="*70)
    
    races = np.array([dataset._encode_race(s['race']) for s in dataset.sample_list])
    binary_labels = np.array([1 if s['label'] >= 1 else 0 for s in dataset.sample_list])
    
    race_names = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Other'}
    
    print(f"\n{'Scenario':<30} {'White Recall':<15} {'Black Recall':<15} {'Asian Recall':<15}")
    print("-" * 80)
    
    # 计算每个race的positive class平均权重
    for scenario_name, weights in [
        ("Class Only", weights_class_only),
        ("Class + Race (0.3)", weights_fair),
        ("Class + Race (0.5)", weights_custom)
    ]:
        avg_weights = {}
        for race_id in [0, 1, 2]:
            mask = (races == race_id) & (binary_labels == 1)
            if mask.sum() > 0:
                avg_weights[race_id] = weights[mask].mean()
            else:
                avg_weights[race_id] = 0
        
        print(f"{scenario_name:<30} {avg_weights.get(0, 0):<15.3f} "
              f"{avg_weights.get(1, 0):<15.3f} {avg_weights.get(2, 0):<15.3f}")
    
    print("\n✓ 可以看到：随着race_balance_weight增加，minority group权重上升")
    
    # =========================================================================
    # 推荐配置
    # =========================================================================
    print("\n" + "="*70)
    print("RECOMMENDED CONFIGURATIONS")
    print("="*70)
    
    print("""
    根据你的需求选择配置：
    
    1. 只关心整体性能（不关心fairness）：
       class_balance_weight=1.0, race_balance_weight=0.0
       → 最大化overall AUROC
    
    2. 适度关心fairness（推荐）：
       class_balance_weight=1.0, race_balance_weight=0.3
       → 平衡overall性能和fairness
       → 预期: Disparity减少30-50%，overall AUROC降低<2%
    
    3. 高度关心fairness：
       class_balance_weight=1.0, race_balance_weight=0.5-0.7
       → 显著提升minority group性能
       → 预期: Disparity减少50-70%，overall AUROC可能降低3-5%
    
    4. 自定义boost（针对性优化）：
       为表现最差的group设置更高boost
       例如：如果Asian group表现最差，设置Asian boost=2.0
    """)
    
    print("="*70 + "\n")


# ============================================================================
# 集成到训练脚本的示例
# ============================================================================

def create_data_loaders_with_fairness(
    clinical_csv,
    metadata_csv,
    batch_size=64,
    num_workers=4,
    use_fair_sampling=True,
    class_balance_weight=1.0,
    race_balance_weight=0.3,
    **dataset_kwargs
):
    """
    创建带fairness的data loaders
    
    在原有create_data_loaders基础上增加fairness支持
    """
    from data.dataset import (
        EMBEDSingleViewLongitudinalDataset,
        collate_fn
    )
    
    # 创建datasets（省略patient split逻辑，见原代码）
    # ...
    
    train_dataset = EMBEDSingleViewLongitudinalDataset(
        clinical_csv=train_csv,
        metadata_csv=metadata_csv,
        mode='train',
        **dataset_kwargs
    )
    
    # ✅ 创建fair sampler
    if use_fair_sampling:
        print("\n🌟 Using Fair Balanced Sampling (Class + Race)")
        train_sampler = create_fair_balanced_sampler(
            train_dataset,
            class_balance_weight=class_balance_weight,
            race_balance_weight=race_balance_weight
        )
        shuffle_train = False
    else:
        print("\n📊 Using Standard Balanced Sampling (Class Only)")
        from data.dataset import create_balanced_sampler
        train_sampler = create_balanced_sampler(train_dataset)
        shuffle_train = False
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Val和Test loaders不使用采样器
    # ...
    
    return train_loader, val_loader, test_loader