#!/usr/bin/env python3
"""
智能去重：只删除真正冗余的记录
规则：
1. 同一个acc_anon，如果有 side='B' 或 NaN，同时又有 side='L'/'R'
   → 删除 L/R 记录（因为B/NaN已经涵盖了所有图像）

2. 同一个acc_anon，只有 side='L' 和 side='R'
   → 保留！这是bilateral exam的正常表示

3. 同一个acc_anon，side完全相同（如两条都是'L'）
   → 删除重复，保留第一条
"""

import pandas as pd
import numpy as np


def smart_deduplication(clinical_csv, output_csv='clinical_deduplicated.csv'):
    """
    智能去重clinical数据
    
    Args:
        clinical_csv: 输入CSV路径
        output_csv: 输出去重后的CSV路径
    """
    print("\n" + "="*80)
    print("SMART DEDUPLICATION FOR CLINICAL DATA")
    print("="*80)
    
    print(f"\nReading: {clinical_csv}")
    df = pd.read_csv(clinical_csv, low_memory=False)
    
    print(f"Original records: {len(df):,}")
    
    # ============================================================================
    # 分析每个acc_anon的side组合
    # ============================================================================
    print("\n[1] Analyzing side combinations per exam...")
    
    side_combos = df.groupby('acc_anon')['side'].apply(list).apply(lambda x: sorted(x, key=str))
    
    # 统计side组合类型
    combo_types = side_combos.apply(lambda x: tuple(sorted(set(str(s) for s in x))))
    combo_dist = combo_types.value_counts()
    
    print(f"\nSide combination distribution:")
    for combo, count in combo_dist.head(20).items():
        pct = count / len(combo_types) * 100
        print(f"  {combo}: {count:>8,} exams ({pct:>5.2f}%)")
    
    # ============================================================================
    # 识别需要处理的情况
    # ============================================================================
    print("\n[2] Identifying records to remove...")
    
    # 标记要删除的记录
    to_remove = []
    
    for acc_id, group in df.groupby('acc_anon'):
        sides = group['side'].values
        
        # 检查是否有B或NaN
        has_B = any(pd.isna(s) or str(s).strip().upper() == 'B' for s in sides)
        has_L = any(not pd.isna(s) and str(s).strip().upper() == 'L' for s in sides)
        has_R = any(not pd.isna(s) and str(s).strip().upper() == 'R' for s in sides)
        
        if has_B and (has_L or has_R):
            # 情况1: 既有B/NaN，又有L/R → 删除L/R，保留B/NaN
            for idx in group.index:
                side_val = group.loc[idx, 'side']
                if not pd.isna(side_val) and str(side_val).strip().upper() in ['L', 'R']:
                    to_remove.append(idx)
        
        elif len(group) > 2:
            # 情况2: 有超过2条记录，且不是B+L/R的情况
            # 可能是真重复，检查是否有完全相同的side
            side_counts = pd.Series(sides).value_counts()
            for side_val, count in side_counts.items():
                if count > 1:
                    # 同一个side有多条记录，保留第一条
                    duplicates = group[group['side'] == side_val].index[1:]
                    to_remove.extend(duplicates)
        
        # 情况3: 只有L和R，或只有单条记录 → 不删除（正常）
    
    print(f"\nRecords to remove: {len(to_remove):,}")
    
    # ============================================================================
    # 执行去重 - Step 1: 删除冗余的L/R
    # ============================================================================
    df_step1 = df.drop(index=to_remove).copy()
    
    print(f"After step 1 (remove redundant L/R): {len(df_step1):,}")
    print(f"Removed in step 1: {len(to_remove):,} records")
    
    # ============================================================================
    # 执行去重 - Step 2: 删除完全相同的重复记录（保留BIRADS等级更高的）
    # ============================================================================
    print("\n[Step 2] Removing exact duplicates (keeping higher BIRADS)...")
    
    # 定义BIRADS优先级（等级越高，优先级越高）
    birads_priority = {
        'K': 6,  # BIRADS 6 - 已知恶性
        'M': 5,  # BIRADS 5 - 高度可疑恶性
        'S': 4,  # BIRADS 4 - 可疑
        'P': 3,  # BIRADS 3 - 可能良性
        'B': 2,  # BIRADS 2 - 良性
        'N': 1,  # BIRADS 1 - 阴性
        'A': 0,  # BIRADS 0 - 需要额外检查
    }
    
    # 添加priority列用于排序
    df_step1['_birads_priority'] = df_step1['asses'].map(birads_priority).fillna(-1)
    
    # 对于每个(acc_anon, side)组合，按BIRADS优先级降序排序，然后保留第一条
    df_step1_sorted = df_step1.sort_values(
        by=['acc_anon', 'side', '_birads_priority'],
        ascending=[True, True, False]  # priority降序，优先级高的在前
    )
    
    df_deduplicated = df_step1_sorted.drop_duplicates(
        subset=['acc_anon', 'side'],
        keep='first'  # 保留第一条（即BIRADS优先级最高的）
    ).copy()
    
    # 删除临时列
    df_deduplicated = df_deduplicated.drop(columns=['_birads_priority'])
    
    removed_step2 = len(df_step1) - len(df_deduplicated)
    
    print(f"After step 2 (remove exact duplicates): {len(df_deduplicated):,}")
    print(f"Removed in step 2: {removed_step2:,} records")
    
    if removed_step2 > 0:
        # 显示保留了哪些BIRADS等级
        removed_in_step2 = df_step1[~df_step1.index.isin(df_deduplicated.index)]
        kept_birads = df_deduplicated[df_deduplicated['acc_anon'].isin(removed_in_step2['acc_anon'])]['asses'].value_counts()
        removed_birads = removed_in_step2['asses'].value_counts()
        
        print(f"\n  BIRADS distribution in removed records:")
        for birads in ['K', 'M', 'S', 'P', 'B', 'N', 'A']:
            count = removed_birads.get(birads, 0)
            if count > 0:
                print(f"    {birads}: {count:>6,}")
        
        print(f"\n  BIRADS distribution in kept records (from same exams):")
        for birads in ['K', 'M', 'S', 'P', 'B', 'N', 'A']:
            count = kept_birads.get(birads, 0)
            if count > 0:
                print(f"    {birads}: {count:>6,}")
    
    print(f"\nTotal removed: {len(to_remove) + removed_step2:,} records ({(len(to_remove) + removed_step2)/len(df)*100:.2f}%)")
    
    # ============================================================================
    # 验证去重后的结果
    # ============================================================================
    print("\n[3] Validating deduplicated data...")
    
    # 检查每个exam现在有多少条记录
    records_per_exam = df_deduplicated.groupby('acc_anon').size()
    
    print(f"\nRecords per exam after deduplication:")
    record_dist = records_per_exam.value_counts().sort_index()
    for n_records, n_exams in record_dist.items():
        pct = n_exams / len(records_per_exam) * 100
        print(f"  {n_records} record(s): {n_exams:>8,} exams ({pct:>5.2f}%)")
    
    # 检查side组合
    side_combos_after = df_deduplicated.groupby('acc_anon')['side'].apply(
        lambda x: tuple(sorted(set(str(s) for s in x)))
    )
    combo_dist_after = side_combos_after.value_counts()
    
    print(f"\nSide combinations after deduplication:")
    for combo, count in combo_dist_after.head(15).items():
        pct = count / len(side_combos_after) * 100
        print(f"  {combo}: {count:>8,} exams ({pct:>5.2f}%)")
    
    # ============================================================================
    # 保存结果
    # ============================================================================
    print(f"\n[4] Saving deduplicated data to: {output_csv}")
    df_deduplicated.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(df_deduplicated):,} records")
    
    # ============================================================================
    # 显示被删除的示例
    # ============================================================================
    total_removed = len(to_remove) + removed_step2
    
    if total_removed > 0:
        print("\n[5] Examples of removed records...")
        
        # Step 1移除的记录
        if len(to_remove) > 0:
            removed_df_step1 = df.loc[to_remove].copy()
            
            print(f"\n[Step 1] Removed {len(to_remove):,} redundant L/R records (first 10):")
            print("-" * 120)
            
            display_cols = ['acc_anon', 'empi_anon', 'study_date_anon', 'side', 
                           'asses', 'new_label', 'birads_updated']
            available_cols = [col for col in display_cols if col in removed_df_step1.columns]
            
            print(removed_df_step1[available_cols].head(10).to_string(index=False))
        
        # Step 2移除的记录
        if removed_step2 > 0:
            # 找出在step2被删除的记录
            removed_in_step2 = df_step1[~df_step1.index.isin(df_deduplicated.index)]
            
            print(f"\n[Step 2] Removed {removed_step2:,} exact duplicate records")
            print(f"         (Kept records with higher BIRADS priority)")
            print("-" * 120)
            
            display_cols = ['acc_anon', 'empi_anon', 'study_date_anon', 'side', 
                           'asses', 'new_label', 'birads_updated']
            available_cols = [col for col in display_cols if col in removed_in_step2.columns]
            
            # 显示前10个被删除的记录
            print("\nFirst 10 removed records:")
            print(removed_in_step2[available_cols].head(10).to_string(index=False))
            
            # 显示对应保留的记录
            removed_exam_sides = removed_in_step2[['acc_anon', 'side']].drop_duplicates().head(5)
            if len(removed_exam_sides) > 0:
                print(f"\n\nCorresponding kept records (showing BIRADS priority):")
                for _, row in removed_exam_sides.iterrows():
                    exam_id = row['acc_anon']
                    side = row['side']
                    
                    kept_record = df_deduplicated[
                        (df_deduplicated['acc_anon'] == exam_id) & 
                        (df_deduplicated['side'] == side)
                    ][available_cols]
                    
                    if len(kept_record) > 0:
                        print(f"\n  Exam {exam_id}, side={side}:")
                        print(f"    Kept: asses={kept_record.iloc[0]['asses']}")
        
        # 显示几个具体的例子，对比删除前后
        print(f"\n\nDetailed examples (showing before/after with BIRADS priority):")
        print("=" * 120)
        
        # 找几个典型的被删除的exam
        if len(to_remove) > 0:
            removed_exams = df.loc[to_remove]['acc_anon'].unique()[:2]
        else:
            removed_exams = []
        
        if removed_step2 > 0:
            removed_exams_step2 = removed_in_step2['acc_anon'].unique()[:2]
            removed_exams = list(removed_exams) + list(removed_exams_step2)
            removed_exams = removed_exams[:3]  # 最多3个例子
        
        for i, exam_id in enumerate(removed_exams, 1):
            print(f"\nExample {i}: Exam {exam_id}")
            print("-" * 120)
            
            # 原始数据
            original = df[df['acc_anon'] == exam_id][available_cols]
            print(f"BEFORE (original {len(original)} records):")
            print(original.to_string(index=False))
            
            # 去重后
            after = df_deduplicated[df_deduplicated['acc_anon'] == exam_id][available_cols]
            print(f"\nAFTER (kept {len(after)} records):")
            if len(after) > 0:
                print(after.to_string(index=False))
            else:
                print("  (All records removed - this shouldn't happen!)")
            
            print()
    
    # ============================================================================
    # 总结
    # ============================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_removed = len(to_remove) + removed_step2
    
    print(f"\n✓ Deduplication completed!")
    print(f"  Original: {len(df):,} records")
    print(f"  Removed (Step 1 - redundant L/R): {len(to_remove):,} records")
    print(f"  Removed (Step 2 - exact duplicates): {removed_step2:,} records")
    print(f"  Total removed: {total_removed:,} records ({total_removed/len(df)*100:.2f}%)")
    print(f"  Final: {len(df_deduplicated):,} records")
    print(f"  Unique exams: {df_deduplicated['acc_anon'].nunique():,}")
    
    print(f"\n💡 What was removed:")
    if total_removed > 0:
        print(f"  Step 1: Redundant L/R records when B/NaN exists")
        print(f"  Step 2: Exact duplicates (same acc_anon + same side)")
        print(f"          → Kept records with HIGHER BIRADS priority")
        print(f"          → Priority: K(6) > M(5) > S(4) > P(3) > B(2) > N(1) > A(0)")
    else:
        print(f"  Nothing! No redundant records found.")
    
    print(f"\n💡 What was kept:")
    print(f"  - Bilateral exams (same acc_anon with side='L' and side='R')")
    print(f"  - Exams with side='B' or NaN (covering all images)")
    print(f"  - Single-sided exams")
    print(f"  - Only ONE record per (acc_anon, side) combination")
    print(f"  - When duplicates exist, the one with HIGHER BIRADS is kept")
    
    print("\n" + "="*80 + "\n")
    
    return df_deduplicated


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart deduplication for clinical data')
    parser.add_argument('--clinical_csv', type=str,
                       default="/projects/standard/lin01231/song0760/embed_recall_reduction/EMBED_OpenData_clinical_relabeled.csv",
                       help='Input clinical CSV')
    parser.add_argument('--output_csv', type=str,
                       default='EMBED_OpenData_clinical_deduplicated.csv',
                       help='Output deduplicated CSV')
    
    args = parser.parse_args()
    
    import os
    if not os.path.exists(args.clinical_csv):
        print(f"❌ File not found: {args.clinical_csv}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} --clinical_csv <input> --output_csv <output>")
        sys.exit(1)
    
    try:
        df_dedup = smart_deduplication(args.clinical_csv, args.output_csv)
        print("✅ Deduplication completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)