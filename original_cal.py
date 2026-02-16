#!/usr/bin/env python3
"""
基于去重后的clinical数据统计各种指标
包括：exam-level和image-level的分布、召回率、公平性等
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from collections import Counter
import sys


def map_asses_to_binary(asses_value):
    if pd.isna(asses_value):
        return np.nan
    asses_str = str(asses_value).strip().upper()
    if asses_str in ['N', 'B']:  # BIRADS 1/2
        return 0
    elif asses_str in ['A', 'P', 'S', 'M', 'K']:  # BIRADS 0/3/4/5/6
        return 1
    else:
        return np.nan


def map_new_label_to_binary(new_label):
    if pd.isna(new_label):
        return np.nan
    return 1 if new_label >= 1 else 0


def encode_race(ethnicity_str):
    if pd.isna(ethnicity_str) or ethnicity_str == '':
        return 3
    ethnicity_str = str(ethnicity_str).strip().lower()
    if 'white' in ethnicity_str or 'caucasian' in ethnicity_str:
        return 0
    elif 'black' in ethnicity_str or 'african' in ethnicity_str:
        return 1
    elif 'asian' in ethnicity_str:
        return 2
    else:
        return 3


def calculate_fairness_metrics(y_true, y_pred, race):
    unique_races = [r for r in np.unique(race) if not pd.isna(r)]
    dp_rates = {}
    acc_rates = {}
    for r in unique_races:
        mask = (race == r)
        if mask.sum() == 0:
            continue
        dp_rates[r] = y_pred[mask].mean()
        acc_rates[r] = (y_true[mask] == y_pred[mask]).mean()
    if len(dp_rates) >= 2:
        dp_diff = max(dp_rates.values()) - min(dp_rates.values())
        if 0 in acc_rates and 1 in acc_rates:
            acc_diff_wb = acc_rates[0] - acc_rates[1]
        else:
            acc_diff_wb = max(acc_rates.values()) - min(acc_rates.values())
    else:
        dp_diff = 0.0
        acc_diff_wb = 0.0
    return dp_diff, acc_diff_wb, dp_rates, acc_rates


def comprehensive_statistics(clinical_csv, metadata_csv):
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICS - DEDUPLICATED DATA")
    print("="*80)
    
    # [1/6] Loading data...
    print("\n[1/6] Loading data...")
    dtype_spec = {'empi_anon': str, 'acc_anon': str, 'side': str, 'ImageLateralityFinal': str}
    clinical_df = pd.read_csv(clinical_csv, dtype=dtype_spec, low_memory=False)
    metadata_df = pd.read_csv(metadata_csv, dtype=dtype_spec, low_memory=False)
    
    print(f"  Clinical records (deduplicated): {len(clinical_df):,}")
    print(f"  Metadata records (all): {len(metadata_df):,}")
    
    if 'ETHNICITY_DESC' in clinical_df.columns:
        clinical_df['race'] = clinical_df['ETHNICITY_DESC'].apply(encode_race)
    elif 'race' not in clinical_df.columns:
        clinical_df['race'] = 3
    
    if 'FinalImageType' in metadata_df.columns:
        metadata_2d = metadata_df[metadata_df['FinalImageType'] == '2D'].copy()
        print(f"  2D images only: {len(metadata_2d):,}")
        img_type_dist = metadata_df['FinalImageType'].value_counts()
        print(f"\n  Image Type Distribution:")
        for img_type, count in img_type_dist.items():
            pct = count / len(metadata_df) * 100
            print(f"    {img_type}: {count:>8,} ({pct:>5.2f}%)")
    else:
        metadata_2d = metadata_df.copy()
        print(f"  ⚠️  'FinalImageType' column not found, using all images")

# [2/6] Exam-Level Statistics...
    print("\n[2/6] Exam-Level Statistics...")
    print("="*80)
    n_exams = len(clinical_df)
    n_unique_patients = clinical_df['empi_anon'].nunique()
    print(f"\n📊 Basic Statistics:")
    print(f"  Total exams: {n_exams:,}")
    print(f"  Unique patients: {n_unique_patients:,}")
    
    clinical_df['y_true'] = clinical_df['new_label'].apply(map_new_label_to_binary)
    clinical_df['y_pred'] = clinical_df['asses'].apply(map_asses_to_binary)
    valid_mask = clinical_df['y_true'].notna() & clinical_df['y_pred'].notna()
    clinical_valid = clinical_df[valid_mask].copy()
    y_true_exam = clinical_valid['y_true'].values.astype(int)
    y_pred_exam = clinical_valid['y_pred'].values.astype(int)
    
    # 计算 Exam-level 指标
    tn, fp, fn, tp = confusion_matrix(y_true_exam, y_pred_exam).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_true_exam, y_pred_exam, zero_division=0)
    recall_tpr = recall_score(y_true_exam, y_pred_exam, zero_division=0)
    f1 = f1_score(y_true_exam, y_pred_exam, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # AUROC
    clinical_valid['asses_priority'] = clinical_valid['asses'].map({'A': 0, 'N': 1, 'B': 2, 'P': 3, 'S': 4, 'M': 5, 'K': 6})
    y_score = clinical_valid['asses_priority'].values / 6.0
    auroc = roc_auc_score(y_true_exam, y_score) if len(np.unique(y_true_exam)) > 1 else 0.5
    aupr = average_precision_score(y_true_exam, y_score) if len(np.unique(y_true_exam)) > 1 else 0.5
    
    # Fairness
    dp_diff, acc_diff_wb, _, _ = calculate_fairness_metrics(y_true_exam, y_pred_exam, clinical_valid['race'].values)

    print(f"  Accuracy: {accuracy:.4f} (Exam-Level)")

    # [3/6] Image-Level Statistics...
    print("\n[3/6] Image-Level Statistics...")
    print("="*80)

    # 修正点 3: 核心合并逻辑，解决 NaN 同时匹配 L/R 的问题
    # A. 侧性精准匹配 (L/R)
    clinical_lr = clinical_df[clinical_df['side'].isin(['L', 'R'])].copy()
    merged_lr = pd.merge(
        metadata_2d,
        clinical_lr[['empi_anon', 'acc_anon', 'side', 'new_label', 'asses', 'y_true', 'y_pred', 'race']],
        left_on=['empi_anon', 'acc_anon', 'ImageLateralityFinal'],
        right_on=['empi_anon', 'acc_anon', 'side'],
        how='inner'
    )

    # B. 广播匹配 (side 为 NaN 或 'B')
    clinical_broad = clinical_df[clinical_df['side'].isna() | (clinical_df['side'] == 'B')].copy()
    merged_broad = pd.merge(
        metadata_2d,
        clinical_broad[['empi_anon', 'acc_anon', 'new_label', 'asses', 'y_true', 'y_pred', 'race']],
        on=['empi_anon', 'acc_anon'],
        how='inner'
    )

    # C. 合并去重，赋值给 merged_filtered 适配后续所有逻辑
    merged = pd.concat([merged_lr, merged_broad], ignore_index=True)
    merged_filtered = merged.drop_duplicates(subset=metadata_2d.columns.tolist())
    n_images = len(merged_filtered)

    print(f"  Merge completed. Matched {n_images:,} images.")

    # 修正点 4: 增强 Debug 打印，显示 empi_anon | acc_anon | side
    debug_outer = pd.merge(
        metadata_2d[['empi_anon', 'acc_anon', 'ImageLateralityFinal']],
        clinical_df[['empi_anon', 'acc_anon', 'side']],
        left_on=['empi_anon', 'acc_anon', 'ImageLateralityFinal'],
        right_on=['empi_anon', 'acc_anon', 'side'],
        how='left',
        indicator=True
    )
    # 彻底无法匹配的（排除已通过 NaN 广播匹配上的）
    no_label_images = debug_outer[
        (debug_outer['_merge'] == 'left_only') & 
        (~debug_outer['acc_anon'].isin(clinical_broad['acc_anon']))
    ]

    if len(no_label_images) > 0:
        print(f"\n[Debug] Images with NO matching Clinical Labels (Total: {len(no_label_images):,}):")
        print(f"Sample (first 20) - empi_anon | acc_anon | ImageLateralityFinal")
        print(no_label_images[['empi_anon', 'acc_anon', 'ImageLateralityFinal']].head(20).to_string(index=False))
    
    # 每个 exam 的图像数
    images_per_exam = merged_filtered.groupby('acc_anon').size()
    
    print(f"\n📸 Images per Exam:")
    print(f"  Average: {images_per_exam.mean():.2f}")
    print(f"  Median:  {images_per_exam.median():.0f}")
    print(f"  Min:     {images_per_exam.min()}")
    print(f"  Max:     {images_per_exam.max()}")
    
    # 分布统计
    img_count_dist = images_per_exam.value_counts().sort_values(ascending=False)
    print(f"\n  Distribution (top 5 most common):")
    for n_imgs, n_exams_count in img_count_dist.head(5).items():
        pct = n_exams_count / len(images_per_exam) * 100
        print(f"    {n_imgs} images: {n_exams_count:>8,} exams ({pct:>5.2f}%)")
    
    # 找到图像数最多的 exam
    max_images_exam_id = images_per_exam.idxmax()
    max_images_count = images_per_exam.max()
    
    print(f"\n  📌 Exam with most images ({max_images_count} images after filter):")
    max_exam_images = merged_filtered[merged_filtered['acc_anon'] == max_images_exam_id]
    max_exam_clinical = clinical_df[clinical_df['acc_anon'] == max_images_exam_id]
    
    if len(max_exam_clinical) > 0:
        print(f"     Exam ID: {max_images_exam_id}")
        print(f"     Patient ID: {max_exam_clinical.iloc[0]['empi_anon']}")
        if 'study_date_anon' in max_exam_clinical.columns:
            print(f"     Study Date: {max_exam_clinical.iloc[0]['study_date_anon']}")
        print(f"     Side: {max_exam_clinical.iloc[0]['side']}")
        print(f"     BIRADS: {max_exam_clinical.iloc[0]['asses']}")
        print(f"     Ground Truth: {max_exam_clinical.iloc[0]['new_label']}")
        
        if len(max_exam_clinical) > 1:
            print(f"\n     ⚠️  WARNING: This exam has {len(max_exam_clinical)} records in clinical!")
            print(f"        Clinical records for this exam:")
            for idx, row in max_exam_clinical.iterrows():
                print(f"          Record {idx}: side={row['side']}, asses={row['asses']}, new_label={row['new_label']}")
        
        exam_in_metadata = metadata_2d[metadata_2d['acc_anon'] == max_images_exam_id]
        print(f"\n     📊 Image count verification:")
        print(f"        In metadata (2D): {len(exam_in_metadata)} images")
        print(f"        After merge + filter: {max_images_count} images")
        
        if len(exam_in_metadata) > 0:
            dup_factor = max_images_count / len(exam_in_metadata)
            if dup_factor > 1.01:
                print(f"        ⚠️  Duplication factor: {dup_factor:.1f}x")
        
        if 'ImageLateralityFinal' in exam_in_metadata.columns:
            lat_dist = exam_in_metadata['ImageLateralityFinal'].value_counts()
            print(f"        Laterality in metadata: {dict(lat_dist)}")
        
        print(f"\n     💡 To diagnose this exam, run:")
        print(f"        python diagnose_exam.py --exam_id {max_images_exam_id}")
    
    # View 分布
    if 'ViewPosition' in merged_filtered.columns and 'ImageLateralityFinal' in merged_filtered.columns:
        merged_filtered['view_name'] = (
            merged_filtered['ImageLateralityFinal'] + 
            merged_filtered['ViewPosition']
        )
        view_dist = merged_filtered['view_name'].value_counts()
        print(f"\n📷 View Distribution:")
        for view in ['LCC', 'RCC', 'LMLO', 'RMLO']:
            count = view_dist.get(view, 0)
            pct = count / n_images * 100
            print(f"  {view}: {count:>8,} ({pct:>5.2f}%)")
    
    # 修正点 5: 初始化所有性能指标变量，防止 assignment 错误
    accuracy_img = precision_img = recall_img = f1_img = fpr_img = 0.0
    auroc_img = aupr_img = dp_diff_img = acc_diff_wb_img = 0.0
    y_true_img = np.array([])

    # Image-level 统计
    valid_mask_img = merged_filtered['y_true'].notna() & merged_filtered['y_pred'].notna()
    merged_valid = merged_filtered[valid_mask_img].copy()
    
    if len(merged_valid) > 0:
        y_true_img = merged_valid['y_true'].values.astype(int)
        y_pred_img = merged_valid['y_pred'].values.astype(int)
        
        print(f"\n📊 Binary Classification (Image-Level):")
        print(f"  Valid images: {len(merged_valid):,}")
        print(f"\n  Ground Truth:")
        print(f"    No Recall:   {(y_true_img==0).sum():>8,} ({(y_true_img==0).mean()*100:>5.2f}%)")
        print(f"    Need Recall: {(y_true_img==1).sum():>8,} ({(y_true_img==1).mean()*100:>5.2f}%)")
        if (y_true_img==1).sum() > 0:
            print(f"    Imbalance Ratio: {(y_true_img==0).sum()/(y_true_img==1).sum():.2f}:1")
        
        tn_img, fp_img, fn_img, tp_img = confusion_matrix(y_true_img, y_pred_img).ravel()
        print(f"\n📋 Confusion Matrix:")
        print(f"  TN: {tn_img:>8,}  |  FP: {fp_img:>8,}")
        print(f"  FN: {fn_img:>8,}  |  TP: {tp_img:>8,}")
        
        accuracy_img = (tp_img + tn_img) / len(y_true_img)
        precision_img = precision_score(y_true_img, y_pred_img, zero_division=0)
        recall_img = recall_score(y_true_img, y_pred_img, zero_division=0)
        f1_img = f1_score(y_true_img, y_pred_img, zero_division=0)
        fpr_img = fp_img / (fp_img + tn_img) if (fp_img + tn_img) > 0 else 0
        
        print(f"\n📈 Performance Metrics (Image-Level):")
        print(f"  Accuracy:   {accuracy_img:.4f}")
        print(f"  Precision:  {precision_img:.4f}")
        print(f"  Recall/TPR: {recall_img:.4f}")
        print(f"  F1 Score:   {f1_img:.4f}")
        print(f"  FPR:        {fpr_img:.4f}")
        
        merged_valid['asses_prio'] = merged_valid['asses'].map({'A': 0, 'N': 1, 'B': 2, 'P': 3, 'S': 4, 'M': 5, 'K': 6})
        y_score_img = merged_valid['asses_prio'].values / 6.0
        if len(np.unique(y_true_img)) > 1:
            auroc_img = roc_auc_score(y_true_img, y_score_img)
            aupr_img = average_precision_score(y_true_img, y_score_img)
            print(f"  AUROC:      {auroc_img:.4f}")
            print(f"  AUPR:       {aupr_img:.4f}")
        
        if 'race' in merged_valid.columns:
            dp_diff_img, acc_diff_wb_img, _, _ = calculate_fairness_metrics(y_true_img, y_pred_img, merged_valid['race'].values)
            print(f"\n⚖️  Fairness Metrics (Image-Level):")
            print(f"  DP Diff:        {dp_diff_img:.4f}")
            print(f"  Acc Diff (W-B): {acc_diff_wb_img:.4f}")

    # [4/6] COMPARISON
    print("\n[4/6] Comparison: Exam-Level vs Image-Level...")
    print("="*80)
    print(f"{'Metric':<40} {'Exam-Level':<20} {'Image-Level':<20}")
    print(f"{'-'*40} {'-'*20} {'-'*20}")
    print(f"{'Total Samples':<40} {len(clinical_valid):>19,} {len(merged_valid):>19,}")
    pos_rate_img = y_true_img.mean() if len(y_true_img) > 0 else 0
    print(f"{'Positive Rate':<40} {(y_true_exam==1).mean():>18.2%} {pos_rate_img:>18.2%}")
    print(f"{'Accuracy':<40} {accuracy:>18.4f} {accuracy_img:>18.4f}")
    print(f"{'Precision':<40} {precision:>18.4f} {precision_img:>18.4f}")
    print(f"{'Recall/TPR':<40} {recall_tpr:>18.4f} {recall_img:>18.4f}")
    print(f"{'F1 Score':<40} {f1:>18.4f} {f1_img:>18.4f}")
    print(f"{'FPR':<40} {fpr:>18.4f} {fpr_img:>18.4f}")
    print(f"{'AUROC':<40} {auroc:>18.4f} {auroc_img:>18.4f}")
    print(f"{'DP Diff':<40} {dp_diff:>18.4f} {dp_diff_img:>18.4f}")
    print(f"{'Acc Diff (W-B)':<40} {acc_diff_wb:>18.4f} {acc_diff_wb_img:>18.4f}")

    # [5/6] Recall Rate by View
    print("\n[5/6] Recall Rate by View...")
    print("="*80)
    if 'view_name' in merged_valid.columns:
        print(f"\n📷 Ground Truth Recall Rate by View:")
        for view in ['LCC', 'RCC', 'LMLO', 'RMLO']:
            view_data = merged_valid[merged_valid['view_name'] == view]
            if len(view_data) > 0:
                recall_v = (view_data['y_true'] == 1).mean() * 100
                print(f"  {view}: {recall_v:>5.2f}% ({len(view_data):>8,} images)")

    # [6/6] Training Recommendations
    print("\n[6/6] Training Recommendations...")
    print("="*80)
    if len(y_true_img) > 0 and (y_true_img == 1).sum() > 0:
        n_neg, n_pos = (y_true_img == 0).sum(), (y_true_img == 1).sum()
        weight_0, weight_1 = len(y_true_img)/(2*n_neg), len(y_true_img)/(2*n_pos)
        print(f"\n💡 Class Imbalance:\n  Imbalance Ratio: {n_neg/n_pos:.2f}:1")
        print(f"💡 Suggested pos_weight: {weight_1/weight_0:.4f}")
    
    print(f"\n💡 Baseline to Beat:\n  Precision: {precision_img:.4f} | Recall: {recall_img:.4f} | FPR: {fpr_img:.4f}")

    # DATA VALIDATION SUMMARY
    print("\n" + "="*80 + "\nDATA VALIDATION SUMMARY\n" + "="*80)
    print(f"  1. Clinical (deduplicated): {len(clinical_df):,} exams")
    print(f"  2. Metadata (2D only): {len(metadata_2d):,} images")
    print(f"  3. After merge: {len(merged):,} records")
    print(f"  4. After laterality filter: {len(merged_filtered):,} images")
    
    dup_exams = clinical_df['acc_anon'].value_counts()
    print(f"  ✓ No duplicate exams in clinical" if (dup_exams > 1).sum() == 0 else f"  ✗ {(dup_exams > 1).sum()} dups found")
    print(f"  - Merge match rate: {len(merged)/len(metadata_2d)*100:.1f}%")
    print(f"  - Average images per exam: {images_per_exam.mean():.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive statistics on deduplicated data')
    parser.add_argument('--clinical_csv', type=str,
                       default='EMBED_OpenData_clinical_deduplicated.csv',
                       help='Deduplicated clinical CSV')
    parser.add_argument('--metadata_csv', type=str,
                       default='/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_metadata_png.csv',
                       help='Metadata CSV')
    
    args = parser.parse_args()
    
    import os
    if not os.path.exists(args.clinical_csv):
        print(f"❌ File not found: {args.clinical_csv}")
        print("\nPlease run smart_deduplication.py first to generate the deduplicated file.")
        print("\nOr specify the correct path:")
        print(f"  python {sys.argv[0]} --clinical_csv <path> --metadata_csv <path>")
        sys.exit(1)
    
    try:
        comprehensive_statistics(args.clinical_csv, args.metadata_csv)
        print("✅ Statistical analysis completed!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)