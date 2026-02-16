#!/usr/bin/env python3
"""
处理BIRADS评分的脚本（完整版）
对于BIRADS=0/1/2的记录，查找指定时间窗口内的后续记录
基于"后见之明"重新标注真实风险等级
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys
from datetime import timedelta

def process_birads_labels(input_csv, output_csv, followup_window_days=365):
    """
    处理BIRADS评分标签（支持0/1/2的追踪）
    
    Parameters:
    -----------
    input_csv : str
        输入CSV文件路径
    output_csv : str
        输出CSV文件路径
    followup_window_days : int
        随访时间窗口（天），默认365天
    """
    
    # 读取数据
    print(f"正在读取数据: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"总记录数: {len(df):,}")
    
    # 检查必要的列
    required_cols = ['empi_anon', 'study_date_anon', 'asses']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    
    # 转换日期列为datetime类型
    df['study_date_anon'] = pd.to_datetime(df['study_date_anon'])
    
    # 按患者ID和日期排序
    df = df.sort_values(['empi_anon', 'study_date_anon']).reset_index(drop=True)
    
    # 统计原始BIRADS分布
    print("\n" + "="*70)
    print("处理前的BIRADS评分分布:")
    print("="*70)
    original_birads_counts = df['asses'].value_counts().sort_index()
    birads_names = {
        'A': 'BIRADS 0 (需要额外检查)',
        'N': 'BIRADS 1 (阴性)',
        'B': 'BIRADS 2 (良性)',
        'P': 'BIRADS 3 (可能良性)',
        'S': 'BIRADS 4 (可疑)',
        'M': 'BIRADS 5 (高度可疑恶性)',
        'K': 'BIRADS 6 (已知恶性)'
    }
    for birads, count in original_birads_counts.items():
        birads_name = birads_names.get(birads, f'Unknown ({birads})')
        percentage = (count / len(df)) * 100
        print(f"{birads_name}: {count:,} 条记录 ({percentage:.2f}%)")
    
    # 创建新列存储处理后的BIRADS评分和诊断信息
    df['birads_updated'] = df['asses'].copy()
    df['followup_count'] = 0
    df['days_to_diagnosis'] = np.nan
    df['diagnosis_in_window'] = False
    df['was_updated'] = False  # 标记是否被更新过
    
    # ✅ 定义需要追踪的BIRADS评分
    track_birads = ['A', 'N', 'B']  # BIRADS 0, 1, 2
    
    print(f"\n正在处理需要追踪的BIRADS评分: {', '.join(track_birads)}")
    print(f"时间窗口: {followup_window_days}天")
    
    # 统计各类型的更新
    update_stats = {
        'A': {'total': 0, 'updated': 0, 'no_followup': 0},
        'N': {'total': 0, 'updated': 0, 'no_followup': 0},
        'B': {'total': 0, 'updated': 0, 'no_followup': 0}
    }
    
    # 记录所有的转换
    all_transitions = []
    
    # 处理需要追踪的BIRADS记录
    for track_birads_value in track_birads:
        birads_indices = df[df['asses'] == track_birads_value].index
        update_stats[track_birads_value]['total'] = len(birads_indices)
        
        print(f"\n处理 BIRADS {birads_names[track_birads_value]}: {len(birads_indices):,} 条记录")
        
        for idx in birads_indices:
            current_patient = df.loc[idx, 'empi_anon']
            current_date = df.loc[idx, 'study_date_anon']
            
            # 查找时间窗口内的后续记录
            same_patient_mask = (df['empi_anon'] == current_patient)
            future_records_mask = (df['study_date_anon'] > current_date)
            window_end_date = current_date + timedelta(days=followup_window_days)
            within_window_mask = (df['study_date_anon'] <= window_end_date)
            
            followup_mask = same_patient_mask & future_records_mask & within_window_mask
            followup_records = df[followup_mask]
            
            if len(followup_records) > 0:
                # 定义BIRADS评分的优先级
                birads_priority = {
                    'N': 1, 'B': 2, 'P': 3, 'S': 4, 'M': 5, 'K': 6, 'A': 0
                }
                
                # 找到优先级最高的BIRADS评分
                max_priority = 0
                max_birads = track_birads_value  # 默认保持原值
                days_to_max = np.nan
                
                for i, birads in enumerate(followup_records['asses'].values):
                    if pd.notna(birads) and birads in birads_priority:
                        priority = birads_priority[birads]
                        if priority > max_priority:
                            max_priority = priority
                            max_birads = birads
                            days_to_max = (followup_records.iloc[i]['study_date_anon'] - current_date).days
                
                # ✅ 更新逻辑
                # 如果后续诊断更严重，则更新
                current_priority = birads_priority[track_birads_value]
                if max_priority > current_priority:
                    df.loc[idx, 'birads_updated'] = max_birads
                    df.loc[idx, 'was_updated'] = True
                    update_stats[track_birads_value]['updated'] += 1
                    all_transitions.append((
                        current_patient,
                        track_birads_value,
                        max_birads,
                        days_to_max
                    ))
                
                df.loc[idx, 'followup_count'] = len(followup_records)
                df.loc[idx, 'days_to_diagnosis'] = days_to_max
                df.loc[idx, 'diagnosis_in_window'] = True
            else:
                update_stats[track_birads_value]['no_followup'] += 1
    
    # 打印更新统计
    print("\n" + "="*70)
    print("追踪更新统计:")
    print("="*70)
    for birads_code, stats in update_stats.items():
        print(f"\n{birads_names[birads_code]}:")
        print(f"  总数: {stats['total']:,}")
        print(f"  已更新: {stats['updated']:,} ({stats['updated']/max(stats['total'],1)*100:.1f}%)")
        print(f"  无随访: {stats['no_followup']:,} ({stats['no_followup']/max(stats['total'],1)*100:.1f}%)")
    
    # 显示所有转换统计
    if all_transitions:
        print("\n" + "="*70)
        print("BIRADS 转换详细统计:")
        print("="*70)
        transitions_df = pd.DataFrame(
            all_transitions,
            columns=['patient', 'original', 'updated', 'days']
        )
        
        # 按原始BIRADS分组
        for orig_birads in track_birads:
            orig_transitions = transitions_df[transitions_df['original'] == orig_birads]
            if len(orig_transitions) > 0:
                print(f"\n{birads_names[orig_birads]} 的转换:")
                transition_counts = orig_transitions['updated'].value_counts().sort_index()
                for new_birads, count in transition_counts.items():
                    percentage = (count / len(orig_transitions)) * 100
                    avg_days = orig_transitions[orig_transitions['updated'] == new_birads]['days'].mean()
                    print(f"  → {birads_names[new_birads]}: {count:,} ({percentage:.2f}%), "
                          f"平均 {avg_days:.1f} 天")
    
    # 统计更新后的BIRADS分布
    print("\n" + "="*70)
    print("更新后的BIRADS评分分布（丢弃前）:")
    print("="*70)
    updated_birads_counts = df['birads_updated'].value_counts().sort_index()
    for birads, count in updated_birads_counts.items():
        birads_name = birads_names.get(birads, f'Unknown ({birads})')
        percentage = (count / len(df)) * 100
        print(f"{birads_name}: {count:,} 条记录 ({percentage:.2f}%)")
    
    # ✅ 丢弃无有效随访的BIRADS 0记录
    print("\n" + "="*70)
    print("正在丢弃无有效随访的BIRADS=0记录...")
    print("="*70)
    
    # 只丢弃原始是A且未被更新的记录
    df_before = len(df)
    df = df[~((df['asses'] == 'A') & (df['birads_updated'] == 'A'))].copy()
    df_after = len(df)
    dropped_count = df_before - df_after
    
    print(f"已丢弃: {dropped_count:,} 条记录（原始BIRADS 0且无随访）")
    print(f"剩余记录: {df_after:,} 条")
    
    # 映射BIRADS评分到三分类标签
    def map_birads_to_label(birads):
        """映射BIRADS评分到三分类标签"""
        if pd.isna(birads):
            return np.nan
        
        birads = str(birads).strip()
        
        if birads in ['N', 'B']:
            return 0  # 阴性/良性
        elif birads == 'P':
            return 1  # 可能良性（需要随访）
        elif birads in ['S', 'M', 'K']:
            return 2  # 可疑/恶性
        else:
            return np.nan
    
    df['new_label'] = df['birads_updated'].apply(map_birads_to_label)
    
    # 添加额外的标注列
    df['original_birads'] = df['asses']  # 保留原始BIRADS
    
    # 统计新标签分布
    print("\n" + "="*70)
    print("映射后的三分类标签分布:")
    print("="*70)
    label_counts = df['new_label'].value_counts().sort_index()
    label_names = {
        0: "Class 0 (BIRADS 1/2: 阴性/良性)",
        1: "Class 1 (BIRADS 3: 可能良性)",
        2: "Class 2 (BIRADS 4/5/6: 可疑/恶性)"
    }
    
    for label in sorted(label_counts.index):
        if not pd.isna(label):
            count = label_counts[label]
            percentage = (count / len(df)) * 100
            
            # 统计被更新的样本数
            updated_count = df[(df['new_label'] == label) & (df['was_updated'] == True)].shape[0]
            
            print(f"\n{label_names[int(label)]}:")
            print(f"  总计: {count:,} 条记录 ({percentage:.2f}%)")
            print(f"  其中重新标注: {updated_count:,} 条")
    
    if df['new_label'].isna().sum() > 0:
        print(f"\n缺失值: {df['new_label'].isna().sum():,} 条记录")
    
    # 详细分析重新标注的样本
    print("\n" + "="*70)
    print("重新标注样本的详细分析:")
    print("="*70)
    
    updated_df = df[df['was_updated'] == True]
    if len(updated_df) > 0:
        print(f"\n总共重新标注: {len(updated_df):,} 条记录\n")
        
        # 按原始BIRADS分组统计
        for orig_birads in track_birads:
            orig_updated = updated_df[updated_df['original_birads'] == orig_birads]
            if len(orig_updated) > 0:
                print(f"{birads_names[orig_birads]} 的重新标注:")
                
                # 统计标签变化
                label_dist = orig_updated['new_label'].value_counts().sort_index()
                for label, count in label_dist.items():
                    if not pd.isna(label):
                        percentage = (count / len(orig_updated)) * 100
                        avg_days = orig_updated[orig_updated['new_label'] == label]['days_to_diagnosis'].mean()
                        
                        print(f"  → {label_names[int(label)]}")
                        print(f"     数量: {count:,} ({percentage:.2f}%)")
                        print(f"     平均确诊时间: {avg_days:.1f} 天")
                print()
    
    # 保存结果
    print("="*70)
    print(f"正在保存结果到: {output_csv}")
    
    # 选择要保存的列
    output_columns = [
        'empi_anon', 'acc_anon', 'study_date_anon',
        'original_birads', 'birads_updated', 'new_label',
        'was_updated', 'followup_count', 'days_to_diagnosis',
        'diagnosis_in_window'
    ]
    
    # 保留其他原始列
    for col in df.columns:
        if col not in output_columns:
            output_columns.append(col)
    
    # 去重并保存
    df[output_columns].to_csv(output_csv, index=False)
    print("✓ 处理完成!")
    
    # 返回统计信息
    stats = {
        'original_total': df_before,
        'final_total': len(df),
        'dropped_no_followup': dropped_count,
        'update_stats': update_stats,
        'all_transitions': all_transitions,
        'followup_window_days': followup_window_days
    }
    
    return stats


if __name__ == "__main__":
    # 设置输入输出路径
    input_csv = "/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_clinical.csv"
    output_csv = "EMBED_OpenData_clinical_relabeled.csv"
    
    # 设置时间窗口（天）
    followup_window = 365
    
    try:
        stats = process_birads_labels(input_csv, output_csv, followup_window_days=followup_window)
        
        # 打印最终总结
        print("\n" + "="*70)
        print("处理总结:")
        print("="*70)
        print(f"原始总记录数: {stats['original_total']:,}")
        print(f"最终记录数: {stats['final_total']:,}")
        print(f"丢弃记录数: {stats['dropped_no_followup']:,}")
        print(f"随访时间窗口: {stats['followup_window_days']} 天")
        
        print(f"\n重新标注统计:")
        total_updated = sum(s['updated'] for s in stats['update_stats'].values())
        print(f"  总计重新标注: {total_updated:,} 条记录")
        
        for birads_code, s in stats['update_stats'].items():
            birads_name = {'A': 'BIRADS 0', 'N': 'BIRADS 1', 'B': 'BIRADS 2'}[birads_code]
            print(f"  {birads_name}: {s['updated']:,} / {s['total']:,} ({s['updated']/max(s['total'],1)*100:.1f}%)")
        
        print(f"\n输出文件已保存到:")
        print(f"  {output_csv}")
        
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)