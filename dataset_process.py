#!/usr/bin/env python3
"""
处理BIRADS评分的脚本（优化版）
对于BIRADS=0的记录，查找指定时间窗口内的后续记录并取最大BIRADS评分
然后将评分映射为三分类标签
丢弃无有效随访的BIRADS=0记录
"""

import pandas as pd
import numpy as np
from collections import Counter
import sys
from datetime import timedelta

def process_birads_labels(input_csv, output_csv, followup_window_days=365):
    """
    处理BIRADS评分标签
    
    Parameters:
    -----------
    input_csv : str
        输入CSV文件路径
    output_csv : str
        输出CSV文件路径
    followup_window_days : int
        随访时间窗口（天），默认365天
        只考虑此窗口内的后续记录来更新BIRADS 0
    """
    
    # 读取数据
    print(f"正在读取数据: {input_csv}")
    df = pd.read_csv(input_csv)
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
    df['followup_count'] = 0  # 随访次数
    df['days_to_diagnosis'] = np.nan  # 确诊时间（天）
    df['diagnosis_in_window'] = False  # 是否在时间窗口内有随访
    
    # 处理BIRADS=0的记录
    print(f"\n正在处理BIRADS=0的记录（时间窗口: {followup_window_days}天）...")
    birads_0_indices = df[df['asses'] == 'A'].index
    print(f"发现 {len(birads_0_indices):,} 条BIRADS=0的记录")
    
    updated_count = 0
    no_followup_count = 0
    
    # 用于统计更新后的BIRADS分布
    birads_0_transitions = []
    
    for idx in birads_0_indices:
        current_patient = df.loc[idx, 'empi_anon']
        current_date = df.loc[idx, 'study_date_anon']
        
        # 查找同一患者的后续记录
        same_patient_mask = (df['empi_anon'] == current_patient)
        future_records_mask = (df['study_date_anon'] > current_date)
        
        # 添加时间窗口限制
        window_end_date = current_date + timedelta(days=followup_window_days)
        within_window_mask = (df['study_date_anon'] <= window_end_date)
        
        followup_mask = same_patient_mask & future_records_mask & within_window_mask
        followup_records = df[followup_mask]
        
        if len(followup_records) > 0:
            # 获取后续所有记录的BIRADS评分
            followup_birads = followup_records['asses'].values
            
            # 定义BIRADS评分的优先级（数值越大优先级越高）
            birads_priority = {
                'N': 1,  # BIRADS 1
                'B': 2,  # BIRADS 2
                'P': 3,  # BIRADS 3
                'S': 4,  # BIRADS 4
                'M': 5,  # BIRADS 5
                'K': 6,  # BIRADS 6 (已知恶性)
                'A': 0   # BIRADS 0
            }
            
            # 找到优先级最高（数值最大）的BIRADS评分
            max_priority = 0
            max_birads = 'A'
            days_to_max = np.nan
            
            for i, birads in enumerate(followup_birads):
                if pd.notna(birads) and birads in birads_priority:
                    priority = birads_priority[birads]
                    if priority > max_priority:
                        max_priority = priority
                        max_birads = birads
                        # 计算到此次诊断的天数
                        days_to_max = (followup_records.iloc[i]['study_date_anon'] - current_date).days
            
            # 更新BIRADS评分和诊断信息
            df.loc[idx, 'birads_updated'] = max_birads
            df.loc[idx, 'followup_count'] = len(followup_records)
            df.loc[idx, 'days_to_diagnosis'] = days_to_max
            df.loc[idx, 'diagnosis_in_window'] = True
            
            updated_count += 1
            birads_0_transitions.append((current_patient, 'A', max_birads, days_to_max))
        else:
            # 没有后续记录，保持原值
            no_followup_count += 1
    
    print(f"✓ 已更新: {updated_count:,} 条记录")
    print(f"✗ 无有效随访（将被丢弃）: {no_followup_count:,} 条记录")
    
    # 显示BIRADS 0的转换统计
    if birads_0_transitions:
        print("\n" + "="*70)
        print("BIRADS 0 转换统计:")
        print("="*70)
        transitions_df = pd.DataFrame(
            birads_0_transitions, 
            columns=['patient', 'original', 'updated', 'days']
        )
        transition_counts = transitions_df['updated'].value_counts().sort_index()
        for birads, count in transition_counts.items():
            birads_name = birads_names.get(birads, f'Unknown ({birads})')
            percentage = (count / len(transitions_df)) * 100
            avg_days = transitions_df[transitions_df['updated'] == birads]['days'].mean()
            print(f"A → {birads} ({birads_name}): {count:,} ({percentage:.2f}%), "
                  f"平均 {avg_days:.1f} 天确诊")
    
    # 统计更新后的BIRADS分布（更新前）
    print("\n" + "="*70)
    print("更新后的BIRADS评分分布（丢弃前）:")
    print("="*70)
    updated_birads_counts = df['birads_updated'].value_counts().sort_index()
    for birads, count in updated_birads_counts.items():
        birads_name = birads_names.get(birads, f'Unknown ({birads})')
        percentage = (count / len(df)) * 100
        print(f"{birads_name}: {count:,} 条记录 ({percentage:.2f}%)")
    
    # 丢弃仍然是 'A' 的记录（无有效随访）
    print("\n" + "="*70)
    print("正在丢弃无有效随访的BIRADS=0记录...")
    print("="*70)
    df_before = len(df)
    df = df[df['birads_updated'] != 'A'].copy()
    df_after = len(df)
    dropped_count = df_before - df_after
    print(f"已丢弃: {dropped_count:,} 条记录")
    print(f"剩余记录: {df_after:,} 条")
    
    # 映射BIRADS评分到三分类标签
    # 0: N (BIRADS 1), B (BIRADS 2) - 阴性/良性（假阳性召回）
    # 1: P (BIRADS 3) - 可能良性，需要短期随访
    # 2: S (BIRADS 4), M (BIRADS 5), K (BIRADS 6) - 可疑/恶性（真阳性召回）
    
    def map_birads_to_label(birads):
        """映射BIRADS评分到三分类标签"""
        if pd.isna(birads):
            return np.nan
        
        birads = str(birads).strip()
        
        if birads in ['N', 'B']:
            return 0  # 阴性/良性（对于原BIRADS 0：假阳性召回）
        elif birads == 'P':
            return 1  # 可能良性（需要随访）
        elif birads in ['S', 'M', 'K']:
            return 2  # 可疑/恶性（对于原BIRADS 0：真阳性召回）
        else:
            return np.nan
    
    df['new_label'] = df['birads_updated'].apply(map_birads_to_label)
    
    # 添加额外的标注列：区分原始BIRADS 0和其他
    df['was_birads_0'] = (df['asses'] == 'A').astype(int)
    
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
    
    total_valid = 0
    for label in sorted(label_counts.index):
        if not pd.isna(label):
            count = label_counts[label]
            percentage = (count / len(df)) * 100
            
            # 进一步统计原BIRADS 0的分布
            was_0_count = df[(df['new_label'] == label) & (df['was_birads_0'] == 1)].shape[0]
            
            print(f"\n{label_names[int(label)]}:")
            print(f"  总计: {count:,} 条记录 ({percentage:.2f}%)")
            print(f"  其中原BIRADS 0: {was_0_count:,} 条")
            total_valid += count
    
    if df['new_label'].isna().sum() > 0:
        print(f"\n缺失值: {df['new_label'].isna().sum():,} 条记录")
    
    # 详细统计原BIRADS 0的标签分布
    print("\n" + "="*70)
    print("原BIRADS=0记录的重标注分布:")
    print("="*70)
    birads_0_df = df[df['was_birads_0'] == 1]
    if len(birads_0_df) > 0:
        birads_0_label_dist = birads_0_df['new_label'].value_counts().sort_index()
        for label, count in birads_0_label_dist.items():
            if not pd.isna(label):
                percentage = (count / len(birads_0_df)) * 100
                avg_days = birads_0_df[birads_0_df['new_label'] == label]['days_to_diagnosis'].mean()
                avg_followups = birads_0_df[birads_0_df['new_label'] == label]['followup_count'].mean()
                
                print(f"{label_names[int(label)]}:")
                print(f"  数量: {count:,} ({percentage:.2f}%)")
                print(f"  平均确诊时间: {avg_days:.1f} 天")
                print(f"  平均随访次数: {avg_followups:.1f} 次")
                print()
    
    # 保存结果
    print("="*70)
    print(f"正在保存结果到: {output_csv}")
    df.to_csv(output_csv, index=False)
    print("✓ 处理完成!")
    
    # 返回统计信息
    stats = {
        'original_total': df_before,
        'final_total': len(df),
        'dropped_no_followup': dropped_count,
        'original_birads': dict(original_birads_counts),
        'updated_birads': dict(updated_birads_counts),
        'new_labels': dict(label_counts),
        'updated_count': updated_count,
        'no_followup_count': no_followup_count,
        'followup_window_days': followup_window_days
    }
    
    return stats


if __name__ == "__main__":
    # 设置输入输出路径
    input_csv = "/projects/standard/lin01231/public/datasets/embed/tables/EMBED_OpenData_clinical.csv"
    output_csv = "EMBED_OpenData_clinical_relabeled_new.csv"
    
    # 设置时间窗口（天）
    # 可以根据需要调整：180天(6个月), 365天(1年), 730天(2年)
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
        print(f"\nBIRADS=0 处理:")
        print(f"  已更新（有随访）: {stats['updated_count']:,}")
        print(f"  已丢弃（无随访）: {stats['no_followup_count']:,}")
        print(f"\n输出文件已保存到:")
        print(f"  {output_csv}")
        
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)