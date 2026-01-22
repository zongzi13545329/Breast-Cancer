#!/usr/bin/env python3
"""
筛选Clinical CSV中side不为R和L的记录
"""

import pandas as pd
import numpy as np

def check_side_column(clinical_csv):
    """
    检查Clinical CSV中的side列，输出不为R和L的记录
    
    Parameters:
    -----------
    clinical_csv : str
        Clinical CSV文件路径
    """
    print("="*70)
    print("检查Clinical CSV的side列")
    print("="*70)
    
    # 读取数据
    print(f"\n正在读取: {clinical_csv}")
    df = pd.read_csv(clinical_csv, low_memory=False)
    print(f"总记录数: {len(df):,}")
    
    # 检查side列是否存在
    if 'side' not in df.columns:
        print("\n⚠️  错误: CSV中没有'side'列!")
        print(f"可用的列: {df.columns.tolist()}")
        return
    
    # 统计side列的分布
    print("\n" + "="*70)
    print("Side列的完整分布:")
    print("="*70)
    side_counts = df['side'].value_counts(dropna=False)
    total = len(df)
    
    for side_value, count in side_counts.items():
        percentage = (count / total) * 100
        if pd.isna(side_value):
            print(f"  NaN (缺失值): {count:,} ({percentage:.2f}%)")
        else:
            print(f"  '{side_value}': {count:,} ({percentage:.2f}%)")
    
    # 筛选不为R和L的记录
    print("\n" + "="*70)
    print("筛选side不为'R'和'L'的记录:")
    print("="*70)
    
    # 条件：side != 'R' AND side != 'L'
    # 注意：这会包括NaN值
    filtered_df = df[(df['side'] != 'R') & (df['side'] != 'L')]
    
    print(f"\n找到 {len(filtered_df):,} 条记录不是R或L")
    
    if len(filtered_df) > 0:
        # 显示这些记录的side值分布
        print("\n这些记录的side值分布:")
        filtered_side_counts = filtered_df['side'].value_counts(dropna=False)
        for side_value, count in filtered_side_counts.items():
            percentage = (count / len(filtered_df)) * 100
            if pd.isna(side_value):
                print(f"  NaN (缺失值): {count:,} ({percentage:.2f}%)")
            else:
                print(f"  '{side_value}': {count:,} ({percentage:.2f}%)")
        
        # 显示前20条记录的关键信息
        print("\n" + "="*70)
        print("前20条记录的详细信息:")
        print("="*70)
        
        # 选择要显示的列
        display_cols = ['empi_anon', 'acc_anon', 'study_date_anon', 'side', 
                       'asses', 'new_label']
        # 只保留存在的列
        display_cols = [col for col in display_cols if col in filtered_df.columns]
        
        print("\n" + filtered_df[display_cols].head(20).to_string(index=False))
        
        # 保存完整结果
        output_csv = "side_not_R_or_L.csv"
        filtered_df.to_csv(output_csv, index=False)
        print(f"\n✓ 已保存完整结果到: {output_csv}")
        
        # 额外分析：这些记录的其他特征
        print("\n" + "="*70)
        print("这些记录的其他统计信息:")
        print("="*70)
        
        if 'new_label' in filtered_df.columns:
            print("\nLabel分布:")
            label_counts = filtered_df['new_label'].value_counts(dropna=False)
            for label, count in label_counts.items():
                percentage = (count / len(filtered_df)) * 100
                if pd.isna(label):
                    print(f"  NaN: {count:,} ({percentage:.2f}%)")
                else:
                    print(f"  Label {int(label)}: {count:,} ({percentage:.2f}%)")
        
        if 'asses' in filtered_df.columns:
            print("\nBIRADS分布:")
            birads_counts = filtered_df['asses'].value_counts(dropna=False)
            for birads, count in birads_counts.items():
                percentage = (count / len(filtered_df)) * 100
                print(f"  '{birads}': {count:,} ({percentage:.2f}%)")
        
        # 检查这些记录是否有对应的metadata
        print("\n" + "="*70)
        print("检查这些exam在metadata中的图像情况:")
        print("="*70)
        
        # 这里只是示例，如果需要可以加载metadata进行匹配
        unique_exams = filtered_df['acc_anon'].nunique()
        print(f"涉及 {unique_exams:,} 个唯一的exam")
        
    else:
        print("\n✓ 所有记录的side都是'R'或'L'")
    
    # 额外检查：R和L的分布是否合理
    print("\n" + "="*70)
    print("R和L的分布检查:")
    print("="*70)
    
    r_count = (df['side'] == 'R').sum()
    l_count = (df['side'] == 'L').sum()
    
    print(f"side='R': {r_count:,} ({r_count/total*100:.2f}%)")
    print(f"side='L': {l_count:,} ({l_count/total*100:.2f}%)")
    
    if r_count > 0 and l_count > 0:
        ratio = max(r_count, l_count) / min(r_count, l_count)
        print(f"R/L比例: {ratio:.2f}:1")
        
        if ratio > 2:
            print("⚠️  警告: R和L的数量差异较大，可能存在数据问题")


if __name__ == "__main__":
    import sys
    
    # 设置输入路径
    clinical_csv = "EMBED_OpenData_clinical_relabeled_new.csv"
    
    # 如果命令行提供了参数，使用命令行参数
    if len(sys.argv) > 1:
        clinical_csv = sys.argv[1]
    
    try:
        check_side_column(clinical_csv)
        print("\n" + "="*70)
        print("✓ 检查完成!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)