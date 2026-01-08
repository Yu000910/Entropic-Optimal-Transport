import os
import glob
import json
import pandas as pd
import numpy as np
from collections import Counter

# --- 配置路径 ---
# 请确保这些路径与你的实际文件位置一致
BEDR_PATH = 'graph_deep_attackg/BEDR_resampled_dataset.csv'
REPORT_DIR = 'generated_reports' 

def analyze_bedr(path):
    print("="*60)
    print(f"1. BEDR 知识库数据集分析")
    print(f"   路径: {path}")
    print("="*60)
    
    if not os.path.exists(path):
        print(f"[Error] 找不到文件: {path}")
        return None, None

    try:
        # 读取 CSV，假设没有表头，前两列分别是 ID 和 Description
        # 使用 quotechar='"' 处理描述中可能包含的逗号
        df = pd.read_csv(path, header=None, usecols=[0, 1], names=['label', 'text'], quotechar='"')
        
        # 清洗数据：去除标签和文本的前后空格
        df['label'] = df['label'].astype(str).str.strip()
        df['text'] = df['text'].astype(str).str.strip()
        
        # 1. 基础计数
        total_samples = len(df)
        unique_classes = df['label'].nunique()
        
        # 2. 类别分布统计
        class_counts = df['label'].value_counts()
        max_samples = class_counts.max()
        min_samples = class_counts.min()
        avg_samples = class_counts.mean()
        std_samples = class_counts.std()
        
        # 3. 文本长度统计 (按单词数)
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        avg_text_len = df['word_count'].mean()
        max_text_len = df['word_count'].max()
        min_text_len = df['word_count'].min()
        
        print(f"Total Prototype Samples (总样本数): {total_samples}")
        print(f"Total Unique Techniques (类别总数): {unique_classes}")
        print(f"-"*30)
        print(f"Samples per Class (每类样本分布):")
        print(f"  - Mean (平均): {avg_samples:.2f}")
        print(f"  - Min  (最小): {min_samples}")
        print(f"  - Max  (最大): {max_samples}")
        print(f"  - Std  (标准差): {std_samples:.2f} (反映类别不平衡程度)")
        print(f"-"*30)
        print(f"Text Length (Words) (文本长度分布):")
        print(f"  - Mean: {avg_text_len:.2f}")
        print(f"  - Max:  {max_text_len}")
        print(f"  - Min:  {min_text_len}")
        
        # 返回 unique 标签集合，用于后续交叉分析
        return set(df['label'].unique()), class_counts
        
    except Exception as e:
        print(f"[Error] 读取 BEDR 失败: {e}")
        return None, None

def analyze_cti_reports(dir_path):
    print("\n" + "="*60)
    print(f"2. CTI 报告数据集分析 (测试集)")
    print(f"   路径: {dir_path}")
    print("="*60)
    
    if not os.path.exists(dir_path):
        print(f"[Error] 找不到目录: {dir_path}")
        return None

    json_files = glob.glob(os.path.join(dir_path, "*.json"))
    total_reports = len(json_files)
    
    if total_reports == 0:
        print("[Warning] 该目录下没有找到 .json 文件。")
        return None

    all_ground_truth_labels = [] # 存储所有报告的所有标签
    report_lengths = []          # 存储每篇报告的长度
    labels_per_report = []       # 存储每篇报告的标签数量
    
    valid_count = 0
    
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取真实标签 (actual_found_ids)
            # 这里的逻辑是：actual_found_ids 是经过验证的 Ground Truth
            gt = data.get('actual_found_ids', [])
            if not isinstance(gt, list): continue
            
            # 清洗标签 (去除空白)
            gt = [str(t).strip() for t in gt]
            
            # 提取文本 (clean_text)
            text = data.get('clean_text', '')
            
            # 记录数据
            labels_per_report.append(len(gt))
            all_ground_truth_labels.extend(gt)
            report_lengths.append(len(text.split()))
            valid_count += 1
            
        except Exception as e:
            print(f"解析文件 {os.path.basename(jf)} 失败: {e}")
            
    if valid_count == 0:
        print("没有有效的报告数据。")
        return None

    # --- 统计指标 ---
    unique_techs_in_reports = set(all_ground_truth_labels)
    num_unique_techs = len(unique_techs_in_reports)
    
    # 1. Label Cardinality (LCard): 平均每篇报告的标签数
    lcard = np.mean(labels_per_report)
    
    # 2. Label Density (LDens): LCard / 总标签空间大小 (这里用测试集出现的总标签数作为分母)
    # 注意：在论文中，分母通常是整个知识库的大小(BEDR unique classes)，我们在交叉分析里算
    
    print(f"Total Reports (报告总数): {valid_count}")
    print(f"Total Technique Instances (标签总实例数): {len(all_ground_truth_labels)}")
    print(f"Unique Techniques Covered (覆盖技术类别数): {num_unique_techs}")
    print(f"-"*30)
    print(f"Multi-label Statistics (多标签难度指标):")
    print(f"  - Label Cardinality (Avg labels/report): {lcard:.2f}")
    print(f"  - Max Labels in a Report: {np.max(labels_per_report)}")
    print(f"  - Min Labels in a Report: {np.min(labels_per_report)}")
    print(f"-"*30)
    print(f"Report Length (Words):")
    print(f"  - Mean: {np.mean(report_lengths):.2f}")
    print(f"  - Max:  {np.max(report_lengths)}")
    print(f"  - Min:  {np.min(report_lengths)}")

    return unique_techs_in_reports

def cross_analysis(bedr_techs, report_techs):
    print("\n" + "="*60)
    print(f"3. 交叉分析 (Cross Analysis)")
    print("="*60)
    
    if bedr_techs is None or report_techs is None:
        return

    # 计算交集
    intersection = bedr_techs.intersection(report_techs)
    missing_in_bedr = report_techs - bedr_techs
    
    bedr_count = len(bedr_techs)
    report_count = len(report_techs)
    overlap_count = len(intersection)
    
    # 覆盖率 (Coverage)
    coverage = (overlap_count / bedr_count * 100) if bedr_count > 0 else 0
    
    # 未见过的技术 (Zero-shot test for unseen classes? 或者单纯的数据不一致)
    unseen_count = len(missing_in_bedr)
    
    print(f"Knowledge Base Size (BEDR): {bedr_count}")
    print(f"Test Set Vocabulary Size: {report_count}")
    print(f"Overlapping Techniques: {overlap_count}")
    print(f"Coverage of Knowledge Base: {coverage:.2f}%")
    
    if unseen_count > 0:
        print(f"[Info] 有 {unseen_count} 个技术出现在报告中但不在 BEDR 库里。")
        print(f"这可能是因为 BEDR 是重采样过的子集。")
    
    # 计算 Label Density (基于完整知识库)
    # Label Density = Label Cardinality / Total Labels in KB
    # 这是一个非常小的数字，但能反映标签空间的稀疏性
    # 我们需要从上面的函数传参过来，这里简单估算一下 LCard 无妨
    # 为了严谨，建议在写论文时直接用 (Avg labels per report) / (BEDR Size)

if __name__ == "__main__":
    # 1. 分析 BEDR
    bedr_techs, _ = analyze_bedr(BEDR_PATH)
    
    # 2. 分析 Reports
    report_techs = analyze_cti_reports(REPORT_DIR)
    
    # 3. 交叉分析
    cross_analysis(bedr_techs, report_techs)