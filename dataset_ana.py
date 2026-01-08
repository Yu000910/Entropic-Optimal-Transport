import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 数据路径
data_path = '/Users/nnn/Desktop/temp/博士毕业/第五篇/verb-tool-project/datasets/D_BEDR.npz'

def analyze_bedr_dataset(path):
    print(f"--- 正在加载数据集: {path} ---")
    try:
        data = np.load(path, allow_pickle=True)
        print(f"包含的 Keys: {list(data.keys())}")
        
        # 假设常见的 Key 名称，根据实际情况调整
        # 通常是 'x', 'y' 或者 'sentences', 'labels'
        if 'sentences' in data and 'labels' in data:
            X = data['sentences']
            y = data['labels']
        elif 'x' in data and 'y' in data:
            X = data['x']
            y = data['y']
        else:
            print("未找到预期的 keys (sentences/labels 或 x/y)，请检查打印的 Keys 列表。")
            return

        print(f"\n样本总数: {len(X)}")
        print(f"示例样本 1: {X[0]} -> Label: {y[0]}")
        print(f"示例样本 2: {X[10]} -> Label: {y[10]}")

        # 1. 类别分布分析
        label_counts = Counter(y)
        df_labels = pd.DataFrame.from_dict(label_counts, orient='index', columns=['count'])
        df_labels = df_labels.sort_values(by='count', ascending=False)
        
        print(f"\n覆盖技术类别数: {len(df_labels)}")
        print(f"Top 5 高频技术:\n{df_labels.head(5)}")
        print(f"Tail 5 低频技术:\n{df_labels.tail(5)}")
        
        # 2. 文本长度分析
        lengths = [len(str(t).split()) for t in X]
        print(f"\n文本长度统计 (词数):")
        print(f"平均长度: {np.mean(lengths):.2f}")
        print(f"最大长度: {np.max(lengths)}")
        print(f"最小长度: {np.min(lengths)}")

        # 3. 适宜性评估建议
        print("\n--- 原型库适宜性评估 ---")
        if len(df_labels) > 100:
            print("[√] 覆盖度良好：包含较多技术类别。")
        else:
            print("[!] 覆盖度预警：技术类别较少，可能无法覆盖多样化的 CTI 报告。")
            
        if np.mean(lengths) < 5:
            print("[!] 信息量预警：句子过短，可能缺乏语义上下文。")
        else:
            print("[√] 信息量良好：句子长度适中。")
            
    except Exception as e:
        print(f"加载或分析出错: {e}")

if __name__ == "__main__":
    analyze_bedr_dataset(data_path)