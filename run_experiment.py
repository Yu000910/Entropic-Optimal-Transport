import os
import glob
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 导入工具模块
from utils_llm import deepseek_extract_attack_graph
from utils_embedding import embed_sentences

# --- 配置路径 ---
# 1. CSV 数据集路径
BEDR_CSV_PATH = 'BEDR_dataset.csv'
# 2. 向量缓存保存路径 (生成一次后，下次直接读这个)
EMBED_CACHE_PATH = 'bedr_embeddings_cache.npy'
# 3. CTI 报告目录
REPORT_DIR = 'CTI_reports'
# 4. 选取前 5 篇报告进行测试
REPORT_FILES = sorted(glob.glob(os.path.join(REPORT_DIR, '*.json')))[:5]

def load_prototypes_from_csv(csv_path, cache_path):
    """
    加载 CSV 数据集，并处理嵌入向量的生成与缓存
    """
    print(f"1. [原型准备] 读取 CSV: {csv_path}")
    
    # 读取 CSV (假设没有 header，第一列是 ID，第二列是 Text)
    # 格式示例: T1213,"Text...",original_retained
    try:
        df = pd.read_csv(csv_path, header=None, names=['label', 'text', 'metadata'])
    except Exception:
        # 如果只有两列或读取失败，尝试更通用的读取
        df = pd.read_csv(csv_path, header=None)
        df.columns = ['label', 'text'] + [f'col_{i}' for i in range(2, df.shape[1])]
    
    # 提取标签和文本
    y_labels = df.iloc[:, 0].astype(str).tolist() # 第一列
    X_text = df.iloc[:, 1].astype(str).tolist()   # 第二列
    
    print(f"   原型库样本数: {len(X_text)}")
    print(f"   示例数据: {y_labels[0]} -> {X_text[0][:50]}...")

    # --- 向量缓存逻辑 ---
    if os.path.exists(cache_path):
        print(f"   [缓存命中] 发现本地向量文件: {cache_path}")
        X_emb = np.load(cache_path)
        
        # 简单校验
        if len(X_emb) != len(X_text):
            print(f"   [警告] 缓存数量 ({len(X_emb)}) 与 CSV 行数 ({len(X_text)}) 不一致！")
            print("   正在重新生成向量...")
            X_emb = embed_sentences(X_text)
            np.save(cache_path, X_emb)
        else:
            print("   成功加载本地向量。")
    else:
        print("   [缓存未命中] 正在调用 API 生成原型向量 (这将花费一些时间)...")
        # 调用 utils_embedding 中的函数
        X_emb = embed_sentences(X_text)
        
        # 保存缓存
        np.save(cache_path, X_emb)
        print(f"   已将向量保存至: {cache_path}")
        
    return X_emb, y_labels

def calculate_metrics(gt_set, pred_set):
    """计算 Precision, Recall, F1"""
    tp = len(gt_set.intersection(pred_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    return p, r, f1

def run_experiment():
    # --- Step 1: 加载原型 (带缓存) ---
    proto_emb, proto_labels = load_prototypes_from_csv(BEDR_CSV_PATH, EMBED_CACHE_PATH)
    
    results = []
    
    print(f"\n2. [开始测试] 处理 {len(REPORT_FILES)} 篇报告...")
    
    for report_path in REPORT_FILES:
        file_name = os.path.basename(report_path)
        print(f"\n--- 正在处理: {file_name} ---")
        
        # --- Step 2: 读取报告 ---
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
        except Exception as e:
            print(f"   [错误] 读取失败: {e}")
            continue
            
        # 获取 GT 和 输入文本
        gt_techniques = set(report_data.get('actual_found_ids', []))
        # 优先使用 clean_text，防止标签泄露
        input_text = report_data.get('clean_text', '') 
        
        if not gt_techniques or not input_text:
            print("   [跳过] 缺少 Ground Truth 或 clean_text")
            continue
            
        # --- Step 3: LLM 抽取行为 ---
        print("   > (LLM) 抽取攻击行为描述...")
        extracted_actions = deepseek_extract_attack_graph(input_text)
        
        if not extracted_actions:
            print("   [失败] LLM 未提取到有效行为")
            results.append({'file': file_name, 'p': 0, 'r': 0, 'f1': 0})
            continue
        
        print(f"   > 提取到 {len(extracted_actions)} 条行为，正在向量化...")
        
        # --- Step 4: 行为向量化 ---
        action_embs = embed_sentences(extracted_actions)
        
        # --- Step 5: 相似度匹配 (Retrieval) ---
        # 计算相似度矩阵 (M_actions, N_prototypes)
        sim_matrix = cosine_similarity(action_embs, proto_emb)
        
        pred_techniques = set()
        SIM_THRESHOLD = 0.45  # 相似度阈值 (根据实验结果微调)
        
        for i in range(len(extracted_actions)):
            # 找到最相似的原型
            best_idx = np.argmax(sim_matrix[i])
            best_score = sim_matrix[i][best_idx]
            best_label = str(proto_labels[best_idx]) # 确保是字符串
            
            if best_score > SIM_THRESHOLD:
                pred_techniques.add(best_label)
                # print(f"     [Match] '{extracted_actions[i][:30]}...' -> {best_label} ({best_score:.2f})")
        
        # --- Step 6: 评估 ---
        p, r, f1 = calculate_metrics(gt_techniques, pred_techniques)
        
        print(f"   [预测] {list(pred_techniques)}")
        print(f"   [真实] {list(gt_techniques)}")
        print(f"   [指标] P: {p:.2f} | R: {r:.2f} | F1: {f1:.2f}")
        
        results.append({'file': file_name, 'p': p, 'r': r, 'f1': f1})

    # --- Step 7: 汇总 ---
    if results:
        avg_p = np.mean([x['p'] for x in results])
        avg_r = np.mean([x['r'] for x in results])
        avg_f1 = np.mean([x['f1'] for x in results])
        
        print("\n" + "="*40)
        print("FINAL RESULTS (AVG)")
        print("="*40)
        print(f"Precision : {avg_p:.4f}")
        print(f"Recall    : {avg_r:.4f}")
        print(f"F1 Score  : {avg_f1:.4f}")
        print("="*40)
    else:
        print("\n未产生有效结果。")

if __name__ == "__main__":
    # 确保 pandas 已安装: pip install pandas
    run_experiment()