import os
import glob
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# 导入新的选择函数
from utils_llm import deepseek_extract_attack_graph, deepseek_contrastive_select, client
from utils_embedding import embed_sentences

# 配置 (保持不变)
BEDR_CSV_PATH = 'graph_deep_attackg/BEDR_resampled_dataset.csv'
EMBED_CACHE_PATH = 'bedr_embeddings_cache.npy'
REPORT_DIR = '/Users/nnn/Desktop/temp/博士毕业/第六篇/elsarticle/experiment/generated_reports/'
REPORT_FILES = sorted(glob.glob(os.path.join(REPORT_DIR, '*.json')))[:200]

class ManifoldDenoiser:
    def __init__(self, n_components=1):
        self.pca = PCA(n_components=n_components)
        self.generic_terms = ["cyber attack", "malware", "system", "network", "hacker", "execute", "run", "file", "data"]
        self.noise_basis = None
    def fit(self):
        self.pca.fit(embed_sentences(self.generic_terms))
        self.noise_basis = self.pca.components_
    def transform(self, embeddings):
        if self.noise_basis is None: self.fit()
        return embeddings - np.dot(np.dot(embeddings, self.noise_basis.T), self.noise_basis)

def load_data_and_prototypes():
    print(f"1. 加载 BEDR 原型: {BEDR_CSV_PATH}")
    try:
        df = pd.read_csv(BEDR_CSV_PATH, header=None)
        y_labels = df.iloc[:, 0].astype(str).tolist()
        x_texts = df.iloc[:, 1].astype(str).tolist()
    except: return None, None, None
    
    if os.path.exists(EMBED_CACHE_PATH):
        x_embs = np.load(EMBED_CACHE_PATH)
    else:
        x_embs = embed_sentences(x_texts)
        np.save(EMBED_CACHE_PATH, x_embs)
    return x_embs, y_labels, x_texts

def run_experiment():
    proto_embs, proto_labels, proto_texts = load_data_and_prototypes()
    if proto_embs is None: return

    # Innovation 1: 去噪
    denoiser = ManifoldDenoiser(n_components=1)
    denoiser.fit()
    proto_embs_clean = denoiser.transform(proto_embs)

    results = []
    print(f"\n2. [开始 Contrastive Neuro-Symbolic 测试] 处理 {len(REPORT_FILES)} 篇报告...")

    for report_path in REPORT_FILES:
        file_name = os.path.basename(report_path)
        print(f"\n--- 处理: {file_name} ---")
        
        with open(report_path, 'r') as f: data = json.load(f)
        gt_techniques = set(data.get('actual_found_ids', []))
        text = data.get('clean_text', '')
        if not gt_techniques or not text: continue
        
        # Step 1: 抽取
        actions = deepseek_extract_attack_graph(text)
        if not actions: continue
        
        # Step 2: 向量化 & 去噪
        action_embs = embed_sentences(actions)
        action_embs_clean = denoiser.transform(action_embs)
        
        # Step 3: 几何检索
        sim_matrix = cosine_similarity(action_embs_clean, proto_embs_clean)
        
        pred_techniques = set()
        
        # Step 4: 对比推理重排序 (Innovation 5)
        for i, action_desc in enumerate(actions):
            # 获取 Top-5 候选 (增加候选数量，给 LLM 更多选择)
            top_k_indices = np.argsort(sim_matrix[i])[::-1][:5]
            
            # 只有当 Top-1 分数够高时才处理，否则视为噪声
            if sim_matrix[i][top_k_indices[0]] < 0.35:
                continue

            candidates = []
            for idx in top_k_indices:
                candidates.append({
                    'id': proto_labels[idx],
                    'desc': proto_texts[idx],
                    'score': sim_matrix[i][idx]
                })
            
            # [关键] 调用对比推理
            # 这是一个"多选一"的过程
            best_id = deepseek_contrastive_select(action_desc, candidates)
            
            if best_id != "None":
                pred_techniques.add(best_id)
                # print(f"    Action: {action_desc[:30]}... -> Selected: {best_id}")

        # 评估
        tp = len(gt_techniques.intersection(pred_techniques))
        fp = len(pred_techniques - gt_techniques)
        fn = len(gt_techniques - pred_techniques)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        print(f"   [预测] {list(pred_techniques)}")
        print(f"   [真实] {list(gt_techniques)}")
        print(f"   [指标] P: {p:.2f} | R: {r:.2f} | F1: {f1:.2f}")
        results.append({'file': file_name, 'p': p, 'r': r, 'f1': f1})

    if results:
        avg_f1 = np.mean([x['f1'] for x in results])
        avg_p = np.mean([x['p'] for x in results])
        avg_r = np.mean([x['r'] for x in results])
        print(f"\n=== 平均 Precision: {avg_p:.4f} | Recall: {avg_r:.4f} | F1: {avg_f1:.4f} ===")

if __name__ == "__main__":
    run_experiment()