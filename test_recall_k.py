import os
import glob
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from utils_llm import deepseek_extract_attack_graph
from utils_embedding import embed_sentences

# --- 配置路径 ---
BEDR_CSV_PATH = 'graph_deep_attackg/BEDR_resampled_dataset.csv'
EMBED_CACHE_PATH = 'bedr_embeddings_cache.npy'
REPORT_DIR = 'generated_reports' # 请确认路径
REPORT_FILES = sorted(glob.glob(os.path.join(REPORT_DIR, '*.json')))[:5] # 只测5篇

# --- 复用之前的类和函数 ---
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

def compute_sinkhorn_plan(cost_matrix, reg=50, max_iter=50):
    n, m = cost_matrix.shape
    K = np.exp(-reg * cost_matrix)
    K /= K.sum()
    r = np.ones(n) / n
    c = np.ones(m) / m
    u = np.zeros(n)
    for _ in range(max_iter):
        u = K.sum(1)
        K *= (r / (u + 1e-9)).reshape((-1, 1))
        K *= (c / (K.sum(0) + 1e-9)).reshape((1, -1))
    return K

def load_data():
    try:
        df = pd.read_csv(BEDR_CSV_PATH, header=None, usecols=[0, 1], names=['label', 'text'], quotechar='"')
        labels = df['label'].astype(str).tolist()
        texts = df['text'].astype(str).tolist()
    except: return None, None, None
    if os.path.exists(EMBED_CACHE_PATH):
        embs = np.load(EMBED_CACHE_PATH)
    else:
        embs = embed_sentences(texts)
    return embs, labels, texts

def run_recall_test():
    proto_embs, proto_labels, _ = load_data()
    if proto_embs is None: return

    denoiser = ManifoldDenoiser(n_components=1)
    denoiser.fit()
    proto_embs_clean = denoiser.transform(proto_embs)

    r10_list = []
    r30_list = []

    print(f"正在测试 5 篇报告的 Recall@K (Stage 1 Performance)...")

    for report_path in REPORT_FILES:
        with open(report_path, 'r') as f: data = json.load(f)
        gt_techniques = set(data.get('actual_found_ids', []))
        text = data.get('clean_text', '')
        if not gt_techniques or not text: continue
        
        # 1. 抽取
        actions = deepseek_extract_attack_graph(text)
        if not actions: continue
        
        # 2. 向量化 & 去噪
        action_embs = embed_sentences(actions)
        action_embs_clean = denoiser.transform(action_embs)
        
        # 3. 局部候选池 (为了加速 OT)
        full_sim = cosine_similarity(action_embs_clean, proto_embs_clean)
        pool_indices = set()
        for i in range(len(actions)):
            pool_indices.update(np.argsort(full_sim[i])[::-1][:50]) # 扩大池子以测试 R@30
        pool_indices = list(pool_indices)
        pool_embs = proto_embs_clean[pool_indices]
        pool_labels = [proto_labels[idx] for idx in pool_indices]
        
        # 4. OT 计算
        sub_sim = cosine_similarity(action_embs_clean, pool_embs)
        cost_matrix = 1.0 - sub_sim
        T_matrix = compute_sinkhorn_plan(cost_matrix, reg=50)
        
        # 5. 计算 Recall@K
        # 定义：整个报告检索回来的 Top-K 集合（所有动作的 Top-K 并集）覆盖了多少 GT
        
        # 获取 Top-10 集合
        retrieved_10 = set()
        for i in range(len(actions)):
            top_k_idx = np.argsort(T_matrix[i])[::-1][:10]
            for idx in top_k_idx: retrieved_10.add(pool_labels[idx])
            
        # 获取 Top-30 集合
        retrieved_30 = set()
        for i in range(len(actions)):
            top_k_idx = np.argsort(T_matrix[i])[::-1][:30]
            for idx in top_k_idx: retrieved_30.add(pool_labels[idx])

        # 计算重叠率
        recall_10 = len(gt_techniques.intersection(retrieved_10)) / len(gt_techniques)
        recall_30 = len(gt_techniques.intersection(retrieved_30)) / len(gt_techniques)
        
        r10_list.append(recall_10)
        r30_list.append(recall_30)
        
        print(f"Report {os.path.basename(report_path)} -> R@10: {recall_10:.2f}, R@30: {recall_30:.2f}")

    print("-" * 30)
    print(f"Avg Recall@10: {np.mean(r10_list):.4f}")
    print(f"Avg Recall@30: {np.mean(r30_list):.4f}")

if __name__ == "__main__":
    run_recall_test()