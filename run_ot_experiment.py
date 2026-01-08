import os
import glob
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# 引入你的工具模块
from utils_llm import deepseek_extract_attack_graph, deepseek_contrastive_select
from utils_embedding import embed_sentences

# --- 配置路径 ---
BEDR_CSV_PATH = 'graph_deep_attackg/BEDR_resampled_dataset.csv'
EMBED_CACHE_PATH = 'bedr_embeddings_cache.npy'
REPORT_DIR = '/Users/nnn/Desktop/temp/博士毕业/第六篇/elsarticle/experiment/generated_reports/'
REPORT_FILES = sorted(glob.glob(os.path.join(REPORT_DIR, '*.json')))[:200]

# ==============================================================================
# 模块一：流形去噪 (Innovation 2)
# ==============================================================================
class ManifoldDenoiser:
    def __init__(self, n_components=1):
        self.pca = PCA(n_components=n_components)
        # 定义通用噪声词
        self.generic_terms = ["cyber attack", "malware", "system", "network", "hacker", "execute", "run", "file", "data", "connection"]
        self.noise_basis = None

    def fit(self):
        # 计算噪声主成分
        noise_embs = embed_sentences(self.generic_terms)
        self.pca.fit(noise_embs)
        self.noise_basis = self.pca.components_

    def transform(self, embeddings):
        if self.noise_basis is None: self.fit()
        # 正交投影去噪
        return embeddings - np.dot(np.dot(embeddings, self.noise_basis.T), self.noise_basis)

# ==============================================================================
# 模块二：Sinkhorn 最优传输 (Innovation 3)
# ==============================================================================
def compute_sinkhorn_plan(cost_matrix, reg=50, max_iter=50, tol=1e-5):
    """
    计算最优传输方案 T。
    T[i, j] 表示第 i 个 Action 传输给第 j 个 Candidate 的概率质量。
    """
    n, m = cost_matrix.shape
    # 熵正则化核
    K = np.exp(-reg * cost_matrix)
    K /= K.sum()
    
    # 均匀边际分布
    r = np.ones(n) / n
    c = np.ones(m) / m
    
    u = np.zeros(n)
    # Sinkhorn 迭代
    for _ in range(max_iter):
        if np.max(np.abs(u - K.sum(1))) < tol: break
        u = K.sum(1)
        K *= (r / u).reshape((-1, 1))
        K *= (c / K.sum(0)).reshape((1, -1))
        
    return K

# ==============================================================================
# 主流程
# ==============================================================================
def load_data():
    print(f"1. 加载 BEDR 原型库...")
    try:
        df = pd.read_csv(BEDR_CSV_PATH, header=None)
        labels = df.iloc[:, 0].astype(str).tolist()
        texts = df.iloc[:, 1].astype(str).tolist()
    except: return None, None, None
    
    if os.path.exists(EMBED_CACHE_PATH):
        embs = np.load(EMBED_CACHE_PATH)
    else:
        embs = embed_sentences(texts)
        np.save(EMBED_CACHE_PATH, embs)
    return embs, labels, texts

def calculate_metrics(gt_set, pred_set):
    tp = len(gt_set.intersection(pred_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def run_experiment():
    # 1. 准备数据
    proto_embs, proto_labels, proto_texts = load_data()
    if proto_embs is None: return

    # 初始化去噪器
    denoiser = ManifoldDenoiser(n_components=1)
    denoiser.fit()
    proto_embs_clean = denoiser.transform(proto_embs)

    results = []
    print(f"\n2. [开始 Full Neuro-Symbolic (OT + Contrastive) 测试]...")

    for report_path in REPORT_FILES:
        file_name = os.path.basename(report_path)
        print(f"\n--- 处理: {file_name} ---")
        
        with open(report_path, 'r') as f: data = json.load(f)
        gt_techniques = set(data.get('actual_found_ids', []))
        text = data.get('clean_text', '')
        if not gt_techniques or not text: continue
        
        # --- Step 1: 语义解耦 (Innovation 1) ---
        actions = deepseek_extract_attack_graph(text)
        if not actions: continue
        # print(f"   > 原子行为数: {len(actions)}")
        
        # --- Step 2: 向量化与去噪 (Innovation 2) ---
        action_embs = embed_sentences(actions)
        action_embs_clean = denoiser.transform(action_embs)
        
        # --- Step 3: 构建 OT 局部环境 (Context Pool) ---
        # 因为全量 OT 太慢，我们先用 Cosine 选出 Top-30 候选集作为 "局部流形"
        full_sim = cosine_similarity(action_embs_clean, proto_embs_clean)
        
        pool_indices = set()
        for i in range(len(actions)):
            pool_indices.update(np.argsort(full_sim[i])[::-1][:30])
        
        pool_indices = list(pool_indices)
        pool_embs = proto_embs_clean[pool_indices]
        pool_labels = [proto_labels[idx] for idx in pool_indices]
        pool_texts = [proto_texts[idx] for idx in pool_indices]
        
        # --- Step 4: OT 集合对齐 (Innovation 3) ---
        # 计算局部代价矩阵 Cost = 1 - Cosine
        sub_sim = cosine_similarity(action_embs_clean, pool_embs)
        cost_matrix = 1.0 - sub_sim
        
        # 运行 Sinkhorn
        # T_matrix[i, j] 表示 Action[i] 匹配 Candidate[j] 的"结构化概率"
        T_matrix = compute_sinkhorn_plan(cost_matrix, reg=50)
        
        pred_techniques = set()
        
        # --- Step 5: 对比推理重排序 (Innovation 5) ---
        for i, action_desc in enumerate(actions):
            # 基于 OT 矩阵 T_matrix 选择 Top-5 候选
            # 注意：这里用的是 T 的权重，而不是原始 Cosine，体现了全局优化的结果
            top_k_local_indices = np.argsort(T_matrix[i])[::-1][:5]
            
            # 准备候选列表给 LLM
            candidates = []
            max_ot_weight = T_matrix[i][top_k_local_indices[0]]
            
            # 过滤：如果 OT 权重太分散(最大值也很小)，或者原始相似度太低，可能是噪声
            # 取对应的原始相似度
            best_raw_sim = sub_sim[i][top_k_local_indices[0]]
            
            if best_raw_sim < 0.35: # 保持一个基本的门槛
                continue

            for local_idx in top_k_local_indices:
                candidates.append({
                    'id': pool_labels[local_idx],
                    'desc': pool_texts[local_idx],
                    'score': T_matrix[i][local_idx] # 这里展示 OT Score
                })
            
            # 调用 LLM 做单选题
            best_id = deepseek_contrastive_select(action_desc, candidates)
            
            if best_id != "None":
                pred_techniques.add(best_id)

        # 评估
        p, r, f1 = calculate_metrics(gt_techniques, pred_techniques)
        print(f"   [预测] {list(pred_techniques)}")
        print(f"   [真实] {list(gt_techniques)}")
        print(f"   [指标] P: {p:.2f} | R: {r:.2f} | F1: {f1:.2f}")
        results.append({'p': p, 'r': r, 'f1': f1})

    if results:
        avg_f1 = np.mean([x['f1'] for x in results])
        avg_p = np.mean([x['p'] for x in results])
        avg_r = np.mean([x['r'] for x in results])
        print(f"\n=== 平均 Precision: {avg_p:.4f} | Recall: {avg_r:.4f} | F1: {avg_f1:.4f} ===")

if __name__ == "__main__":
    run_experiment()