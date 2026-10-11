#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baseline_techniquerag.py — 复现 TechniqueRAG (Findings of ACL 2025) 于本文数据集

原论文方法（TechniqueRAG: Retrieval Augmented Generation for Adversarial Technique
Annotation in Cyber Threat Intelligence Text, ACL 2025 Findings）：
  1. Retriever R（冻结）：BM25，取 top-K 文本-技术对（原论文 K=40）
  2. Re-ranker（冻结，零样本 LLM）：对检索到的候选做显式推理重排
     —— 三步推理：拆解攻击步骤 → 映射到 tactics → 链接到 techniques
     —— 批大小 40、重叠 20（滑窗，同 RankGPT）
  3. Generator G（唯一训练）：从重排后的候选集里生成技术集合

本复现的适配（会在论文中如实说明）：
  * 检索语料：以 BEDR 参照库（21,453 条 / 679 类）替代原论文的 (文本, 标签) 训练集
  * Retriever：BM25（与原论文一致），K=40
  * Re-ranker：以 DeepSeek 替代原论文的 DeepSeek V3（同为 DeepSeek 系列）
  * **Generator 阶段**：原论文需 LoRA 微调 Ministral-8B；本复现以"零样本生成 +
    约束在重排候选集内"实现（原论文也强调生成被约束在候选集内以抑制幻觉）

用法：
    python run_baseline_techniquerag.py --limit 200
    python run_baseline_techniquerag.py --limit 10 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter

import numpy as np

import config as C
from bedr_data import load_bedr_prototypes
from metrics import (
    build_label_matrix, macro_micro_prf, per_report_mean_prf, recall_at_k,
)
from llm_clients import get_client

# ---------------------------------------------------------------------------
# BM25（纯 Python 实现，避免依赖 rank_bm25）
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus_tokens, k1=1.6, b=0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(self.N, 1)
        self.df = Counter()
        for d in corpus_tokens:
            for t in set(d):
                self.df[t] += 1
        self.idf = {t: math.log(1 + (self.N - c + 0.5) / (c + 0.5))
                    for t, c in self.df.items()}
        self.tf = [Counter(d) for d in corpus_tokens]

    def score(self, query_tokens, doc_idx):
        s = 0.0
        dl = len(self.corpus[doc_idx])
        for t in query_tokens:
            if t not in self.idf:
                continue
            f = self.tf[doc_idx].get(t, 0)
            if f == 0:
                continue
            s += self.idf[t] * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9)))
        return s

    def top_k(self, query_tokens, k):
        scores = [(self.score(query_tokens, i), i) for i in range(self.N)]
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]{1,}")


def tok(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# Re-ranker（零样本 LLM，显式推理）
# ---------------------------------------------------------------------------
RERANK_SYSTEM = """You are a MITRE ATT&CK mapping expert.
You are given an observed sentence and a list of retrieved candidate techniques.
Rank the candidates by how well each matches the observed sentence.

### REASONING PROCEDURE
Step 1: identify the core adversarial behaviour in the sentence.
Step 2: map that behaviour to the relevant tactic(s).
Step 3: link it to the most appropriate technique or sub-technique, paying attention to
        fine-grained differences between related sub-techniques.

### RULES
- Consider that a sentence may involve more than one technique (direct or implied).
- Return only candidate identifiers that were provided to you.
- Return at most 10 identifiers, most relevant first.

### OUTPUT FORMAT (JSON)
{"ranking": ["Txxxx", "Txxxx.xxx"]}
"""


def rerank(client, model, query_text, candidates, retries=3):
    """candidates: list of (label, text)。返回排序后的 label 列表。"""
    block = "\n".join(f"{i+1}. [{lab}] {txt[:180]}"
                      for i, (lab, txt) in enumerate(candidates))
    user = (f'### OBSERVED SENTENCE\n"{query_text}"\n\n'
            f'### CANDIDATE TECHNIQUES\n{block}\n\n'
            f'### TASK\nRank the candidates by relevance.')
    valid = {lab for lab, _ in candidates}
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": RERANK_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=300,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            ranking = data.get("ranking", [])
            if isinstance(ranking, str):
                ranking = [ranking]
            return [str(r).strip() for r in ranking if str(r).strip() in valid]
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return []


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=C.DEFAULT_TEST_LIMIT)
    ap.add_argument("--topk", type=int, default=40, help="BM25 检索数 K（原论文 K=40）")
    ap.add_argument("--topn", type=int, default=5, help="重排后保留数")
    ap.add_argument("--llm", default="deepseek-chat")
    ap.add_argument("--offset", type=int, default=1, choices=[0, 1])
    ap.add_argument("--tag", default="techniquerag")
    ap.add_argument("--out", default="", help="自定义结果文件名（默认 tag+llm）")
    ap.add_argument("--cache", default="")
    args = ap.parse_args()

    # ---------- 参照库 ----------
    proto_embs, proto_labels, proto_texts = load_bedr_prototypes(offset=args.offset)
    label_space = sorted(set(proto_labels))
    print(f"[RefLib] {len(proto_texts)} 条参照句 / {len(label_space)} 类")

    print("[BM25] 构建索引 ...")
    t0 = time.time()
    corpus_tokens = [tok(t) for t in proto_texts]
    bm25 = BM25(corpus_tokens)
    print(f"[BM25] 索引完成（{time.time()-t0:.1f}s）")

    # ---------- 报告 ----------
    cache_path = args.cache or os.path.join(C.CACHE_DIR, "report_action_embeddings_casd.npz")
    Z = np.load(cache_path, allow_pickle=True)
    rids = list(Z["report_ids"])[: args.limit]
    gt_sets = [set(x) for x in Z["gt_ids"]][: args.limit]
    tkey = "casd_texts" if "casd_texts" in Z else "sent_texts"
    texts = list(Z[tkey])[: args.limit]
    print(f"[CTI] {len(rids)} 篇报告 | K={args.topk} | top-n={args.topn}")

    client, model = get_client(args.llm)
    print(f"[LLM] {args.llm} ({model})")

    preds, r10, r30 = [], [], []
    n_llm = 0
    t0 = time.time()

    for idx, units in enumerate(texts, 1):
        # Recall@K：把 BM25 检索到的标签并集作为召回集合
        s10, s30 = set(), set()
        report_pred = set()
        for u in units:
            qt = tok(str(u))
            if not qt:
                continue
            idxs = bm25.top_k(qt, args.topk)
            cand = [(proto_labels[j], proto_texts[j]) for j in idxs]
            s10.update([c[0] for c in cand[:10]])
            s30.update([c[0] for c in cand[:30]])
            ranked = rerank(client, model, str(u), cand)
            n_llm += 1
            report_pred.update(ranked[: args.topn])
        preds.append(report_pred); r10.append(s10); r30.append(s30)
        if idx % 5 == 0 or idx == len(rids):
            print(f"  [{idx}/{len(rids)}] LLM 调用 {n_llm} 次，用时 {time.time()-t0:.0f}s",
                  flush=True)

    yt, yp = build_label_matrix(gt_sets, preds, label_space)
    M = macro_micro_prf(yt, yp)
    rp, rr, rf1 = per_report_mean_prf(gt_sets, preds)
    R10, R30 = recall_at_k(gt_sets, r10), recall_at_k(gt_sets, r30)

    print("\n" + "=" * 74)
    print(f"  TechniqueRAG (复现) | {len(rids)} 篇 | LLM {n_llm} 次")
    print("=" * 74)
    print(f"  Macro-P : {M['macro_p']:.4f}   Macro-R : {M['macro_r']:.4f}   "
          f"Macro-F1 : {M['macro_f1']:.4f}")
    print(f"  Micro-P : {M['micro_p']:.4f}   Micro-R : {M['micro_r']:.4f}   "
          f"Micro-F1 : {M['micro_f1']:.4f}")
    print(f"  逐报告平均 P/R/F1 : {rp:.4f} / {rr:.4f} / {rf1:.4f}")
    print(f"  Recall@10 : {R10:.4f}   Recall@30 : {R30:.4f}")
    print("=" * 74)

    out = {
        "config": {"method": "TechniqueRAG", "limit": len(rids), "topk": args.topk,
                   "topn": args.topn, "llm": args.llm, "offset": args.offset},
        "name": "TechniqueRAG", "metrics": M,
        "per_report_mean": {"p": rp, "r": rr, "f1": rf1},
        "recall@10": R10, "recall@30": R30, "n_llm_calls": n_llm,
        "predictions": {r: sorted(p) for r, p in zip(rids, preds)},
        "ground_truth": {r: sorted(g) for r, g in zip(rids, gt_sets)},
    }
    fname = args.out or f"baseline_{args.tag}_{args.llm}.json"
    fp = os.path.join(C.RESULTS_DIR, fname)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[Saved] {fp}")


if __name__ == "__main__":
    main()
