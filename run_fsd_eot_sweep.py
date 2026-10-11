#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_fsd_eot_sweep.py — FSD / EOT 的论文口径复现

针对实测发现的偏差（FSD 增益仅 +0.003，EOT 增益 −0.012，而论文称 +5.3% / +2.7%）：

  1. FSD 噪声词典：原代码 13 个硬编码词 → 论文 §3.3 的「50 种子词 + 5-NN 扩展 = 250」
  2. FSD 主成分数 k：原代码 1 → 论文的 k=5（且论文称试过 k∈{3,5,7,10}）
  3. 噪声基拟合对象：原代码用**噪声词向量**做 PCA → 论文 §3.3 也是（这里保持一致）
  4. Sinkhorn 核：论文式(6) 为 K = exp(-C/λ)，λ=50

用法：
    python run_fsd_eot_sweep.py --limit 200
    python run_fsd_eot_sweep.py --limit 200 --eot-reg 50 --eot-sinkhorn paper
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import config as C
from bedr_data import load_bedr_prototypes
from metrics import (
    build_label_matrix, macro_micro_prf, per_report_mean_prf, recall_at_k,
)
from build_embeddings import NOISE_SEED_TERMS, embed_texts

CACHE_CASD = os.path.join(C.CACHE_DIR, "report_action_embeddings_casd.npz")
NOISE_TOTAL_CACHE = os.path.join(C.CACHE_DIR, "noise_dict_embeddings_paper.npy")


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def build_noise_dictionary(bedr_texts, bedr_embs, target=250, nn=5):
    """论文 §3.3 口径：50 个种子词 + 在嵌入空间取 5-NN 扩展，最终 250 条。"""
    if os.path.exists(NOISE_TOTAL_CACHE):
        return np.load(NOISE_TOTAL_CACHE)

    seed_emb = embed_texts(NOISE_SEED_TERMS, desc="seed")
    P = l2(bedr_embs)
    S = l2(seed_emb)
    sim = S @ P.T
    picked = []
    seen = set()
    for r in range(sim.shape[0]):
        for j in np.argsort(sim[r])[::-1][:nn]:
            if int(j) not in seen:
                seen.add(int(j))
                picked.append(int(j))
    extra_texts = [bedr_texts[j] for j in picked][: target - len(NOISE_SEED_TERMS)]
    extra_emb = embed_texts(extra_texts, desc="nn")
    all_emb = np.vstack([seed_emb, extra_emb]).astype(np.float32)
    np.save(NOISE_TOTAL_CACHE, all_emb)
    print(f"[Noise] 构造完成：{len(NOISE_SEED_TERMS)} 种子 + {len(extra_texts)} 近邻 "
          f"= {all_emb.shape[0]} 条")
    return all_emb


def fit_basis(noise_embs, k):
    Xc = noise_embs - noise_embs.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k]


def project(E, B):
    return E - (E @ B.T) @ B


def sinkhorn(cost, reg=50.0, max_iter=200, tol=1e-8, mode="paper"):
    """mode='paper' → K = exp(-C/λ)（论文式(6)）；mode='legacy' → K = exp(-λC)。"""
    n, m = cost.shape
    if mode == "paper":
        K = np.exp(-cost / max(reg, 1e-9))
    else:
        K = np.exp(-reg * cost)
    K = np.maximum(K, 1e-300)
    K /= K.sum()
    r = np.ones(n) / n
    c = np.ones(m) / m
    v = np.ones(m)
    for _ in range(max_iter):
        u = r / (K @ v + 1e-300)
        v_new = c / (K.T @ u + 1e-300)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    return u[:, None] * K * v[None, :]


def predict_doc(A, P, lab, B=None, use_eot=False, eot_mode="paper",
                reg=50.0, n_cand=30, thr=0.35):
    if A is None or len(A) == 0:
        return set(), set(), set()
    A = np.asarray(A, dtype=np.float32)
    if B is not None:
        A = project(A, B)
    Q = l2(A)
    sim = Q @ P.T

    s10, s30 = set(), set()
    for r in range(sim.shape[0]):
        o = np.argsort(sim[r])[::-1]
        s10.update(lab[o[:10]].tolist())
        s30.update(lab[o[:30]].tolist())

    pool = set()
    for r in range(sim.shape[0]):
        pool.update(np.argsort(sim[r])[::-1][:n_cand].tolist())
    pool = sorted(pool)
    ps = sim[:, pool]

    if use_eot and len(pool) > 1:
        T = sinkhorn(1.0 - ps, reg=reg, mode=eot_mode)
    else:
        T = ps

    pred = set()
    for r in range(sim.shape[0]):
        if ps[r].max() <= thr:
            continue
        j = int(np.argmax(T[r]))
        pred.add(lab[pool[j]])
    return pred, s10, s30


def evaluate(embs, gt_sets, P, lab, label_space, **kw):
    preds, r10, r30 = [], [], []
    for A in embs:
        p, s10, s30 = predict_doc(A, P, lab, **kw)
        preds.append(p); r10.append(s10); r30.append(s30)
    yt, yp = build_label_matrix(gt_sets, preds, label_space)
    M = macro_micro_prf(yt, yp)
    rp, rr, rf1 = per_report_mean_prf(gt_sets, preds)
    return M, rf1, recall_at_k(gt_sets, r10), recall_at_k(gt_sets, r30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=C.DEFAULT_TEST_LIMIT)
    ap.add_argument("--offset", type=int, default=1, choices=[0, 1])
    ap.add_argument("--cache", default=CACHE_CASD)
    ap.add_argument("--n-cand", type=int, default=C.DEFAULT_N_CAND)
    ap.add_argument("--thr", type=float, default=C.DEFAULT_SIM_THRESHOLD)
    args = ap.parse_args()

    proto_embs, proto_labels, proto_texts = load_bedr_prototypes(offset=args.offset,
                                                                 verbose=False)
    label_space = sorted(set(proto_labels))
    lab = np.array(proto_labels)

    Z = np.load(args.cache, allow_pickle=True)
    rids = list(Z["report_ids"])[: args.limit]
    gt_sets = [set(x) for x in Z["gt_ids"]][: args.limit]
    key = "casd_embs" if "casd_embs" in Z else "sent_embs"
    embs = list(Z[key])[: args.limit]
    print(f"[CTI] {len(rids)} 篇 | 动作来源 {key} | 候选 {args.n_cand} | thr {args.thr}")

    # --- 噪声词典（论文口径 250 条） ---
    noise250 = build_noise_dictionary(proto_texts, proto_embs)
    noise13 = embed_texts(
        ["cyber attack", "malware", "system", "network", "hacker", "execute", "run",
         "file", "data", "connection", "process", "command", "tool"], desc="legacy13")

    rows = []

    def add(name, **kw):
        t0 = time.time()
        P = l2(proto_embs) if kw.get("B") is None else l2(project(proto_embs, kw["B"]))
        M, rf1, R10, R30 = evaluate(embs, gt_sets, P, lab, label_space, **kw)
        rows.append({
            "name": name, "macro_f1": M["macro_f1"], "micro_f1": M["micro_f1"],
            "per_report_f1": rf1, "recall@10": R10, "recall@30": R30,
            "macro_p": M["macro_p"], "macro_r": M["macro_r"],
        })
        print(f"  {name:<34} 逐报告F1={rf1:.4f}  Micro={M['micro_f1']:.4f}  "
              f"Macro={M['macro_f1']:.4f}  R@30={R30:.4f}  ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 96)
    print("  FSD 噪声词典 / 主成分数 k 扫描（EOT 关闭）")
    print("=" * 96)
    add("no FSD (baseline)")
    for k in (1, 3, 5, 7, 10):
        add(f"FSD legacy13 k={k}", B=fit_basis(noise13, k))
    for k in (1, 3, 5, 7, 10):
        add(f"FSD paper250 k={k}", B=fit_basis(noise250, k))

    print("\n" + "=" * 96)
    print("  EOT 对照（固定 FSD = paper250 k=5）")
    print("=" * 96)
    B5 = fit_basis(noise250, 5)
    add("paper250 k=5 + cosine", B=B5, use_eot=False)
    for mode in ("paper", "legacy"):
        for reg in (10.0, 50.0, 100.0):
            add(f"paper250 k=5 + EOT {mode} λ={reg:g}",
                B=B5, use_eot=True, eot_mode=mode, reg=reg)

    out = {"config": {"limit": len(rids), "offset": args.offset,
                      "cache": os.path.basename(args.cache),
                      "n_cand": args.n_cand, "thr": args.thr},
           "rows": rows}
    fp = os.path.join(C.RESULTS_DIR, "fsd_eot_sweep.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[Saved] {fp}")

    best = max(rows, key=lambda r: r["per_report_f1"])
    print(f"\n最优配置：{best['name']}  逐报告F1={best['per_report_f1']:.4f}")


if __name__ == "__main__":
    main()
