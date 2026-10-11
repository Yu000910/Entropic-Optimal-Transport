#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tram_eval.py — 在 MITRE 官方 TRAM 基准上评测本框架（外部验证）

目的
----
回应"评测语料与参照库同源、结果是否自圆其说"的关切：在一个**非本工作构造的、
由 MITRE CTID 人工标注**的公开基准上评测本框架，并与朴素余弦检索对比。

数据
----
    查询集：TRAM multi_label.json 中带标注的句子（4,041 条，avg 24.4 词）
    参照库：BEDR 逐条参考句原型（21,453 条 / 679 类）+ FSD(PCA k=1)

配置
----
    --mode greedy : FSD 去噪 + 贪心余弦（几何对齐层，无 LLM）
    --mode full   : CASD(对 TRAM 句已为原子动作，跳过) + FSD + EOT + SLV（完整框架）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

import config as C
from bedr_data import load_bedr_prototypes
from metrics import build_label_matrix, macro_micro_prf
from run_full_pipeline import SYSTEM_SLV, llm_score
from llm_clients import REGISTRY, get_client

NOISE = os.path.join(C.CACHE_DIR, "noise_dict_embeddings_paper.npy")


def make_basis(k=1):
    X = np.load(NOISE).astype(np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k]


def l2(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def sinkhorn(cost, reg=50.0, it=200, tol=1e-9):
    n, m = cost.shape
    K = np.maximum(np.exp(-cost / max(reg, 1e-12)), 1e-300)
    r, c = np.ones(n) / n, np.ones(m) / m
    v = np.ones(m)
    for _ in range(it):
        u = r / (K @ v + 1e-300)
        vn = c / (K.T @ u + 1e-300)
        if np.max(np.abs(vn - v)) < tol:
            v = vn; break
        v = vn
    return u[:, None] * K * v[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="greedy", choices=["greedy", "full"])
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--n-cand", type=int, default=30)
    ap.add_argument("--sim-threshold", type=float, default=0.35)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--eot-reg", type=float, default=50.0)
    ap.add_argument("--llm", default="deepseek-chat")
    ap.add_argument("--out", default="tram_eval.json")
    args = ap.parse_args()

    B = make_basis(1)
    pe, plab, ptext = load_bedr_prototypes(offset=1, verbose=False)
    P = l2(np.asarray(pe, dtype=np.float32) - (np.asarray(pe, dtype=np.float32) @ B.T) @ B)
    lab = np.array(plab); ptext = list(ptext)
    classes = sorted(set(plab))

    z = np.load(os.path.join(C.CACHE_DIR, "tram_embeddings.npz"), allow_pickle=True)
    E = z["emb"].astype(np.float32)
    Q = json.load(open(os.path.join(C.CACHE_DIR, "tram_queries.json"), encoding="utf-8"))
    if args.limit:
        E = E[: args.limit]; Q = Q[: args.limit]
    Qe = l2(E - (E @ B.T) @ B)
    texts = [q["text"] for q in Q]
    gts = [set(q["labels"]) for q in Q]
    print(f"[TRAM] 查询 {len(texts)} | 参照原型 {P.shape[0]} | 类别 {len(classes)} | mode={args.mode}")

    # 预构建候选池
    sim_all = Qe @ P.T
    pools = [sorted(set(np.argsort(-sim_all[i])[:, ] if False else np.argsort(-sim_all[i])[: args.n_cand])) for i in range(len(Qe))]
    pools = []
    for i in range(len(Qe)):
        o = np.argsort(-sim_all[i])[: args.n_cand]
        pools.append(sorted(set(o.tolist())))

    client = None; model = None
    if args.mode == "full":
        if args.llm not in REGISTRY:
            raise SystemExit(f"未知 LLM {args.llm}")
        client, model = get_client(args.llm)
        print(f"[LLM] {args.llm} ({model})")

    cache = {}; n_calls = [0]
    preds = []
    t0 = time.time()
    for i in range(len(Qe)):
        pool = pools[i]
        if len(pool) < 2:
            preds.append(set()); continue
        Sp = sim_all[i][pool]
        # 单查询：n=1，用相似度归一化作为 P_align（论文式）
        pa = Sp / (Sp.sum() + 1e-12)
        order = np.argsort(-Sp)[: args.topk]
        cands = [{"id": str(lab[pool[j]]), "desc": ptext[pool[j]]} for j in order]
        mass = pa[order]

        if args.mode == "greedy":
            # 纯余弦：取最相似的那一个（论文口径：每动作选 1 个）
            preds.append({cands[0]["id"]})
        else:
            if Sp.max() <= args.sim_threshold:
                preds.append(set()); continue
            key = (texts[i], tuple(c["id"] for c in cands))
            if key in cache:
                sc = cache[key]
            else:
                sc = llm_score(client, model, texts[i], cands)
                cache[key] = sc; n_calls[0] += 1
                if n_calls[0] % 200 == 0:
                    print(f"    LLM {n_calls[0]} 次 | {i}/{len(Qe)} | {time.time()-t0:.0f}s", flush=True)
            best, bs = None, -1e18
            for j, cd in enumerate(cands):
                sl = sc.get(cd["id"])
                f = math.log(float(mass[j]) + 1e-12) + (args.alpha * math.log(sl) if sl and sl > 0 else 0.0)
                if f > bs:
                    bs, best = f, cd["id"]
            preds.append({best} if best else set())

    yt, yp = build_label_matrix(gts, preds, classes)
    m = macro_micro_prf(yt, yp)
    hit = sum(1 for p, g in zip(preds, gts) if p & g) / len(gts)
    res = {"config": vars(args), "n_query": len(gts),
           "macro_f1": m["macro_f1"], "micro_f1": m["micro_f1"],
           "micro_p": m["micro_p"], "micro_r": m["micro_r"],
           "hit_rate": hit, "n_llm_calls": n_calls[0],
           "seconds": time.time() - t0}
    print(f"\n=== TRAM 外部基准（{len(gts)} 条）mode={args.mode} ===")
    print(f"  Macro-F1={m['macro_f1']:.4f}  Micro-F1={m['micro_f1']:.4f}  "
          f"P={m['micro_p']:.4f}  R={m['micro_r']:.4f}  命中率={hit:.4f}")
    print(f"  LLM 调用={n_calls[0]}  用时={time.time()-t0:.0f}s")
    fp = os.path.join(C.RESULTS_DIR, args.out)
    json.dump(res, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[Saved] {fp}")


if __name__ == "__main__":
    main()
