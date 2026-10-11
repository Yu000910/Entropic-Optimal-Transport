#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_slv_candidate_quality.py — 决定性实验：EOT 通过改善候选质量 → 提升下游 SLV

假设链
------
    EOT 在候选池内联合分配质量
        → 每个动作送入 SLV 的 top-K 候选召回更高（已实测 K=5: +1.50%）
        → SLV 能验证出更多真实技术
        → 端到端 F1 提升

设计（严格控制变量）
--------------------
    同一批报告、同一 CASD 动作、同一 FSD 去噪、同一候选池、
    同一 SLV 提示词与模型（temperature=0.0）、同一融合公式与 top-K。
    唯一变量：候选的排序依据
        A) greedy —— 按余弦相似度排序（+ 余弦归一化作为 P_align）
        B) eot    —— 按 Sinkhorn 传输质量排序（+ 传输质量归一化作为 P_align）

为避免重复计费，SLV 结果按「动作文本 + 候选集合」缓存。
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
from run_full_pipeline import (SYSTEM_SLV, build_slv_prompt, llm_score)
from llm_clients import get_client

CASD = os.path.join(C.CACHE_DIR, "report_action_embeddings_casd.npz")
NOISE = os.path.join(C.CACHE_DIR, "noise_dict_embeddings_paper.npy")


def make_basis(k):
    X = np.load(NOISE).astype(np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k]


def proj(E, B):
    E = np.asarray(E, dtype=np.float32)
    return E - (E @ B.T) @ B


def l2(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


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
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--n-cand", type=int, default=30)
    ap.add_argument("--sim-threshold", type=float, default=0.35)
    ap.add_argument("--fsd-k", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--eot-reg", type=float, default=50.0)
    ap.add_argument("--proto-mode", default="per_ref",
                    choices=["per_ref", "centroid"],
                    help="per_ref=逐条参考句原型（论文口径）；centroid=类均值")
    ap.add_argument("--llm", default="deepseek-chat",
                    help="SLV 使用的 LLM 后端（见 llm_clients.REGISTRY）")
    ap.add_argument("--out", default="slv_candidate_quality.json")
    args = ap.parse_args()

    B = make_basis(args.fsd_k)
    pe, plab, ptext = load_bedr_prototypes(offset=1, verbose=False)
    pe = l2(proj(pe, B))
    if args.proto_mode == "per_ref":
        # 论文/原仓库口径：每一条参考句都是一个原型
        P, label_of, ptext = pe, np.array(plab), list(ptext)
    else:
        classes = sorted(set(plab))
        arr = np.array(plab)
        P = l2(np.stack([pe[arr == c].mean(axis=0) for c in classes]))
        label_of, ptext = np.array(classes), [ptext[arr == c][0] for c in classes]
    classes = sorted(set(str(x) for x in label_of))
    print(f"[原型] 模式={args.proto_mode}  原型数={P.shape[0]}  "
          f"唯一类别={len(classes)}")

    z = np.load(CASD, allow_pickle=True)
    rids = [str(x) for x in z["report_ids"]][: args.limit]
    gts = [set(x) for x in z["gt_ids"]][: args.limit]
    acts = [l2(proj(np.asarray(a, dtype=np.float32), B))
            for a in list(z["casd_embs"])[: args.limit]]
    atexts = [list(x) for x in z["casd_texts"]][: args.limit]

    from llm_clients import REGISTRY
    if args.llm not in REGISTRY:
        raise SystemExit(f"未知 LLM {args.llm}；可选 {list(REGISTRY)}")
    client, model = get_client(args.llm)
    print(f"[SLV] model={model} | topk={args.topk} | n_cand={args.n_cand} "
          f"| alpha={args.alpha} | eot_reg={args.eot_reg}")

    cache = {}                     # (action_text, tuple(ids)) -> {id: score}
    n_calls = [0]

    def score(action_text, cands):
        key = (action_text, tuple(c["id"] for c in cands))
        if key in cache:
            return cache[key]
        out = llm_score(client, model, action_text, cands)
        n_calls[0] += 1
        cache[key] = out
        if n_calls[0] % 100 == 0:
            print(f"    SLV 调用 {n_calls[0]} 次（去重后）", flush=True)
        return out

    def run(mode):
        preds = []
        for ri, A in enumerate(acts):
            A = np.asarray(A, dtype=np.float32)
            if A.shape[0] == 0:
                preds.append(set()); continue
            sim = A @ P.T
            pool = sorted(set(np.argsort(-sim, axis=1)[:, : args.n_cand].ravel()))
            if len(pool) < 2:
                preds.append(set()); continue
            Sp = sim[:, pool]
            if mode == "eot":
                S = sinkhorn(1.0 - Sp, reg=args.eot_reg)
            else:
                S = Sp / (Sp.sum(axis=1, keepdims=True) + 1e-12)

            pred = set()
            for r in range(S.shape[0]):
                if Sp[r].max() <= args.sim_threshold:
                    continue
                order = np.argsort(S[r])[::-1][: args.topk]
                cands = [{"id": str(label_of[pool[j]]),
                          "desc": ptext[pool[j]]} for j in order]
                mass = S[r][order]
                pa = mass / (mass.sum() + 1e-12)
                sc = score(str(atexts[ri][r]) if r < len(atexts[ri]) else "", cands)
                best, bs = None, -1e18
                for j, cd in enumerate(cands):
                    sl = sc.get(cd["id"])
                    fused = (math.log(float(pa[j]) + 1e-12)
                             + (args.alpha * math.log(sl) if sl and sl > 0 else 0.0))
                    if fused > bs:
                        bs, best = fused, cd["id"]
                if best is not None:
                    pred.add(best)
            preds.append(pred)
        return preds

    res = {}
    for mode in ("greedy", "eot"):
        t0 = time.time()
        pr = run(mode)
        yt, yp = build_label_matrix(gts, pr, classes)
        m = macro_micro_prf(yt, yp)
        per = [2 * len(p & g) / max(len(p) + len(g), 1) for p, g in zip(pr, gts)]
        res[mode] = {"macro_f1": m["macro_f1"], "micro_f1": m["micro_f1"],
                     "micro_p": m["micro_p"], "micro_r": m["micro_r"],
                     "per_report_f1": float(np.mean(per)),
                     "n_slv_calls": n_calls[0], "seconds": time.time() - t0,
                     "predictions": {r: sorted(s) for r, s in zip(rids, pr)}}
        print(f"\n[{mode}] macro={m['macro_f1']:.4f} micro={m['micro_f1']:.4f} "
              f"P={m['micro_p']:.4f} R={m['micro_r']:.4f} "
              f"perReport={np.mean(per):.4f}  用时 {time.time()-t0:.0f}s")

    g, e = res["greedy"], res["eot"]
    res["delta"] = {"macro_f1": e["macro_f1"] - g["macro_f1"],
                    "micro_f1": e["micro_f1"] - g["micro_f1"],
                    "per_report_f1": e["per_report_f1"] - g["per_report_f1"]}
    print("\n=== EOT 相对 greedy（同 SLV、同候选池、同提示词）===")
    for k, v in res["delta"].items():
        print(f"  Δ{k:<16}{v:+.4f}")

    fp = os.path.join(C.RESULTS_DIR, args.out)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": res}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n[Saved] {fp}")


if __name__ == "__main__":
    main()
