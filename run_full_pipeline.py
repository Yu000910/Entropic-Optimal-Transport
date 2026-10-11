#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_pipeline.py — 完整四阶段流水线（CASD + FSD + EOT + SLV），验证论文 0.738

SLV 严格按论文 §3.4.2 / §3.4.3 实现：
    三维打分  S_logic = Σ ω_j · Eval(a, t_k; D_j) · I_constraint
              ω = (intent 0.4, artifact 0.3, implementation 0.3)
    分数融合  S_final = log P_align(t_k|a) + α · log S_logic(t_k)
    拒识规则  若 max S_final < τ 则输出空集

**批处理优化**：把同一 action 的所有候选放进**一次** LLM 请求，
请求数从 n_act × topk 降到 n_act（约 10 倍加速）。

用法：
    # 先小样本验证方向
    python run_full_pipeline.py --limit 20 --tag smoke

    # 全量
    python run_full_pipeline.py --limit 200 --tag full

    # 消融：关掉某个模块
    python run_full_pipeline.py --limit 200 --no-eot --tag no_eot
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

import config as C
from bedr_data import load_bedr_prototypes
from metrics import (
    build_label_matrix, macro_micro_prf, per_report_mean_prf, recall_at_k,
)

# ---------------------------------------------------------------------------
# LLM 打分（三维）
# ---------------------------------------------------------------------------
SYSTEM_SLV = """You are a senior MITRE ATT&CK mapping expert.
For ONE observed action, score how well EACH candidate technique matches,
along three independent dimensions:
  - intent:         same adversarial goal?  (e.g. Persistence vs Execution)
  - artifact:       does it involve the specific system objects named in the
                    technique (registry keys, scheduled tasks, browser data, ...)?
  - implementation: does it use the specific tool / mechanism the technique
                    requires (e.g. strictly distinguish PowerShell from cmd.exe)?

Rules:
  * Score each dimension in [0.0, 1.0].
  * If the observed action uses Tool A, but a candidate explicitly requires
    Tool B, set that candidate's "impl" score to 0.0.
  * Be discriminative; do not give every candidate the same score.

Return ONLY valid JSON of the form:
{"scores": [{"id": "Txxxx", "intent": 0.0, "artifact": 0.0, "impl": 0.0}, ...]}
Every candidate id provided must appear exactly once.
"""

W_INTENT, W_ARTIFACT, W_IMPL = 0.4, 0.3, 0.3


def build_slv_prompt(action_text, candidates):
    lines = []
    for i, c in enumerate(candidates, 1):
        desc = str(c["desc"]).replace("\n", " ")[:220]
        lines.append(f"{i}. [{c['id']}] {desc}")
    cand_block = "\n".join(lines)
    user = (f'### OBSERVED ACTION\n"{action_text}"\n\n'
            f"### CANDIDATE TECHNIQUES\n{cand_block}\n\n"
            f"### TASK\nScore each candidate on intent / artifact / implementation.")
    return user


def llm_score(client, model, action_text, candidates, retries=3):
    """返回 {id: S_logic}；失败返回 {}。"""
    if not candidates:
        return {}
    user = build_slv_prompt(action_text, candidates)
    ids = [c["id"] for c in candidates]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_SLV},
                          {"role": "user", "content": user}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            out = {}
            for item in data.get("scores", []):
                cid = str(item.get("id", "")).strip()
                if cid not in ids:
                    continue
                s = (W_INTENT * float(item.get("intent", 0.0))
                     + W_ARTIFACT * float(item.get("artifact", 0.0))
                     + W_IMPL * float(item.get("impl", 0.0)))
                out[cid] = min(max(s, 0.0), 1.0)
            if out:
                return out
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return {}


# ---------------------------------------------------------------------------
# Sinkhorn
# ---------------------------------------------------------------------------
def sinkhorn(cost, reg=50.0, max_iter=100, tol=1e-6):
    n, m = cost.shape
    K = np.exp(-cost / max(reg, 1e-9))
    K = np.maximum(K, 1e-300)
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


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=C.DEFAULT_TEST_LIMIT)
    ap.add_argument("--cache", default=os.path.join(C.CACHE_DIR,
                                                    "report_action_embeddings_casd.npz"))
    ap.add_argument("--offset", type=int, default=1, choices=[0, 1])
    ap.add_argument("--fsd", action="store_true", default=True)
    ap.add_argument("--no-fsd", dest="fsd", action="store_false")
    ap.add_argument("--fsd-k", type=int, default=C.DEFAULT_N_COMPONENTS)
    ap.add_argument("--eot", action="store_true", default=True)
    ap.add_argument("--no-eot", dest="eot", action="store_false")
    ap.add_argument("--topk", type=int, default=C.DEFAULT_TOPK_LLM,
                    help="交给 LLM 打分的候选数 K")
    ap.add_argument("--n-cand", type=int, default=C.DEFAULT_N_CAND)
    ap.add_argument("--alpha", type=float, default=1.0, help="逻辑分权重 α")
    ap.add_argument("--tau", type=float, default=-1e9,
                    help="拒识阈值 τ（基于融合分，默认不拒识）")
    ap.add_argument("--sim-threshold", type=float, default=C.DEFAULT_SIM_THRESHOLD)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--llm", default="deepseek-chat",
                    help="SLV 使用的 LLM 后端（见 llm_clients.REGISTRY）")
    ap.add_argument("--cache-prefix", default="", help="结果文件名前缀")
    ap.add_argument("--dump-scores", default="", help="将每个动作的融合分数写入该 JSON 文件（用于阈值分析）")
    args = ap.parse_args()

    from llm_clients import get_client, REGISTRY
    if args.llm not in REGISTRY:
        raise SystemExit(f"未知 LLM {args.llm}；可选 {list(REGISTRY)}")
    client, llm_model = get_client(args.llm)
    print(f"[LLM] 后端 = {args.llm} ({llm_model})")

    # ---------- 原型 ----------
    proto_embs, proto_labels, proto_texts = load_bedr_prototypes(offset=args.offset)
    label_space = sorted(set(proto_labels))

    # ---------- FSD ----------
    if args.fsd:
        from build_embeddings import embed_texts, NOISE_SEED_TERMS
        noise_embs = embed_texts(NOISE_SEED_TERMS, desc="noise")
        Xc = noise_embs - noise_embs.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        B = Vt[: args.fsd_k]

        def proj(E):
            return E - (E @ B.T) @ B
        proto_embs = proj(proto_embs)
        print(f"[FSD] k={args.fsd_k}，噪声词典 {noise_embs.shape[0]} 词")

    # ---------- 报告 ----------
    Z = np.load(args.cache, allow_pickle=True)
    rids = list(Z["report_ids"])[: args.limit]
    gt_sets = [set(x) for x in Z["gt_ids"]][: args.limit]
    embs = list(Z["casd_embs"])[: args.limit] if "casd_embs" in Z else list(Z["sent_embs"])[: args.limit]
    texts = list(Z["casd_texts"])[: args.limit] if "casd_texts" in Z else list(Z["sent_texts"])[: args.limit]
    print(f"[CTI] {len(rids)} 篇 | K={args.topk} | EOT={args.eot} | FSD={args.fsd} "
          f"| α={args.alpha} | τ={args.tau}")

    if args.fsd:
        embs = [proj(np.asarray(e, dtype=np.float32)) for e in embs]

    P = l2(np.asarray(proto_embs, dtype=np.float32))
    lab_arr = np.array(proto_labels)

    preds, r10, r30 = [], [], []
    score_log = []
    n_llm = 0
    t0 = time.time()

    for idx, (A, acts) in enumerate(zip(embs, texts), 1):
        A = np.asarray(A, dtype=np.float32)
        if A.size == 0:
            preds.append(set()); r10.append(set()); r30.append(set()); continue
        Q = l2(A)
        sim = Q @ P.T                                  # (n_act, n_proto)

        # Recall@K（用去噪后的余弦 top-K 并集近似 OT 后的候选覆盖）
        s10, s30 = set(), set()
        for r in range(sim.shape[0]):
            o = np.argsort(sim[r])[::-1]
            s10.update(lab_arr[o[:10]].tolist())
            s30.update(lab_arr[o[:30]].tolist())
        r10.append(s10); r30.append(s30)

        # 候选池
        pool = set()
        for r in range(sim.shape[0]):
            pool.update(np.argsort(sim[r])[::-1][: args.n_cand].tolist())
        pool = sorted(pool)
        pool_sim = sim[:, pool]

        if args.eot and len(pool) > 1:
            T = sinkhorn(1.0 - pool_sim, reg=50.0, max_iter=100)
        else:
            T = pool_sim / (pool_sim.sum(axis=1, keepdims=True) + 1e-12)

        report_pred = set()
        action_scores = []
        for r in range(sim.shape[0]):
            if pool_sim[r].max() <= args.sim_threshold:
                continue
            order = np.argsort(T[r])[::-1][: args.topk]
            cands = [{"id": proto_labels[pool[j]], "desc": proto_texts[pool[j]]}
                     for j in order]
            # P_align 归一化
            mass = T[r][order]
            p_align = mass / (mass.sum() + 1e-12)

            scores = llm_score(client, llm_model, str(acts[r]), cands)
            n_llm += 1

            best_id, best_score = None, -1e18
            for j, cand in enumerate(cands):
                pa = float(p_align[j])
                sl = scores.get(cand["id"])
                if sl is None or sl <= 0.0:
                    fused = math.log(pa + 1e-12)
                else:
                    fused = math.log(pa + 1e-12) + args.alpha * math.log(sl)
                if fused > best_score:
                    best_score, best_id = fused, cand["id"]
            action_scores.append(float(best_score))
            if best_id is not None and best_score > args.tau:
                report_pred.add(best_id)

        preds.append(report_pred)
        if args.dump_scores:
            score_log.extend(action_scores)
        if idx % 10 == 0 or idx == len(rids):
            print(f"  [{idx}/{len(rids)}] LLM 请求 {n_llm} 次，"
                  f"用时 {time.time()-t0:.0f}s", flush=True)

    # ---------- 指标 ----------
    yt, yp = build_label_matrix(gt_sets, preds, label_space)
    M = macro_micro_prf(yt, yp)
    rp, rr, rf1 = per_report_mean_prf(gt_sets, preds)
    R10, R30 = recall_at_k(gt_sets, r10), recall_at_k(gt_sets, r30)

    print("\n" + "=" * 72)
    print(f"  Full Pipeline  |  {len(rids)} 篇  |  LLM 请求 {n_llm} 次")
    print("=" * 72)
    print("  【论文口径】")
    print(f"    Macro-P : {M['macro_p']:.4f}   Macro-R : {M['macro_r']:.4f}   "
          f"Macro-F1 : {M['macro_f1']:.4f}")
    print(f"    Micro-P : {M['micro_p']:.4f}   Micro-R : {M['micro_r']:.4f}   "
          f"Micro-F1 : {M['micro_f1']:.4f}")
    print(f"    Recall@10 : {R10:.4f}   Recall@30 : {R30:.4f}")
    print("  【原仓库口径：逐报告平均】")
    print(f"    P : {rp:.4f}   R : {rr:.4f}   F1 : {rf1:.4f}")
    print("=" * 72)

    out = {
        "config": {"llm": args.llm, "limit": len(rids), "topk": args.topk, "n_cand": args.n_cand,
                   "alpha": args.alpha, "tau": args.tau, "fsd": args.fsd,
                   "fsd_k": args.fsd_k if args.fsd else None, "eot": args.eot,
                   "offset": args.offset, "cache": os.path.basename(args.cache)},
        "name": "FullPipeline", "metrics": M,
        "per_report_mean": {"p": rp, "r": rr, "f1": rf1},
        "recall@10": R10, "recall@30": R30, "n_llm_calls": n_llm,
        "predictions": {r: sorted(p) for r, p in zip(rids, preds)},
        "ground_truth": {r: sorted(g) for r, g in zip(rids, gt_sets)},
    }
    if args.dump_scores:
        with open(args.dump_scores, "w", encoding="utf-8") as f:
            json.dump(score_log, f)
        print(f"[Scores] 已保存 {len(score_log)} 个动作的融合分数至 {args.dump_scores}")
    fp = os.path.join(C.RESULTS_DIR,
                      f"{args.cache_prefix}full_pipeline_{args.tag}_{args.llm}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[Saved] {fp}")


if __name__ == "__main__":
    main()
