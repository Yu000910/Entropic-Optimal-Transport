#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation.py — 离线复现论文 Table 3（Naive RAG）与 Table 4（消融）的各档配置

配置（对齐论文 Table 4 的行）：
  1. Baseline (Naive RAG)       : 无 CASD / 无 FSD / 无 EOT / 无 SLV，贪心余弦
  2. + CASD                     : 用 LLM 抽取的原子动作替代整句
  3. + FSD                      : 加特征空间去噪
  4. + EOT                      : 用 Sinkhorn 最优传输替代贪心匹配
  5. + SLV                      : 加 LLM 逻辑验证（需 --slv，会调用 DeepSeek）
  4a. CASD + SLV
  4b. CASD + FSD + SLV

用法：
    python run_ablation.py --limit 200                 # 离线档（1/3/4）
    python run_ablation.py --limit 200 --casd          # 加 CASD 档
    python run_ablation.py --limit 200 --casd --slv    # 完整流水线
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import config as C
from metrics import (
    build_label_matrix, macro_micro_prf, per_report_mean_prf,
    recall_at_k, support_counts, per_class_f1,
)

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


class ManifoldDenoiser:
    """对齐原仓库 run_ot_experiment.py 的 ManifoldDenoiser。"""

    def __init__(self, noise_embs=None, n_components=1, generic_terms=None):
        self.n_components = n_components
        self.noise_basis = None
        self._noise_embs = noise_embs
        self._terms = generic_terms

    def fit(self):
        if self._noise_embs is not None:
            X = self._noise_embs
        else:
            from build_embeddings import embed_texts
            X = embed_texts(self._terms)
        Xc = X - X.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.noise_basis = Vt[: self.n_components]

    def transform(self, E):
        if self.noise_basis is None:
            self.fit()
        B = self.noise_basis
        return E - (E @ B.T) @ B


def sinkhorn(cost, reg=50.0, max_iter=100, tol=1e-6):
    """熵正则最优传输（Sinkhorn-Knopp）。对齐论文式(6)。"""
    n, m = cost.shape
    K = np.exp(-cost / max(reg, 1e-9))
    K = np.maximum(K, 1e-300)
    r = np.ones(n) / n
    c = np.ones(m) / m
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(max_iter):
        u_prev = u.copy()
        u = r / (K @ v + 1e-300)
        v = c / (K.T @ u + 1e-300)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    return u[:, None] * K * v[None, :]


# ---------------------------------------------------------------------------
# 预测核心
# ---------------------------------------------------------------------------
def predict_report(action_embs, proto_embs, proto_labels, use_eot,
                   threshold, n_cand, topk_llm, llm_client=None,
                   slv=False):
    """返回 (预测标签集合, 候选并集top10, 候选并集top30, 诊断信息)。"""
    if action_embs is None or len(action_embs) == 0:
        return set(), set(), set(), {}

    Q = l2norm(action_embs)
    P = l2norm(proto_embs)
    sim = Q @ P.T                                  # (n_act, n_proto)
    diag = {"n_actions": int(sim.shape[0])}

    retr10, retr30 = set(), set()
    for r in range(sim.shape[0]):
        order = np.argsort(sim[r])[::-1]
        for j in order[:10]:
            retr10.add(proto_labels[j])
        for j in order[:30]:
            retr30.add(proto_labels[j])

    pred = set()

    if not use_eot:
        # ---- 贪心余弦 ----
        pools = []
        for r in range(sim.shape[0]):
            order = np.argsort(sim[r])[::-1]
            pools.append(order[:n_cand])
        diag["pool_size"] = int(len(set(np.concatenate(pools).tolist())))

        for r in range(sim.shape[0]):
            order = np.argsort(sim[r])[::-1]
            if sim[r, order[0]] <= threshold:
                continue
            if slv and llm_client is not None:
                cand = [{"id": proto_labels[j], "desc": proto_texts_global[j],
                         "score": float(sim[r, j])} for j in order[:topk_llm]]
                sel = llm_select(llm_client, action_texts_global[r], cand)
                if sel and sel != "None":
                    pred.add(sel)
            else:
                pred.add(proto_labels[order[0]])
    else:
        # ---- EOT ----
        pool_idx = set()
        for r in range(sim.shape[0]):
            pool_idx.update(np.argsort(sim[r])[::-1][:n_cand].tolist())
        pool_idx = sorted(pool_idx)
        diag["pool_size"] = len(pool_idx)

        pool_sim = sim[:, pool_idx]                       # (n_act, m)
        cost = 1.0 - pool_sim
        T = sinkhorn(cost, reg=50.0, max_iter=100)        # (n_act, m)

        for r in range(sim.shape[0]):
            if pool_sim[r].max() <= threshold:
                continue
            local_order = np.argsort(T[r])[::-1]
            if slv and llm_client is not None:
                cand = [{"id": proto_labels[pool_idx[j]], "desc": proto_texts_global[pool_idx[j]],
                         "score": float(T[r, j])} for j in local_order[:topk_llm]]
                sel = llm_select(llm_client, action_texts_global[r], cand)
                if sel and sel != "None":
                    pred.add(sel)
            else:
                pred.add(proto_labels[pool_idx[local_order[0]]])

    return pred, retr10, retr30, diag


# 全局（由 main 注入，避免层层传参）
proto_texts_global = []
action_texts_global = []


SYSTEM_PROMPT_SLV = """You are a MITRE ATT&CK Mapping Expert.
Your task is to select the **SINGLE BEST** technique ID that matches the Observed Action.

### INSTRUCTIONS:
1. Compare the Observed Action against the Candidate Options.
2. Select the option that matches the **specific tool, method, or intent** described.
3. Be discriminative! Distinguish between similar techniques (e.g., "Replication via USB" [T1091] vs "Exfiltration via USB" [T1052]).
4. If NONE of the options are a strong semantic match, return "None".

### FORMAT:
Return JSON: {"selected_id": "Txxxx" or "None"}
"""


def llm_select(client, action_text, candidates):
    options = ""
    for i, c in enumerate(candidates):
        options += f"Option {i+1}: [{c['id']}] {str(c['desc'])[:150]}...\n"
    user = (f'\n    ### OBSERVED ACTION:\n    "{action_text}"\n\n'
            f"    ### CANDIDATE OPTIONS:\n    {options}\n\n"
            f"    ### DECISION:\n    Which Option ID is the best fit?\n    ")
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=C.LLM_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT_SLV},
                          {"role": "user", "content": user}],
                temperature=0.0, response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content).get("selected_id", "None")
        except Exception:
            time.sleep(1.0)
    return "None"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    global proto_texts_global

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=C.DEFAULT_TEST_LIMIT)
    ap.add_argument("--casd", action="store_true", help="使用 LLM 原子动作")
    ap.add_argument("--fsd", action="store_true", help="开启特征空间去噪")
    ap.add_argument("--fsd-k", type=int, default=C.DEFAULT_N_COMPONENTS)
    ap.add_argument("--fsd-paper", action="store_true",
                    help="使用论文口径的噪声词典（250 词，需先跑 build_embeddings --noise）")
    ap.add_argument("--eot", action="store_true", help="使用 Sinkhorn 最优传输")
    ap.add_argument("--no-fsd", dest="fsd", action="store_false", help="显式关闭 FSD")
    ap.add_argument("--no-eot", dest="eot", action="store_false", help="显式关闭 EOT")
    ap.add_argument("--slv", action="store_true", help="使用 LLM 逻辑验证（慢，耗 API）")
    ap.add_argument("--threshold", type=float, default=C.DEFAULT_SIM_THRESHOLD)
    ap.add_argument("--n-cand", type=int, default=C.DEFAULT_N_CAND)
    ap.add_argument("--topk-llm", type=int, default=C.DEFAULT_TOPK_LLM)
    ap.add_argument("--tag", default="", help="结果文件名后缀")
    ap.add_argument("--cache", default="", help="指定报告向量缓存文件（默认句级缓存）")
    ap.add_argument("--offset", type=int, default=C.BEDR_EMB_OFFSET, choices=[0, 1],
                    help="BEDR 向量偏移：1=正确对齐（默认），0=复现原代码错位")
    args = ap.parse_args()

    if args.slv:
        args.casd = True

    # ---------- 载入原型的文本与向量（正确对齐） ----------
    from bedr_data import load_bedr_prototypes
    proto_embs, proto_labels, proto_texts_global = load_bedr_prototypes(offset=args.offset)
    print(f"[BEDR] offset={args.offset} "
          f"({'正确对齐' if args.offset == 1 else '复现原代码错位'})")

    label_space = sorted(set(proto_labels))

    # ---------- 载入报告向量 ----------
    cache_path = args.cache or C.REPORT_EMB_CACHE
    if not os.path.exists(cache_path):
        print(f"[FATAL] 缺少报告向量缓存 {cache_path}\n"
              f"        请先运行：python build_embeddings.py "
              f"--limit {args.limit}{' --casd' if args.casd else ''}", file=sys.stderr)
        sys.exit(2)
    Z = np.load(cache_path, allow_pickle=True)
    print(f"[CTI] 缓存: {cache_path}")
    report_ids = list(Z["report_ids"])
    gt_sets = [set(x) for x in Z["gt_ids"]]
    # --- 关键的消融约束：SLV 的输入是 CASD 抽出的原子动作，
    #     因此当 --casd 未指定但 --slv 指定时，SLV 无法在文档级“单独”运行。
    #     此处回退到句级单元，并在结果中记录该约束（详见补充表脚注）。
    if args.slv and not args.casd:
        print("[NOTE] SLV 需要原子动作作为输入；本次以句子级单元回退运行，"
              "该配置在补充表中标注为 'restricted'。")
    if args.casd and "casd_embs" in Z:
        key, txt_key = "casd_embs", "casd_texts"
    elif "sent_embs" in Z:
        key, txt_key = "sent_embs", "sent_texts"
    else:
        print("[FATAL] 缓存中既无 casd_embs 也无 sent_embs", file=sys.stderr)
        sys.exit(2)
    embs_list = list(Z[key])
    texts_list = list(Z[txt_key])
    print(f"[CTI] {len(gt_sets)} 篇报告，动作来源 = {key}")

    # ---------- FSD ----------
    if args.fsd:
        if args.fsd_paper:
            if not os.path.exists(C.NOISE_EMB_CACHE):
                print("[FATAL] 缺少噪声词典向量，请先运行 build_embeddings.py --noise",
                      file=sys.stderr)
                sys.exit(2)
            noise_embs = np.load(C.NOISE_EMB_CACHE)
            print(f"[FSD] 论文口径噪声词典 {noise_embs.shape}")
        else:
            from build_embeddings import embed_texts, NOISE_SEED_TERMS
            noise_embs = embed_texts(NOISE_SEED_TERMS, desc="noise-seed")
            print(f"[FSD] 种子词噪声词典 {noise_embs.shape}（原仓库口径）")
        den = ManifoldDenoiser(noise_embs=noise_embs, n_components=args.fsd_k)
        den.fit()
        proto_embs = den.transform(proto_embs)
        embs_list = [den.transform(e) for e in embs_list]
        print(f"[FSD] 已去噪 k={args.fsd_k}")

    # ---------- LLM 客户端 ----------
    llm_client = None
    if args.slv:
        from openai import OpenAI
        if not C.DEEPSEEK_API_KEY:
            print("[FATAL] --slv 需要 DEEPSEEK_API_KEY", file=sys.stderr)
            sys.exit(2)
        llm_client = OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL)

    # ---------- 逐报告预测 ----------
    preds, r10, r30 = [], [], []
    t0 = time.time()
    for i, (rid, A, txts) in enumerate(zip(report_ids, embs_list, texts_list), 1):
        global action_texts_global
        action_texts_global = list(txts)
        p, s10, s30, _ = predict_report(
            np.asarray(A, dtype=np.float32), proto_embs, proto_labels,
            use_eot=args.eot, threshold=args.threshold, n_cand=args.n_cand,
            topk_llm=args.topk_llm, llm_client=llm_client, slv=args.slv,
        )
        preds.append(p); r10.append(s10); r30.append(s30)
        if i % 20 == 0 or i == len(report_ids):
            print(f"  [{i}/{len(report_ids)}] 用时 {time.time()-t0:.0f}s", flush=True)

    # ---------- 指标 ----------
    y_true, y_pred = build_label_matrix(gt_sets, preds, label_space)
    M = macro_micro_prf(y_true, y_pred)
    rp, rr, rf1 = per_report_mean_prf(gt_sets, preds)
    R10, R30 = recall_at_k(gt_sets, r10), recall_at_k(gt_sets, r30)

    cfg = {
        "offset": args.offset, "action_unit": key,
        "casd": args.casd, "fsd": args.fsd, "fsd_k": args.fsd_k if args.fsd else None,
        "fsd_paper": args.fsd_paper, "eot": args.eot, "slv": args.slv,
        "threshold": args.threshold, "n_cand": args.n_cand, "topk_llm": args.topk_llm,
        "limit": len(gt_sets),
    }

    name = ("CASD" if args.casd else "noCASD") + \
           ("+FSD" if args.fsd else "") + \
           ("+EOT" if args.eot else "") + \
           ("+SLV" if args.slv else "")
    print("\n" + "=" * 70)
    print(f"  配置: {name}   报告数: {len(gt_sets)}")
    print("=" * 70)
    print("  【论文口径】")
    print(f"    Macro-P : {M['macro_p']:.4f}   Macro-R : {M['macro_r']:.4f}   "
          f"Macro-F1 : {M['macro_f1']:.4f}")
    print(f"    Micro-P : {M['micro_p']:.4f}   Micro-R : {M['micro_r']:.4f}   "
          f"Micro-F1 : {M['micro_f1']:.4f}")
    print(f"    Recall@10 : {R10:.4f}   Recall@30 : {R30:.4f}")
    print("  【原仓库代码口径（逐报告平均）】")
    print(f"    P : {rp:.4f}   R : {rr:.4f}   F1 : {rf1:.4f}")
    print("=" * 70)

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    out = {
        "config": cfg, "name": name, "metrics": M,
        "per_report_mean": {"p": rp, "r": rr, "f1": rf1},
        "recall@10": R10, "recall@30": R30,
        "per_class_f1": per_class_f1(y_true, y_pred, label_space),
        "support": support_counts(gt_sets, label_space).tolist(),
        "predictions": {rid: sorted(p) for rid, p in zip(report_ids, preds)},
        "ground_truth": {rid: sorted(g) for rid, g in zip(report_ids, gt_sets)},
    }
    fp = os.path.join(C.RESULTS_DIR, f"ablation_{name}{('_' + args.tag) if args.tag else ''}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[Saved] {fp}")


if __name__ == "__main__":
    main()
