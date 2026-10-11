#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baseline_ttpfshot.py — 复现 TTPFShot (ARES 2025) 于本文数据集

原论文方法（TTP Classification with Minimal Labeled Data, ARES 2025）：
  1. 从 MITRE ATT&CK Enterprise v15.1 的 procedure examples 切句，自动继承 technique ID，
     构成检索库 DSA（原论文：19,747 句 / 780 个 ID）
  2. 用领域句编码器（原论文 SecureBERT）把检索库编码为向量库
  3. 新句用同一编码器编码，按 **L2 距离**取 N=65 个最近句
  4. 拼 Few-Shot 模板（Text / Similarity / Classification 三段），LLM 输出 TTP ID；
     无关则输出 "T0000"
  5. 句级预测 → 文档级聚合

本复现的适配（会在论文中如实说明）：
  * 检索库：以本研究的 BEDR 参照库替代原论文的 DSA（21,453 条 / 679 类）
  * 句编码器：以本研究使用的 text-embedding-v4（768 维）替代 SecureBERT
  * LLM：以 DeepSeek 替代原论文的 GPT-4o
  * N（最近邻数）：保持原论文的 65
  * 输入句：以 CASD 抽取的原子动作作为句级查询单元

用法：
    python run_baseline_ttpfshot.py --limit 200
    python run_baseline_ttpfshot.py --limit 20 --n-shot 65   # 小样本验证
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
from llm_clients import get_client

# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a MITRE ATT&CK TTP classification expert.
You are given an observed sentence from a cyber threat intelligence report, followed by
a set of similar labelled sentences retrieved from a reference knowledge base.
Each reference example shows: [text] the sentence, [similarity] its distance to the
observed sentence, and [technique] its ATT&CK technique or sub-technique identifier.

### TASK
Assign the ATT&CK technique or sub-technique identifier that best matches the observed
sentence.

### RULES
1. Ground your decision in the retrieved reference examples.
2. If the observed sentence does not correspond to any adversarial technique, output "T0000".
3. Multiple identifiers are allowed if the sentence clearly describes several techniques.
4. Output only identifiers that appear in the retrieved examples or in the ATT&CK taxonomy.

### OUTPUT FORMAT (JSON)
{"techniques": ["Txxxx", "Txxxx.xxx"]}
"""


def build_user_prompt(query_text, exemplars):
    lines = [f'[text] {query_text}']
    lines.append('')
    lines.append('### RETRIEVED REFERENCE EXAMPLES')
    for ex in exemplars:
        lines.append(f"[text] {ex['text']}")
        lines.append(f"[similarity] {ex['dist']:.4f}")
        lines.append(f"[technique] {ex['id']}")
        lines.append('')
    lines.append('### OUTPUT')
    return '\n'.join(lines)


def classify(client, model, query_text, exemplars, retries=3):
    user = build_user_prompt(query_text, exemplars)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            ids = data.get("techniques", [])
            if isinstance(ids, str):
                ids = [ids]
            return [str(i).strip() for i in ids if str(i).strip() and str(i).strip() != "T0000"]
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return []


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=C.DEFAULT_TEST_LIMIT)
    ap.add_argument("--n-shot", type=int, default=65, help="检索的最近邻数（原论文 N=65）")
    ap.add_argument("--llm", default="deepseek-chat")
    ap.add_argument("--offset", type=int, default=1, choices=[0, 1])
    ap.add_argument("--tag", default="ttpfshot")
    ap.add_argument("--cache", default="", help="报告句向量缓存（默认用 CASD 缓存）")
    args = ap.parse_args()

    # ---------- 参照库 ----------
    proto_embs, proto_labels, proto_texts = load_bedr_prototypes(offset=args.offset)
    label_space = sorted(set(proto_labels))
    lib = np.asarray(proto_embs, dtype=np.float32)
    lib = lib / (np.linalg.norm(lib, axis=1, keepdims=True) + 1e-9)
    lab_arr = np.array(proto_labels)
    print(f"[RefLib] {lib.shape[0]} 条参照句 / {len(label_space)} 类")

    # ---------- 报告 ----------
    cache_path = args.cache or os.path.join(C.CACHE_DIR, "report_action_embeddings_casd.npz")
    Z = np.load(cache_path, allow_pickle=True)
    rids = list(Z["report_ids"])[: args.limit]
    gt_sets = [set(x) for x in Z["gt_ids"]][: args.limit]
    key = "casd_embs" if "casd_embs" in Z else "sent_embs"
    tkey = "casd_texts" if "casd_texts" in Z else "sent_texts"
    embs = list(Z[key])[: args.limit]
    texts = list(Z[tkey])[: args.limit]
    print(f"[CTI] {len(rids)} 篇报告 | 输入单元 = {key} | N-shot = {args.n_shot}")

    client, model = get_client(args.llm)
    print(f"[LLM] {args.llm} ({model})")

    preds, r10, r30 = [], [], []
    n_llm = 0
    t0 = time.time()

    for idx, (A, units) in enumerate(zip(embs, texts), 1):
        A = np.asarray(A, dtype=np.float32)
        if A.size == 0:
            preds.append(set()); r10.append(set()); r30.append(set()); continue
        Q = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        sim = Q @ lib.T                                   # 余弦相似度
        dist = 1.0 - sim                                  # 转成"距离"用于展示

        # 每篇报告的候选并集（用于 Recall@K）
        s10, s30 = set(), set()
        for r in range(sim.shape[0]):
            o = np.argsort(sim[r])[::-1]
            s10.update(lab_arr[o[:10]].tolist())
            s30.update(lab_arr[o[:30]].tolist())
        r10.append(s10); r30.append(s30)

        report_pred = set()
        for r in range(sim.shape[0]):
            order = np.argsort(sim[r])[::-1][: args.n_shot]
            exemplars = [{"text": proto_texts[j][:200], "dist": float(dist[r, j]),
                          "id": proto_labels[j]} for j in order]
            got = classify(client, model, str(units[r]), exemplars)
            n_llm += 1
            report_pred.update(got)

        preds.append(report_pred)
        if idx % 5 == 0 or idx == len(rids):
            print(f"  [{idx}/{len(rids)}] LLM 调用 {n_llm} 次，用时 {time.time()-t0:.0f}s",
                  flush=True)

    # ---------- 指标 ----------
    yt, yp = build_label_matrix(gt_sets, preds, label_space)
    M = macro_micro_prf(yt, yp)
    rp, rr, rf1 = per_report_mean_prf(gt_sets, preds)
    R10, R30 = recall_at_k(gt_sets, r10), recall_at_k(gt_sets, r30)

    print("\n" + "=" * 74)
    print(f"  TTPFShot (复现) | {len(rids)} 篇 | LLM {n_llm} 次 | N-shot={args.n_shot}")
    print("=" * 74)
    print(f"  Macro-P : {M['macro_p']:.4f}   Macro-R : {M['macro_r']:.4f}   "
          f"Macro-F1 : {M['macro_f1']:.4f}")
    print(f"  Micro-P : {M['micro_p']:.4f}   Micro-R : {M['micro_r']:.4f}   "
          f"Micro-F1 : {M['micro_f1']:.4f}")
    print(f"  逐报告平均 P/R/F1 : {rp:.4f} / {rr:.4f} / {rf1:.4f}")
    print(f"  Recall@10 : {R10:.4f}   Recall@30 : {R30:.4f}")
    print("=" * 74)

    out = {
        "config": {"method": "TTPFShot", "limit": len(rids), "n_shot": args.n_shot,
                   "llm": args.llm, "offset": args.offset, "cache": os.path.basename(cache_path)},
        "name": "TTPFShot", "metrics": M,
        "per_report_mean": {"p": rp, "r": rr, "f1": rf1},
        "recall@10": R10, "recall@30": R30, "n_llm_calls": n_llm,
        "predictions": {r: sorted(p) for r, p in zip(rids, preds)},
        "ground_truth": {r: sorted(g) for r, g in zip(rids, gt_sets)},
    }
    fp = os.path.join(C.RESULTS_DIR, f"baseline_{args.tag}_{args.llm}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[Saved] {fp}")


if __name__ == "__main__":
    main()
