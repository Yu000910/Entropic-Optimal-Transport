#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_intent_dependency.py — 回应 Reviewer 3 意见 1（互信息/意图依赖的可计算化）

目的
----
原稿式 (2) 用互信息描述原子动作的独立性，但 Intent(a) 没有数学定义、互信息也从未计算。
本脚本给出可计算定义并实测：

    (1) Intent(a) ∈ Δ^14 —— 动作 a 在 14 个 ATT&CK tactic 上的分布
        用 a 的嵌入与该 tactic 下全部 technique prototype 的相似度经 softmax 得到
    (2) Dep(a_p, a_q) = JS( Intent(a_p) || Intent(a_q) )   —— 意图依赖度（0 = 完全独立）
    (3) 对比 CASD 解耦后的原子动作 vs 未解耦的原始句子：
        解耦应使同一报告内动作之间的意图依赖显著降低（即更接近相互独立）

数据
----
    * tactic 映射：ATT&CK v14 enterprise STIX
    * 参照库：BEDR（21,453 条 / 679 类）+ 已缓存的原型嵌入
    * 待测单元：cache/report_action_embeddings_casd.npz
        - sent_embs  —— 未解耦的原始句子
        - casd_embs  —— CASD 解耦后的原子动作

用法
----
    python run_intent_dependency.py --limit 200
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re

import numpy as np

import config as C
from bedr_data import load_bedr_prototypes

# ATT&CK STIX 定义（可选；由环境变量 ATTACK_JSON 指定，默认见 config.py）
ATTACK_JSON = C.ATTACK_JSON

TECH_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


# ---------------------------------------------------------------------------
def load_tactic_map(path=ATTACK_JSON):
    """返回 {technique_id: [tactic_shortname, ...]}（含子技术继承父技术的 tactic）。"""
    with open(path, encoding="utf-8") as f:
        bundle = json.load(f)
    objs = bundle.get("objects", bundle)

    tactics = {}          # x_mitre_shortname -> display name
    for o in objs:
        if o.get("type") == "x-mitre-tactic":
            ext = o.get("external_references", [])
            tid = ext[0].get("external_id") if ext else None
            tactics[o.get("x_mitre_shortname", "")] = {
                "id": tid, "name": o.get("name", "")}

    tech2tac, tech_name = {}, {}
    for o in objs:
        if o.get("type") != "attack-pattern":
            continue
        ext = o.get("external_references", [])
        tid = next((e.get("external_id") for e in ext
                    if TECH_RE.match(str(e.get("external_id", "")))), None)
        if not tid:
            continue
        phases = [p.get("phase_name") for p in o.get("kill_chain_phases", [])
                  if p.get("kill_chain_name") == "mitre-attack"]
        tech2tac[tid] = phases
        tech_name[tid] = o.get("name", "")
    return tech2tac, tech_name, tactics


def build_prototype_index(proto_embs, proto_labels, tech2tac, tactic_order):
    """返回 (P_norm, label_of_class, classes)。P 为每类均值并归一化。"""
    classes = sorted(set(proto_labels))
    P = np.zeros((len(classes), proto_embs.shape[1]), dtype=np.float32)
    lab_arr = np.array(proto_labels)
    for i, c in enumerate(classes):
        P[i] = proto_embs[lab_arr == c].mean(axis=0)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)

    # 每个原型归属的 tactic 集合（子技术继承父技术的 tactic）
    cls_tactics = []
    for c in classes:
        ph = list(tech2tac.get(c, []))
        if not ph and "." in c:
            ph = list(tech2tac.get(c.split(".")[0], []))
        cls_tactics.append([t for t in ph if t in tactic_order])
    return P, classes, cls_tactics


def intent_distribution(action_embs, P, cls_tactics, tactic_index, tau=0.05):
    """动作嵌入 -> tactic 分布（对每个动作，tactic 得分为其下原型相似度的 logsumexp 代理）。"""
    A = np.asarray(action_embs, dtype=np.float32)
    if A.size == 0:
        return np.zeros((0, len(tactic_index)), dtype=np.float64)
    Q = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    S = Q @ P.T                                    # (n_actions, n_classes)
    out = np.zeros((S.shape[0], len(tactic_index)), dtype=np.float64)
    for j, tl in enumerate(cls_tactics):
        if not tl:
            continue
        for t in tl:
            out[:, tactic_index[t]] = np.maximum(out[:, tactic_index[t]], S[:, j])
    # 温度 softmax（只保留正相似度，避免负值带来噪声）
    out = np.maximum(out, 0.0) / max(tau, 1e-9)
    out -= out.max(axis=1, keepdims=True)
    p = np.exp(out)
    p /= (p.sum(axis=1, keepdims=True) + 1e-12)
    return p


def js_divergence(p, q):
    """Jensen-Shannon 散度（以 2 为底，范围 [0,1]）。"""
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mean_pairwise_js(dists):
    """同一报告内所有动作两两之间的平均 JS 散度。"""
    n = dists.shape[0]
    if n < 2:
        return None
    vals = [js_divergence(dists[i], dists[j])
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=C.DEFAULT_TEST_LIMIT)
    ap.add_argument("--tau", type=float, default=0.05, help="tactic 分布 softmax 温度")
    ap.add_argument("--out", default="intent_dependency.json")
    args = ap.parse_args()

    # ---------- tactic 映射 ----------
    tech2tac, tech_name, tactics = load_tactic_map()
    print(f"[ATT&CK] 技术条目 {len(tech2tac)}，tactic {len(tactics)} 个")

    proto_embs, proto_labels, _ = load_bedr_prototypes(offset=1, verbose=False)
    cov = sum(1 for l in set(proto_labels) if tech2tac.get(l) or
              (("." in l) and tech2tac.get(l.split(".")[0])))
    print(f"[BEDR] {len(set(proto_labels))} 类，其中 {cov} 类可映射到 tactic "
          f"({100*cov/len(set(proto_labels)):.1f}%)")

    # ---------- 原型索引 ----------
    # 该数据集内实际出现的 tactic（稳定排序）
    used = set()
    for l in set(proto_labels):
        ph = tech2tac.get(l) or (tech2tac.get(l.split(".")[0]) if "." in l else [])
        used.update(ph)
    tactic_order = sorted(used)
    tactic_index = {t: i for i, t in enumerate(tactic_order)}
    print(f"[Tactics] 使用 {len(tactic_order)} 个: {tactic_order}")

    P, classes, cls_tactics = build_prototype_index(
        proto_embs, proto_labels, tech2tac, tactic_order)
    print(f"[Prototypes] {P.shape[0]} 个类均值原型，维度 {P.shape[1]}")

    # ---------- 对比：未解耦句子 vs CASD 原子动作 ----------
    z = np.load(os.path.join(C.CACHE_DIR, "report_action_embeddings_casd.npz"),
                allow_pickle=True)
    import json as _json
    sent = list(z["sent_embs"])[: args.limit]
    casd = list(z["casd_embs"])[: args.limit]
    print(f"[CTI] {len(sent)} 篇报告（句子 vs CASD 动作）")

    res = {"n_reports": len(sent), "tau": args.tau,
           "n_tactics": len(tactic_order), "tactics": tactic_order,
           "sent": {}, "casd": {}}

    for key, units, label in (("sent", sent, "未解耦句子"),
                              ("casd", casd, "CASD 原子动作")):
        vals, n_units, entropies = [], [], []
        for u in units:
            d = intent_distribution(u, P, cls_tactics, tactic_index, args.tau)
            if d.shape[0] == 0:
                continue
            n_units.append(d.shape[0])
            entropies.extend((-np.sum(d * np.log2(np.maximum(d, 1e-12)), axis=1)).tolist())
            v = mean_pairwise_js(d)
            if v is not None:
                vals.append(v)
        a = np.array(vals)
        res[key] = {
            "mean_pairwise_js": float(a.mean()),
            "median_pairwise_js": float(np.median(a)),
            "std": float(a.std()),
            "n_reports_with_pairs": int(len(a)),
            "mean_units_per_report": float(np.mean(n_units)),
            "mean_intent_entropy": float(np.mean(entropies)),
        }
        print(f"\n[{label}]")
        print(f"  报告数（含≥2单元）: {len(a)}")
        print(f"  单元数/报告: {np.mean(n_units):.2f}")
        print(f"  意图熵 (bits): {np.mean(entropies):.4f}")
        print(f"  报告内平均两两 JS 散度: {a.mean():.4f}  (中位 {np.median(a):.4f}, "
              f"std {a.std():.4f})")

    # ---------- 显著性 ----------
    from scipy import stats
    a_sent = np.array([mean_pairwise_js(
        intent_distribution(u, P, cls_tactics, tactic_index, args.tau)) or np.nan
        for u in sent])
    a_casd = np.array([mean_pairwise_js(
        intent_distribution(u, P, cls_tactics, tactic_index, args.tau)) or np.nan
        for u in casd])
    m = ~(np.isnan(a_sent) | np.isnan(a_casd))
    if m.sum() > 5:
        stat, p = stats.wilcoxon(a_sent[m], a_casd[m])
        res["wilcoxon"] = {"statistic": float(stat), "p": float(p),
                           "n_pairs": int(m.sum())}
        print(f"\n[Wilcoxon 配对检验] 未解耦 vs CASD：统计量={stat:.1f}，p={p:.3e}，"
              f"n={m.sum()}")
        print(f"  平均降低: {a_sent[m].mean()-a_casd[m].mean():+.4f}"
              f"  （相对 {(a_sent[m].mean()-a_casd[m].mean())/max(a_sent[m].mean(),1e-9)*100:.1f}%）")

    fp = os.path.join(C.RESULTS_DIR, args.out)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n[Saved] {fp}")


if __name__ == "__main__":
    main()
