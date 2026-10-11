#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_with_paper.py — 把实测结果与论文里的数字逐项对照

用法：
    python compare_with_paper.py                  # 汇总 results/ 下所有结果
    python compare_with_paper.py results/ablation_xxx.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

import config as C

# ---------------------------------------------------------------------------
# 论文中报告的数字（CAEE 版 sn-article.tex）
# ---------------------------------------------------------------------------
PAPER_TABLE3 = {
    "TF-IDF + SVM":              {"macro_f1": 0.154, "micro_f1": 0.154},
    "TTPDrill":                  {"macro_f1": 0.165, "micro_f1": 0.168},
    "AttacKG":                   {"macro_f1": 0.178, "micro_f1": 0.179},
    "ATHRNN":                    {"macro_f1": 0.185, "micro_f1": 0.185},
    "HM-ACNN":                   {"macro_f1": 0.210, "micro_f1": 0.209},
    "Fine-tuned SecureBERT":     {"macro_f1": 0.231, "micro_f1": 0.231},
    "ACRCNN":                    {"macro_f1": 0.249, "micro_f1": 0.248},
    "Fine-tuned LLaMA-3-8B":     {"macro_f1": 0.215, "micro_f1": 0.215},
    "DeepSeek-V3.2 (Vanilla)":   {"macro_f1": 0.132, "micro_f1": 0.132},
    "DeepSeek-V3.2 (CoT)":       {"macro_f1": 0.147, "micro_f1": 0.147},
    "GPT-4o (Vanilla)":          {"macro_f1": 0.175, "micro_f1": 0.175},
    "GPT-4o (CoT)":              {"macro_f1": 0.196, "micro_f1": 0.196},
    "AttacKG+ (Hierarchical)":   {"macro_f1": 0.185, "micro_f1": 0.185},
    "Naive RAG (Cosine)":        {"macro_f1": 0.496, "micro_f1": 0.495, "recall@30": 0.684},
    "Ours (Test Set, N=200)":    {"macro_f1": 0.738, "micro_f1": 0.744,
                                  "precision": 0.736, "recall": 0.753, "recall@30": 0.937},
    "Ours (Full Set, N=1002)":   {"macro_f1": 0.729, "micro_f1": 0.730, "recall@30": 0.931},
}

PAPER_TABLE4_ABLATION = {
    "1. Baseline (Naive RAG)":   {"P": 0.452, "R": 0.548, "macro_f1": 0.496},
    "2. + Semantic Decoupling":  {"P": 0.584, "R": 0.650, "macro_f1": 0.615},
    "3. + Feature Denoising":    {"P": 0.635, "R": 0.705, "macro_f1": 0.668},
    "4. + OT Alignment":         {"P": 0.651, "R": 0.745, "macro_f1": 0.695},
    "4a. CASD + SLV":            {"P": 0.658, "R": 0.662, "macro_f1": 0.660},
    "4b. CASD + FSD + SLV":      {"P": 0.708, "R": 0.713, "macro_f1": 0.710},
    "5. Full Framework":         {"P": 0.736, "R": 0.753, "macro_f1": 0.738},
}

PAPER_LONGTAIL = {
    "SecureBERT head": 0.78, "SecureBERT torso": 0.45, "SecureBERT tail": 0.12,
    "Ours head": 0.76, "Ours torso": 0.74, "Ours tail": 0.71,
}


def fmt(v):
    return "  n/a " if v is None else f"{v:.4f}"


def table(title, paper, measured):
    print("\n" + "=" * 84)
    print(f"  {title}")
    print("=" * 84)
    print(f"  {'配置 / 方法':<30} {'论文值':>10} {'实测值':>10} {'差值':>10}  判定")
    print("-" * 84)
    for name, pv in paper.items():
        key = pv.get("macro_f1")
        mv = measured.get(name)
        if mv is None:
            print(f"  {name:<30} {fmt(key):>10} {'--':>10} {'--':>10}  未实测")
            continue
        d = mv - key
        flag = "一致" if abs(d) < 0.01 else ("偏大" if d > 0 else "偏小")
        print(f"  {name:<30} {fmt(key):>10} {fmt(mv):>10} {d:>+10.4f}  {flag}")
    print("-" * 84)


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(C.RESULTS_DIR, "*.json")))
    if not files:
        print("results/ 下暂无结果文件，请先运行 run_ablation.py")
        return

    print(f"共 {len(files)} 个结果文件：")
    measured = {}
    for f in files:
        r = json.load(open(f, encoding="utf-8"))
        m = r["metrics"]
        measured[r["name"]] = m["macro_f1"]
        print(f"  - {os.path.basename(f):<45} "
              f"Macro-F1={m['macro_f1']:.4f}  Micro-F1={m['micro_f1']:.4f}  "
              f"R@30={r.get('recall@30', float('nan')):.4f}  cfg={r['config']}")

    # 与论文 Table 4（消融）对照
    print("\n\n" + "#" * 84)
    print("#  与论文 Table 4（消融）对照")
    print("#" * 84)
    alias = {
        "noCASD": "1. Baseline (Naive RAG)",
        "CASD": "2. + Semantic Decoupling",
        "CASD+FSD": "3. + Feature Denoising",
        "CASD+FSD+EOT": "4. + OT Alignment",
        "CASD+SLV": "4a. CASD + SLV",
        "CASD+FSD+SLV": "4b. CASD + FSD + SLV",
        "CASD+FSD+EOT+SLV": "5. Full Framework",
    }
    rows = []
    for f in files:
        r = json.load(open(f, encoding="utf-8"))
        key = alias.get(r["name"])
        if key:
            rows.append((key, r["metrics"]["macro_f1"], r["metrics"]["micro_f1"],
                         r.get("recall@30"), r["name"]))
    rows.sort(key=lambda x: list(PAPER_TABLE4_ABLATION).index(x[0]) if x[0] in PAPER_TABLE4_ABLATION else 99)
    print(f"  {'论文中的配置':<28} {'论文 Macro-F1':>14} {'实测 Macro-F1':>14} {'差值':>10}")
    print("-" * 84)
    for key, mf1, mif1, r30, name in rows:
        pv = PAPER_TABLE4_ABLATION[key]["macro_f1"]
        print(f"  {key:<28} {pv:>14.3f} {mf1:>14.4f} {mf1 - pv:>+10.4f}")
    print("-" * 84)
    print("\n提示：论文 Table 3 的 Naive RAG = 0.496；Table 4 的 Baseline = 0.496（两者同源）。")
    print("      若实测 Naive RAG 与此差距很大，说明论文数字无法由现有代码/数据复现。")


if __name__ == "__main__":
    main()
