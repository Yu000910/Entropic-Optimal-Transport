#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metric_variants.py — 把"同一个预测结果"用 5 种常见口径全部算一遍

目的：定位论文里的 0.738 究竟是哪一种口径。

    A. 逐报告平均 F1           (per-report mean)   <- 原仓库代码口径
    B. Micro-F1                (全局 TP/FP/FN)     <- 多标签分类标准 Micro
    C. Macro-F1（679 类平均）   (论文 §4.2.3 声明)  <- 论文声称的口径
    D. Macro-F1（仅支持类）     (排除零支持类)
    E. Weighted-F1             (按支持数加权)

用法：
    python metric_variants.py results/full_pipeline_full200.json
    python metric_variants.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import config as C


def compute(pred_map, gt_map):
    labels = sorted({l for v in gt_map.values() for l in v} |
                    {l for v in pred_map.values() for l in v})
    idx = {l: i for i, l in enumerate(labels)}
    n, c = len(gt_map), len(labels)
    yt = np.zeros((n, c), dtype=np.int8)
    yp = np.zeros((n, c), dtype=np.int8)
    keys = list(gt_map)
    for i, k in enumerate(keys):
        for l in gt_map[k]:
            yt[i, idx[l]] = 1
        for l in pred_map.get(k, []):
            yp[i, idx[l]] = 1

    tp = (yt * yp).sum(0).astype(float)
    fp = ((1 - yt) * yp).sum(0).astype(float)
    fn = (yt * (1 - yp)).sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pc = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rc = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        fc = np.where(pc + rc > 0, 2 * pc * rc / (pc + rc), 0.0)

    sup = yt.sum(0)
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    mi_p = TP / (TP + FP) if TP + FP else 0.0
    mi_r = TP / (TP + FN) if TP + FN else 0.0
    mi_f = 2 * mi_p * mi_r / (mi_p + mi_r) if mi_p + mi_r else 0.0

    # A. 逐报告平均
    fs = []
    for i, k in enumerate(keys):
        t = (yt[i] * yp[i]).sum(); f = ((1 - yt[i]) * yp[i]).sum(); n_ = (yt[i] * (1 - yp[i])).sum()
        p = t / (t + f) if t + f else 0.0
        r = t / (t + n_) if t + n_ else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    per_report = float(np.mean(fs))

    sup_mask = sup > 0
    return {
        "A_per_report_mean_F1": per_report,
        "B_micro_F1": float(mi_f),
        "B_micro_P": float(mi_p),
        "B_micro_R": float(mi_r),
        "C_macro_F1_all_679": float(fc.mean()),
        "D_macro_F1_supported_only": float(fc[sup_mask].mean()) if sup_mask.any() else 0.0,
        "E_weighted_F1": float((fc * sup).sum() / sup.sum()) if sup.sum() else 0.0,
        "n_reports": n,
        "n_labels_total": c,
        "n_labels_supported": int(sup_mask.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    files = args.files
    if args.all or not files:
        files = sorted(glob.glob(os.path.join(C.RESULTS_DIR, "*.json")))

    print(f"{'结果文件':<42} {'A逐报告':>9} {'B Micro':>9} {'C Macro679':>11} {'D Macro支持':>11} {'E Weighted':>11}")
    print("-" * 100)
    for f in files:
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if "predictions" not in r or "ground_truth" not in r:
            continue
        m = compute(r["predictions"], r["ground_truth"])
        print(f"{os.path.basename(f):<42} {m['A_per_report_mean_F1']:>9.4f} "
              f"{m['B_micro_F1']:>9.4f} {m['C_macro_F1_all_679']:>11.4f} "
              f"{m['D_macro_F1_supported_only']:>11.4f} {m['E_weighted_F1']:>11.4f}")

    print("\n论文对应值：")
    print("  Table 3 / Table 4 中 'Ours' 与 'Full Framework' 行均写作 Macro-F1 = 0.738")
    print("  Naive RAG 基线写作 Macro-F1 = 0.496")
    print("  => 请对照上表判断 0.738 / 0.496 实际对应哪一列。")


if __name__ == "__main__":
    main()
