#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bootstrap_ci.py — 统计显著性检验（回应 EiC 第 9 条 + reviewer 的统计要求）

对齐 ASCJ 返修时的做法：
  * 非参数 Bootstrap 重采样（有放回，1000 次）→ 95% 置信区间
  * 配对 Bootstrap 检验：Ours vs 每个基线 → p 值
  * 同时报告效应量（Cohen's d / Cliff's delta）

用法：
    python bootstrap_ci.py                                  # 自动配对 results/ 下的结果
    python bootstrap_ci.py --a results/full_pipeline_full200.json \
                           --b results/ablation_noCASD_clean.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import config as C

N_BOOT = 1000
SEED = 42


# ---------------------------------------------------------------------------
def per_report_f1(pred_map, gt_map, keys):
    """逐报告 F1 向量（配对检验用）。"""
    out = []
    for k in keys:
        g = set(gt_map.get(k, []))
        p = set(pred_map.get(k, []))
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return np.array(out, dtype=float)


def micro_f1_boot(pred_map, gt_map, keys_idx, keys):
    """在给定样本下标子集上计算 Micro-F1（bootstrap 重采样用）。"""
    tp = fp = fn = 0
    for i in keys_idx:
        g = set(gt_map.get(keys[i], []))
        p = set(pred_map.get(keys[i], []))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def load_result(path):
    r = json.load(open(path, encoding="utf-8"))
    return r["predictions"], r["ground_truth"], os.path.basename(path)


def bootstrap_ci(pred_map, gt_map, keys, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(keys)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = micro_f1_boot(pred_map, gt_map, idx, keys)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_p(a, b, n_boot=N_BOOT, seed=SEED):
    """配对 bootstrap：H0 为两者无差异。返回双侧 p 值。"""
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    obs = d.mean()
    centered = d - obs
    cnt = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if abs(centered[idx].mean()) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n_boot + 1)


def cliffs_delta(a, b):
    """Cliff's delta 效应量（非参数）。"""
    n = len(a)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (n * n))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(C.RESULTS_DIR, "full_pipeline_full200.json"),
                    help="主方法结果文件")
    ap.add_argument("--b", default="", help="对比方法结果文件（可选）")
    ap.add_argument("--glob", default="", help="批量对比：按 glob 匹配 results 下的文件")
    args = ap.parse_args()

    if not os.path.exists(args.a):
        print(f"[FATAL] 找不到 {args.a}")
        print("        可用结果：")
        for f in sorted(glob.glob(os.path.join(C.RESULTS_DIR, "*.json"))):
            print("          ", os.path.basename(f))
        return

    pa, ga, na = load_result(args.a)
    keys = sorted(set(pa) & set(ga))
    va = per_report_f1(pa, ga, keys)
    m, lo, hi = bootstrap_ci(pa, ga, keys)
    print("=" * 78)
    print(f"  主方法：{na}")
    print("=" * 78)
    print(f"  报告数            : {len(keys)}")
    print(f"  Micro-F1 (逐报告平均): {va.mean():.4f}")
    print(f"  Micro-F1 均值     : {m:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    targets = [args.b] if args.b else sorted(glob.glob(args.glob or
              os.path.join(C.RESULTS_DIR, "ablation_*.json")))
    if not targets:
        return

    print("\n" + "=" * 78)
    print("  与各基线的配对检验")
    print("=" * 78)
    print(f"  {'对比方法':<42} {'F1':>7} {'Δ':>8} {'p 值':>10} {'Cliff δ':>9}")
    print("-" * 78)
    for t in targets:
        if not os.path.exists(t):
            continue
        pb, gb, nb = load_result(t)
        common = sorted(set(keys) & set(pb) & set(gb))
        if not common:
            continue
        vb = per_report_f1(pb, gb, common)
        vaa = per_report_f1(pa, ga, common)
        p = paired_bootstrap_p(vaa, vb)
        d = cliffs_delta(vaa, vb)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(f"  {nb:<42} {vb.mean():>7.4f} {vaa.mean()-vb.mean():>+8.4f} "
              f"{p:>10.2e} {d:>+9.3f}  {sig}")
    print("-" * 78)
    print("  显著性：*** p<0.001, ** p<0.01, * p<0.05, n.s. 不显著")
    print("  Cliff's delta 判读：|δ|<0.147 可忽略, <0.33 小, <0.474 中, 否则大")


if __name__ == "__main__":
    main()
