#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py — 严格按论文定义实现的评测指标

论文 §4.2.3 定义：
  Micro-F1 : 全局汇总 TP/FP/FN
  Macro-F1 : 对 K=679 个类各算 F1，再取无权重平均   <-- 这是关键
  Recall@K : |GT ∩ TopK| / |GT|，按报告平均

注意：原始仓库代码（run_experiment.py 等）只做了「逐报告 P/R/F1 再平均」，
      那是 Micro 口径，不是 Macro。本模块提供两种口径以便对照。
"""
from __future__ import annotations

import numpy as np


def build_label_matrix(gt_sets, pred_sets, label_space):
    """构造 (N, C) 0/1 矩阵；列顺序固定为 label_space。"""
    idx = {lab: j for j, lab in enumerate(label_space)}
    n, c = len(gt_sets), len(label_space)
    y_true = np.zeros((n, c), dtype=np.int8)
    y_pred = np.zeros((n, c), dtype=np.int8)
    for i, (g, p) in enumerate(zip(gt_sets, pred_sets)):
        for lab in g:
            if lab in idx:
                y_true[i, idx[lab]] = 1
        for lab in p:
            if lab in idx:
                y_pred[i, idx[lab]] = 1
    return y_true, y_pred


def macro_micro_prf(y_true, y_pred):
    """返回 dict：macro/micro 的 P、R、F1。"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = (y_true * y_pred).sum(axis=0).astype(np.float64)
    fp = ((1 - y_true) * y_pred).sum(axis=0).astype(np.float64)
    fn = (y_true * (1 - y_pred)).sum(axis=0).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_c = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r_c = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f_c = np.where(p_c + r_c > 0, 2 * p_c * r_c / (p_c + r_c), 0.0)

    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    mi_p = TP / (TP + FP) if TP + FP > 0 else 0.0
    mi_r = TP / (TP + FN) if TP + FN > 0 else 0.0
    mi_f = 2 * mi_p * mi_r / (mi_p + mi_r) if (mi_p + mi_r) > 0 else 0.0

    return {
        "macro_p": float(p_c.mean()),
        "macro_r": float(r_c.mean()),
        "macro_f1": float(f_c.mean()),
        "micro_p": float(mi_p),
        "micro_r": float(mi_r),
        "micro_f1": float(mi_f),
        "n_classes": int(y_true.shape[1]),
        "n_classes_with_support": int((y_true.sum(axis=0) > 0).sum()),
    }


def per_class_f1(y_true, y_pred, label_space):
    """返回 {label: F1}，用于长尾分层分析。"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = (y_true * y_pred).sum(axis=0).astype(np.float64)
    fp = ((1 - y_true) * y_pred).sum(axis=0).astype(np.float64)
    fn = (y_true * (1 - y_pred)).sum(axis=0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return {lab: float(f[j]) for j, lab in enumerate(label_space)}


def support_counts(gt_sets, label_space):
    idx = {lab: j for j, lab in enumerate(label_space)}
    sup = np.zeros(len(label_space), dtype=np.int64)
    for g in gt_sets:
        for lab in g:
            if lab in idx:
                sup[idx[lab]] += 1
    return sup


def per_report_mean_prf(gt_sets, pred_sets):
    """原始仓库代码口径：逐报告 P/R/F1 再取平均（≈Micro）。"""
    ps, rs, fs = [], [], []
    for g, p in zip(gt_sets, pred_sets):
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        ps.append(pr); rs.append(rc); fs.append(f1)
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs))


def recall_at_k(gt_sets, retrieved_sets):
    vals = [len(g & r) / len(g) for g, r in zip(gt_sets, retrieved_sets) if g]
    return float(np.mean(vals)) if vals else 0.0
