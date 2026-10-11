#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baseline_tfidf_svm.py — TF-IDF + SVM 监督基线（按论文口径）

论文 §4.2.1 中该基线的描述：
    "A classic text classification baseline using One-Vs-Rest Support Vector
     Machines with TF-IDF features."

论文 §4.2.4 中的调参配置（用户确认）：
    * 线性核
    * 正则化 C = 1
    * 类别权重 = balanced
    * One-vs-Rest 判定阈值 = 0.5
    * TF-IDF 词表上限 = 50,000 特征
    * unigram 至 bigram

划分：§4.1.3 口径 —— 固定随机种子（seed=42）随机抽取 200 篇测试 / 802 篇训练。

任务形式：**文档级多标签分类**（与本文其余方法同口径），因此采用
    OneVsRestClassifier(LinearSVC) + 决策函数阈值 0.5。

用法：
    python run_baseline_tfidf_svm.py
    python run_baseline_tfidf_svm.py --seed 42 --test-size 200
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import config as C
from metrics import (
    build_label_matrix, macro_micro_prf, per_report_mean_prf, recall_at_k,
)
from splits import load_report_files, load_ground_truth, all_report_files


def report_id_of_local(path: str) -> str:
    """读取报告的 report_id（与 splits.report_id_of 等价，避免额外 import）。"""
    import json as _json
    return _json.load(open(path, encoding="utf-8"))["report_id"]


def topk_recall_sets(clf, X, label_space, k_list=(10, 30)):
    """用决策函数值构造 top-K 召回集合（用于 Recall@K）。"""
    scores = clf.decision_function(X)          # (n, n_classes)
    out = {k: [] for k in k_list}
    for i in range(scores.shape[0]):
        order = np.argsort(scores[i])[::-1]
        for k in k_list:
            out[k].append({label_space[j] for j in order[:k]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--max-features", type=int, default=50000)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="baseline_tfidf_svm.json")
    ap.add_argument("--split-mode", default="random", choices=["random", "first_n"],
                    help="random=论文口径(seed划分)；first_n=历史口径(前N篇作测试)")
    ap.add_argument("--calibrate", action="store_true",
                    help="在校准集上选取决策阈值（避免因阈值不当低估基线）")
    ap.add_argument("--from-cache", default=None,
                    help="从主流程的动作嵌入缓存读取测试集 report_id，"
                         "确保与其它方法评测于完全相同的报告上")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC

    # ---------- 划分（论文口径）----------
    if args.from_cache:
        # 与主流程（run_full_pipeline）使用完全相同的测试报告
        import numpy as _np
        _z = _np.load(os.path.join(C.CACHE_DIR, args.from_cache), allow_pickle=True)
        wanted = [str(x) for x in _z["report_ids"]][: args.test_size]
        test_files = sorted(
            f for f in all_report_files() if report_id_of_local(f) in set(wanted)
        )
        # 同一 report_id 保留一个文件即可，避免重复
        seen, uniq = set(), []
        for f in test_files:
            rid = report_id_of_local(f)
            if rid in seen:
                continue
            seen.add(rid); uniq.append(f)
        test_files = uniq
        print(f"[Split] 从缓存 {args.from_cache} 读取 {len(wanted)} 个 report_id，"
              f"匹配到 {len(test_files)} 个唯一报告")
    else:
        test_files = load_report_files(mode=args.split_mode, n=args.test_size, seed=args.seed)
    test_set = set(test_files)
    train_files = [f for f in all_report_files() if f not in test_set]
    print(f"[Split] 训练 {len(train_files)} 篇 / 测试 {len(test_files)} 篇 "
          f"(mode={args.split_mode}, seed={args.seed})")

    def read(files):
        ids, texts, gts = [], [], []
        for f in files:
            rid, txt, gt = load_ground_truth(f)
            ids.append(rid); texts.append(txt); gts.append(set(gt))
        return ids, texts, gts

    tr_ids, tr_texts, tr_gts = read(train_files)
    te_ids, te_texts, te_gts = read(test_files)

    label_space = sorted({l for g in tr_gts for l in g} | {l for g in te_gts for l in g})
    print(f"[Label space] {len(label_space)} 类")

    Ytr = build_label_matrix(tr_gts, tr_gts, label_space)[0]   # (n_train, C) 多热

    # ---------- 特征 ----------
    print("[TF-IDF] 拟合 ...")
    t0 = time.time()
    vec = TfidfVectorizer(max_features=args.max_features,
                          ngram_range=(1, args.ngram_max),
                          sublinear_tf=True)
    Xtr = vec.fit_transform(tr_texts)
    Xte = vec.transform(te_texts)
    print(f"[TF-IDF] 训练矩阵 {Xtr.shape}，用时 {time.time()-t0:.1f}s")

    # ---------- 分类器 ----------
    def make_clf():
        return OneVsRestClassifier(
            LinearSVC(C=args.C, class_weight="balanced", max_iter=5000, dual="auto"),
            n_jobs=-1,
        )

    threshold = args.threshold
    if args.calibrate:
        # 从训练集划出 20% 作校准集，在其上选取使 Macro-F1 最大的决策阈值
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(tr_texts))
        i_fit, i_cal = train_test_split(idx, test_size=0.2, random_state=args.seed)
        vec_c = TfidfVectorizer(max_features=args.max_features,
                                ngram_range=(1, args.ngram_max), sublinear_tf=True)
        Xf = vec_c.fit_transform([tr_texts[i] for i in i_fit])
        Xc = vec_c.transform([tr_texts[i] for i in i_cal])
        clf_c = make_clf(); clf_c.fit(Xf, Ytr[i_fit])
        sc = clf_c.decision_function(Xc)
        gt_cal = [tr_gts[i] for i in i_cal]
        best, best_thr = -1.0, 0.0
        for thr in np.arange(-1.0, 0.55, 0.05):
            preds_c = [{label_space[j] for j in np.where(sc[i] > thr)[0]}
                       for i in range(sc.shape[0])]
            f1 = per_report_mean_prf(gt_cal, preds_c)[2]
            if f1 > best:
                best, best_thr = f1, float(thr)
        threshold = best_thr
        print(f"[Calib] 校准集最优阈值 = {threshold:.2f}（校准集 Macro-F1 = {best:.4f}）")

    print(f"[SVM] OneVsRest(LinearSVC, C={args.C}, class_weight=balanced) 训练 ...")
    t0 = time.time()
    clf = make_clf()
    clf.fit(Xtr, Ytr)
    print(f"[SVM] 训练完成，用时 {time.time()-t0:.1f}s，决策阈值 = {threshold}")

    # ---------- 预测 ----------
    scores = clf.decision_function(Xte)
    preds = []
    for i in range(scores.shape[0]):
        preds.append({label_space[j] for j in np.where(scores[i] > threshold)[0]})
    r = topk_recall_sets(clf, Xte, label_space, k_list=(10, 30))

    # ---------- 指标 ----------
    yt, yp = build_label_matrix(te_gts, preds, label_space)
    M = macro_micro_prf(yt, yp)
    rp, rr, rf1 = per_report_mean_prf(te_gts, preds)
    R10, R30 = recall_at_k(te_gts, r[10]), recall_at_k(te_gts, r[30])

    print("\n" + "=" * 74)
    print(f"  TF-IDF + SVM（监督基线） | {len(te_gts)} 篇测试 | {len(label_space)} 类")
    print("=" * 74)
    print(f"  Macro-P : {M['macro_p']:.4f}   Macro-R : {M['macro_r']:.4f}   "
          f"Macro-F1 : {M['macro_f1']:.4f}")
    print(f"  Micro-P : {M['micro_p']:.4f}   Micro-R : {M['micro_r']:.4f}   "
          f"Micro-F1 : {M['micro_f1']:.4f}")
    print(f"  逐报告平均 P/R/F1 : {rp:.4f} / {rr:.4f} / {rf1:.4f}")
    print(f"  Recall@10 : {R10:.4f}   Recall@30 : {R30:.4f}")
    print("=" * 74)

    out = {
        "config": {"method": "TF-IDF + SVM", "seed": args.seed,
                   "test_size": args.test_size, "max_features": args.max_features,
                   "ngram_range": [1, args.ngram_max], "C": args.C,
                   "class_weight": "balanced", "threshold": threshold, "calibrated": bool(args.calibrate),
                   "n_classes": len(label_space),
                   "n_train": len(tr_gts), "n_test": len(te_gts)},
        "name": "TF-IDF + SVM", "metrics": M,
        "per_report_mean": {"p": rp, "r": rr, "f1": rf1},
        "recall@10": R10, "recall@30": R30,
        "predictions": {rid: sorted(p) for rid, p in zip(te_ids, preds)},
        "ground_truth": {rid: sorted(g) for rid, g in zip(te_ids, te_gts)},
    }
    fp = os.path.join(C.RESULTS_DIR, args.out)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[Saved] {fp}")


if __name__ == "__main__":
    main()
