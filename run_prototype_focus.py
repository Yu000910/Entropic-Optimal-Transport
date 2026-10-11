#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_prototype_focus.py — 复现论文 §5.2 的「单元与其最佳原型的平均余弦相似度」度量

论文表述（中文对照）：
    "每个单元变得更聚焦于其所指技术：单元与其最佳匹配原型的平均余弦相似度
     由 0.71 升至 0.82。"

本脚本对两种单元表示分别计算该度量：
    · sent —— 未解耦的句级单元（对应论文的 0.71）
    · casd —— CASD 解耦后的原子动作（对应论文的 0.82）

并对两种「原型」口径分别计算，以便核对论文采用的是哪一种：
    · ref    —— 全部 21,453 条参考句原型（per-reference，论文 §3.4.1 的定义）
    · class  —— 679 个技术类的质心原型

用法：
    python run_prototype_focus.py
    python run_prototype_focus.py --out prototype_focus.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import config as C


# ---------------------------------------------------------------------------
def l2(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def load_prototypes(offset: int = None):
    """载入 BEDR 参考句原型（含偏移校正）。

    注意：BEDR 嵌入缓存比 CSV 多 1 行，原代码等价于 offset=0（整体错位 1 位）；
    config.BEDR_EMB_OFFSET = 1 为正确对齐。
    """
    offset = C.BEDR_EMB_OFFSET if offset is None else offset
    E = np.load(C.BEDR_EMB_CACHE)
    csv = pd.read_csv(C.BEDR_CSV)
    labs = csv["tech_id"].astype(str).tolist()
    embs = E[offset:offset + len(labs)].astype(np.float32)
    assert len(embs) == len(labs), f"嵌入/标签数量不匹配: {len(embs)} vs {len(labs)}"
    return embs, np.array(labs)


def class_centroids(proto_embs: np.ndarray, proto_labels: np.ndarray):
    """按技术类求质心。"""
    classes = sorted(set(proto_labels.tolist()))
    P = np.zeros((len(classes), proto_embs.shape[1]), dtype=np.float32)
    for i, c in enumerate(classes):
        P[i] = proto_embs[proto_labels == c].mean(axis=0)
    return l2(P), classes


def stack_units(seq) -> np.ndarray:
    """把 object 数组（每篇一个 (n_i, d)）堆成 (N, d)。"""
    arrs = [np.asarray(x, dtype=np.float32) for x in seq if len(x)]
    return np.vstack(arrs) if arrs else np.zeros((0, 0), dtype=np.float32)


def mean_best_cosine(units: np.ndarray, protos: np.ndarray) -> float:
    """每个单元与全部原型取最大余弦，再对单元求平均。"""
    if units.size == 0:
        return float("nan")
    S = l2(units) @ l2(protos).T
    return float(S.max(axis=1).mean())


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="report_action_embeddings_casd.npz")
    ap.add_argument("--out", default="prototype_focus.json")
    ap.add_argument("--offset", type=int, default=None)
    args = ap.parse_args()

    cache = os.path.join(C.CACHE_DIR, os.path.basename(args.cache))
    if not os.path.exists(cache):
        print(f"[FATAL] 找不到动作嵌入缓存: {cache}")
        return 1

    z = np.load(cache, allow_pickle=True)
    if "casd_embs" not in z or "sent_embs" not in z:
        print(f"[FATAL] 缓存缺少 casd_embs / sent_embs，实际键: {list(z.keys())}")
        return 1

    proto_embs, proto_labels = load_prototypes(args.offset)
    print(f"[BEDR] 参考句原型 {len(proto_labels)} 条，{len(set(proto_labels.tolist()))} 类")

    sent = stack_units(z["sent_embs"])
    casd = stack_units(z["casd_embs"])
    print(f"[单元] 句级 {sent.shape[0]} 个 → {sent.shape}；CASD 动作 {casd.shape[0]} 个 → {casd.shape}")

    result = {
        "config": {
            "cache": os.path.basename(cache),
            "bedr_offset": C.BEDR_EMB_OFFSET if args.offset is None else args.offset,
            "n_prototypes": int(len(proto_labels)),
            "n_classes": int(len(set(proto_labels.tolist()))),
            "n_sent_units": int(sent.shape[0]),
            "n_casd_units": int(casd.shape[0]),
        },
        "against_reference_prototypes": {
            "sent_mean_best_cosine": mean_best_cosine(sent, proto_embs),
            "casd_mean_best_cosine": mean_best_cosine(casd, proto_embs),
        },
    }

    Pc, classes = class_centroids(proto_embs, proto_labels)
    result["against_class_centroids"] = {
        "n_classes": len(classes),
        "sent_mean_best_cosine": mean_best_cosine(sent, Pc),
        "casd_mean_best_cosine": mean_best_cosine(casd, Pc),
    }

    # ---------- 报告 ----------
    print("\n" + "=" * 68)
    print("  单元与其最佳原型的平均余弦相似度")
    print("=" * 68)
    print(f"  {'原型口径':<28}{'句级(未解耦)':>14}{'CASD 动作':>12}")
    print("-" * 68)
    a = result["against_reference_prototypes"]
    b = result["against_class_centroids"]
    print(f"  {'全部 21,453 条参考句原型':<28}{a['sent_mean_best_cosine']:>14.4f}{a['casd_mean_best_cosine']:>12.4f}")
    print(f"  {'679 个技术类质心':<28}{b['sent_mean_best_cosine']:>14.4f}{b['casd_mean_best_cosine']:>12.4f}")
    print("-" * 68)
    print("  论文 §5.2 报告值:            0.71          0.82")
    print("=" * 68)

    out = os.path.join(C.RESULTS_DIR, os.path.basename(args.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
