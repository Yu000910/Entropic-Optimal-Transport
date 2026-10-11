#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bedr_data.py — BEDR 原型与向量缓存的**正确对齐**加载器

背景（重要）：
    历史产物 bedr_embeddings_cache.npy 的形状是 (21454, 768)，而
    BEDR_dataset.csv 只有 21453 条数据（第 1 行是表头）。

    已验证：缓存第 0 行是表头文本 "tech_id,text,source" 的向量，
    缓存第 j+1 行才对应 CSV 第 j 条数据（cos=1.0000）。

    原始仓库代码（Entropic-Optimal-Transport/run_experiment.py:46-67）直接
    返回 X_emb 与 y_labels，未做偏移，因此 labels[j] 实际索引到的是第 j+1 条
    数据的向量 —— 标签与向量整体错配 1 位。

    本模块提供两种模式：
      offset=1 : 正确对齐（推荐，用于复现"应有的"结果）
      offset=0 : 复现原代码的错位状态（用于诊断论文数字来源）
"""
from __future__ import annotations

import os

import numpy as np

import config as C


def load_bedr_prototypes(offset: int = C.BEDR_EMB_OFFSET, verbose: bool = True):
    """返回 (embs, labels, texts)，三者严格一一对应。

    offset=1: embs = cache[1:]        （正确）
    offset=0: embs = cache[:21453]    （复现原代码的错位）
    """
    import pandas as pd

    df = pd.read_csv(
        C.BEDR_CSV, header=0, usecols=[0, 1],
        names=["label", "text"], quotechar='"',
    )
    labels = df["label"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()

    if not os.path.exists(C.BEDR_EMB_CACHE):
        raise FileNotFoundError(
            f"缺少 BEDR 向量缓存：{C.BEDR_EMB_CACHE}\n"
            f"请先运行 build_embeddings.py 生成。"
        )
    cache = np.load(C.BEDR_EMB_CACHE)
    n = len(labels)

    if offset == 1:
        if cache.shape[0] != n + 1:
            raise ValueError(
                f"缓存行数 {cache.shape[0]} 与预期 {n + 1} 不符，无法用 offset=1 对齐。"
            )
        embs = cache[1:]
    elif offset == 0:
        if cache.shape[0] < n:
            raise ValueError(f"缓存行数 {cache.shape[0]} < 标签数 {n}，无法用 offset=0。")
        embs = cache[:n]
    else:
        raise ValueError("offset 只能是 0（复现错位）或 1（正确对齐）")

    if verbose:
        tag = "正确对齐" if offset == 1 else "原代码错位复现"
        print(f"[BEDR] {tag}: {len(labels)} 条数据 / "
              f"{len(set(labels))} 类 / 向量 {embs.shape}")
    assert embs.shape[0] == len(labels) == len(texts)
    return embs, labels, texts


def verify_alignment(probe_idx=(0, 1, 2)):
    """用重新嵌入的方式验证偏移量，返回 (offset, 相似度)。需 API。"""
    from build_embeddings import embed_texts

    embs, labels, texts = load_bedr_prototypes(verbose=False)
    cache = np.load(C.BEDR_EMB_CACHE)
    probe = embed_texts([texts[i] for i in probe_idx], desc="verify")

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    results = {}
    for off in (0, 1):
        sims = [cos(probe[k], cache[i + off]) for k, i in enumerate(probe_idx)]
        results[off] = float(np.mean(sims))
    best = max(results, key=results.get)
    return best, results


if __name__ == "__main__":
    for off in (0, 1):
        load_bedr_prototypes(offset=off)
    print("\n[验证] 重新嵌入并比对偏移量 ...")
    best, res = verify_alignment()
    for off, s in res.items():
        print(f"  offset={off}  平均 cos = {s:.4f}")
    print(f"  => 正确的偏移量是 offset={best}")
