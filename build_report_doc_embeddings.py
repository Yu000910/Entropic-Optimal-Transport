#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_report_doc_embeddings.py — 生成"整篇报告"向量（用于严格复现 Naive RAG 基线）

论文 §4.2.1 的 Naive RAG 描述为 "using raw sentence embeddings"，
即把整段文本直接嵌入后贪心匹配，不做 CASD 解耦。本脚本生成这一份向量。

用法：
    python build_report_doc_embeddings.py --limit 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import config as C
from build_embeddings import embed_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=C.DEFAULT_TEST_LIMIT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = os.path.join(C.CACHE_DIR, "report_doc_embeddings.npz")
    if os.path.exists(out) and not args.force:
        print(f"[Skip] {out} 已存在")
        return

    files = sorted(glob.glob(os.path.join(C.CTI_REPORTS_DIR, "*_ground_truth.json")))[: args.limit]
    print(f"[CTI] {len(files)} 篇报告 -> 整篇向量")

    ids, texts, gts = [], [], []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        ids.append(d["report_id"])
        texts.append(d["clean_text"])
        gts.append(sorted(set(d["actual_found_ids"])))

    embs = embed_texts(texts, desc="doc")
    np.savez_compressed(
        out,
        report_ids=np.array(ids, dtype=object),
        raw_texts=np.array(texts, dtype=object),
        gt_ids=np.array(gts, dtype=object),
        doc_embs=embs,
    )
    print(f"[Done] {out}  {embs.shape}")


if __name__ == "__main__":
    main()
