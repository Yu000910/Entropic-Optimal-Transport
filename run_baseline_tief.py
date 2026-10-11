#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baseline_tief.py — 复现 TIEF (J. Cybersecurity and Privacy 2025) 于本文数据集

原论文方法（Threat Intelligence Extraction Framework (TIEF) for TTP Extraction）：
  1. PDF 清洗 / 分词 / 去停用词 / 词形还原
  2. BERTopic 句子分组（UMAP + HDBSCAN）
  3. IOC 抽取（13 类 regex）与 soft tagging
  4. **微调 DistilBERT 多标签分类**（sigmoid，阈值 0.391）
     超参：batch 32，lr 1.023e-5，4 epochs，warmup 0.0866，wd 0.0475，max_len 128
  5. STIX 2.1 输出

本复现的适配（会在论文中如实说明）：
  * 段落分组：原论文用 BERTopic（依赖 UMAP/HDBSCAN）；本复现以**句子切分**作为
    分类单元，省略 IOC 与 STIX 模块（这两者不影响 technique 分类指标）
  * 训练集：以 BEDR 参照库（21,453 条句-技术对 / 679 类）作为训练语料
  * 分类头：输出维度 = 679，sigmoid 多标签，阈值保持原论文的 0.391
  * 超参：保持原论文的 batch 32 / lr 1.023e-5 / 4 epochs / warmup 0.0866 / wd 0.0475

用法：
    python run_baseline_tief.py --epochs 4                 # 训练 + 评测（默认 200 篇）
    python run_baseline_tief.py --epochs 1 --limit 20      # 快速验证
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import config as C
from metrics import (
    build_label_matrix, macro_micro_prf, per_report_mean_prf, recall_at_k,
)

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
class TTPDataset(Dataset):
    def __init__(self, texts, multihot, tokenizer, max_len=128):
        self.texts = texts
        self.y = multihot
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.y[i], dtype=torch.float),
        }


def build_multihot(labels, label_space):
    idx = {l: i for i, l in enumerate(label_space)}
    Y = np.zeros((len(labels), len(label_space)), dtype=np.float32)
    for i, l in enumerate(labels):
        if l in idx:
            Y[i, idx[l]] = 1.0
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=C.DEFAULT_TEST_LIMIT)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--train-limit", type=int, default=0, help="只用前 N 条训练（0=全量）")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.023e-5)
    ap.add_argument("--warmup", type=float, default=0.0866)
    ap.add_argument("--wd", type=float, default=0.0475)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.391, help="原论文阈值 0.391")
    ap.add_argument("--out", default="baseline_tief.json")
    ap.add_argument("--save-model", default="cache/tief_model")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[Device] {device}")

    # ---------- 训练语料：BEDR ----------
    df = pd.read_csv(C.BEDR_CSV, header=0, usecols=[0, 1],
                     names=["label", "text"], quotechar='"')
    proto_labels = df["label"].astype(str).tolist()
    proto_texts = df["text"].astype(str).tolist()
    label_space = sorted(set(proto_labels))
    if args.train_limit:
        proto_texts = proto_texts[: args.train_limit]
        proto_labels = proto_labels[: args.train_limit]
    print(f"[Train] BEDR {len(proto_texts)} 条 / {len(label_space)} 类")
    Ytr = build_multihot(proto_labels, label_space)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_space),
        problem_type="multi_label_classification",
    ).to(device)

    ds = TTPDataset(proto_texts, Ytr, tokenizer, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    total_steps = len(dl) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup, anneal_strategy="linear",
    )
    lossf = torch.nn.BCEWithLogitsLoss()

    print(f"[Train] epochs={args.epochs} batch={args.batch} steps/epoch={len(dl)} "
          f"lr={args.lr} warmup={args.warmup} wd={args.wd}")
    t0 = time.time()
    model.train()
    for ep in range(args.epochs):
        run_loss, nb = 0.0, 0
        for step, b in enumerate(dl, 1):
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device))
            loss = lossf(out.logits, b["labels"].to(device))
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            run_loss += loss.item(); nb += 1
            if step % 100 == 0:
                print(f"    ep{ep+1} step{step}/{len(dl)} loss={run_loss/nb:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"  [ep{ep+1}] mean loss = {run_loss/max(nb,1):.4f}  "
              f"用时 {time.time()-t0:.0f}s", flush=True)

    os.makedirs(args.save_model, exist_ok=True)
    model.save_pretrained(args.save_model)
    tokenizer.save_pretrained(args.save_model)
    print(f"[Train] 模型已保存 {args.save_model}")

    # ---------- 评测 ----------
    Z = np.load(os.path.join(C.CACHE_DIR, "report_action_embeddings_casd.npz"),
                allow_pickle=True)
    rids = list(Z["report_ids"])[: args.limit]
    gt_sets = [set(x) for x in Z["gt_ids"]][: args.limit]
    tkey = "casd_texts" if "casd_texts" in Z else "sent_texts"
    texts = list(Z[tkey])[: args.limit]
    print(f"[Eval] {len(rids)} 篇报告 | 阈值 {args.threshold}")

    model.eval()
    preds, r10, r30 = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, units in enumerate(texts, 1):
            doc_pred = set()
            s10, s30 = set(), set()
            if units:
                enc = tokenizer(list(units), truncation=True, padding=True,
                                max_length=args.max_len, return_tensors="pt")
                logits = model(input_ids=enc["input_ids"].to(device),
                               attention_mask=enc["attention_mask"].to(device)).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                for r in range(probs.shape[0]):
                    for j in np.argsort(probs[r])[::-1][:30]:
                        if probs[r, j] >= args.threshold:
                            doc_pred.add(label_space[j])
                    # Recall@K：取每句概率 top-K 的并集（能力上界）
                    for j in np.argsort(probs[r])[::-1][:10]:
                        s10.add(label_space[j])
                    for j in np.argsort(probs[r])[::-1][:30]:
                        s30.add(label_space[j])
            preds.append(doc_pred); r10.append(s10); r30.append(s30)
            if i % 20 == 0 or i == len(rids):
                print(f"  [{i}/{len(rids)}] 用时 {time.time()-t0:.0f}s", flush=True)

    yt, yp = build_label_matrix(gt_sets, preds, label_space)
    M = macro_micro_prf(yt, yp)
    rp, rr, rf1 = per_report_mean_prf(gt_sets, preds)
    R10, R30 = recall_at_k(gt_sets, r10), recall_at_k(gt_sets, r30)

    print("\n" + "=" * 74)
    print(f"  TIEF (复现) | {len(rids)} 篇 | epochs={args.epochs} | thr={args.threshold}")
    print("=" * 74)
    print(f"  Macro-P : {M['macro_p']:.4f}   Macro-R : {M['macro_r']:.4f}   "
          f"Macro-F1 : {M['macro_f1']:.4f}")
    print(f"  Micro-P : {M['micro_p']:.4f}   Micro-R : {M['micro_r']:.4f}   "
          f"Micro-F1 : {M['micro_f1']:.4f}")
    print(f"  逐报告平均 P/R/F1 : {rp:.4f} / {rr:.4f} / {rf1:.4f}")
    print(f"  Recall@10 : {R10:.4f}   Recall@30 : {R30:.4f}")
    print("=" * 74)

    out = {
        "config": {"method": "TIEF", "limit": len(rids), "epochs": args.epochs,
                   "batch": args.batch, "lr": args.lr, "threshold": args.threshold,
                   "max_len": args.max_len, "n_classes": len(label_space)},
        "name": "TIEF", "metrics": M,
        "per_report_mean": {"p": rp, "r": rr, "f1": rf1},
        "recall@10": R10, "recall@30": R30,
        "predictions": {r: sorted(p) for r, p in zip(rids, preds)},
        "ground_truth": {r: sorted(g) for r, g in zip(rids, gt_sets)},
    }
    fp = os.path.join(C.RESULTS_DIR, args.out)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[Saved] {fp}")


if __name__ == "__main__":
    main()
