#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_embeddings.py — 生成评测所需的向量缓存

产出：
  1. cache/report_action_embeddings.npz
       - report_ids : (N,) 报告 ID
       - raw_text   : (N,) clean_text
       - sent_text  : list[list[str]]  按句切分（近似 CASD 的离线替代）
       - sent_emb   : list[(n_sent, 768)] 句级向量
       - casd_text  : list[list[str]]  LLM(CASD) 抽取的原子动作（--casd 时）
       - casd_emb   : list[(n_act, 768)]
  2. cache/noise_dict_embeddings.npy
       - 论文 §3.3 的噪声词典向量（50 个种子词 + 5-NN 扩展到 250）

用法：
    # 只做句级（快，用于复现 Naive RAG 基线）
    python build_embeddings.py --limit 200

    # 同时做 CASD（需要 DeepSeek API，较慢）
    python build_embeddings.py --limit 200 --casd
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

import config as C

# ---------------------------------------------------------------------------
# 向量 API
# ---------------------------------------------------------------------------
_embed_client = None


def get_embed_client():
    global _embed_client
    if _embed_client is None:
        from openai import OpenAI
        key = C.DASHSCOPE_API_KEY
        if not key:
            print("[FATAL] 未设置 DASHSCOPE_API_KEY", file=sys.stderr)
            sys.exit(2)
        _embed_client = OpenAI(api_key=key, base_url=C.EMBED_BASE_URL)
    return _embed_client


def embed_texts(texts, desc="embedding", max_retry=4):
    """批量嵌入，返回 (len(texts), 768) float32。"""
    if not texts:
        return np.zeros((0, C.EMBED_DIM), dtype=np.float32)
    client = get_embed_client()
    out = []
    for i in range(0, len(texts), C.EMBED_BATCH):
        batch = [t if t.strip() else "N/A" for t in texts[i:i + C.EMBED_BATCH]]
        for attempt in range(max_retry):
            try:
                resp = client.embeddings.create(
                    model=C.EMBED_MODEL, input=batch,
                    dimensions=C.EMBED_DIM, encoding_format="float",
                )
                out.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
                break
            except Exception as e:
                if attempt == max_retry - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        if (i // C.EMBED_BATCH) % 20 == 0:
            print(f"    [{desc}] {i + len(batch)}/{len(texts)}", flush=True)
    return np.vstack(out)


# ---------------------------------------------------------------------------
# 句切分（CASD 的离线近似）
# ---------------------------------------------------------------------------
def sentence_split(text):
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.split()) >= 4]


# ---------------------------------------------------------------------------
# CASD：调用 LLM 抽取原子动作（对齐原仓库 utils_llm.deepseek_extract_attack_graph）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_CASD = """You are a CTI Semantic Analysis Engine.
Your goal is to extract technical attack behaviors for MITRE ATT&CK mapping.

### CRITICAL RULE: CONTEXT-AWARE DISENTANGLEMENT
You must identify distinct technical behaviors.
1. **KEEP TOGETHER (Do NOT Split):**
   - **Tool + Action:** "Adversary used PowerShell to execute a script." -> Keep as one string.
   - **Action + Objective:** "Modified registry keys to establish persistence." -> Keep as one string.
2. **SPLIT (Decouple):**
   - **Unrelated Actions:** If a sentence combines two distinct phases of the kill chain.

### FORMAT
Return a valid JSON object with key "attack_actions" containing a list of strings.
"""


def casd_extract(client, report_text):
    user_prompt = (
        f"### REPORT CONTENT:\n{report_text[:4000]}\n\n"
        f"### TASK:\nExtract distinct technical behaviors."
    )
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=C.LLM_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT_CASD},
                          {"role": "user", "content": user_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            res = json.loads(resp.choices[0].message.content)
            if "attack_actions" in res:
                acts = [re.sub(r"\[T\d+(\.\d+)?\]", "", a).strip()
                        for a in res["attack_actions"]]
                return [a for a in acts if len(a.split()) >= 4]
            for v in res.values():
                if isinstance(v, list) and v and isinstance(v[0], str):
                    return [a for a in v if len(str(a).split()) >= 4]
        except Exception:
            time.sleep(1.0)
    return []


# ---------------------------------------------------------------------------
# 噪声词典（论文 §3.3：50 个种子词 + 5-NN 扩展到 250）
# ---------------------------------------------------------------------------
NOISE_SEED_TERMS = [
    "system", "network", "file", "execute", "process", "command", "tool",
    "attacker", "adversary", "victim", "user", "server", "computer", "host",
    "information", "activity", "action", "operation", "data", "connection",
    "malware", "software", "application", "service", "account", "credential",
    "script", "code", "payload", "content", "resource", "device", "machine",
    "environment", "configuration", "setting", "access", "permission", "log",
    "event", "monitor", "detect", "response", "analysis", "target", "source",
    "destination", "protocol", "traffic", "session",
]


def build_noise_dict(bedr_texts, bedr_embs):
    """种子词向量 + 在 BEDR 词表中找 5-NN 扩展，共 250 个词。"""
    seed_emb = embed_texts(NOISE_SEED_TERMS, desc="noise-seed")
    # 在 BEDR 样本嵌入里找与种子词最相近的样本，取其文本作为扩展词
    P = bedr_embs / (np.linalg.norm(bedr_embs, axis=1, keepdims=True) + 1e-9)
    S = seed_emb / (np.linalg.norm(seed_emb, axis=1, keepdims=True) + 1e-9)
    sim = S @ P.T                                   # (50, 21453)
    picked = set()
    for r in range(sim.shape[0]):
        for j in np.argsort(sim[r])[::-1][:5]:
            picked.add(int(j))
    # 扩大到 250：把扩展样本的文本里的高频词也纳入
    # 说明：论文声称「嵌入空间 5 近邻扩展得到 250 个词」，此处按可复现方式实现：
    #       50 个种子词 + 200 个来自 BEDR 近邻样本的短文本，共 250 条，向量化后做 PCA。
    extra = [bedr_texts[j] for j in sorted(picked)][: 250 - len(NOISE_SEED_TERMS)]
    terms = NOISE_SEED_TERMS + extra
    terms = list(dict.fromkeys(terms))[:250]
    embs = np.vstack([seed_emb, embed_texts(extra, desc="noise-nn")])[: len(terms)]
    return terms, embs.astype(np.float32)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=C.DEFAULT_TEST_LIMIT)
    ap.add_argument("--casd", action="store_true", help="同时调用 LLM 做 CASD 抽取")
    ap.add_argument("--noise", action="store_true", help="生成论文口径的噪声词典向量")
    ap.add_argument("--force", action="store_true", help="忽略已有缓存")
    ap.add_argument("--split", default="first_n", choices=["first_n", "random"],
                    help="报告选择口径：first_n=历史口径；random=论文口径（seed 划分）")
    ap.add_argument("--split-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---------- BEDR 原型（使用正确对齐的加载器） ----------
    from bedr_data import load_bedr_prototypes
    proto_embs, proto_labels, proto_texts = load_bedr_prototypes(offset=C.BEDR_EMB_OFFSET)

    # ---------- 可选：噪声词典 ----------
    if args.noise and (args.force or not os.path.exists(C.NOISE_EMB_CACHE)):
        terms, embs = build_noise_dict(proto_texts, proto_embs)
        np.save(C.NOISE_EMB_CACHE, embs)
        with open(os.path.join(C.CACHE_DIR, "noise_dict_terms.json"), "w", encoding="utf-8") as f:
            json.dump(terms, f, ensure_ascii=False, indent=1)
        print(f"[Noise] 已保存 {len(terms)} 词的噪声词典向量")

    # ---------- 报告 ----------
    out_path = C.REPORT_EMB_CACHE
    split_tag = f"_seed{args.seed}" if args.split == "random" else ""
    if args.casd:
        out_path = os.path.join(C.CACHE_DIR,
                                f"report_action_embeddings_casd{split_tag}.npz")
    if os.path.exists(out_path) and not args.force:
        print(f"[Skip] 已存在 {out_path}（--force 可覆盖）")
        return

    all_files = sorted(glob.glob(os.path.join(C.CTI_REPORTS_DIR, "*_ground_truth.json")))
    if args.split == "random":
        # 论文 §4.1.3 口径：固定随机种子随机抽取的测试划分
        from splits import load_report_files
        files = load_report_files(mode="random", n=args.split_size, seed=args.seed)
        print(f"[CTI] 论文口径划分：seed={args.seed}，随机抽取 {len(files)} 篇测试报告"
              f"（原始文件 {len(all_files)} 个）")
    else:
        # 历史口径：按文件名排序取前 N 篇（仅用于复现旧结果）
        files, seen_ids = [], set()
        for f in all_files:
            rid = json.load(open(f, encoding="utf-8"))["report_id"]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            files.append(f)
            if len(files) >= args.limit:
                break
        print(f"[CTI] 历史口径：取前 {len(files)} 篇（原始文件 {len(all_files)} 个）")

    report_ids, raw_texts, gt_sets = [], [], []
    sent_text_list, sent_emb_list = [], []
    casd_text_list, casd_emb_list = [], []

    llm_client = None
    if args.casd:
        from openai import OpenAI
        if not C.DEEPSEEK_API_KEY:
            print("[FATAL] --casd 需要 DEEPSEEK_API_KEY", file=sys.stderr)
            sys.exit(2)
        llm_client = OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL)

    for n, f in enumerate(files, 1):
        d = json.load(open(f, encoding="utf-8"))
        text = d["clean_text"]
        report_ids.append(d["report_id"])
        raw_texts.append(text)
        gt_sets.append(sorted(set(d["actual_found_ids"])))

        # 句级
        sents = sentence_split(text)
        sent_text_list.append(sents)
        sent_emb_list.append(embed_texts(sents, desc=f"sent#{n}"))

        # CASD
        if args.casd:
            acts = casd_extract(llm_client, text)
            if not acts:
                acts = sents
            casd_text_list.append(acts)
            casd_emb_list.append(embed_texts(acts, desc=f"casd#{n}"))
            print(f"  [{n}/{len(files)}] {d['report_id']}: "
                  f"{len(sents)} 句 -> {len(acts)} 个原子动作", flush=True)

    payload = {
        "report_ids": np.array(report_ids, dtype=object),
        "raw_texts": np.array(raw_texts, dtype=object),
        "gt_ids": np.array(gt_sets, dtype=object),
        "sent_texts": np.array(sent_text_list, dtype=object),
        "sent_embs": np.array(sent_emb_list, dtype=object),
    }
    if args.casd:
        payload["casd_texts"] = np.array(casd_text_list, dtype=object)
        payload["casd_embs"] = np.array(casd_emb_list, dtype=object)

    np.savez_compressed(out_path, **payload)
    print(f"[Done] 已保存 {out_path}")


if __name__ == "__main__":
    main()
