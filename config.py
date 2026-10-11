#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py — 全局配置与路径

设计原则
--------
1. **不含任何硬编码密钥**。API key 一律从环境变量读取；未设置时给出清晰提示。
2. **路径可移植**。所有路径相对于本仓库自身解析，克隆到任何位置都能运行；
   同时支持用环境变量覆盖，便于把数据放在仓库之外。
3. 所有产出物写入本仓库目录（cache/、results/），这两个目录会被自动创建。

环境变量
--------
    DASHSCOPE_API_KEY     阿里云百炼（DashScope）密钥，用于 text-embedding-v4
    DEEPSEEK_API_KEY      DeepSeek 密钥，用于 CASD 与 SLV 的 LLM 调用
    BEDR_CSV              （可选）BEDR 数据集 CSV 路径
    CTI_REPORTS_DIR       （可选）CTI-1002 语料目录
    BEDR_EMB_CACHE        （可选）BEDR 原型向量缓存 .npy 路径
    EOT_DATA_ROOT         （可选）上述数据文件所在根目录，一次性覆盖前三项
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# 目录（全部相对于仓库自身，保证可移植）
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 仓库根目录（即 github.com/Yu000910/Entropic-Optimal-Transport 的克隆根）
REPO_ROOT = os.environ.get("EOT_DATA_ROOT") or PROJECT_DIR

# ---------------------------------------------------------------------------
# 数据路径
# ---------------------------------------------------------------------------
# 数据文件默认放在仓库根目录下；未找到时提示用户用环境变量指定。
BEDR_CSV = os.environ.get("BEDR_CSV") or os.path.join(REPO_ROOT, "BEDR_dataset.csv")
CTI_REPORTS_DIR = os.environ.get("CTI_REPORTS_DIR") or os.path.join(REPO_ROOT, "CTI_reports")

# BEDR 原型向量缓存（可由 build_embeddings.py 生成，也可从仓库下载）
#   ⚠️ 该缓存有 21,454 行，比 CSV 的 21,453 条数据多一行：
#      第 0 行是 CSV 表头被拼成的字符串（"tech_id,text,source"）的向量，
#      因此从第 1 行起才与 CSV 数据行一一对应（已用 re-embedding 验证，sim=1.0000）。
BEDR_EMB_CACHE = os.environ.get("BEDR_EMB_CACHE") or os.path.join(
    REPO_ROOT, "bedr_embeddings_cache.npy"
)
BEDR_EMB_OFFSET = 1  # 正确的对齐偏移

# 本仓库自身的缓存
REPORT_EMB_CACHE = os.path.join(CACHE_DIR, "report_action_embeddings.npz")
NOISE_EMB_CACHE = os.path.join(CACHE_DIR, "noise_dict_embeddings.npy")

# ---------------------------------------------------------------------------
# 辅助实验脚本用到的外部数据（可选；仅在运行对应脚本时才需要）
# ---------------------------------------------------------------------------
# ATT&CK STIX 定义（官方公布，可在 https://github.com/mitre-attack/attack-stix-data 获取）。
# 各脚本的 --attack-json 参数会覆盖此默认值。
ATTACK_JSON = os.environ.get("ATTACK_JSON", os.path.join(REPO_ROOT, "enterprise-attack.json"))

# 仅 run_eot_attack_base*.py / run_eot_base_ref_on_reports.py / run_eot_contribution*.py
# 等早期消融脚本使用的“技术-文本”配对表（可选）。
TEXT_TECH_CSV = os.environ.get("TEXT_TECH_CSV", os.path.join(REPO_ROOT, "text_tech.csv"))

# ---------------------------------------------------------------------------
# 向量模型（DashScope / 阿里云百炼，OpenAI 兼容接口）
# ---------------------------------------------------------------------------
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-v4")
EMBED_DIM = 768
EMBED_BATCH = 10  # 阿里云限制 batch <= 10

# ---------------------------------------------------------------------------
# LLM（DeepSeek，OpenAI 兼容接口）
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# API keys —— 仅从环境变量读取，本文件不含任何密钥
# ---------------------------------------------------------------------------
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def require_api_keys(*names: str) -> None:
    """在真正需要调用外部服务前检查密钥，缺失时给出可执行的提示。"""
    missing = [n for n in names if not globals().get(n)]
    if missing:
        sys.stderr.write(
            "[config] 缺少 API 密钥: " + ", ".join(missing) + "\n"
            "         请先设置环境变量，例如:\n"
            "             export DASHSCOPE_API_KEY=sk-...\n"
            "             export DEEPSEEK_API_KEY=sk-...\n"
        )
        raise SystemExit(2)


def describe_paths() -> str:
    """返回一份可读的路径自检信息，便于用户排查'文件找不到'的问题。"""
    lines = [
        f"  PROJECT_DIR      {PROJECT_DIR}",
        f"  REPO_ROOT        {REPO_ROOT}",
        f"  BEDR_CSV         {BEDR_CSV}  {'[OK]' if os.path.exists(BEDR_CSV) else '[MISSING]'}",
        f"  CTI_REPORTS_DIR  {CTI_REPORTS_DIR}  {'[OK]' if os.path.isdir(CTI_REPORTS_DIR) else '[MISSING]'}",
        f"  BEDR_EMB_CACHE   {BEDR_EMB_CACHE}  {'[OK]' if os.path.exists(BEDR_EMB_CACHE) else '[MISSING]'}",
        f"  CACHE_DIR        {CACHE_DIR}",
        f"  RESULTS_DIR      {RESULTS_DIR}",
        f"  ATTACK_JSON      {ATTACK_JSON}  {'[OK]' if os.path.exists(ATTACK_JSON) else '[MISSING]'}",
        f"  TEXT_TECH_CSV    {TEXT_TECH_CSV}  {'[OK]' if os.path.exists(TEXT_TECH_CSV) else '[MISSING]'}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 实验默认超参（对齐论文）
# ---------------------------------------------------------------------------
DEFAULT_N_COMPONENTS = 1      # FSD 噪声子空间维数 k（论文 §3.3 取 k=1）
DEFAULT_N_CAND = 30           # 每动作候选数（论文 N_cand = 30）
DEFAULT_TOPK_LLM = 5          # 交给逻辑验证的候选数（论文 K = 5）
DEFAULT_SIM_THRESHOLD = 0.35  # 相似度门槛（在论文配置下从不触发）
DEFAULT_TEST_LIMIT = 200      # 测试集报告数（论文 N = 200）


if __name__ == "__main__":
    print("config.py 路径自检：")
    print(describe_paths())
