#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
splits.py — 统一的报告划分协议（**与论文 §4.1.3 一致**）

论文 §4.1.3 的描述：
    1,002 篇报告被**随机划分**为 802 篇训练（80%）与 200 篇测试（20%）。

数据事实（已核验）：
    * CTI_reports/ 共 1,002 个 ground-truth 文件
    * 唯一 clean_text 数 = **1,002**（内容完全不重复）
    * 因此 1,002 篇是 1,002 份不同内容，随机划分合法

    注：dataset 中存在 report_id 命名重叠（200 个 id 被复用于不同版本的报告），
    这是数据集构建时的标识符疏漏；**报告内容彼此独立**，因此划分以「文件/内容」
    为单位，而非以 report_id 为单位。

本模块是论文所述划分的**权威实现**（canonical implementation）：

    random   : 固定随机种子（默认 42）随机抽取 200 篇测试 / 802 篇训练  ← 论文口径
    first_n  : 按文件名排序取前 n 篇（历史实验代码的口径，仅用于复现旧结果）
    index    : 使用历史产物 test_set_index.json 中的划分

用法：
    from splits import load_report_files, load_ground_truth
    files = load_report_files(mode="random", n=200, seed=42)   # 论文口径
"""
from __future__ import annotations

import glob
import json
import os
import random

import config as C

# 划分索引文件（论文承诺公开的可复现划分）。
# 默认在仓库根目录查找 partition_index.json；也可用环境变量 PARTITION_INDEX 指定。
# 兼容两种记录格式：
#   (a) {"seed":…, "test_size":…, "n_total":…, "test_files":[…], "train_files":[…]}
#   (b) [{"report_id": "report_691", …}, …]
INDEX_FILE = os.environ.get(
    "PARTITION_INDEX", os.path.join(C.REPO_ROOT, "partition_index.json")
)


def all_report_files():
    return sorted(glob.glob(os.path.join(C.CTI_REPORTS_DIR, "*_ground_truth.json")))


def report_id_of(path: str) -> str:
    return json.load(open(path, encoding="utf-8"))["report_id"]


def load_report_files(mode: str = "first_n", n: int = C.DEFAULT_TEST_LIMIT,
                      seed: int = 42) -> list[str]:
    """返回选中的报告文件路径列表。"""
    files = all_report_files()

    if mode == "first_n":
        return files[:n]

    if mode == "random":
        # 论文 §4.1.3 口径：以固定随机种子从 1,002 个文件中抽取 n 篇作为测试集，
        # 其余全部作为训练集。排序后返回以保证结果可复现。
        rng = random.Random(seed)
        picked = rng.sample(files, min(n, len(files)))
        return sorted(picked)

    if mode == "index":
        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError(
                f"缺少划分索引 {INDEX_FILE}\n"
                f"请从仓库下载 partition_index.json，或用环境变量 PARTITION_INDEX 指定其路径。"
            )
        idx = json.load(open(INDEX_FILE, encoding="utf-8"))
        if isinstance(idx, dict):
            # 格式 (a)：直接给出测试文件名列表
            names = set(idx.get("test_files", []))
            if not names:
                raise ValueError(f"{INDEX_FILE} 中缺少 'test_files' 字段")
            return [f for f in files if os.path.basename(f) in names]
        # 格式 (b)：按 report_id 匹配（同一 id 可能有多个时间戳版本，全部纳入）
        wanted = set(x["report_id"] for x in idx)
        return [f for f in files if report_id_of(f) in wanted]

    raise ValueError(f"未知 mode: {mode}")


def load_ground_truth(path: str):
    d = json.load(open(path, encoding="utf-8"))
    return d["report_id"], d["clean_text"], sorted(set(d["actual_found_ids"]))


if __name__ == "__main__":
    for mode in ("first_n", "random", "index"):
        try:
            fs = load_report_files(mode=mode)
            ids = sorted(set(report_id_of(f) for f in fs))
            print(f"{mode:8s}: {len(fs)} 文件 / {len(ids)} 个不同 report_id")
        except Exception as e:
            print(f"{mode:8s}: {type(e).__name__}: {e}")
