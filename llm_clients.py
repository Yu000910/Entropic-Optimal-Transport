#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_clients.py — 多后端 LLM 统一接口

用途：
  1. 跨 LLM 消融（回应 Reviewer 1 意见 2："与更新的 AI 模型对比"）
  2. 对称消融（对齐 ASCJ 的做法：同一模型既作独立基线、又作流水线引擎）

已探测可用后端（2026-10）：
  DeepSeek  : deepseek-chat, deepseek-flash, deepseek-v4-pro
  DashScope : qwen-max, qwen-plus, qwen-turbo, qwen3-235b-a22b
"""
from __future__ import annotations

import json
import time

import config as C

# 名称 -> (base_url, api_key_env, model_id)
REGISTRY = {
    # ---- DeepSeek ----
    "deepseek-chat":     ("deepseek", "deepseek-chat"),
    "deepseek-flash":    ("deepseek", "deepseek-flash"),
    "deepseek-v4-pro":   ("deepseek", "deepseek-v4-pro"),
    # ---- DashScope / Qwen ----
    "qwen-max":          ("dashscope", "qwen-max"),
    "qwen-plus":         ("dashscope", "qwen-plus"),
    "qwen3-235b":        ("dashscope", "qwen3-235b-a22b"),
}

_BASES = {
    "deepseek": C.LLM_BASE_URL,
    "dashscope": C.EMBED_BASE_URL,
}


def get_client(name: str):
    from openai import OpenAI
    if name not in REGISTRY:
        raise KeyError(f"未知模型 {name}；可选：{list(REGISTRY)}")
    provider, model = REGISTRY[name]
    key = C.DEEPSEEK_API_KEY if provider == "deepseek" else C.DASHSCOPE_API_KEY
    if not key:
        raise RuntimeError(f"{name} 需要 {provider} 的 API key")
    return OpenAI(api_key=key, base_url=_BASES[provider]), model


def chat_json(client, model, system, user, retries=3, max_tokens=1500):
    """调用 LLM 并要求返回 JSON 对象；失败返回 {}。"""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=C.LLM_TEMPERATURE,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return {}


def smoke_test():
    for name in REGISTRY:
        try:
            client, model = get_client(name)
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "reply OK"}],
                temperature=0.0, max_tokens=5,
            )
            print(f"  OK   {name:18s} -> {resp.choices[0].message.content.strip()[:20]}")
        except Exception as e:
            print(f"  FAIL {name:18s} -> {str(e)[:90]}")


if __name__ == "__main__":
    print("=== LLM 后端连通性测试 ===")
    smoke_test()
