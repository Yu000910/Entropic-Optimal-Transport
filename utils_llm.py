# graph_deep_attackg/utils_llm.py

import json
import time
import re
from openai import OpenAI

import os

client = OpenAI(
    # 密钥仅从环境变量读取；请先 export DEEPSEEK_API_KEY=sk-...
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
)

# 1. 抽取函数 (保持 Version 3.0 上下文感知逻辑不变)
def deepseek_extract_attack_graph(report_text: str) -> list[str]:
    system_prompt = """You are a CTI Semantic Analysis Engine.
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
    user_prompt = f"### REPORT CONTENT:\n{report_text[:4000]}\n\n### TASK:\nExtract distinct technical behaviors."

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    for _ in range(3):
        try:
            resp = client.chat.completions.create(model="deepseek-chat", messages=messages, temperature=0.1, response_format={'type': 'json_object'})
            res = json.loads(resp.choices[0].message.content)
            if "attack_actions" in res: return [re.sub(r'\[T\d+(\.\d+)?\]', '', a).strip() for a in res["attack_actions"] if len(a.split()) >= 4]
            for v in res.values(): 
                if isinstance(v, list) and v and isinstance(v[0], str): return v
        except: time.sleep(1)
    return []

# 2. [新增] 对比推理选择器 (Contrastive Selector)
def deepseek_contrastive_select(action_text: str, candidates: list[dict]) -> str:
    """
    输入：
        action_text: 报告中的行为
        candidates: [{'id': 'T1059', 'desc': '...'}, ...] Top-K 候选
    输出：
        选中的最佳 ID，或者 "None"
    """
    
    # 构建选项字符串
    options_str = ""
    for idx, cand in enumerate(candidates):
        # 截断描述以节省 token
        desc_preview = cand['desc'][:150] + "..."
        options_str += f"Option {idx+1}: [{cand['id']}] {desc_preview}\n"
        
    system_prompt = """You are a MITRE ATT&CK Mapping Expert.
Your task is to select the **SINGLE BEST** technique ID that matches the Observed Action.

### INSTRUCTIONS:
1. Compare the Observed Action against the Candidate Options.
2. Select the option that matches the **specific tool, method, or intent** described.
3. Be discriminative! Distinguish between similar techniques (e.g., "Replication via USB" [T1091] vs "Exfiltration via USB" [T1052]).
4. If NONE of the options are a strong semantic match, return "None".

### FORMAT:
Return JSON: {"selected_id": "Txxxx" or "None"}
"""

    user_prompt = f"""
    ### OBSERVED ACTION:
    "{action_text}"

    ### CANDIDATE OPTIONS:
    {options_str}

    ### DECISION:
    Which Option ID is the best fit?
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            response_format={'type': 'json_object'}
        )
        content = json.loads(response.choices[0].message.content)
        return content.get("selected_id", "None")
    except:
        return "None"

# ==============================================================================
# 模块三：高变异性报告生成 (High-Variance Report Generation)
# 用于生成更具挑战性的合成数据
# ==============================================================================

def deepseek_generate_report_json(scenario_data: list[dict], style_reference: str) -> dict:
    """
    功能：生成合成 CTI 报告，强制进行改写和混淆。
    """
    
    system = ('''You are a Senior Cyber Threat Intelligence (CTI) Analyst.
Your task is to write a FICTIONAL, EDUCATIONAL Cybersecurity Advisory report.

### CRITICAL INSTRUCTION: PARAPHRASE & OBFUSCATE
You are provided with a "Reference Context" for each technique. **DO NOT COPY IT VERBATIM.**
1. **Rewrite Completely:** Change vocabulary, sentence structure, and tone.
   - Source: "The malware uses PowerShell to download files."
   - Target: "Analysts observed the adversary leveraging PS scripts to retrieve secondary payloads from remote infrastructure."
2. **Multi-Label:** Combine related techniques into single, complex sentences naturally.
3. **Format:** Return valid JSON.

### JSON STRUCTURE:
{
    "title": "...",
    "summary": "...",
    "clean_text": "The full report text WITHOUT any [Txxxx] tags.", 
    "full_report_text": "The text WITH [Txxxx] tags embedded.",
    "techniques_used": ["Txxxx", ...]
}
''')

    # 构建剧本
    tech_list_str = ""
    for i, item in enumerate(scenario_data, 1):
        tech_list_str += f"{i}. ID: {item['id']} - Name: {item['name']}\n   Ref Context: {item['context']}\n"

    user_prompt = f"""
    ### TARGET SCENARIO:
    {tech_list_str}

    ### REFERENCE STYLE:
    {style_reference[:500]}...

    Generate the report JSON now.
    """
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_prompt}
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=1.1, # 高温增加多样性
                max_tokens=3000,
                response_format={'type': 'json_object'}
            )
            
            content = response.choices[0].message.content
            result_dict = json.loads(content)
            
            # 确保 clean_text 存在
            if "clean_text" not in result_dict and "full_report_text" in result_dict:
                clean = re.sub(r'\[T\d+(\.\d+)?\]', '', result_dict['full_report_text'])
                result_dict['clean_text'] = clean

            if "clean_text" not in result_dict:
                raise KeyError("Missing 'clean_text'")
                
            return result_dict
            
        except Exception as e:
            # print(f"[Generate Error] {e}")
            time.sleep(2)
    
    return {"error": "Max retries exceeded"}

# 自测代码
if __name__ == "__main__":
    print("--- Testing Context-Aware Extraction ---")
    text = "Payload demonstrated the ability to download secondary malware from Google Drive, leveraging web services for command and control."
    print("Input:", text)
    print("Output:", deepseek_extract_attack_graph(text))