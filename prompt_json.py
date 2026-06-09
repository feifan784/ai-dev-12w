import os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import requests
import json
from dotenv import load_dotenv

# 加载配置
load_dotenv("/Users/xufeifan/ai-dev-12w/.env")

# ====================== 核心：强制格式约束 Prompt ======================
SYSTEM_PROMPT = """
你的任务是：永远只返回标准 JSON 格式，绝对不要输出多余文字、解释、markdown、注释。
输出必须是合法 JSON，格式如下：
{
    "输入内容": "用户输入的原文",
    "处理结果": "你的回答",
    "情绪": "正面/负面/中性",
    "关键词": ["关键词1", "关键词2", "关键词3"]
}
只输出 JSON，不要输出任何其他内容！
"""
# ======================================================================


def _get_url():
    """获取完整的 API URL"""
    url = os.getenv("LLM_API_URL")
    if "/chat/completions" not in url:
        url = url.rstrip('/') + "/chat/completions"
    return url


def structured_output(user_input: str, stream: bool = True):
    """
    结构化输出 —— 让 LLM 始终返回 JSON 格式
    参数:
        user_input: 用户输入文本
        stream: 是否使用流式输出（默认True，可看到逐token生成过程）
    返回:
        解析后的 dict
    """
    api_key = os.getenv("LLM_API_KEY")
    api_url = _get_url()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-turbo",
        "messages": messages,
        "temperature": 0.1,  # 低温度 → 格式更稳定
        "stream": stream
    }

    resp = requests.post(api_url, json=payload, headers=headers, stream=stream, timeout=60)

    if resp.status_code != 200:
        return {"错误": f"HTTP {resp.status_code}", "详情": resp.text[:300]}

    if stream:
        # ---------- 流式模式：逐 token 接收并拼接 ----------
        print("模型生成中：", end="", flush=True)
        full_content = []
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        full_content.append(content)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
        print()  # 换行
        raw = "".join(full_content)
    else:
        # ---------- 非流式模式：一次性获取 ----------
        try:
            json_result = resp.json()
            raw = json_result["choices"][0]["message"]["content"]
        except Exception as e:
            return {"错误": f"响应解析失败: {e}", "原始返回": resp.text[:300]}

    # 把返回的字符串转成 JSON 对象
    try:
        # 容错：去掉可能的 markdown 代码块包裹 ```json ... ```
        if raw.strip().startswith("```"):
            lines = raw.strip().split("\n")
            # 去掉首尾的 ``` 行
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines)
        parsed = json.loads(raw)
        return parsed
    except json.JSONDecodeError as e:
        return {"错误": f"JSON解析失败: {e}", "原始返回": raw}


# ===================== 测试 =====================
if __name__ == "__main__":
    # 测试1：非流式
    print("=" * 60)
    print("【测试1：非流式结构化输出】")
    print("=" * 60)
    result1 = structured_output("我今天学习了AI大模型，非常有收获！", stream=False)
    print("=== 结构化输出结果 ===")
    print(json.dumps(result1, ensure_ascii=False, indent=2))

    print("\n")

    # 测试2：流式输出（可看到生成过程）
    print("=" * 60)
    print("【测试2：流式结构化输出】")
    print("=" * 60)
    result2 = structured_output("最近工作压力很大，有点焦虑，不知道该怎么办。")
    print("=== 结构化输出结果 ===")
    print(json.dumps(result2, ensure_ascii=False, indent=2))