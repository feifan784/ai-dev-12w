import os
import certifi
import requests
import json
from dotenv import load_dotenv

# SSL配置
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv(dotenv_path="/Users/xufeifan/ai-dev-12w/.env")

def stream_chat(question: str):
    """流式聊天"""
    url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")
    
    # 确保URL正确
    if "/chat/completions" not in url:
        url = url.rstrip('/') + "/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "qwen-turbo",  # 修正模型名称
        "messages": [{"role": "user", "content": question}],
        "stream": True,
        "temperature": 0.7
    }
    
    response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
    
    # 处理流式响应
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data = line[6:]  # 去掉 "data: " 前缀
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)
            except:
                pass

if __name__ == "__main__":
    print("🤖: ", end="")
    stream_chat("说明一下IOS和Android的区别")
    print()  # 换行