import os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import requests
from dotenv import load_dotenv

# 关键：强制加载指定目录下的 .env
load_dotenv(dotenv_path="/Users/xufeifan/ai-dev-12w/.env")

def call_qwen(prompt):
    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_API_URL")
    
    print("正在使用 API KEY:", api_key)  # 调试用
    print("正在使用 API URL:", api_url)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post(api_url, json=data, headers=headers)
    print("API 原始返回:", resp.text)  # 看真实错误
    return resp.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    print(call_qwen("你好"))