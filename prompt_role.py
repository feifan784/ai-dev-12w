# prompt_role.py 角色设定Prompt
import os
import certifi
import requests
from dotenv import load_dotenv

# SSL配置（解决可能的证书问题）
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv(dotenv_path="/Users/xufeifan/ai-dev-12w/.env")

def role_chat(user_text: str):
    url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")
    
    # 确保URL正确
    if "/chat/completions" not in url:
        url = url.rstrip('/') + "/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    
    # system 定义角色
    messages = [
        {"role": "system", "content": "你是专业文案优化师，语言简洁、正式、通顺"},
        {"role": "user", "content": user_text}
    ]
    
    payload = {
        "model": "qwen-turbo",  # 修正：qwen → qwen-turbo
        "messages": messages
    }
    
    res = requests.post(url, json=payload, headers=headers, timeout=30)
    
    # 检查响应状态
    if res.status_code != 200:
        print(f"错误: {res.status_code}")
        print(f"响应: {res.text[:200]}")
        return None
    
    return res.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    result = role_chat("优化这段文案：我们去公园散步，今天天气真好")
    if result:
        print(result)