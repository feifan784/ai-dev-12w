import os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import requests
from dotenv import load_dotenv
from typing import Optional

load_dotenv(dotenv_path="/Users/xufeifan/ai-dev-12w/.env")

class QwenClient:
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_API_URL")
        
        # 构建完整的 API endpoint
        if not base_url:
            raise ValueError("LLM_API_URL not found in .env file")
        
        # 确保 URL 包含正确的路径
        if "/chat/completions" not in base_url:
            base_url = base_url.rstrip('/')
            self.api_url = f"{base_url}/chat/completions"
        else:
            self.api_url = base_url
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in .env file")
    
    def call(self, prompt: str, model: str = "qwen-turbo") -> Optional[str]:
        """调用通义千问 API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        try:
            print(f"发送请求到: {self.api_url}")
            resp = requests.post(self.api_url, json=data, headers=headers, timeout=30)
            
            print(f"状态码: {resp.status_code}")
            
            if resp.status_code == 200:
                result = resp.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"API 错误响应: {resp.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("请求超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {e}")
            return None
        except Exception as e:
            print(f"未知错误: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    try:
        client = QwenClient()
        response = client.call("你好，请简单介绍一下自己")
        if response:
            print(f"回答: {response}")
        else:
            print("调用失败")
    except ValueError as e:
        print(f"配置错误: {e}")