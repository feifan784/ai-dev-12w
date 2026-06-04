#!/usr/bin/env python3
"""
Ollama 本地模型调用脚本
支持同步调用、流式输出、多轮对话
"""

import os
import json
import requests
from typing import Optional, Generator, List, Dict
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()


class OllamaClient:
    """Ollama 本地模型客户端"""
    
    def __init__(self, base_url: str = None, model: str = None):
        """
        初始化 Ollama 客户端
        
        Args:
            base_url: Ollama API 地址，默认从环境变量读取
            model: 模型名称，默认从环境变量读取
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen:7b")
        
        # 模型参数
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "2048"))
        self.top_p = float(os.getenv("OLLAMA_TOP_P", "0.9"))
        
        # API 端点
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.tags_url = f"{self.base_url}/api/tags"
        
    def is_running(self) -> bool:
        """检查 Ollama 服务是否运行"""
        try:
            response = requests.get(self.tags_url, timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """列出已下载的模型"""
        if not self.is_running():
            return []
        
        response = requests.get(self.tags_url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m['name'] for m in models]
        return []
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        简单生成接口（非对话格式）
        
        Args:
            prompt: 提示词
            stream: 是否流式输出
        
        Returns:
            生成的文本
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": self.top_p
            }
        }
        
        if stream:
            return self._stream_generate(payload)
        else:
            response = requests.post(self.generate_url, json=payload)
            response.raise_for_status()
            return response.json().get('response', '')
    
    def _stream_generate(self, payload: dict) -> str:
        """流式生成（逐字输出）"""
        response = requests.post(self.generate_url, json=payload, stream=True)
        response.raise_for_status()
        
        full_text = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                if 'response' in chunk:
                    print(chunk['response'], end='', flush=True)
                    full_text += chunk['response']
                if chunk.get('done', False):
                    break
        print()  # 换行
        return full_text
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """
        对话接口（支持多轮对话）
        
        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "hello"}]
            stream: 是否流式输出
        
        Returns:
            助手的回复
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": self.top_p
            }
        }
        
        if stream:
            return self._stream_chat(payload)
        else:
            response = requests.post(self.chat_url, json=payload)
            response.raise_for_status()
            return response.json().get('message', {}).get('content', '')
    
    def _stream_chat(self, payload: dict) -> str:
        """流式对话（逐字输出）"""
        response = requests.post(self.chat_url, json=payload, stream=True)
        response.raise_for_status()
        
        full_text = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                    full_text += content
                if chunk.get('done', False):
                    break
        print()  # 换行
        return full_text


def main():
    """主函数：演示各种调用方式"""
    print("=" * 60)
    print("🤖 Ollama 本地模型调用演示")
    print("=" * 60)
    
    # 创建客户端
    client = OllamaClient()
    
    # 1. 检查服务状态
    print("\n📡 检查 Ollama 服务状态...")
    if not client.is_running():
        print("❌ Ollama 服务未运行！请启动 Ollama 应用（菜单栏小狗图标）")
        print("   下载地址: https://ollama.com/download")
        return
    print("✅ Ollama 服务运行正常")
    
    # 2. 显示已下载的模型
    models = client.list_models()
    print(f"\n📦 已下载的模型: {models}")
    print(f"🔧 当前使用的模型: {client.model}")
    
    # 检查指定模型是否存在
    if client.model not in models:
        print(f"\n⚠️  警告: 模型 '{client.model}' 未下载")
        print(f"   请运行: ollama pull {client.model}")
        # 继续执行，让程序尝试调用（可能会报错）
    
    # 3. 简单生成示例（非流式）
    print("\n" + "=" * 60)
    print("示例 1: 简单生成（非流式）")
    print("=" * 60)
    
    prompt = "用一句话介绍什么是人工智能"
    print(f"👤 用户: {prompt}")
    print("🤖 模型: ", end="")
    response = client.generate(prompt, stream=False)
    print(response)
    
    # 4. 流式输出示例
    print("\n" + "=" * 60)
    print("示例 2: 流式输出（逐字显示）")
    print("=" * 60)
    
    prompt = "写一首关于夏天的五言绝句"
    print(f"👤 用户: {prompt}")
    print("🤖 模型: ", end="")
    client.generate(prompt, stream=True)
    
    # 5. 多轮对话示例
    print("\n" + "=" * 60)
    print("示例 3: 多轮对话")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "你好，我叫小明"}
    ]
    
    print(f"👤 用户: {messages[0]['content']}")
    print("🤖 模型: ", end="")
    response1 = client.chat(messages, stream=False)
    print(response1)
    
    # 继续对话（需要传递历史消息）
    messages.append({"role": "assistant", "content": response1})
    messages.append({"role": "user", "content": "我叫什么名字？"})
    
    print(f"👤 用户: {messages[-1]['content']}")
    print("🤖 模型: ", end="")
    response2 = client.chat(messages, stream=False)
    print(response2)
    
    print("\n" + "=" * 60)
    print("✨ 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()