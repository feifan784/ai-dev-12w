#!/usr/bin/env python3
"""
Ollama 交互式对话脚本
支持连续多轮对话，保留上下文
"""

import os
import sys
from dotenv import load_dotenv
from ollama_client import OllamaClient

load_dotenv()


def interactive_chat():
    """交互式对话"""
    client = OllamaClient()
    
    print("=" * 60)
    print("🤖 Ollama 本地模型交互式对话")
    print(f"📦 当前模型: {client.model}")
    print("-" * 60)
    print("💡 命令说明:")
    print("   /clear   - 清空对话历史")
    print("   /models  - 查看已下载的模型")
    print("   /switch <模型名> - 切换模型")
    print("   /quit    - 退出程序")
    print("=" * 60)
    
    # 检查服务
    if not client.is_running():
        print("\n❌ Ollama 服务未运行！请先启动 Ollama 应用")
        print("   下载地址: https://ollama.com/download")
        return
    
    # 显示可用模型
    models = client.list_models()
    print(f"\n✅ 已连接的模型: {models}")
    
    if client.model not in models:
        print(f"⚠️  当前配置的模型 '{client.model}' 未下载")
        print(f"   请运行: ollama pull {client.model}")
        
        if models:
            # 使用第一个可用模型
            client.model = models[0]
            print(f"   已自动切换到: {client.model}")
    
    print("\n✨ 开始对话吧！\n")
    
    # 对话历史
    messages = []
    
    while True:
        try:
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input == "/quit":
                print("👋 再见！")
                break
            
            elif user_input == "/clear":
                messages = []
                print("✅ 对话历史已清空\n")
                continue
            
            elif user_input == "/models":
                models = client.list_models()
                print(f"📦 已下载的模型: {models}")
                print(f"🔧 当前使用: {client.model}\n")
                continue
            
            elif user_input.startswith("/switch "):
                new_model = user_input[8:].strip()
                if new_model in client.list_models():
                    client.model = new_model
                    print(f"✅ 已切换到模型: {client.model}\n")
                else:
                    print(f"❌ 模型 '{new_model}' 未下载")
                    print(f"   请运行: ollama pull {new_model}\n")
                continue
            
            # 正常对话
            messages.append({"role": "user", "content": user_input})
            
            print("🤖 助手: ", end="", flush=True)
            
            # 使用流式输出
            response = client.chat(messages, stream=True)
            
            # 将助手回复加入历史
            messages.append({"role": "assistant", "content": response})
            print()  # 空行美化
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")


if __name__ == "__main__":
    interactive_chat()