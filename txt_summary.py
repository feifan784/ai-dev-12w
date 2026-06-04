import os
import certifi
import requests
from dotenv import load_dotenv
from typing import Optional

# 设置 SSL 证书路径（解决可能的证书问题）
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# 加载环境变量 - 指定完整路径
load_dotenv(dotenv_path="/Users/xufeifan/ai-dev-12w/.env")

# 读取文本
def read_txt_file(file_path: str) -> str:
    """读取文本文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 调用LLM生成摘要（改进版）
def get_text_summary(text: str, model: str = "qwen-turbo") -> Optional[str]:
    """
    调用通义千问 API 生成文本摘要
    
    Args:
        text: 要总结的文本内容
        model: 使用的模型名称（默认 qwen-turbo）
    
    Returns:
        摘要文本，失败返回 None
    """
    # 获取配置
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_URL")
    
    # 验证配置
    if not api_key:
        print("❌ 错误：未找到 LLM_API_KEY，请检查 .env 文件")
        return None
    
    if not base_url:
        print("❌ 错误：未找到 LLM_API_URL，请检查 .env 文件")
        return None
    
    # 构建完整的 API endpoint（参考 QwenClient 类）
    if "/chat/completions" not in base_url:
        base_url = base_url.rstrip('/')
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = base_url
    
    # 准备请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 限制文本长度避免超出 token 限制（qwen-turbo 支持约 8k token）
    max_text_length = 3000
    if len(text) > max_text_length:
        text = text[:max_text_length]
        print(f"⚠️ 文本已截断至 {max_text_length} 字符")
    
    # 准备请求数据
    payload = {
        "model": model,  # 使用 qwen-turbo 而不是 qwen
        "messages": [
            {
                "role": "system", 
                "content": "你是一个专业的文本摘要助手。请用简洁、准确的语言总结用户提供的文本内容。"
            },
            {
                "role": "user", 
                "content": f"请对以下内容生成简短摘要（100-200字）：\n\n{text}"
            }
        ],
        "temperature": 0.3  # 降低随机性，使摘要更稳定
    }
    
    try:
        print(f"📤 发送请求到: {api_url}")
        print(f"📊 文本长度: {len(text)} 字符")
        
        # 发送请求（设置超时时间）
        res = requests.post(api_url, json=payload, headers=headers, timeout=60)
        
        print(f"📊 HTTP 状态码: {res.status_code}")
        
        # 检查响应状态
        if res.status_code != 200:
            print(f"❌ API 错误: {res.status_code}")
            print(f"响应内容: {res.text[:500]}")
            return None
        
        # 解析 JSON 响应
        result = res.json()
        
        # 提取摘要内容
        if "choices" in result and len(result["choices"]) > 0:
            if "message" in result["choices"][0]:
                summary = result["choices"][0]["message"]["content"]
                return summary
            elif "text" in result["choices"][0]:
                summary = result["choices"][0]["text"]
                return summary
        
        print(f"❌ 意外的响应格式: {result}")
        return None
        
    except requests.exceptions.Timeout:
        print("❌ 请求超时，请检查网络连接")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {e}")
        print("提示：请检查网络连接和代理设置")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {e}")
        return None
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return None

# 测试 API 连接
def test_api_connection() -> bool:
    """测试通义千问 API 连接是否正常"""
    print("\n=== 测试 API 连接 ===")
    
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_API_URL")
    
    if not api_key or not base_url:
        print("❌ 配置不完整")
        return False
    
    # 构建完整 URL
    if "/chat/completions" not in base_url:
        base_url = base_url.rstrip('/')
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = base_url
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 简单测试请求
    payload = {
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": "回复'OK'"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API 连接成功")
            return True
        else:
            print(f"❌ API 连接失败: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
        return False

# 主函数
if __name__ == "__main__":
    print("=== 文本摘要程序 ===")
    
    # 1. 测试 API 连接
    if not test_api_connection():
        print("\n请检查 .env 文件配置：")
        print("LLM_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        print("LLM_API_KEY=sk-你的API密钥")
        exit(1)
    
    # 2. 读取文本文件
    txt_file_path = "/Users/xufeifan/ai-dev-12w/test.txt"
    try:
        txt_content = read_txt_file(txt_file_path)
        print(f"\n✅ 成功读取文件: {txt_file_path}")
        print(f"📄 文本长度: {len(txt_content)} 字符")
        print(f"📝 文本预览: {txt_content[:200]}...")
    except FileNotFoundError:
        print(f"❌ 找不到文件: {txt_file_path}")
        exit(1)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        exit(1)
    
    # 3. 生成摘要
    print("\n=== 生成摘要 ===")
    summary = get_text_summary(txt_content)
    
    if summary:
        print("\n✅ 文本摘要生成成功！")
        print("=" * 50)
        print("📝 摘要结果：")
        print(summary)
        print("=" * 50)
    else:
        print("\n❌ 生成摘要失败")