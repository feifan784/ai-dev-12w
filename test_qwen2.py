import os
import requests
import json

def call_qwen(user_message):
    api_key = "sk-900bc88a9559452a8b34989077283d7a"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "qwen-turbo",
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7
    }
    
    print(f"发送请求到: {base_url}/chat/completions")
    print(f"请求头: {headers}")
    print(f"请求数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"\n响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print(f"原始响应内容: {response.text}")
        
        # 检查状态码
        if response.status_code != 200:
            return f"API错误 (状态码: {response.status_code}): {response.text}"
        
        # 尝试解析JSON
        result = response.json()
        print(f"解析后的JSON: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        return f"请求异常: {str(e)}"
    except json.JSONDecodeError as e:
        return f"JSON解析错误: {str(e)}, 原始内容: {response.text}"

# 测试调用
print("=== 测试Qwen API ===\n")
result = call_qwen("你好，请简单介绍一下自己")
print(f"\n最终结果: {result}")