# json_demo.py
import json

# 字典 -> JSON字符串（序列化）
data_dict = {"title": "AI学习笔记", "content": "LLM基础调用"}
json_str = json.dumps(data_dict, ensure_ascii=False, indent=2)
print("JSON字符串：\n", json_str)

# JSON字符串 -> 字典（反序列化）
parse_dict = json.loads(json_str)
print("\n解析后字典：", parse_dict["title"])