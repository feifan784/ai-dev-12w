# prompt_cot.py 思维链（Chain of Thought）讲解演示
# CoT 核心思想：通过Prompt引导模型展示推理步骤，而非直接输出答案
# 参考：https://arxiv.org/abs/2201.11903
import os
import json
import certifi
import requests
from dotenv import load_dotenv

# SSL配置（解决可能的证书问题）
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv(dotenv_path="/Users/xufeifan/ai-dev-12w/.env")


def llm_chat_stream(messages: list, temperature: float = 0.7):
    """通用LLM流式调用函数，逐token输出"""
    url = os.getenv("LLM_API_URL")
    api_key = os.getenv("LLM_API_KEY")

    if "/chat/completions" not in url:
        url = url.rstrip('/') + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "qwen-turbo",
        "messages": messages,
        "stream": True,
        "temperature": temperature
    }

    response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)

    if response.status_code != 200:
        print(f"错误: {response.status_code}")
        print(f"响应: {response.text[:200]}")
        return

    full_content = []
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data = line[6:]  # 去掉 "data: " 前缀
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

    print()  # 输出结束后换行
    return "".join(full_content)


def demo_standard_vs_cot():
    """
    【对比演示】普通Prompt vs CoT思维链Prompt
    问题：一道需要多步推理的数学题
    """
    question = "小明有15个苹果，他给了小红3个，又从小刚那里得到了一些苹果，此时他有20个。请问小刚给了小明多少个苹果？"

    print("=" * 60)
    print("【CoT思维链对比实验】")
    print("=" * 60)
    print(f"\n问题：{question}\n")

    # ---------------- 方式1：普通Prompt（无思维链）----------------
    print("-" * 60)
    print(">>> 方式1：普通Prompt（直接要答案）")
    print("-" * 60)
    print("模型回答：", end="")
    messages_standard = [
        {"role": "user", "content": question}
    ]
    llm_chat_stream(messages_standard)

    # ---------------- 方式2：CoT Zero-Shot Prompt -----------------
    print("\n" + "-" * 60)
    print(">>> 方式2：CoT Zero-Shot（加上'让我们一步步思考'）")
    print("-" * 60)
    print("模型回答：", end="")
    messages_cot_zero = [
        {"role": "user", "content": f"{question}\n\n请一步步推理，写出思考过程，最后给出答案。"}
    ]
    llm_chat_stream(messages_cot_zero)

    # ---------------- 方式3：CoT Few-Shot Prompt -----------------
    print("\n" + "-" * 60)
    print(">>> 方式3：CoT Few-Shot（给出一个带推理过程的示例）")
    print("-" * 60)
    print("模型回答：", end="")
    messages_cot_few = [
        {"role": "user", "content": f"""在回答之前，请参考以下示例的推理方式：

【示例问题】小红有10颗糖，吃掉3颗后又买了5颗，现在有多少颗？
【示例推理】
1. 初始糖果数：10颗
2. 吃掉3颗后：10 - 3 = 7颗
3. 又买了5颗：7 + 5 = 12颗
4. 最终答案：12颗

现在请用同样的步骤推理以下问题：
{question}"""}
    ]
    llm_chat_stream(messages_cot_few)


def demo_cot_logic_puzzle():
    """
    【进阶演示】逻辑推理题 —— CoT优势更明显
    """
    puzzle = "房间里有A、B、C三人。A说：B在说谎。B说：C在说谎。C说：A和B都在说谎。请问谁在说真话？"

    print("=" * 60)
    print("【CoT逻辑推理题演示】")
    print("=" * 60)
    print(f"\n问题：{puzzle}\n")

    print("-" * 60)
    print(">>> 普通回答")
    print("-" * 60)
    print("模型回答：", end="")
    messages_normal = [
        {"role": "user", "content": puzzle}
    ]
    llm_chat_stream(messages_normal)

    print("\n" + "-" * 60)
    print(">>> CoT 思维链回答（要求逐步分析）")
    print("-" * 60)
    print("模型回答：", end="")
    messages_cot = [
        {"role": "user", "content": f"""{puzzle}

请按照以下步骤逐一分析：
步骤1：假设A说真话，推导B和C的情况，检查是否矛盾。
步骤2：假设B说真话，推导A和C的情况，检查是否矛盾。
步骤3：假设C说真话，推导A和B的情况，检查是否矛盾。
步骤4：综合以上分析，给出最终结论。"""}
    ]
    llm_chat_stream(messages_cot)


def demo_cot_code_debug():
    """
    【实战演示】代码Debug场景 —— CoT帮模型找出bug
    """
    code = """
def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(5))
print(fibonacci(0))
print(fibonacci(-3))
"""

    print("=" * 60)
    print("【CoT代码Debug演示】")
    print("=" * 60)
    print(f"\n待分析代码：\n{code}")

    print("-" * 60)
    print(">>> CoT Debug Prompt（要求逐步分析）")
    print("-" * 60)
    print("模型回答：", end="")
    messages_cot_debug = [
        {"role": "user", "content": f"""请逐步分析以下代码，找出所有潜在问题：

{code}

请按以下步骤分析：
步骤1：逐行解释代码功能。
步骤2：检查逻辑是否正确（特别是边界条件）。
步骤3：检查是否有隐藏的bug或性能问题。
步骤4：给出改进建议和修正后的代码。"""}
    ]
    llm_chat_stream(messages_cot_debug)


if __name__ == "__main__":
    # ============ 思维链(CoT)核心概念说明 ============
    print("""
╔══════════════════════════════════════════════════════════╗
║        CoT 思维链 (Chain of Thought) 讲解               ║
╠══════════════════════════════════════════════════════════╣
║                                                        ║
║  CoT 是一种Prompt工程技术，核心思想是：                  ║
║  引导大模型在回答问题时"展示推理过程"，                  ║
║  而不是直接给出答案。                                   ║
║                                                        ║
║  三种主要形式：                                         ║
║  1. Zero-Shot CoT：仅加"让我们一步步思考"              ║
║  2. Few-Shot CoT：给出带推理步骤的示例                  ║
║  3. 结构化CoT：明确列出分析步骤                         ║
║                                                        ║
║  为什么有效？                                           ║
║  - 将复杂问题分解为子问题，降低单步推理难度             ║
║  - 推理过程可见，便于验证和纠错                         ║
║  - 对于数学、逻辑、编程等任务效果显著                   ║
║                                                        ║
╚══════════════════════════════════════════════════════════╝
    """)

    # 运行三个演示
    demo_standard_vs_cot()
    demo_cot_logic_puzzle()
    demo_cot_code_debug()
