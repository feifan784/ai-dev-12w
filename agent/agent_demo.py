import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
# v1新标准接口，替代langgraph.prebuilt.create_react_agent
from langchain.agents import create_agent

load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",
    temperature=0
)

@tool
def calculator(num1: float, num2: float, op: str) -> str:
    """
    简易计算器，支持加减乘除运算
    :param num1: 第一个数字
    :param num2: 第二个数字
    :param op: 运算符，可选 add 加 / sub 减 / mul 乘 / div 除
    """
    if op == "add":
        res = num1 + num2
    elif op == "sub":
        res = num1 - num2
    elif op == "mul":
        res = num1 * num2
    elif op == "div":
        res = num1 / num2
    else:
        return "运算符错误，仅支持 add/sub/mul/div"
    return f"计算结果：{num1} {op} {num2} = {res}"

@tool
def get_current_time() -> str:
    """获取当前系统时间"""
    from datetime import datetime
    return f"当前时间：{datetime.now().strftime('%Y年%m月%d日 %H点%M分%S秒')}"

tools = [calculator, get_current_time]
# 新标准创建，底层封装LangGraph，无弃用警告
agent = create_agent(model=llm, tools=tools)

if __name__ == "__main__":
    res = agent.invoke({"messages": [("user", "125乘以8再加36等于多少？")]})
    print(res["messages"][-1].content)

    res2 = agent.invoke({"messages": [("user", "当前时间是多少？")]})
    print(res2["messages"][-1].content)