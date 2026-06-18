import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

# 1. 初始化通义千问LLM，原生支持Function Calling
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",
    temperature=0
)

# ===================== 工具1：天气查询 =====================
@tool
def get_weather(city: str) -> str:
    """
    查询指定城市实时天气、温度
    :param city: 中文城市名，如上海、北京
    """
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=8)
        data = resp.json()
        temp = data["current_condition"][0]["temp_C"]
        desc = data["current_condition"][0]["weatherDesc"][0]["value"]
        return f"{city}：{desc}，气温{temp}℃"
    except Exception as e:
        return f"天气查询失败：{str(e)}"

# ===================== 工具2：模拟用户数据库 =====================
mock_user_db = {
    1001: {"name": "张三", "city": "北京", "phone": "13800138000"},
    1002: {"name": "李四", "city": "上海", "phone": "13900139000"},
    1003: {"name": "王五", "city": "广州", "phone": "13700137000"}
}

@tool
def query_user_info(user_id: int) -> str:
    """
    根据用户ID查询用户姓名、所在城市、手机号
    :param user_id: 数字用户ID，仅支持1001/1002/1003
    """
    if user_id in mock_user_db:
        u = mock_user_db[user_id]
        return f"用户{user_id}：姓名{u['name']}，居住城市{u['city']}，手机号{u['phone']}"
    else:
        return f"数据库不存在ID={user_id}的用户"

# 注册工具列表
tools = [get_weather, query_user_info]

# 创建标准ReAct智能体（底层自动实现Reason+Act交替循环）
react_agent = create_agent(llm, tools)

def run_complex_task(query: str):
    """执行复杂多步骤任务，流式打印完整思考+行动链路"""
    sys_prompt = """
    你是ReAct智能体，遵循思考→行动交替逻辑分步解决用户问题。
    规则：
    1. 不能编造任何用户数据、天气信息，必须调用工具获取；
    2. 复杂问题需要分步调用多个工具，完成所有子任务再汇总回答；
    3. 拿到工具返回结果后，重新判断是否还需要继续调用工具；
    4. 所有子问题全部获取数据后，再输出完整综合答案。
    """
    print("========== ReAct 流式执行链路 ==========\n")
    final_ans = ""

    # stream + stream_mode="messages" 实现逐 token 流式输出
    for msg, _ in react_agent.stream(
        {
            "messages": [
                ("system", sys_prompt),
                ("user", query)
            ]
        },
        stream_mode="messages"
    ):
        msg_type = type(msg).__name__

        if msg_type == "AIMessageChunk":
            # LLM 逐 token 流式输出文本
            if msg.content:
                print(msg.content, end="", flush=True)
                final_ans += msg.content
            # 检测工具调用（流式分片到达，首次出现 name 时打印）
            if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                for tc in msg.tool_call_chunks:
                    if tc.get("name"):
                        print(f"\n  → 调用工具: {tc['name']}")

        elif msg_type == "ToolMessage":
            # 工具执行结果返回
            print(f"\n  ← 工具返回: {msg.content}\n")
            final_ans = ""  # 重置，下一段 AI 输出为新一轮思考

    print("\n\n========== 最终综合答案 ==========")
    return final_ans

if __name__ == "__main__":
    # 复杂多步骤测试用例（必须两轮工具调用）
    complex_question = "先查询用户ID1002的完整信息，再查询该用户所在城市的实时天气，最后整合所有信息给我"
    answer = run_complex_task(complex_question)
    print(answer)