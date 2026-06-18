import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

# 1. 加载密钥，初始化支持Function Calling的通义千问LLM
load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",
    temperature=0
)

# ===================== 工具1：天气查询工具（真实API调用） =====================
@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气与温度
    :param city: 城市中文名，例如 北京、上海、广州
    """
    # 免费公开简易天气接口
    url = f"https://wttr.in/{city}?format=j1"
    try:
        resp = requests.get(url, timeout=8)
        data = resp.json()
        temp = data["current_condition"][0]["temp_C"]
        desc = data["current_condition"][0]["weatherDesc"][0]["value"]
        return f"{city} 当前天气：{desc}，气温 {temp}℃"
    except Exception as e:
        return f"天气查询失败：{str(e)}，请检查城市名称是否正确"

# ===================== 工具2：模拟用户数据库查询工具 =====================
# 模拟私有数据库内存表
mock_user_database = {
    1001: {"name": "张三", "phone": "13800138000"},
    1002: {"name": "李四", "phone": "13900139000"},
    1003: {"name": "王五", "phone": "13700137000"}
}

@tool
def query_user_db(user_id: int) -> str:
    """
    根据用户ID查询数据库内用户姓名和手机号
    :param user_id: 用户数字ID，仅支持 1001、1002、1003
    """
    if user_id in mock_user_database:
        info = mock_user_database[user_id]
        return f"用户ID{user_id}信息：姓名{info['name']}，手机号{info['phone']}"
    else:
        return f"数据库无ID为{user_id}的用户数据"

# 注册工具列表
tools = [get_weather, query_user_db]

# 创建React Agent（底层基于Function Calling）
agent = create_agent(llm, tools)

# 封装调用函数
def run_agent_query(user_input: str):
    result = agent.invoke({"messages": [("system", "你需要使用工具获取外部实时/数据库信息，禁止编造数据；有对应工具必须调用，不要凭空猜测"), ("user", user_input)]})
    # 取出最终回答
    final_answer = result["messages"][-1].content
    return final_answer

if __name__ == "__main__":
    # 测试1：触发天气工具调用
    print("===== 测试1：查询上海天气 =====")
    ans1 = run_agent_query("滨海现在多少度，天气怎么样？")
    print(ans1, "\n")

    # 测试2：触发数据库查询工具
    print("===== 测试2：查询用户1002信息 =====")
    ans2 = run_agent_query("帮我查一下用户ID1002的姓名和手机号")
    print(ans2, "\n")

    # 测试3：无需调用工具，直接回答
    print("===== 测试3：无工具需求纯文本问答 =====")
    ans3 = run_agent_query("什么是Function Calling函数调用？")
    print(ans3, "\n")

    # 测试4：需要模型自主判断参数、工具不存在的数据
    print("===== 测试4：查询不存在的用户 =====")
    ans4 = run_agent_query("用户ID9999的信息是什么")
    print(ans4)