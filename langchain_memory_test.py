import os
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from dotenv import load_dotenv

# 加载你的 .env 文件（路径正确）
load_dotenv("/Users/xufeifan/ai-dev-12w/.env")

# --------------------- 1. 通义千问模型配置 ---------------------
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",
    temperature=0.1
)

# --------------------- 2. 新版【全量记忆】核心（替代 ConversationBufferMemory） ---------------------
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 存储所有会话的历史（全量存储，一句不漏）
store = {}

# 获取/创建会话记忆
def get_session_history(session_id: str):
    if session_id not in store:
        # 这里 = 新版全量存储记忆（和 ConversationBufferMemory 完全一样）
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 提示词模板（必须留位置给聊天历史）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的AI助手，拥有完整的对话记忆。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 记忆自动放这里
    ("human", "{input}")
])

# 构建链
chain = prompt | llm

# 绑定记忆 = 开启全量存储
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# --------------------- 3. 测试多轮对话 ---------------------
print("=" * 50)
print("第一轮对话：")
res1 = chain_with_memory.invoke(
    {"input": "你好，我叫小明，我喜欢打篮球"},
    config={"configurable": {"session_id": "my_session"}}
)
print("AI：", res1.content)

print("=" * 50)
print("第二轮对话（测试记忆）：")
res2 = chain_with_memory.invoke(
    {"input": "我叫什么名字？我喜欢做什么？"},
    config={"configurable": {"session_id": "my_session"}}
)
print("AI：", res2.content)

print("=" * 50)
print("第三轮对话：")
res3 = chain_with_memory.invoke(
    {"input": "我刚才和你说过什么？"},
    config={"configurable": {"session_id": "my_session"}}
)
print("AI：", res3.content)

# --------------------- 4. 查看完整聊天记录（验证全存储） ---------------------
print("\n" + "=" * 50)
print("✅ 确认：当前使用【新版全量记忆】（替代 ConversationBufferMemory）")
print("📝 完整存储的聊天记录：")
for msg in store["my_session"].messages:
    print(f"{msg.type}: {msg.content}")