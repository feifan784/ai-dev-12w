# prompt_streamlit.py —— Streamlit 网页端 AI 对话
# 整合前文 LLM 调用代码，实现多轮对话 + 流式输出
import os
import json
import certifi
import requests
import streamlit as st
from dotenv import load_dotenv

# ---------- SSL & 环境配置 ----------
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv(dotenv_path="/Users/xufeifan/ai-dev-12w/.env")

# ---------- LLM 调用函数 ----------
def _get_url():
    """获取完整的 API URL"""
    url = os.getenv("LLM_API_URL")
    if "/chat/completions" not in url:
        url = url.rstrip('/') + "/chat/completions"
    return url

def llm_chat_stream(messages: list, temperature: float = 0.7):
    """
    LLM 流式调用，返回生成器，逐 token yield
    用于 Streamlit 中实时显示
    """
    api_key = os.getenv("LLM_API_KEY")
    url = _get_url()

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
        yield f"❌ 错误 HTTP {response.status_code}: {response.text[:300]}"
        return

    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

# ---------- Streamlit 页面 ----------
st.set_page_config(
    page_title="AI 对话助手",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI 对话助手")
st.caption("基于 Qwen-Turbo 模型 · 支持流式输出 · 多轮对话")

# ============ 侧边栏：对话模式 ============
with st.sidebar:
    st.header("⚙️ 设置")

    mode = st.radio(
        "选择对话模式",
        options=["普通聊天", "角色扮演（文案优化师）", "CoT 思维链"],
        index=0,
        help="""
        - 普通聊天：自由对话
        - 角色扮演：设定模型为专业文案优化师
        - CoT 思维链：引导模型逐步推理
        """
    )

    temperature = st.slider("创造性 (Temperature)", min_value=0.0, max_value=1.5, value=0.7, step=0.1,
                           help="越低越保守精确，越高越随机多样")

    # 系统提示词预设
    system_prompts = {
        "普通聊天": "你是一个乐于助人的AI助手，回答简洁、准确、友好。",
        "角色扮演（文案优化师）": "你是专业文案优化师，语言简洁、正式、通顺。请对用户的文案进行优化改写。",
        "CoT 思维链": "你是一个严谨的推理助手。在回答问题时，请一步步展示你的思考过程，最后再给出结论。"
    }

    st.divider()
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

    st.caption("系统提示词：")
    st.code(system_prompts[mode], language="text")

# ============ 初始化对话历史 ============
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============ 渲染历史消息 ============
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============ 用户输入 ============
user_input = st.chat_input("输入你的问题...")

if user_input:
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 构建 messages 列表
    system_content = system_prompts[mode]
    api_messages = [{"role": "system", "content": system_content}]

    # 只取最近 10 轮对话，避免 token 过多
    recent_history = st.session_state.messages[-20:]  # 20条 = 10轮
    api_messages.extend(recent_history)

    # 3. 调用 LLM 流式输出
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = []

        for token in llm_chat_stream(api_messages, temperature=temperature):
            full_response.append(token)
            # 实时更新显示
            placeholder.markdown("".join(full_response) + "▌")

        # 移除光标
        final_text = "".join(full_response)
        placeholder.markdown(final_text)

    # 4. 保存助手消息
    st.session_state.messages.append({"role": "assistant", "content": final_text})

# ============ 底部信息 ============
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"对话模式：{mode}")
with col2:
    st.caption(f"Temperature：{temperature}")
with col3:
    st.caption(f"历史消息数：{len(st.session_state.messages)}")
