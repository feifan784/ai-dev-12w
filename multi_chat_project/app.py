import streamlit as st
import requests

st.title("🤖 AI多轮对话机器人")
st.subheader("基于通义千问 + LangChain新版记忆 + FastAPI")

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_001"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("输入消息...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    res = requests.post(
        "http://127.0.0.1:8000/chat",
        params={"message": user_input, "session_id": st.session_state.session_id}
    )
    reply = res.json()["reply"]

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)