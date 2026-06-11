# AI多轮对话机器人
通义千问 + FastAPI + Streamlit + LangChain新版记忆

## 功能
- 多轮对话
- 上下文记忆
- 前后端分离
- 可部署上线

## 启动
1. 安装依赖（已在虚拟环境安装可跳过此步骤）
pip install -r requirements.txt

2. 启动后端
fastapi dev /Users/xufeifan/ai-dev-12w/multi_chat_project/main.py

3. 启动前端
streamlit run /Users/xufeifan/ai-dev-12w/multi_chat_project/app.py