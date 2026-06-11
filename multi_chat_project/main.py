from fastapi import FastAPI
from bot import chat_bot

app = FastAPI(title="多轮对话机器人API")

@app.get("/")
def home():
    return {"message": "AI对话机器人运行成功"}

@app.post("/chat")
def chat(message: str, session_id: str = "default"):
    res = chat_bot.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )
    return {"reply": res.content}