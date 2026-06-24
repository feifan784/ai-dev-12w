import os
import requests
import logging
from typing import TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv

# LangChain / LangGraph 核心
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
# v1.x 混合检索正确导入
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# ===================== 1. 日志配置（全链路日志） =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===================== 2. 初始化LLM、Embedding =====================
load_dotenv()
API_KEY = os.getenv("LLM_API_KEY")
# 通义千问 LLM
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",
    temperature=0
)
# 在线Embedding
embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=API_KEY
)

# ===================== 3. 全局PDF知识库初始化 =====================
PDF_PATH = "/Users/xufeifan/ai-dev-12w/agentic_rag_langgraph/test02.pdf"
# 分块处理
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100, separators=["\n\n", "\n", "。"])
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
chunks = splitter.split_documents(docs)
logger.info(f"PDF分块完成，总分块数量：{len(chunks)}")

# 内存向量库
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="agentic_rag"
)
# 混合检索：稠密向量 + BM25
dense_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10
hybrid_retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.5, 0.5])

# ===================== 4. 全部工具定义 =====================
# 工具1：RAG知识库检索工具
@tool
def query_knowledge_base(question: str) -> str:
    """
    仅当问题和本地PDF文档内容相关时调用，检索内部知识库获取参考资料
    :param question: 用户关于文档内容的提问
    """
    try:
        retrieve_docs = hybrid_retriever.invoke(question)
        context = ""
        for idx, doc in enumerate(retrieve_docs):
            src = doc.metadata.get("source", "未知文档")
            page = doc.metadata.get("page", "无页码")
            context += f"【片段{idx+1} 文件:{src} 页码:{page}】{doc.page_content}\n\n"
        logger.info(f"知识库检索完成，召回片段数：{len(retrieve_docs)}")
        return context if context else "知识库未查询到相关内容"
    except Exception as e:
        logger.error(f"知识库检索异常：{str(e)}")
        return f"知识库查询失败：{str(e)}"

# 工具2：天气查询工具
@tool
def get_city_weather(city: str) -> str:
    """
    查询指定城市实时天气、温度，实时外部数据，PDF文档内不存在
    :param city: 中文城市名称
    """
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=8)
        data = resp.json()
        temp = data["current_condition"][0]["temp_C"]
        desc = data["current_condition"][0]["weatherDesc"][0]["value"]
        res = f"{city}实时天气：{desc}，气温{temp}℃"
        logger.info(f"天气查询成功：{res}")
        return res
    except Exception as e:
        logger.error(f"天气接口异常：{str(e)}")
        return f"天气查询失败：{str(e)}"

# 工具3：模拟用户数据库查询
mock_user_db = {1001:{"name":"张三","city":"北京"},1002:{"name":"李四","city":"上海"},1003:{"name":"王五","city":"广州"}}
@tool
def get_user_info(user_id: int) -> str:
    """
    根据用户数字ID查询用户姓名、居住城市，私有数据库数据，文档不存在
    :param user_id: 用户ID，仅支持1001/1002/1003
    """
    try:
        if user_id in mock_user_db:
            info = mock_user_db[user_id]
            res = f"用户{user_id}：姓名{info['name']}，居住城市{info['city']}"
            logger.info(f"用户数据库查询成功：{res}")
            return res
        else:
            return f"数据库无用户ID {user_id} 的记录"
    except Exception as e:
        logger.error(f"用户数据库查询异常：{str(e)}")
        return f"用户信息查询失败：{str(e)}"

# 工具集合
tools = [query_knowledge_base, get_city_weather, get_user_info]
tool_map = {tool.name: tool for tool in tools}

# ===================== 5. LangGraph 状态定义 =====================
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    route: str  # 路由决策结果（rag / tool / direct）

# ===================== 6. 图节点函数 =====================
# 6.1 路由判断节点：决定走知识库RAG / 外部工具 / 直接回答
def route_judge(state: GraphState):
    messages = state["messages"]
    sys_prompt = SystemMessage(content="""
    你是路由判断专家，根据用户问题选择下一步执行路径：
    1. 如果问题是PDF文档内的业务、接口、功能相关内容 → 返回 "rag"
    2. 如果需要实时天气、用户私有数据库信息 → 返回 "tool"
    3. 通用常识、无需外部数据、无需文档内容 → 返回 "direct"
    仅输出一个单词：rag / tool / direct，不要多余文字
    """)
    res = llm.invoke([sys_prompt] + messages)
    decision = res.content.strip().split()[0]  # 取第一个词，防止多行输出
    logger.info(f"路由决策结果：{decision}")
    return {"route": decision}

# 6.2 RAG知识库检索节点
def node_rag(state: GraphState):
    messages = state["messages"]
    user_q = messages[-1].content
    # 调用RAG工具
    rag_context = query_knowledge_base.invoke({"question": user_q})
    # 把检索上下文加入对话
    new_msg = HumanMessage(content=f"知识库参考资料：\n{rag_context}\n用户原始问题：{user_q}，仅根据参考资料回答，结尾标注文档来源")
    return {"messages": [new_msg]}

# 6.3 外部工具调用节点
def node_tool(state: GraphState):
    messages = state["messages"]
    # 绑定工具，让LLM生成工具调用参数
    llm_with_tools = llm.bind_tools(tools)
    tool_call_res = llm_with_tools.invoke(messages)
    return {"messages": [tool_call_res]}

# 6.4 工具执行解析节点
def node_execute_tool(state: GraphState):
    messages = state["messages"]
    last_msg = messages[-1]
    tool_calls = last_msg.tool_calls
    new_messages = []
    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        tool = tool_map[tool_name]
        try:
            tool_result = tool.invoke(tool_args)
            logger.info(f"执行工具 {tool_name}，返回结果：{tool_result[:100]}")
            new_messages.append(HumanMessage(content=f"工具{tool_name}执行结果：{tool_result}"))
        except Exception as e:
            err_msg = f"工具{tool_name}执行异常：{str(e)}"
            logger.error(err_msg)
            new_messages.append(HumanMessage(content=err_msg))
    return {"messages": new_messages}

# 6.5 最终生成回答节点
def node_generate_answer(state: GraphState):
    messages = state["messages"]
    final_sys = SystemMessage(content="你是专业问答助手，整合所有参考资料、工具结果回答用户问题，禁止编造信息，有文档片段必须标注来源。")
    answer = llm.invoke([final_sys] + messages)
    return {"messages": [answer]}

# ===================== 7. 构建LangGraph流程图 =====================
def build_graph():
    graph = StateGraph(GraphState)
    # 注册所有节点
    graph.add_node("route_judge", route_judge)
    graph.add_node("node_rag", node_rag)
    graph.add_node("node_tool", node_tool)
    graph.add_node("node_execute_tool", node_execute_tool)
    graph.add_node("node_generate_answer", node_generate_answer)
    # 入口
    graph.set_entry_point("route_judge")
    # 分支路由：从状态中提取 route 字段作为路由键
    graph.add_conditional_edges(
        "route_judge",
        lambda state: state["route"],
        {
            "rag": "node_rag",
            "tool": "node_tool",
            "direct": "node_generate_answer"
        }
    )
    # RAG检索完成后进入生成节点
    graph.add_edge("node_rag", "node_generate_answer")
    # 工具生成调用指令 → 执行工具
    graph.add_edge("node_tool", "node_execute_tool")
    # 工具执行完毕 → 生成最终答案
    graph.add_edge("node_execute_tool", "node_generate_answer")
    # 生成回答后结束流程
    graph.add_edge("node_generate_answer", END)
    return graph.compile()

# ===================== 8. 对外调用封装 + 异常捕获 =====================
graph_app = build_graph()
def run_agentic_rag(user_question: str) -> str:
    try:
        logger.info(f"接收用户提问：{user_question}")
        init_msg = [HumanMessage(content=user_question)]
        result = graph_app.invoke({"messages": init_msg})
        final_answer = result["messages"][-1].content
        logger.info("问答流程执行完成")
        return final_answer
    except Exception as main_err:
        logger.error(f"全局流程异常：{str(main_err)}", exc_info=True)
        return f"系统运行异常，请稍后重试：{str(main_err)}"

# ===================== 主程序测试 =====================
if __name__ == "__main__":
    print("===== Agentic RAG 一体化AI助手（LangGraph编排）=====\n")
    test_cases = [
        # 1. 文档相关问题 → 走RAG分支
        "自动播放是否静音，调用的方法名是什么",
        # 2. 实时天气 → 外部工具分支
        "上海今天天气怎么样",
        # 3. 用户数据库查询 → 外部工具分支
        "用户ID1002是谁，住在哪个城市",
        # 4. 混合复杂问题（知识库+外部数据）
        "文档里自动播放相关方法是什么，同时查北京的天气",
        # 5. 通用常识，无需工具/检索
        "什么是LangGraph"
    ]
    for idx, q in enumerate(test_cases, 1):
        print(f"【测试用例{idx}】提问：{q}")
        ans = run_agentic_rag(q)
        print(f"AI回答：{ans}\n" + "-"*80 + "\n")