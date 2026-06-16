import os
import shutil
# HuggingFace国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 消除hf废弃警告（可选）
# os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from typing import List
# BGE向量化模型
from FlagEmbedding import FlagModel
# LangChain基础组件
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
# v1.x 混合检索正确导入
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

# ===================== 1. 初始化本地BGE稠密Embedding =====================
def init_bge_embedding():
    # 线上仓库自动缓存，规避本地目录不存在问题
    bge_model = FlagModel(
        model_name_or_path="BAAI/bge-base-zh-v1.5",
        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
        use_fp16=True
    )
    # 适配LangChain标准Embedding接口
    class LocalBGEEmbedding(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """批量文档向量化（知识库分块使用）"""
            return bge_model.encode(texts).tolist()
        def embed_query(self, text: str) -> List[float]:
            """单条问题向量化（用户提问使用）"""
            return bge_model.encode_queries([text])[0].tolist()
    return LocalBGEEmbedding()

embedding = init_bge_embedding()

# ===================== 2. 初始化通义千问LLM =====================
def init_llm():
    load_dotenv("/Users/xufeifan/ai-dev-12w/.env")
    llm = ChatOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        temperature=0.01  # 低随机性，严格依赖参考文档作答
    )
    return llm
llm = init_llm()

# ===================== 3. 文档加载 + 重叠语义分块 =====================
def load_and_split_doc(file_path: str):
    """
    加载PDF/TXT文档，执行重叠分块
    :param file_path: 文档绝对路径
    :return: 分块后的Document列表
    """
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("仅支持 .txt / .pdf 格式文档")
    raw_docs = loader.load()
    # 递归语义分割器：优先按段落/句子切割，保证语义完整
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，"]
    )
    chunk_docs = splitter.split_documents(raw_docs)
    print(f"文档分块完成，总分块数量：{len(chunk_docs)}")
    return chunk_docs

# ===================== 4. Chroma稠密向量库构建 =====================
CHROMA_DB_PATH = "./chroma_hybrid_db"
def build_chroma_vector_store(chunks):
    """构建持久化Chroma向量库"""
    # 清空旧库，避免历史数据干扰检索
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DB_PATH,
        collection_name="hybrid_kb"
    )
    print("稠密向量库构建完成")
    return vector_db

# ===================== 5. 混合检索器：稠密向量 + BM25稀疏关键词融合 =====================
def get_hybrid_ensemble_retriever(all_chunks, top_k=10):
    """
    混合两路检索融合
    1. 稠密向量检索：捕捉深层语义、同义表述
    2. BM25稀疏检索：精准匹配专有名词、方法名、关键词字面匹配
    :param all_chunks: 全部分块文档，用于初始化BM25索引
    :param top_k: 单路检索返回候选数量
    :return: 融合检索器
    """
    # 加载向量库，创建稠密检索器
    vec_db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding,
        collection_name="hybrid_kb"
    )
    dense_retriever = vec_db.as_retriever(search_kwargs={"k": top_k})

    # 初始化BM25关键词检索器
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = top_k

    # 多路结果融合，weights控制两路检索权重 0.5:0.5均衡
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

# ===================== 6. 溯源Prompt模板：强制输出引用来源 =====================
trace_prompt = PromptTemplate.from_template("""
你是专业文档问答助手，只能依据下方参考资料回答，禁止编造任何不存在的信息。
回答完成后必须单独分段输出【引用来源】，每条格式严格遵循：
[序号] 文档路径 | 页码 | 原文摘要（控制100字以内）

参考资料：
{context}
用户问题：{question}
""")

def format_reference_context(doc_list):
    """将检索文档拼接为LLM可读上下文"""
    context_text = ""
    for idx, doc in enumerate(doc_list):
        context_text += f"片段{idx+1}：{doc.page_content}\n\n"
    return context_text

# ===================== 7. 构建RAG问答链路 =====================
def build_rag_chain():
    chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough()
        }
        | trace_prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ===================== 程序入口执行 =====================
if __name__ == "__main__":
    # 1. 加载并分块文档
    doc_path = "/Users/xufeifan/ai-dev-12w/rag_demo/test02.pdf"
    chunk_list = load_and_split_doc(doc_path)

    # 2. 构建向量库
    vector_store = build_chroma_vector_store(chunk_list)

    # 3. 初始化混合融合检索器
    hybrid_retriever = get_hybrid_ensemble_retriever(chunk_list, top_k=10)

    # 4. 初始化问答链路
    rag_chain = build_rag_chain()

    # 5. 用户提问
    user_query = "自动播放是否静音，调用的方法名是什么"
    print(f"\n===== 用户提问：{user_query} =====\n")

    # 6. 混合检索获取融合后的候选文档
    retrieve_docs = hybrid_retriever.invoke(user_query)
    print(f"混合检索融合后候选片段总数：{len(retrieve_docs)}")

    # 7. 拼接上下文并调用LLM生成带溯源答案
    context_str = format_reference_context(retrieve_docs)
    answer_result = rag_chain.invoke({"context": context_str, "question": user_query})

    # 输出AI回答
    print("🤖 问答结果：")
    print(answer_result)

    # 打印完整溯源元数据（用于校验原文出处）
    print("\n===== 检索完整来源溯源详情 =====")
    for idx, doc in enumerate(retrieve_docs):
        source_file = doc.metadata.get("source", "未知文档")
        page_num = doc.metadata.get("page", "无页码")
        brief_text = doc.page_content[:120] + "..."
        print(f"【片段{idx+1}】文档：{source_file} | 页码：{page_num}")
        print(f"原文摘要：{brief_text}\n")