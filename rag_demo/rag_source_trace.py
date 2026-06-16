import os
import shutil
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List
from FlagEmbedding import FlagModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ===================== 1. 初始化本地BGE Embedding（和建库统一模型） =====================
local_bge_path = "BAAI/bge-base-zh-v1.5"
bge_model = FlagModel(
    model_name_or_path=local_bge_path,
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True
)

# LangChain 兼容封装
class LocalBGEEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return bge_model.encode(texts).tolist()
    def embed_query(self, text: str) -> List[float]:
        return bge_model.encode_queries([text])[0].tolist()
embedding = LocalBGEEmbedding()

# ===================== 2. 初始化通义千问LLM =====================
from dotenv import load_dotenv
load_dotenv("/Users/xufeifan/ai-dev-12w/.env")
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",
    temperature=0.01  # 降低随机性，严格基于文档作答
)

# ===================== 3. 文档加载、分块 =====================
def load_document(file_path: str):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("仅支持 .txt / .pdf")
    docs = loader.load()
    print(f"加载文档：{file_path}，原始片段数：{len(docs)}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，"]
    )
    chunks = splitter.split_documents(docs)
    print(f"文本分块完成，总分块：{len(chunks)}")
    return chunks

# ===================== 4. 构建/加载向量库 =====================
CHROMA_PATH = "./chroma_db"
def build_vector_store(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_PATH,
        collection_name="source_trace_kb"
    )
    print("向量库构建完成")
    return db

def get_retriever(top_k=3):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
        collection_name="source_trace_kb"
    )
    # 返回带分数的检索结果
    return db.as_retriever(search_kwargs={"k": top_k})

# ===================== 5. 溯源专用Prompt（强制输出引用） =====================
trace_prompt = PromptTemplate.from_template("""
你是文档问答助手，仅能依据下方参考资料回答问题，禁止编造任何内容。
回答结束后必须单独一段输出【引用来源】，每条格式：
[序号] 文档名 | 页码 | 原文摘要（截取100字内）

参考资料：
{context}
用户问题：{question}
""")

# 拼接检索上下文，同时保留完整Document对象用于溯源
def format_context_with_source(docs):
    context_text = ""
    for idx, doc in enumerate(docs):
        context_text += f"片段{idx+1}：{doc.page_content}\n\n"
    return context_text

# ===================== 6. 构建带溯源的RAG链 =====================
def create_trace_rag_chain(retriever):
    rag_chain = (
        {
            "context": retriever | format_context_with_source,
            "raw_docs": retriever,  # 透传原始文档对象，用于外部打印来源
            "question": RunnablePassthrough()
        }
        | trace_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# ===================== 主程序：问答 + 打印完整引用元数据 =====================
if __name__ == "__main__":
    # 1. 上传/加载你的文档
    doc_file = "/Users/xufeifan/ai-dev-12w/rag_demo/test02.pdf"
    chunks = load_document(doc_file)
    vector_db = build_vector_store(chunks)
    retriever = get_retriever(top_k=3)
    rag_chain = create_trace_rag_chain(retriever)

    # 2. 用户提问
    user_q = "⾃动播放是否静⾳，调用的方法名是什么"
    print(f"\n===== 用户提问：{user_q} =====\n")

    # 3. 带相似度分数检索（推荐）
    vector_store = retriever.vectorstore
    search_result = vector_store.similarity_search_with_score(user_q, k=3)
    search_docs = [doc for doc, score in search_result]

    # 4. 生成答案
    answer = rag_chain.invoke(user_q)

    # 5. 输出AI回答
    print("🤖 AI回答：")
    print(answer)

    # 6. 控制台打印完整溯源详情（额外展示元数据，便于调试）
    print("\n===== 完整检索来源详情（调试溯源） =====")
    for idx, doc in enumerate(search_docs):
        source_path = doc.metadata.get("source", "未知文档")
        page = doc.metadata.get("page", "无页码")
        short_content = doc.page_content[:120] + "..."
        print(f"【片段{idx+1}】")
        print(f"文档路径：{source_path}")
        print(f"页码：{page}")
        print(f"原文片段：{short_content}\n")