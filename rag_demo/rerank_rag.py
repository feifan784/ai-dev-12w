import os
import shutil
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List
# BGE向量 + 重排序模型
from FlagEmbedding import FlagModel, FlagReranker
# LangChain基础组件
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# ===================== 1. 初始化BGE稠密Embedding =====================
def init_bge_embedding():
    bge_model = FlagModel(
        model_name_or_path="BAAI/bge-base-zh-v1.5",
        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
        use_fp16=True
    )
    class LocalBGEEmbedding(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return bge_model.encode(texts).tolist()
        def embed_query(self, text: str) -> List[float]:
            return bge_model.encode_queries([text])[0].tolist()
    return LocalBGEEmbedding()
embedding = init_bge_embedding()

# ===================== 2. 初始化Reranker交叉编码器精排模型 =====================
def init_reranker_model():
    """BGE官方重排序模型，成对输入问题+文档精细打分"""
    rerank_model = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
    return rerank_model
reranker = init_reranker_model()

# ===================== 3. 初始化通义千问LLM =====================
def init_llm():
    load_dotenv("/Users/xufeifan/ai-dev-12w/.env")
    llm = ChatOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        temperature=0.01
    )
    return llm
llm = init_llm()

# ===================== 4. 文档加载与重叠分块 =====================
def load_and_split_doc(file_path: str):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("仅支持 .txt / .pdf")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，"]
    )
    chunk_docs = splitter.split_documents(raw_docs)
    print(f"文档分块完成，总分块数量：{len(chunk_docs)}")
    return chunk_docs

# ===================== 5. Chroma稠密向量库构建 =====================
CHROMA_DB_PATH = "./chroma_rerank_db"
def build_chroma_vector_store(chunks):
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DB_PATH,
        collection_name="rerank_kb"
    )
    print("稠密向量库构建完成")
    return vector_db

# ===================== 6. 获取单路稠密向量检索器（粗召回） =====================
def get_dense_retriever(top_k=10):
    """粗召回取10条候选，给Reranker筛选空间"""
    vec_db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding,
        collection_name="rerank_kb"
    )
    retriever = vec_db.as_retriever(search_kwargs={"k": top_k})
    return retriever

# ===================== 7. Reranker重排序核心逻辑 =====================
def rerank_filter_docs(query: str, raw_candidate_docs, reserve_top_n=3):
    """
    对粗召回候选文档二次精细打分、排序、过滤
    :param query: 用户原始问题
    :param raw_candidate_docs: 向量粗召回全部候选文档
    :param reserve_top_n: 最终保留送入LLM的高相关片段数量
    :return: 重排后top_n高分文档，全部候选带分数列表（用于复盘）
    """
    # 构造 [问题, 文档片段] 成对输入reranker
    input_pairs = [[query, doc.page_content] for doc in raw_candidate_docs]
    # 批量计算相关性分数，分数越高相关性越强
    score_list = reranker.compute_score(input_pairs)
    # 文档与分数绑定，按分数降序重排
    doc_score_zip = list(zip(raw_candidate_docs, score_list))
    doc_score_zip.sort(key=lambda x: x[1], reverse=True)
    # 截取topN高分片段
    top_filter_docs = [item[0] for item in doc_score_zip[:reserve_top_n]]
    return top_filter_docs, doc_score_zip

# ===================== 8. 溯源Prompt模板 =====================
trace_prompt = PromptTemplate.from_template("""
你是专业文档问答助手，只能依据下方参考资料回答，禁止编造任何不存在的信息。
回答完成后必须单独分段输出【引用来源】，每条格式严格遵循：
[序号] 文档路径 | 页码 | 原文摘要（控制100字以内）

参考资料：
{context}
用户问题：{question}
""")

def format_reference_context(doc_list):
    context_text = ""
    for idx, doc in enumerate(doc_list):
        context_text += f"片段{idx+1}：{doc.page_content}\n\n"
    return context_text

# ===================== 9. 构建RAG问答链路 =====================
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
    # 1. 加载分块文档
    doc_path = "/Users/xufeifan/ai-dev-12w/rag_demo/test02.pdf"
    chunk_list = load_and_split_doc(doc_path)

    # 2. 构建向量库
    vector_store = build_chroma_vector_store(chunk_list)

    # 3. 稠密粗召回检索器，一次性取10条候选
    dense_retriever = get_dense_retriever(top_k=10)

    # 4. 问答链路初始化
    rag_chain = build_rag_chain()

    # 5. 用户提问
    user_query = "自动播放是否静音，调用的方法名是什么"
    print(f"\n===== 用户提问：{user_query} =====\n")

    # 6. 第一步：向量粗召回获取候选集
    raw_candidates = dense_retriever.invoke(user_query)
    print(f"向量粗召回候选片段总数：{len(raw_candidates)}")

    # 7. 第二步：Reranker精细重排序，过滤保留Top3高相关片段
    final_top_docs, all_scored_candidates = rerank_filter_docs(user_query, raw_candidates, reserve_top_n=3)
    print(f"Reranker精排过滤后保留高相关片段：{len(final_top_docs)}")

    # 8. 拼接上下文生成答案
    context_str = format_reference_context(final_top_docs)
    answer_result = rag_chain.invoke({"context": context_str, "question": user_query})

    # 输出LLM回答（自带引用来源）
    print("🤖 优化后问答结果：")
    print(answer_result)

    # 打印最终送入模型的溯源片段
    print("\n===== Reranker筛选后最终参考片段 =====")
    for idx, doc in enumerate(final_top_docs):
        source_file = doc.metadata.get("source", "未知文档")
        page_num = doc.metadata.get("page", "无页码")
        brief_text = doc.page_content[:120] + "..."
        print(f"【片段{idx+1}】文档：{source_file} | 页码：{page_num}")
        print(f"摘要：{brief_text}\n")

    # 打印全部候选打分列表，用于复盘优化前后召回差异
    print("\n===== 全部候选Reranker相关性打分（复盘对比用） =====")
    for idx, (doc, score) in enumerate(all_scored_candidates):
        print(f"候选{idx+1} 相关性分数：{score:.4f} 片段前80字：{doc.page_content[:80]}...")