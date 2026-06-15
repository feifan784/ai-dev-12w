import os
import shutil
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from dotenv import load_dotenv
load_dotenv("/Users/xufeifan/ai-dev-12w/.env")

# 导入模块
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ===================== 1. 大模型（通义千问对话模型） =====================
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",
    temperature=0.1
)

# ===================== 2. 关键修复：通义Embedding（替代OpenAI Embedding） =====================
# 使用 DashScope 原生 Embedding，不再用 OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("LLM_API_KEY")
)

# ===================== 3. 文档加载 =====================
def load_file(file_path: str):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("仅支持 .txt、.pdf 格式")
    docs = loader.load()
    print(f"✅ 文档加载完成，页数/片段数：{len(docs)}")
    return docs

# ===================== 4. 文本分块 =====================
def split_document(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    split_docs = splitter.split_documents(docs)
    print(f"✅ 文本分块完成，总分块数：{len(split_docs)}")
    return split_docs

# ===================== 5. 构建向量库 =====================
def build_vector_db(split_docs):
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ 向量库构建完成")
    retriever = vector_db.as_retriever(k=3)
    return retriever

# ===================== 6. 新版LCEL RAG链 =====================
def create_rag_chain(llm, retriever):
    prompt = PromptTemplate.from_template("""
请严格根据下面的参考资料回答问题，禁止编造内容。
参考资料：{context}
问题：{question}
""")
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 改为你的文档绝对路径
    file_path = "/Users/xufeifan/ai-dev-12w/agp_demo/test02.pdf"
    # file_path = "/Users/xufeifan/ai-dev-12w/agp_demo/test.txt"

    raw_docs = load_file(file_path)
    chunk_docs = split_document(raw_docs)
    retriever = build_vector_db(chunk_docs)
    rag_chain = create_rag_chain(llm, retriever)

    question = "⾃渲染信息流⼴告的简介是什么？"
    print(f"\n❓ 用户问题：{question}")
    answer = rag_chain.invoke(question)

    print(f"\n🤖 AI回答：\n{answer}")