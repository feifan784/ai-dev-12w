from FlagEmbedding import FlagModel

# ========== 在线自动下载模式 ==========
# 首次运行自动下载 BAAI/bge-base-zh-v1.5 到本地缓存
model = FlagModel(
    model_name_or_path="BAAI/bge-base-zh-v1.5",
    # 检索任务必须加查询指令（BGE官方规范，提升检索精度）
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True  # Mac芯片加速，CPU也兼容
)

# 1. 用户查询文本（用encode_queries，自动带上指令）
query = "什么是RAG检索增强生成"
query_vec = model.encode_queries([query])
print(f"【查询向量】文本：{query}")
print(f"向量维度：{len(query_vec[0])}")
print(f"向量前10位数值：{query_vec[0][:10]}\n")

# 2. 知识库文档（用encode，不加指令）
docs = [
    "RAG全称检索增强生成，解决大模型幻觉、知识滞后问题",
    "文本分块Chunk是RAG流程核心步骤，控制上下文长度",
    "Embedding向量化将文本转为语义向量，用于相似度检索"
]
doc_vecs = model.encode(docs)

# 打印文档向量
for idx, text in enumerate(docs):
    print(f"【文档{idx+1}】{text}")
    print(f"向量维度：{len(doc_vecs[idx])}\n")

# 3. 计算余弦相似度（直观验证语义匹配）
similarity = query_vec @ doc_vecs.T
print("=== 查询与各文档相似度 ===")
for i, sim in enumerate(similarity[0]):
    print(f"文档{i+1} 相似度：{sim:.4f}")