"""
Vector databases:
FAISS: by Meta, closed source,
fast in‑memory similarity search and clustering,
no external service — runs inside your Python process.
suitable for personal development.

Chroma: Lightweight embedding store + retrieval layer (Python‑first).
easy to set up and use.
Great for prototyping RAG, chatbots, or small/mid-size apps.
Ideal if you want simplicity over scalability.

Milvus: full‑blown vector database (open‑source, cloud‑native).
Large‑scale, production‑grade vector search with persistence and metadata.
"""
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from new_langchaing_practice.embeddings import bge_hf_model, Qwen3CustomEmbedding

# FAISS needs to be installed with pip install faiss-cpu
qwen3_embedding_model = Qwen3CustomEmbedding("Qwen/Qwen3-Embedding-0.6B")

# Create index
dim  = len(qwen3_embedding_model.embed_query("Hello")) # set up the dimension of the embedding
index = faiss.IndexFlatL2(dim)

# Initialize the vector database client
vector_store = FAISS(
    embedding_function = qwen3_embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id = {}
)

# Prepare data (Document)
# page_content will be vectorized, and can be used for semantic retrieval
# metadata is for filter retrieval
document_1 = Document(
    page_content="今天早餐我吃了巧克力薄煎饼和炒蛋。",
    metadata={"source": "tweet", "time": "上午"},
)

document_2 = Document(
    page_content="明天的天气预报是阴天多云，最高气温62华氏度。",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="正在用LangChain构建一个激动人心的新项目——快来看看吧！",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="劫匪闯入城市银行，盗走了100万美元现金。",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="哇！那部电影太精彩了，我已经迫不及待想再看一遍。",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="新iPhone值得这个价格吗？阅读这篇评测一探究竟。",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="当今世界排名前十的足球运动员。",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph是构建有状态智能体应用的最佳框架！",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="由于对经济衰退的担忧，今日股市下跌500点。",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="我有种不好的预感，我要被删除了 :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]

ids = ["id" + str(i+1) for i in range(len(documents))]

# Add the documents to FAISS vector database
vector_store.add_documents(documents, ids = ids)

#TODO: retrieve information from the database
results = vector_store.similarity_search("今天的投资建议", k=2)
for res in results:
    print(type(res))
    print(res.id)
    print(f"* {res.page_content} [{res.metadata}]")
