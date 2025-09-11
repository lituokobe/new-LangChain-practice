from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer

from new_langchaing_practice.env_util import OPENAI_API_KEY, OPENAI_BASE_URL
"""
4 libraries for embedding:
1. sentence-transformers
Purpose: Specialized for sentence, text, and document embeddings. Built on top of HuggingFace Transformers.
When to use: If you specifically need embeddings for semantic search, retrieval, clustering, or similarity scoring.

2. huggingface/transformers
General-purpose library for all NLP tasks (not just embeddings) that lets you load any HuggingFace model (BERT, GPT, etc.).
More flexible, but lower-level than sentence-transformers.
When to use: If you want full control over the model pipeline (e.g., fine-tuning embeddings, or using non-embedding models).

3. Langchain
LangChain provides wrappers (HuggingFaceEmbeddings, OpenAIEmbeddings, HuggingFaceBgeEmbeddings, etc.)\
so you can plug embedding models into LangChain pipelines without worrying about low‑level details.
Can switch between embeddings (OpenAIEmbeddings, HuggingFaceEmbeddings, CohereEmbeddings, etc.) with minimal code changes.
When to use: If you’re building end-to-end pipelines (RAG, chatbots, knowledge bases) and want to abstract away embedding backends.

4. FlagEmbedding
Purpose: Embedding-focused library from BAAI (Beijing Academy of AI). Hosts the BGE (BAAI General Embedding) family (bge-small, bge-large, bge-m3 etc.).
Specialized in retrieval / semantic similarity / RAG. Strong multilingual Chinese-English performance.
When to use: If you want state-of-the-art embeddings (esp. for Chinese + English) or need dense retrieval performance.

For RAG project, understanding semantic meaning is very important, Sentence-Transformers is recommended.

"""
#TODO: User default OpenAI library to embed
#
# client = OpenAI(
#     api_key = OPENAI_API_KEY,
#     base_url = OPENAI_BASE_URL
# )
#
# text = "this is very beautiful."
# resp = client.embeddings.create(
#     model = "text-embedding-3-large",
#     dimensions = 512,
#     input = text
# )
#
# print(resp)
# print(resp.data[0].embedding)
# print(len(resp.data[0].embedding))

# TODO: Use LangChain library to embed
# openai_embedding = OpenAIEmbeddings(
#     api_key = OPENAI_API_KEY,
#     base_url = OPENAI_BASE_URL,
#     model = "text-embedding-3-large",
#     dimensions = 512,
# )
#
# resp = openai_embedding.embed_documents(
#     [
#         "This is very beautiful.",
#         "The weather is regulated."
#     ]
# )
#
# print(resp[0])
# print(len(resp[0]))

# TODO: BGE embedding with huggingface
model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# This will download the embedding model to local machine.
# First time to run the model, it will download it from HuggingFace
# You can change environment variable HF_HOME to reset the download directory
bge_hf_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
resp = bge_hf_model.embed_documents(
    [
        "This is very beautiful.",
        "The weather is regulated."
    ]
)

print(resp[0])
print(len(resp[0]))

# TODO: Sentence Transformer embedding practice with Qwen3-Embedding-4B
# below code will download the model to local machine as well
# qwen3_embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
#
# resp = qwen3_embedding_model.encode(
#     [
#         "This is very beautiful.",
#         "The weather is regulated."
#     ]
# )
# print(resp[0])
# print(len(resp[0]))

# TODO: Integrate Qwen3-Embedding with LangChain class
# As of Aug 2025, Qwen3-Embedding is not directly supported by LangChain. We can self create a class to make it happen
# This will be a good example for future possible customization.

class Qwen3CustomEmbedding(Embeddings):
    """
    Customize a Qwen3 Embedding class, integrated with LangChain
    """
    def __init__(self, model_name):
        self.qwen3_embedding = SentenceTransformer(model_name)
    def embed_query(self, text : str) ->list[float]:
        return self.embed_documents([text])[0]
    def embed_documents(self, texts : list[str]) -> list[list[float]]:
        return self.qwen3_embedding.encode(texts)

qwen3_embedding_model = Qwen3CustomEmbedding("Qwen/Qwen3-Embedding-0.6B")
resp = qwen3_embedding_model.embed_documents(
    [
        "This is very beautiful.",
        "The weather is regulated."
    ]
)
print(resp[0])
print(len(resp[0]))