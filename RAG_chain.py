import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from new_langchaing_practice.embeddings import Qwen3CustomEmbedding
from new_langchaing_practice.models import llm

# Prepare vector database
qwen3_embedding_model = Qwen3CustomEmbedding("Qwen/Qwen3-Embedding-0.6B")
vector_store = Chroma(
    collection_name = "t_agent_blog",
    embedding_function = qwen3_embedding_model,
    persist_directory = "./chroma_db"
)

def create_dense_db():
    """
    write a blog to the database
    :return:
    """
    loader = WebBaseLoader(
        web_path = ("https://lilianweng.github.io/posts/2023-06-23-agent/"),
        bs_kwargs = dict(
            parse_only = bs4.SoupStrainer(
                class_= ("post-content", "post-title", "post-header")
            )
        )
    )

    docs_list = loader.load()

    # Cutting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splits = text_splitter.split_documents(docs_list)

    print("amount of docs: ", len(splits))

    # write docs to vector database
    ids = ["id" + str(i + 1) for i in range(len(splits))]
    vector_store.add_documents(documents = splits, ids = ids)

# create_dense_db()

# Create chat template
contextualize_q_system_prompt = (
    "Given the chat history and the latest user question (which may reference context from the chat history), "
    "rephrase it into a standalone question (one that can be understood without the chat history). "
    "Do not answer the question — only rephrase it when necessary; otherwise, leave it unchanged."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a retriever for vector database
retriever = vector_store.as_retriever(search_kwargs = {"k" : 2})

# Create a retriever with context awareness
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# Actual RAG
system_prompt = (
    "You are a question‑answering task assistant. "
    "Use the following retrieved context to answer the question. "
    "If you don’t know the answer, say you don’t know. "
    "Answer in no more than three sentences, keeping it concise."
    "\n\n"
    "{context}"  # docs retrieved from vector database
)

# Create answer template
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_chain = create_stuff_documents_chain(
    llm, qa_prompt
)

rag_chain = create_retrieval_chain(history_aware_retriever, question_chain)

# history storage
store = {}
def get_session_history(session_id : str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Add history storage
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key = "input",
    history_messages_key = "chat_history" ,# this is to match variable_name in MessagesPlaceholder
    output_messages_key = "answer"
)

result1 = conversational_rag_chain.invoke({"input" :"What is task decomposition?"}, config = {"configurable" : {"session_id" : "bbn123"}} )
print(result1["answer"])

result2 = conversational_rag_chain.invoke({"input" :"What are common ways of doing it?"}, config = {"configurable" : {"session_id" : "bbn123"}})
print(result2["answer"])
#
# result2 = conversational_rag_chain.invoke({"input" :"I want to have a Chinese name that matches my Spanish name.", "config" : {"configurable" : {"session_id" : "bbn123"}}}, config = {"configurable" : {"session_id" : "bbn123"}})
# print(result2.content)