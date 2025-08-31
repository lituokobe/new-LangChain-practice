from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from new_langchaing_practice.models import llm

# We make the system message dynamic, so that it can handle the summarized context
# We put the summarized context to system message because the next node - RunnableWithMessageHistory only takes 2 input variables: input_messages_key, history_messages_key
# This is more like a compromise solution
prompt = ChatPromptTemplate([
    ("system", "{system_message}"),
    MessagesPlaceholder(variable_name = "chat_history", optional = True),
    # ("placeholder", "{chat_history}"), #this is also a way to input history, but not recommended
    ("human", "{input}"),
])

chain = prompt | llm

# TODO: 1. chat history solution 1: use dictionary to store chat history
# # the dictionary of store has all the chat history from all users.
# # each user has a key of its session id
# store = {}
# def get_session_history(session_id : str):
#     """
#     Get chat history from RMA
#     :param session_id:
#     :return:
#     """
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#         # InMemoryChatMessageHistory is a child class of Basemodel
#     return store[session_id]

# TODO: chat history solution 2: use SQLite to store chat history
def get_session_history(session_id : str):
    return SQLChatMessageHistory(
        session_id = session_id,
        connection = "sqlite:///chat_history.db"
    )

# TODO: Summarize context for history record. Only directly keep the recent 2 messages
def summarize_messages(current_input):
    """"clip and summarize context
    """
    session_id = current_input["config"]["configurable"]["session_id"]
    if not session_id:
        raise ValueError("No session id available")

    # Get all chat history from current session id
    chat_history = get_session_history(session_id)
    stored_messages = chat_history.messages #stored_messages is a list

    if len(stored_messages) <= 2:
        return {
        "original_messages" : stored_messages,
        "summary" : None
    }

    # clip messages
    last_two_messages = stored_messages[-2:]
    messages_to_summarize = stored_messages[:-2]

    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "Please summarize the following chat history to an abstract that keeps all the key information."),
        MessagesPlaceholder(variable_name = "chat_history", optional = True),
        # ("placeholder", "{chat_history}"),
        ("human", "Please generate an abstract to covers the core content of the above conversation. Keep important facts and decisions.")
    ])

    summarization_chain = summarization_prompt | llm

    # this will be AI message
    summary_message = summarization_chain.invoke({"chat_history" : messages_to_summarize})

    return {
        "original_messages" : last_two_messages,
        "summary" : summary_message
    }

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key = "input",
    history_messages_key = "chat_history" # this is to match variable_name in MessagesPlaceholder
)

# RunnablePassthrough will pass exactly the same value/content to the next
# assign() will add/edit a key in the value/content
final_chain = RunnablePassthrough.assign(messages_summarized = summarize_messages) | RunnablePassthrough.assign(
    input = lambda x : x["input"],
    chat_history = lambda x : x["messages_summarized"]["original_messages"],
    system_message = lambda x : f"You are an assistant who tries to answer all the questions. Previous chat abstract: {x["messages_summarized"]["summary"].content}" if x["messages_summarized"].get("summary") else "No abstract"
) | chain_with_message_history

result1 = final_chain.invoke({"input" :"Hi, my name is Luis.", "config" : {"configurable" : {"session_id" : "bbn123"}}}, config = {"configurable" : {"session_id" : "bbn123"}} )
print(result1.content)

result2 = final_chain.invoke({"input" :"What is my name?", "config" : {"configurable" : {"session_id" : "bbn123"}}}, config = {"configurable" : {"session_id" : "bbn123"}})
print(result2.content)

result2 = final_chain.invoke({"input" :"I want to have a Chinese name that matches my Spanish name.", "config" : {"configurable" : {"session_id" : "bbn123"}}}, config = {"configurable" : {"session_id" : "bbn123"}})
print(result2.content)