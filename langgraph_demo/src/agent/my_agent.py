import os
import sqlite3

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

from new_langchaing_practice.langgraph_demo.src.agent.tools.tools_BaseTool import my_search_tool

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(
    model = 'gpt-4.1-nano',
    temperature = 0.8,
    api_key = OPENAI_API_KEY,
    base_url = OPENAI_BASE_URL,
    max_tokens = 200
)

# TODO: save the short-term history to memory
# checkpointer = InMemorySaver()

# TODO: save the short/long-term history to SQL
conn = sqlite3.connect("chat_history2.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
store= SqliteSaver(conn)

graph = create_react_agent(
    llm,
    tools = [my_search_tool],
    prompt = "You are a helpful assistant.",
    checkpointer = checkpointer, # checkpointer is for short-term memory
    store = store, # store is for long term memory
)

config = {
    "configurable" : {
        "thread_id" : "2"
    }
}

# TODO: get context from short-term memory
# rest = list(graph.get_state(config))
# print(rest)
# # TODO: get context from long-term memory
# rest2 = list(graph.get_state_history(config))
# print(rest2)


# TODO: Test with multiple rounds of chat
resp1 = graph.invoke(
    {"messages" : [{"role":"user", "content":"Who is the greatest player in basketball?"}]},
    config
)
print(resp1["messages"][-1].content)

resp2 = graph.invoke(
    {"messages" : [{"role":"user", "content":"What about football?"}]},
    config
)
print(resp2["messages"][-1].content)