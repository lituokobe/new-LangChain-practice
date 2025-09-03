import os

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from src.agent.tools.tool_args_description_practice import calculate4
from src.agent.tools.tools_runnable import runnable_tool
from src.agent.tools.tools_BaseTool import my_search_tool
from src.agent.tools.tools_get_user_info import get_user_info_by_name
from dotenv import load_dotenv
from src.agent.tools.tools_get_user_info import get_user_name, greet_user
from src.agent.my_state import CustomState

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

# save the short-term history to memory
checkpointer = InMemorySaver()


graph = create_react_agent(
    llm,
    tools = [runnable_tool, my_search_tool],
    prompt = "You are a helpful assistant.",
    checkpointer = checkpointer
)

config = {
    "configurable" : {
        "thread_id" : "1"
    }
}

resp1 = graph.invoke(
    {"messages" : [{"role":"user", "content":"What's the weather like in Beijing today?"}]},
    config
)

print(resp1["messages"][-1].content)

resp2 = graph.invoke(
    {"messages" : [{"role":"user", "content":"What about Shanghai?"}]},
    config
)

print(resp1["messages"][-1].content)