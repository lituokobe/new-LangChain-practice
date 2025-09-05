import os

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from new_langchaing_practice.langgraph_demo.src.agent.tools.tool_args_description_practice import calculate4
from new_langchaing_practice.langgraph_demo.src.agent.tools.tools_runnable import runnable_tool
from new_langchaing_practice.langgraph_demo.src.agent.tools.tools_BaseTool import my_search_tool
from new_langchaing_practice.langgraph_demo.src.agent.tools.tools_get_user_info import get_user_info_by_name
from new_langchaing_practice.langgraph_demo.src.agent.tools.tools_get_user_info import get_user_name, greet_user
from new_langchaing_practice.langgraph_demo.src.agent.my_state import CustomState
"""
Because we are running this project on LangGraph server, so the file import way below follows the
instruction of LangGraph server import requirements. You will fail to import if you directly run in on
Pycharm platform.

In langgraph.json, it is set up this way:
"agent": "./src/agent/graph.py:graph"
"""
# from src.agent.tools.tool_args_description_practice import calculate4
# from src.agent.tools.tools_runnable import runnable_tool
# from src.agent.tools.tools_BaseTool import my_search_tool
# from src.agent.tools.tools_get_user_info import get_user_info_by_name
# from src.agent.tools.tools_get_user_info import get_user_name, greet_user
# from src.agent.my_state import CustomState

from dotenv import load_dotenv
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


# def get_weather(city:str) -> str:
#     """Return the weather info for a given city."""
#     return f"It's a good day in {city} at 26 degree."

# prompt template function, content passed by user to form a dynamic system prompt
def prompt(state : AgentState, config: RunnableConfig) ->list[AnyMessage]:
    user_name = config["configurable"].get("user_name", "Luis")
    print(user_name)
    system_message = f"You are a smart assistant. Now user's name is {user_name}"
    return [{"role":"system", "content":system_message}] + state["messages"]


graph = create_react_agent(
    llm,
    # tools = [get_weather, calculate4, runnable_tool, my_search_tool, get_user_info_by_name],
    tools = [calculate4, runnable_tool, my_search_tool, get_user_name, greet_user],
    # prompt = "You are a helpful assistant.",
    prompt = prompt,
    # state_schema = CustomState # Customize our own state, by default, there is also AgentState generated if this parameter is None
)