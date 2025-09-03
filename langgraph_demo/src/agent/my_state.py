from langchain_core.messages import BaseMessage
from langgraph.prebuilt.chat_agent_executor import AgentState

# Self customized Agent state class
class CustomState(AgentState):
    username : str
