import asyncio
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.my_llm import llm

# TODO: Create MCPs
load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
zhipu_mcp_server_config = {
    "url" : "https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization=" + ZHIPU_API_KEY,
    "transport" : "sse",
}

# MCPs from modelscope
# 12306 MCP, visit https://modelscope.cn/mcp/servers/@Joooook/12306-mcp to update url
my12306_mcp_server_config = {
    "url" : "https://mcp.api-inference.modelscope.net/23ff105a9f6b4b/sse",
    "transport" : "sse",
}
# chart visualization, visit https://modelscope.cn/mcp/servers/@antvis/mcp-server-chart to update url
chart_mcp_server_config = {
    "url" : "https://mcp.api-inference.modelscope.net/4206bdd2532946/sse",
    "transport" : "sse",
}

mcp_client = MultiServerMCPClient(
    {
        "chart_mcp": chart_mcp_server_config,
        "my12306_mcp": my12306_mcp_server_config,
        "zhipuai_mcp": zhipu_mcp_server_config
    }
)

#TODO: create tool call class


class State(MessagesState):
    pass


async def create_graph():
    tools = await mcp_client.get_tools() # over 30 tools, all from MCP server

    builder = StateGraph(State)

    llm_with_tools = llm.bind_tools(tools)

    async def chatbot(state:State):
        return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}

    builder.add_node("chatbot", chatbot)

    tools_node = ToolNode(tools)

    builder.add_node("tools", tools_node)

    builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    builder.add_edge(
        "tools",
        "chatbot"
    )

    builder.add_edge(
        START,
        "chatbot"
    )

    graph = builder.compile()

    return graph

agent = asyncio.run(create_graph())
