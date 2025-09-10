import asyncio
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from new_langchaing_practice.langgraph_demo2.src.agent.my_llm import llm

# from src.agent.my_llm import llm

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

    memory = MemorySaver()
    graph = builder.compile(checkpointer = memory,interrupt_before = ['tools']) # when there is a tool call, there is an interruption

    return graph

# agent = asyncio.run(create_graph())

# We won't run this project on Studio UI for customized UI information.
async def run_graph():
    graph = await create_graph()
    config = {
        "configurable":{
            "thread_id":"luis123"
        }
    }
    def get_answer(tool_message, user_answer):
        """human interference to provide answer to question"""
        tool_name = tool_message.tool_calls[0]["name"]
        answer = (
            f"Human ends the tool execution of {tool_name}, the reason is {user_answer}"
        )
        # create a message
        new_message = [
            ToolMessage(content = answer, tool_call_id = tool_message.tool_calls[0]["id"]),
            AIMessage(content = answer)
        ]

        # add new message to workflow's state
        graph.update_state(
            config = config,
            values = {"messages": new_message}
        )
    def print_message(event, result):
        """formatted message output"""
        messages = event.get("messages")
        if messages:
            if isinstance(messages, list):
                message = messages[-1]
            if message.__class__.__name__ == "AIMessage":
                if message.content:
                    result = message.content
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > 1500:
                msg_repr = msg_repr[:1500] + "... and more"
            print(msg_repr)
        return result

    async def execute_graph(user_input: str) -> str:
        """function to execute workflow"""
        result = "" # last message from AI assistant
        if user_input.strip().lower()!= "y": # if user doesn't agree to do tool call
            current_state = graph.get_state(config)
            if current_state.next: # if there is next step, current workflow is paused
                tools_script_message = current_state.values["messages"][-1]
                # provide change request to satisfy tool call
                get_answer(tools_script_message, user_input)
                message = graph.get_state(config).values["messages"][-1]
                result = message.content
                return result

            else: # process user_input to continue the chat based on it
                async for chunk in graph.astream({"messages": ("user", user_input)}, config, stream_mode = "values"):
                    result = print_message(chunk, result)
        else: # user input Y to continue tool call
            async for chunk in graph.astream(None, config, stream_mode = "values"):
                result = print_message(chunk, result)

        current_state = graph.get_state(config)
        if current_state.next:
            ai_message = current_state.values["messages"][-1]
            tool_name = ai_message.tool_calls[0]["name"]
            result = f"AI assistant will execute {tool_name}. Input 'y' to approve, otherwise, state your reason."

        return result

    while True:
        user_input = input("User: ")
        resp = await execute_graph(user_input)
        print("AI: ", resp)

if __name__ == '__main__':
    asyncio.run(run_graph())