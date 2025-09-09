import asyncio
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph

from src.agent.my_llm import llm

# TODO: Create MCPs
load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
zhipu_mcp_server_config = {
    "url" : "https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization=" + ZHIPU_API_KEY,
    "transport" : "sse",
}

# MCPs from modelscope
# 12306 MCP
my12306_mcp_server_config = {
    "url" : "https://mcp.api-inference.modelscope.net/cebe8ee5aa2a44/sse",
    "transport" : "sse",
}
# chart visualization
chart_mcp_server_config = {
    "url" : "https://mcp.api-inference.modelscope.net/90c6f438dc1640/sse",
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
class BasicToolsNode:
    """
    Async tool call node, used for concurrent tool calls in AIMessage.
    Features:
    1. receive tool list and create name index
    2. Run tool call requests in the messages concurrently
    3. Automatically choose appropriate sync/async tool
    """
    def __init__(self, tools: list):
        """
        Initialize tools node
        Args:
            tools: tool lists, each tool includes the attribute of name
        """
        self.tools_by_name = {tool.name : tool for tool in tools} # set of tool names without duplication

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, List[ToolMessage]]:
        """
        Async call enter point
        Args:
            inputs: input dictionary, including key of "messages"
        Returns:
            Dictionary that includes list of ToolMessage
        Raises:
            ValueError: when input is invalid
        """
        # input validation
        if not (messages := state.get("messages")): # walrus operator (:=) allows you to assign a value to a variable as part of an expression
            raise ValueError("No message found in input.")
        message: AIMessage = messages[-1]

        # Concurrent run tool calls
        outputs = await self._execute_tool_calls(message.tool_calls)
        return{"messages":outputs}

    async def _execute_tool_calls(self, tool_calls: list[Dict]) -> List[ToolMessage]:
        """
        Execute tool calls
        Args:
            tool_calls: list of tool calls
        Returns:
            list of ToolMessage results
        """

        async def _invoke_tool(tool_call: Dict) -> ToolMessage:
            """
            Execute single tool call
            Args:
                tool_call: Dictionary of tool call, including name/args/id keys
            Returns:
                encapsulated ToolMessage
            Raises:
                KeyError: when the tool is not registered
                RuntimeError: when the tool call fails
            """
            try:
                # call async tool
                tool = self.tools_by_name.get(tool_call['name'])
                if not tool:
                    raise KeyError(f"{tool_call["name"]} is not registered")

                if hasattr(tool, "ainvoke"): # prioritize async approach
                    tool_result = await tool.ainvoke(tool_call["args"])

                else: # convert sync tool to async
                    loop = asyncio.get_running_loop()
                    tool_result = await loop.run_in_executor(
                        None, # default thread pool
                        tool.invoke, # sync call
                        tool_call["args"] #parameters
                    )

                # Create ToolMessage
                return ToolMessage(
                    content = json.dumps(tool_result, ensure_ascii = False),
                    name = tool_call["name"],
                    tool_call_id = tool_call["id"]
                    )
            except Exception as e:
                print(e)
                raise RuntimeError(f"Failed to call {tool_call["name"]}") from e

        try:
            # concurrently run all the tool calls
            # asyncio.gather() is the core function of Python async to initiate multiple threads. Its actions include:
            # concurrent execution: all the input threads will be called to the event loop at the same time and run concurrently
            # result collection: return thread results in the order of input, regardless of the completion order
            # error management: by default, any task failure will cancel all the other tasks and raise an error
            # if set return_exceptions = True, the exceptions will become result
            return await asyncio.gather( *[_invoke_tool(tool_call) for tool_call in tool_calls])
        except Exception as e:
            print(e)
            raise RuntimeError("Error in concurrent tool calls") from e

class State(MessagesState):
    pass

def route_tools_func(state:State):
    """
    Router function, if the output AIMessage includes tool calls, it will go to the tool node. Otherwise, END node.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages",[]):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls)>0:
        return "tools"
    return END

async def create_graph():
    tools = await mcp_client.get_tools() # over 30 tools, all from MCP server

    builder = StateGraph(State)

    llm_with_tools = llm.bind_tools(tools)

    async def chatbot(state:State):
        return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}

    builder.add_node("chatbot", chatbot)

    tools_node = BasicToolsNode(tools)

    builder.add_node("tools", tools_node)

    builder.add_conditional_edges(
        "chatbot",
        route_tools_func,
        {"tools":"tools", END:END}
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
