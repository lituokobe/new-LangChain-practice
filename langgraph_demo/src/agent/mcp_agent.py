import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

from new_langchaing_practice.langgraph_demo.src.agent.tools.tools_BaseTool import my_search_tool
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

llm = ChatOpenAI(
    model = 'gpt-4.1-nano',
    temperature = 0.8,
    api_key = OPENAI_API_KEY,
    base_url = OPENAI_BASE_URL,
    max_tokens = 200
)

## Python MCP server configuration
python_mcp_server_config = {
    # "url" : "http://127.0.0.1:8000/sse",
    # "transport" : "sse",
    "url" : "http://127.0.0.1:8000/streamable",
    "transport" : "streamable_http",
}

# Public internet MCP
zhipu_mcp_server_config = {
    "url" : "https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization=" + ZHIPU_API_KEY,
    "transport" : "sse",
}

# MCP client
mcp_client = MultiServerMCPClient(
    {
        # "python_mcp":python_mcp_server_config,
        "zhipu_mcp":zhipu_mcp_server_config,
    }
)

# call the tool from mcp server
async def create_agent():
    """This agent has to be async"""
    mcp_tools = await mcp_client.get_tools()
    print(mcp_tools) # check all the tools

    # # get python mcp prompt
    # p =  await mcp_client.get_prompt(server_name = "python_mcp", #name of the server
    #                                  prompt_name = "ask_about_topic", # the tool name (function name)
    #                                  arguments ={"topic":"basketball"} # the input parameter
    #                                  )
    # print(p)
    #
    # # get python mcp resources
    # data = await mcp_client.get_resources(server_name="python_mcp",
    #                                       uris="resource://config")
    # print(data[0])
    # print(data[0].data) #json data

    return create_react_agent(
        llm,
        tools = mcp_tools,
        prompt = "You are a helpful assistant.",
    )

agent = asyncio.run(create_agent())

#Test the agent
async def test_agent():
    response = await agent.ainvoke({"messages": [{"role": "user", "content": "What is weather like today in Beijing?"}]})
    return response

# Run the async test
result = asyncio.run(test_agent())
print(result)