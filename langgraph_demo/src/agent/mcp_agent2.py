import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

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

# Python MCP server configuration
# This is only for server with authentication, i.e. when tool_server2 is running
# We need to pass the token in the client side
# Copy paste the token here every time when the server starts to run
test_token = ""

python_mcp_server_config = {
    "url" : "http://127.0.0.1:8000/streamable",
    "transport" : "streamable_http",
    "headers": {
        "Authorization": f"Bearer {test_token}",
    }
}


# MCP client
mcp_client = MultiServerMCPClient(
    {
        "python_mcp":python_mcp_server_config,
    }
)

# call the tool from mcp server
async def create_agent():
    """This agent has to be async"""
    mcp_tools = await mcp_client.get_tools()
    print(mcp_tools) # check all the tools

    return create_react_agent(
        llm,
        tools = mcp_tools,
        prompt = "You are a helpful assistant.",
    )

agent = asyncio.run(create_agent())

#Test the agent
# async def test_agent():
#     response = await agent.ainvoke({"messages": [{"role": "user", "content": "What is weather like today in Beijing?"}]})
#     return response
#
# # Run the async test
# result = asyncio.run(test_agent())
# print(result)