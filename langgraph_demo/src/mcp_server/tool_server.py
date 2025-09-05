from fastmcp import FastMCP
from langchain_tavily import TavilySearch
from fastmcp.prompts.prompt import PromptMessage, TextContent
import os
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

luis_miguel_server = FastMCP(name = "my_mcp", instructions = "Luis Miguel's Python MCP")

# TODO: Create tool MCP
@luis_miguel_server.tool(name = "Tavily_search_tool") #mcp server tool decorator
def my_search(query : str) -> str:
    """
    tool to search public content on the Internet, including real-time weather
    """
    try:
        print("Use the search tool, input parameter is ", query)
        search = TavilySearch(max_results=3, api_key=TAVILY_API_KEY)
        response = search.run(query)
        if response["results"]:
            return "\n\n".join([d["content"] for d in response["results"]])
        else:
            return "No content found in searching"
    except Exception as e:
        print(e)
        return "No content found in searching"

@luis_miguel_server.tool()
def say_hello(username : str) -> str:
    """
    Greet the designated user.
    """
    return f"Hello, {username}! It's a good day today!"

# TODO: Create prompt MCP
@luis_miguel_server.prompt()
def ask_about_topic(topic: str)->str:
    return f"Can you explain {topic} in detail?"

@luis_miguel_server.prompt()
def generate_code_request(language: str, task_description : str)->PromptMessage:
    """
    template for prompt to generate code
    """
    content = f"Please use {language} to write a function that delivers following works: {task_description}"
    return PromptMessage(
        role = "user",
        content = TextContent(type = "text", text = content)
    )

# TODO: create resource MCP
@luis_miguel_server.resource("resource://config")
def get_config()-> dict:
    """return configuration in json"""
    return{
        "theme" : "dark",
        "version" : "1.2.0",
        "features" : ["tools", "resources"],
    }