# Searching tool
from typing import Type

from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class SearchArgs(BaseModel):
    query : str = Field(description = "information that needs to search online")
class MySearchTool(BaseTool):
    name : str = "search_tool"
    description : str = "tool to search public content on the Internet, including real-time weather"
    return_direct : bool = False
    args_schema : Type[BaseModel]= SearchArgs
    def _run(self, query) -> str:
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
my_search_tool = MySearchTool()

# print(my_search_tool.name)
# print(my_search_tool.description)
# print(my_search_tool.args_schema.model_json_schema())

try:
    search = TavilySearch(max_results=2, api_key=TAVILY_API_KEY)
    response = search.run("What's the latest news on AI?")
    if response["results"]:
        print("\n\n".join([d["content"] for d in response["results"]]))
    else:
        print("No content found in searching")
except Exception as e:
    print(e)
    print("No content found in searching")