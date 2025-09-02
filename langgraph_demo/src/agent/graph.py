import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(
    model = 'gpt-4.1-nano',
    temperature = 0.8,
    api_key = OPENAI_API_KEY,
    base_url = OPENAI_BASE_URL,
    max_tokens = 200
)


def get_weather(city:str) -> str:
    """Return the weather info for a given city."""
    return f"It's a good day in {city} at 26 degree."

graph = create_react_agent(
    llm,
    tools = [get_weather],
    prompt = "You are a helpful assistant."
)