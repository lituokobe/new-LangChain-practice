from langchain_openai import ChatOpenAI

from new_langchaing_practice.env_util import OPENAI_API_KEY, OPENAI_BASE_URL

llm = ChatOpenAI(
    model = 'gpt-4.1-nano',
    # model = 'gpt-4o',
    temperature = 0.8,
    api_key = OPENAI_API_KEY,
    base_url = OPENAI_BASE_URL,
    max_tokens = 200
)