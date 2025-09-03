from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from new_langchaing_practice.env_util import OPENAI_API_KEY, OPENAI_BASE_URL, ALI_API_KEY, ALI_BASE_URL

llm = ChatOpenAI(
    model = 'gpt-4.1-nano',
    # model = 'gpt-4o',
    temperature = 0.8,
    api_key = OPENAI_API_KEY,
    base_url = OPENAI_BASE_URL,
    max_tokens = 200
)

# multimodal_llm = ChatOpenAI(
#     model = 'gpt-4o-mini-2024-07-18',
#     temperature = 0.8,
#     api_key = OPENAI_API_KEY,
#     base_url = OPENAI_BASE_URL,
#     max_tokens = 200
# )

# multimodal_llm = ChatOpenAI(
#     model = 'qwen2.5-omni-7b',
#     temperature = 0.8,
#     api_key = ALI_API_KEY,
#     base_url = ALI_BASE_URL,
#     max_tokens = 200
# )
# multimodal_llm = ChatTongyi(
#     # model_name="qwen-vl-max",
#     model_name="qwen2.5-omni-7b",
#     dashscope_api_key=ALI_API_KEY,
#     temperature=0.8,
#     max_tokens=200
# )