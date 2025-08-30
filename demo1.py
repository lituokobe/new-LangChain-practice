from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, MessagesPlaceholder, ChatPromptTemplate, \
    FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

from new_langchaing_practice.env_util import OPENAI_API_KEY, OPENAI_BASE_URL

llm = ChatOpenAI(
    model = 'gpt-4.1-nano',
    temperature = 0.8,
    api_key = OPENAI_API_KEY,
    base_url = OPENAI_BASE_URL,
    max_tokens = 200
)

# TODO: Let LLM answer the question with different style by adjusting the system message
# message = [
#     # ("system", "You are a smart assistant, and you answer question in a witty way."),
#     # ("system", "You are a serious assistant, and you answer question in an academic way."),
#     ("system", "You are an innocent cute baby, and you don't know too many things, but you answer question in a childish way."),
#     ("human", "Why is the sky blue?")
# ]
#
# resp = llm.invoke(message)
#
# print(resp.content)

# TODO: Use prompt template - variable placeholder
# prompt_template = PromptTemplate.from_template("Help me generate a story of {subject}.")
# chain = prompt_template | llm
#
# resp = chain.invoke({"subject" : "cute panda"})
#
# print(resp.content)

# TODO: FewShotPromptTemplate
# examples = [
#     {
#         "question": "穆罕默德·阿里和艾伦·图灵谁活得更久？",
#         "answer": """
# 是否需要后续问题：是。
# 后续问题：穆罕默德·阿里去世时多大？
# 中间答案：穆罕默德·阿里去世时74岁。
# 后续问题：艾伦·图灵去世时多大？
# 中间答案：艾伦·图灵去世时41岁。
# 所以最终答案是：穆罕默德·阿里
# """,
#     },
#     {
#         "question": "乔治·华盛顿的外祖父是谁？",
#         "answer": """
# 是否需要后续问题：是。
# 后续问题：乔治·华盛顿的母亲是谁？
# 中间答案：乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿。
# 后续问题：玛丽·鲍尔·华盛顿的父亲是谁？
# 中间答案：玛丽·鲍尔·华盛顿的父亲是约瑟夫·鲍尔。
# 所以最终答案是：约瑟夫·鲍尔
# """,
#     },
#     {
#         "question": "《大白鲨》和《007：大战皇家赌场》的导演是否来自同一个国家？",
#         "answer": """
# 是否需要后续问题：是。
# 后续问题：《大白鲨》的导演是谁？
# 中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
# 后续问题：史蒂文·斯皮尔伯格来自哪里？
# 中间答案：美国。
# 后续问题：《007：大战皇家赌场》的导演是谁？
# 中间答案：《007：大战皇家赌场》的导演是马丁·坎贝尔。
# 后续问题：马丁·坎贝尔来自哪里？
# 中间答案：新西兰。
# 所以最终答案是：否
# """,
#     },
# ]
# base_template = PromptTemplate.from_template("Question:{question}\n{answer}")
# final_template = FewShotPromptTemplate(
#     examples = examples,
#     example_prompt = base_template,
#     suffix = "Question: {input}",
#     input_variables = ["input"]
# )
#
# chain = final_template | llm
# resp = chain.invoke({"input" : "中国历史上，隋朝和秦朝哪个持续的时间比较长？"})
#
# print(resp)

# TODO: Use prompt template - message placeholder
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "You are a basketball fan, and you favorite player is Kobe Bryant, you answer all the questions from your passion of basketball and Kobe Bryant."),
#     MessagesPlaceholder("input")
# ]
# )
# chain = prompt_template | llm
#
# resp = chain.invoke({"input" : [HumanMessage(content = "How to study math?")]})
#
# print(resp.content)

# TODO: ICL in chat template
examples = [
    {"input" : "2 🦜 2", "output" : "5"},
    {"input" : "3 🦜 2", "output" : "7"},
    {"input" : "4 🦜 6", "output" : "25"}
]

base_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples = examples,
    example_prompt = base_prompt
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a smart AI assistant. " "The following examples show how to interpret 🦜."),
        # Must ask the model to learn from the examples
        few_shot_prompt,
        MessagesPlaceholder("msg")
    ]
)

chain = prompt_template | llm | StrOutputParser()

resp = chain.invoke({"msg" : [HumanMessage(content = "What is 5 🦜 8?")]})

print(resp)