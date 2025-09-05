from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from new_langchaing_practice.models import llm

prompt = (
    PromptTemplate.from_template("Help me generate a speech opening of {topic}.")
    + " requirement: 1. Be casual and funny;"
    + "2. use language of {language}."
)

chain = prompt | llm | StrOutputParser()

#TODO create tool from a chain
class ToolArgs(BaseModel):
    topic: str = Field(description = "topic of the speech")
    language : str = Field(description = "language of the speech")

runnable_tool = chain.as_tool(
    name = "chain_tool",
    description = "tool to generate speech opening",
    args_schema = ToolArgs,
)

# print(runnable_tool.args_schema.model_json_schema())
# print(runnable_tool.name)
# print(runnable_tool.description)

# resp = chain.invoke({"topic":"basketball", "language":"Chinese"})
# print(resp.content)