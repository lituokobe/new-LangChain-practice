from typing import TypedDict, Literal

from langchain_core.output_parsers import StrOutputParser
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from new_langchaing_practice.langgraph_demo2.src.agent.my_llm import llm

class State(TypedDict):
    """
    State class, we use TypedDict class. This is only one way to set up State class.
    This project is to create a workflow that can tell a joke
    """
    joke: str # content of joke
    topic:str # topic specified by user
    feedback: str # suggestions to improve
    funny_or_not: str # level of funniness


class Feedback(BaseModel):
    """
    Structured output class, used for LLM evaluation
    This class builds a schema for the output
    """
    grade : Literal["funny", "not funny"] = Field(
        description = "Decide if the joke is funny or not",
        examples = ["funny", "not funny"]
    )
    feedback : str = Field(
        description = "If the joke is not funny, provide suggestions to revise it.",
        examples = ["Add puns or surprising endings."]
    )

def joke_generator_func(state: State):
    """This is the node to use LLM to generate a joke"""
    prompt = (
        f"Revise the joke based on the feedback: {state['feedback']}\n topic: {state['topic']}." if state.get("feedback", None)
        else f"Create a joke about {state['topic']}."
    )

    # # the returned value should be an update to the state
    # # first way to return
    # resp = llm.invoke(prompt)
    # return {"joke":resp.content}

    # second way to return
    chain = llm|StrOutputParser()
    resp = chain.invoke(prompt)
    return {"joke": resp}

def joke_evaluator_func(state: State):
    """This is the node to use LLM to evaluate the joke in the state"""
    # if the llm supports structured output
    chain = llm.with_structured_output(Feedback)
    resp = chain.invoke(
        f"Evaluate the level of funniness of the joke: {state['joke']}\n"
        "Attention: jokes with surprises or puns are considered funny"
    )
    return {"feedback": resp.feedback, "funny_or_not":resp.grade}

    # # if the llm doesn't support structured output
    # chain = llm.bind_tools([Feedback]) #bind the structured class as a tool to the model
    # evaluation = chain.invoke(
    #     f"Evaluate the level of funniness of the joke: {state['joke']}\n"
    #     "Attention: jokes with surprises or puns are considered funny"
    # )
    # evaluation = evaluation.tool_calls[-1]["args"]

    return {"feedback" : evaluation["feedback"], "funny_or_not" : evaluation["grade"]}

def router_func(state: State) -> str:
    """The router function for conditional edges"""
    return "Accepted" if state.get("funny_or_not", None) == "funny" else "Rejected + Feedback"

# Create the workflow
builder = StateGraph(State)

builder.add_node("joke_generator", joke_generator_func)
builder.add_node("joke_evaluator", joke_evaluator_func)

builder.add_edge(START, "joke_generator")
builder.add_edge("joke_generator", "joke_evaluator")
builder.add_conditional_edges(
    "joke_evaluator",
    router_func,
    {
        "Accepted" : END,
        "Rejected + Feedback" : "joke_generator"
    }
)

graph = builder.compile()