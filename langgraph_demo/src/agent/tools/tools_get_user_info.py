from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from src.agent.my_state import CustomState


@tool
def get_user_info_by_name(config: RunnableConfig):
    """
    Get all user information including gender, age, etc.
    :return:
    """
    user_name = config["configurable"].get("user_name", "Miguel")
    print(f"tool is called, the user name passed is {user_name}")
    # simulate obtaining user info
    return {"username":user_name, "gender":"male", "age":18}

@tool
def get_user_name(tool_call_id: Annotated[str, InjectedToolCallId],
                  config: RunnableConfig) -> Command:
    # InjectedToolCallId will let the tool call id injected to this function from state
    # In the old version of langgraph, you need to get the toolcall id from the last message in the state, much more complex
    """
    Get current user' name, in order to generate content.
    """
    user_name = config["configurable"].get("user_name", "Miguel")
    print(f"tool is called, the user name passed in is {user_name}")
    # simulate obtaining user info
    return Command(update = {
        "user_name" : user_name, # Update user name in state
        # Update a message after tool is executed
        "messages" : [
            ToolMessage(
                content = "Obtained current username successfully.",
                tool_call_id = tool_call_id
            )
        ]
    })

@tool
def greet_user(state: Annotated[CustomState, InjectedState]) -> None:
    """
    After getting user's name, generate greeting message
    """
    username = state["username"]
    return f"Congrats, {username}!"