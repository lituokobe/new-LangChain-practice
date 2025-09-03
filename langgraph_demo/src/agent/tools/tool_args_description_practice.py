from typing import Annotated
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# # @tool(return_direct = True) #this means the tool result will be directly returned to user, by default is False
# @tool
# def calculate1 (a : float, b : float, operation: str) -> float:
#     """function to calculate the math operation between 2 numbers"""
#     print(f"first number: {a}, second number: {b}, operation: {operation}")
#     result = 0.0
#     match operation:
#         case "add":
#             result = a + b
#         case "subtract":
#             result = a-b
#         case "multiply":
#             result = a*b
#         case "divide":
#             if b != 0 :
#                 result = a/b
#             else:
#                 raise ValueError("denominator cannot be zero")
#     return result
#
# print(calculate1.name)
# print(calculate1.description)
# print(calculate1.args)
# print(calculate1.args_schema.model_json_schema())
# print(calculate1.return_direct)
# print(type(calculate1))
#
# # TODO: Describe the arguments with a class in args_schema
# class CalculateArgs(BaseModel):
#     a : float = Field(description = "first number to input")
#     b: float = Field(description="second number to input")
#     operation: str = Field(description="calculation type, can only be add, subtract, multiply or divide.")
#
# @tool("calculate", args_schema = CalculateArgs) # args_schema uses a class to describe the input arguments
# def calculate2 (a : float, b : float, operation: str) -> float:
#     """function to calculate the math operation between 2 numbers"""
#     print(f"first number: {a}, second number: {b}, operation: {operation}")
#     result = 0.0
#     match operation:
#         case "add":
#             result = a + b
#         case "subtract":
#             result = a-b
#         case "multiply":
#             result = a*b
#         case "divide":
#             if b != 0 :
#                 result = a/b
#             else:
#                 raise ValueError("denominator cannot be zero")
#     return result
#
# print(calculate2.name)
# print(calculate2.description)
# print(calculate2.args)
# print(calculate2.args_schema.model_json_schema())
# print(calculate2.return_direct)
# print(type(calculate2))

# TODO: Describe the arguments with Annotated
@tool("calculate") # args_schema uses a class to describe the input arguments
def calculate3 (a : Annotated[float, "first number to input"],
                b : Annotated[float, 'second number to input'],
                operation: Annotated[str, 'calculation type, can only be add, subtract, multiply or divide.']) -> float:
    """function to calculate the math operation between 2 numbers"""
    print(f"first number: {a}, second number: {b}, operation: {operation}")
    result = 0.0
    match operation:
        case "add":
            result = a + b
        case "subtract":
            result = a-b
        case "multiply":
            result = a*b
        case "divide":
            if b != 0 :
                result = a/b
            else:
                raise ValueError("denominator cannot be zero")
    return result
print(calculate3.name)
print(calculate3.description)
print(calculate3.args)
print(calculate3.args_schema.model_json_schema())
print(calculate3.return_direct)
print(type(calculate3))
print(calculate3.invoke({"a":4, "b":3, "operation":"add"}))

# TODO: Describe the arguments in the note, this is Google style
@tool("calculate", parse_docstring = True) #parse_doc_string must be set true to make the description be read
def calculate4 (a : float, b : float, operation: str) -> float:
    """
    function to calculate the math operation between 2 numbers

    Args:
        a (float): first number to input
        b (float): second number to input
        operation (str): calculation type, can only be add, subtract, multiply or divide

    Returns:
        float: the operation result of the 2 numbers' operation
    """
    print(f"first number: {a}, second number: {b}, operation: {operation}")
    result = 0.0
    match operation:
        case "add":
            result = a + b
        case "subtract":
            result = a-b
        case "multiply":
            result = a*b
        case "divide":
            if b != 0 :
                result = a/b
            else:
                raise ValueError("denominator cannot be zero")
    return result

print(calculate4.name)
print(calculate4.description)
print(calculate4.args)
print(calculate4.args_schema.model_json_schema())
print(calculate4.return_direct)
print(type(calculate4))
print(calculate4.invoke({"a":4, "b":3, "operation":"add"}))