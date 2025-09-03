from langchain_core.tools import tool, StructuredTool


def calculate5 (a : float, b : float, operation: str) -> float:
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

async def calculate6 (a : float, b : float, operation: str) -> float:
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

calculator = StructuredTool.from_function(
    func = calculate5,
    name = "calculator",
    description = "function to calculate the math operation between 2 numbers",
    return_direct = False,
    coroutine = calculate6 # the async version of the function, it will execute in async environment
)

print(calculator.description)