# LCEL practice
import time
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers import Run

# TODO: with_listener: lifecycle management
def test4 (n: int):
    time.sleep(n)
    return n*2

r1 = RunnableLambda(test4)

# on_start function is an instance of Run class
def on_start(run_obj: Run):
    """
    call when r1 is initialized
    :param run_obj:
    :return:
    """
    print("ri is initialized at ", run_obj.start_time)

def on_end(run_obj: Run):
    """
    call when r1 is initialized
    :param run_obj:
    :return:
    """
    print("ri is ended at ", run_obj.end_time)

chain = r1.with_listeners(on_start = on_start, on_end = on_end)
chain.invoke(2)