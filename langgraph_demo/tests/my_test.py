# test langgraph API
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:2024")

for chunk in client.runs.stream(
    None,  # Threadless run
    "agent", # Name of assistant. Defined in langgraph.json.
    input={
        "messages": [{
            "role": "human",
            # "content" : "What's the weather like in Jakarta?",
            # "content": "What's the age of current user?",
            # "content": "计算一下(3 + 5) x 12的结果",
            "content": "Can you greet current user?",
        }],

    },
    stream_mode="messages-tuple", # output messages token by token with metadata
    # stream_mode="messages", # it outputs message token by token
    # stream_mode="updates",
    config={"configurable": {"user_name": "user_456"}}
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    # if isinstance(chunk.data, list) and 'type' in chunk.data[0] and chunk.data[0]['type'] == 'AIMessageChunk':
    #     print(chunk.data[0]['content'], end='|')

