from new_langchaing_practice.langgraph_demo.src.mcp_server.tool_server2 import luis_miguel_server

if __name__ == "__main__":
    luis_miguel_server.run(
        transport = "streamable-http",
        host = "127.0.0.1",
        port = 8000,
        log_level = "debug",
        path = "/streamable"
    )