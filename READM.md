# 🧠 LangChain & LangGraph Practice Repository

This repository documents my hands-on exploration of the latest **LangChain** and **LangGraph** techniques, starting from March 2025. It includes prompt engineering, embedding workflows, multimodal agents, and LangGraph CLI-based agent orchestration.

---

## 📦 Script Overview

### 🔧 Prompt Engineering & Structured Output
- `demo1.py`: Demonstrates various prompt construction methods using `PromptTemplate`, `ChatPromptTemplate`, and `FewShotPromptTemplate`, along with structured output formatting.
- `demo2.py`: Explores the use of `with_listeners` on a `Runnable` object for enhanced observability and debugging.

### 🧬 Embeddings & Vector Search
- `embeddings.py`: Compares embedding generation across multiple libraries—OpenAI, LangChain, Hugging Face, and Sentence Transformers.
- `embedding_information_query.py`: Implements cosine similarity-based querying from a vector database using custom embeddings.
- `vector_database_chroma.py`: Builds a vector store using **ChromaDB**.
- `vector_database_FAISS.py`: Builds a vector store using **FAISS**.

### 📚 Retrieval-Augmented Generation (RAG)
- `RAG_chain.py`: Constructs a RAG pipeline using an online article as the knowledge base, with retrieval and question-answering capabilities.

### 🗣️ Multimodal Chatbots
- `Multimodal_chatbot.py`: GUI-based chatbot that maintains recent chat history (last 2 messages) and summaries, with voice input support.
- `Multimodal_chatbot2.py`: Enhanced chatbot supporting both voice and image inputs.

---

## ⚙️ LangGraph Projects (Built with LangGraph CLI)

I created two LangGraph projects using the LangGraph CLI, both designed to run on a local LangGraph server. These projects explore agent orchestration, memory management, tool integration, and human-in-the-loop workflows.

---

### 📁 `langgraph_demo`

#### 🔧 `src/agent`: Agent Implementations
- `graph.py`: Defines a simple agent using a custom prompt function.
- `mcp_agent.py`: Agent integrated with an MCP tool using SSE connection.
- `mcp_agent2.py`: Variant of `mcp_agent.py` that requires an API key for tool access.
- `my_agent.py`: Agent with persistent memory—chat history stored in a SQLite database.
- `my_state.py`: Custom `AgentState` class used in `graph.py` to manage structured state.

#### 🛠️ `src/mcp_server`: MCP Server Setup
- `tool_server.py`: MCP server hosting multiple tools, prompts, and resources.
- `tool_server2.py`: MCP server with key-based access and extended toolset.
- `start_sse_server.py`: Launches an SSE-based MCP server.
- `start_streamable_server.py`: Launches a streamable MCP server for real-time interaction.

---

### 📁 `langgraph_demo2`

#### 🧠 `src/agent`: Advanced LangGraph Workflows
- `graph.py`: Generates jokes and evaluates their humor level using a LangGraph workflow.
- `graph2.py`: Dynamically decides whether to invoke tools and supports synchronous multi-tool execution.
- `graph3_toolnode.py`: Simplified version of `graph2.py` using `ToolNode` and `tools_condition` for cleaner logic.
- `graph4_human_interference.py`: Adds human approval before executing any tool calls.
- `graph5.py`: Adds human approval only before invoking the **search tool**; other tools execute automatically.

---

## 🚀 Purpose

This repository serves as a sandbox for:
- Practicing agentic workflows and orchestration
- Exploring LangChain and LangGraph integration
- Building multimodal, memory-aware agents
- Evaluating tool-based reasoning and human-in-the-loop designs

Feel free to fork, explore, and contribute!
