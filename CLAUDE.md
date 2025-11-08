# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a learning repository for implementing the **A2A (Agent-to-Agent) Protocol** using LangGraph. The project demonstrates intelligent agent communication with helpfulness evaluation and multi-turn conversation capabilities.

### Core Architecture

The repository contains **two separate implementations** of the same agent:

1. **`app/`** - Original monolithic implementation (for reference/learning)
2. **`a2a_service/`** - Refactored service with clean architecture (production-ready)

Both implement the same LangGraph agent with A2A Protocol, but `a2a_service/` follows a layered architecture pattern:
- `core/` - Protocol-agnostic business logic (graph, tools, RAG)
- `adapters/` - A2A Protocol-specific implementation
- Clear separation of concerns for maintainability and extensibility

### Key Concept: The Wrapper Pattern

This codebase uses a wrapper pattern that often confuses developers:

- **`core/graph.py`**: Pure LangGraph state machine definition - the "engine"
- **`core/agent.py`**: Application wrapper providing streaming interface - the "dashboard"

The `Agent` class wraps the graph to provide streaming responses and state management without duplicating graph logic.

## Common Development Commands

### Setup and Installation

```bash
# Quick setup (recommended)
./quickstart.sh

# Manual setup
uv sync
python3 setup.py
mkdir -p data
```

### Running the A2A Service

```bash
# Start the A2A server (default: localhost:10000)
cd a2a_service
uv run python -m a2a_service

# Custom host/port
uv run python -m a2a_service --host 0.0.0.0 --port 8080
```

### Testing

```bash
# Test the A2A service with client
uv run python a2a_client_examples/test_client.py

# Run environment check
uv run python check_env.py
```

### LangGraph Development

```bash
# Start LangGraph dev server (from root directory)
uv run langgraph dev

# Access LangGraph Studio
# API: http://localhost:2024
# Studio: https://smith.langchain.com/studio?baseUrl=http://localhost:2024
```

## Environment Configuration

Required environment variables (`.env` file):

```bash
# LLM Configuration
OPENAI_API_KEY=your_key_here
TOOL_LLM_URL=https://api.openai.com/v1
TOOL_LLM_NAME=gpt-4o-mini

# Web Search
TAVILY_API_KEY=your_key_here

# RAG Configuration
RAG_DATA_DIR=data
OPENAI_CHAT_MODEL=gpt-4o-mini
```

## Agent Architecture

### The Helpfulness Evaluation Loop

The core innovation is a **post-response evaluation cycle**:

1. **Agent Node**: LLM processes query and decides if tools are needed
2. **Action Node**: Executes tools (Tavily web search, ArXiv papers, RAG retrieval)
3. **Helpfulness Node**: Self-evaluation loop where the same LLM assesses if its response is satisfactory ('Y'/'N'), triggering retry if needed (max 10 iterations)
4. **Decision Logic**:
   - Helpful (Y) → END
   - Not helpful (N) → Loop back to Agent (max 10 iterations)

### State Management

```python
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]  # Conversation history
    structured_response: Any                 # ResponseFormat for A2A
```

### Available Tools

Configured in `core/tools.py`:
- **Tavily Search**: Real-time web search (max 5 results)
- **ArXiv Search**: Academic paper retrieval
- **RAG Retrieval**: Document search via Qdrant vector store

## Important Implementation Notes

### A2A Protocol Compliance

The `adapters/agent_executor.py` implements the A2A `AgentExecutor` interface:
- Translates A2A requests → core Agent calls
- Handles streaming via EventQueue
- Manages multi-turn conversations with context_id

### AgentCard Components

Defined in `a2a_service/__main__.py`:
- **Capabilities**: `streaming=True`, `push_notifications=True`
- **Skills**: Describes web_search, arxiv_search, rag_search capabilities
- **Content Types**: Supported input/output modes from Agent class

### RAG Implementation Details

The `core/rag.py` uses token-aware chunking:
- PDFs loaded recursively from `RAG_DATA_DIR`
- Chunking via `tiktoken` for accurate token counting (not just characters)
- In-memory Qdrant vector store (no persistence by default)
- Two-node LangGraph: retrieve → generate

### Helpfulness Loop Protection

Multiple safeguards prevent infinite loops:
1. Message count tracking (limit: 10 iterations)
2. Hard stop returns `HELPFULNESS:END` when exceeded
3. Decision router terminates on limit markers

## Code Organization Patterns

### Import Conventions

When working with `a2a_service/`:

```python
# Core business logic
from core.agent import Agent
from core.graph import build_graph
from core.tools import get_tool_belt
from core.rag import retrieve_information

# Protocol adapters
from adapters.agent_executor import GeneralAgentExecutor
```

### Extending the System

**Adding a new tool:**
1. Implement in `core/tools.py` using `@tool` decorator
2. Add to `get_tool_belt()` function
3. Tool automatically available to graph

**Modifying graph logic:**
1. Edit `core/graph.py` for node/edge changes
2. Update `AgentState` if new state fields needed
3. No changes needed to adapter layer

**Adding a new protocol:**
1. Create `adapters/new_protocol/`
2. Implement protocol-specific adapter
3. Core logic remains untouched

## Testing and Troubleshooting

### Common Issues

| Error | Solution |
|-------|----------|
| Internal Error (-32603) | Check model names in `.env` (common: wrong model like gpt-4.1 instead of gpt-4o) |
| Tool failures | Verify `TAVILY_API_KEY` and `OPENAI_API_KEY` are set |
| RAG errors | Ensure PDFs in `data/` directory and OpenAI key configured |
| Timeout on helpfulness | Evaluation can take 10-30s; this is expected behavior |

### Quick Diagnostics

```bash
# Verify environment
uv run python check_env.py

# Check available tools
uv run python -c "from a2a_service.core.tools import get_tool_belt; print([t.name for t in get_tool_belt()])"

# Test RAG loading
uv run python -c "from a2a_service.core.rag import get_rag_graph; get_rag_graph(); print('OK')"
```

## Version Constraints

The project uses specific LangChain/LangGraph versions to avoid deprecation warnings:

```toml
# Critical constraints
langgraph = ">=0.3.18,<1.0"
langchain-openai = ">=0.1.0,<1.0"
langchain-community = ">=0.3.0,<1.0"
```

When adding dependencies, maintain these version constraints.

## Learning Objectives

This is an educational repository. Key concepts to understand:

1. **A2A Protocol**: Agent discovery, communication, and evaluation
2. **Wrapper Pattern**: Separation between graph definition and streaming interface
3. **Helpfulness Loop**: Autonomous quality evaluation and iteration
4. **Clean Architecture**: Protocol-agnostic core vs. protocol-specific adapters

When making changes, preserve the educational structure and clear separation of concerns.