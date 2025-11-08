# A2A Service - Research Agent

This directory contains the **service provider** code for the A2A Research Agent, reorganized from the original `app/` structure for better clarity and maintainability.

## Architecture Overview

```
+---------------------------------------------+
|  External Client (A2A Protocol)             |
+---------------------------------------------+
                    | HTTP/JSON-RPC
                    v
+---------------------------------------------+
|  Entry Point                                |
|  __main__.py - Server startup & AgentCard   |
+---------------------------------------------+
                    |
                    v
+---------------------------------------------+
|  Protocol Adapter Layer                     |
|  adapters/agent_executor.py                 |
|  Translates A2A <-> Core Agent              |
+---------------------------------------------+
                    |
                    v
+---------------------------------------------+
|  Core Business Logic                        |
|                                             |
|  +-------------------------------------+    |
|  | agent.py (Streaming Wrapper)       |    |
|  | - Provides streaming interface     |    |
|  | - Handles response formatting      |    |
|  +-------------------------------------+    |
|                |                            |
|                v                            |
|  +-------------------------------------+    |
|  | graph.py    |    |
|  | - LangGraph state machine          |    |
|  | - Node & edge definitions          |    |
|  +-------------------------------------+    |
|                |                            |
|                v                            |
|  +-------------------------------------+    |
|  | tools.py + rag_graph.py            |    |
|  | - Tavily, ArXiv, RAG tools         |    |
|  +-------------------------------------+    |
+---------------------------------------------+
```

## File Mapping from app/ to a2a_service/

| Original Location (`app/`) | New Location (`a2a_service/`) | Purpose |
|---------------------------|------------------------------|---------|
| `__main__.py` | `__main__.py` | Entry point - starts A2A server |
| `__init__.py` | `__init__.py` | Package initialization |
| `agent_executor.py` | `adapters/agent_executor.py` | A2A Protocol adapter - implements AgentExecutor interface |
| `agent.py` | `core/agent.py` | Streaming wrapper around the graph |
| `agent_graph_with_helpfulness.py` | `core/graph.py` | LangGraph state machine definition (function renamed from `build_agent_graph_with_helpfulness` to `build_graph`) |
| `tools.py` | `core/tools.py` | Tool implementations (Tavily, ArXiv) |
| `rag.py` | `core/rag_graph.py` | RAG implementation |
| `test_client.py` | *(moved to `a2a_client_examples/`)* | Client code - not part of service |
| `README.md` | *(replaced with this file)* | Documentation |

## Understanding the Wrapper Pattern

### Why Two "Agent" Files?

This confuses many people, but they serve different purposes:

#### 1. `core/graph.py` - The State Machine
- **What it is**: Pure LangGraph state machine definition
- **Purpose**: Defines the graph structure, nodes, and edges
- **Key function**: `build_graph()`
- **Reusable**: Can be used in any context (A2A, REST, CLI)
- **Analogy**: Like a car engine - the core mechanism

#### 2. `core/agent.py` - The Wrapper
- **What it is**: Application-specific wrapper class
- **Purpose**: Provides streaming interface and response formatting
- **Key class**: `Agent` with `stream()` method
- **Why needed**:
  - Converts graph execution to async streams
  - Formats responses for A2A protocol
  - Manages state and configuration
- **Analogy**: Like a car dashboard - the interface to the engine

### Example of the Wrapper Pattern

```python
# In core/agent.py
class Agent:
    def __init__(self):
        # Uses the graph, doesn't duplicate it
        self.graph = build_graph(...)

    async def stream(self, query, context_id):
        # Wraps graph execution with streaming
        for item in self.graph.stream(...):
            # Transform raw graph output
            # Into application-specific format
            yield formatted_response
```

Without this wrapper, the `adapters/agent_executor.py` would need to directly handle all the streaming logic, state management, and response formatting.

## Directory Structure Explained

```
a2a_service/
├── core/                    # Business logic (protocol-agnostic)
│   ├── __init__.py
│   ├── agent.py            # Streaming wrapper class
│   ├── graph.py  # LangGraph definition
│   ├── tools.py            # Tool implementations
│   └── rag_graph.py         # RAG functionality
│
├── adapters/               # Protocol-specific code
│   ├── __init__.py
│   └── agent_executor.py   # A2A Protocol adapter
│
├── __main__.py             # Entry point
├── __init__.py             # Package init
└── README.md               # This file
```

### Design Rationale

1. **Separation of Concerns**
   - `core/`: Contains all business logic that could work with any protocol
   - `adapters/`: Contains protocol-specific implementations
   - Clear boundary between "what the agent does" vs "how it communicates"

2. **Reusability**
   - The `core/` module can be reused with different protocols:
     - Add `adapters/rest/` for REST API
     - Add `adapters/grpc/` for gRPC
     - Core logic remains unchanged

3. **Maintainability**
   - New tools? Add to `core/tools.py`
   - New graph logic? Modify `core/graph.py`
   - New protocol? Add new adapter in `adapters/`

## Import Changes

All imports were updated from `app.*` to reflect the new structure:

| Old Import (app/) | New Import (a2a_service/) |
|------------|------------|
| `from app.agent import Agent` | `from .core.agent import Agent` |
| `from app.agent_executor import GeneralAgentExecutor` | `from .adapters.agent_executor import GeneralAgentExecutor` |
| `from app.graph import ...` | `from .core.graph import ...` |
| `from app.tools import get_tool_belt` | `from .tools import get_tool_belt` |
| `from app.rag import retrieve_information` | `from .rag_graph import retrieve_information` |

## Running the Service

**Workspace Operation**: This service shares the root project configuration. Run from the project root:

```bash
# From the project root directory
cd /path/to/15_A2A_LangGraph
uv run python -m a2a_service
```

**Alternative ports**: If port 10000 is in use, specify a different port:

```bash
# From the project root
uv run python -m a2a_service --port 10002
```

The service will start on `http://localhost:10000` (or the specified port)

## Package Configuration

The service is configured in the root `pyproject.toml` alongside the original `app`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["app", "a2a_service"]
```

This configuration allows both `app` and `a2a_service` to share dependencies while maintaining independent package structures. Run from the project root with `uv run python -m a2a_service`.

## Key Components

### Entry Point (`__main__.py`)
- Creates AgentCard with service capabilities
- Initializes GeneralAgentExecutor
- Starts A2A server with Starlette/uvicorn

### Protocol Adapter (`adapters/agent_executor.py`)
- Implements A2A's `AgentExecutor` interface
- Translates A2A requests to Agent calls
- Handles streaming responses via EventQueue

### Core Agent (`core/agent.py`)
- Wraps the LangGraph for streaming
- Manages conversation state via context_id
- Formats responses for protocol layer

### Graph Definition (`core/graph.py`)
- Defines the LangGraph state machine
- Includes helpfulness evaluation loop
- Pure graph logic, no protocol concerns

### Tools (`core/tools.py`, `core/rag_graph.py`)
- Web search via Tavily
- Academic search via ArXiv
- Document retrieval via RAG
- Assembled into tool belt for the graph

## Environment Variables

Required:
- `OPENAI_API_KEY`: For LLM operations

Optional:
- `TOOL_LLM_NAME`: Model name (default: gpt-4o-mini)
- `TOOL_LLM_URL`: API endpoint (default: https://api.openai.com/v1)
- `TAVILY_API_KEY`: For web search (if using Tavily tool)
- `OPENAI_CHAT_MODEL`: Model for RAG generation (default: gpt-4.1-nano)
- `RAG_DATA_DIR`: Directory containing PDF documents for RAG (default: data)

## Extension Guide

### Adding a New Tool
1. Implement tool in `core/tools.py` or separate file
2. Add to `get_tool_belt()` function
3. Tool automatically available to graph

### Modifying Graph Logic
1. Edit `core/graph.py`
2. Add/modify nodes or edges
3. Update state schema if needed

### Adding a New Protocol
1. Create `adapters/new_protocol/`
2. Implement protocol-specific adapter
3. No changes needed to `core/`

## Troubleshooting

### Common Issues

**"No module named a2a_service"**
- **Cause**: Running from the wrong directory or workspace not configured
- **Solution**: Run from the project root: `cd /path/to/15_A2A_LangGraph && uv run python -m a2a_service`

**"Address already in use"**
- **Cause**: Port 10000 is already occupied by another service
- **Solution**: Use a different port: `uv run python -m a2a_service --port 10002`

**Import errors**
- **Cause**: Incorrect import structure for standalone operation
- **Solution**: All imports within a2a_service should use relative imports with dots: `from .core.*` or `from .adapters.*`

### Verification

To verify the service is running correctly:

```bash
# Check if service is responding
curl http://localhost:10000/.well-known/agent-card

# Or check process
ps aux | grep a2a_service
```

## Summary

This structure provides:
- **Clear separation** between business logic and protocol concerns
- **Better organization** than flat `app/` structure
- **Reusability** of core components
- **Maintainability** through clear boundaries
- **Educational value** by making architecture explicit

The key insight is understanding that `agent.py` is not a duplicate of `graph.py` - it's a wrapper that provides the streaming interface on top of the pure graph definition. This is a common pattern in production systems where you separate the "what" (graph logic) from the "how" (streaming interface).