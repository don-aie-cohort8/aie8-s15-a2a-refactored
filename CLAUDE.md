# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a learning repository for implementing the **A2A (Agent-to-Agent) Protocol** using LangGraph. The project demonstrates intelligent agent communication with helpfulness evaluation and multi-turn conversation capabilities.

### Core Architecture

The repository contains **three major components**:

1. **`app/`** - Original monolithic implementation (for reference/learning)
2. **`a2a_service/`** - Refactored service with clean architecture (production-ready)
   - `core/` - Protocol-agnostic business logic (graph, tools, RAG)
   - `adapters/` - A2A Protocol-specific implementation
   - Clear separation of concerns for maintainability and extensibility

3. **Repository Analyzer Framework** (`ra_*` directories) - Portable multi-domain analysis toolkit
   - `ra_orchestrators/` - Multi-agent workflow orchestrators for different domains
   - `ra_agents/` - Reusable agent definitions (JSON-based)
   - `ra_tools/` - Tool integrations (MCP registry, Figma)
   - `ra_output/` - Timestamped analysis results (gitignored)

### Relationship Between Components

- **A2A Service** (`app/` and `a2a_service/`): Demonstrates agent-to-agent protocol implementation with a specific use case (helpfulness evaluation)
- **Repository Analyzer**: Applies the same multi-agent patterns at a higher abstraction level - orchestrators coordinate multiple specialized agents across different domains
- **Shared Patterns**: Both use LangGraph, specialized agents, and tool integration, but at different scales and for different purposes

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

### Repository Analyzer Framework

```bash
# Architecture analysis (generates ra_output/architecture_{timestamp}/)
python -m ra_orchestrators.architecture_orchestrator "Project Name"

# UX design workflow (generates ra_output/ux_{timestamp}/)
python -m ra_orchestrators.ux_orchestrator "Project Name"

# List available agents
python -c "from ra_agents.registry import AgentRegistry; print(AgentRegistry().discover_agents())"
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
| Internal Error (-32603) | Check model names in `.env` (common: wrong model like gpt-4.1 instead of gpt-4o-mini or gpt-4o) |
| Tool failures | Verify `TAVILY_API_KEY` and `OPENAI_API_KEY` are set |
| RAG errors | Ensure PDFs in `data/` directory and OpenAI key configured |
| Timeout on helpfulness | Evaluation can take 10-30s; this is expected behavior |
| ModuleNotFoundError for `ra_*` | Ensure running from repository root, not subdirectory |
| Agent not writing files | Verify agent prompt includes explicit Write tool instruction |

### Quick Diagnostics

```bash
# Verify environment
uv run python check_env.py

# Check available tools
uv run python -c "from a2a_service.core.tools import get_tool_belt; print([t.name for t in get_tool_belt()])"

# Test RAG loading
uv run python -c "from a2a_service.core.rag_graph import get_rag_graph; get_rag_graph(); print('OK')"

# Check RA framework agents
python -c "from ra_agents.registry import AgentRegistry; r = AgentRegistry(); print('Available agents:', list(r.discover_agents().keys()))"

# List MCP servers
python -c "from ra_tools.mcp_registry import MCPRegistry; r = MCPRegistry(); print('MCP servers:', r.discover_servers())"
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

## Repository Analyzer Framework

### Overview

The `ra_*` directories contain a **portable, drop-in analysis framework** designed to be added to any repository for comprehensive multi-domain analysis. Key features:

- **Portability**: Drop into any repository without modification
- **No Collisions**: `ra_` prefix avoids conflicts with existing code
- **Timestamped Outputs**: Each run creates `ra_output/{domain}_{YYYYMMDD_HHMMSS}/`
- **Multi-Domain**: Architecture, UX, DevOps (future), Testing (future)

### Framework Components

**Base Orchestrator** (`ra_orchestrators/base_orchestrator.py`):
- Phase execution engine for sequential/concurrent workflows
- Agent lifecycle management
- Progress tracking with tool usage visibility
- Cost monitoring per phase
- Output verification and error handling

**Domain Orchestrators**:
- `architecture_orchestrator.py` - Repository structure, diagrams, data flows, API docs
- `ux_orchestrator.py` - User research, IA, visual design, prototyping

**Agent Registry** (`ra_agents/registry.py`):
- JSON-based agent definitions in `ra_agents/{domain}/`
- Lazy loading and caching
- Cross-domain agent reusability

**Tool Integrations** (`ra_tools/`):
- `mcp_registry.py` - Discover and manage MCP server connections
- `figma_integration.py` - Figma MCP and REST API wrapper

### Adding a New Domain Orchestrator

Target: <1 day to implement a new domain

1. Create orchestrator class inheriting from `BaseOrchestrator`
2. Define agents as JSON in `ra_agents/{domain}/`
3. Implement `get_agent_definitions()`, `get_allowed_tools()`, and `run()`
4. Run from repository root: `python -m ra_orchestrators.custom_orchestrator`

See `ra_orchestrators/CLAUDE.md` for detailed patterns and best practices.

### Output Structure

All analyses generate timestamped outputs to avoid collisions:

```
ra_output/
├── architecture_20251108_122754/
│   ├── docs/
│   ├── diagrams/
│   └── README.md
├── ux_20251108_140000/
│   ├── 01_research/
│   ├── 02_ia/
│   └── ...
```

## Learning Objectives

This is an educational repository. Key concepts to understand:

1. **A2A Protocol**: Agent discovery, communication, and evaluation
2. **Wrapper Pattern**: Separation between graph definition and streaming interface
3. **Helpfulness Loop**: Autonomous quality evaluation and iteration
4. **Clean Architecture**: Protocol-agnostic core vs. protocol-specific adapters
5. **Multi-Agent Orchestration**: Framework-based orchestrators with specialized agents
6. **Portable Analysis**: Drop-in toolkit design for repository analysis

When making changes, preserve the educational structure and clear separation of concerns.