# Component Inventory

## Overview

This document provides a comprehensive inventory of all components in the `a2a-sample-agent-langgraph` codebase. The project implements a LangGraph-based AI agent with A2A (Agent-to-Agent) Protocol support, featuring web search, academic paper search, and RAG (Retrieval-Augmented Generation) capabilities.

The codebase is organized into two main implementations:
1. **app/** - Original/legacy implementation
2. **a2a_service/** - Refactored implementation with better separation of concerns

## Public API

### Top-Level Modules

#### a2a_service Package
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/__init__.py`
- **Lines 1-5:** Package initialization and exports
- **Exported Classes:**
  - `Agent` - Core agent class
  - `GeneralAgentExecutor` - A2A protocol executor
- **Version:** 0.1.0
- **Purpose:** Main package for the refactored A2A service with clean architecture

### Core Business Logic (a2a_service/core/)

#### Agent Class
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/agent.py`

**Purpose:** Core agent implementation with LangGraph integration and A2A protocol compatibility

**Public Classes:**

1. **ResponseFormat** (Lines 18-22)
   - Pydantic BaseModel for structured responses
   - Fields:
     - `status`: Literal["input_required", "completed", "error"]
     - `message`: str
   - Purpose: Standardize agent response format for A2A protocol

2. **Agent** (Lines 25-113)
   - Main agent class with streaming and state management
   - **Constants:**
     - `SYSTEM_INSTRUCTION` (Lines 28-33): System prompt for agent behavior
     - `FORMAT_INSTRUCTION` (Lines 36-39): Response format instructions
     - `SUPPORTED_CONTENT_TYPES` (Line 113): ["text", "text/plain"]

   - **Public Methods:**
     - `__init__(self)` (Lines 42-55): Initializes model and graph with memory
     - `async stream(self, query, context_id)` (Lines 57-80): Streams agent responses
       - Yields status updates during tool execution
       - Returns task completion status
     - `get_agent_response(self, config)` (Lines 82-111): Extracts structured response from state
       - Maps ResponseFormat status to task completion flags

#### Graph Builder
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/graph.py`

**Purpose:** LangGraph state graph with helpfulness evaluation loop

**Public Classes:**

1. **AgentState** (Lines 19-23)
   - TypedDict for graph state
   - Fields:
     - `messages`: Annotated[List, add_messages]
     - `structured_response`: Any (ResponseFormat | None)

**Public Functions:**

1. **build_graph(model, system_instruction, format_instruction, checkpointer=None)** (Lines 102-163)
   - Builds and compiles agent graph with helpfulness evaluation
   - Parameters:
     - model: LLM model instance
     - system_instruction: System prompt
     - format_instruction: Response format instructions
     - checkpointer: Optional state persistence
   - Returns: Compiled StateGraph
   - Graph nodes: "agent", "action", "helpfulness"
   - Includes feedback loop for response quality

2. **build_model_with_tools(model)** (Lines 26-30)
   - Binds tool belt to model
   - Returns: Model with tools attached

#### RAG System
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/rag_graph.py`

**Purpose:** Retrieval-Augmented Generation with PDF document processing

**Public Tool:**

1. **@tool retrieve_information(query)** (Lines 115-125)
   - LangChain tool decorator for RAG functionality
   - Parameters:
     - query: Annotated[str, "query to ask the retrieve information tool"]
   - Returns: Generated response from context
   - Purpose: Retrieve and generate answers from loaded PDF documents
   - Uses cached RAG graph from environment variable `RAG_DATA_DIR`

**Public Classes:**

1. **_RAGState** (Lines 35-39)
   - TypedDict for RAG graph state
   - Fields:
     - question: str
     - context: List[Document]
     - response: str

**Public Functions:**

1. **_get_rag_graph()** (Lines 108-112)
   - LRU cached function (maxsize=1)
   - Returns compiled RAG graph
   - Loads from environment: `RAG_DATA_DIR` (default: "data")

#### Tool Belt
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/tools.py`

**Purpose:** Centralized tool assembly for agent capabilities

**Public Functions:**

1. **get_tool_belt()** (Lines 15-18)
   - Returns: List of available tools
   - Tools included:
     1. TavilySearch (max_results=5) - Web search
     2. ArxivQueryRun() - Academic paper search
     3. retrieve_information - RAG tool
   - Purpose: Provide unified tool interface to agent graphs

### Protocol Adapters (a2a_service/adapters/)

#### GeneralAgentExecutor
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/adapters/agent_executor.py`

**Purpose:** A2A protocol adapter bridging core agent to A2A server

**Public Classes:**

1. **GeneralAgentExecutor(AgentExecutor)** (Lines 27-93)
   - Implements A2A AgentExecutor interface
   - **Public Methods:**
     - `__init__(self)` (Lines 30-31): Initializes core agent
     - `async execute(self, context: RequestContext, event_queue: EventQueue)` (Lines 33-85)
       - Main execution method for A2A requests
       - Handles task creation, streaming, and status updates
       - Maps agent responses to A2A TaskState
     - `async cancel(self, context: RequestContext, event_queue: EventQueue)` (Lines 90-93)
       - Raises UnsupportedOperationError
   - **Private Methods:**
     - `_validate_request(self, context: RequestContext)` (Lines 87-88): Currently returns False

### Legacy Implementation (app/)

#### Legacy Agent Class
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent.py`

**Note:** Similar structure to a2a_service/core/agent.py but imports from different modules

**Public Classes:**

1. **ResponseFormat** (Lines 17-21)
2. **Agent** (Lines 24-115)
   - Uses `app.agent_graph_with_helpfulness.build_agent_graph_with_helpfulness`
   - Identical API to refactored version

#### Legacy Graph Builder
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent_graph_with_helpfulness.py`

**Public Functions:**

1. **build_agent_graph_with_helpfulness(model, system_instruction, format_instruction, checkpointer=None)** (Lines 99-160)
   - Same functionality as refactored version

#### Legacy RAG System
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/rag.py`

**Public Tool:**

1. **@tool retrieve_information(query)** (Lines 115-125)
   - Identical to refactored version

#### Legacy Tools
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/tools.py`

**Public Functions:**

1. **get_tool_belt()** (Lines 15-18)
   - Returns same tool list as refactored version

#### Legacy Agent Executor
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent_executor.py`

**Public Classes:**

1. **GeneralAgentExecutor(AgentExecutor)** (Lines 27-93)
   - Identical to refactored version but imports from `app.agent`

## Internal Implementation

### Internal Graph Functions (a2a_service/core/graph.py)

1. **call_model(state, model)** (Lines 33-38)
   - Internal node function for invoking model
   - Returns: Dict with messages

2. **route_to_action_or_helpfulness(state)** (Lines 41-46)
   - Conditional edge router
   - Returns: "action" if tool calls present, else "helpfulness"

3. **helpfulness_node(state, model)** (Lines 49-84)
   - Evaluates response helpfulness
   - Loop limit: 10 messages (Line 52)
   - Returns: Dict with helpfulness decision message
   - Uses PromptTemplate for evaluation

4. **helpfulness_decision(state)** (Lines 87-99)
   - Routes based on helpfulness evaluation
   - Returns: "end", "continue", or END
   - Guards against infinite loops

### Internal RAG Functions (a2a_service/core/rag_graph.py)

1. **_tiktoken_len(text)** (Lines 29-32)
   - Calculates token length using tiktoken
   - Model: "gpt-4o"
   - Purpose: Chunk size measurement

2. **_build_rag_graph(data_dir)** (Lines 42-105)
   - Constructs RAG pipeline
   - Steps:
     - Loads PDFs with DirectoryLoader and PyMuPDFLoader (Lines 54-59)
     - Splits with RecursiveCharacterTextSplitter (Lines 70-73)
       - chunk_size: 750
       - chunk_overlap: 0
     - Creates OpenAI embeddings (model: "text-embedding-3-small")
     - Builds in-memory Qdrant vector store
     - Defines retrieve and generate nodes
   - Returns: Compiled StateGraph

3. **retrieve(state)** (Lines 91-93)
   - Internal node: retrieves relevant documents
   - Returns: Dict with context

4. **generate(state)** (Lines 95-100)
   - Internal node: generates response from context
   - Uses ChatOpenAI with model from env (default: "gpt-4.1-nano")
   - Returns: Dict with response

### Setup and Configuration Utilities

#### Environment Setup Script
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/setup.py`

**Purpose:** Interactive setup script for environment configuration

**Functions:**

1. **create_env_file()** (Lines 9-69)
   - Creates .env file with user input
   - Prompts for:
     - OpenAI API key
     - OpenAI model name (default: "gpt-4.1-mini")
     - Tavily API key
     - RAG data directory (default: "data")
   - Creates data directory
   - Overwrites existing .env with confirmation

2. **main()** (Lines 72-81)
   - Entry point with error handling
   - Catches KeyboardInterrupt and general exceptions

#### Environment Checker
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/check_env.py`

**Purpose:** Validates environment configuration

**Inline Script** (Lines 1-57)
   - Checks API keys:
     - OPENAI_API_KEY
     - TAVILY_API_KEY
   - Validates LLM configuration:
     - TOOL_LLM_URL (default: "https://api.openai.com/v1")
     - TOOL_LLM_NAME (default: "gpt-4o-mini")
   - Checks RAG configuration:
     - RAG_DATA_DIR (default: "data")
     - OPENAI_CHAT_MODEL (default: "gpt-4o-mini")
   - Counts PDF files in data directory
   - Masks sensitive keys for security

## Entry Points

### Primary Entry Points

#### A2A Service Entry Point (Refactored)
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/__main__.py`

**Purpose:** Main entry point for refactored A2A service

**Entry Point:** `python -m a2a_service`

**Components:**

1. **MissingAPIKeyError Exception** (Lines 33-34)
   - Custom exception for missing API key

2. **@click.command main(host, port)** (Lines 37-109)
   - CLI command with Click framework
   - **Options:**
     - `--host` (default: 'localhost')
     - `--port` (default: 10000)
   - **Initialization:**
     - Validates OPENAI_API_KEY (Lines 43-46)
     - Creates AgentCapabilities (Line 48)
     - Defines AgentSkills (Lines 49-71):
       - web_search
       - arxiv_search
       - rag_search
     - Creates AgentCard (Lines 72-81)
     - Initializes A2A server components:
       - httpx.AsyncClient
       - InMemoryPushNotificationConfigStore
       - BasePushNotificationSender
       - DefaultRequestHandler with GeneralAgentExecutor
       - A2AStarletteApplication
   - Runs uvicorn server (Line 99)
   - Error handling (Lines 102-109)

3. **if __name__ == '__main__'** (Lines 112-113)
   - Invokes main()

#### Legacy App Entry Point
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/__main__.py`

**Purpose:** Main entry point for legacy implementation

**Entry Point:** `python -m app`

**Note:** Identical structure to a2a_service/__main__.py but imports from `app` module

### Client Examples

#### A2A Client Test (Refactored)
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_client_examples/test_client.py`

**Purpose:** Example A2A client demonstrating protocol usage

**Entry Point:** `python a2a_client_examples/test_client.py`

**async main() Function** (Lines 21-193)
   - Configures logging (Lines 23-24)
   - Creates httpx client with 60s timeout (Line 31)
   - Initializes A2ACardResolver (Lines 33-37)
   - Fetches agent cards:
     - Public card from AGENT_CARD_WELL_KNOWN_PATH (Lines 44-57)
     - Extended card if supported (Lines 59-93)
   - Creates A2A client via ClientFactory (Lines 111-119)
   - Demonstrates:
     - **Simple message send** (Lines 122-137)
     - **Multi-turn conversation** (Lines 140-180)
       - First message
       - Follow-up with task_id and context_id
     - **Streaming message** (Lines 183-192)

**if __name__ == '__main__'** (Lines 195-198)
   - Runs asyncio.run(main())

#### Legacy App Client Test
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/test_client.py`

**Purpose:** Legacy client test (identical to refactored version)

**Entry Point:** `python app/test_client.py`

### Utility Entry Points

#### Setup Script
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/setup.py`

**Purpose:** Interactive environment configuration

**Entry Point:** `python setup.py`

**if __name__ == "__main__"** (Lines 84-85)
   - Calls main() function

#### Environment Checker
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/check_env.py`

**Purpose:** Validate environment setup

**Entry Point:** `python check_env.py`

**Script execution:** Lines 1-57 (runs on import)

## Package Structure

### Project Metadata
**Location:** `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/pyproject.toml`

**Package Information:**
- Name: "a2a-sample-agent-langgraph"
- Version: "0.1.0"
- Python: ">=3.13"

**Packaged Modules:** (Lines 29-30)
- `app`
- `a2a_service`

**Key Dependencies:**
- langgraph: >=0.3.18,<1.0
- langchain-openai: >=0.1.0,<1.0
- langchain-community: >=0.3.0,<1.0
- langchain-tavily: >=0.1.0,<1.0
- a2a-sdk: >=0.3.0
- pydantic: >=2.10.6
- uvicorn: >=0.34.2
- httpx: >=0.28.1
- click: >=8.1.8
- tiktoken: >=0.5.1
- qdrant-client: >=1.7.0
- pymupdf: >=1.23.0

## Architecture Patterns

### Separation of Concerns

1. **Core Business Logic** (a2a_service/core/)
   - Protocol-agnostic agent implementation
   - Graph construction and state management
   - RAG and tool integration
   - No A2A-specific code

2. **Protocol Adapters** (a2a_service/adapters/)
   - Bridges core logic to A2A protocol
   - Handles A2A-specific types and events
   - Minimal business logic

3. **Entry Points** (a2a_service/__main__.py)
   - Server initialization
   - Configuration and dependency injection
   - Error handling

### State Management

1. **LangGraph Checkpointer**
   - MemorySaver for conversation state
   - Thread-based context isolation
   - Shared across both implementations

2. **Structured Responses**
   - ResponseFormat Pydantic model
   - Three states: input_required, completed, error
   - JSON schema validation

### Tool Integration

1. **Unified Tool Belt**
   - get_tool_belt() provides consistent interface
   - Three tool types:
     - External API (Tavily, ArXiv)
     - Internal RAG
   - LangChain tool decorator pattern

2. **RAG Pipeline**
   - In-memory Qdrant vector store
   - OpenAI embeddings
   - PDF document processing
   - Token-aware chunking

## Duplication Analysis

### Duplicated Code Paths

The codebase contains two nearly identical implementations:

**Original (app/):**
- app/agent.py
- app/agent_graph_with_helpfulness.py
- app/tools.py
- app/rag.py
- app/agent_executor.py
- app/__main__.py

**Refactored (a2a_service/):**
- a2a_service/core/agent.py
- a2a_service/core/graph.py
- a2a_service/core/tools.py
- a2a_service/core/rag_graph.py
- a2a_service/adapters/agent_executor.py
- a2a_service/__main__.py

**Key Differences:**
1. Import paths (app.* vs a2a_service.core.*)
2. Module organization (flat vs core/adapters separation)
3. Naming (agent_graph_with_helpfulness vs graph)
4. Package structure (legacy vs clean architecture)

**Recommendation:** Deprecate app/ directory after validating a2a_service/ implementation

## Environment Variables

### Required Configuration

1. **OPENAI_API_KEY** (Required)
   - OpenAI API authentication
   - Used by: Agent, RAG embeddings
   - Validation: Checked at startup

2. **TAVILY_API_KEY** (Optional)
   - Tavily web search API
   - Used by: TavilySearch tool
   - No validation at startup

### Optional Configuration

1. **TOOL_LLM_URL** (default: "https://api.openai.com/v1")
   - OpenAI API base URL
   - Used by: Agent model initialization

2. **TOOL_LLM_NAME** (default: "gpt-4o-mini")
   - OpenAI model name for agent
   - Used by: Agent model initialization

3. **OPENAI_CHAT_MODEL** (default: "gpt-4.1-nano")
   - Model for RAG generation
   - Used by: RAG graph builder

4. **RAG_DATA_DIR** (default: "data")
   - Directory containing PDF documents
   - Used by: RAG graph builder
   - Created by setup.py

## Summary

### Component Count

**Public API Components:**
- 4 main classes (Agent, ResponseFormat, GeneralAgentExecutor, AgentState)
- 3 public functions (build_graph, get_tool_belt, retrieve_information)
- 1 decorated tool (retrieve_information)

**Internal Components:**
- 7 internal functions (graph routing, RAG processing)
- 2 utility scripts (setup, check_env)

**Entry Points:**
- 2 main server entry points (__main__.py in app/ and a2a_service/)
- 2 client examples
- 2 utility scripts

### Key Strengths

1. **Protocol Independence:** Core logic separated from A2A protocol
2. **Tool Modularity:** Clean tool belt abstraction
3. **State Management:** Proper conversation state with checkpointing
4. **Helpfulness Loop:** Quality control with feedback mechanism
5. **Structured Responses:** Type-safe response format

### Technical Debt

1. **Code Duplication:** app/ and a2a_service/ contain identical logic
2. **Legacy Code:** app/ directory should be deprecated
3. **Error Handling:** Bare except clauses in RAG (Lines 58, 64, 132 in rag_graph.py)
4. **Validation:** _validate_request always returns False (needs implementation)

### Recommended Usage

**Primary Entry Point:**
```bash
python -m a2a_service --host localhost --port 10000
```

**Client Usage:**
```bash
python a2a_client_examples/test_client.py
```

**Setup:**
```bash
python setup.py
python check_env.py
```
