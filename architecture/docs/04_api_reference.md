# API Reference

## Overview

This document provides comprehensive API reference documentation for the A2A Research Agent codebase. The system is built on LangGraph and implements the A2A (Agent-to-Agent) protocol, providing a general-purpose AI assistant with web search, academic paper search, and document retrieval capabilities.

The codebase is organized into two main packages:
- **app**: Legacy implementation with direct A2A integration
- **a2a_service**: Refactored implementation with clean separation of concerns (core logic + adapters)

---

## Table of Contents

1. [Core Classes](#core-classes)
2. [Public Functions](#public-functions)
3. [Configuration](#configuration)
4. [Usage Patterns](#usage-patterns)
5. [Best Practices](#best-practices)
6. [Complete Examples](#complete-examples)

---

## Core Classes

### Class: Agent (app package)
**File**: `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent.py` (lines 24-116)

**Description**: A general-purpose AI assistant agent with access to web search, academic papers (arXiv), and RAG (Retrieval-Augmented Generation) capabilities. The agent uses LangGraph with a helpfulness evaluation loop to ensure quality responses that comply with the A2A protocol.

**Responsibilities**:
- Manages the LangGraph agent graph with helpfulness evaluation
- Provides streaming response capabilities
- Handles response format structuring for A2A protocol
- Maintains conversation memory using MemorySaver

**Initialization**:
```python
from app.agent import Agent

# Initialize agent (uses environment variables for configuration)
agent = Agent()
```

**Class Attributes**:
- `SYSTEM_INSTRUCTION` (str): System prompt defining agent behavior
- `FORMAT_INSTRUCTION` (str): Instructions for response formatting
- `SUPPORTED_CONTENT_TYPES` (list): Supported content types ['text', 'text/plain']

**Methods**:

#### async stream()
**Signature**: `async stream(query: str, context_id: str) -> AsyncIterable[dict[str, Any]]`

**Description**: Streams agent responses for a given query, yielding intermediate states and final response.

**Parameters**:
- `query` (str): The user's question or request
- `context_id` (str): Unique identifier for the conversation context/thread

**Returns**: AsyncIterable yielding dictionaries with:
- `is_task_complete` (bool): Whether the task is finished
- `require_user_input` (bool): Whether user input is needed
- `content` (str): The response content or status message

**Example**:
```python
import asyncio
from app.agent import Agent

async def chat_example():
    agent = Agent()
    context_id = "unique-thread-123"

    async for response in agent.stream("What is quantum computing?", context_id):
        print(f"Complete: {response['is_task_complete']}")
        print(f"Needs Input: {response['require_user_input']}")
        print(f"Content: {response['content']}\n")

asyncio.run(chat_example())
```

#### get_agent_response()
**Signature**: `get_agent_response(config: dict) -> dict[str, Any]`

**Description**: Retrieves and formats the final agent response from the graph state.

**Parameters**:
- `config` (dict): LangGraph configuration with thread_id

**Returns**: Dictionary with response structure matching A2A protocol expectations

**Example**:
```python
config = {'configurable': {'thread_id': 'thread-123'}}
response = agent.get_agent_response(config)
print(response['content'])
```

---

### Class: Agent (a2a_service package)
**File**: `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/agent.py` (lines 25-114)

**Description**: Refactored version of the Agent class with cleaner separation of concerns. This is the preferred implementation for new integrations.

**Differences from app.Agent**:
- Uses modular graph builder from `a2a_service.core.graph`
- Cleaner dependency management
- Same interface and capabilities

**Initialization**:
```python
from a2a_service.core.agent import Agent

agent = Agent()
```

**Methods**: Same as app.Agent (see above)

---

### Class: ResponseFormat
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent.py` (lines 17-22)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/agent.py` (lines 18-23)

**Description**: Pydantic model defining the structured response format for agent outputs. Ensures consistent response formatting across the system.

**Fields**:
- `status` (Literal['input_required', 'completed', 'error']): Response status indicator
- `message` (str): The response message content

**Example**:
```python
from app.agent import ResponseFormat

# Example structured response
response = ResponseFormat(
    status='completed',
    message='The answer to your question is...'
)

print(response.status)  # 'completed'
print(response.message)  # 'The answer to your question is...'
```

---

### Class: GeneralAgentExecutor (app package)
**File**: `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent_executor.py` (lines 27-94)

**Description**: A2A Protocol adapter that bridges the Agent class to the A2A server framework. Handles request/response translation, task management, and event streaming.

**Responsibilities**:
- Implements A2A AgentExecutor interface
- Manages task lifecycle (creation, updates, completion)
- Translates agent streams to A2A events
- Error handling and validation

**Initialization**:
```python
from app.agent_executor import GeneralAgentExecutor

executor = GeneralAgentExecutor()
```

**Methods**:

#### async execute()
**Signature**: `async execute(context: RequestContext, event_queue: EventQueue) -> None`

**Description**: Executes an agent request and streams results through the A2A event queue.

**Parameters**:
- `context` (RequestContext): A2A request context containing user message and task info
- `event_queue` (EventQueue): Queue for streaming task updates and results

**Raises**:
- `ServerError`: On validation failure or execution errors

**Example**:
```python
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

async def handle_request(context: RequestContext):
    executor = GeneralAgentExecutor()
    event_queue = EventQueue()

    await executor.execute(context, event_queue)

    # Events are automatically pushed to the queue
    async for event in event_queue:
        print(event)
```

#### async cancel()
**Signature**: `async cancel(context: RequestContext, event_queue: EventQueue) -> None`

**Description**: Not currently supported. Raises UnsupportedOperationError.

---

### Class: GeneralAgentExecutor (a2a_service package)
**File**: `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/adapters/agent_executor.py` (lines 27-94)

**Description**: Identical to app.GeneralAgentExecutor but uses the refactored Agent from a2a_service.core.

---

### Class: AgentState
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent_graph_with_helpfulness.py` (lines 18-22)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/graph.py` (lines 19-23)

**Description**: TypedDict schema defining the state structure for the LangGraph agent. Uses LangGraph's add_messages reducer for message accumulation.

**Fields**:
- `messages` (Annotated[List, add_messages]): List of conversation messages with automatic merging
- `structured_response` (Any): Optional ResponseFormat object with final structured response

**Example**:
```python
from app.agent_graph_with_helpfulness import AgentState
from langchain_core.messages import HumanMessage

# Example state
state: AgentState = {
    'messages': [HumanMessage(content='Hello!')],
    'structured_response': None
}
```

---

## Public Functions

### Function: build_agent_graph_with_helpfulness()
**File**: `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent_graph_with_helpfulness.py` (line 99)

**Signature**: `build_agent_graph_with_helpfulness(model, system_instruction: str, format_instruction: str, checkpointer=None) -> CompiledGraph`

**Description**: Constructs a LangGraph workflow with an integrated helpfulness evaluation loop. The graph includes three nodes: agent (LLM with tools), action (tool execution), and helpfulness (response quality evaluation).

**Parameters**:
- `model`: ChatOpenAI or compatible LangChain model instance
- `system_instruction` (str): System prompt for the agent
- `format_instruction` (str): Instructions for response formatting
- `checkpointer` (optional): LangGraph checkpointer for state persistence (e.g., MemorySaver)

**Returns**: Compiled LangGraph instance ready for invocation

**Graph Flow**:
1. **agent** node: Invokes LLM with tools
2. Routes to **action** (if tool calls) or **helpfulness** (if response)
3. **action** node: Executes tools and loops back to agent
4. **helpfulness** node: Evaluates response quality
5. Terminates on helpful response or continues loop (max 10 iterations)

**Example**:
```python
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from app.agent_graph_with_helpfulness import build_agent_graph_with_helpfulness

# Initialize model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Build graph
graph = build_agent_graph_with_helpfulness(
    model=model,
    system_instruction="You are a helpful assistant.",
    format_instruction="Respond in a clear and concise manner.",
    checkpointer=MemorySaver()
)

# Use the graph
result = graph.invoke(
    {"messages": [("user", "What is Python?")]},
    config={"configurable": {"thread_id": "123"}}
)
print(result['messages'][-1].content)
```

---

### Function: build_graph()
**File**: `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/graph.py` (line 102)

**Signature**: `build_graph(model, system_instruction: str, format_instruction: str, checkpointer=None) -> CompiledGraph`

**Description**: Identical to build_agent_graph_with_helpfulness() but in the refactored a2a_service package. This is the preferred function for new implementations.

**Example**: See build_agent_graph_with_helpfulness() above.

---

### Function: get_tool_belt()
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/tools.py` (line 15)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/tools.py` (line 15)

**Signature**: `get_tool_belt() -> List[Tool]`

**Description**: Returns a list of tools available to the agent. This includes third-party tools (Tavily for web search, ArxivQueryRun for academic papers) and custom tools (RAG for document retrieval).

**Returns**: List containing:
1. TavilySearch (configured for max 5 results)
2. ArxivQueryRun
3. retrieve_information (RAG tool)

**Example**:
```python
from app.tools import get_tool_belt

tools = get_tool_belt()
for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}\n")

# Output:
# Tool: tavily_search_results_json
# Description: A search engine...
#
# Tool: arxiv
# Description: A wrapper around Arxiv.org...
#
# Tool: retrieve_information
# Description: Use Retrieval Augmented Generation...
```

---

### Function: retrieve_information()
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/rag.py` (lines 115-126)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/rag_graph.py` (lines 115-126)

**Signature**: `retrieve_information(query: str) -> str`

**Description**: LangChain tool that implements Retrieval-Augmented Generation (RAG) for querying loaded PDF documents. Uses OpenAI embeddings and Qdrant vector store with a two-step retrieve-then-generate workflow.

**Parameters**:
- `query` (str): Query to search against loaded documents

**Returns**: Generated response based on retrieved context, or "I don't know" if no relevant information found

**Tool Metadata**:
- **Name**: retrieve_information
- **Description**: "Use Retrieval Augmented Generation to retrieve information about student loan policies"

**Example**:
```python
from app.rag import retrieve_information

# Direct invocation
result = retrieve_information("What are the eligibility requirements?")
print(result)

# As part of agent workflow (automatic)
# Agent will call this tool when appropriate
```

**Note**: Requires PDF documents in the RAG_DATA_DIR (default: "data") and OPENAI_API_KEY for embeddings.

---

### Function: create_env_file()
**File**: `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/setup.py` (lines 9-69)

**Signature**: `create_env_file() -> None`

**Description**: Interactive setup wizard that creates a .env configuration file by prompting the user for necessary API keys and configuration values.

**Prompts For**:
1. OpenAI API key
2. OpenAI model name (default: gpt-4.1-mini)
3. Tavily API key for web search
4. Data directory for PDFs (default: data)

**Creates**:
- `.env` file with all configuration
- Data directory for RAG documents

**Example**:
```python
from setup import create_env_file

# Run interactive setup
create_env_file()

# Example interaction:
# 🚀 LangGraph Agent Setup
# ==================================================
#
# This script will help you set up your environment configuration.
#
# 1. OpenAI Configuration:
# Enter your OpenAI API key: sk-...
# Enter OpenAI model name (default: gpt-4.1-mini):
#
# 2. Web Search Configuration
# ...
```

---

### Internal Functions (Graph Utilities)

#### Function: call_model()
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent_graph_with_helpfulness.py` (line 30)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/graph.py` (line 33)

**Signature**: `call_model(state: Dict[str, Any], model) -> Dict[str, Any]`

**Description**: LangGraph node function that invokes the model with bound tools and returns the response.

---

#### Function: route_to_action_or_helpfulness()
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent_graph_with_helpfulness.py` (line 38)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/graph.py` (line 41)

**Signature**: `route_to_action_or_helpfulness(state: Dict[str, Any]) -> str`

**Description**: Conditional edge function that routes to "action" if the last message contains tool calls, otherwise routes to "helpfulness" evaluation.

---

#### Function: helpfulness_node()
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent_graph_with_helpfulness.py` (line 46)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/graph.py` (line 49)

**Signature**: `helpfulness_node(state: Dict[str, Any], model) -> Dict[str, Any]`

**Description**: Evaluates whether the agent's response is helpful by comparing initial query with final response. Uses an LLM-as-judge pattern.

**Evaluation Criteria**:
- Provides accurate and relevant information
- Complete and addresses user's specific need
- Uses appropriate tools when necessary

**Returns**: Message with "HELPFULNESS:Y" (helpful), "HELPFULNESS:N" (not helpful), or "HELPFULNESS:END" (loop limit reached)

---

#### Function: helpfulness_decision()
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent_graph_with_helpfulness.py` (line 86)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/graph.py` (line 87)

**Signature**: `helpfulness_decision(state: Dict[str, Any]) -> str`

**Description**: Conditional edge function that decides whether to end the graph, continue looping, or terminate due to loop limit.

---

#### Function: build_model_with_tools()
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/agent_graph_with_helpfulness.py` (line 24)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/graph.py` (line 26)

**Signature**: `build_model_with_tools(model) -> ChatModel`

**Description**: Binds the tool belt (web search, arxiv, RAG) to the provided model instance.

---

### RAG Internal Functions

#### Function: _build_rag_graph()
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/rag.py` (line 42)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/rag_graph.py` (line 42)

**Signature**: `_build_rag_graph(data_dir: str) -> CompiledGraph`

**Description**: Constructs a two-node RAG pipeline: retrieve (from vector store) then generate (using LLM).

**Parameters**:
- `data_dir` (str): Directory containing PDF files

**Process**:
1. Loads PDFs from data_dir recursively
2. Splits documents into 750-token chunks (no overlap)
3. Creates OpenAI embeddings (text-embedding-3-small)
4. Stores in in-memory Qdrant vector store
5. Builds retrieve -> generate pipeline

**Returns**: Compiled LangGraph with retrieve and generate nodes

---

#### Function: _get_rag_graph()
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/rag.py` (line 108)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/rag_graph.py` (line 108)

**Signature**: `_get_rag_graph() -> CompiledGraph`

**Description**: Returns cached RAG graph built from RAG_DATA_DIR. Uses @lru_cache to ensure graph is only built once.

---

#### Function: _tiktoken_len()
**Files**:
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/app/rag.py` (line 29)
- `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/a2a_service/core/rag_graph.py` (line 29)

**Signature**: `_tiktoken_len(text: str) -> int`

**Description**: Calculates token count using tiktoken for gpt-4o model. Used by text splitter for accurate chunk sizing.

---

## Configuration

### Environment Variables

The system requires several environment variables for proper operation. Use `setup.py` to interactively configure, or manually create a `.env` file.

#### Required Variables

##### OPENAI_API_KEY
**Type**: String
**Required**: Yes
**Description**: OpenAI API key for LLM and embeddings
**Example**: `OPENAI_API_KEY=sk-proj-...`
**Used By**:
- Agent class (LLM inference)
- RAG system (embeddings)

##### TAVILY_API_KEY
**Type**: String
**Required**: Yes (for web search)
**Description**: Tavily API key for web search functionality
**Example**: `TAVILY_API_KEY=tvly-...`
**Get Key From**: https://tavily.com
**Used By**: TavilySearch tool

#### Optional Variables

##### TOOL_LLM_NAME
**Type**: String
**Default**: `gpt-4o-mini`
**Description**: OpenAI model name for agent inference
**Example**: `TOOL_LLM_NAME=gpt-4.1-mini`
**Valid Values**: Any OpenAI chat model (gpt-4o-mini, gpt-4o, gpt-4-turbo, etc.)

##### TOOL_LLM_URL
**Type**: String
**Default**: `https://api.openai.com/v1`
**Description**: OpenAI API base URL (useful for compatible APIs)
**Example**: `TOOL_LLM_URL=https://api.openai.com/v1`

##### OPENAI_CHAT_MODEL
**Type**: String
**Default**: `gpt-4.1-nano`
**Description**: Model used for RAG generation step
**Example**: `OPENAI_CHAT_MODEL=gpt-4.1`

##### RAG_DATA_DIR
**Type**: String
**Default**: `data`
**Description**: Directory containing PDF files for RAG
**Example**: `RAG_DATA_DIR=data`
**Note**: Directory is created automatically by setup.py

### Configuration File Example

Complete `.env` file example:

```bash
# LangGraph Agent Configuration

# OpenAI Configuration (Required)
OPENAI_API_KEY=sk-proj-your-key-here
TOOL_LLM_URL=https://api.openai.com/v1
TOOL_LLM_NAME=gpt-4.1-mini

# Web Search Configuration (Required)
TAVILY_API_KEY=tvly-your-key-here

# RAG Configuration
RAG_DATA_DIR=data
OPENAI_CHAT_MODEL=gpt-4.1
```

### Server Configuration

When running the A2A server, additional configuration is available via command-line options:

```bash
# Start server with default settings (localhost:10000)
python -m app

# Custom host and port
python -m app --host 0.0.0.0 --port 8080

# Or using a2a_service
python -m a2a_service --host localhost --port 10000
```

**Server Options**:
- `--host`: Server bind address (default: localhost)
- `--port`: Server port (default: 10000)

---

## Usage Patterns

### Pattern 1: Basic Agent Query

Simple question-answer interaction with streaming responses.

```python
import asyncio
from app.agent import Agent

async def basic_query():
    """Basic agent query with streaming."""
    agent = Agent()
    context_id = "thread-001"

    query = "What is the latest research on transformers?"

    print(f"Query: {query}\n")

    async for response in agent.stream(query, context_id):
        if response['is_task_complete']:
            print(f"Final Answer: {response['content']}")
        else:
            print(f"Status: {response['content']}")

asyncio.run(basic_query())
```

**Output Example**:
```
Query: What is the latest research on transformers?

Status: Searching for information...
Status: Processing the results...
Final Answer: Recent research on transformers focuses on...
```

---

### Pattern 2: Multi-Turn Conversation

Maintaining context across multiple exchanges using the same context_id.

```python
import asyncio
from app.agent import Agent

async def multi_turn_conversation():
    """Multi-turn conversation with context preservation."""
    agent = Agent()
    context_id = "conversation-123"  # Same ID for entire conversation

    # First query
    print("=== Turn 1 ===")
    async for response in agent.stream(
        "Find papers about quantum computing",
        context_id
    ):
        if response['is_task_complete']:
            print(f"Answer: {response['content']}\n")

    # Follow-up query (uses same context)
    print("=== Turn 2 ===")
    async for response in agent.stream(
        "Can you summarize the main findings?",
        context_id
    ):
        if response['is_task_complete']:
            print(f"Answer: {response['content']}\n")

    # Another follow-up
    print("=== Turn 3 ===")
    async for response in agent.stream(
        "Who are the key researchers in this field?",
        context_id
    ):
        if response['is_task_complete']:
            print(f"Answer: {response['content']}")

asyncio.run(multi_turn_conversation())
```

**Key Points**:
- Use the same `context_id` for all related queries
- Agent automatically maintains conversation history
- Can reference previous exchanges ("them", "it", etc.)

---

### Pattern 3: RAG Document Search

Searching loaded PDF documents using the RAG tool.

```python
import asyncio
import os
from app.agent import Agent

async def rag_search_example():
    """Search through loaded PDF documents."""
    # Ensure documents are loaded
    data_dir = os.getenv('RAG_DATA_DIR', 'data')
    print(f"Searching documents in: {data_dir}\n")

    agent = Agent()
    context_id = "rag-session-001"

    # Query that triggers RAG
    query = "What are the eligibility requirements for student loans?"

    async for response in agent.stream(query, context_id):
        if response['is_task_complete']:
            print(f"Answer from documents: {response['content']}")
        elif not response['require_user_input']:
            print(f"Status: {response['content']}")

asyncio.run(rag_search_example())
```

**Prerequisites**:
1. Place PDF files in the data directory
2. Set RAG_DATA_DIR environment variable
3. Ensure OPENAI_API_KEY is configured

---

### Pattern 4: Running the A2A Server

Starting the agent as an A2A protocol server for agent-to-agent communication.

```python
# File: my_server.py
import os
import uvicorn
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
import httpx

from a2a_service.core.agent import Agent
from a2a_service.adapters.agent_executor import GeneralAgentExecutor

load_dotenv()

# Define agent capabilities
capabilities = AgentCapabilities(streaming=True, push_notifications=True)

# Define agent skills
skills = [
    AgentSkill(
        id='web_search',
        name='Web Search Tool',
        description='Search the web for current information',
        tags=['search', 'web'],
        examples=['What are the latest AI news?'],
    ),
    AgentSkill(
        id='arxiv_search',
        name='Academic Paper Search',
        description='Search arXiv for academic papers',
        tags=['research', 'papers'],
        examples=['Find papers on transformers'],
    ),
    AgentSkill(
        id='rag_search',
        name='Document Retrieval',
        description='Search loaded documents',
        tags=['documents', 'rag'],
        examples=['What do the policies say about loans?'],
    ),
]

# Create agent card
agent_card = AgentCard(
    name='Research Assistant',
    description='AI assistant with search and document capabilities',
    url='http://localhost:10000/',
    version='1.0.0',
    default_input_modes=Agent.SUPPORTED_CONTENT_TYPES,
    default_output_modes=Agent.SUPPORTED_CONTENT_TYPES,
    capabilities=capabilities,
    skills=skills,
)

# Setup request handler
httpx_client = httpx.AsyncClient()
push_config_store = InMemoryPushNotificationConfigStore()
push_sender = BasePushNotificationSender(
    httpx_client=httpx_client,
    config_store=push_config_store
)

request_handler = DefaultRequestHandler(
    agent_executor=GeneralAgentExecutor(),
    task_store=InMemoryTaskStore(),
    push_config_store=push_config_store,
    push_sender=push_sender
)

# Create server
server = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler
)

# Run server
if __name__ == '__main__':
    uvicorn.run(server.build(), host='localhost', port=10000)
```

**Run with**:
```bash
python my_server.py
```

**Or use built-in entry point**:
```bash
# Using app package
python -m app --host localhost --port 10000

# Using a2a_service package (recommended)
python -m a2a_service --host localhost --port 10000
```

---

### Pattern 5: A2A Client Usage

Connecting to the A2A server from a client application.

```python
import asyncio
from uuid import uuid4
import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import MessageSendParams, SendMessageRequest

async def client_example():
    """Example A2A client interaction."""
    base_url = 'http://localhost:10000'

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        # Resolve agent card
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url
        )
        agent_card = await resolver.get_agent_card()

        # Create client
        factory = ClientFactory(ClientConfig(httpx_client=httpx_client))
        client = factory.create(card=agent_card)

        # Send message
        payload = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': 'What is LangGraph?'}
                ],
                'message_id': uuid4().hex,
            },
        }

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload)
        )

        response = await client.send_message(request)
        print(response.model_dump(mode='json', exclude_none=True))

asyncio.run(client_example())
```

---

### Pattern 6: Custom Tool Integration

Adding custom tools to the agent's toolbelt.

```python
# File: custom_tools.py
from typing import List
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from a2a_service.core.rag_graph import retrieve_information

@tool
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def get_custom_tool_belt() -> List:
    """Return tool belt with custom tools."""
    return [
        TavilySearch(max_results=5),
        ArxivQueryRun(),
        retrieve_information,
        calculate_fibonacci,  # Custom tool
    ]

# Use in graph building
from a2a_service.core.graph import build_graph
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# Patch get_tool_belt
import a2a_service.core.tools as tools_module
tools_module.get_tool_belt = get_custom_tool_belt

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
graph = build_graph(
    model=model,
    system_instruction="You are a helpful assistant with calculation capabilities.",
    format_instruction="Provide clear responses.",
    checkpointer=MemorySaver()
)
```

---

### Pattern 7: Error Handling

Robust error handling for production deployments.

```python
import asyncio
import logging
from app.agent import Agent
from app.agent_executor import GeneralAgentExecutor
from a2a.utils.errors import ServerError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def robust_query(query: str, context_id: str):
    """Query with comprehensive error handling."""
    agent = Agent()

    try:
        logger.info(f"Processing query: {query}")

        response_received = False
        async for response in agent.stream(query, context_id):
            response_received = True

            if response.get('is_task_complete'):
                logger.info("Task completed successfully")
                return response['content']
            elif response.get('require_user_input'):
                logger.warning("User input required")
                return response['content']
            else:
                logger.info(f"Intermediate: {response['content']}")

        if not response_received:
            logger.error("No response received from agent")
            return "Error: No response from agent"

    except ServerError as e:
        logger.error(f"Server error: {e.error}")
        return f"Server error: {e.error.message}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return f"An unexpected error occurred: {str(e)}"

async def main():
    result = await robust_query(
        "What is machine learning?",
        "session-001"
    )
    print(f"Result: {result}")

asyncio.run(main())
```

---

## Best Practices

### 1. Context ID Management

**DO**:
```python
# Use consistent, meaningful context IDs
context_id = f"user-{user_id}-session-{session_id}"

# Generate unique IDs for isolated conversations
import uuid
context_id = str(uuid.uuid4())
```

**DON'T**:
```python
# Don't reuse context IDs across different users
context_id = "global-session"  # Bad - mixes user conversations

# Don't use predictable/guessable IDs for sensitive data
context_id = "123"  # Bad - predictable
```

---

### 2. Environment Configuration

**DO**:
```python
# Always use environment variables for secrets
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("OPENAI_API_KEY not set")
```

**DON'T**:
```python
# Never hardcode API keys
api_key = "sk-proj-..."  # Bad - security risk

# Don't commit .env files
# Add to .gitignore
```

---

### 3. Streaming Response Handling

**DO**:
```python
# Always consume the entire stream
async for response in agent.stream(query, context_id):
    if response['is_task_complete']:
        final_response = response['content']
        break
    # Handle intermediate states
```

**DON'T**:
```python
# Don't break early without handling state
async for response in agent.stream(query, context_id):
    print(response)
    break  # Bad - doesn't get final response
```

---

### 4. RAG Document Management

**DO**:
```python
# Organize documents by category
RAG_DATA_DIR/
  ├── policies/
  │   ├── policy1.pdf
  │   └── policy2.pdf
  └── research/
      ├── paper1.pdf
      └── paper2.pdf

# Validate documents are loaded
import os
data_dir = os.getenv('RAG_DATA_DIR', 'data')
if os.path.exists(data_dir):
    pdf_count = len([f for f in os.listdir(data_dir) if f.endswith('.pdf')])
    print(f"Loaded {pdf_count} PDFs")
```

**DON'T**:
```python
# Don't load extremely large documents without chunking strategy
# Already handled by _build_rag_graph, but be aware of limits

# Don't assume documents are present
# Always check and provide fallback behavior
```

---

### 5. Error Recovery

**DO**:
```python
# Implement retry logic for transient failures
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def resilient_query(agent, query, context_id):
    async for response in agent.stream(query, context_id):
        if response['is_task_complete']:
            return response['content']
    raise Exception("No response received")
```

**DON'T**:
```python
# Don't swallow errors silently
try:
    result = await agent.stream(query, context_id)
except:
    pass  # Bad - error hidden
```

---

### 6. Performance Optimization

**DO**:
```python
# Reuse agent instances
agent = Agent()  # Create once

# Use for multiple queries
for query in queries:
    response = await agent.stream(query, context_id)
```

**DON'T**:
```python
# Don't create new agent for each query
for query in queries:
    agent = Agent()  # Bad - expensive initialization
    response = await agent.stream(query, context_id)
```

---

### 7. Testing

**DO**:
```python
# Write tests with mocked LLM responses
import pytest
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_agent_stream():
    with patch('app.agent.ChatOpenAI') as mock_llm:
        mock_llm.return_value.invoke = AsyncMock(
            return_value="Test response"
        )

        agent = Agent()
        responses = []
        async for resp in agent.stream("test", "test-id"):
            responses.append(resp)

        assert len(responses) > 0
        assert responses[-1]['is_task_complete']
```

---

### 8. Logging

**DO**:
```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log important events
logger.info(f"Query received: {query[:100]}...")
logger.info(f"Task completed for context: {context_id}")
```

---

## Complete Examples

### Example 1: Production-Ready Server

Full production server with monitoring, error handling, and graceful shutdown.

```python
"""
Production-ready A2A server with monitoring and error handling.

Run with:
    python production_server.py --host 0.0.0.0 --port 10000
"""

import asyncio
import logging
import os
import signal
import sys
from typing import Optional

import click
import httpx
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from a2a_service.core.agent import Agent
from a2a_service.adapters.agent_executor import GeneralAgentExecutor

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ServerManager:
    """Manages server lifecycle and graceful shutdown."""

    def __init__(self):
        self.server: Optional[uvicorn.Server] = None
        self.should_exit = False

    def handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_exit = True
        if self.server:
            self.server.should_exit = True

    def validate_environment(self):
        """Validate required environment variables."""
        required_vars = ['OPENAI_API_KEY', 'TAVILY_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            logger.error("Run 'python setup.py' to configure")
            raise ValueError(f"Missing environment variables: {missing_vars}")

        logger.info("Environment validation passed")

    def create_agent_card(self, host: str, port: int) -> AgentCard:
        """Create agent card with capabilities and skills."""
        capabilities = AgentCapabilities(
            streaming=True,
            push_notifications=True
        )

        skills = [
            AgentSkill(
                id='web_search',
                name='Web Search',
                description='Search the web for current information',
                tags=['search', 'web', 'internet'],
                examples=['What are the latest AI developments?'],
            ),
            AgentSkill(
                id='arxiv_search',
                name='Academic Papers',
                description='Search arXiv for research papers',
                tags=['research', 'papers', 'academic'],
                examples=['Find papers on transformer models'],
            ),
            AgentSkill(
                id='rag_search',
                name='Document Search',
                description='Search loaded PDF documents',
                tags=['documents', 'rag', 'retrieval'],
                examples=['What do the policies say about eligibility?'],
            ),
        ]

        return AgentCard(
            name='Production Research Agent',
            description='Enterprise AI assistant with search and document capabilities',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            default_input_modes=Agent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=Agent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )

    def create_server(self, host: str, port: int) -> uvicorn.Server:
        """Create and configure the server."""
        agent_card = self.create_agent_card(host, port)

        # Initialize components
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )

        # Create request handler
        request_handler = DefaultRequestHandler(
            agent_executor=GeneralAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )

        # Create A2A application
        app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        # Configure uvicorn
        config = uvicorn.Config(
            app.build(),
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

        return uvicorn.Server(config)

    async def run(self, host: str, port: int):
        """Run the server with graceful shutdown."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        # Validate environment
        self.validate_environment()

        # Create and run server
        self.server = self.create_server(host, port)

        logger.info(f"Starting server on {host}:{port}")
        logger.info(f"Agent card available at http://{host}:{port}/.well-known/agent.json")

        try:
            await self.server.serve()
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
        finally:
            logger.info("Server shutdown complete")


@click.command()
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=10000, type=int, help='Server port')
def main(host: str, port: int):
    """Start the production A2A agent server."""
    manager = ServerManager()

    try:
        asyncio.run(manager.run(host, port))
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
```

**Run with**:
```bash
python production_server.py --host 0.0.0.0 --port 10000
```

---

### Example 2: Interactive CLI Client

Interactive command-line client for testing the agent.

```python
"""
Interactive CLI client for testing the A2A agent.

Run with:
    python interactive_client.py
"""

import asyncio
import logging
from uuid import uuid4
from typing import Optional

import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import MessageSendParams, SendMessageRequest

logging.basicConfig(level=logging.WARNING)


class InteractiveClient:
    """Interactive client for A2A agent."""

    def __init__(self, base_url: str = 'http://localhost:10000'):
        self.base_url = base_url
        self.client = None
        self.context_id: Optional[str] = None

    async def connect(self):
        """Connect to the agent server."""
        print(f"Connecting to {self.base_url}...")

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=self.base_url
            )

            try:
                agent_card = await resolver.get_agent_card()
                print(f"Connected to: {agent_card.name}")
                print(f"Description: {agent_card.description}")
                print(f"\nAvailable skills:")
                for skill in agent_card.skills or []:
                    print(f"  - {skill.name}: {skill.description}")
                print()

                factory = ClientFactory(ClientConfig(httpx_client=httpx_client))
                self.client = factory.create(card=agent_card)

                return True
            except Exception as e:
                print(f"Connection failed: {e}")
                return False

    async def send_message(self, text: str) -> Optional[str]:
        """Send a message and get response."""
        if not self.client:
            print("Not connected!")
            return None

        # Create new context for first message
        if not self.context_id:
            self.context_id = str(uuid4())

        payload = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': text}],
                'message_id': uuid4().hex,
                'context_id': self.context_id,
            },
        }

        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**payload)
        )

        try:
            print("Thinking...", end='', flush=True)
            response = await self.client.send_message(request)
            print("\r" + " " * 20 + "\r", end='', flush=True)

            # Extract response text
            if response.root.result.artifacts:
                artifact = response.root.result.artifacts[0]
                if artifact.parts:
                    return artifact.parts[0].root.text

            return "No response received"
        except Exception as e:
            print(f"\nError: {e}")
            return None

    async def start_session(self):
        """Start interactive session."""
        print("=" * 60)
        print("A2A Agent Interactive Client")
        print("=" * 60)
        print("Commands:")
        print("  /new    - Start new conversation")
        print("  /quit   - Exit")
        print("  /help   - Show this help")
        print("=" * 60)
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower()

                    if command == 'quit':
                        print("Goodbye!")
                        break
                    elif command == 'new':
                        self.context_id = None
                        print("Started new conversation\n")
                        continue
                    elif command == 'help':
                        print("\nCommands:")
                        print("  /new    - Start new conversation")
                        print("  /quit   - Exit")
                        print("  /help   - Show this help\n")
                        continue
                    else:
                        print(f"Unknown command: {command}\n")
                        continue

                # Send message
                response = await self.send_message(user_input)
                if response:
                    print(f"Agent: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


async def main():
    """Main entry point."""
    client = InteractiveClient()

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        # Store httpx_client for the session
        if await client.connect():
            await client.start_session()
        else:
            print("Failed to connect to agent server.")
            print("Make sure the server is running:")
            print("  python -m a2a_service")


if __name__ == '__main__':
    asyncio.run(main())
```

**Run with**:
```bash
# Start server first
python -m a2a_service

# In another terminal, run client
python interactive_client.py
```

**Example session**:
```
==============================================================
A2A Agent Interactive Client
==============================================================
Commands:
  /new    - Start new conversation
  /quit   - Exit
  /help   - Show this help
==============================================================

Connecting to http://localhost:10000...
Connected to: General Purpose Agent
Description: A helpful AI assistant with web search, academic paper search, and document retrieval capabilities

Available skills:
  - Web Search Tool: Search the web for current information
  - Academic Paper Search: Search for academic papers on arXiv
  - Document Retrieval: Search through loaded documents for specific information

You: What is LangGraph?
Agent: LangGraph is a framework for building stateful, multi-actor applications with LLMs...

You: Can you find recent papers about it?
Agent: I found several recent papers on arXiv about LangGraph...

You: /new
Started new conversation

You: /quit
Goodbye!
```

---

### Example 3: Batch Processing Pipeline

Processing multiple queries efficiently with progress tracking.

```python
"""
Batch query processing with progress tracking and error handling.

Usage:
    python batch_processor.py queries.txt results.json
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from app.agent import Agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a single query."""
    query: str
    response: str
    success: bool
    error: str = ""
    timestamp: str = ""
    context_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class BatchProcessor:
    """Process multiple queries in batch."""

    def __init__(self, max_concurrent: int = 3):
        self.agent = Agent()
        self.max_concurrent = max_concurrent
        self.results: List[QueryResult] = []

    async def process_query(
        self,
        query: str,
        context_id: str
    ) -> QueryResult:
        """Process a single query."""
        try:
            logger.info(f"Processing: {query[:50]}...")

            response_text = ""
            async for response in self.agent.stream(query, context_id):
                if response['is_task_complete']:
                    response_text = response['content']
                    break

            result = QueryResult(
                query=query,
                response=response_text,
                success=True,
                context_id=context_id
            )

            logger.info(f"Completed: {query[:50]}...")
            return result

        except Exception as e:
            logger.error(f"Failed: {query[:50]}... - {e}")
            return QueryResult(
                query=query,
                response="",
                success=False,
                error=str(e),
                context_id=context_id
            )

    async def process_batch(
        self,
        queries: List[str],
        use_shared_context: bool = False
    ) -> List[QueryResult]:
        """
        Process multiple queries.

        Args:
            queries: List of query strings
            use_shared_context: If True, all queries share context
        """
        logger.info(f"Processing {len(queries)} queries...")

        # Create tasks
        tasks = []
        shared_context = "batch-context" if use_shared_context else None

        for i, query in enumerate(queries):
            context_id = shared_context or f"batch-{i}"
            tasks.append(self.process_query(query, context_id))

        # Process with concurrency limit
        results = []
        for i in range(0, len(tasks), self.max_concurrent):
            batch = tasks[i:i + self.max_concurrent]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)

            # Progress update
            progress = (i + len(batch)) / len(tasks) * 100
            logger.info(f"Progress: {progress:.1f}% ({i + len(batch)}/{len(tasks)})")

        self.results = results
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        if not self.results:
            return {}

        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful

        return {
            'total': len(self.results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(self.results) * 100,
            'results': [asdict(r) for r in self.results]
        }

    def save_results(self, output_path: Path):
        """Save results to JSON file."""
        summary = self.get_summary()

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {output_path}")


async def main(input_file: str, output_file: str):
    """Main entry point."""
    # Read queries
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    with open(input_path, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(queries)} queries from {input_file}")

    # Process
    processor = BatchProcessor(max_concurrent=3)
    await processor.process_batch(queries, use_shared_context=False)

    # Save results
    output_path = Path(output_file)
    processor.save_results(output_path)

    # Print summary
    summary = processor.get_summary()
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total queries: {summary['total']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python batch_processor.py <input_file> <output_file>")
        print("\nExample:")
        print("  python batch_processor.py queries.txt results.json")
        sys.exit(1)

    asyncio.run(main(sys.argv[1], sys.argv[2]))
```

**Create queries.txt**:
```
What is artificial intelligence?
Explain quantum computing
Find papers about transformers
What are the latest developments in robotics?
```

**Run**:
```bash
python batch_processor.py queries.txt results.json
```

**Output (results.json)**:
```json
{
  "total": 4,
  "successful": 4,
  "failed": 0,
  "success_rate": 100.0,
  "results": [
    {
      "query": "What is artificial intelligence?",
      "response": "Artificial intelligence (AI) is...",
      "success": true,
      "error": "",
      "timestamp": "2025-11-08T12:34:56.789",
      "context_id": "batch-0"
    },
    ...
  ]
}
```

---

## Conclusion

This API reference provides comprehensive documentation for the A2A Research Agent codebase. For additional information:

- **Architecture Documentation**: See `01_component_inventory.md` for system architecture
- **Data Flow Documentation**: See `03_data_flows.md` for data flow diagrams
- **Source Code**: Browse `/home/donbr/don-aie-cohort8/aie8-s15-a2a-refactored/`

For support or questions, refer to the project repository or contact the development team.

---

**Document Information**:
- **Generated**: 2025-11-08
- **Version**: 1.0.0
- **Codebase**: A2A Research Agent (aie8-s15-a2a-refactored)
- **Python Version**: 3.13+
