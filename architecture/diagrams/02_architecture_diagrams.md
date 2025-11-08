# Architecture Diagrams

## System Architecture

The system follows a layered architecture pattern with clear separation of concerns. There are two implementations present: a legacy `app/` package and a refactored `a2a_service/` package that demonstrates better architectural practices.

**Key Architectural Layers:**
- **Presentation Layer**: A2A protocol server (Starlette-based HTTP server)
- **Adapter Layer**: Protocol-specific adapters (GeneralAgentExecutor) that bridge protocol to core logic
- **Core Business Logic Layer**: Agent, Graph building, and orchestration
- **Tool/Service Layer**: External integrations (Tavily, ArXiv, RAG)
- **Data Access Layer**: RAG with vector store (Qdrant in-memory), PDF loaders

The refactored architecture (`a2a_service/`) demonstrates the Hexagonal/Ports and Adapters pattern, where:
- Core business logic is protocol-agnostic
- Adapters translate between protocols (A2A) and the core domain
- Dependencies point inward (adapters depend on core, not vice versa)

```mermaid
graph TB
    subgraph PRES["Presentation Layer"]
        A2A["A2A Starlette Server<br/>Port: 10000"]
        Client["A2A Test Client"]
    end

    subgraph ADAPT["Adapter Layer"]
        Executor["GeneralAgentExecutor<br/>AgentExecutor implementation"]
        RequestHandler["DefaultRequestHandler<br/>A2A Protocol Handler"]
    end

    subgraph CORE["Core Business Logic"]
        Agent["Agent<br/>Main orchestrator"]
        Graph["LangGraph Builder<br/>Stateful workflow engine"]
        ResponseFormat["ResponseFormat<br/>Pydantic models"]
    end

    subgraph TOOLS["Tool/Service Layer"]
        ToolBelt["Tool Belt Assembly"]
        Tavily["TavilySearch<br/>Web search"]
        ArxivTool["ArxivQueryRun<br/>Academic papers"]
        RAGTool["retrieve_information<br/>Document retrieval"]
    end

    subgraph DATA["Data Access Layer"]
        RAGGraph["RAG Graph<br/>2-step: retrieve+generate"]
        VectorStore["Qdrant Vector Store<br/>In-memory"]
        PDFLoader["PyMuPDF Loader<br/>Document ingestion"]
        Embeddings["OpenAI Embeddings<br/>text-embedding-3-small"]
    end

    subgraph EXT["External Services"]
        OpenAI["OpenAI API<br/>gpt-4o-mini"]
        TavilyAPI["Tavily API"]
        ArxivAPI["ArXiv API"]
        FileSystem["PDF Files<br/>data/"]
    end

    Client -->|HTTP/JSON-RPC| A2A
    A2A --> RequestHandler
    RequestHandler --> Executor
    Executor -->|execute/stream| Agent
    Agent --> Graph
    Graph --> ToolBelt

    ToolBelt --> Tavily
    ToolBelt --> ArxivTool
    ToolBelt --> RAGTool

    RAGTool --> RAGGraph
    RAGGraph --> VectorStore
    RAGGraph -->|generate| OpenAI
    VectorStore -->|embeddings| Embeddings
    PDFLoader -->|chunks| VectorStore
    FileSystem --> PDFLoader

    Tavily --> TavilyAPI
    ArxivTool --> ArxivAPI
    Graph -->|LLM calls| OpenAI
    Embeddings --> OpenAI

    style PRES fill:#E0F7FA,stroke:#00BCD4,stroke-width:2px
    style ADAPT fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style CORE fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style TOOLS fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style DATA fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style EXT fill:#ECEFF1,stroke:#607D8B,stroke-width:2px

    style Agent fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style Graph fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style ResponseFormat fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style Executor fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style RequestHandler fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style A2A fill:#E0F7FA,stroke:#00BCD4,stroke-width:2px
    style Client fill:#E0F7FA,stroke:#00BCD4,stroke-width:2px
    style ToolBelt fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style Tavily fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style ArxivTool fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style RAGTool fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style RAGGraph fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style VectorStore fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style PDFLoader fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style Embeddings fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style OpenAI fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
    style TavilyAPI fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
    style ArxivAPI fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
    style FileSystem fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
```

## Component Relationships

The system is organized around a central Agent component that coordinates between the LangGraph state machine and various tools. The Agent to A2A protocol bridge is handled by the GeneralAgentExecutor adapter.

**Key Relationships:**
1. **Agent ↔ Graph**: Agent instantiates and manages the compiled LangGraph, streaming results
2. **Graph ↔ Tools**: Graph binds tools to the LLM and orchestrates tool calls via ToolNode
3. **Executor ↔ Agent**: Executor translates A2A protocol events to/from Agent stream
4. **Graph ↔ Helpfulness**: Post-response evaluation loop ensures quality before completion
5. **RAG Tool ↔ RAG Graph**: RAG tool encapsulates a separate 2-node LangGraph for retrieval+generation

```mermaid
graph LR
    subgraph ENTRY["Server Entry Point"]
        Main["__main__.py"]
    end

    subgraph PROTO["A2A Protocol Layer"]
        A2AApp["A2AStarletteApplication"]
        Handler["DefaultRequestHandler"]
        TaskStore["InMemoryTaskStore"]
        EventQueue["EventQueue"]
    end

    subgraph EXEC["Agent Execution"]
        Executor["GeneralAgentExecutor"]
        TaskUpdater["TaskUpdater"]
    end

    subgraph CORE_LOGIC["Core Agent Logic"]
        Agent["Agent"]
        StreamMethod["stream method"]
        GetResponse["get_agent_response"]
    end

    subgraph GRAPH_ORCH["Graph Orchestration"]
        CompiledGraph["Compiled LangGraph"]
        AgentNode["agent node"]
        ActionNode["action node"]
        HelpNode["helpfulness node"]
        Memory["MemorySaver checkpointer"]
    end

    subgraph TOOL_EXEC["Tool Execution"]
        ToolNode["ToolNode"]
        TavilyTool["Tavily Search"]
        ArxivTool["ArXiv Query"]
        RAGTool["RAG retrieve_information"]
    end

    subgraph MODEL["Model Interaction"]
        ChatOpenAI["ChatOpenAI<br/>with tools bound"]
        StructuredOutput["with_structured_output<br/>ResponseFormat"]
    end

    Main -->|initializes| A2AApp
    Main -->|creates| Handler
    Handler -->|uses| Executor
    Handler -->|manages| TaskStore
    Executor -->|creates| TaskUpdater
    Executor -->|calls| Agent

    Agent -->|owns| CompiledGraph
    Agent -->|invokes| StreamMethod
    StreamMethod -->|streams from| CompiledGraph
    StreamMethod -->|yields| GetResponse

    CompiledGraph -->|contains| AgentNode
    CompiledGraph -->|contains| ActionNode
    CompiledGraph -->|contains| HelpNode
    CompiledGraph -->|persists via| Memory

    AgentNode -->|invokes| ChatOpenAI
    AgentNode -->|structures via| StructuredOutput
    ActionNode -->|executes| ToolNode
    HelpNode -->|evaluates| ChatOpenAI

    ToolNode -->|dispatches to| TavilyTool
    ToolNode -->|dispatches to| ArxivTool
    ToolNode -->|dispatches to| RAGTool

    TaskUpdater -->|publishes to| EventQueue
    EventQueue -->|streams to| A2AApp

    style ENTRY fill:#EFEBE9,stroke:#795548,stroke-width:2px
    style PROTO fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style EXEC fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style CORE_LOGIC fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style GRAPH_ORCH fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style TOOL_EXEC fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style MODEL fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px

    style Main fill:#EFEBE9,stroke:#795548,stroke-width:2px
    style A2AApp fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Handler fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style TaskStore fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style EventQueue fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Executor fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style TaskUpdater fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Agent fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style StreamMethod fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style GetResponse fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style CompiledGraph fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style AgentNode fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style ActionNode fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style HelpNode fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style Memory fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style ToolNode fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style TavilyTool fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style ArxivTool fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style RAGTool fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style ChatOpenAI fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style StructuredOutput fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
```

## Class Hierarchies

The codebase demonstrates inheritance from the A2A SDK base classes, with custom implementations for the agent domain. The core Agent class is a standalone orchestrator without inheritance, while the Executor extends the A2A protocol's AgentExecutor base class.

**Key Class Structures:**
- **GeneralAgentExecutor** extends `AgentExecutor` from a2a-sdk to implement protocol-specific execution
- **Agent** is a standalone class that encapsulates ChatOpenAI model and compiled LangGraph
- **ResponseFormat** extends Pydantic's `BaseModel` for structured output validation
- **AgentState** is a TypedDict defining the shape of graph state with message accumulation

```mermaid
classDiagram
    class AgentExecutor {
        <<abstract>>
        +execute(context, event_queue)*
        +cancel(context, event_queue)*
    }

    class GeneralAgentExecutor {
        -Agent agent
        +execute(context, event_queue)
        +cancel(context, event_queue)
        -_validate_request(context)
    }

    class BaseModel {
        <<pydantic>>
    }

    class ResponseFormat {
        +str status
        +str message
    }

    class TypedDict {
        <<typing>>
    }

    class AgentState {
        +List~Message~ messages
        +Any structured_response
    }

    class Agent {
        +ChatOpenAI model
        +CompiledGraph graph
        +str SYSTEM_INSTRUCTION
        +str FORMAT_INSTRUCTION
        +List~str~ SUPPORTED_CONTENT_TYPES
        +stream(query, context_id) AsyncIterable
        +get_agent_response(config) dict
    }

    class StateGraph {
        <<langgraph>>
        +add_node(name, func)
        +add_edge(from, to)
        +add_conditional_edges(from, router, mapping)
        +set_entry_point(node)
        +compile(checkpointer) CompiledGraph
    }

    class CompiledGraph {
        <<langgraph>>
        +stream(inputs, config, stream_mode) Iterator
        +invoke(inputs, config) dict
        +get_state(config) StateSnapshot
    }

    class MemorySaver {
        <<langgraph.checkpoint>>
        +get(config) StateSnapshot
        +put(config, state) void
    }

    class ToolNode {
        <<langgraph.prebuilt>>
        +List~Tool~ tools
        +__call__(state) dict
    }

    class BaseTool {
        <<langchain.core.tools>>
        +str name
        +str description
        +_run(input)*
    }

    class TavilySearch {
        +int max_results
        +_run(query) str
    }

    class ArxivQueryRun {
        +_run(query) str
    }

    class CustomTool {
        +retrieve_information(query)
    }

    AgentExecutor <|-- GeneralAgentExecutor : implements
    BaseModel <|-- ResponseFormat : extends
    TypedDict <|-- AgentState : extends

    GeneralAgentExecutor *-- Agent : contains
    Agent *-- CompiledGraph : owns
    Agent *-- ChatOpenAI : configures

    StateGraph ..> CompiledGraph : compiles to
    StateGraph ..> MemorySaver : uses for persistence
    StateGraph *-- ToolNode : contains node

    ToolNode o-- TavilySearch : executes
    ToolNode o-- ArxivQueryRun : executes
    ToolNode o-- CustomTool : executes

    BaseTool <|-- TavilySearch : implements
    BaseTool <|-- ArxivQueryRun : implements

    CompiledGraph ..> AgentState : operates on
    Agent ..> ResponseFormat : produces

    %% Style definitions
    classDef coreClass fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    classDef adapterClass fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    classDef toolClass fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    classDef dataClass fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    classDef externalClass fill:#ECEFF1,stroke:#607D8B,stroke-width:2px

    %% Attach styles
    class Agent:::coreClass
    class CompiledGraph:::coreClass
    class StateGraph:::coreClass
    class MemorySaver:::coreClass
    class ResponseFormat:::coreClass
    class AgentState:::coreClass
    class GeneralAgentExecutor:::adapterClass
    class AgentExecutor:::adapterClass
    class ToolNode:::toolClass
    class BaseTool:::toolClass
    class TavilySearch:::toolClass
    class ArxivQueryRun:::toolClass
    class CustomTool:::toolClass
    class BaseModel:::externalClass
    class TypedDict:::externalClass
```

## Module Dependencies

The module dependency graph shows clear layering with minimal circular dependencies. The refactored `a2a_service/` package demonstrates better separation between core logic and adapters compared to the legacy `app/` package.

**Import Patterns:**
- `__main__.py` imports from both `core/` and `adapters/`, orchestrating server startup
- `adapters/agent_executor` imports from `core/agent` (adapter → core, correct direction)
- `core/agent` imports from `core/graph` (within core layer)
- `core/graph` imports from `core/tools` (within core layer)
- `core/tools` imports from `core/rag_graph` (within core layer)
- External SDK imports: `a2a`, `langgraph`, `langchain_*`, `openai`

```mermaid
graph TB
    subgraph REFACTORED["a2a_service Package"]
        subgraph ENTRY_PTS["Entry Points"]
            MainService["a2a_service/__main__.py"]
        end

        subgraph ADAPTERS["adapters/ - Protocol Layer"]
            AdapterInit["adapters/__init__.py"]
            AgentExecutor["adapters/agent_executor.py"]
        end

        subgraph CORE_MODS["core/ - Business Logic"]
            CoreInit["core/__init__.py"]
            CoreAgent["core/agent.py"]
            CoreGraph["core/graph.py"]
            CoreTools["core/tools.py"]
            CoreRAG["core/rag_graph.py"]
        end
    end

    subgraph LEGACY["app Package (Legacy)"]
        MainApp["app/__main__.py"]
        AppAgent["app/agent.py"]
        AppExecutor["app/agent_executor.py"]
        AppGraph["app/agent_graph_with_helpfulness.py"]
        AppTools["app/tools.py"]
        AppRAG["app/rag.py"]
        TestClient["app/test_client.py"]
    end

    subgraph CLIENTS["Client Examples"]
        ClientExample["a2a_client_examples/test_client.py"]
    end

    subgraph UTILS["Utilities"]
        Setup["setup.py"]
        CheckEnv["check_env.py"]
    end

    subgraph EXT_DEPS["External Dependencies"]
        A2ASDK["a2a-sdk"]
        LangGraph["langgraph"]
        LangChain["langchain-*"]
        OpenAI["langchain-openai"]
        Tavily["langchain-tavily"]
        Community["langchain-community"]
        Qdrant["qdrant-client"]
        Pydantic["pydantic"]
        Starlette["starlette"]
    end

    MainService --> AgentExecutor
    MainService --> CoreAgent
    MainService --> A2ASDK
    MainService --> Starlette

    AgentExecutor --> CoreAgent
    AgentExecutor --> A2ASDK

    CoreAgent --> CoreGraph
    CoreAgent --> OpenAI
    CoreAgent --> LangGraph
    CoreAgent --> Pydantic

    CoreGraph --> CoreTools
    CoreGraph --> LangGraph
    CoreGraph --> LangChain

    CoreTools --> CoreRAG
    CoreTools --> Tavily
    CoreTools --> Community

    CoreRAG --> OpenAI
    CoreRAG --> Qdrant
    CoreRAG --> LangGraph
    CoreRAG --> LangChain
    CoreRAG --> Community

    MainApp --> AppExecutor
    MainApp --> AppAgent
    MainApp --> A2ASDK
    MainApp --> Starlette

    AppExecutor --> AppAgent
    AppExecutor --> A2ASDK

    AppAgent --> AppGraph
    AppAgent --> OpenAI
    AppAgent --> LangGraph

    AppGraph --> AppTools
    AppGraph --> LangGraph

    AppTools --> AppRAG
    AppTools --> Tavily
    AppTools --> Community

    AppRAG --> OpenAI
    AppRAG --> Qdrant
    AppRAG --> LangGraph

    TestClient --> A2ASDK
    ClientExample --> A2ASDK

    Setup -.->|configures| MainApp
    Setup -.->|configures| MainService
    CheckEnv -.->|validates| MainApp
    CheckEnv -.->|validates| MainService

    style REFACTORED fill:#E8F5E9,stroke:#4CAF50,stroke-width:3px
    style LEGACY fill:#FFEBEE,stroke:#F44336,stroke-width:3px
    style CLIENTS fill:#E0F7FA,stroke:#00BCD4,stroke-width:2px
    style UTILS fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style EXT_DEPS fill:#ECEFF1,stroke:#607D8B,stroke-width:2px

    style ENTRY_PTS fill:#EFEBE9,stroke:#795548,stroke-width:2px
    style ADAPTERS fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style CORE_MODS fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px

    style CoreAgent fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style CoreGraph fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style CoreTools fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style CoreRAG fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style AgentExecutor fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style MainService fill:#EFEBE9,stroke:#795548,stroke-width:2px

    style AppAgent fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style AppGraph fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style AppExecutor fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style AppTools fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style AppRAG fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style MainApp fill:#FFEBEE,stroke:#F44336,stroke-width:2px
```

## LangGraph State Machine Flow

The agent uses a sophisticated LangGraph state machine with a helpfulness evaluation loop. This ensures responses meet quality standards before returning to the user.

**Graph Structure:**
1. **Entry Point**: "agent" node receives user messages
2. **Agent Node**: Invokes LLM with bound tools, produces AIMessage
3. **Routing Decision**: If tool calls needed → "action", else → "helpfulness"
4. **Action Node**: Executes tools via ToolNode, returns ToolMessages
5. **Helpfulness Node**: Evaluates response quality (Y/N decision)
6. **Helpfulness Decision**: If Y → END, if N → back to "agent" (max 10 iterations)

```mermaid
stateDiagram-v2
    [*] --> agent: User query arrives

    agent --> route_decision: AIMessage produced

    route_decision --> action: tool_calls present
    route_decision --> helpfulness: no tool_calls

    action --> agent: ToolMessage returned

    helpfulness --> helpfulness_eval: Evaluate response quality

    helpfulness_eval --> agent: HELPFULNESS N (continue improving)
    helpfulness_eval --> structured_response: HELPFULNESS Y (response is helpful)
    helpfulness_eval --> [*]: HELPFULNESS END (max iterations exceeded)

    structured_response --> [*]: Return to user

    state agent {
        [*] --> model_with_tools
        model_with_tools --> check_tool_calls
        check_tool_calls --> structured_output: No tool calls
        check_tool_calls --> return_response: Has tool calls
        structured_output --> return_response
        return_response --> [*]
    }

    state action {
        [*] --> tool_node
        tool_node --> execute_tools
        execute_tools --> [*]: ToolMessage
    }

    state helpfulness {
        [*] --> check_message_count
        check_message_count --> evaluate_with_llm: Count <= 10
        check_message_count --> [*]: Count > 10 (END marker)
        evaluate_with_llm --> create_decision
        create_decision --> [*]: HELPFULNESS Y/N
    }

    note right of helpfulness
        Evaluates if response is:
        - Accurate and relevant
        - Complete and addresses need
        - Uses appropriate tools
    end note

    note right of agent
        Uses ChatOpenAI with tools:
        - TavilySearch (web)
        - ArxivQueryRun (papers)
        - retrieve_information (RAG)
    end note
```

## RAG Architecture Detail

The RAG (Retrieval-Augmented Generation) system is implemented as a separate mini-LangGraph exposed as a tool. It demonstrates a clean two-step pattern: retrieve → generate.

**RAG Components:**
1. **Document Loading**: PyMuPDFLoader recursively loads PDFs from `RAG_DATA_DIR`
2. **Text Splitting**: RecursiveCharacterTextSplitter with tiktoken-based length (750 tokens)
3. **Embedding**: OpenAI's text-embedding-3-small model
4. **Vector Store**: Qdrant in-memory (location=":memory:")
5. **Retrieval**: Similarity search via Qdrant retriever
6. **Generation**: ChatOpenAI with constrained context prompt

```mermaid
graph TB
    subgraph TOOL_IFACE["RAG Tool Interface"]
        ToolDef["@tool decorator<br/>retrieve_information"]
        ToolInput["query: str"]
        ToolOutput["response: str"]
    end

    subgraph RAG_PIPELINE["RAG Graph - 2 Step Pipeline"]
        RagStart["START"]
        RetrieveNode["retrieve node"]
        GenerateNode["generate node"]
        RagEnd["END"]
    end

    subgraph RAG_STATE["RAG State"]
        StateQuestion["question: str"]
        StateContext["context: List Document"]
        StateResponse["response: str"]
    end

    subgraph INIT["Initialization (Cached)"]
        LoadPDFs["DirectoryLoader<br/>glob: **/*.pdf"]
        SplitDocs["RecursiveCharacterTextSplitter<br/>chunk_size: 750<br/>tiktoken length"]
        CreateEmbeddings["OpenAIEmbeddings<br/>text-embedding-3-small"]
        BuildVectorStore["Qdrant.from_documents<br/>location: :memory:"]
        CreateRetriever["as_retriever"]
    end

    subgraph RETRIEVE["Retrieve Step"]
        RetrieverInvoke["retriever.invoke"]
        SimilaritySearch["Similarity search in Qdrant"]
        ReturnDocs["Retrieved documents → context"]
    end

    subgraph GENERATE["Generate Step"]
        GenPrompt["ChatPromptTemplate<br/>Context + Query"]
        GenLLM["ChatOpenAI<br/>gpt-4.1-nano"]
        GenParser["StrOutputParser"]
        GenOutput["Generated response"]
    end

    subgraph DATA_SRC["Data Source"]
        FileSystem["File System<br/>data/*.pdf"]
    end

    ToolInput --> RagStart
    RagStart --> RetrieveNode
    RetrieveNode --> GenerateNode
    GenerateNode --> RagEnd
    RagEnd --> ToolOutput

    StateQuestion -.->|input to| RetrieveNode
    RetrieveNode -.->|updates| StateContext
    StateContext -.->|input to| GenerateNode
    GenerateNode -.->|updates| StateResponse

    FileSystem -->|load| LoadPDFs
    LoadPDFs --> SplitDocs
    SplitDocs --> CreateEmbeddings
    CreateEmbeddings --> BuildVectorStore
    BuildVectorStore --> CreateRetriever

    CreateRetriever -.->|provides| RetrieverInvoke
    RetrieveNode --> RetrieverInvoke
    RetrieverInvoke --> SimilaritySearch
    SimilaritySearch --> ReturnDocs

    GenerateNode --> GenPrompt
    GenPrompt --> GenLLM
    GenLLM --> GenParser
    GenParser --> GenOutput

    ToolDef -.->|caches via @lru_cache| LoadPDFs

    style TOOL_IFACE fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style RAG_PIPELINE fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style RAG_STATE fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style INIT fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style RETRIEVE fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style GENERATE fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style DATA_SRC fill:#ECEFF1,stroke:#607D8B,stroke-width:2px

    style RetrieveNode fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style GenerateNode fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style BuildVectorStore fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style RagStart fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style RagEnd fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style LoadPDFs fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style SplitDocs fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style CreateEmbeddings fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style CreateRetriever fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style RetrieverInvoke fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style SimilaritySearch fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style ReturnDocs fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style GenPrompt fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style GenLLM fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style GenParser fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style GenOutput fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style ToolDef fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style ToolInput fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style ToolOutput fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style FileSystem fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
```

## A2A Protocol Integration

The A2A (Agent-to-Agent) protocol integration demonstrates the Adapter pattern, translating between the protocol's event-driven streaming model and the core Agent's streaming interface.

**Integration Points:**
1. **AgentCard**: Metadata describing agent capabilities and skills
2. **DefaultRequestHandler**: Handles HTTP requests, manages tasks and push notifications
3. **GeneralAgentExecutor**: Bridges A2A protocol to Agent.stream()
4. **TaskUpdater**: Publishes state transitions to EventQueue
5. **EventQueue**: Streams updates back to client via SSE (Server-Sent Events)

```mermaid
sequenceDiagram
    participant Client as A2A Client
    participant Server as A2AStarletteApp
    participant Handler as DefaultRequestHandler
    participant Executor as GeneralAgentExecutor
    participant Agent as Agent
    participant Graph as CompiledGraph
    participant Tools as Tools<br/>(Tavily, ArXiv, RAG)
    participant Queue as EventQueue

    Client->>Server: GET /.well-known/agent.json
    Server-->>Client: AgentCard<br/>(capabilities, skills)

    Client->>Server: POST /send_message<br/>(JSON-RPC)
    Server->>Handler: handle_request(message)
    Handler->>Executor: execute(context, event_queue)

    Executor->>Executor: validate_request(context)
    Executor->>Agent: stream(query, context_id)

    activate Agent
    Agent->>Graph: stream(inputs, config)
    activate Graph

    Graph->>Graph: agent node<br/>(invoke LLM)
    Graph-->>Agent: AIMessage<br/>with tool_calls
    Agent->>Executor: {is_task_complete: False,<br/>require_user_input: False}
    Executor->>Queue: enqueue TaskState.working
    Queue-->>Client: SSE: "Searching for information..."

    Graph->>Tools: action node<br/>(execute tools)
    activate Tools
    Tools-->>Graph: ToolMessage (results)
    deactivate Tools

    Graph->>Graph: agent node<br/>(process results)
    Graph-->>Agent: AIMessage (response)
    Agent->>Executor: {is_task_complete: False,<br/>require_user_input: False}
    Executor->>Queue: enqueue TaskState.working
    Queue-->>Client: SSE: "Processing the results..."

    Graph->>Graph: helpfulness node<br/>(evaluate)
    Graph-->>Agent: AIMessage "HELPFULNESS:Y"

    deactivate Graph

    Agent->>Agent: get_agent_response(config)
    Agent-->>Executor: {is_task_complete: True,<br/>content: "..."}
    deactivate Agent

    Executor->>Queue: add_artifact(TextPart)
    Executor->>Queue: complete()
    Queue-->>Client: SSE: Final result artifact
    Queue-->>Client: SSE: TaskState.completed

    Client->>Client: Display final result

    rect rgb(224, 242, 250)
        note over Client,Server: Presentation Layer
    end
    rect rgb(255, 243, 224)
        note over Handler,Executor: Adapter Layer
    end
    rect rgb(227, 242, 253)
        note over Agent,Graph: Core Business Logic
    end
    rect rgb(232, 245, 233)
        note over Tools: Tool/Service Layer
    end
```

## File Organization Comparison

The codebase contains two implementations side-by-side: the original `app/` package and the refactored `a2a_service/` package. This comparison highlights architectural improvements.

**app/ Package (Original - Flat Structure):**
```
app/
├── __init__.py          (minimal)
├── __main__.py          (server entry, A2A setup)
├── agent.py             (Agent class)
├── agent_executor.py    (GeneralAgentExecutor)
├── agent_graph_with_helpfulness.py (graph builder)
├── tools.py             (tool assembly)
├── rag.py               (RAG graph)
└── test_client.py       (test client)
```

**a2a_service/ Package (Refactored - Layered Structure):**
```
a2a_service/
├── __init__.py          (package exports)
├── __main__.py          (server entry)
├── adapters/            (Protocol Layer)
│   ├── __init__.py
│   └── agent_executor.py (A2A → Core bridge)
└── core/                (Business Logic Layer)
    ├── __init__.py
    ├── agent.py         (Agent orchestrator)
    ├── graph.py         (graph builder)
    ├── tools.py         (tool assembly)
    └── rag_graph.py     (RAG implementation)
```

```mermaid
graph LR
    subgraph LEGACY_STRUCT["app Package - Flat Organization"]
        direction TB
        AppMain["__main__.py<br/>Server + Protocol"]
        AppAgent["agent.py<br/>Agent Logic"]
        AppExec["agent_executor.py<br/>A2A Adapter"]
        AppGraph["agent_graph_with_helpfulness.py<br/>Graph Builder"]
        AppTools["tools.py<br/>Tool Belt"]
        AppRAG["rag.py<br/>RAG Graph"]

        AppMain -.->|imports| AppAgent
        AppMain -.->|imports| AppExec
        AppExec -.->|imports| AppAgent
        AppAgent -.->|imports| AppGraph
        AppGraph -.->|imports| AppTools
        AppTools -.->|imports| AppRAG
    end

    subgraph REFACTORED_STRUCT["a2a_service Package - Layered Organization"]
        direction TB

        subgraph ENTRY_LAYER["Entry"]
            SvcMain["__main__.py<br/>Server Setup"]
        end

        subgraph ADAPTER_LAYER["adapters/ - Protocol Concerns"]
            SvcExec["agent_executor.py<br/>A2A Protocol Bridge"]
        end

        subgraph CORE_LAYER["core/ - Business Logic"]
            SvcAgent["agent.py<br/>Agent Orchestrator"]
            SvcGraph["graph.py<br/>LangGraph Builder"]
            SvcTools["tools.py<br/>Tool Assembly"]
            SvcRAG["rag_graph.py<br/>RAG Pipeline"]
        end

        SvcMain -.->|imports| SvcExec
        SvcMain -.->|imports| SvcAgent
        SvcExec -.->|imports| SvcAgent
        SvcAgent -.->|imports| SvcGraph
        SvcGraph -.->|imports| SvcTools
        SvcTools -.->|imports| SvcRAG
    end

    style LEGACY_STRUCT fill:#FFEBEE,stroke:#F44336,stroke-width:3px
    style REFACTORED_STRUCT fill:#E8F5E9,stroke:#4CAF50,stroke-width:3px

    style ENTRY_LAYER fill:#EFEBE9,stroke:#795548,stroke-width:2px
    style ADAPTER_LAYER fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style CORE_LAYER fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px

    style AppMain fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style AppExec fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style AppAgent fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style AppGraph fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style AppTools fill:#FFEBEE,stroke:#F44336,stroke-width:2px
    style AppRAG fill:#FFEBEE,stroke:#F44336,stroke-width:2px

    style SvcMain fill:#EFEBE9,stroke:#795548,stroke-width:2px
    style SvcExec fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style SvcAgent fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style SvcGraph fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style SvcTools fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style SvcRAG fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px

    note1["Flat structure:<br/>All modules at same level<br/>Protocol + Logic mixed"]
    note1 -.-> AppMain

    note2["Layered structure:<br/>- adapters/ for protocols<br/>- core/ for business logic<br/>Clear dependency direction"]
    note2 -.-> SvcExec
```

## Configuration and Environment

The application uses environment-based configuration with validation utilities. Configuration is loaded via python-dotenv and validated before server startup.

**Configuration Files:**
- `.env` - Environment variables (API keys, model config, RAG settings)
- `.env.example` - Template for required configuration
- `setup.py` - Interactive setup wizard
- `check_env.py` - Configuration validation utility

```mermaid
graph TB
    subgraph CONFIG_MGMT["Configuration Management"]
        EnvFile[".env file"]
        EnvExample[".env.example<br/>Template"]
        Setup["setup.py<br/>Interactive wizard"]
        Check["check_env.py<br/>Validation"]
    end

    subgraph ENV_VARS["Environment Variables"]
        OAIKEY["OPENAI_API_KEY<br/>Required for LLM + embeddings"]
        TavilyKey["TAVILY_API_KEY<br/>Required for web search"]
        LLMName["TOOL_LLM_NAME<br/>Default: gpt-4o-mini"]
        LLMURL["TOOL_LLM_URL<br/>Default: api.openai.com"]
        ChatModel["OPENAI_CHAT_MODEL<br/>Default: gpt-4.1-nano"]
        DataDir["RAG_DATA_DIR<br/>Default: data/"]
    end

    subgraph APP_COMP["Application Components"]
        Main["__main__.py"]
        Agent["Agent.__init__"]
        RAG["RAG _build_rag_graph"]
        Tools["get_tool_belt"]
    end

    subgraph EXT_SERV["External Services"]
        OpenAI["OpenAI API"]
        Tavily["Tavily API"]
    end

    Setup -->|creates| EnvFile
    EnvExample -.->|template for| Setup
    Check -->|validates| EnvFile

    EnvFile --> OAIKEY
    EnvFile --> TavilyKey
    EnvFile --> LLMName
    EnvFile --> LLMURL
    EnvFile --> ChatModel
    EnvFile --> DataDir

    Main -->|checks| OAIKEY
    Agent -->|uses| OAIKEY
    Agent -->|uses| LLMName
    Agent -->|uses| LLMURL

    RAG -->|uses| OAIKEY
    RAG -->|uses| ChatModel
    RAG -->|uses| DataDir

    Tools -->|uses| TavilyKey

    Agent -->|authenticates| OpenAI
    RAG -->|authenticates| OpenAI
    Tools -->|authenticates| Tavily

    style CONFIG_MGMT fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style ENV_VARS fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style APP_COMP fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style EXT_SERV fill:#ECEFF1,stroke:#607D8B,stroke-width:2px

    style EnvFile fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style EnvExample fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Setup fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Check fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style OAIKEY fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style TavilyKey fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style LLMName fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style LLMURL fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style ChatModel fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style DataDir fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Main fill:#EFEBE9,stroke:#795548,stroke-width:2px
    style Agent fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style RAG fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style Tools fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style OpenAI fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
    style Tavily fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
```

## Summary

The architecture demonstrates:

1. **Hexagonal/Ports and Adapters Pattern** (in `a2a_service/`): Clean separation between core business logic and protocol adapters
2. **Stateful Workflow Orchestration**: LangGraph manages complex conversation flow with helpfulness evaluation
3. **Tool Abstraction**: Unified interface for multiple data sources (web, academic, documents)
4. **RAG as a Mini-Graph**: Encapsulated retrieval-generation pipeline exposed as a tool
5. **Protocol-Agnostic Core**: Business logic independent of A2A protocol (could swap to REST/gRPC)
6. **Event-Driven Streaming**: A2A's event queue enables real-time progress updates to clients
7. **Configuration as Code**: Environment-based config with validation utilities

The refactored `a2a_service/` package shows significant architectural improvements over the original `app/` package, with better testability, maintainability, and adherence to SOLID principles.
