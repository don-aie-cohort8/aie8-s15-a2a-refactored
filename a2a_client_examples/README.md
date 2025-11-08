# A2A Client Examples

This directory contains examples of how to interact with A2A Protocol agents.

## Contents

### test_client.py

A complete Python client demonstrating:

- Agent discovery via AgentCard
- Single-turn message sending
- Multi-turn conversations with context management
- Streaming responses via Server-Sent Events (SSE)

### json-rpc/

Example JSON-RPC message formats used by the A2A Protocol:

- `single_message_request.json` - Basic message request
- `multi_turn_request.json` - Conversation with context tracking
- `streaming_request.json` - Request for streaming responses
- `sample_response.json` - Example response formats

## Running the Test Client

```bash
# Ensure the agent service is running first
cd a2a_client_examples
uv run python test_client.py
```

## Understanding the Examples

The A2A Protocol uses JSON-RPC 2.0 over HTTP/SSE for agent communication. Key concepts:

1. **AgentCard**: Service discovery document at `/.well-known/agent-card`
2. **Context Management**: Use `task_id` and `context_id` for multi-turn conversations
3. **Streaming**: Real-time responses via Server-Sent Events

See the `json-rpc/` directory for detailed message formats.