# Databricks Model Context Protocol (MCP) Documentation

**Source**: https://docs.databricks.com/aws/en/generative-ai/mcp/
**Domain**: docs.databricks.com
**Fetched**: 2025-08-26
**Type**: Technical Documentation - AI/ML Infrastructure

## Overview

Model Context Protocol (MCP) is an open source standard that connects AI agents to tools, resources, prompts, and other contextual information. It enables standardized interactions between AI agents and various tools, allowing for seamless integration and reusability across different agents and organizations within the Databricks ecosystem.

## Key Concepts

### What is MCP?
- **Open standard** for AI agent connectivity
- **Tool abstraction layer** that enables agents to interact with various resources
- **Interoperability framework** for sharing tools across different AI agents
- **Secure integration** mechanism leveraging Unity Catalog permissions

### Architecture Options

1. **Databricks-managed MCP servers** - Pre-configured connections to Databricks services
2. **Custom MCP servers** - User-defined servers hosted on Databricks Apps

### Core Features
- Standardized tool and resource sharing across agents
- Unity Catalog-based access control
- OAuth and PAT authentication support
- Integration with Databricks Vector Search, Genie Spaces, and UC Functions
- Support for both serverless compute and SQL compute

## Managed MCP Servers

### Available Managed Servers

#### 1. Vector Search Server
- **Purpose**: Query Vector Search indexes to find relevant documents and data
- **URL Pattern**: `https://<workspace-hostname>/api/2.0/mcp/vector-search/{catalog}/{schema}`
- **Use Case**: Semantic search, RAG applications, document retrieval

#### 2. Unity Catalog Functions Server
- **Purpose**: Run Unity Catalog functions like custom Python or SQL tools
- **URL Pattern**: `https://<workspace-hostname>/api/2.0/mcp/functions/{catalog}/{schema}`
- **Use Case**: Custom business logic, data transformations, ML model invocations

#### 3. Genie Space Server
- **Purpose**: Query Genie spaces to get insights from structured data tables
- **URL Pattern**: `https://<workspace-hostname>/api/2.0/mcp/genie/{genie_space_id}`
- **Use Case**: Natural language queries over structured data, analytics insights
- **Limitation**: Does not pass conversation history

### Authentication & Access
- Requires Databricks workspace authentication
- OAuth for local development
- Secured by default with Unity Catalog permissions
- Resource specification required during agent deployment

## Custom MCP Servers

### Definition
Custom MCP servers are specialized HTTP-compatible servers hosted as Databricks apps, enabling:
- Deployment of existing MCP servers
- Integration of third-party MCP tools
- Creation of custom tool ecosystems

### Requirements
- HTTP-compatible transport implementation
- Databricks app deployment framework compatibility
- Proper authentication configuration

### Deployment Process

```bash
# 1. Create the app
databricks apps create mcp-my-custom-server

# 2. Sync files to workspace
databricks sync . "/Users/$DATABRICKS_USERNAME/mcp-my-custom-server"

# 3. Deploy the app
databricks apps deploy mcp-my-custom-server
```

### Configuration Files

#### app.yaml
```yaml
# Specifies server launch command and configuration
command: ["python", "-m", "your_mcp_server"]
```

#### requirements.txt
```text
# Python dependencies for your custom MCP server
databricks-mcp
mcp
# Additional dependencies...
```

## Code Examples

### Connecting to Managed MCP Server (Python)

```python
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

# Initialize client
workspace_client = WorkspaceClient()
mcp_client = DatabricksMCPClient(workspace_client)

# Connect to Vector Search server
vector_search_url = "https://<workspace>/api/2.0/mcp/vector-search/catalog/schema"
async with mcp_client.connect(vector_search_url) as session:
    tools = await session.list_tools()
    # Use tools for search operations
```

### LangGraph Agent with MCP Tools

```python
from langgraph.graph import StateGraph
from databricks_mcp import DatabricksMCPClient

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Build agent graph
builder = StateGraph(AgentState)

# Add MCP tool discovery
async def discover_tools(state):
    async with mcp_client.connect(mcp_server_url) as session:
        tools = await session.list_tools()
        return {"tools": tools}

builder.add_node("discover_tools", discover_tools)
```

### Custom MCP Server Connection

```python
from mcp.client import ClientSession
from mcp.client.stdio import streamablehttp_client
from databricks_mcp import DatabricksOAuthClientProvider

# Connect with OAuth authentication
async with streamablehttp_client(
    f"{custom_mcp_server_url}", 
    auth=DatabricksOAuthClientProvider(workspace_client)
) as (read_stream, write_stream, _):
    async with ClientSession(read_stream, write_stream) as session:
        tools = await session.list_tools()
        # Execute tools
        result = await session.call_tool("tool_name", arguments={})
```

## Implementation Notes

### Best Practices
1. **Authentication**: Always use OAuth for production deployments
2. **Resource Management**: Explicitly log all required resources when deploying agents
3. **Error Handling**: Implement proper error handling for MCP connections
4. **Tool Discovery**: Use dynamic tool discovery for flexible agent capabilities
5. **Permissions**: Leverage Unity Catalog for fine-grained access control

### Common Patterns

#### Agent Deployment with Resources
```python
from mlflow.models import ModelConfig

# Define required MCP resources
resources = ModelConfig(
    resources=[
        {"type": "uc_function", "key": "catalog.schema.function"},
        {"type": "vector_search_index", "key": "catalog.schema.index"},
        {"type": "genie_space", "key": "space_id"}
    ]
)
```

#### Multi-Server Integration
```python
# Connect to multiple MCP servers
servers = {
    "vector_search": "https://.../mcp/vector-search/...",
    "functions": "https://.../mcp/functions/...",
    "genie": "https://.../mcp/genie/..."
}

async def use_multiple_servers():
    for server_type, url in servers.items():
        async with mcp_client.connect(url) as session:
            # Use server-specific tools
            pass
```

### Limitations & Considerations
- Beta feature with ongoing development
- Genie MCP server doesn't maintain conversation history
- Custom servers subject to Databricks Apps pricing
- Serverless compute required for certain operations
- MCP standard evolving - expect updates

## Getting Started Checklist

1. **Environment Setup**
   - [ ] Install Databricks CLI
   - [ ] Configure authentication (OAuth/PAT)
   - [ ] Install `databricks-mcp` Python library

2. **Managed MCP Servers**
   - [ ] Create required Unity Catalog resources
   - [ ] Set up Vector Search indexes if needed
   - [ ] Configure Genie spaces for analytics
   - [ ] Deploy UC functions for custom logic

3. **Custom MCP Servers**
   - [ ] Prepare server code with HTTP transport
   - [ ] Create `app.yaml` configuration
   - [ ] Deploy as Databricks App
   - [ ] Test connection and authentication

4. **Agent Development**
   - [ ] Implement tool discovery logic
   - [ ] Add error handling
   - [ ] Configure resource requirements
   - [ ] Deploy with MLflow

## Related Resources

### Internal Documentation
- Reference [CLAUDE.md](../CLAUDE.md) for project-specific MCP implementation guidelines
- Check [PRD.md](../PRD.md) for manufacturing use case requirements

## Version Notes
- Documentation fetched: 2025-08-26
- MCP Feature Status: Beta
- Compatibility: Databricks Runtime 14.0+
- Required Libraries: databricks-mcp, mcp, mlflow>=3.1