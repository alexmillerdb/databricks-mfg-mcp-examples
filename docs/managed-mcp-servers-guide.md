# Managed MCP Servers - Local IDE Agent Building Guide

**Source**: https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp#local-ide-build-an-agent-with-managed-mcp-servers
**Domain**: docs.databricks.com
**Fetched**: 2025-08-26
**Type**: Implementation Guide

## Overview

Model Context Protocol (MCP) servers are bridges that enable AI agents to access external data and tools on Databricks. Managed MCP servers provide pre-configured connections to Databricks services without requiring manual setup, making it easy to build powerful agents with minimal configuration.

## Key Concepts

### Available Managed MCP Servers

1. **Vector Search Server**
   - Purpose: Query Vector Search indexes for similarity searches
   - URL Pattern: `https://<workspace-hostname>/api/2.0/mcp/vector-search/{catalog}/{schema}`
   - Use Case: Semantic search, document retrieval, RAG applications

2. **Unity Catalog Functions Server**
   - Purpose: Execute custom Python or SQL functions as tools
   - URL Pattern: `https://<workspace-hostname>/api/2.0/mcp/functions/{catalog}/{schema}`
   - Use Case: Custom business logic, data transformations, ML model inference

3. **Genie Space Server**
   - Purpose: Query structured data tables using natural language
   - URL Pattern: `https://<workspace-hostname>/api/2.0/mcp/genie/{genie_space_id}`
   - Use Case: Business intelligence queries, KPI retrieval, data exploration

### Beta Feature Notice
This feature is currently in Beta and subject to changes. Production deployments should account for potential API updates.

## Prerequisites

### Environment Setup
- Python 3.12 or above
- Databricks workspace with serverless compute enabled
- OAuth authentication configured

### Required Packages
```bash
pip install -U "mcp>=1.9" \
              "databricks-sdk[openai]" \
              "mlflow>=3.1.0" \
              "databricks-agents>=1.0.0" \
              "databricks-mcp"
```

### Authentication
```bash
# Authenticate via OAuth
databricks auth login --host https://<your-workspace-hostname>
```

## Code Examples

### Basic MCP Client Setup
```python
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

# Initialize workspace client
w = WorkspaceClient()

# Create MCP client
mcp_client = DatabricksMCPClient(workspace_client=w)

# Connect to a managed server
vector_search_url = "https://<workspace>/api/2.0/mcp/vector-search/catalog/schema"
await mcp_client.connect_server_session(vector_search_url)

# List available tools
tools = await mcp_client.list_tools()
for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"  Description: {tool.description}")
```

### Building an Agent with Multiple MCP Servers
```python
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient
import mlflow

class MultiMCPAgent:
    def __init__(self):
        self.w = WorkspaceClient()
        self.mcp_client = DatabricksMCPClient(workspace_client=self.w)
        self.servers = []
    
    async def connect_servers(self, server_urls):
        """Connect to multiple MCP servers"""
        for url in server_urls:
            await self.mcp_client.connect_server_session(url)
            self.servers.append(url)
    
    async def discover_tools(self):
        """Discover all available tools from connected servers"""
        all_tools = {}
        for server in self.servers:
            tools = await self.mcp_client.list_tools(server=server)
            for tool in tools:
                all_tools[tool.name] = {
                    'description': tool.description,
                    'server': server,
                    'schema': tool.input_schema
                }
        return all_tools
    
    async def call_tool(self, tool_name, arguments):
        """Execute a tool call"""
        result = await self.mcp_client.call_tool(
            name=tool_name,
            arguments=arguments
        )
        return result

# Usage example
agent = MultiMCPAgent()

# Connect to multiple servers
server_urls = [
    "https://workspace/api/2.0/mcp/vector-search/catalog/schema",
    "https://workspace/api/2.0/mcp/functions/catalog/schema",
    "https://workspace/api/2.0/mcp/genie/space_id"
]

await agent.connect_servers(server_urls)
tools = await agent.discover_tools()
```

### Deploying an MCP Agent with MLflow
```python
import mlflow
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

class MCPAgent:
    def __init__(self):
        self.w = WorkspaceClient()
        self.mcp_client = DatabricksMCPClient(workspace_client=self.w)
    
    def predict(self, model_input):
        """Agent prediction method for MLflow deployment"""
        messages = model_input.get("messages", [])
        
        # Process messages and determine tool calls
        # ... agent logic here ...
        
        return {"response": "Agent response"}

# Log and register the agent
with mlflow.start_run():
    # Get required Databricks resources
    resources = DatabricksMCPClient().get_databricks_resources()
    
    # Log the agent
    logged_agent = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=MCPAgent(),
        pip_requirements=[
            "databricks-mcp>=1.0.0",
            "databricks-sdk>=0.20.0",
            "mlflow>=3.1.0"
        ],
        resources=resources,
        registered_model_name="mcp_agent"
    )
```

### Single-Turn vs Multi-Turn Agents

#### Single-Turn Agent
```python
async def single_turn_agent(query):
    """Process a single query and return response"""
    # Connect to MCP server
    await mcp_client.connect_server_session(server_url)
    
    # Analyze query and determine tool
    tool_name = determine_tool(query)
    
    # Call tool
    result = await mcp_client.call_tool(
        name=tool_name,
        arguments={"query": query}
    )
    
    # Generate response
    return generate_response(result)
```

#### Multi-Turn Agent with State
```python
class MultiTurnAgent:
    def __init__(self):
        self.conversation_history = []
        self.mcp_client = DatabricksMCPClient()
    
    async def process_turn(self, user_input):
        """Process a conversation turn"""
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Determine if tool call is needed
        if needs_tool_call(user_input, self.conversation_history):
            tool_result = await self.call_appropriate_tool(user_input)
            response = self.generate_response(tool_result)
        else:
            response = self.generate_response(user_input)
        
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
```

## Implementation Notes

### Resource Specification
When deploying agents, explicitly specify all required resources:
```python
resources = [
    {
        "type": "databricks_vector_search_index",
        "name": "catalog.schema.index_name"
    },
    {
        "type": "databricks_uc_function",
        "name": "catalog.schema.function_name"
    },
    {
        "type": "databricks_genie_space",
        "name": "space_id"
    }
]
```

### Error Handling
```python
try:
    result = await mcp_client.call_tool(name=tool_name, arguments=args)
except Exception as e:
    # Handle connection errors
    if "connection" in str(e).lower():
        await mcp_client.reconnect()
        result = await mcp_client.call_tool(name=tool_name, arguments=args)
    else:
        raise e
```

### Performance Optimization
- Cache tool discoveries to avoid repeated API calls
- Use connection pooling for multiple server connections
- Implement retry logic with exponential backoff
- Batch tool calls when possible

### Security Best Practices
- Always use OAuth authentication in production
- Implement proper access controls via Unity Catalog
- Audit tool usage through MLflow tracking
- Use on-behalf-of-user (OBO) authentication for multi-tenant scenarios

## Related Resources
- Reference [CLAUDE.md](../CLAUDE.md) for project-specific MCP implementation guidelines
- Check [PRD.md](../PRD.md) for manufacturing use case requirements
- See [langgraph-mcp-agent.md](./langgraph-mcp-agent.md) for LangGraph integration examples
- Review Databricks Agent Framework documentation for deployment options