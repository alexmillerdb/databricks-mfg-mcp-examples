# Databricks Vector Search Client Guide

This guide provides comprehensive documentation for using Databricks Vector Search with LangChain to build intelligent retrieval-augmented generation (RAG) applications.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Configuration Options](#configuration-options)
5. [Index Types](#index-types)
6. [Query Types](#query-types)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)

## Overview

Databricks Vector Search enables you to store and query vector embeddings at scale, making it ideal for building RAG applications, semantic search, and recommendation systems. The `VectorSearchRetrieverTool` from `databricks-langchain` provides a seamless integration with LangChain for building AI applications.

### Key Features

- **Scalable vector storage**: Handle millions of vectors efficiently
- **Hybrid search**: Combine semantic and keyword search capabilities
- **Real-time updates**: Automatically sync with Delta tables
- **Security**: Enterprise-grade security and governance
- **Integration**: Native LangChain and LLM integration

## Prerequisites

Before using Databricks Vector Search, ensure you have:

1. **Databricks Workspace**: With Vector Search enabled
2. **Authentication**: Valid workspace token or service principal
3. **Vector Search Index**: Created and populated with your data
4. **Python Packages**:
   ```bash
   pip install databricks-langchain langchain
   ```

### Environment Setup

Set up your environment variables:

```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-access-token"
```

## Quick Start

```python
from databricks_langchain import VectorSearchRetrieverTool

# Basic usage for Delta Sync indexes
vs_tool = VectorSearchRetrieverTool(
    index_name="catalog.schema.my_index",
    tool_name="my_retriever",
    tool_description="Retrieves relevant documents"
)

# Query the index
results = vs_tool.invoke("your search query")
```

## Configuration Options

### Basic Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `index_name` | str | Full index name (catalog.schema.index) | Yes |
| `tool_name` | str | Name for the tool (used by LLM) | Yes |
| `tool_description` | str | Description of tool purpose | Yes |

### Advanced Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_results` | int | Maximum results to return | 10 |
| `columns` | List[str] | Columns to include in results | All |
| `filters` | Dict | Filter conditions for search | None |
| `query_type` | str | "ANN" or "HYBRID" | "ANN" |
| `text_column` | str | Column containing text for embeddings | None |
| `embedding` | Embeddings | Custom embedding model | None |

## Index Types

### 1. Delta Sync Index

- **Use case**: Auto-sync with Delta tables
- **Embeddings**: Managed by Databricks
- **Configuration**: Minimal setup required

```python
vs_tool = VectorSearchRetrieverTool(
    index_name="catalog.schema.delta_sync_index",
    tool_name="delta_retriever",
    tool_description="Retrieves from auto-synced Delta table"
)
```

### 2. Direct Vector Access Index

- **Use case**: Direct vector uploads
- **Embeddings**: Self-managed
- **Configuration**: Requires embedding model

```python
from databricks_langchain import DatabricksEmbeddings

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

vs_tool = VectorSearchRetrieverTool(
    index_name="catalog.schema.direct_access_index",
    tool_name="direct_retriever",
    tool_description="Retrieves from direct access index",
    text_column="content",
    embedding=embedding_model
)
```

## Query Types

### Approximate Nearest Neighbor (ANN)

- **Purpose**: Pure semantic search
- **Performance**: Fastest
- **Use case**: When you want similarity-based retrieval

```python
vs_tool = VectorSearchRetrieverTool(
    index_name="catalog.schema.my_index",
    query_type="ANN",
    # ... other parameters
)
```

### Hybrid Search

- **Purpose**: Combines semantic and keyword search
- **Performance**: Slower but more comprehensive
- **Use case**: When you need both semantic understanding and exact keyword matches

```python
vs_tool = VectorSearchRetrieverTool(
    index_name="catalog.schema.my_index",
    query_type="HYBRID",
    # ... other parameters
)
```

## Best Practices

### Performance Optimization

1. **Limit Results**: Use appropriate `num_results` to balance quality and speed
2. **Column Selection**: Only retrieve necessary columns
3. **Query Type**: Use ANN for pure semantic search
4. **Filtering**: Apply filters to reduce search space
5. **Caching**: Implement caching for frequent queries

### Security Considerations

1. **Authentication**: Use service principals for production
2. **Access Control**: Implement proper table/index permissions
3. **Data Privacy**: Ensure sensitive data is properly handled
4. **Network Security**: Use private endpoints when available

## On-Behalf-Of (OBO) Authentication

For multi-tenant applications where vector search should respect individual user permissions, configure OBO authentication with proper resource declarations and API scopes.

### Resource Configuration

When logging agents that use vector search, declare all vector search indexes as resources:

```python
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
from mlflow.models.auth_policy import UserAuthPolicy, SystemAuthPolicy, AuthPolicy

# Declare vector search indexes as resources
resources = [
    DatabricksVectorSearchIndex(index_name="catalog.schema.docs_index"),
    DatabricksVectorSearchIndex(index_name="catalog.schema.products_index"),
    DatabricksServingEndpoint(endpoint_name="databricks-claude-3-7-sonnet"),  # LLM endpoint
    # Add other resources as needed
]

# System resources (agent identity)
system_auth_policy = SystemAuthPolicy(resources=resources)
```

### User Authentication Policy

Configure API scopes for OBO vector search access:

```python
# User authentication policy for OBO access
user_auth_policy = UserAuthPolicy(
    api_scopes=[
        "serving.serving-endpoints",           # For LLM endpoints
        "vectorsearch.vector-search-endpoints", # For vector search endpoints
        "vectorsearch.vector-search-indexes",   # For vector search indexes
        # Add other scopes as needed (e.g., "dashboards.genie")
    ]
)

# Combined authentication policy
auth_policy = AuthPolicy(
    system_auth_policy=system_auth_policy,
    user_auth_policy=user_auth_policy
)
```

### OBO Agent Implementation

Initialize vector search clients inside the predict method for proper OBO:

```python
from uuid import uuid4
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials
from databricks_langchain import VectorSearchRetrieverTool

class OBOVectorSearchAgent(ResponsesAgent):
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # Initialize OBO client inside predict method
        user_client = WorkspaceClient(
            credentials_strategy=ModelServingUserCredentials()
        )
        
        # Create vector search tool with user credentials
        vs_tool = VectorSearchRetrieverTool(
            index_name="catalog.schema.user_docs_index",
            tool_name="user_docs_retriever",
            tool_description="Retrieves user-accessible documents",
            workspace_client=user_client  # Enforces OBO access
        )
        
        # Process user query
        user_query = request.input[0].content
        search_results = vs_tool.invoke(user_query)
        
        # Generate response based on search results
        response_text = f"Found {len(search_results)} relevant documents: {search_results}"
        
        output_item = self.create_text_output_item(
            text=response_text,
            id=str(uuid4())
        )
        
        return ResponsesAgentResponse(output=[output_item])
```

### MLflow Logging with OBO

Log the agent with proper resource and authentication policies:

```python
import mlflow

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="obo_vector_search_agent",
        python_model="obo_agent.py",
        auth_policy=auth_policy,  # Includes both system and user policies
        pip_requirements=[
            "databricks-langchain",
            "databricks-ai-bridge",
            "mlflow>=3.1.0"
        ]
    )
```

### OBO Best Practices

1. **Resource Declaration**: Always declare vector search indexes in `SystemAuthPolicy`
2. **Scope Minimization**: Only request necessary API scopes in `UserAuthPolicy`
3. **Client Initialization**: Initialize user-authenticated clients inside `predict` method
4. **Permission Testing**: Validate that users can only access permitted indexes
5. **Error Handling**: Implement proper error handling for permission denied scenarios

## MCP Server Integration

Databricks provides managed MCP servers for vector search, enabling dynamic tool discovery and standardized integration across different agent frameworks.

### Managed Vector Search MCP Server

Connect to the managed vector search MCP server to automatically discover available vector search tools:

```python
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

# Initialize workspace client
workspace_client = WorkspaceClient()

# Construct MCP server URL for vector search
host = workspace_client.config.host
mcp_server_url = f"{host}/api/2.0/mcp/vector-search/<catalog>/<schema>"

# Connect to the managed MCP server
mcp_client = DatabricksMCPClient(
    server_url=mcp_server_url, 
    workspace_client=workspace_client
)

# Discover available vector search tools
tools = mcp_client.list_tools()
print("Vector search tools available:", [t.name for t in tools])
```

### MCP Tool Usage

Use discovered MCP tools in your agents:

```python
# Call a specific vector search tool via MCP
tool_name = "vector_search_query"  # Replace with actual tool name
search_result = mcp_client.call_tool(
    name=tool_name,
    arguments={
        "query": "What is machine learning?",
        "num_results": 5,
        "filters": {"category": "documentation"}
    }
)

print("MCP search results:", search_result)
```

### Integration with LangGraph Agents

Combine MCP vector search tools with LangGraph for dynamic agent workflows:

```python
from databricks_mcp import DatabricksMCPClient
from langchain_core.tools import Tool
import asyncio

class MCPVectorSearchTool:
    """Wrapper to convert MCP vector search tools to LangChain-compatible tools"""
    
    def __init__(self, mcp_client: DatabricksMCPClient, tool_name: str, tool_description: str):
        self.mcp_client = mcp_client
        self.tool_name = tool_name
        self.tool_description = tool_description
    
    async def _arun(self, query: str, **kwargs):
        """Async execution of MCP vector search tool"""
        result = await self.mcp_client.call_tool(
            name=self.tool_name,
            arguments={"query": query, **kwargs}
        )
        return result
    
    def _run(self, query: str, **kwargs):
        """Sync wrapper for async tool execution"""
        return asyncio.run(self._arun(query, **kwargs))
    
    def to_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        return Tool(
            name=self.tool_name,
            description=self.tool_description,
            func=self._run,
            coroutine=self._arun
        )

# Create LangChain-compatible tools from MCP
def create_mcp_vector_search_tools(catalog: str, schema: str):
    """Create vector search tools from MCP server"""
    workspace_client = WorkspaceClient()
    host = workspace_client.config.host
    mcp_server_url = f"{host}/api/2.0/mcp/vector-search/{catalog}/{schema}"
    
    mcp_client = DatabricksMCPClient(
        server_url=mcp_server_url,
        workspace_client=workspace_client
    )
    
    # Discover available tools
    mcp_tools = mcp_client.list_tools()
    langchain_tools = []
    
    for tool in mcp_tools:
        mcp_tool_wrapper = MCPVectorSearchTool(
            mcp_client=mcp_client,
            tool_name=tool.name,
            tool_description=tool.description
        )
        langchain_tools.append(mcp_tool_wrapper.to_langchain_tool())
    
    return langchain_tools
```

### MCP vs Direct Integration

Choose the appropriate integration method based on your use case:

| Approach | Use Case | Benefits | Considerations |
|----------|----------|----------|----------------|
| **Direct VectorSearchRetrieverTool** | Single index, simple queries | Direct control, minimal overhead | Less flexible, manual configuration |
| **Managed MCP Server** | Multiple indexes, dynamic discovery | Standardized, discoverable, multi-framework | Additional abstraction layer |
| **Custom MCP Server** | Complex logic, external systems | Full customization, specialized workflows | Requires custom implementation |

### MCP Best Practices

1. **Server URL Format**: Use the correct catalog/schema in the MCP server URL
2. **Tool Discovery**: Always call `list_tools()` to discover available tools dynamically
3. **Error Handling**: Implement proper error handling for MCP server connectivity
4. **Authentication**: MCP servers inherit workspace authentication automatically
5. **Tool Naming**: Use descriptive tool names for better LLM understanding

### Example: Complete MCP Integration

```python
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

def setup_mcp_vector_search(catalog: str, schema: str):
    """Complete example of MCP vector search setup"""
    try:
        # Initialize clients
        workspace_client = WorkspaceClient()
        host = workspace_client.config.host
        mcp_server_url = f"{host}/api/2.0/mcp/vector-search/{catalog}/{schema}"
        
        print(f"Connecting to MCP server: {mcp_server_url}")
        
        mcp_client = DatabricksMCPClient(
            server_url=mcp_server_url,
            workspace_client=workspace_client
        )
        
        # Discover tools
        tools = mcp_client.list_tools()
        print(f"Discovered {len(tools)} vector search tools:")
        
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Test a tool (if available)
        if tools:
            test_tool = tools[0]
            result = mcp_client.call_tool(
                name=test_tool.name,
                arguments={"query": "test query", "num_results": 3}
            )
            print(f"Test result: {result}")
        
        return mcp_client, tools
        
    except Exception as e:
        print(f"MCP setup error: {e}")
        print("Check catalog/schema names and permissions")
        return None, []

# Usage
mcp_client, tools = setup_mcp_vector_search("my_catalog", "my_schema")
```

### Error Handling

```python
try:
    results = vs_tool.invoke(query)
    if not results:
        print("No results found")
except Exception as e:
    print(f"Search failed: {e}")
    # Implement fallback strategy
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**: Check DATABRICKS_HOST and DATABRICKS_TOKEN environment variables
2. **Index Not Found**: Confirm index name format: `catalog.schema.index` and verify index exists
3. **Empty Results**: Verify index has data, check query relevance, and review filter conditions
4. **Performance Issues**: Reduce `num_results`, limit returned columns, use appropriate query type

### Performance Optimization Tips

```python
# Optimized configuration for performance
vs_tool = VectorSearchRetrieverTool(
    index_name="catalog.schema.optimized_index",
    num_results=3,  # Lower number for faster responses
    columns=["id", "text"],  # Minimal columns to reduce data transfer
    query_type="ANN",  # Faster than HYBRID for pure semantic search
    tool_name="fast_retriever",
    tool_description="Optimized retriever for fast responses"
)
```

**Key Performance Tips**:
- Use fewer `num_results` for faster queries
- Limit columns to only what you need
- Use ANN query type for pure semantic search
- Apply filters to reduce search space
- Consider caching for frequently asked questions

## Code Examples

### Basic Usage Example

```python
from databricks_langchain import VectorSearchRetrieverTool

def basic_vector_search():
    """Simple retrieval with minimal configuration for Delta Sync indexes."""
    vs_tool = VectorSearchRetrieverTool(
        index_name="catalog.schema.my_databricks_docs_index",
        tool_name="databricks_docs_retriever",
        tool_description="Retrieves information about Databricks products from official documentation."
    )
    
    # Test the retriever
    results = vs_tool.invoke("What is Databricks Agent Framework?")
    print(f"Retrieved {len(results)} results")
    return vs_tool
```

### Advanced Configuration Example

```python
from databricks_langchain import VectorSearchRetrieverTool, DatabricksEmbeddings

def advanced_vector_search():
    """Advanced configuration with custom embeddings and filtering."""
    # Initialize custom embedding model
    embedding_model = DatabricksEmbeddings(
        endpoint="databricks-bge-large-en"
    )
    
    # Create retriever with advanced configuration
    vs_tool = VectorSearchRetrieverTool(
        index_name="catalog.schema.index_name",
        num_results=5,
        columns=["id", "content", "source", "metadata"],
        filters={"source": "databricks_docs"},
        query_type="HYBRID",  # Semantic + keyword search
        tool_name="advanced_databricks_retriever",
        tool_description="Advanced retriever with filtering and hybrid search",
        text_column="content",
        embedding=embedding_model
    )
    
    return vs_tool
```

### LLM Integration Example

```python
from databricks_langchain import ChatDatabricks
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

def create_rag_agent():
    """Creates a conversational agent using vector search for RAG."""
    # Initialize components
    vs_tool = basic_vector_search()
    llm = ChatDatabricks(endpoint="databricks-claude-3-7-sonnet")
    
    # Create agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful Databricks expert assistant. Use the vector search tool 
        to find relevant information from documentation to answer questions accurately. 
        Always cite your sources."""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])
    
    # Create and test agent
    agent = create_tool_calling_agent(llm, [vs_tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[vs_tool], verbose=True)
    
    # Test query
    response = agent_executor.invoke({
        "input": "How do I create and manage a vector search index in Databricks?"
    })
    
    return agent_executor
```

### Error Handling and Validation

```python
import os

def safe_vector_search():
    """Demonstrates proper error handling for vector search operations."""
    try:
        # Validate environment
        if not os.getenv("DATABRICKS_HOST") and not os.getenv("DATABRICKS_TOKEN"):
            print("Warning: Databricks authentication not configured")
        
        vs_tool = VectorSearchRetrieverTool(
            index_name="catalog.schema.my_index",
            tool_name="safe_retriever",
            tool_description="Safely retrieves information with error handling",
            num_results=3
        )
        
        # Test with different query types
        test_queries = [
            "What is machine learning?",
            "very_specific_technical_term_that_might_not_exist"
        ]
        
        for query in test_queries:
            try:
                print(f"Testing query: '{query}'")
                results = vs_tool.invoke(query)
                
                if results:
                    print(f"  ✓ Found {len(results)} results")
                else:
                    print("  ⚠ No results found")
                    
            except Exception as query_error:
                print(f"  ✗ Query failed: {query_error}")
                
    except Exception as setup_error:
        print(f"Setup error: {setup_error}")
        print("Check your index name, authentication, and network connectivity")
```

## Usage Checklist

Before deploying to production:

1. ✅ Replace index names with your actual indexes
2. ✅ Configure proper authentication (service principal recommended)
3. ✅ Test with your specific data and use case
4. ✅ Monitor performance and adjust parameters as needed
5. ✅ Implement proper error handling and fallback strategies
6. ✅ Set up appropriate logging and monitoring

## Additional Resources

- [Databricks Vector Search Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Databricks AI Examples](https://github.com/databricks/databricks-ml-examples)