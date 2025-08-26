# LangGraph MCP Tool-Calling Agent Implementation

**Source**: https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html
**Domain**: docs.databricks.com
**Fetched**: 2025-08-26
**Type**: Notebook Implementation Example

## Overview

This notebook demonstrates building a flexible LangGraph agent that integrates with Databricks MCP servers for dynamic tool discovery and execution. The agent supports both managed and custom MCP servers, enabling sophisticated tool-calling workflows with state management and MLflow tracking.

## Key Concepts

### Architecture Components
- **LangGraph**: Provides stateful agent workflow orchestration
- **MCP Integration**: Dynamic tool discovery and execution
- **MLflow Tracking**: Automatic tracing and monitoring
- **Mosaic AI Compatibility**: Seamless deployment to Databricks Model Serving

### Agent Capabilities
- Connect to multiple MCP servers simultaneously
- Discover tools dynamically at runtime
- Execute tool calls with proper error handling
- Maintain conversation state across turns
- Stream responses for better user experience

## Code Examples

### Complete Agent Implementation

#### 1. Dependencies Installation
```python
%pip install -U -qqq mcp>=1.9 \
                     databricks-sdk \
                     databricks-agents \
                     databricks-mcp \
                     databricks-langchain \
                     uv \
                     langgraph==0.3.4
%restart_python
```

#### 2. Core Imports and Setup
```python
import asyncio
import json
from typing import Any, Dict, List, Optional, Literal, TypedDict, Annotated
from dataclasses import dataclass, field, asdict

from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient
from databricks_agents import Agent

from langchain_core.tools import Tool
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, 
    SystemMessage, ToolMessage
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import mlflow
mlflow.langchain.autolog()
```

#### 3. MCP Tool Wrapper Class
```python
class MCPTool:
    """Wrapper to convert MCP tools to LangChain-compatible tools"""
    
    def __init__(self, mcp_client: DatabricksMCPClient, tool_name: str, tool_description: str, tool_schema: dict):
        self.mcp_client = mcp_client
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_schema = tool_schema
    
    async def _arun(self, *args, **kwargs):
        """Async execution of MCP tool"""
        # Convert LangChain args to MCP format
        mcp_args = self._convert_args(kwargs)
        
        # Call MCP tool
        result = await self.mcp_client.call_tool(
            name=self.tool_name,
            arguments=mcp_args
        )
        
        return self._format_result(result)
    
    def _run(self, *args, **kwargs):
        """Sync wrapper for async tool execution"""
        return asyncio.run(self._arun(*args, **kwargs))
    
    def _convert_args(self, langchain_args):
        """Convert LangChain arguments to MCP format"""
        # Implementation depends on tool schema
        return langchain_args
    
    def _format_result(self, mcp_result):
        """Format MCP result for LangChain"""
        if isinstance(mcp_result, dict):
            return json.dumps(mcp_result, indent=2)
        return str(mcp_result)
    
    def to_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        return Tool(
            name=self.tool_name,
            description=self.tool_description,
            func=self._run,
            coroutine=self._arun
        )
```

#### 4. Agent State Definition
```python
class AgentState(TypedDict):
    """State for the LangGraph agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    tools: List[Tool]
    next_action: Optional[str]
```

#### 5. Tool Discovery and Registration
```python
async def discover_and_register_tools(mcp_client: DatabricksMCPClient, server_urls: List[str]) -> List[Tool]:
    """Discover tools from MCP servers and convert to LangChain tools"""
    all_tools = []
    
    for server_url in server_urls:
        # Connect to server
        await mcp_client.connect_server_session(server_url)
        
        # List available tools
        mcp_tools = await mcp_client.list_tools()
        
        # Convert each tool
        for tool in mcp_tools:
            mcp_tool_wrapper = MCPTool(
                mcp_client=mcp_client,
                tool_name=tool.name,
                tool_description=tool.description,
                tool_schema=tool.input_schema
            )
            all_tools.append(mcp_tool_wrapper.to_langchain_tool())
    
    return all_tools
```

#### 6. LangGraph Agent Workflow
```python
def create_tool_calling_agent(llm, tools: List[Tool]):
    """Create a LangGraph agent with tool-calling capabilities"""
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Define agent node
    async def agent_node(state: AgentState):
        """Main agent decision node"""
        messages = state["messages"]
        
        # Get LLM response
        response = await llm_with_tools.ainvoke(messages)
        
        # Check if tool calls are needed
        if response.tool_calls:
            state["next_action"] = "tools"
        else:
            state["next_action"] = "end"
        
        return {"messages": [response], "next_action": state["next_action"]}
    
    # Define tool execution node
    tool_node = ToolNode(tools)
    
    # Build state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        lambda x: x["next_action"],
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()
```

#### 7. Mosaic AI Compatible Agent Wrapper
```python
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)

class LangGraphResponsesAgent(ResponsesAgent):
    """ResponsesAgent wrapper for Mosaic AI deployment"""
    
    def __init__(self, mcp_server_urls: List[str], llm_endpoint: str):
        self.mcp_server_urls = mcp_server_urls
        self.llm_endpoint = llm_endpoint
        self.mcp_client = None
        self.agent = None
        self.tools = []
    
    async def _initialize(self):
        """Lazy initialization of MCP client and tools"""
        if self.mcp_client is None:
            # Initialize MCP client
            w = WorkspaceClient()
            self.mcp_client = DatabricksMCPClient(workspace_client=w)
            
            # Discover tools
            self.tools = await discover_and_register_tools(
                self.mcp_client, 
                self.mcp_server_urls
            )
            
            # Create LLM
            from langchain_databricks import ChatDatabricks
            llm = ChatDatabricks(endpoint=self.llm_endpoint)
            
            # Create agent
            self.agent = create_tool_calling_agent(llm, self.tools)
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Synchronous prediction method for MLflow"""
        return asyncio.run(self.apredict(request))
    
    async def apredict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Async prediction with streaming support"""
        await self._initialize()
        
        # Convert input to messages
        messages = self._parse_messages([msg.model_dump() for msg in request.input])
        
        # Create initial state
        initial_state = {
            "messages": messages,
            "tools": self.tools,
            "next_action": None
        }
        
        # Run agent
        result = await self.agent.ainvoke(initial_state)
        
        # Extract response
        response_message = result["messages"][-1]
        
        # Create output item
        output_item = self.create_text_output_item(
            text=response_message.content,
            id=str(uuid4())
        )
        
        return ResponsesAgentResponse(output=[output_item])
    
    def _parse_messages(self, messages: List[Dict]) -> List[BaseMessage]:
        """Convert dict messages to LangChain messages"""
        parsed = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "user":
                parsed.append(HumanMessage(content=content))
            elif role == "assistant":
                parsed.append(AIMessage(content=content))
            elif role == "system":
                parsed.append(SystemMessage(content=content))
        
        return parsed
```

#### 8. Configuration and Deployment
```python
# Configuration for managed MCP servers
MCP_SERVER_URLS = [
    "https://<workspace>/api/2.0/mcp/vector-search/catalog/schema",
    "https://<workspace>/api/2.0/mcp/functions/catalog/schema",
    "https://<workspace>/api/2.0/mcp/genie/genie_space_id"
]

# LLM endpoint (e.g., DBRX, Llama, or custom model)
LLM_ENDPOINT = "databricks-claude-3-7-sonnet"

# Create agent instance
agent = LangGraphResponsesAgent(
    mcp_server_urls=MCP_SERVER_URLS,
    llm_endpoint=LLM_ENDPOINT
)

# Log with MLflow
with mlflow.start_run():
    # Get required resources
    mcp_client = DatabricksMCPClient()
    resources = mcp_client.get_databricks_resources()
    
    # Log model
    logged_model = mlflow.pyfunc.log_model(
        artifact_path="langgraph_mcp_agent",
        python_model=agent,
        pip_requirements=[
            "mcp>=1.9",
            "databricks-mcp>=1.0.0",
            "databricks-agents>=1.0.0",
            "langgraph==0.3.4",
            "langchain-databricks>=0.1.0"
        ],
        resources=resources,
        registered_model_name="langgraph_mcp_agent"
    )
    
    print(f"Model URI: {logged_model.model_uri}")
```

## Implementation Notes

### Connection Modes

#### Managed MCP Server Connection
```python
# Simple connection using workspace authentication
mcp_client = DatabricksMCPClient(workspace_client=WorkspaceClient())
await mcp_client.connect_server_session(
    "https://workspace/api/2.0/mcp/vector-search/catalog/schema"
)
```

#### Custom MCP Server with OAuth
```python
# OAuth configuration for custom servers
custom_client = DatabricksMCPClient(
    workspace_client=WorkspaceClient(),
    oauth_config={
        "client_id": "your-client-id",
        "client_secret": "your-client-secret",
        "token_url": "https://custom-server/oauth/token"
    }
)
```

### Error Handling and Retries
```python
async def robust_tool_call(mcp_client, tool_name, arguments, max_retries=3):
    """Execute tool call with retry logic"""
    for attempt in range(max_retries):
        try:
            result = await mcp_client.call_tool(
                name=tool_name,
                arguments=arguments
            )
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Testing the Agent
```python
# Test with simple query
from mlflow.types.responses import ResponsesAgentRequest

test_request = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "What is 7*6 in Python?"}
    ]
)

result = agent.predict(test_request)
print(f"Agent response: {result.output[0].content[0].text}")

# Test with tool-requiring query
test_request_with_tools = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "Search for customer churn predictions in the vector index"}
    ]
)

result = agent.predict(test_request_with_tools)
print(f"Final response: {result.output[0].content[0].text}")
```

### Performance Optimization

For production deployments, consider:
- **Tool caching**: Cache discovered tools with TTL to reduce server calls
- **Parallel discovery**: Discover tools from multiple servers concurrently
- **Connection pooling**: Reuse MCP client connections when possible

### Monitoring and Observability

```python
# Enable comprehensive MLflow autologging
mlflow.langchain.autolog(
    log_models=True,
    log_traces=True
)

# Track key metrics during agent execution
with mlflow.start_run():
    mlflow.log_metric("num_tools_discovered", len(tools))
    mlflow.log_metric("num_mcp_servers", len(MCP_SERVER_URLS))
```

## Related Resources
- Reference [CLAUDE.md](../CLAUDE.md) for project-specific agent implementation patterns
- See [managed-mcp-servers-guide.md](./managed-mcp-servers-guide.md) for detailed MCP server configuration
- Check [PRD.md](../PRD.md) for manufacturing-specific agent requirements
- Review MLflow documentation for advanced tracking and deployment options