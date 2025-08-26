# MLflow Agent Development Guide

This guide demonstrates how to develop, log, evaluate, and deploy AI agents using MLflow's `ResponsesAgent` interface and PyFunc model logging capabilities.

## Overview

MLflow provides two key components for agent development:

1. **ResponsesAgent Interface**: Modern, production-grade agent authoring with streaming support
2. **PyFunc Model Logging**: Comprehensive agent deployment with resource management

## ResponsesAgent Interface

Databricks recommends the MLflow `ResponsesAgent` interface to create production-grade agents. `ResponsesAgent` lets you build agents with any third-party framework, then integrate it with Databricks AI features for robust logging, tracing, evaluation, deployment, and monitoring capabilities.

### Key Benefits

- **Advanced agent capabilities**: Multi-agent support and comprehensive tool-calling
- **Streaming output**: Stream the output in smaller chunks for interactive user experiences
- **Comprehensive tool-calling message history**: Return multiple messages, including intermediate tool-calling messages
- **Tool-calling confirmation support**: Enhanced conversation management
- **Long-running tool support**: Handle complex, time-intensive operations
- **Framework-agnostic**: Wrap any existing agent for MLflow compatibility
- **Typed authoring**: Python classes with IDE autocomplete support
- **Automatic signature inference**: Simplified registration and deployment
- **Automatic tracing**: MLflow automatically traces your predict and predict_stream functions
- **AI Gateway-enhanced inference tables**: Automatic access to detailed request log metadata

### Basic ResponsesAgent Implementation

```python
from uuid import uuid4
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)

class MyWrappedAgent(ResponsesAgent):
    """
    Example wrapper for integrating a custom agent with the MLflow ResponsesAgent interface.
    
    This class demonstrates how to adapt an existing agent to the MLflow ResponsesAgent API,
    enabling compatibility with MLflow's responses schema, streaming, and deployment features.
    """

    def __init__(self, agent):
        """Initialize the wrapped agent."""
        # Reference your existing agent
        self.agent = agent

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Generate a response using the wrapped agent.
        
        Args:
            request (ResponsesAgentRequest): The incoming request with messages
            
        Returns:
            ResponsesAgentResponse: The agent's response in MLflow's responses schema
        """
        # Convert incoming messages to your agent's format
        # prep_msgs_for_llm is a function you write to convert the incoming messages
        messages = self.prep_msgs_for_llm([i.model_dump() for i in request.input])

        # Call your existing agent (non-streaming)
        agent_response = self.agent.invoke(messages)

        # Convert your agent's output to ResponsesAgent format, assuming agent_response is a str
        output_item = self.create_text_output_item(text=agent_response, id=str(uuid4()))

        # Return the response
        return ResponsesAgentResponse(output=[output_item])

    def prep_msgs_for_llm(self, messages):
        """Convert ResponsesAgent messages to your agent's expected format."""
        # Implement your message conversion logic here
        return messages
```

### Streaming ResponsesAgent Implementation

```python
from typing import Generator
from uuid import uuid4
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

class MyWrappedStreamingAgent(ResponsesAgent):
    """
    Streaming ResponsesAgent that reuses logic to avoid code duplication.
    """

    def __init__(self, agent):
        """Initialize the wrapped agent."""
        # Reference your existing agent
        self.agent = agent

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming predict: collects all streaming chunks into a single response."""
        # Reuse the streaming logic and collect all output items
        output_items = []
        for stream_event in self.predict_stream(request):
            if stream_event.type == "response.output_item.done":
                output_items.append(stream_event.item)

        # Return all collected items as a single response
        return ResponsesAgentResponse(output=output_items)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming predict: the core logic that both methods use."""
        # Convert incoming messages to your agent's format
        messages = self.prep_msgs_for_llm([i.model_dump() for i in request.input])

        # Stream from your existing agent
        item_id = str(uuid4())
        aggregated_stream = ""
        
        for chunk in self.agent.stream(messages):
            # Convert each chunk to ResponsesAgent format
            aggregated_stream += chunk
            
            # Yield streaming delta event
            yield ResponsesAgentStreamEvent(
                type="response.output_item.delta",
                item=self.create_text_output_item(
                    text=chunk, 
                    id=item_id
                )
            )

        # Yield final completion event
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(
                text=aggregated_stream, 
                id=item_id
            )
        )

    def prep_msgs_for_llm(self, messages):
        """Convert ResponsesAgent messages to your agent's expected format."""
        # Implement your message conversion logic here
        return messages
```

## Agent Logging and Deployment Workflow

### Prerequisites

- An existing agent defined in `agent.py` with tools, `LLM_ENDPOINT_NAME`, and `AGENT`
- Databricks workspace with Unity Catalog enabled
- Appropriate permissions for model registration and deployment

### Step 1: Prepare Resources

```python
import mlflow
from agent import tools, LLM_ENDPOINT_NAME, AGENT
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool
from pkg_resources import get_distribution

# Initialize resources list with the LLM serving endpoint
resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]

# Iterate through agent tools to collect additional resources
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        # Add vector search index resources for RAG functionality
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        # Add Unity Catalog function resources for data access
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))
```

### Step 2: Log Agent as MLflow PyFunc Model

```python
# Start an MLflow run to track the model logging process
with mlflow.start_run():
    # Log the agent as a PyFunc model with all dependencies
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",  # Model name in the MLflow run
        python_model="agent.py",  # Python file containing the agent implementation
        resources=resources,  # Databricks resources the model depends on
        pip_requirements=[
            # Pin dependency versions to ensure reproducible deployments
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
        ],
    )
```

### Step 3: Evaluate Agent Performance

```python
from mlflow.genai.scorers import RelevanceToQuery, Safety

# Define evaluation dataset with test queries using ResponsesAgent schema
eval_dataset = [
    {
        "inputs": {
            "input": [{"role": "user", "content": "What is an LLM?"}]
        },
        "expected_response": None,  # Set to None for auto-evaluation
    }
    # Add more test cases covering different agent capabilities
]

# Evaluate the agent using MLflow's built-in scorers
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input_data: AGENT.predict(
        ResponsesAgentRequest(input=input_data["input"])
    ),
    scorers=[
        RelevanceToQuery(),  # Measures how relevant responses are to the query
        Safety()  # Checks for potentially harmful content
    ],
)
```

### Step 4: Test Logged Model Locally

```python
# Test the logged model locally before deployment using ResponsesAgent schema
mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={
        "input": [{"role": "user", "content": "Hello!"}]
    },
    env_manager="uv",  # Use uv for faster dependency resolution
)
```

### Step 5: Register Model to Unity Catalog

```python
# Configure MLflow to use Unity Catalog as the model registry
mlflow.set_registry_uri("databricks-uc")

# Define Unity Catalog model location
catalog = ""  # e.g., "main" or your workspace catalog
schema = ""   # e.g., "ai_agents" or your schema name
model_name = ""  # e.g., "customer_support_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# Register the logged model to Unity Catalog
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name=UC_MODEL_NAME
)
```

### Step 6: Deploy Agent to Model Serving

```python
from databricks import agents

# Deploy the registered model to a Databricks Model Serving endpoint
agents.deploy(
    UC_MODEL_NAME,  # Model name in Unity Catalog
    uc_registered_model_info.version,  # Specific model version to deploy
    tags={"endpointSource": "docs"}  # Tags for tracking and organization
)
```

## Best Practices

### Resource Dependencies
- Ensure all tools in your agent have proper resource dependencies defined
- Vector search indices must be properly configured and accessible
- Unity Catalog functions must exist and have appropriate permissions

### Model Versioning
- Each run creates a new model version in Unity Catalog
- Use meaningful model names and tags for easier tracking
- Consider implementing automated versioning strategies

### Evaluation
- Expand the evaluation dataset with diverse test cases
- Monitor evaluation metrics over time to track model performance
- Consider adding custom scorers for domain-specific evaluation
- Use ResponsesAgent schema for evaluation datasets
- Leverage automatic tracing for better evaluation insights

### Deployment
- Test thoroughly in development before production deployment
- Monitor endpoint performance and costs after deployment
- Implement proper error handling and logging in production
- Use synchronous code patterns to avoid event loop conflicts
- Design for distributed serving environments

### Security
- Ensure proper access controls are in place for Unity Catalog models
- Review and audit model permissions regularly
- Follow your organization's security guidelines for AI/ML deployments

## Advanced Features

### Custom Inputs and Outputs

Some scenarios might require additional agent inputs, such as `client_type` and `session_id`, or outputs like retrieval source links that should not be included in the chat history for future interactions.

MLflow ResponsesAgent natively supports the fields `custom_inputs` and `custom_outputs`. You can access the custom inputs via `request.custom_inputs` in your agent implementation.

### Custom Retriever Schemas

AI agents commonly use retrievers to find and query unstructured data from vector search indices. Trace these retrievers within your agent with MLflow RETRIEVER spans to enable Databricks product features.

```python
import mlflow

# Define the retriever's schema by providing your column names
mlflow.models.set_retriever_schema(
    # Specify the name of your retriever span
    name="mlflow_docs_vector_search",
    # Specify the output column name to treat as the primary key (ID) of each retrieved document
    primary_key="document_id",
    # Specify the output column name to treat as the text content (page content) of each retrieved document
    text_column="chunk_text",
    # Specify the output column name to treat as the document URI of each retrieved document
    doc_uri="doc_uri",
    # Specify any other columns returned by the retriever
    other_columns=["title"],
)
```

## Deployment Considerations

### Prepare for Databricks Model Serving

Databricks deploys ResponsesAgents in a distributed environment on Databricks Model Serving. Pay attention to the following implications:

- **Avoid local caching**: Don't assume the same replica handles all requests in a multi-turn conversation
- **Thread-safe state**: Design agent state to be thread-safe, preventing conflicts in multi-threaded environments
- **Initialize state in the predict function**: Initialize state each time the predict function is called, not during ResponsesAgent initialization

### Use Synchronous Code

To ensure stability and compatibility, use synchronous code or callback-based patterns in your agent implementation. Databricks automatically manages asynchronous communication to provide optimal concurrency and performance when you deploy an agent.

## Key Features

### ResponsesAgent Interface Features
- **Multi-agent support**: Advanced agent capabilities
- **Streaming output**: Stream output in smaller chunks for interactive experiences
- **Comprehensive tool-calling message history**: Return multiple messages including intermediate tool calls
- **Tool-calling confirmation support**: Enhanced conversation management
- **Long-running tool support**: Handle complex, time-intensive operations
- **Automatic tracing**: MLflow automatically traces predict and predict_stream functions
- **AI Gateway-enhanced inference tables**: Automatic access to detailed request log metadata

### PyFunc Model Features
- **Resource management**: Automatic discovery and inclusion of dependencies
- **Reproducible deployments**: Pinned dependency versions
- **Comprehensive evaluation**: Built-in scorers for quality assessment
- **Unity Catalog integration**: Centralized model governance and access control
- **Production deployment**: REST API endpoints via Model Serving
