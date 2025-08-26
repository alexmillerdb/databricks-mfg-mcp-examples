# Databricks Manufacturing MCP Examples - Project Context

## Overview
This project implements a multi-tenant Supply Chain & Sales agentic system on Databricks using Model Context Protocol (MCP), Databricks Agent Framework, and Unity Catalog governance. See [PRD.md](./PRD.md) for complete requirements.

## Architecture Summary
The system demonstrates MCP-based agent orchestration with:
- **Managed MCP Servers**: Vector Search, Genie Spaces, UC Functions
- **Custom MCP Servers**: External APIs, IoT connectors on Databricks Apps
- **Multi-tenant isolation** via Unity Catalog and on-behalf-of-user (OBO) authentication
- **MLflow 3.0** for monitoring, tracing, and evaluation

## Key Use Cases

### Supply Chain Optimization
- Query inventory/shipments via Genie MCP
- Run predictive models for shortages via UC Functions
- Retrieve incident reports via Vector Search
- Generate actionable recommendations

### Sales/Customer Insights  
- Query sales KPIs via Genie MCP
- Run churn/upsell models via UC Functions
- Retrieve past proposals via Vector Search
- Generate next-best-actions

## Implementation Guidelines

### MCP vs UC Functions
- **UC Functions Only**: Simple, tightly-coupled use cases in notebooks
- **MCP**: Standardized, reusable tool calling across agents and external systems
- **Managed MCP**: Instant access to Databricks services without infrastructure
- **Custom MCP**: Specialized logic and external API integrations

## Technical Stack
- **Databricks Agent Framework** for LLM orchestration
- **Unity Catalog** for governance and multi-tenancy
- **Delta Lake** for data storage
- **Databricks Apps** for UI and custom MCP servers
- **MLflow 3.0** for observability

## Documentation References

### Local Implementation Guides
**IMPORTANT**: Always reference the `docs/` folder for detailed implementation examples and code patterns:
- **[docs/managed-mcp-servers-guide.md](./docs/managed-mcp-servers-guide.md)** - Complete guide for building agents with managed MCP servers
- **[docs/langgraph-mcp-agent.md](./docs/langgraph-mcp-agent.md)** - LangGraph agent implementation with full code examples
- **[docs/databricks-mcp-documentation.md](./docs/databricks-mcp-documentation.md)** - Core MCP concepts and architecture
- **[docs/deploying-on-behalf-of-user-agents.md](./docs/deploying-on-behalf-of-user-agents.md)** - On-Behalf-Of authentication for multi-tenant agents with Unity Catalog integration
-- **[docs/mlflow3-documentation-guide.md](./docs/mlflow3-documentation-guide.md)** - MLflow3 GenAI apps documentation guide

When planning, researching, or implementing MCP-related code, ALWAYS consult these local documentation files first as they contain:
- Production-ready code examples
- Error handling patterns
- Performance optimization strategies
- Testing approaches
- Deployment configurations

### Core MCP Documentation
- [Databricks MCP Overview](https://docs.databricks.com/aws/en/generative-ai/mcp/)
- [Managed MCP Servers](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp)
- [Custom MCP Servers](https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp)

### Implementation Examples
- [Build an agent with managed MCP servers (Local IDE)](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp#local-ide-build-an-agent-with-managed-mcp-servers)
- [Deploy MCP tool-calling LangGraph Agent](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html)

### Authentication & Security
- [On-Behalf-Of User Authentication](https://docs.databricks.com/aws/en/generative-ai/agent-framework/authenticate-on-behalf-of-user)
- [Agents OBO Example Notebook](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/agents-obo-example.html)

### Monitoring & Observability
- [MLflow 3 Demo Repository](https://github.com/databricks-solutions/mlflow-demo)
- [MLflow GenAI API Reference](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html)

## Project Structure
```
databricks-mfg-mcp-examples/
├── src/
│   ├── agents/          # Agent orchestrator code
│   ├── mcp_servers/     # Custom MCP server implementations
│   ├── data/            # Data generation and processing
│   ├── app/             # Databricks App UI
│   └── utils/           # Shared utilities
├── notebooks/           # Databricks notebooks for demos
├── tests/              # Unit and integration tests
├── config/             # Configuration files
└── databricks.yml      # Databricks asset bundle configuration
```

## Development Workflow
1. Set up Databricks workspace connection
2. Deploy Delta tables with sample data
3. Configure managed MCP servers (Genie, Vector Search, UC Functions)
4. Deploy custom MCP server on Databricks Apps
5. Deploy agent orchestrator to Model Serving
6. Launch Databricks App UI for user interaction
7. Monitor with MLflow tracing and evaluation

## Testing Commands
```bash
# Run unit tests
pytest tests/

# Deploy to Databricks (dev environment)
databricks bundle deploy

# Validate bundle configuration
databricks bundle validate
```