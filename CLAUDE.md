# Databricks Manufacturing MCP Examples - Project Context

## Quick start priorities
- Default to managed MCP for Databricks services (Genie, Vector Search, Unity Catalog Functions) to maximize standardization, governance, and least‑privilege access; reserve custom MCP (Databricks Apps) for specialized logic or third‑party systems.
- Pin versions and verify environment
- Log Databricks resource dependencies with the model so Databricks can inject short‑lived credentials automatically at deploy time (Vector Search indexes, UC Functions, Genie spaces, serving endpoints)
- Prefer automatic auth passthrough for Databricks‑managed resources; use on‑behalf‑of‑user (OBO) only when per‑user scoping is required, with minimal scopes aligned to Unity Catalog policies.

## Overview
This project implements a multi-tenant Supply Chain & Sales agentic system on Databricks using Model Context Protocol (MCP), Databricks Agent Framework, and Unity Catalog governance. See [PRD.md](./PRD.md) for complete requirements.

## Architecture Summary
The system demonstrates MCP-based agent orchestration with:
- **Managed MCP Servers**: Vector Search, Genie Spaces, UC Functions to expose governed tools instantly without self‑hosting.
- **Custom MCP Servers**: External APIs, IoT connectors on Databricks Apps when managed MCP does not cover a needed capability.
- **Multi-tenant isolation** via Unity Catalog and on-behalf-of-user (OBO) authentication available to scope access at query time to the invoking user where appropriate.
- **MLflow 3.0** observability for tracing, inference logging, and evaluation aligned with Databricks agent deployment features.

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
- Prefer managed MCP for Databricks services for immediate availability and automatic UC policy enforcement; switch to custom MCP only for bespoke logic or non‑Databricks systems.

## Technical Stack
- **Databricks Agent Framework** for LLM orchestration
- **MCP** for building scalable, reusable, security tools for Agents
- **Unity Catalog** for governance and multi-tenancy
- **Delta Lake** for data storage
- **Databricks Apps** for UI and custom MCP servers
- **MLflow 3.0** for observability

## Example End-to-End Process Flow (ALWAYS FOLLOW THIS WORKFLOW FOR DEVELOPING AGENTS)

1. **Author agent** with correct interface/schema with explicit tools and inputs/outputs. 
2. **Instrument** with MLflow tracing/autologging to capture prompts, tool calls, and results.
3. **Develop/capture golden data** for offline evaluation and keep it versioned alongside code and prompts.  
4. **Register scorers** (LLM-based or code-based) for both offline and live monitoring.
5. **Evaluate** agent's quality (on code & live traffic).  
6. **Log/track** app+prompt version with clear model registry lineage in Unity Catalog.  
7. **Register & deploy** agent with explicit resource/auth policies via deploy().  
8. **Monitor production** with the same scorers, traces, and inference tables for feedback loops.  
9. **Collect & use feedback** for iterative improvement via the Review App and unify it with offline golden datasets for regression checks.  
10. **Repeat**—integrate improvements, retest, redeploy, monitor.

## Resource declaration (required at logging time)
- Declare all Databricks‑managed dependencies used by the agent so deploy() can inject short‑lived credentials with least privilege, including Vector Search indexes, UC Functions, Genie spaces, and any serving endpoints the agent relies on.
- Validate that the endpoint owner holds the necessary UC permissions so Databricks can issue the credentials safely at deployment time without privilege escalation.

## Authentication and security policy
- Prefer automatic Databricks authentication for managed resources to avoid distributing secrets and to align with UC permissions by default.
- Use OBO when per‑user scoping is required so Unity Catalog row‑level and column‑level policies apply to each request, with downscoped tokens limited to specific agent APIs.
-  long‑lived secrets for non‑Databricks external systems, never log tokens, and audit access paths in Databricks Apps for custom MCP servers.

## Documentation References

### Local Implementation Guides
**IMPORTANT**: Always reference the `docs/` folder for detailed implementation examples and code patterns:

#### Agent Development (`docs/agents/`)
- **[docs/agents/best-practices-deploying-agents-workflow.md](./docs/agents/best-practices-deploying-agents-workflow.md)** - Best practices for developing, evaluating, and deploying Agents on Databricks. ALWAYS FOLLOW THIS WORKFLOW
- **[docs/agents/databricks-agent-uc-tools.md](./docs/agents/databricks-agent-uc-tools.md)** - Unity Catalog functions integration with Databricks Agent Framework for custom tool creation
- **[docs/agents/langgraph-mcp-agent.md](./docs/agents/langgraph-mcp-agent.md)** - LangGraph agent implementation with MCP integration and full code examples
- **[docs/agents/deploying-on-behalf-of-user-agents.md](./docs/agents/deploying-on-behalf-of-user-agents.md)** - On-Behalf-Of authentication for multi-tenant agents with Unity Catalog integration

#### MCP Integration (`docs/mcp/`)
- **[docs/mcp/managed-mcp-servers-guide.md](./docs/mcp/managed-mcp-servers-guide.md)** - Complete guide for building agents with managed MCP servers
- **[docs/mcp/databricks-mcp-documentation.md](./docs/mcp/databricks-mcp-documentation.md)** - Core MCP concepts and architecture

#### MLflow & Observability (`docs/mlflow/`)
- **[docs/mlflow/mlflow-agent-development-guide.md](./docs/mlflow/mlflow-agent-development-guide.md)** - Complete guide for developing, logging, and deploying agents using MLflow's ResponsesAgent interface
- **[docs/mlflow/mlflow3-documentation-guide.md](./docs/mlflow/mlflow3-documentation-guide.md)** - MLflow 3 GenAI apps documentation guide with tracing, evaluation, and monitoring

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
│   ├── config/          # Environment and parameterized variables
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
4. Deploy custom MCP server on Databricks Apps (if required but optional)
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