# Databricks Manufacturing MCP Examples

A multi-tenant Supply Chain & Sales agentic system built on Databricks using Model Context Protocol (MCP), Databricks Agent Framework, and Unity Catalog governance.

## Quick Start

### Environment Setup

This project uses [UV](https://docs.astral.sh/uv/) for fast Python package management with Python 3.11.

```bash
# Activate the environment (includes setup verification)
./activate.sh

# Or manually activate
source .venv/bin/activate

# Verify setup
python verify_setup.py
```

### Key Dependencies

- **MCP (Model Context Protocol)** - Standardized tool calling interface
- **LangGraph 0.3.4** - Agent orchestration framework  
- **Databricks SDK** - Databricks platform integration
- **Databricks Agents** - Databricks Agent Framework
- **Databricks Connect** - Databricks runtime connectivity
- **MLflow 3.3.0** - Monitoring, tracing, and evaluation
- **FastAPI** - Web framework for Databricks Apps

## Architecture

This system demonstrates MCP-based agent orchestration with:

- **Managed MCP Servers**: Vector Search, Genie Spaces, UC Functions
- **Custom MCP Servers**: External APIs, IoT connectors on Databricks Apps  
- **Multi-tenant isolation** via Unity Catalog and on-behalf-of-user (OBO) authentication
- **MLflow 3.0** for monitoring, tracing, and evaluation

## Use Cases

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

## Development

```bash
# Install project in editable mode
uv pip install -e .

# Install with development dependencies
uv pip install -e .[dev]

# Run tests
pytest tests/

# Deploy to Databricks
databricks bundle deploy

# Validate configuration
databricks bundle validate
```

## Documentation

- [Databricks MCP Overview](https://docs.databricks.com/aws/en/generative-ai/mcp/)
- [Deploy MCP tool-calling LangGraph Agent](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html)
- [Managed MCP Servers](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp)
- [Custom MCP Servers](https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp)

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

See [CLAUDE.md](./CLAUDE.md) for detailed project context and [PRD.md](./PRD.md) for complete requirements.
