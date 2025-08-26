#!/bin/bash
# Activation script for the Databricks MCP Examples project

echo "🚀 Activating Databricks MCP Examples environment..."
echo "📍 Project: Multi-tenant Supply Chain & Sales Agentic System"
echo ""

# Activate the virtual environment
source .venv/bin/activate

# Display environment info
echo "✅ Virtual environment activated"
echo "🐍 Python version: $(python --version)"
echo "📦 UV version: $(uv --version)"
echo ""
echo "📚 Key dependencies available:"
echo "   - MCP (Model Context Protocol)"
echo "   - LangGraph for agent orchestration"
echo "   - Databricks SDK and Agent Framework"
echo "   - MLflow for monitoring and tracing"
echo ""
echo "🔧 Development commands:"
echo "   uv pip install -e .          # Install project in editable mode"
echo "   uv pip install -e .[dev]     # Install with dev dependencies"
echo "   pytest tests/                # Run tests"
echo "   databricks bundle deploy     # Deploy to Databricks"
echo ""
echo "📖 Documentation: https://docs.databricks.com/aws/en/generative-ai/mcp/"
echo "🎯 Ready to build MCP-powered agents!"
