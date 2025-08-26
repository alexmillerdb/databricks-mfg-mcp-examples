#!/usr/bin/env python3
"""
Verification script for Databricks MCP Examples environment setup.
This script verifies that all required dependencies are properly installed.
"""

import sys
from importlib import import_module

def check_import(module_name: str, description: str) -> bool:
    """Check if a module can be imported successfully."""
    try:
        import_module(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description} - Error: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 Verifying Databricks MCP Examples Environment Setup")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 11):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python version {python_version.major}.{python_version.minor} (requires 3.11+)")
        return False
    
    print("\n📦 Checking Dependencies:")
    print("-" * 30)
    
    # Core dependencies to check
    dependencies = [
        ("mcp", "Model Context Protocol (MCP)"),
        ("langgraph", "LangGraph for agent orchestration"),
        ("databricks.sdk", "Databricks SDK"),
        ("databricks.agents", "Databricks Agents Framework"),
        ("databricks.connect", "Databricks Connect"),
        ("mlflow", "MLflow for monitoring"),
        ("pandas", "Pandas for data processing"),
        ("numpy", "NumPy for numerical computing"),
        ("pydantic", "Pydantic for data validation"),
        ("fastapi", "FastAPI for web framework"),
        ("uvicorn", "Uvicorn ASGI server"),
        ("httpx", "HTTPX for HTTP requests"),
        ("aiofiles", "Aiofiles for async file operations"),
    ]
    
    all_good = True
    for module, description in dependencies:
        if not check_import(module, description):
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Environment setup verification PASSED!")
        print("\n🚀 Ready to build MCP-powered agents on Databricks!")
        print("\n📚 Next steps:")
        print("   1. Configure Databricks workspace connection")
        print("   2. Set up managed MCP servers (Genie, Vector Search, UC Functions)")
        print("   3. Deploy custom MCP servers on Databricks Apps")
        print("   4. Build and deploy agent orchestrator")
        print("\n📖 Documentation: https://docs.databricks.com/aws/en/generative-ai/mcp/")
        return True
    else:
        print("❌ Environment setup verification FAILED!")
        print("Please install missing dependencies and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
