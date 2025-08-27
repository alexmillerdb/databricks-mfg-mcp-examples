#!/usr/bin/env python3
"""
Databricks Environment Setup Utilities

Common functions for setting up Databricks environments for both notebook and local IDE execution.
Supports automatic environment detection, Databricks Connect, and proper resource management.
"""

import os
from typing import Dict, Any, Optional


def setup_databricks_environment() -> Dict[str, Any]:
    """
    Universal setup for local IDE or notebook execution.
    
    Detects the execution environment and sets up the appropriate configuration.
    
    Returns:
        Dict containing environment configuration with keys:
        - environment: 'notebook' or 'local'
        - spark: SparkSession instance or None
        - workspace_client: WorkspaceClient instance or None
        - catalog: Default catalog name
        - schema: Default schema name
    """
    # Detect execution environment
    is_notebook = 'DATABRICKS_RUNTIME_VERSION' in os.environ
    
    if is_notebook:
        print("🟢 Databricks Notebook Environment")
        return setup_notebook_env()
    else:
        print("🔵 Local IDE Environment")
        return setup_local_env()


def setup_notebook_env() -> Dict[str, Any]:
    """
    Setup for Databricks notebook execution.
    
    In notebook environments, spark and dbutils are available globally.
    
    Returns:
        Dict with notebook environment configuration
    """
    from databricks.sdk import WorkspaceClient
    
    # In notebook, spark is available globally
    global spark  # Declare global to avoid undefined variable warning
    return {
        'environment': 'notebook',
        'spark': spark,  # Available globally in notebooks
        'workspace_client': WorkspaceClient(),
        'catalog': os.getenv('UC_DEFAULT_CATALOG', 'mfg_mcp_demo'),
        'schema': os.getenv('UC_DEFAULT_SCHEMA', 'supply_chain')
    }


def setup_local_env() -> Dict[str, Any]:
    """
    Setup for local IDE execution using Databricks Connect.
    
    Loads environment variables, creates Databricks Connect session,
    and configures MLflow tracking.
    
    Returns:
        Dict with local environment configuration
    """
    from dotenv import load_dotenv
    from databricks.connect import DatabricksSession
    from databricks.sdk import WorkspaceClient
    import mlflow
    
    # Load environment variables
    load_dotenv()
    
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "aws-apps")
    catalog = os.getenv("UC_DEFAULT_CATALOG", "mfg_mcp_demo")
    schema = os.getenv("UC_DEFAULT_SCHEMA", "supply_chain")
    
    try:
        # Initialize Databricks Connect
        spark = DatabricksSession.builder.profile(profile).serverless(True).getOrCreate()
        
        # Set catalog context only (no schema since we use fully qualified table names)
        spark.sql(f"USE CATALOG {catalog}")
        
        # Configure MLflow (optional but good to have)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
        mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))
        
        return {
            'environment': 'local',
            'spark': spark,
            'workspace_client': WorkspaceClient(profile=profile),
            'catalog': catalog,
            'schema': schema
        }
    except Exception as e:
        print(f"⚠️  Error setting up local environment: {e}")
        print("   Running in dry-run mode (operations will be printed but not executed)")
        return {
            'environment': 'local',
            'spark': None,  # Dry-run mode
            'workspace_client': None,
            'catalog': catalog,
            'schema': schema
        }


def cleanup_environment(config: Dict[str, Any]) -> None:
    """
    Clean up resources (local only).
    
    Properly stops Spark session for local environments to free resources.
    
    Args:
        config: Configuration dictionary from setup_databricks_environment()
    """
    if config.get('environment') == 'local' and config.get('spark'):
        try:
            config['spark'].stop()
            print("   🧹 Cleaned up Spark session")
        except Exception:
            pass  # Ignore errors during cleanup


def get_environment_info(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get human-readable environment information.
    
    Args:
        config: Configuration dictionary from setup_databricks_environment()
        
    Returns:
        Dict with environment details for logging
    """
    info = {
        'environment': config.get('environment', 'unknown'),
        'catalog': config.get('catalog', 'unknown'),
        'schema': config.get('schema', 'unknown'),
        'spark_available': 'Yes' if config.get('spark') else 'No (dry-run mode)',
    }
    
    if config.get('environment') == 'local':
        info['profile'] = os.getenv("DATABRICKS_CONFIG_PROFILE", "aws-apps")
        info['host'] = os.getenv("DATABRICKS_HOST", "not set")
    
    return info