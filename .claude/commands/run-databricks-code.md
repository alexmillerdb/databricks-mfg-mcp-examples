---
description: Run Python code locally with Databricks Connect or in Databricks notebook environment
argument-hint: [code-type] [optional-file-path]
---

## Why this matters
Enable seamless execution of Databricks code in both local IDE (using environment variables and Databricks Connect) and Databricks notebook environments (using native workspace context). This command automatically detects the environment and configures the appropriate authentication, compute, and MLflow settings.

## Usage Examples
- `run-databricks-code spark` - Run Spark/Delta code with environment detection
- `run-databricks-code mlflow` - Run MLflow logging/tracking code
- `run-databricks-code convert path/to/script.py` - Convert existing file to be environment-agnostic
- `run-databricks-code test` - Run complete environment test

## Core Environment Setup

### Universal Environment Detection
```python
import os

def setup_databricks_environment():
    """Universal setup for local IDE or notebook execution"""
    
    # Detect execution environment
    is_notebook = 'DATABRICKS_RUNTIME_VERSION' in os.environ
    
    if is_notebook:
        print("🟢 Databricks Notebook Environment")
        return setup_notebook_env()
    else:
        print("🔵 Local IDE Environment")
        return setup_local_env()

def setup_notebook_env():
    """Setup for Databricks notebook"""
    from databricks.sdk import WorkspaceClient
    
    return {
        'environment': 'notebook',
        'spark': spark,  # Available globally
        'workspace_client': WorkspaceClient(),
        'catalog': 'manufacturing',  # Update as needed
        'schema': 'supply_chain'     # Update as needed
    }

def setup_local_env():
    """Setup for local IDE"""
    from dotenv import load_dotenv
    from databricks.connect import DatabricksSession
    from databricks.sdk import WorkspaceClient
    import mlflow
    
    # Load environment variables
    load_dotenv()
    
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "aws-apps")
    catalog = os.getenv("UC_DEFAULT_CATALOG", "users")
    schema = os.getenv("UC_DEFAULT_SCHEMA", "alex_miller")
    
    # Initialize Databricks Connect
    spark = DatabricksSession.builder.profile(profile).serverless(True).getOrCreate()
    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"USE SCHEMA {schema}")
    
    # Configure MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "/Users/alex.miller@databricks.com/mfg-mcp-agent"))
    
    return {
        'environment': 'local',
        'spark': spark,
        'workspace_client': WorkspaceClient(profile=profile),
        'catalog': catalog,
        'schema': schema
    }

def cleanup_environment(config):
    """Clean up resources (local only)"""
    if config.get('environment') == 'local' and config.get('spark'):
        config['spark'].stop()
```

## Code Examples

### 1. Spark/Delta Operations
```python
def run_spark_example():
    """Example Spark operations with environment detection"""
    config = setup_databricks_environment()
    
    try:
        spark = config['spark']
        catalog = config['catalog']
        schema = config['schema']
        
        # Test Spark operation
        df = spark.range(1, 1000)
        print(f"✓ Spark test: {df.count()} records")
        
        # Use Unity Catalog variables
        # df = spark.table(f"{catalog}.{schema}.my_table")
        
        print("✅ Spark operations completed")
        return True
        
    except Exception as e:
        print(f"❌ Spark operations failed: {e}")
        return False
    finally:
        cleanup_environment(config)

if __name__ == "__main__":
    run_spark_example()
```

### 2. MLflow Operations
```python
def run_mlflow_example():
    """Example MLflow operations with environment detection"""
    config = setup_databricks_environment()
    
    try:
        import mlflow
        
        # MLflow is automatically configured by setup
        with mlflow.start_run() as run:
            mlflow.log_param("environment", config['environment'])
            mlflow.log_metric("test_metric", 0.95)
            
        print(f"✓ MLflow run: {run.info.run_id}")
        print("✅ MLflow operations completed")
        return True
        
    except Exception as e:
        print(f"❌ MLflow operations failed: {e}")
        return False
    finally:
        cleanup_environment(config)

if __name__ == "__main__":
    run_mlflow_example()
```

### 3. Complete Environment Test
```python
def test_environment():
    """Test complete Databricks environment setup"""
    
    print("=" * 60)
    print("🚀 DATABRICKS ENVIRONMENT TEST")
    print("=" * 60)
    
    config = setup_databricks_environment()
    
    if not config:
        print("❌ Environment setup failed")
        return False
    
    results = {}
    
    # Test Spark
    try:
        df = config['spark'].range(1, 10)
        count = df.count()
        print(f"✓ Spark: {count} records")
        results['spark'] = True
    except Exception as e:
        print(f"❌ Spark failed: {e}")
        results['spark'] = False
    
    # Test MLflow
    try:
        import mlflow
        with mlflow.start_run() as run:
            mlflow.log_metric("test", 1.0)
        print(f"✓ MLflow: {run.info.run_id}")
        results['mlflow'] = True
    except Exception as e:
        print(f"❌ MLflow failed: {e}")
        results['mlflow'] = False
    
    # Test Vector Search (optional)
    try:
        from databricks_langchain import VectorSearchRetrieverTool
        print("✓ Vector Search libraries available")
        results['vector_search'] = True
    except ImportError:
        print("⚠️ Vector Search libraries not installed")
        results['vector_search'] = False
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print("-" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test.upper():15} {status}")
    
    cleanup_environment(config)
    return passed == total

if __name__ == "__main__":
    test_environment()
```

## Usage

### Local IDE
```bash
# Test environment
claude run-databricks-code test

# Run specific examples
claude run-databricks-code spark
claude run-databricks-code mlflow

# Convert existing file
claude run-databricks-code convert path/to/file.py
```

### Databricks Notebook
```python
# Copy any of the example code above directly into notebook cells
# Environment detection will automatically use notebook context
```

## Key Benefits

1. **Automatic Environment Detection**: No manual configuration needed
2. **Unified Code**: Same code works locally and in notebooks  
3. **Proper Authentication**: CLI profiles locally, native auth in notebooks
4. **MLflow Integration**: Automatic tracking URI configuration
5. **Unity Catalog**: Dynamic catalog/schema handling
6. **Error Handling**: Graceful fallbacks and cleanup

This enables true hybrid development where you can prototype locally with full IDE features and seamlessly transition to notebook-based development.
