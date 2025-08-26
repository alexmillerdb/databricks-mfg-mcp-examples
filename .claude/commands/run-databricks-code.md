---
description: Run Python code locally with Databricks Connect or in Databricks notebook environment
argument-hint: [code-type] [optional-file-path]
---

## Why this matters
Enable seamless execution of Databricks code in both local IDE (using environment variables and Databricks Connect) and Databricks notebook environments (using native workspace context). This command automatically detects the environment and configures the appropriate authentication, compute, and MLflow settings for consistent development across local and remote contexts.

## Usage Examples
- `run-databricks-code spark` - Run Spark/Delta code with environment detection
- `run-databricks-code mlflow` - Run MLflow logging/tracking code
- `run-databricks-code vector-search` - Run vector search operations
- `run-databricks-code mcp` - Run MCP server integration code
- `run-databricks-code agent` - Run agent development/testing code
- `run-databricks-code full-test path/to/script.py` - Run complete test from file

## Environment Detection & Configuration

### Local IDE Environment
When running locally, the command will:
1. Load environment variables from `.env` file
2. Use Databricks CLI profile authentication
3. Initialize Databricks Connect for Spark operations
4. Configure MLflow tracking to workspace
5. Set up proper Unity Catalog context

### Databricks Notebook Environment
When running in Databricks notebooks, the command will:
1. Use native workspace authentication (no env vars needed)
2. Use existing Spark session
3. Use native MLflow integration
4. Access Unity Catalog directly

## Code Templates by Type

### 1. Spark/Delta Operations
```python
# Environment detection and setup
import os
import sys

def setup_databricks_environment():
    """Setup environment for local or notebook execution"""
    
    # Detect if running in Databricks notebook
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        print("🟢 Running in Databricks notebook environment")
        
        # Use existing Spark session in notebook
        spark = spark  # Available globally in notebooks
        
        # Set Unity Catalog context (optional, usually set by notebook)
        catalog = "manufacturing"  # Default for this project
        schema = "supply_chain"
        
        try:
            spark.sql(f"USE CATALOG {catalog}")
            spark.sql(f"USE SCHEMA {schema}")
            print(f"✓ Using Unity Catalog: {catalog}.{schema}")
        except Exception as e:
            print(f"Note: Could not set catalog/schema: {e}")
            
        return spark, catalog, schema
    
    else:
        print("🔵 Running in local IDE environment")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize Databricks Connect
        from databricks.connect import DatabricksSession
        
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "aws-apps")
        serverless_id = os.getenv("DATABRICKS_SERVERLESS_COMPUTE_ID", "auto")
        catalog = os.getenv("UC_DEFAULT_CATALOG", "users")
        schema = os.getenv("UC_DEFAULT_SCHEMA", "alex_miller")
        
        print(f"Using profile: {profile}")
        print(f"Using serverless compute: {serverless_id}")
        
        # Create Spark session
        spark = DatabricksSession.builder \
            .profile(profile) \
            .serverless(True) \
            .getOrCreate()
        
        # Set Unity Catalog context
        try:
            spark.sql(f"USE CATALOG {catalog}")
            spark.sql(f"USE SCHEMA {schema}")
            print(f"✓ Connected to Unity Catalog: {catalog}.{schema}")
        except Exception as e:
            print(f"Warning: Could not set catalog/schema: {e}")
        
        return spark, catalog, schema

# Example Spark/Delta operations
def test_spark_operations():
    """Test basic Spark and Delta operations"""
    spark, catalog, schema = setup_databricks_environment()
    
    try:
        # Test basic Spark operation
        print("\n📊 Testing Spark operations...")
        df = spark.range(1, 1000)
        count = df.count()
        print(f"✓ Spark test successful: {count} records")
        
        # Test Delta table operations (if tables exist)
        try:
            tables = spark.sql("SHOW TABLES").collect()
            print(f"✓ Found {len(tables)} tables in {catalog}.{schema}")
            
            # Example: Read from a Delta table (adjust table name as needed)
            # df = spark.table(f"{catalog}.{schema}.inventory")
            # print(f"✓ Successfully read Delta table: {df.count()} rows")
            
        except Exception as e:
            print(f"Note: Could not access Delta tables: {e}")
        
        print("✅ Spark/Delta operations completed successfully")
        
    except Exception as e:
        print(f"❌ Spark operations failed: {e}")
        return False
    
    finally:
        # Clean up Spark session (only in local mode)
        if 'DATABRICKS_RUNTIME_VERSION' not in os.environ:
            spark.stop()
    
    return True

# Run the test
if __name__ == "__main__":
    test_spark_operations()
```

### 2. MLflow Operations
```python
# MLflow setup for local and notebook environments
import os
import mlflow

def setup_mlflow_environment():
    """Setup MLflow for local or notebook execution"""
    
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        print("🟢 MLflow: Using Databricks notebook environment")
        
        # MLflow is pre-configured in notebooks
        # Set experiment if needed
        experiment_name = "/Users/alex.miller@databricks.com/mfg-mcp-agent"
        
    else:
        print("🔵 MLflow: Using local IDE environment")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Configure MLflow for workspace tracking
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Users/alex.miller@databricks.com/mfg-mcp-agent")
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        
        print(f"✓ MLflow tracking URI: {tracking_uri}")
        print(f"✓ MLflow registry URI: {registry_uri}")
    
    # Set or create experiment
    try:
        mlflow.set_experiment(experiment_name)
        print(f"✓ Using MLflow experiment: {experiment_name}")
    except Exception as e:
        print(f"Warning: Could not set experiment: {e}")
    
    return experiment_name

def test_mlflow_operations():
    """Test MLflow logging and tracking"""
    experiment_name = setup_mlflow_environment()
    
    try:
        print("\n📈 Testing MLflow operations...")
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("environment", "local" if 'DATABRICKS_RUNTIME_VERSION' not in os.environ else "notebook")
            mlflow.log_param("test_type", "mlflow_operations")
            
            # Log metrics
            mlflow.log_metric("test_metric", 0.95)
            mlflow.log_metric("accuracy", 0.87)
            
            # Log artifact (simple text file)
            with open("test_artifact.txt", "w") as f:
                f.write("This is a test artifact from MLflow operations")
            mlflow.log_artifact("test_artifact.txt")
            
            print(f"✓ MLflow run completed: {run.info.run_id}")
            print(f"✓ Run URL: {mlflow.get_tracking_uri()}/#/experiments/{mlflow.get_experiment_by_name(experiment_name).experiment_id}/runs/{run.info.run_id}")
        
        # Clean up
        if os.path.exists("test_artifact.txt"):
            os.remove("test_artifact.txt")
        
        print("✅ MLflow operations completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ MLflow operations failed: {e}")
        return False

# Run the test
if __name__ == "__main__":
    test_mlflow_operations()
```

### 3. Vector Search Operations
```python
# Vector Search setup for local and notebook environments
import os

def setup_vector_search_environment():
    """Setup Vector Search for local or notebook execution"""
    
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        print("🟢 Vector Search: Using Databricks notebook environment")
        
        # Use native workspace client in notebooks
        from databricks.sdk import WorkspaceClient
        workspace_client = WorkspaceClient()
        
    else:
        print("🔵 Vector Search: Using local IDE environment")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize workspace client with profile
        from databricks.sdk import WorkspaceClient
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "aws-apps")
        
        workspace_client = WorkspaceClient(profile=profile)
        print(f"✓ Using Databricks profile: {profile}")
    
    return workspace_client

def test_vector_search_operations():
    """Test Vector Search and LangChain integration"""
    workspace_client = setup_vector_search_environment()
    
    try:
        print("\n🔍 Testing Vector Search operations...")
        
        # Test VectorSearchRetrieverTool (if indexes exist)
        try:
            from databricks_langchain import VectorSearchRetrieverTool
            
            # Example vector search tool (adjust index name as needed)
            vs_tool = VectorSearchRetrieverTool(
                index_name="manufacturing.supply_chain.sop_index",  # Adjust as needed
                tool_name="test_retriever",
                tool_description="Test vector search retriever",
                num_results=3,
                workspace_client=workspace_client
            )
            
            # Test query (this will fail if index doesn't exist, which is expected)
            try:
                results = vs_tool.invoke("test query")
                print(f"✓ Vector search successful: {len(results)} results")
            except Exception as e:
                print(f"Note: Vector search test skipped (index may not exist): {e}")
            
        except ImportError:
            print("Note: databricks-langchain not installed, skipping vector search test")
        
        # Test MCP integration (if available)
        try:
            from databricks_mcp import DatabricksMCPClient
            
            host = workspace_client.config.host
            mcp_server_url = f"{host}/api/2.0/mcp/vector-search/manufacturing/supply_chain"
            
            mcp_client = DatabricksMCPClient(
                server_url=mcp_server_url,
                workspace_client=workspace_client
            )
            
            # Test tool discovery
            try:
                tools = mcp_client.list_tools()
                print(f"✓ MCP server connected: {len(tools)} tools discovered")
            except Exception as e:
                print(f"Note: MCP server test skipped: {e}")
                
        except ImportError:
            print("Note: databricks-mcp not installed, skipping MCP test")
        
        print("✅ Vector Search operations completed")
        return True
        
    except Exception as e:
        print(f"❌ Vector Search operations failed: {e}")
        return False

# Run the test
if __name__ == "__main__":
    test_vector_search_operations()
```

### 4. Agent Development Operations
```python
# Agent development setup for local and notebook environments
import os
from uuid import uuid4

def setup_agent_environment():
    """Setup agent development environment"""
    
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        print("🟢 Agent Development: Using Databricks notebook environment")
        
        from databricks.sdk import WorkspaceClient
        workspace_client = WorkspaceClient()
        
    else:
        print("🔵 Agent Development: Using local IDE environment")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        from databricks.sdk import WorkspaceClient
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "aws-apps")
        workspace_client = WorkspaceClient(profile=profile)
        
        print(f"✓ Using Databricks profile: {profile}")
    
    return workspace_client

def test_agent_operations():
    """Test ResponsesAgent and related operations"""
    workspace_client = setup_agent_environment()
    
    try:
        print("\n🤖 Testing Agent operations...")
        
        # Test ResponsesAgent interface
        try:
            from mlflow.pyfunc import ResponsesAgent
            from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
            
            class TestAgent(ResponsesAgent):
                def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
                    # Simple test implementation
                    user_message = request.input[-1].content if request.input else "No message"
                    
                    output_item = self.create_text_output_item(
                        text=f"Test response to: {user_message}",
                        id=str(uuid4())
                    )
                    
                    return ResponsesAgentResponse(output=[output_item])
            
            # Test agent
            agent = TestAgent()
            
            # Create test request
            test_request = ResponsesAgentRequest(
                input=[{"role": "user", "content": "Hello, test agent!"}]
            )
            
            response = agent.predict(test_request)
            print(f"✓ ResponsesAgent test successful: {response.output[0].content[0].text}")
            
        except ImportError as e:
            print(f"Note: MLflow ResponsesAgent not available: {e}")
        
        # Test LLM endpoint connection
        try:
            from databricks_langchain import ChatDatabricks
            
            llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")
            
            # Simple test (this may fail if endpoint doesn't exist)
            try:
                response = llm.invoke("Hello, this is a test message.")
                print(f"✓ LLM endpoint test successful: {response.content[:50]}...")
            except Exception as e:
                print(f"Note: LLM endpoint test skipped: {e}")
                
        except ImportError:
            print("Note: databricks-langchain not installed, skipping LLM test")
        
        print("✅ Agent operations completed")
        return True
        
    except Exception as e:
        print(f"❌ Agent operations failed: {e}")
        return False

# Run the test
if __name__ == "__main__":
    test_agent_operations()
```

### 5. Complete Integration Test
```python
# Complete integration test combining all components
import os
import sys

def run_complete_integration_test():
    """Run complete integration test of all Databricks components"""
    
    print("=" * 80)
    print("🚀 DATABRICKS COMPLETE INTEGRATION TEST")
    print("=" * 80)
    
    # Detect environment
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
        print("🟢 Environment: Databricks Notebook")
        print(f"Runtime Version: {os.environ['DATABRICKS_RUNTIME_VERSION']}")
    else:
        print("🔵 Environment: Local IDE")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "aws-apps")
        print(f"Using CLI Profile: {profile}")
    
    print("-" * 80)
    
    # Track test results
    results = {}
    
    # Test 1: Spark/Delta Operations
    print("\n1️⃣ Testing Spark/Delta Operations...")
    try:
        exec(open('spark_test.py').read()) if os.path.exists('spark_test.py') else exec("""
# Inline Spark test
spark, catalog, schema = setup_databricks_environment()
df = spark.range(1, 100)
count = df.count()
print(f"✓ Spark test: {count} records")
if 'DATABRICKS_RUNTIME_VERSION' not in os.environ:
    spark.stop()
results['spark'] = True
""")
    except Exception as e:
        print(f"❌ Spark test failed: {e}")
        results['spark'] = False
    
    # Test 2: MLflow Operations
    print("\n2️⃣ Testing MLflow Operations...")
    try:
        import mlflow
        
        # Setup MLflow
        if 'DATABRICKS_RUNTIME_VERSION' not in os.environ:
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
        
        # Quick MLflow test
        with mlflow.start_run() as run:
            mlflow.log_param("test_param", "integration_test")
            mlflow.log_metric("test_metric", 1.0)
        
        print(f"✓ MLflow test successful: {run.info.run_id}")
        results['mlflow'] = True
        
    except Exception as e:
        print(f"❌ MLflow test failed: {e}")
        results['mlflow'] = False
    
    # Test 3: Vector Search (if available)
    print("\n3️⃣ Testing Vector Search...")
    try:
        from databricks_langchain import VectorSearchRetrieverTool
        print("✓ Vector Search libraries available")
        results['vector_search'] = True
    except ImportError:
        print("⚠️ Vector Search libraries not installed")
        results['vector_search'] = False
    
    # Test 4: MCP Integration (if available)
    print("\n4️⃣ Testing MCP Integration...")
    try:
        from databricks_mcp import DatabricksMCPClient
        print("✓ MCP libraries available")
        results['mcp'] = True
    except ImportError:
        print("⚠️ MCP libraries not installed")
        results['mcp'] = False
    
    # Test 5: Agent Framework (if available)
    print("\n5️⃣ Testing Agent Framework...")
    try:
        from mlflow.pyfunc import ResponsesAgent
        from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
        print("✓ Agent Framework available")
        results['agents'] = True
    except ImportError:
        print("⚠️ Agent Framework not available")
        results['agents'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name.upper():20} {status}")
    
    print("-" * 80)
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Environment is ready for development.")
    else:
        print("⚠️ Some tests failed. Check installation and configuration.")
    
    return passed == total

# Run complete test
if __name__ == "__main__":
    success = run_complete_integration_test()
    sys.exit(0 if success else 1)
```

## Command Execution Logic

The command will detect the execution context and run the appropriate code:

1. **Environment Detection**: Check for `DATABRICKS_RUNTIME_VERSION` to determine if running in notebook
2. **Authentication Setup**: Use CLI profiles locally, native auth in notebooks  
3. **Compute Configuration**: Databricks Connect locally, existing Spark session in notebooks
4. **MLflow Setup**: Configure tracking URIs locally, use native integration in notebooks
5. **Error Handling**: Graceful fallbacks and informative error messages

## Usage in Practice

### Local IDE Development
```bash
# Test Spark operations locally
claude run-databricks-code spark

# Test MLflow integration locally  
claude run-databricks-code mlflow

# Run complete integration test
claude run-databricks-code full-test
```

### Databricks Notebook
```python
# Copy and paste the generated code directly into notebook cells
# No environment variables or authentication setup needed
# Code will automatically detect notebook environment and use native integrations
```

## Benefits

1. **Seamless Development**: Same code works in both local and notebook environments
2. **Automatic Configuration**: Environment detection handles setup differences
3. **Best Practices**: Uses recommended authentication and configuration patterns
4. **Comprehensive Testing**: Covers all major Databricks components
5. **Error Resilience**: Graceful handling of missing dependencies or configuration issues

This command enables true hybrid development where you can prototype locally with full IDE features and then seamlessly transition to notebook-based development and deployment.
