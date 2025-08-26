#!/usr/bin/env python3
"""
MLflow configuration smoke test
Tests MLflow connection to Databricks workspace and Unity Catalog
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mlflow_connection():
    """Test MLflow connection to Databricks"""
    try:
        import mlflow
        from mlflow import MlflowClient
        
        # Configure MLflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Users/alex.miller@databricks.com/mfg-mcp-agent")
        
        print(f"Configuring MLflow:")
        print(f"  Tracking URI: {tracking_uri}")
        print(f"  Registry URI: {registry_uri}")
        print(f"  Experiment: {experiment_name}")
        
        # Set MLflow URIs
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.set_experiment(experiment_name)
            print(f"✓ Connected to experiment: {experiment.name}")
            print(f"  Experiment ID: {experiment.experiment_id}")
        except Exception as e:
            print(f"❌ Could not set experiment: {e}")
            return False
        
        # Test logging a simple run
        print("\nTesting MLflow logging...")
        with mlflow.start_run(run_name=f"smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("test_param", "hello")
            mlflow.log_param("environment", os.getenv("ENVIRONMENT", "dev"))
            
            # Log metrics
            mlflow.log_metric("test_metric", 42.0)
            mlflow.log_metric("accuracy", 0.95)
            
            # Log tags
            mlflow.set_tag("test_type", "smoke_test")
            mlflow.set_tag("project", os.getenv("PROJECT_NAME", "databricks-mfg-mcp-examples"))
            
            run_id = mlflow.active_run().info.run_id
            print(f"✓ Successfully logged run: {run_id}")
        
        # Test Unity Catalog model registry (if available)
        catalog = os.getenv("UC_DEFAULT_CATALOG", "users")
        schema = os.getenv("UC_DEFAULT_SCHEMA", "alex_miller")
        
        print(f"\nTesting Unity Catalog model registry...")
        print(f"  Catalog: {catalog}")
        print(f"  Schema: {schema}")
        
        client = MlflowClient()
        try:
            # List registered models in the catalog
            models = client.search_registered_models(
                filter_string=f"name LIKE '{catalog}.{schema}.%'",
                max_results=5
            )
            print(f"✓ Found {len(models)} models in {catalog}.{schema}")
            for model in models[:3]:  # Show first 3
                print(f"  - {model.name}")
        except Exception as e:
            print(f"Note: Could not list UC models (this is normal if no models exist yet)")
        
        print("\n✓ MLflow configuration successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Error: mlflow not installed: {e}")
        print("Run: pip install 'mlflow[databricks]'")
        return False
    except Exception as e:
        print(f"❌ MLflow connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Databricks CLI is authenticated: databricks auth login --profile aws-apps")
        print("2. Check MLFLOW_TRACKING_URI is set to 'databricks'")
        print("3. Verify you have permissions to create experiments in the workspace")
        return False

def test_mlflow_tracing():
    """Test MLflow 3.0 tracing capabilities"""
    try:
        import mlflow
        from mlflow import MlflowClient
        
        print("\nTesting MLflow 3.0 Tracing...")
        
        # Enable autologging for tracing
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "/Users/alex.miller@databricks.com/mfg-mcp-agent"))
        
        with mlflow.start_run(run_name="tracing_test"):
            # Log a trace (MLflow 3.0 feature)
            with mlflow.start_span(name="test_span") as span:
                span.set_attributes({
                    "test_attribute": "value",
                    "span_type": "test"
                })
                print("✓ MLflow 3.0 tracing is available")
            
        return True
    except AttributeError:
        print("Note: MLflow tracing features not available (requires MLflow 3.0+)")
        return True
    except Exception as e:
        print(f"Warning: Tracing test failed: {e}")
        return True  # Not critical for basic functionality

if __name__ == "__main__":
    print("=" * 60)
    print("MLflow Configuration Smoke Test")
    print("=" * 60)
    
    success = test_mlflow_connection()
    
    if success:
        test_mlflow_tracing()
    
    sys.exit(0 if success else 1)