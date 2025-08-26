#!/usr/bin/env python3
"""
Databricks Connect smoke test
Tests connection to Databricks using profile-based authentication
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_databricks_connect():
    """Test Databricks Connect with serverless compute"""
    try:
        from databricks.connect import DatabricksSession
        
        # Get configuration from environment
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "aws-apps")
        serverless_id = os.getenv("DATABRICKS_SERVERLESS_COMPUTE_ID", "auto")
        
        print(f"Testing Databricks Connect with profile: {profile}")
        print(f"Using serverless compute: {serverless_id}")
        
        # Create Spark session using serverless compute
        spark = DatabricksSession.builder \
            .profile(profile) \
            .serverless(True) \
            .getOrCreate()
        
        # Test the connection
        print("\nTesting Spark connection...")
        df = spark.range(1, 10)
        count = df.count()
        print(f"✓ Successfully connected! Test count: {count}")
        
        # Show Spark version
        print(f"Spark version: {spark.version}")
        
        # Test catalog access if Unity Catalog is configured
        catalog = os.getenv("UC_DEFAULT_CATALOG", "users")
        schema = os.getenv("UC_DEFAULT_SCHEMA", "alex_miller")
        
        try:
            spark.sql(f"USE CATALOG {catalog}")
            spark.sql(f"USE SCHEMA {schema}")
            print(f"✓ Successfully set Unity Catalog: {catalog}.{schema}")
            
            # List tables in the schema
            tables = spark.sql("SHOW TABLES").collect()
            print(f"Tables in {catalog}.{schema}: {len(tables)} found")
        except Exception as e:
            print(f"Note: Could not set catalog/schema: {e}")
        
        spark.stop()
        return True
        
    except ImportError as e:
        print(f"❌ Error: databricks-connect not installed: {e}")
        print("Run: pip install databricks-connect")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you've authenticated: databricks auth login --profile aws-apps")
        print("2. Check your .env file has correct DATABRICKS_CONFIG_PROFILE")
        print("3. Verify profile with: databricks current-user me --profile aws-apps")
        return False

def test_databricks_connect_with_cluster():
    """Alternative test using a specific cluster (if serverless is not available)"""
    try:
        from databricks.connect import DatabricksSession
        
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE", "aws-apps")
        cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")
        
        if not cluster_id:
            print("Note: DATABRICKS_CLUSTER_ID not set, skipping cluster-based test")
            return True
        
        print(f"\nTesting cluster-based connection with cluster: {cluster_id}")
        
        spark = DatabricksSession.builder \
            .profile(profile) \
            .clusterId(cluster_id) \
            .getOrCreate()
        
        df = spark.range(1, 5)
        count = df.count()
        print(f"✓ Cluster connection successful! Count: {count}")
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"❌ Cluster connection failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Databricks Connect Smoke Test")
    print("=" * 60)
    
    success = test_databricks_connect()
    
    # Optionally test cluster-based connection
    if success and os.getenv("DATABRICKS_CLUSTER_ID"):
        test_databricks_connect_with_cluster()
    
    sys.exit(0 if success else 1)