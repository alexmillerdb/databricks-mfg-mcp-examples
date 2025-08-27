#!/usr/bin/env python3
"""
Unity Catalog Setup Script for Manufacturing MCP Examples
Phase 1.2: Create Unity Catalog Structure

This script sets up the required Unity Catalog structure including:
- Manufacturing catalog
- Domain-specific schemas (supply_chain, sales, iot, support)
- Proper permissions for multi-tenant access

Can be run from Databricks notebooks or locally via Databricks Connect.
"""

import os
import sys
from typing import Optional, List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import utilities from utils package
from utils import setup_databricks_environment, cleanup_environment, execute_sql


def create_catalog(spark, catalog_name: str = "mfg_mcp_demo") -> bool:
    """
    Create the main manufacturing catalog.
    """
    sql = f"CREATE CATALOG IF NOT EXISTS {catalog_name}"
    return execute_sql(
        spark, 
        sql, 
        f"Creating catalog: {catalog_name}"
    )


def create_schemas(spark, catalog_name: str = "mfg_mcp_demo") -> Dict[str, bool]:
    """
    Create domain-specific schemas within the catalog.
    """
    schemas = [
        ("supply_chain", "Supply chain management and logistics"),
        ("sales", "Sales and customer management"),
        ("iot", "IoT sensor and telemetry data"),
        ("support", "Customer support and ticketing"),
        ("agent_logs", "Agent execution logs and monitoring")
    ]
    
    results = {}
    
    for schema_name, comment in schemas:
        sql = f"""
        CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}
        COMMENT '{comment}'
        """
        results[schema_name] = execute_sql(
            spark,
            sql,
            f"Creating schema: {catalog_name}.{schema_name}"
        )
    
    return results


def grant_permissions(spark, catalog_name: str = "mfg_mcp_demo") -> Dict[str, bool]:
    """
    Grant appropriate permissions for multi-tenant access.
    
    Note: In production, you would grant more specific permissions to different groups.
    This is a simplified version for the demo.
    """
    permissions = [
        # Grant catalog usage to all users (read-only by default)
        (f"GRANT USAGE ON CATALOG {catalog_name} TO `account users`",
         "Granting catalog USAGE to all users"),
        
        # Grant schema usage for different domains
        (f"GRANT USE SCHEMA ON SCHEMA {catalog_name}.supply_chain TO `account users`",
         "Granting supply_chain schema usage"),
        (f"GRANT USE SCHEMA ON SCHEMA {catalog_name}.sales TO `account users`",
         "Granting sales schema usage"),
        (f"GRANT USE SCHEMA ON SCHEMA {catalog_name}.iot TO `account users`",
         "Granting iot schema usage"),
        (f"GRANT USE SCHEMA ON SCHEMA {catalog_name}.support TO `account users`",
         "Granting support schema usage"),
        
        # Grant CREATE permissions for tables/functions in schemas
        (f"GRANT CREATE TABLE ON SCHEMA {catalog_name}.supply_chain TO `account users`",
         "Granting CREATE TABLE on supply_chain"),
        (f"GRANT CREATE TABLE ON SCHEMA {catalog_name}.sales TO `account users`",
         "Granting CREATE TABLE on sales"),
        (f"GRANT CREATE TABLE ON SCHEMA {catalog_name}.iot TO `account users`",
         "Granting CREATE TABLE on iot"),
        (f"GRANT CREATE TABLE ON SCHEMA {catalog_name}.support TO `account users`",
         "Granting CREATE TABLE on support"),
        
        # Grant CREATE FUNCTION permissions for UC functions
        (f"GRANT CREATE FUNCTION ON SCHEMA {catalog_name}.supply_chain TO `account users`",
         "Granting CREATE FUNCTION on supply_chain"),
        (f"GRANT CREATE FUNCTION ON SCHEMA {catalog_name}.sales TO `account users`",
         "Granting CREATE FUNCTION on sales"),
    ]
    
    results = {}
    
    for sql, description in permissions:
        # Use the SQL as the key for results
        results[description] = execute_sql(spark, sql, description)
    
    return results


def verify_setup(spark, catalog_name: str = "mfg_mcp_demo") -> bool:
    """
    Verify that the Unity Catalog structure was created successfully.
    """
    print("\n🔍 Verifying Unity Catalog setup...")
    
    try:
        # Check if catalog exists
        catalogs_df = spark.sql("SHOW CATALOGS")
        catalogs = [row.catalog for row in catalogs_df.collect()]
        
        if catalog_name not in catalogs:
            print(f"   ❌ Catalog '{catalog_name}' not found")
            return False
        
        print(f"   ✅ Catalog '{catalog_name}' exists")
        
        # Check schemas
        schemas_df = spark.sql(f"SHOW SCHEMAS IN {catalog_name}")
        schemas = [row.databaseName for row in schemas_df.collect()]
        
        expected_schemas = ["supply_chain", "sales", "iot", "support", "agent_logs"]
        missing_schemas = [s for s in expected_schemas if s not in schemas]
        
        if missing_schemas:
            print(f"   ❌ Missing schemas: {missing_schemas}")
            return False
        
        print(f"   ✅ All schemas created: {expected_schemas}")
        
        # Set the catalog as current
        spark.sql(f"USE CATALOG {catalog_name}")
        print(f"   ✅ Set current catalog to '{catalog_name}'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False


def main():
    """
    Main execution function to set up Unity Catalog structure.
    """
    print("=" * 60)
    print("Unity Catalog Setup for Manufacturing MCP Examples")
    print("Phase 1.2: Creating Unity Catalog Structure")
    print("=" * 60)
    
    # Setup environment (universal for notebook or local)
    config = setup_databricks_environment()
    spark = config.get('spark')
    catalog_name = config.get('catalog', 'manufacturing')
    
    if not spark:
        print("\n⚠️  No Spark session available. Running in dry-run mode.")
        print("   SQL statements will be printed but not executed.")
    
    print(f"\n📦 Target catalog: {catalog_name}")
    print("-" * 40)
    
    # Create catalog
    success = create_catalog(spark, catalog_name)
    if not success and spark:
        print("\n❌ Failed to create catalog. Exiting.")
        return 1
    
    # Create schemas
    print(f"\n📁 Creating schemas in {catalog_name}...")
    print("-" * 40)
    schema_results = create_schemas(spark, catalog_name)
    
    failed_schemas = [s for s, result in schema_results.items() if not result]
    if failed_schemas and spark:
        print(f"\n❌ Failed to create schemas: {failed_schemas}")
        return 1
    
    # Grant permissions
    print(f"\n🔐 Granting permissions...")
    print("-" * 40)
    permission_results = grant_permissions(spark, catalog_name)
    
    failed_permissions = [p for p, result in permission_results.items() if not result]
    if failed_permissions:
        print(f"\n⚠️  Some permissions could not be granted: {failed_permissions}")
        print("   This might be expected if you don't have admin privileges.")
    
    # Verify setup
    if spark:
        verification_success = verify_setup(spark, catalog_name)
        
        if verification_success:
            print("\n✅ Unity Catalog setup completed successfully!")
            print(f"\n📊 Summary:")
            print(f"   - Catalog: {catalog_name}")
            print(f"   - Schemas: {len(schema_results)} created")
            print(f"   - Permissions: {len(permission_results) - len(failed_permissions)} granted")
        else:
            print("\n⚠️  Setup completed with warnings. Please review the output above.")
            return 1
    else:
        print("\n✅ Dry-run completed. Run with Spark session to execute.")
    
    print("\n" + "=" * 60)
    print("Next step: Run create_delta_tables.py to create the data tables")
    print("=" * 60)
    
    # Clean up resources if running locally
    cleanup_environment(config)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())