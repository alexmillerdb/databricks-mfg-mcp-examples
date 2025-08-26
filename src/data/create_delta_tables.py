#!/usr/bin/env python3
"""
Delta Tables Creation Script for Manufacturing MCP Examples
Phase 2.1: Create Delta Tables

This script creates all required Delta tables for the manufacturing demo:
- Supply Chain: inventory, shipments, suppliers, standard_operating_procedures, incident_reports
- Sales: transactions, customers, sales_proposals  
- IoT: telemetry
- Support: tickets, support_tickets

Each table includes proper schema definition, partitioning, and optimization settings.
Can be run from Databricks notebooks or locally via Databricks Connect.
"""

import os
import sys
from typing import Dict, Tuple


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
    
    # In notebook, spark is available globally
    global spark  # Declare global to avoid undefined variable warning
    return {
        'environment': 'notebook',
        'spark': spark,  # Available globally in notebooks
        'workspace_client': WorkspaceClient(),
        'catalog': os.getenv('UC_DEFAULT_CATALOG', 'mfg_mcp_demo'),
        'schema': 'supply_chain'  # Default schema
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
    catalog = os.getenv("UC_DEFAULT_CATALOG", "mfg_mcp_demo")
    schema = os.getenv("UC_DEFAULT_SCHEMA", "supply_chain")
    
    try:
        # Initialize Databricks Connect
        spark = DatabricksSession.builder.profile(profile).serverless(True).getOrCreate()
        
        # Set catalog context only (no schema since we use fully qualified table names)
        spark.sql(f"USE CATALOG {catalog}")
        
        # Configure MLflow (optional for this script but good to have)
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
        print("   Running in dry-run mode (SQL statements will be printed but not executed)")
        return {
            'environment': 'local',
            'spark': None,  # Dry-run mode
            'workspace_client': None,
            'catalog': catalog,
            'schema': schema
        }


def cleanup_environment(config):
    """Clean up resources (local only)"""
    if config.get('environment') == 'local' and config.get('spark'):
        try:
            config['spark'].stop()
        except:
            pass  # Ignore errors during cleanup


def execute_sql(spark, sql: str, description: str = None) -> bool:
    """
    Execute SQL statement with error handling and logging.
    """
    try:
        if description:
            print(f"📋 {description}")
        
        if spark:
            result = spark.sql(sql)
            if result is not None and hasattr(result, 'collect'):
                result.collect()
        else:
            print(f"   SQL: {sql[:200]}...")  # Show first 200 chars in dry-run
        
        print("   ✅ Success")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg.lower():
            print("   ℹ️  Table already exists (skipping)")
            return True
        else:
            print(f"   ❌ Error: {error_msg}")
            return False


def create_supply_chain_tables(spark, catalog: str = "mfg_mcp_demo") -> Dict[str, bool]:
    """
    Create supply chain domain tables.
    """
    results = {}
    
    # Inventory table
    inventory_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.supply_chain.inventory (
        inventory_id STRING COMMENT 'Unique identifier for inventory record',
        part_id STRING COMMENT 'Part identifier',
        part_name STRING COMMENT 'Human-readable part name',
        warehouse_id STRING COMMENT 'Warehouse identifier',
        warehouse_location STRING COMMENT 'Warehouse location/city',
        current_quantity INT COMMENT 'Current inventory count',
        reorder_level INT COMMENT 'Minimum quantity before reorder',
        reorder_quantity INT COMMENT 'Standard reorder quantity',
        unit_cost DOUBLE COMMENT 'Cost per unit in USD',
        last_updated TIMESTAMP COMMENT 'Last inventory update time',
        category STRING COMMENT 'Part category',
        supplier_id STRING COMMENT 'Primary supplier identifier'
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true'
    )
    COMMENT 'Current inventory levels across all warehouses'
    """
    results['inventory'] = execute_sql(spark, inventory_sql, 
                                      f"Creating {catalog}.supply_chain.inventory")
    
    # Shipments table
    shipments_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.supply_chain.shipments (
        shipment_id STRING COMMENT 'Unique shipment identifier',
        order_id STRING COMMENT 'Associated order ID',
        part_id STRING COMMENT 'Part being shipped',
        part_name STRING COMMENT 'Part name',
        quantity INT COMMENT 'Quantity being shipped',
        supplier_id STRING COMMENT 'Supplier identifier',
        supplier_name STRING COMMENT 'Supplier name',
        origin_location STRING COMMENT 'Shipment origin',
        destination_location STRING COMMENT 'Shipment destination',
        shipment_date DATE COMMENT 'Date shipped',
        expected_delivery DATE COMMENT 'Expected delivery date',
        actual_delivery DATE COMMENT 'Actual delivery date',
        status STRING COMMENT 'Current shipment status',
        carrier STRING COMMENT 'Shipping carrier',
        tracking_number STRING COMMENT 'Carrier tracking number',
        cost DOUBLE COMMENT 'Shipping cost in USD'
    )
    USING DELTA
    PARTITIONED BY (shipment_date)
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.autoOptimize.optimizeWrite' = 'true'
    )
    COMMENT 'Shipment tracking and logistics data'
    """
    results['shipments'] = execute_sql(spark, shipments_sql,
                                       f"Creating {catalog}.supply_chain.shipments")
    
    # Suppliers table
    suppliers_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.supply_chain.suppliers (
        supplier_id STRING COMMENT 'Unique supplier identifier',
        supplier_name STRING COMMENT 'Supplier company name',
        contact_name STRING COMMENT 'Primary contact person',
        contact_email STRING COMMENT 'Contact email address',
        contact_phone STRING COMMENT 'Contact phone number',
        address STRING COMMENT 'Supplier address',
        city STRING COMMENT 'Supplier city',
        country STRING COMMENT 'Supplier country',
        rating DOUBLE COMMENT 'Supplier rating (0-5)',
        lead_time_days INT COMMENT 'Average lead time in days',
        payment_terms STRING COMMENT 'Payment terms',
        status STRING COMMENT 'Supplier status (ACTIVE/INACTIVE)',
        created_date DATE COMMENT 'Supplier onboarding date',
        last_order_date DATE COMMENT 'Date of last order'
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    )
    COMMENT 'Supplier master data and relationships'
    """
    results['suppliers'] = execute_sql(spark, suppliers_sql,
                                       f"Creating {catalog}.supply_chain.suppliers")
    
    # Standard Operating Procedures table (for Vector Search)
    sop_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.supply_chain.standard_operating_procedures (
        id STRING COMMENT 'Unique SOP identifier',
        procedure_name STRING COMMENT 'Name of the procedure',
        procedure_text STRING COMMENT 'Full procedure text content',
        category STRING COMMENT 'Procedure category',
        department STRING COMMENT 'Responsible department',
        version STRING COMMENT 'SOP version number',
        effective_date DATE COMMENT 'Effective from date',
        last_reviewed DATE COMMENT 'Last review date',
        next_review DATE COMMENT 'Next scheduled review',
        created_by STRING COMMENT 'Author of the SOP',
        tags ARRAY<STRING> COMMENT 'Searchable tags'
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    )
    COMMENT 'Standard operating procedures for supply chain processes'
    """
    results['standard_operating_procedures'] = execute_sql(spark, sop_sql,
        f"Creating {catalog}.supply_chain.standard_operating_procedures")
    
    # Incident Reports table (for Vector Search)
    incident_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.supply_chain.incident_reports (
        id STRING COMMENT 'Unique incident identifier',
        incident_title STRING COMMENT 'Brief incident title',
        incident_description STRING COMMENT 'Detailed incident description',
        incident_date TIMESTAMP COMMENT 'When incident occurred',
        severity STRING COMMENT 'Severity level (LOW/MEDIUM/HIGH/CRITICAL)',
        category STRING COMMENT 'Incident category',
        affected_systems ARRAY<STRING> COMMENT 'Systems affected',
        resolution STRING COMMENT 'How the incident was resolved',
        resolution_time_hours DOUBLE COMMENT 'Time to resolution in hours',
        root_cause STRING COMMENT 'Root cause analysis',
        preventive_measures STRING COMMENT 'Measures to prevent recurrence',
        reported_by STRING COMMENT 'Person who reported',
        resolved_by STRING COMMENT 'Person who resolved'
    )
    USING DELTA
    PARTITIONED BY (incident_date)
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    )
    COMMENT 'Historical incident reports and resolutions'
    """
    results['incident_reports'] = execute_sql(spark, incident_sql,
        f"Creating {catalog}.supply_chain.incident_reports")
    
    return results


def create_sales_tables(spark, catalog: str = "mfg_mcp_demo") -> Dict[str, bool]:
    """
    Create sales domain tables.
    """
    results = {}
    
    # Customers table
    customers_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.sales.customers (
        customer_id STRING COMMENT 'Unique customer identifier',
        customer_name STRING COMMENT 'Customer company name',
        contact_name STRING COMMENT 'Primary contact person',
        email STRING COMMENT 'Contact email address',
        phone STRING COMMENT 'Contact phone number',
        industry STRING COMMENT 'Customer industry',
        segment STRING COMMENT 'Customer segment (Enterprise/SMB/Startup)',
        address STRING COMMENT 'Customer address',
        city STRING COMMENT 'Customer city',
        country STRING COMMENT 'Customer country',
        created_date DATE COMMENT 'Account creation date',
        status STRING COMMENT 'Customer status (ACTIVE/INACTIVE/CHURNED)',
        total_orders INT COMMENT 'Total number of orders',
        total_spent DOUBLE COMMENT 'Total amount spent in USD',
        last_purchase_date DATE COMMENT 'Date of last purchase',
        credit_limit DOUBLE COMMENT 'Credit limit in USD',
        payment_terms STRING COMMENT 'Payment terms'
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.autoOptimize.optimizeWrite' = 'true'
    )
    COMMENT 'Customer master data and profiles'
    """
    results['customers'] = execute_sql(spark, customers_sql,
                                       f"Creating {catalog}.sales.customers")
    
    # Transactions table
    transactions_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.sales.transactions (
        transaction_id STRING COMMENT 'Unique transaction identifier',
        order_id STRING COMMENT 'Order identifier',
        customer_id STRING COMMENT 'Customer identifier',
        customer_name STRING COMMENT 'Customer name',
        transaction_date TIMESTAMP COMMENT 'Transaction timestamp',
        product_id STRING COMMENT 'Product identifier',
        product_name STRING COMMENT 'Product name',
        quantity INT COMMENT 'Quantity purchased',
        unit_price DOUBLE COMMENT 'Price per unit in USD',
        total_amount DOUBLE COMMENT 'Total transaction amount',
        discount_amount DOUBLE COMMENT 'Discount applied',
        tax_amount DOUBLE COMMENT 'Tax amount',
        payment_method STRING COMMENT 'Payment method used',
        sales_rep_id STRING COMMENT 'Sales representative ID',
        sales_rep_name STRING COMMENT 'Sales representative name',
        region STRING COMMENT 'Sales region',
        status STRING COMMENT 'Transaction status'
    )
    USING DELTA
    PARTITIONED BY (transaction_date)
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.autoOptimize.optimizeWrite' = 'true'
    )
    COMMENT 'Sales transaction data'
    """
    results['transactions'] = execute_sql(spark, transactions_sql,
                                          f"Creating {catalog}.sales.transactions")
    
    # Sales Proposals table (for Vector Search)
    proposals_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.sales.sales_proposals (
        id STRING COMMENT 'Unique proposal identifier',
        proposal_id STRING COMMENT 'Proposal reference number',
        customer_id STRING COMMENT 'Customer identifier',
        customer_name STRING COMMENT 'Customer name',
        proposal_date DATE COMMENT 'Proposal creation date',
        proposal_content STRING COMMENT 'Full proposal text content',
        executive_summary STRING COMMENT 'Executive summary',
        proposal_value DOUBLE COMMENT 'Total proposal value in USD',
        products ARRAY<STRING> COMMENT 'Products included',
        status STRING COMMENT 'Proposal status (DRAFT/SENT/WON/LOST)',
        valid_until DATE COMMENT 'Proposal validity date',
        created_by STRING COMMENT 'Sales rep who created',
        win_probability DOUBLE COMMENT 'Estimated win probability',
        competitor_info STRING COMMENT 'Known competitor information',
        notes STRING COMMENT 'Internal notes'
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    )
    COMMENT 'Sales proposals and customer communications'
    """
    results['sales_proposals'] = execute_sql(spark, proposals_sql,
        f"Creating {catalog}.sales.sales_proposals")
    
    return results


def create_iot_tables(spark, catalog: str = "mfg_mcp_demo") -> Dict[str, bool]:
    """
    Create IoT domain tables.
    """
    results = {}
    
    # Telemetry table
    telemetry_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.iot.telemetry (
        reading_id STRING COMMENT 'Unique reading identifier',
        device_id STRING COMMENT 'IoT device identifier',
        device_type STRING COMMENT 'Type of device',
        location_id STRING COMMENT 'Device location',
        timestamp TIMESTAMP COMMENT 'Reading timestamp',
        temperature DOUBLE COMMENT 'Temperature in Celsius',
        humidity DOUBLE COMMENT 'Humidity percentage',
        pressure DOUBLE COMMENT 'Pressure in PSI',
        vibration DOUBLE COMMENT 'Vibration level',
        power_consumption DOUBLE COMMENT 'Power usage in kW',
        operational_status STRING COMMENT 'Device operational status',
        error_code STRING COMMENT 'Error code if any',
        maintenance_required BOOLEAN COMMENT 'Maintenance flag',
        reading_date DATE COMMENT 'Date of reading for partitioning'
    )
    USING DELTA
    PARTITIONED BY (reading_date)
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.autoOptimize.optimizeWrite' = 'true'
    )
    COMMENT 'IoT sensor telemetry data'
    """
    results['telemetry'] = execute_sql(spark, telemetry_sql,
                                       f"Creating {catalog}.iot.telemetry")
    
    return results


def create_support_tables(spark, catalog: str = "mfg_mcp_demo") -> Dict[str, bool]:
    """
    Create support domain tables.
    """
    results = {}
    
    # Support tickets table (main)
    tickets_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.support.tickets (
        ticket_id STRING COMMENT 'Unique ticket identifier',
        customer_id STRING COMMENT 'Customer identifier',
        customer_name STRING COMMENT 'Customer name',
        created_date TIMESTAMP COMMENT 'Ticket creation timestamp',
        category STRING COMMENT 'Ticket category',
        priority STRING COMMENT 'Priority level (LOW/MEDIUM/HIGH/CRITICAL)',
        subject STRING COMMENT 'Ticket subject',
        description STRING COMMENT 'Detailed description',
        status STRING COMMENT 'Current status',
        assigned_to STRING COMMENT 'Assigned support agent',
        resolved_date TIMESTAMP COMMENT 'Resolution timestamp',
        resolution_time_hours DOUBLE COMMENT 'Time to resolution',
        satisfaction_score INT COMMENT 'Customer satisfaction (1-5)',
        tags ARRAY<STRING> COMMENT 'Ticket tags'
    )
    USING DELTA
    PARTITIONED BY (created_date)
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    )
    COMMENT 'Customer support tickets'
    """
    results['tickets'] = execute_sql(spark, tickets_sql,
                                     f"Creating {catalog}.support.tickets")
    
    # Support tickets for Vector Search (enhanced version)
    support_tickets_sql = f"""
    CREATE TABLE IF NOT EXISTS {catalog}.support.support_tickets (
        id STRING COMMENT 'Unique identifier for vector search',
        ticket_id STRING COMMENT 'Ticket reference number',
        customer_id STRING COMMENT 'Customer identifier',
        customer_name STRING COMMENT 'Customer name',
        ticket_content STRING COMMENT 'Full ticket content for search',
        problem_description STRING COMMENT 'Detailed problem description',
        resolution STRING COMMENT 'How the issue was resolved',
        category STRING COMMENT 'Ticket category',
        product_affected STRING COMMENT 'Affected product/service',
        created_date TIMESTAMP COMMENT 'Creation timestamp',
        resolved_date TIMESTAMP COMMENT 'Resolution timestamp',
        priority STRING COMMENT 'Priority level',
        tags ARRAY<STRING> COMMENT 'Searchable tags'
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    )
    COMMENT 'Support tickets optimized for vector search'
    """
    results['support_tickets'] = execute_sql(spark, support_tickets_sql,
        f"Creating {catalog}.support.support_tickets")
    
    return results


def optimize_tables(spark, catalog: str = "mfg_mcp_demo") -> Dict[str, bool]:
    """
    Optimize tables with Z-ordering for better query performance.
    """
    results = {}
    
    optimizations = [
        (f"{catalog}.supply_chain.inventory", ["part_id", "warehouse_id"]),
        (f"{catalog}.supply_chain.shipments", ["shipment_date", "status"]),
        (f"{catalog}.sales.transactions", ["customer_id", "transaction_date"]),
        (f"{catalog}.sales.customers", ["customer_id", "status"]),
        (f"{catalog}.iot.telemetry", ["device_id", "reading_date"]),
        (f"{catalog}.support.tickets", ["customer_id", "created_date"])
    ]
    
    for table, columns in optimizations:
        try:
            sql = f"OPTIMIZE {table} ZORDER BY ({', '.join(columns)})"
            results[table] = execute_sql(spark, sql, f"Optimizing {table}")
        except Exception as e:
            # Table might be empty, which is OK
            print(f"   ℹ️  Could not optimize {table} (might be empty): {e}")
            results[table] = True
    
    return results


def verify_tables(spark, catalog: str = "mfg_mcp_demo") -> bool:
    """
    Verify all tables were created successfully.
    """
    print("\n🔍 Verifying Delta tables...")
    
    expected_tables = {
        "supply_chain": ["inventory", "shipments", "suppliers", 
                        "standard_operating_procedures", "incident_reports"],
        "sales": ["customers", "transactions", "sales_proposals"],
        "iot": ["telemetry"],
        "support": ["tickets", "support_tickets"]
    }
    
    all_good = True
    
    for schema, tables in expected_tables.items():
        try:
            # Get tables in schema
            tables_df = spark.sql(f"SHOW TABLES IN {catalog}.{schema}")
            existing_tables = [row.tableName for row in tables_df.collect()]
            
            for table in tables:
                if table in existing_tables:
                    print(f"   ✅ {catalog}.{schema}.{table}")
                else:
                    print(f"   ❌ {catalog}.{schema}.{table} not found")
                    all_good = False
                    
        except Exception as e:
            print(f"   ❌ Could not verify {catalog}.{schema}: {e}")
            all_good = False
    
    return all_good


def main():
    """
    Main execution function to create all Delta tables.
    """
    print("=" * 60)
    print("Delta Tables Creation for Manufacturing MCP Examples")
    print("Phase 2.1: Creating Delta Tables")
    print("=" * 60)
    
    # Setup environment (universal for notebook or local)
    config = setup_databricks_environment()
    spark = config.get('spark')
    catalog_name = config.get('catalog', 'mfg_mcp_demo')
    
    if not spark:
        print("\n⚠️  No Spark session available. Running in dry-run mode.")
        print("   SQL statements will be printed but not executed.")
    
    print(f"\n📦 Target catalog: {catalog_name}")
    
    # Set catalog context (if not already set in setup)
    if spark:
        try:
            spark.sql(f"USE CATALOG {catalog_name}")
            print(f"   ✅ Using catalog: {catalog_name}")
        except Exception as e:
            print(f"   ❌ Could not use catalog {catalog_name}: {e}")
            print("   Please run setup_unity_catalog.py first")
            cleanup_environment(config)
            return 1
    
    print("-" * 40)
    
    # Create supply chain tables
    print("\n🏭 Creating Supply Chain tables...")
    print("-" * 40)
    sc_results = create_supply_chain_tables(spark, catalog_name)
    
    # Create sales tables
    print("\n💰 Creating Sales tables...")
    print("-" * 40)
    sales_results = create_sales_tables(spark, catalog_name)
    
    # Create IoT tables
    print("\n📡 Creating IoT tables...")
    print("-" * 40)
    iot_results = create_iot_tables(spark, catalog_name)
    
    # Create support tables
    print("\n🎫 Creating Support tables...")
    print("-" * 40)
    support_results = create_support_tables(spark, catalog_name)
    
    # Combine all results
    all_results = {**sc_results, **sales_results, **iot_results, **support_results}
    failed_tables = [t for t, result in all_results.items() if not result]
    
    if failed_tables:
        print(f"\n❌ Failed to create tables: {failed_tables}")
        return 1
    
    # Optimize tables (only if they exist and have data)
    if spark:
        print("\n⚡ Optimizing tables with Z-ordering...")
        print("-" * 40)
        optimize_results = optimize_tables(spark, catalog_name)
        print(f"   Optimized {sum(optimize_results.values())} tables successfully")
    
    # Verify all tables
    if spark:
        verification_success = verify_tables(spark, catalog_name)
        
        if verification_success:
            print("\n✅ All Delta tables created successfully!")
            print(f"\n📊 Summary:")
            print(f"   - Catalog: {catalog_name}")
            print(f"   - Tables created: {len(all_results)}")
            print(f"   - Supply Chain: {len(sc_results)} tables")
            print(f"   - Sales: {len(sales_results)} tables")
            print(f"   - IoT: {len(iot_results)} tables")
            print(f"   - Support: {len(support_results)} tables")
        else:
            print("\n⚠️  Some tables could not be verified. Please check the output above.")
    else:
        print("\n✅ Dry-run completed. Run with Spark session to execute.")
    
    print("\n" + "=" * 60)
    print("Next step: Run generate_sample_data.py to populate tables with data")
    print("=" * 60)
    
    # Clean up resources if running locally
    cleanup_environment(config)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())