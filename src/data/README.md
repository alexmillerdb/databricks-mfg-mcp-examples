# Data Layer Scripts - Manufacturing MCP Examples

This directory contains the data layer implementation scripts for the Manufacturing MCP Examples project. These scripts set up Unity Catalog structure, create Delta tables, and generate realistic synthetic data for demonstration purposes.

## ✅ Completed Scripts

### 1. Unity Catalog Setup: `setup_unity_catalog.py`

Sets up the Unity Catalog structure for the manufacturing demo with proper governance and permissions.

**Features:**
- Creates catalog `mfg_mcp_demo` 
- Creates 5 schemas: `supply_chain`, `sales`, `iot`, `support`, `agent_logs`
- Grants appropriate permissions for multi-tenant access
- Universal setup for both Databricks notebooks and local IDE execution
- Dry-run mode support for testing

**Usage:**
```bash
# Run locally with Databricks Connect
python src/data/setup_unity_catalog.py

# Or in Databricks notebook
%run ./setup_unity_catalog
```

**Environment Variables:**
- `UC_DEFAULT_CATALOG`: Catalog name (default: "mfg_mcp_demo")
- `DATABRICKS_CONFIG_PROFILE`: CLI profile (default: "aws-apps")

### 2. Delta Tables Creation: `create_delta_tables.py`

Creates all required Delta tables with optimized schemas for the manufacturing use cases.

**Created Tables:**
- **Supply Chain**: `inventory`, `shipments`, `suppliers`, `standard_operating_procedures`, `incident_reports`
- **Sales**: `customers`, `transactions`, `sales_proposals`
- **IoT**: `telemetry` 
- **Support**: `tickets`, `support_tickets`

**Features:**
- Proper data types with INT for integers, TIMESTAMP for dates
- Partitioning by date columns for performance
- Z-ordering optimization for common query patterns
- Change Data Feed enabled for real-time processing
- Schema verification and validation

**Usage:**
```bash
# Run locally
python src/data/create_delta_tables.py

# Or in Databricks notebook
%run ./create_delta_tables
```

### 3. Sample Data Generation: `generate_sample_data.py`

Generates realistic synthetic data for all tables with proper relationships and business logic.

**Generated Data:**
- **50 suppliers** with ratings, lead times, and contact information
- **10,000 inventory items** distributed across 5 warehouses
- **1,000 shipments** with various statuses and tracking
- **500 customers** with purchase history and segments
- **5,000 transactions** with pricing, discounts, and sales reps
- **100,000 IoT readings** with sensor data and anomalies
- **1,000 support tickets** with resolutions and categories
- **100 SOPs** for Vector Search RAG capabilities
- **200 incident reports** for troubleshooting assistance
- **300 sales proposals** for customer insights

**Features:**
- Uses `faker` library for realistic names, addresses, and contact info
- Maintains referential integrity between related tables
- Generates business-realistic data with proper distributions
- Handles schema type conflicts with automatic casting
- Special handling for array columns to avoid type inference errors
- Universal execution for notebooks and local environments

**Usage:**
```bash
# Run locally
python src/data/generate_sample_data.py

# Or in Databricks notebook
%run ./generate_sample_data
```

## Quick Start Guide

### Prerequisites

1. **Databricks Environment Setup:**
   ```bash
   # Configure Databricks CLI (if running locally)
   databricks auth login --profile aws-apps
   
   # Verify connection
   databricks current-user me --profile aws-apps
   ```

2. **Python Environment:**
   ```bash
   # Install required packages
   pip install databricks-connect databricks-sdk faker python-dotenv
   ```

3. **Environment Variables (.env file):**
   ```bash
   DATABRICKS_CONFIG_PROFILE=aws-apps
   UC_DEFAULT_CATALOG=mfg_mcp_demo
   UC_DEFAULT_SCHEMA=supply_chain
   ```

### Execution Order

Run the scripts in this order for complete setup:

```bash
# 1. Set up Unity Catalog structure
python src/data/setup_unity_catalog.py

# 2. Create Delta tables
python src/data/create_delta_tables.py

# 3. Generate and load sample data
python src/data/generate_sample_data.py
```

### Verification

After running all scripts, verify the setup:

```python
from databricks.connect import DatabricksSession

# Connect to Databricks
spark = DatabricksSession.builder.profile("aws-apps").serverless(True).getOrCreate()
spark.sql("USE CATALOG mfg_mcp_demo")

# Check table counts
tables = [
    "supply_chain.suppliers",
    "supply_chain.inventory", 
    "supply_chain.shipments",
    "sales.customers",
    "sales.transactions",
    "iot.telemetry",
    "support.tickets"
]

for table in tables:
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {table}").collect()[0]['cnt']
    print(f"{table}: {count:,} records")
```

Expected output:
```
supply_chain.suppliers: 50 records
supply_chain.inventory: 10,000 records
supply_chain.shipments: 1,000 records
sales.customers: 500 records
sales.transactions: 5,000 records
iot.telemetry: 100,000 records
support.tickets: 1,000 records
```

## Data Schema Details

### Supply Chain Schema
- **Suppliers**: Company info, ratings, lead times, payment terms
- **Inventory**: Parts across warehouses with reorder levels
- **Shipments**: Tracking with carriers, status, and delivery dates
- **SOPs**: Standard Operating Procedures with searchable text
- **Incidents**: Historical problems with resolutions

### Sales Schema  
- **Customers**: Company profiles with purchase history and segments
- **Transactions**: Detailed sales with pricing, discounts, and reps
- **Proposals**: Sales proposals with content for Vector Search

### IoT Schema
- **Telemetry**: Sensor readings with temperature, pressure, vibration, and alerts

### Support Schema
- **Tickets**: Customer support requests with resolutions and satisfaction
- **Support Tickets**: Enhanced version optimized for Vector Search

## Script Features

### Universal Execution Environment
All scripts detect whether they're running in:
- **Databricks Notebook**: Uses global `spark` session
- **Local IDE**: Creates `DatabricksSession` with serverless compute

### Error Handling
- Graceful handling of "already exists" errors
- Dry-run mode when Spark session unavailable  
- Schema type casting to prevent data type conflicts
- Comprehensive logging and status messages

### Performance Optimizations
- **Partitioning**: Tables partitioned by date columns
- **Z-ordering**: Optimized for common query patterns
- **Change Data Feed**: Enabled for real-time processing
- **Auto-optimization**: Compaction and optimize write enabled

### Data Quality
- **Referential Integrity**: Foreign key relationships maintained
- **Business Logic**: Realistic data with proper distributions
- **Data Validation**: Type checking and constraint validation
- **Reproducibility**: Seeded random generation for consistent results

## Troubleshooting

### Common Issues

1. **"Schema not found" errors:**
   - Ensure Unity Catalog setup completed first
   - Check catalog name matches environment variable

2. **Type inference errors:**
   - Array fields automatically handled with consistent types
   - Schema casting applied for existing tables

3. **Permission errors:**
   - Verify Databricks CLI authentication
   - Check Unity Catalog permissions

4. **Connection timeouts:**
   - Ensure serverless compute is enabled
   - Check network connectivity to Databricks workspace

### Debug Mode

Enable debug output by setting environment variable:
```bash
export DATABRICKS_DEBUG=true
python src/data/generate_sample_data.py
```

## Next Steps

After completing the data layer setup:

1. **Create Vector Search Indexes** (Phase 2.3)
   - Set up Vector Search for SOPs, incidents, and proposals
   - Enable RAG capabilities for the manufacturing agent

2. **Configure Genie Spaces** (Phase 2.4)  
   - Create Genie Space for Supply Chain Analytics
   - Create Genie Space for Sales Analytics

3. **Develop Unity Catalog Functions** (Phase 3)
   - Implement predictive functions for shortage and churn
   - Create automation functions for order management

The data foundation is now ready to support the full Manufacturing MCP Examples implementation!