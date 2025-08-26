# Databricks Manufacturing MCP Examples - Execution Plan

This document provides a comprehensive implementation plan for building the multi-tenant Supply Chain & Sales agentic system on Databricks using Model Context Protocol (MCP). This plan follows the updated architecture and best practices defined in CLAUDE.md and the documentation guides.

## Project Overview

Building a production-ready demonstration of MCP-based agent orchestration for manufacturing use cases with:
- **Two primary personas**: Supply Chain Manager and Sales Representative
- **Multi-tenant architecture**: Unity Catalog with optional On-Behalf-Of (OBO) authentication
- **Managed MCP servers (Priority)**: Vector Search, Genie Spaces, UC Functions for instant governance
- **Custom MCP server (Optional)**: Only for specialized logic not covered by managed MCP
- **ResponsesAgent Interface**: Modern MLflow agent authoring with streaming support
- **Full observability**: MLflow 3.0 integration with tracing, evaluation, and monitoring

## Quick Start Priorities (Following CLAUDE.md Guidelines)

1. **Default to managed MCP** for Databricks services (Genie, Vector Search, UC Functions)
2. **Pin versions and verify environment** for stability
3. **Log resource dependencies** with the model for automatic credential injection
4. **Prefer automatic auth passthrough** - use OBO only when per-user scoping required
5. **Follow the 10-step agent development workflow** from CLAUDE.md

## Implementation Phases

### Phase 1: Databricks Workspace & Environment Setup
**Priority: Critical | Duration: 1 day**

#### 1.1 Configure Databricks CLI Authentication
```bash
# Install Databricks CLI with specific version
pip install databricks-cli==0.18.0

# Configure OAuth authentication (preferred over PAT)
databricks auth login --host https://<workspace-hostname>

# Verify connection
databricks workspace ls /Users
```

#### 1.2 Create Unity Catalog Structure
```sql
-- Create catalog for manufacturing demo
CREATE CATALOG IF NOT EXISTS manufacturing;

-- Create schemas for different domains
CREATE SCHEMA IF NOT EXISTS manufacturing.supply_chain;
CREATE SCHEMA IF NOT EXISTS manufacturing.sales;
CREATE SCHEMA IF NOT EXISTS manufacturing.iot;

-- Grant minimal permissions (least privilege principle)
GRANT USAGE ON CATALOG manufacturing TO `account users`;
GRANT USE SCHEMA ON SCHEMA manufacturing.supply_chain TO `supply_chain_users`;
GRANT USE SCHEMA ON SCHEMA manufacturing.sales TO `sales_users`;
```

#### 1.3 Set Up Python Environment with Pinned Versions
```bash
# Create virtual environment with Python 3.12+
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pinned dependencies for managed MCP
pip install -U "mcp>=1.9" \
              "databricks-sdk[openai]==0.20.0" \
              "mlflow>=3.1.0" \
              "databricks-agents>=1.0.0" \
              "databricks-mcp>=0.1.0" \
              "langchain==0.2.0" \
              "langgraph==0.1.0"

# Verify installation
python -c "import mlflow; print(mlflow.__version__)"
```

### Phase 2: Data Layer Implementation
**Priority: High | Duration: 2 days**

#### 2.1 Create Delta Tables
Location: `src/data/create_tables.py`

Tables to create:
- `manufacturing.supply_chain.inventory` - Current inventory levels
- `manufacturing.supply_chain.shipments` - Shipment tracking
- `manufacturing.supply_chain.suppliers` - Supplier information
- `manufacturing.sales.transactions` - Sales data
- `manufacturing.sales.customers` - Customer profiles
- `manufacturing.iot.telemetry` - Sensor data
- `manufacturing.support.tickets` - Support tickets

Schema definitions should include:
- Proper data types (use TIMESTAMP for dates, DOUBLE for metrics)
- Partitioning by date where appropriate
- Z-ordering for query optimization
- Change Data Feed enabled for real-time processing

#### 2.2 Implement Data Generation Utilities
Location: `src/data/generate_data.py`

Generate realistic synthetic data for:
- 10,000+ inventory items across 5 warehouses
- 1,000+ active shipments with various statuses
- 500+ customers with purchase history
- 100,000+ IoT sensor readings
- 1,000+ support tickets with resolutions

Use libraries:
- `faker` for realistic names/addresses
- `numpy` for statistical distributions
- `pandas` for data manipulation

#### 2.3 Create Vector Search Indexes
Location: `src/data/create_vector_indexes.py`

**Following**: docs/vector-search/vector-search-guide.md for comprehensive implementation

Create vector search indexes for RAG capabilities with both managed MCP server integration and direct LangChain tool usage. This implementation supports:

**Indexes to create**:
- `manufacturing.supply_chain.sop_index` - Standard Operating Procedures
- `manufacturing.supply_chain.incident_index` - Incident reports  
- `manufacturing.sales.proposal_index` - Sales proposals
- `manufacturing.support.ticket_index` - Support ticket history

**Configuration**:
- Use embedding model: `databricks-gte-large-en`
- Delta Sync indexes for automatic embedding generation
- Real-time sync with TRIGGERED pipeline mode
- Optimized for both LangChain tools and managed MCP servers

**Implementation**:
```python
from databricks.vector_search import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks_langchain import VectorSearchRetrieverTool
from databricks_mcp import DatabricksMCPClient
import pandas as pd

def create_manufacturing_vector_indexes():
    """
    Create vector search indexes for manufacturing use cases.
    Supports both managed MCP server integration and direct LangChain usage.
    """
    
    # Initialize clients
    workspace_client = WorkspaceClient()
    vs_client = VectorSearchClient(workspace_client=workspace_client)
    
    # Vector search endpoint (create if doesn't exist)
    endpoint_name = "manufacturing_vector_search_endpoint"
    
    try:
        endpoint = vs_client.get_endpoint(endpoint_name)
        print(f"Using existing endpoint: {endpoint_name}")
    except Exception:
        print(f"Creating new endpoint: {endpoint_name}")
        endpoint = vs_client.create_endpoint(
            name=endpoint_name,
            endpoint_type="STANDARD"
        )
    
    # Index configurations for manufacturing use cases
    indexes_config = [
        {
            "name": "manufacturing.supply_chain.sop_index",
            "source_table": "manufacturing.supply_chain.standard_operating_procedures",
            "description": "Standard Operating Procedures for supply chain processes",
            "text_column": "procedure_text",
            "embedding_model": "databricks-gte-large-en",
            "sync_mode": "TRIGGERED"
        },
        {
            "name": "manufacturing.supply_chain.incident_index", 
            "source_table": "manufacturing.supply_chain.incident_reports",
            "description": "Historical incident reports and resolutions",
            "text_column": "incident_description",
            "embedding_model": "databricks-gte-large-en",
            "sync_mode": "TRIGGERED"
        },
        {
            "name": "manufacturing.sales.proposal_index",
            "source_table": "manufacturing.sales.sales_proposals", 
            "description": "Sales proposals and customer communications",
            "text_column": "proposal_content",
            "embedding_model": "databricks-gte-large-en",
            "sync_mode": "TRIGGERED"
        },
        {
            "name": "manufacturing.support.ticket_index",
            "source_table": "manufacturing.support.support_tickets",
            "description": "Customer support tickets and resolutions",
            "text_column": "ticket_content", 
            "embedding_model": "databricks-gte-large-en",
            "sync_mode": "TRIGGERED"
        }
    ]
    
    # Create indexes
    created_indexes = []
    for config in indexes_config:
        try:
            print(f"Creating index: {config['name']}")
            
            # Create Delta Sync index for automatic embedding generation
            index = vs_client.create_delta_sync_index(
                endpoint_name=endpoint_name,
                index_name=config["name"],
                source_table_name=config["source_table"],
                pipeline_type="TRIGGERED",  # Real-time sync
                primary_key="id",
                embedding_source_column=config["text_column"],
                embedding_model_endpoint_name=config["embedding_model"]
            )
            
            created_indexes.append({"index": index, "config": config})
            print(f"✓ Successfully created index: {config['name']}")
            
        except Exception as e:
            print(f"✗ Failed to create index {config['name']}: {e}")
    
    return created_indexes, endpoint

def setup_vector_search_tools():
    """Set up LangChain tools for vector search integration."""
    
    # Create LangChain tools for each index
    vector_tools = []
    
    # Standard Operating Procedures retriever
    sop_tool = VectorSearchRetrieverTool(
        index_name="manufacturing.supply_chain.sop_index",
        tool_name="sop_retriever", 
        tool_description="Retrieves relevant standard operating procedures for manufacturing processes, quality control, and maintenance activities.",
        num_results=5,
        columns=["procedure_name", "procedure_text", "category"]
    )
    vector_tools.append(sop_tool)
    
    # Incident reports retriever
    incident_tool = VectorSearchRetrieverTool(
        index_name="manufacturing.supply_chain.incident_index",
        tool_name="incident_retriever",
        tool_description="Retrieves historical incident reports and resolutions to help troubleshoot current issues and prevent recurring problems.",
        num_results=3,
        columns=["incident_title", "incident_description", "resolution_time_hours"]
    )
    vector_tools.append(incident_tool)
    
    # Sales proposals retriever  
    proposal_tool = VectorSearchRetrieverTool(
        index_name="manufacturing.sales.proposal_index",
        tool_name="proposal_retriever",
        tool_description="Retrieves past sales proposals and customer communications to inform new proposals and customer interactions.",
        num_results=3,
        columns=["customer_name", "proposal_content", "proposal_value", "status"]
    )
    vector_tools.append(proposal_tool)
    
    # Support tickets retriever
    support_tool = VectorSearchRetrieverTool(
        index_name="manufacturing.support.ticket_index", 
        tool_name="support_retriever",
        tool_description="Retrieves customer support ticket history and resolutions to help resolve current customer issues efficiently.",
        num_results=3,
        columns=["customer_name", "ticket_content", "resolution"]
    )
    vector_tools.append(support_tool)
    
    return vector_tools

def setup_managed_mcp_vector_search():
    """Set up managed MCP server integration for vector search."""
    
    # Initialize workspace client
    workspace_client = WorkspaceClient()
    host = workspace_client.config.host
    
    # MCP server URLs for different catalogs/schemas
    mcp_servers = {
        "supply_chain": f"{host}/api/2.0/mcp/vector-search/manufacturing/supply_chain",
        "sales": f"{host}/api/2.0/mcp/vector-search/manufacturing/sales", 
        "support": f"{host}/api/2.0/mcp/vector-search/manufacturing/support"
    }
    
    # Connect to managed MCP servers and discover tools
    mcp_clients = {}
    discovered_tools = {}
    
    for domain, server_url in mcp_servers.items():
        try:
            print(f"Connecting to {domain} MCP server: {server_url}")
            
            mcp_client = DatabricksMCPClient(
                server_url=server_url,
                workspace_client=workspace_client
            )
            
            # Discover available tools
            tools = mcp_client.list_tools()
            print(f"Discovered {len(tools)} tools for {domain}:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            mcp_clients[domain] = mcp_client
            discovered_tools[domain] = tools
            
        except Exception as e:
            print(f"Failed to connect to {domain} MCP server: {e}")
    
    return mcp_clients, discovered_tools
```

#### 2.4 Configure Genie Spaces
Manual steps in Databricks UI:
1. Create Genie Space for Supply Chain Analytics
   - Add tables: inventory, shipments, suppliers
   - Configure natural language descriptions
2. Create Genie Space for Sales Analytics
   - Add tables: transactions, customers
   - Configure KPI definitions

### Phase 3: Unity Catalog Functions for Managed MCP
**Priority: Critical | Duration: 2 days**
**Note**: UC Functions will be exposed through managed MCP servers - no custom infrastructure needed!

#### 3.1 Create predict_shortage UC Function (for Managed MCP)
Location: `src/uc_functions/predict_shortage.py`

Python function implementation with resource dependency tracking:
```python
def predict_shortage(part_id: str, current_inventory: int, usage_rate: float) -> str:
    """
    Predict potential part shortages using historical patterns and current inventory.
    
    Args:
        part_id (str): The unique identifier of the part to analyze
        current_inventory (int): Current inventory count for the part
        usage_rate (float): Daily usage rate of the part
    
    Returns:
        str: JSON string containing shortage prediction with risk score and recommendations
    """
    import json
    import datetime
    
    # Simple predictive logic without external dependencies
    safety_buffer = 14  # 2 week safety buffer
    daily_usage = max(usage_rate, 0.1)  # Prevent division by zero
    
    # Calculate days until shortage
    days_until_shortage = max(1, int(current_inventory / daily_usage))
    
    # Calculate risk score (0-1 scale)
    risk_score = min(1.0, max(0.0, (safety_buffer - days_until_shortage) / safety_buffer))
    
    # Determine risk level and recommendation
    if risk_score > 0.8:
        risk_level = "CRITICAL"
        recommendation = f"URGENT: Order immediately. Critical shortage in {days_until_shortage} days."
    elif risk_score > 0.5:
        risk_level = "HIGH"
        recommendation = f"HIGH: Schedule order soon. Shortage expected in {days_until_shortage} days."
    elif risk_score > 0.2:
        risk_level = "MEDIUM"
        recommendation = f"MEDIUM: Monitor closely. {days_until_shortage} days of inventory remaining."
    else:
        risk_level = "LOW"
        recommendation = f"LOW: Inventory adequate. {days_until_shortage} days remaining."
    
    # Calculate suggested order quantity (30 days supply)
    suggested_quantity = int(daily_usage * 30)
    
    result = {
        "part_id": part_id,
        "current_inventory": current_inventory,
        "daily_usage_rate": daily_usage,
        "days_until_shortage": days_until_shortage,
        "risk_score": round(risk_score, 3),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "suggested_order_quantity": suggested_quantity,
        "predicted_shortage_date": str(datetime.date.today() + datetime.timedelta(days=days_until_shortage))
    }
    
    return json.dumps(result)

# Register function using DatabricksFunctionClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

CATALOG = "manufacturing"
SCHEMA = "supply_chain"

client = DatabricksFunctionClient()
function_info = client.create_python_function(
    func=predict_shortage,
    catalog=CATALOG,
    schema=SCHEMA,
    replace=True
)
```

#### 3.2 Create predict_churn UC Function
Location: `src/uc_functions/predict_churn.py`

Python function implementation:
```python
def predict_churn(customer_id: str, days_since_last_purchase: int, total_purchases: int, avg_order_value: float) -> str:
    """
    Predict customer churn probability using RFM-based analysis.
    
    Args:
        customer_id (str): Unique identifier of the customer to analyze
        days_since_last_purchase (int): Number of days since the last purchase
        total_purchases (int): Total number of purchases made by the customer
        avg_order_value (float): Average value of customer orders
    
    Returns:
        str: JSON string containing churn prediction, risk factors, and recommendations
    """
    import json
    
    # Calculate churn score components (0-1 scale, higher = more likely to churn)
    recency_score = min(1.0, days_since_last_purchase / 90.0)  # Normalize to 90 days
    frequency_score = max(0.0, 1.0 - (total_purchases / 20.0))  # Good customers have 20+ purchases
    monetary_score = max(0.0, 1.0 - (avg_order_value / 1000.0))  # Good customers spend $1000+ avg
    
    # Weighted churn probability
    churn_probability = (
        recency_score * 0.5 +      # Recency is most important
        frequency_score * 0.3 +    # Frequency is important
        monetary_score * 0.2       # Monetary value is least important
    )
    
    # Determine risk level
    if churn_probability > 0.7:
        risk_level = "HIGH"
    elif churn_probability > 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Identify specific risk factors
    risk_factors = []
    if days_since_last_purchase > 60:
        risk_factors.append(f"No purchases in {days_since_last_purchase} days")
    if total_purchases < 5:
        risk_factors.append("Low purchase frequency")
    if avg_order_value < 200:
        risk_factors.append("Low average order value")
    
    # Generate recommendations
    recommendations = []
    if risk_level == "HIGH":
        recommendations.extend([
            "Send immediate re-engagement campaign",
            "Offer personalized discount or incentive",
            "Schedule customer success call"
        ])
    elif risk_level == "MEDIUM":
        recommendations.extend([
            "Monitor customer activity closely", 
            "Send targeted product recommendations",
            "Consider loyalty program enrollment"
        ])
    else:
        recommendations.append("Continue current engagement strategy")
    
    result = {
        "customer_id": customer_id,
        "churn_probability": round(churn_probability, 3),
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "customer_metrics": {
            "days_since_last_purchase": days_since_last_purchase,
            "total_purchases": total_purchases,
            "avg_order_value": round(avg_order_value, 2)
        },
        "recommendations": recommendations
    }
    
    return json.dumps(result)

# Register function using DatabricksFunctionClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

CATALOG = "manufacturing"
SCHEMA = "sales"

client = DatabricksFunctionClient()
function_info = client.create_python_function(
    func=predict_churn,
    catalog=CATALOG,
    schema=SCHEMA,
    replace=True
)
```

#### 3.3 Create create_order UC Function
Location: `src/uc_functions/create_order.py`

Python function implementation:
```python
def create_order(supplier_id: str, part_id: str, quantity: int, priority: str = "NORMAL") -> str:
    """
    Create automated purchase order with validation and cost estimation.
    
    Args:
        supplier_id (str): Unique identifier of the supplier
        part_id (str): Unique identifier of the part to order
        quantity (int): Number of units to order
        priority (str): Order priority level - NORMAL, HIGH, or URGENT
    
    Returns:
        str: JSON string containing order confirmation, costs, and delivery estimates
    """
    import json
    import uuid
    import datetime
    
    # Validate inputs
    if quantity <= 0:
        return json.dumps({
            "error": "Quantity must be greater than zero",
            "order_id": None,
            "status": "FAILED"
        })
    
    if priority not in ["NORMAL", "HIGH", "URGENT"]:
        priority = "NORMAL"
    
    # Generate order ID
    order_id = f"PO-{uuid.uuid4().hex[:8].upper()}"
    
    # Simulate supplier and part data lookup (in real implementation, this would query actual tables)
    suppliers = {
        "SUP001": {"name": "Acme Manufacturing", "lead_time": 7, "reliability": 0.95},
        "SUP002": {"name": "Global Parts Inc", "lead_time": 10, "reliability": 0.90},
        "SUP003": {"name": "Quick Supply Co", "lead_time": 3, "reliability": 0.85}
    }
    
    parts = {
        "PART001": {"name": "Steel Bearing", "unit_cost": 15.50, "min_qty": 10},
        "PART002": {"name": "Aluminum Housing", "unit_cost": 45.00, "min_qty": 5},
        "PART003": {"name": "Rubber Gasket", "unit_cost": 2.25, "min_qty": 50}
    }
    
    # Validate supplier and part
    if supplier_id not in suppliers:
        return json.dumps({
            "error": f"Invalid supplier ID: {supplier_id}",
            "order_id": None,
            "status": "FAILED"
        })
    
    if part_id not in parts:
        return json.dumps({
            "error": f"Invalid part ID: {part_id}",
            "order_id": None,
            "status": "FAILED"
        })
    
    supplier = suppliers[supplier_id]
    part = parts[part_id]
    
    # Validate minimum quantity
    if quantity < part["min_qty"]:
        return json.dumps({
            "error": f"Quantity {quantity} below minimum order quantity {part['min_qty']}",
            "order_id": None,
            "status": "FAILED"
        })
    
    # Calculate costs
    unit_cost = part["unit_cost"]
    subtotal = unit_cost * quantity
    
    # Apply rush fees based on priority
    rush_fee_percent = {"URGENT": 0.20, "HIGH": 0.10, "NORMAL": 0.00}[priority]
    rush_fee = subtotal * rush_fee_percent
    total_cost = subtotal + rush_fee
    
    # Calculate delivery estimate
    base_lead_time = supplier["lead_time"]
    priority_adjustment = {"URGENT": -3, "HIGH": -1, "NORMAL": 0}[priority]
    estimated_delivery_days = max(1, base_lead_time + priority_adjustment)
    
    delivery_date = datetime.date.today() + datetime.timedelta(days=estimated_delivery_days)
    
    # Create order result
    result = {
        "order_id": order_id,
        "status": "PENDING",
        "supplier": {
            "supplier_id": supplier_id,
            "supplier_name": supplier["name"],
            "reliability_score": supplier["reliability"]
        },
        "part": {
            "part_id": part_id,
            "part_name": part["name"],
            "unit_cost": unit_cost,
            "quantity": quantity
        },
        "pricing": {
            "subtotal": round(subtotal, 2),
            "rush_fee": round(rush_fee, 2),
            "rush_fee_percent": rush_fee_percent * 100,
            "total_cost": round(total_cost, 2)
        },
        "delivery": {
            "estimated_delivery_date": str(delivery_date),
            "estimated_days": estimated_delivery_days,
            "priority": priority
        },
        "next_steps": [
            "Purchase order has been generated",
            "Supplier notification will be sent automatically",
            f"Expected delivery: {delivery_date}",
            "Order tracking available in procurement system"
        ]
    }
    
    return json.dumps(result)

# Register function using DatabricksFunctionClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

CATALOG = "manufacturing"
SCHEMA = "supply_chain"

client = DatabricksFunctionClient()
function_info = client.create_python_function(
    func=create_order,
    catalog=CATALOG,
    schema=SCHEMA,
    replace=True
)
```

#### 3.4 Create SQL Retrieval Functions
Location: `src/uc_functions/sql_functions.sql`

```sql
-- Customer lookup function
CREATE OR REPLACE FUNCTION manufacturing.sales.lookup_customer_info(
  customer_name STRING COMMENT 'Name of the customer to look up'
)
RETURNS STRING
COMMENT 'Returns customer details including ID, email, and purchase summary'
RETURN SELECT CONCAT(
    'Customer ID: ', customer_id, ', ',
    'Email: ', email, ', ',
    'Total Orders: ', CAST(total_orders AS STRING), ', ',
    'Total Spent: $', CAST(ROUND(total_spent, 2) AS STRING), ', ',
    'Last Purchase: ', CAST(last_purchase_date AS STRING), ', ',
    'Status: ', status
  )
  FROM manufacturing.sales.customers
  WHERE customer_name LIKE CONCAT('%', customer_name, '%')
  ORDER BY last_purchase_date DESC
  LIMIT 1;

-- Inventory status lookup
CREATE OR REPLACE FUNCTION manufacturing.supply_chain.get_inventory_status(
  part_name STRING COMMENT 'Name or partial name of the part'
)
RETURNS STRING
COMMENT 'Returns current inventory levels and reorder recommendations'
RETURN SELECT CONCAT(
    'Part: ', part_name, ' (', part_id, '), ',
    'Stock: ', CAST(current_quantity AS STRING), ' units, ',
    'Location: ', warehouse_location, ', ',
    'Reorder Level: ', CAST(reorder_level AS STRING), ', ',
    'Status: ', CASE 
        WHEN current_quantity <= reorder_level THEN 'REORDER REQUIRED'
        WHEN current_quantity <= (reorder_level * 1.5) THEN 'LOW STOCK'
        ELSE 'ADEQUATE'
    END
  )
  FROM manufacturing.supply_chain.inventory
  WHERE part_name LIKE CONCAT('%', part_name, '%')
  ORDER BY current_quantity ASC
  LIMIT 1;

-- Shipment tracking function
CREATE OR REPLACE FUNCTION manufacturing.supply_chain.track_shipments(
  days_back INT COMMENT 'Number of days to look back (default 7)'
)
RETURNS STRING
COMMENT 'Returns shipment summary for the specified time period'
RETURN SELECT CONCAT(
    'Period: Last ', CAST(days_back AS STRING), ' days, ',
    'Total Shipments: ', CAST(COUNT(*) AS STRING), ', ',
    'Delivered: ', CAST(SUM(CASE WHEN status = 'DELIVERED' THEN 1 ELSE 0 END) AS STRING), ' (', 
    CAST(ROUND(100.0 * SUM(CASE WHEN status = 'DELIVERED' THEN 1 ELSE 0 END) / COUNT(*), 1) AS STRING), '%), ',
    'In Transit: ', CAST(SUM(CASE WHEN status = 'IN_TRANSIT' THEN 1 ELSE 0 END) AS STRING), ', ',
    'Delayed: ', CAST(SUM(CASE WHEN status = 'DELAYED' THEN 1 ELSE 0 END) AS STRING)
  )
  FROM manufacturing.supply_chain.shipments
  WHERE shipment_date >= CURRENT_DATE - INTERVAL days_back DAYS;
```

#### 3.5 Register Functions for Agent Integration
Location: `src/uc_functions/register_all_functions.py`

```python
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from databricks_langchain import UCFunctionToolkit

# Initialize client
client = DatabricksFunctionClient()

print("Registering UC Functions for Manufacturing Agent...")

# Register Python functions (already done in individual files above)
python_functions = [
    "manufacturing.supply_chain.predict_shortage",
    "manufacturing.sales.predict_churn", 
    "manufacturing.supply_chain.create_order"
]

# SQL functions are created via SQL DDL (see sql_functions.sql)
sql_functions = [
    "manufacturing.sales.lookup_customer_info",
    "manufacturing.supply_chain.get_inventory_status",
    "manufacturing.supply_chain.track_shipments"
]

# Combine all functions for agent toolkit
all_function_names = python_functions + sql_functions

# Create toolkit for agent integration
toolkit = UCFunctionToolkit(
    function_names=all_function_names,
    warehouse_id="your_warehouse_id"  # Replace with actual warehouse ID
)

# Get tools for agent use
tools = toolkit.tools

print(f"Successfully created {len(tools)} UC Function tools:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")

# Test function execution (optional)
if __name__ == "__main__":
    # Test predict_shortage
    test_result = client.execute_function(
        function_name="manufacturing.supply_chain.predict_shortage",
        parameters={
            "part_id": "PART001",
            "current_inventory": 25,
            "usage_rate": 2.5
        }
    )
    print(f"\nTest result for predict_shortage: {test_result}")
```

### Phase 4: ResponsesAgent Development with Managed MCP
**Priority: Critical | Duration: 3 days**
**Following**: docs/agents/best-practices-deploying-agents-workflow.md & docs/mlflow/mlflow-agent-development-guide.md

#### 4.1 Implement ResponsesAgent with Vector Search and MCP Integration
Location: `src/agents/manufacturing_agent.py`

Core ResponsesAgent implementation with vector search and managed MCP integration:
```python
from uuid import uuid4
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient
from databricks_langchain import VectorSearchRetrieverTool, ChatDatabricks
import asyncio

class ManufacturingAgent(ResponsesAgent):
    """
    Manufacturing agent using ResponsesAgent interface with vector search and managed MCP servers.
    Follows best practices from docs/agents/best-practices-deploying-agents-workflow.md
    and docs/vector-search/vector-search-guide.md
    """
    
    def __init__(self):
        """Initialize agent - keep stateless as per best practices"""
        # Configuration will be set at deployment
        self.llm_endpoint = "databricks-meta-llama-3-3-70b-instruct"
        
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Process request using vector search and managed MCP tools.
        Initialize resources inside predict for proper auth isolation.
        """
        # Initialize workspace client inside predict for proper auth
        workspace_client = WorkspaceClient()
        
        # Set up vector search tools
        vector_tools = self._setup_vector_search_tools()
        
        # Set up managed MCP clients
        mcp_clients = self._setup_mcp_clients(workspace_client)
        
        # Initialize LLM
        llm = ChatDatabricks(endpoint=self.llm_endpoint)
        
        # Get user query
        user_query = request.input[-1].content if request.input else ""
        
        # Process query with vector search and MCP tools
        response_text = self._process_query(
            user_query, vector_tools, mcp_clients, llm
        )
        
        # Create response
        output_item = self.create_text_output_item(
            text=response_text,
            id=str(uuid4())
        )
        
        return ResponsesAgentResponse(output=[output_item])
    
    def _setup_vector_search_tools(self):
        """Set up vector search retriever tools"""
        tools = {}
        
        # Standard Operating Procedures retriever
        tools['sop'] = VectorSearchRetrieverTool(
            index_name="manufacturing.supply_chain.sop_index",
            tool_name="sop_retriever",
            tool_description="Retrieves standard operating procedures",
            num_results=3
        )
        
        # Incident reports retriever
        tools['incidents'] = VectorSearchRetrieverTool(
            index_name="manufacturing.supply_chain.incident_index",
            tool_name="incident_retriever", 
            tool_description="Retrieves historical incident reports",
            num_results=3
        )
        
        # Sales proposals retriever
        tools['proposals'] = VectorSearchRetrieverTool(
            index_name="manufacturing.sales.proposal_index",
            tool_name="proposal_retriever",
            tool_description="Retrieves sales proposals and customer communications",
            num_results=3
        )
        
        return tools
    
    def _setup_mcp_clients(self, workspace_client):
        """Set up managed MCP server connections"""
        host = workspace_client.config.host
        mcp_clients = {}
        
        # Vector search MCP servers
        mcp_servers = {
            "supply_chain_vs": f"{host}/api/2.0/mcp/vector-search/manufacturing/supply_chain",
            "uc_functions": f"{host}/api/2.0/mcp/functions/manufacturing/supply_chain",
            "genie": f"{host}/api/2.0/mcp/genie/supply_chain_space_id"
        }
        
        for name, server_url in mcp_servers.items():
            try:
                mcp_client = DatabricksMCPClient(
                    server_url=server_url,
                    workspace_client=workspace_client
                )
                mcp_clients[name] = mcp_client
            except Exception as e:
                print(f"Failed to connect to {name} MCP server: {e}")
        
        return mcp_clients
    
    def _process_query(self, user_query, vector_tools, mcp_clients, llm):
        """Process user query using available tools"""
        
        # Determine query intent and select appropriate tools
        if "procedure" in user_query.lower() or "sop" in user_query.lower():
            # Use SOP retriever
            results = vector_tools['sop'].invoke(user_query)
            context = f"Found {len(results)} relevant procedures: {results}"
            
        elif "incident" in user_query.lower() or "problem" in user_query.lower():
            # Use incident retriever
            results = vector_tools['incidents'].invoke(user_query)
            context = f"Found {len(results)} similar incidents: {results}"
            
        elif "proposal" in user_query.lower() or "customer" in user_query.lower():
            # Use proposals retriever
            results = vector_tools['proposals'].invoke(user_query)
            context = f"Found {len(results)} relevant proposals: {results}"
            
        else:
            # General search across multiple sources
            sop_results = vector_tools['sop'].invoke(user_query)
            incident_results = vector_tools['incidents'].invoke(user_query)
            context = f"SOPs: {sop_results[:2]}, Incidents: {incident_results[:2]}"
        
        # Generate response using LLM with retrieved context
        prompt = f"""
        User Query: {user_query}
        
        Retrieved Context: {context}
        
        Based on the retrieved information, provide a helpful response to the user's query.
        If the context contains relevant procedures or incident resolutions, reference them specifically.
        """
        
        try:
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"I found relevant information: {context[:200]}... but encountered an error generating the response: {e}"
    
    def predict_stream(self, request: ResponsesAgentRequest):
        """Streaming support for better UX"""
        # Implement streaming version following docs/mlflow/mlflow-agent-development-guide.md
        pass
```

#### 4.2 Implement Tool Discovery and Registration
Location: `src/agents/tool_manager.py`

```python
from databricks_mcp import DatabricksMCPClient

class MCPToolManager:
    """Manages tool discovery from multiple managed MCP servers"""
    
    async def discover_all_tools(self, mcp_client, server_urls):
        """Discover and catalog all available tools"""
        tools = {}
        for url in server_urls:
            await mcp_client.connect_server_session(url)
            server_tools = await mcp_client.list_tools()
            for tool in server_tools:
                tools[tool.name] = {
                    'description': tool.description,
                    'schema': tool.input_schema,
                    'server': url
                }
        return tools
```

#### 4.3 Resource Dependency Declaration
Location: `src/agents/resource_config.py`

```python
# Declare all Databricks resources for automatic credential injection
# Following docs/vector-search/vector-search-guide.md for proper resource declaration
AGENT_RESOURCES = {
    "vector_indexes": [
        "manufacturing.supply_chain.sop_index",
        "manufacturing.supply_chain.incident_index", 
        "manufacturing.sales.proposal_index",
        "manufacturing.support.ticket_index"
    ],
    "uc_functions": [
        "manufacturing.supply_chain.predict_shortage",
        "manufacturing.sales.predict_churn",
        "manufacturing.supply_chain.create_order"
    ],
    "genie_spaces": [
        "supply_chain_analytics_space_id",
        "sales_analytics_space_id"
    ],
    "serving_endpoints": [
        "databricks-meta-llama-3-3-70b-instruct"  # LLM endpoint
    ],
    "mcp_servers": [
        # Managed MCP server URLs for vector search
        "https://{host}/api/2.0/mcp/vector-search/manufacturing/supply_chain",
        "https://{host}/api/2.0/mcp/vector-search/manufacturing/sales",
        "https://{host}/api/2.0/mcp/vector-search/manufacturing/support",
        # UC Functions MCP server
        "https://{host}/api/2.0/mcp/functions/manufacturing/supply_chain",
        # Genie MCP server
        "https://{host}/api/2.0/mcp/genie/{genie_space_id}"
    ]
}

# MLflow resource objects for proper logging
from mlflow.models.resources import (
    DatabricksVectorSearchIndex, 
    DatabricksServingEndpoint,
    DatabricksFunction
)

def get_manufacturing_resources():
    """Get properly formatted MLflow resource objects"""
    return [
        # Vector Search Indexes
        DatabricksVectorSearchIndex(index_name="manufacturing.supply_chain.sop_index"),
        DatabricksVectorSearchIndex(index_name="manufacturing.supply_chain.incident_index"),
        DatabricksVectorSearchIndex(index_name="manufacturing.sales.proposal_index"),
        DatabricksVectorSearchIndex(index_name="manufacturing.support.ticket_index"),
        
        # UC Functions
        DatabricksFunction(function_name="manufacturing.supply_chain.predict_shortage"),
        DatabricksFunction(function_name="manufacturing.sales.predict_churn"),
        DatabricksFunction(function_name="manufacturing.supply_chain.create_order"),
        
        # Serving Endpoints
        DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
    ]

### Phase 5: MLflow Integration & Agent Development Workflow
**Priority: Critical | Duration: 3 days**
**Following**: The 10-step workflow from CLAUDE.md

#### 5.1 Step 1-2: Author & Instrument Agent
Location: `src/agents/mlflow_integration.py`

Implement tracing and autologging:
```python
import mlflow
from mlflow.genai.scorers import Correctness, Safety, RelevanceToQuery

def setup_mlflow_tracking():
    """Configure MLflow tracking and autologging"""
    # Set tracking URI
    mlflow.set_tracking_uri("databricks")
    
    # Set experiment
    mlflow.set_experiment("/Users/<user>/manufacturing-mcp-agent")
    
    # Enable autologging for tracing
    mlflow.langchain.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True,
        log_traces=True  # Enable tracing
    )
    
    # Enable OpenAI autologging if using OpenAI models
    mlflow.openai.autolog()
    
    return mlflow.get_tracking_uri()

@mlflow.trace
def agent_predict(request):
    """Traced agent prediction function"""
    # All operations within this function will be traced
    pass
```

#### 5.2 Steps 3-4: Golden Data & Scorers
Location: `src/evaluation/golden_data.py`

Create evaluation datasets and register scorers:
```python
from mlflow.genai.scorers import scorer

# Create golden dataset for supply chain scenarios
supply_chain_dataset = [
    {
        "inputs": {
            "input": [{"role": "user", "content": "Check inventory for steel bearings"}]
        },
        "expectations": {
            "expected_response": "Current inventory shows 150 units in warehouse A",
            "expected_tools": ["get_inventory_status"]
        }
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "Predict shortage for PART001"}]
        },
        "expectations": {
            "expected_response": "HIGH risk of shortage in 5 days",
            "expected_tools": ["predict_shortage"]
        }
    }
]

# Custom scorer for manufacturing domain
@scorer
def manufacturing_accuracy(outputs, expectations):
    """Score accuracy of manufacturing predictions"""
    # Check if correct tools were called
    # Validate numerical predictions
    # Assess recommendation quality
    score = 0.0
    
    # Implementation of scoring logic
    if "expected_tools" in expectations:
        # Check tool usage
        pass
    
    return score
```

#### 5.3 Steps 5-6: Evaluate & Log Agent
Location: `src/evaluation/evaluate_agent.py`

**Documentation**: This section implements the MLflow logging and prediction patterns based on the [Databricks ResponsesAgent LangGraph documentation](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/responses-agent-langgraph.html). The implementation follows the recommended practices for logging ResponsesAgent models with proper resource dependencies, input/output examples, and comprehensive testing.

**Key Features**:
- **Proper MLflow Logging**: Uses `mlflow.pyfunc.log_model` with ResponsesAgent-specific configuration
- **Resource Dependencies**: Logs all Databricks resources (endpoints, functions, MCP servers) for automatic credential injection
- **Input/Output Examples**: Provides proper schema examples for model signature inference
- **Local Testing**: Implements both `mlflow.pyfunc.load_model` and `mlflow.models.predict` testing patterns
- **Comprehensive Metadata**: Logs evaluation metrics, model parameters, and framework information

```python
import mlflow
from mlflow.pyfunc import log_model
from mlflow.genai.scorers import Correctness, Safety, RelevanceToQuery

def evaluate_and_log_agent(agent, golden_dataset, resources):
    """Steps 5-6: Evaluate agent quality and log with version tracking"""
    
    # Step 5: Evaluate agent performance
    results = mlflow.genai.evaluate(
        model=agent,
        data=golden_dataset,
        scorers=[
            Correctness(),
            Safety(),
            RelevanceToQuery(),
            manufacturing_accuracy  # Custom scorer
        ]
    )
    
    # Step 6: Log agent with resource dependencies and proper configuration
    with mlflow.start_run() as run:
        # Log the ResponsesAgent model with all dependencies
        logged_agent_info = mlflow.pyfunc.log_model(
            name="manufacturing_agent",
            python_model="agent.py",  # ResponsesAgent instance needs to be code
            resources=resources,  # Databricks resources (endpoints, functions, etc.)
            pip_requirements=[
                "mlflow>=3.1.0",
                "databricks-agents>=1.0.0", 
                "databricks-mcp>=1.0.0",
                "langgraph>=0.3.4",
                "databricks-langchain>=0.1.0"
            ],
            input_example={
                "input": [
                    {"role": "user", "content": "What is the current inventory status for product ABC123?"}
                ]
            },
            signature=mlflow.models.infer_signature(
                model_input={
                    "input": [
                        {"role": "user", "content": "Sample manufacturing query"}
                    ]
                },
                model_output={
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "text", "text": "Sample response"}]
                        }
                    ]
                }
            )
        )
        
        # Log evaluation metrics
        mlflow.log_metrics({
            "correctness_score": results.metrics.get("correctness", 0.0),
            "safety_score": results.metrics.get("safety", 0.0),
            "relevance_score": results.metrics.get("relevance", 0.0),
            "manufacturing_accuracy": results.metrics.get("manufacturing_accuracy", 0.0)
        })
        
        # Log model metadata
        mlflow.log_param("agent_type", "ResponsesAgent")
        mlflow.log_param("framework", "LangGraph")
        mlflow.log_param("mcp_integration", True)
        
        return run.info.run_id, logged_agent_info

def test_logged_agent(logged_agent_info):
    """Test the logged agent locally before deployment"""
    
    # Test the logged model using mlflow.pyfunc.load_model
    loaded_agent = mlflow.pyfunc.load_model(logged_agent_info.model_uri)
    
    # Test with sample manufacturing queries
    test_inputs = [
        {
            "input": [
                {"role": "user", "content": "What is the current inventory level for product SKU-12345?"}
            ]
        },
        {
            "input": [
                {"role": "user", "content": "Show me the production schedule for next week"}
            ]
        },
        {
            "input": [
                {"role": "user", "content": "Are there any quality issues with batch B789?"}
            ]
        }
    ]
    
    print("Testing logged agent locally...")
    for i, test_input in enumerate(test_inputs, 1):
        try:
            # Use mlflow.pyfunc.predict for local testing
            result = loaded_agent.predict(test_input)
            print(f"Test {i} - Success:")
            print(f"  Input: {test_input['input'][0]['content']}")
            print(f"  Output: {result['output'][0]['content'][0]['text'][:100]}...")
            print()
        except Exception as e:
            print(f"Test {i} - Error: {str(e)}")
    
    return True

def predict_with_mlflow_models(model_uri, input_data):
    """
    Alternative prediction method using mlflow.models.predict
    Useful for testing different model URIs and input formats
    """
    
    # Use mlflow.models.predict for flexible model testing
    result = mlflow.models.predict(
        model_uri=model_uri,
        input_data=input_data,
        env_manager="conda"  # or "virtualenv", "local"
    )
    
    return result

# Usage Example: Complete Evaluation and Testing Workflow
def complete_agent_evaluation_workflow():
    """
    Complete example demonstrating the evaluation, logging, and testing workflow
    Based on Databricks ResponsesAgent LangGraph patterns
    """
    
    # Step 1: Load your trained ResponsesAgent
    from src.agents.manufacturing_agent import ManufacturingResponsesAgent
    agent = ManufacturingResponsesAgent(
        mcp_server_urls=["https://workspace/api/2.0/mcp/genie/space_id"],
        llm_endpoint="databricks-meta-llama-3-3-70b-instruct"
    )
    
    # Step 2: Prepare evaluation dataset
    golden_dataset = [
        {
            "inputs": {
                "input": [{"role": "user", "content": "What is the inventory status for SKU-12345?"}]
            },
            "expected_response": "Current inventory for SKU-12345 is 150 units"
        },
        {
            "inputs": {
                "input": [{"role": "user", "content": "Show production schedule for line A"}]
            },
            "expected_response": "Production line A schedule shows 3 batches planned"
        }
    ]
    
    # Step 3: Define resources (from resource_config.py)
    from src.config.resource_config import get_manufacturing_resources
    resources = get_manufacturing_resources()
    
    # Step 4: Evaluate and log the agent
    run_id, logged_agent_info = evaluate_and_log_agent(
        agent=agent,
        golden_dataset=golden_dataset,
        resources=resources
    )
    
    print(f"Agent logged successfully. Run ID: {run_id}")
    print(f"Model URI: {logged_agent_info.model_uri}")
    
    # Step 5: Test the logged agent locally
    test_success = test_logged_agent(logged_agent_info)
    
    # Step 6: Alternative testing with mlflow.models.predict
    test_input = {
        "input": [
            {"role": "user", "content": "Check quality metrics for batch B789"}
        ]
    }
    
    result = predict_with_mlflow_models(
        model_uri=logged_agent_info.model_uri,
        input_data=test_input
    )
    
    print("Alternative prediction result:", result)
    
    return logged_agent_info

# Example: Testing different model URIs
def test_different_model_uris():
    """
    Example showing how to test models using different URI formats
    """
    
    # Test with run URI
    run_uri = "runs:/abc123def456/manufacturing_agent"
    
    # Test with registered model URI  
    registered_uri = "models:/manufacturing_agent/1"
    
    # Test with latest version
    latest_uri = "models:/manufacturing_agent/latest"
    
    test_input = {
        "input": [
            {"role": "user", "content": "What is the current production status?"}
        ]
    }
    
    for uri_name, model_uri in [
        ("Run URI", run_uri),
        ("Registered Model", registered_uri), 
        ("Latest Version", latest_uri)
    ]:
        try:
            result = predict_with_mlflow_models(model_uri, test_input)
            print(f"{uri_name} - Success: {result['output'][0]['content'][0]['text'][:50]}...")
        except Exception as e:
            print(f"{uri_name} - Error: {str(e)}")
```

#### 5.4 Steps 7-10: Deploy, Monitor, Collect Feedback & Iterate
Location: `src/deployment/deploy_monitor.py`

```python
from mlflow.deployments import get_deploy_client
from mlflow.genai import create_monitor

def deploy_and_monitor_agent(model_name, version):
    """Steps 7-10: Deploy agent and set up monitoring"""
    
    # Step 7: Register & Deploy with auth policies
    client = get_deploy_client("databricks")
    
    # Deploy with resource policies
    deployment = client.create_deployment(
        name="manufacturing-agent-endpoint",
        model_uri=f"models:/{model_name}/{version}",
        config={
            "served_entities": [{
                "name": "manufacturing-agent",
                "entity_version": version,
                "scale_to_zero_enabled": True,
                "workload_size": "Small",
                "min_provisioned_throughput": 0,
                "max_provisioned_throughput": 100
            }],
            "auto_capture_config": {
                "catalog_name": "manufacturing",
                "schema_name": "agent_logs",
                "table_name_prefix": "manufacturing_agent"
            }
        }
    )
    
    # Step 8: Monitor production with same scorers
    monitor = mlflow.genai.create_monitor(
        name="manufacturing_agent_monitor",
        endpoint=f"endpoints:/{deployment.name}",
        scorers=[
            Correctness(),
            Safety(), 
            manufacturing_accuracy
        ],
        sampling_rate=0.1  # Monitor 10% of traffic
    )
    
    # Step 9: Collect feedback (automated via Review App)
    # Step 10: Iterate based on feedback
    
    return deployment, monitor
```

### Phase 6: Optional On-Behalf-Of (OBO) Authentication
**Priority: Medium | Duration: 1 day**
**Note**: Only implement if per-user scoping is required. Default to automatic auth passthrough.

#### 6.1 Configure Auth Policies (If Needed)
Location: `src/auth/policies.py`

Based on `docs/agents/deploying-on-behalf-of-user-agents.md`:
```python
from mlflow.models.auth_policy import UserAuthPolicy, SystemAuthPolicy, AuthPolicy

def create_auth_policies(llm_endpoint: str):
    # System resources (agent's own identity)
    system_auth_policy = SystemAuthPolicy(
        resources=[DatabricksServingEndpoint(endpoint_name=llm_endpoint)]
    )
    
    # User resources (OBO)
    user_auth_policy = UserAuthPolicy(
        api_scopes=[
            "dashboards.genie",  # Genie Spaces
            "vector_search.query",  # Vector Search
            "sql.query"  # SQL queries
        ]
    )
    
    return AuthPolicy(
        system_auth_policy=system_auth_policy,
        user_auth_policy=user_auth_policy
    )
```

#### 6.2 Implement OBO Client Pattern
Location: `src/auth/obo_client.py`

```python
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials

class OBOAgentClient:
    def predict(self, messages, context=None):
        # Initialize client with user credentials INSIDE predict
        user_client = WorkspaceClient(
            credentials_strategy=ModelServingUserCredentials()
        )
        
        # Use user_client for all OBO operations
        # This ensures proper tenant isolation
        pass
```

#### 6.3 Optional OBO Vector Search Agent Example
Location: `src/agents/obo_vector_search_agent.py`

**Following**: docs/vector-search/vector-search-guide.md OBO authentication patterns

```python
from uuid import uuid4
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
from mlflow.models.auth_policy import UserAuthPolicy, SystemAuthPolicy, AuthPolicy
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials
from databricks_langchain import VectorSearchRetrieverTool, ChatDatabricks

class OBOManufacturingAgent(ResponsesAgent):
    """
    Manufacturing agent with On-Behalf-Of authentication for multi-tenant scenarios.
    Ensures vector search respects individual user permissions and Unity Catalog policies.
    """
    
    def __init__(self):
        """Initialize OBO agent - stateless design"""
        self.llm_endpoint = "databricks-meta-llama-3-3-70b-instruct"
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Process request with OBO authentication for vector search.
        Initialize user-authenticated clients inside predict method.
        """
        # Initialize OBO client inside predict method for proper user context
        user_client = WorkspaceClient(
            credentials_strategy=ModelServingUserCredentials()
        )
        
        # Set up vector search tools with user credentials
        vector_tools = self._setup_obo_vector_search_tools(user_client)
        
        # Initialize LLM with system credentials
        llm = ChatDatabricks(endpoint=self.llm_endpoint)
        
        # Get user query
        user_query = request.input[-1].content if request.input else ""
        
        # Process query with user-scoped vector search
        try:
            response_text = self._process_obo_query(user_query, vector_tools, llm)
        except PermissionError as e:
            response_text = f"Access denied: {e}. Please check your permissions for the requested data."
        except Exception as e:
            response_text = f"Error processing request: {e}"
        
        # Create response
        output_item = self.create_text_output_item(
            text=response_text,
            id=str(uuid4())
        )
        
        return ResponsesAgentResponse(output=[output_item])
    
    def _setup_obo_vector_search_tools(self, user_client):
        """Set up vector search tools with user authentication"""
        tools = {}
        
        try:
            # User-scoped SOPs retriever
            tools['sop'] = VectorSearchRetrieverTool(
                index_name="manufacturing.supply_chain.sop_index",
                tool_name="user_sop_retriever",
                tool_description="Retrieves user-accessible standard operating procedures",
                workspace_client=user_client,  # Enforces OBO access
                num_results=3
            )
            
            # User-scoped incident reports retriever
            tools['incidents'] = VectorSearchRetrieverTool(
                index_name="manufacturing.supply_chain.incident_index",
                tool_name="user_incident_retriever",
                tool_description="Retrieves user-accessible incident reports",
                workspace_client=user_client,  # Enforces OBO access
                num_results=3
            )
            
            # User-scoped proposals retriever (sales team only)
            tools['proposals'] = VectorSearchRetrieverTool(
                index_name="manufacturing.sales.proposal_index",
                tool_name="user_proposal_retriever",
                tool_description="Retrieves user-accessible sales proposals",
                workspace_client=user_client,  # Enforces OBO access
                num_results=3
            )
            
        except Exception as e:
            print(f"Error setting up OBO vector search tools: {e}")
            # Return empty tools dict if setup fails
            tools = {}
        
        return tools
    
    def _process_obo_query(self, user_query, vector_tools, llm):
        """Process query with OBO-scoped vector search"""
        
        if not vector_tools:
            return "Unable to access vector search tools. Please check your permissions."
        
        # Determine query intent and use appropriate tools
        context_parts = []
        
        if "procedure" in user_query.lower() or "sop" in user_query.lower():
            if 'sop' in vector_tools:
                results = vector_tools['sop'].invoke(user_query)
                context_parts.append(f"Procedures: {results}")
        
        elif "incident" in user_query.lower() or "problem" in user_query.lower():
            if 'incidents' in vector_tools:
                results = vector_tools['incidents'].invoke(user_query)
                context_parts.append(f"Incidents: {results}")
        
        elif "proposal" in user_query.lower() or "customer" in user_query.lower():
            if 'proposals' in vector_tools:
                results = vector_tools['proposals'].invoke(user_query)
                context_parts.append(f"Proposals: {results}")
        
        else:
            # General search with user permissions
            for tool_name, tool in vector_tools.items():
                try:
                    results = tool.invoke(user_query)
                    if results:
                        context_parts.append(f"{tool_name}: {results[:2]}")
                except Exception as e:
                    print(f"Error accessing {tool_name}: {e}")
        
        if not context_parts:
            return "No accessible information found for your query. This may be due to permission restrictions."
        
        context = " | ".join(context_parts)
        
        # Generate response with retrieved context
        prompt = f"""
        User Query: {user_query}
        
        Retrieved Context (user-scoped): {context}
        
        Based on the information you have access to, provide a helpful response.
        Note that results are filtered based on your permissions.
        """
        
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

# Resource and authentication policy configuration for OBO agent
def get_obo_agent_resources():
    """
    Define resources and authentication policies for OBO vector search agent.
    Following docs/vector-search/vector-search-guide.md OBO patterns.
    """
    
    # Declare all vector search indexes as resources
    resources = [
        DatabricksVectorSearchIndex(index_name="manufacturing.supply_chain.sop_index"),
        DatabricksVectorSearchIndex(index_name="manufacturing.supply_chain.incident_index"),
        DatabricksVectorSearchIndex(index_name="manufacturing.sales.proposal_index"),
        DatabricksVectorSearchIndex(index_name="manufacturing.support.ticket_index"),
        DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
    ]
    
    # System resources (agent identity)
    system_auth_policy = SystemAuthPolicy(resources=resources)
    
    # User authentication policy for OBO access
    user_auth_policy = UserAuthPolicy(
        api_scopes=[
            "serving.serving-endpoints",              # For LLM endpoints
            "vectorsearch.vector-search-endpoints",   # For vector search endpoints  
            "vectorsearch.vector-search-indexes",     # For vector search indexes
            "sql.query",                              # For SQL-based data access
            # Add other scopes as needed
        ]
    )
    
    # Combined authentication policy
    auth_policy = AuthPolicy(
        system_auth_policy=system_auth_policy,
        user_auth_policy=user_auth_policy
    )
    
    return resources, auth_policy

# MLflow logging for OBO agent
def log_obo_agent():
    """Log OBO agent with proper resource and authentication policies"""
    import mlflow
    
    resources, auth_policy = get_obo_agent_resources()
    
    with mlflow.start_run():
        logged_agent_info = mlflow.pyfunc.log_model(
            name="obo_manufacturing_agent",
            python_model="obo_vector_search_agent.py",
            auth_policy=auth_policy,  # Includes both system and user policies
            pip_requirements=[
                "databricks-langchain",
                "databricks-ai-bridge", 
                "databricks-mcp",
                "mlflow>=3.1.0"
            ],
            input_example={
                "input": [
                    {"role": "user", "content": "What procedures should I follow for quality control?"}
                ]
            }
        )
    
    return logged_agent_info
```

#### 6.4 Test Tenant Isolation
Location: `tests/test_tenant_isolation.py`

Test scenarios:
- User A cannot access User B's data
- Proper filtering of Genie Space results  
- Vector Search respects permissions
- UC Functions honor row-level security
- OBO vector search enforces user permissions

### Phase 7: Optional Custom MCP Server (Only if Needed)
**Priority: Low | Duration: 2 days**
**Note**: Only implement if you have specialized logic not covered by managed MCP servers

#### 7.1 Custom MCP Server for External APIs
Location: `src/mcp_servers/custom_external.py`

Only if integrating with non-Databricks systems:
```python
from mcp.server import Server, Tool
from databricks_mcp import DatabricksMCPServer

class ExternalAPIMCPServer(DatabricksMCPServer):
    """Custom MCP server for external integrations"""
    
    def __init__(self):
        super().__init__()
        # Only add tools not available via managed MCP
        self.register_external_tools()
    
    def register_external_tools(self):
        # Example: External ERP system integration
        self.add_tool(Tool(
            name="query_external_erp",
            description="Query external ERP system",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            },
            handler=self.query_external_erp
        ))
```

#### 7.2 Deploy Custom MCP Server on Databricks Apps
```yaml
# src/mcp_servers/app.yaml
name: manufacturing-custom-mcp
description: Custom MCP server for external integrations only
entry_point: main.py
environment:
  DATABRICKS_HOST: ${DATABRICKS_HOST}
  DATABRICKS_TOKEN: ${DATABRICKS_TOKEN}
```

### Phase 8: Databricks App UI Development
**Priority: Medium | Duration: 3 days**

#### 8.1 Create App Structure
Location: `src/app/`

Directory structure:
```
src/app/
├── app.yaml           # Databricks App configuration
├── requirements.txt   # UI dependencies
├── main.py           # FastAPI backend
├── static/           # Frontend assets
│   ├── css/
│   ├── js/
│   └── images/
├── templates/        # HTML templates
│   ├── index.html
│   ├── supply_chain.html
│   └── sales.html
└── api/              # API endpoints
    ├── auth.py
    ├── agent.py
    └── analytics.py
```

#### 8.2 Build Supply Chain Manager Interface
Features:
- Chat interface for natural language queries
- Real-time inventory dashboard
- Shipment tracking map
- Alert notifications panel
- Prediction visualizations

#### 8.3 Build Sales Representative Interface
Features:
- Customer 360 view
- KPI dashboard with trends
- Churn risk heatmap
- Proposal generator
- Lead scoring display

#### 8.4 Implement OAuth/SSO Authentication
```python
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from databricks.sdk import WorkspaceClient

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Validate OAuth token
    # Return user identity for OBO
    pass
```

### Phase 9: Deployment & Model Serving
**Priority: Critical | Duration: 2 days**

#### 9.1 Configure Asset Bundle
Location: `databricks.yml`

Update configuration:
```yaml
bundle:
  name: manufacturing-mcp-agent
  
resources:
  jobs:
    data_pipeline:
      name: manufacturing-data-pipeline
      tasks:
        - task_key: generate_data
          python_wheel_task:
            package_name: manufacturing_mcp
            entry_point: generate_data
            
  models:
    manufacturing_agent:
      name: manufacturing-mcp-agent
      config:
        serving_config:
          served_entities:
            - entity_name: manufacturing-agent
              entity_version: 1
              scale_to_zero_enabled: true
              
  apps:
    mcp_server:
      name: manufacturing-mcp-server
      source_code_path: ./src/mcp_servers
      
    ui:
      name: manufacturing-ui
      source_code_path: ./src/app
```

#### 9.2 Register and Deploy Agent
```bash
# Register model
python src/agents/register_agent.py

# Deploy to Model Serving
databricks models serve --model-name manufacturing-mcp-agent --version 1
```

#### 9.3 Deploy Custom MCP Server
```bash
# Create Databricks App
databricks apps create manufacturing-mcp-server

# Deploy
databricks apps deploy manufacturing-mcp-server --source-code-path ./src/mcp_servers
```

#### 9.4 Deploy UI Application
```bash
# Create UI App
databricks apps create manufacturing-ui

# Deploy
databricks apps deploy manufacturing-ui --source-code-path ./src/app
```

### Phase 10: Testing & Validation
**Priority: High | Duration: 2 days**

#### 10.1 Unit Tests
Location: `tests/unit/`

Test coverage for:
- MCP tool wrappers
- Agent state management
- Auth policy configuration
- Data generation utilities
- UC function logic

#### 10.2 Integration Tests
Location: `tests/integration/`

End-to-end test scenarios:
- Supply chain shortage prediction workflow
- Sales churn analysis workflow
- Multi-turn conversation flow
- Cross-MCP server tool orchestration

#### 10.3 Performance Testing
Location: `tests/performance/`

Benchmarks:
- Response time under load (target: <2s p95)
- Concurrent user handling (target: 100+ users)
- Tool call latency
- Vector search performance

#### 10.4 Security Validation
Location: `tests/security/`

Security tests:
- Tenant data isolation verification
- OBO permission boundaries
- SQL injection prevention
- Authentication flow validation

## Simplified Implementation Priority Order

### Week 1: Foundation & Core Agent
1. **Day 1**: Phase 1 - Environment Setup with pinned versions
2. **Days 2-3**: Phase 2 - Data Layer (Delta tables, sample data)
3. **Days 4-5**: Phase 3 - UC Functions for managed MCP

### Week 2: Agent Development (10-Step Workflow)
1. **Days 6-7**: Phase 4 - ResponsesAgent implementation
2. **Days 8-10**: Phase 5 - MLflow integration (tracing, evaluation, deployment)

### Week 3: Testing & Optional Features
1. **Days 11-12**: Phase 10 - Testing & validation
2. **Day 13**: Phase 6 - OBO authentication (only if needed)
3. **Days 14-15**: Phase 7 - Custom MCP server (only if needed)

### Week 4: UI & Production
1. **Days 16-18**: Phase 8 - Databricks App UI
2. **Days 19-20**: Phase 9 - Production deployment & monitoring

## Key Implementation Files to Create (Prioritized)

### Phase 1-2: Foundation Files
- `requirements.txt` - Pinned dependencies
- `src/data/create_tables.py` - Delta table creation
- `src/data/generate_data.py` - Synthetic data generation
- `src/data/create_vector_indexes.py` - Vector index setup

### Phase 3: UC Functions (for Managed MCP)
- `src/uc_functions/predict_shortage.py` - Shortage prediction
- `src/uc_functions/predict_churn.py` - Churn prediction
- `src/uc_functions/create_order.py` - Order automation
- `src/uc_functions/register_all_functions.py` - Function registration

### Phase 4: ResponsesAgent Files (Priority)
- `src/agents/manufacturing_agent.py` - Main ResponsesAgent implementation
- `src/agents/tool_manager.py` - MCP tool discovery
- `src/agents/resource_config.py` - Resource dependency declaration

### Phase 5: MLflow Integration (10-Step Workflow)
- `src/agents/mlflow_integration.py` - Tracing & autologging setup
- `src/evaluation/golden_data.py` - Evaluation datasets
- `src/evaluation/evaluate_agent.py` - Agent evaluation
- `src/deployment/deploy_monitor.py` - Deployment & monitoring

### Phase 6: Optional Authentication
- `src/auth/policies.py` - Auth policy configuration (if OBO needed)
- `src/auth/obo_client.py` - OBO client implementation (if needed)

### Phase 7: Optional Custom MCP
- `src/mcp_servers/custom_external.py` - External API integration (if needed)
- `src/mcp_servers/app.yaml` - Databricks App config (if needed)

### Phase 8: UI Files
- `src/app/main.py` - FastAPI backend
- `src/app/templates/index.html` - Main UI
- `src/app/api/agent.py` - Agent API endpoints

### Phase 10: Test Files
- `tests/unit/test_agent.py` - ResponsesAgent unit tests
- `tests/integration/test_mcp_integration.py` - MCP integration tests
- `tests/test_golden_data.py` - Evaluation dataset tests

## Success Criteria

1. **Functional Requirements**
   - ✅ ResponsesAgent interface fully implemented
   - ✅ All three managed MCP servers integrated (Vector Search, UC Functions, Genie)
   - ✅ Both use cases working (Supply Chain & Sales)
   - ✅ MLflow 3.0 tracing, evaluation, and monitoring operational
   - ✅ 10-step agent workflow completed

2. **Performance Requirements**
   - ✅ Response time < 2 seconds (p95)
   - ✅ Streaming support for better UX
   - ✅ Support 100+ concurrent users
   - ✅ Vector search latency < 500ms

3. **Security Requirements**
   - ✅ Automatic auth passthrough for managed resources
   - ✅ Unity Catalog permissions enforced
   - ✅ Resource dependencies declared and logged
   - ✅ OBO authentication available when needed

4. **Quality Requirements**
   - ✅ 80%+ test coverage
   - ✅ Golden datasets created and versioned
   - ✅ Custom scorers implemented and passing
   - ✅ Production monitoring with same scorers

## Next Steps

1. **Start simple**: Begin with Phase 1 environment setup
2. **Prioritize managed MCP**: Use managed servers before custom
3. **Follow the 10-step workflow**: From CLAUDE.md for agent development
4. **Test continuously**: Each phase before moving to next
5. **Document resource dependencies**: For automatic credential injection

## Key Documentation References

### Agent Development Guides
- [docs/agents/best-practices-deploying-agents-workflow.md](./docs/agents/best-practices-deploying-agents-workflow.md) - 10-step workflow
- [docs/agents/databricks-agent-uc-tools.md](./docs/agents/databricks-agent-uc-tools.md) - UC Functions integration
- [docs/agents/langgraph-mcp-agent.md](./docs/agents/langgraph-mcp-agent.md) - LangGraph patterns
- [docs/agents/deploying-on-behalf-of-user-agents.md](./docs/agents/deploying-on-behalf-of-user-agents.md) - OBO auth

### MCP Guides
- [docs/mcp/managed-mcp-servers-guide.md](./docs/mcp/managed-mcp-servers-guide.md) - Managed MCP implementation
- [docs/mcp/databricks-mcp-documentation.md](./docs/mcp/databricks-mcp-documentation.md) - Core MCP concepts

### MLflow Guides
- [docs/mlflow/mlflow-agent-development-guide.md](./docs/mlflow/mlflow-agent-development-guide.md) - ResponsesAgent interface
- [docs/mlflow/mlflow3-documentation-guide.md](./docs/mlflow/mlflow3-documentation-guide.md) - MLflow 3.0 features