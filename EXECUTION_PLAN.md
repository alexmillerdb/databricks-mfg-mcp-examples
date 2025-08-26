# Databricks Manufacturing MCP Examples - Execution Plan

This document provides a comprehensive implementation plan for building the multi-tenant Supply Chain & Sales agentic system on Databricks using Model Context Protocol (MCP). This plan should be used as a reference guide for Claude Code or developers implementing the system.

## Project Overview

Building a production-ready demonstration of MCP-based agent orchestration for manufacturing use cases with:
- **Two primary personas**: Supply Chain Manager and Sales Representative
- **Multi-tenant architecture**: Using Unity Catalog and On-Behalf-Of (OBO) authentication
- **Managed MCP servers**: Vector Search, Genie Spaces, UC Functions
- **Custom MCP server**: External APIs and IoT connectors
- **Full observability**: MLflow 3.0 integration for tracing and evaluation

## Implementation Phases

### Phase 1: Databricks Workspace & Environment Setup
**Priority: Critical | Duration: 1 day**

#### 1.1 Configure Databricks CLI Authentication
```bash
# Install Databricks CLI
pip install databricks-cli

# Configure authentication
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

-- Grant permissions for multi-tenant access
GRANT USAGE ON CATALOG manufacturing TO `account users`;
```

#### 1.3 Set Up Python Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Verify installation
python verify_setup.py
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

Indexes to create:
- `manufacturing.supply_chain.sop_index` - Standard Operating Procedures
- `manufacturing.supply_chain.incident_index` - Incident reports
- `manufacturing.sales.proposal_index` - Sales proposals
- `manufacturing.support.ticket_index` - Support ticket history

Configuration:
- Use embedding model: `databricks-gte-large-en`
- Similarity metric: cosine
- Enable delta sync for real-time updates

#### 2.4 Configure Genie Spaces
Manual steps in Databricks UI:
1. Create Genie Space for Supply Chain Analytics
   - Add tables: inventory, shipments, suppliers
   - Configure natural language descriptions
2. Create Genie Space for Sales Analytics
   - Add tables: transactions, customers
   - Configure KPI definitions

### Phase 3: Unity Catalog Functions Development
**Priority: High | Duration: 2 days**

#### 3.1 Create predict_shortage UC Function
Location: `src/uc_functions/predict_shortage.py`

Python function implementation:
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

### Phase 4: MCP Server Implementations
**Priority: High | Duration: 3 days**

#### 4.1 Implement Custom MCP Server Base
Location: `src/mcp_servers/custom_server.py`

Core components:
```python
from mcp.server import Server, Tool
from databricks_mcp import DatabricksMCPServer

class ManufacturingMCPServer(DatabricksMCPServer):
    def __init__(self):
        super().__init__()
        self.register_tools()
    
    def register_tools(self):
        # Register IoT tools
        self.add_tool(Tool(
            name="query_sensor_data",
            description="Query IoT sensor telemetry",
            input_schema={...},
            handler=self.query_sensor_data
        ))
        
        # Register ticketing tools
        self.add_tool(Tool(
            name="create_ticket",
            description="Create support ticket",
            input_schema={...},
            handler=self.create_ticket
        ))
```

#### 4.2 Add IoT Sensor Connector
Location: `src/mcp_servers/iot_connector.py`

Features:
- Query real-time sensor data
- Detect anomalies in telemetry
- Generate maintenance alerts
- Aggregate sensor statistics

#### 4.3 Add Ticketing System Integration
Location: `src/mcp_servers/ticketing_connector.py`

Features:
- Create new tickets
- Query ticket status
- Update ticket resolution
- Search historical tickets

### Phase 5: Agent Orchestrator Development
**Priority: Critical | Duration: 4 days**

#### 5.1 Implement LangGraph Agent Orchestrator
Location: `src/agents/orchestrator.py`

Core implementation based on `docs/langgraph-mcp-agent.md`:
```python
from langgraph.graph import StateGraph
from databricks_mcp import DatabricksMCPClient
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tools: List[Tool]
    next_action: Optional[str]
    
class ManufacturingAgent:
    def __init__(self, mcp_server_urls: List[str], llm_endpoint: str):
        self.mcp_server_urls = mcp_server_urls
        self.llm_endpoint = llm_endpoint
        self.mcp_client = None
        self.tools = []
        self.graph = None
        
    async def initialize(self):
        # Initialize MCP clients
        # Discover and register tools
        # Build LangGraph workflow
        pass
```

#### 5.2 Create MCPTool Wrapper Class
Location: `src/agents/mcp_tool_wrapper.py`

Convert MCP tools to LangChain-compatible format:
```python
class MCPToolWrapper:
    def __init__(self, mcp_client, tool_name, tool_description, tool_schema):
        self.mcp_client = mcp_client
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_schema = tool_schema
    
    async def _arun(self, **kwargs):
        # Convert arguments and execute MCP tool
        pass
    
    def to_langchain_tool(self) -> Tool:
        # Return LangChain-compatible tool
        pass
```

#### 5.3 Implement Multi-MCP Server Connection
Location: `src/agents/multi_server_manager.py`

Features:
- Parallel server discovery
- Tool deduplication
- Connection pooling
- Retry logic with exponential backoff

#### 5.4 Add Conversation State Management
Location: `src/agents/state_manager.py`

Implement:
- Conversation history tracking
- Context window management
- State persistence for long conversations
- Memory summarization for context efficiency

### Phase 6: On-Behalf-Of (OBO) Authentication
**Priority: Critical | Duration: 2 days**

#### 6.1 Configure Auth Policies
Location: `src/auth/policies.py`

Based on `docs/deploying-on-behalf-of-user-agents.md`:
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

#### 6.3 Test Tenant Isolation
Location: `tests/test_tenant_isolation.py`

Test scenarios:
- User A cannot access User B's data
- Proper filtering of Genie Space results
- Vector Search respects permissions
- UC Functions honor row-level security

### Phase 7: MLflow 3.0 Integration
**Priority: High | Duration: 3 days**

#### 7.1 Configure MLflow Tracing
Location: `src/monitoring/mlflow_config.py`

Based on `docs/mlflow3-documentation-guide.md`:
```python
import mlflow

def configure_mlflow():
    # Set tracking URI
    mlflow.set_tracking_uri("databricks")
    
    # Set experiment
    mlflow.set_experiment("/Users/<user>/manufacturing-mcp-agent")
    
    # Enable autologging
    mlflow.openai.autolog()
    mlflow.langchain.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True,
        log_traces=True
    )
```

#### 7.2 Implement Custom Scorers
Location: `src/monitoring/scorers.py`

Manufacturing-specific scorers:
```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer
def supply_chain_accuracy(outputs, expectations):
    """Score accuracy of supply chain predictions"""
    # Check if shortage predictions match actual
    # Validate inventory recommendations
    # Assess delivery time estimates
    pass

@scorer
def sales_relevance(outputs, expectations):
    """Score relevance of sales insights"""
    # Check KPI accuracy
    # Validate churn predictions
    # Assess proposal quality
    pass
```

#### 7.3 Create Evaluation Datasets
Location: `src/evaluation/datasets.py`

Datasets to create:
- Supply chain scenarios (50+ examples)
- Sales interactions (50+ examples)
- Edge cases and error conditions
- Multi-turn conversations

#### 7.4 Set Up Production Monitoring
Location: `src/monitoring/production_monitor.py`

```python
from mlflow.genai import create_monitor

def setup_monitoring(endpoint_name: str):
    monitor = mlflow.genai.create_monitor(
        name="manufacturing_agent_monitor",
        endpoint=f"endpoints:/{endpoint_name}",
        scorers=[
            Safety(),
            RelevanceToQuery(),
            supply_chain_accuracy,
            sales_relevance
        ],
        sampling_rate=0.1  # Monitor 10% of traffic
    )
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

## Implementation Priority Order

### Week 1: Foundation
1. Phase 1: Environment Setup (Day 1)
2. Phase 2.1-2.2: Delta Tables & Data Generation (Days 2-3)
3. Phase 5.1-5.2: Core Agent Implementation (Days 4-5)

### Week 2: Core Functionality
1. Phase 3: UC Functions (Days 6-7)
2. Phase 4: Custom MCP Server (Days 8-10)

### Week 3: Advanced Features
1. Phase 6: OBO Authentication (Days 11-12)
2. Phase 7: MLflow Integration (Days 13-15)

### Week 4: UI & Deployment
1. Phase 8: UI Development (Days 16-18)
2. Phase 9: Deployment (Days 19-20)

### Week 5: Testing & Polish
1. Phase 10: Testing (Days 21-22)
2. Phase 2.3-2.4: Vector Search & Genie (Days 23-24)
3. Final integration and documentation (Day 25)

## Key Implementation Files to Create

### Core Agent Files
- `src/agents/orchestrator.py` - Main agent orchestrator
- `src/agents/mcp_tool_wrapper.py` - MCP to LangChain wrapper
- `src/agents/state_manager.py` - Conversation state management
- `src/agents/register_agent.py` - Agent registration script

### MCP Server Files
- `src/mcp_servers/custom_server.py` - Custom MCP server base
- `src/mcp_servers/iot_connector.py` - IoT integration
- `src/mcp_servers/ticketing_connector.py` - Ticketing system
- `src/mcp_servers/app.yaml` - Databricks App config

### Data Layer Files
- `src/data/create_tables.py` - Delta table creation
- `src/data/generate_data.py` - Synthetic data generation
- `src/data/create_vector_indexes.py` - Vector index setup

### UC Function Files
- `src/uc_functions/predict_shortage.py` - Shortage prediction
- `src/uc_functions/predict_churn.py` - Churn prediction
- `src/uc_functions/create_order.py` - Order automation

### Authentication Files
- `src/auth/policies.py` - Auth policy configuration
- `src/auth/obo_client.py` - OBO client implementation

### Monitoring Files
- `src/monitoring/mlflow_config.py` - MLflow configuration
- `src/monitoring/scorers.py` - Custom scorers
- `src/monitoring/production_monitor.py` - Production monitoring

### UI Files
- `src/app/main.py` - FastAPI backend
- `src/app/templates/supply_chain.html` - Supply chain UI
- `src/app/templates/sales.html` - Sales UI
- `src/app/api/agent.py` - Agent API endpoints

### Test Files
- `tests/unit/test_agent.py` - Agent unit tests
- `tests/integration/test_workflows.py` - Workflow tests
- `tests/security/test_tenant_isolation.py` - Security tests

## Success Criteria

1. **Functional Requirements**
   - ✅ Multi-tenant support with proper isolation
   - ✅ All three managed MCP servers integrated
   - ✅ Custom MCP server deployed and functional
   - ✅ Both use cases (Supply Chain & Sales) working
   - ✅ MLflow tracing and evaluation operational

2. **Performance Requirements**
   - ✅ Response time < 2 seconds (p95)
   - ✅ Support 100+ concurrent users
   - ✅ Vector search latency < 500ms

3. **Security Requirements**
   - ✅ OBO authentication working
   - ✅ Unity Catalog permissions enforced
   - ✅ No cross-tenant data leakage

4. **Quality Requirements**
   - ✅ 80%+ test coverage
   - ✅ All scorers passing thresholds
   - ✅ Documentation complete

## Next Steps

1. Begin with Phase 1 environment setup
2. Follow the priority order for implementation
3. Test each phase before moving to the next
4. Document any deviations or improvements
5. Create demo scripts for showcasing the system

## References

- [CLAUDE.md](./CLAUDE.md) - Project context and guidelines
- [PRD.md](./PRD.md) - Product requirements
- [docs/managed-mcp-servers-guide.md](./docs/managed-mcp-servers-guide.md) - MCP server implementation
- [docs/langgraph-mcp-agent.md](./docs/langgraph-mcp-agent.md) - LangGraph agent patterns
- [docs/deploying-on-behalf-of-user-agents.md](./docs/deploying-on-behalf-of-user-agents.md) - OBO authentication
- [docs/mlflow3-documentation-guide.md](./docs/mlflow3-documentation-guide.md) - MLflow 3.0 integration