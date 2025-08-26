# Databricks Agent Framework with Unity Catalog Tools

**Source**: Multiple URLs
- https://docs.databricks.com/aws/en/generative-ai/agent-framework/create-custom-tool
- https://docs.databricks.com/aws/en/generative-ai/agent-framework/structured-retrieval-tools  
- https://docs.databricks.com/aws/en/generative-ai/agent-framework/langchain-uc-integration

**Domain**: docs.databricks.com
**Fetched**: 2025-08-26
**Type**: Agent Framework Integration Guide

## Overview

This documentation covers the integration of Unity Catalog functions with the Databricks Agent Framework, enabling custom tool creation, structured data retrieval, and LangChain integration for AI agents in a governed multi-tenant environment.

## Key Prerequisites

- **Databricks Runtime**: 15.0 or above
- **Python**: 3.10 or above  
- **Compute**: Serverless generic compute enabled
- **Access**: Unity Catalog with appropriate permissions

## Required Dependencies

```python
%pip install unitycatalog-ai[databricks]
%pip install unitycatalog-langchain[databricks]
%pip install databricks-langchain
```

## Creating Custom Unity Catalog Tools

### 1. Initialize Function Client

```python
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
client = DatabricksFunctionClient()

# Alternative initialization
from unitycatalog.ai.core.base import get_uc_function_client
client = get_uc_function_client()
```

### 2. Function Definition Requirements

**Critical Requirements**:
- Must have complete type hints
- No variable arguments (*args, **kwargs)
- Descriptive Google-style docstring
- All dependencies imported within function scope

```python
def add_numbers(number_1: float, number_2: float) -> float:
    """
    A function that adds two floating point numbers.
    
    Args:
        number_1 (float): First number to add
        number_2 (float): Second number to add
    
    Returns:
        float: Sum of input numbers
    """
    return number_1 + number_2
```

### 3. Function Registration

```python
function_info = client.create_python_function(
    func=add_numbers,
    catalog="my_catalog",
    schema="my_schema", 
    replace=True
)
```

### 4. Tool Integration with Agents

```python
from databricks_langchain import UCFunctionToolkit

func_name = f"{CATALOG}.{SCHEMA}.add_numbers"
toolkit = UCFunctionToolkit(function_names=[func_name])
tools = toolkit.tools
```

## Structured Data Retrieval

### SQL Unity Catalog Functions

For predictable, structured queries with known formats:

```sql
CREATE OR REPLACE FUNCTION main.default.lookup_customer_info(
  customer_name STRING COMMENT 'Name of the customer whose info to look up'
)
RETURNS STRING
COMMENT 'Returns metadata about a particular customer'
RETURN SELECT CONCAT(
    'Customer ID: ', customer_id, ', ',
    'Customer Email: ', customer_email
  )
  FROM main.default.customer_data
  WHERE customer_name = customer_name
  LIMIT 1;
```

**Use Cases**:
- Fixed format queries
- Parameterized data retrieval
- Direct SQL table access
- High-performance structured lookups

### Genie Multi-Agent System

For complex, natural language queries:

**Features**:
- Flexible natural language query processing
- Multi-table reasoning capabilities
- Complex data relationships
- Currently in Public Preview

**Implementation Decision**:
- Use **Unity Catalog SQL Functions** for structured, predictable queries
- Use **Genie** for complex, dynamic data retrieval scenarios

## LangChain Integration Patterns

### Complete Integration Example

```python
# 1. Import required modules
from unitycatalog.ai.core.base import get_uc_function_client
from databricks_langchain import UCFunctionToolkit

# 2. Initialize client
client = get_uc_function_client()

# 3. Define and register function
def calculate_inventory_shortage(item_id: str, demand_forecast: float) -> dict:
    """
    Calculate potential inventory shortage for manufacturing item.
    
    Args:
        item_id (str): Manufacturing item identifier
        demand_forecast (float): Predicted demand units
        
    Returns:
        dict: Shortage analysis with recommendations
    """
    # Import dependencies within function
    import json
    
    # Function logic here
    result = {
        "item_id": item_id,
        "current_stock": 100,  # Query from Delta table
        "demand_forecast": demand_forecast,
        "shortage_risk": "HIGH" if demand_forecast > 100 else "LOW",
        "recommended_action": "Increase production" if demand_forecast > 100 else "Maintain current levels"
    }
    
    return json.dumps(result)

# 4. Register function
function_info = client.create_python_function(
    func=calculate_inventory_shortage,
    catalog="manufacturing",
    schema="supply_chain",
    replace=True
)

# 5. Create toolkit and tools
func_name = "manufacturing.supply_chain.calculate_inventory_shortage" 
toolkit = UCFunctionToolkit(function_names=[func_name])
tools = toolkit.tools
```

## Execution Modes

### Serverless Mode (Production)
- **Default mode** for production workloads
- Uses remote Databricks compute
- Governed by Unity Catalog security
- Automatic scaling and management

### Local Mode (Development)
- Executes in local subprocess
- Useful for debugging and testing
- Limited by local environment constraints

## Configuration and Environment Variables

```python
# Key environment variables for execution control
EXECUTOR_MAX_CPU_TIME_LIMIT = 10  # seconds
```

## Security and Multi-Tenancy

### Unity Catalog Integration
- **Catalog-level isolation**: Each tenant gets dedicated catalog
- **Schema-level organization**: Logical grouping within catalogs  
- **Function-level permissions**: Granular access control
- **Audit logging**: Complete lineage and usage tracking

### On-Behalf-Of User Authentication
```python
# Functions execute with caller's permissions
# Automatic multi-tenant isolation through UC governance
# No additional authentication configuration needed
```

## Best Practices

### Function Design
1. **Clear Documentation**: Use comprehensive Google-style docstrings
2. **Type Safety**: Always include complete type hints
3. **Input Validation**: Validate parameters within functions
4. **Error Handling**: Implement robust error handling
5. **Dependency Management**: Import all dependencies within function scope

### Performance Optimization
1. **Function Caching**: Leverage Unity Catalog caching mechanisms
2. **Batch Operations**: Design functions for batch processing when possible
3. **Resource Limits**: Set appropriate CPU and memory limits
4. **Delta Lake Integration**: Use Delta tables for data access

### Security Considerations
1. **Principle of Least Privilege**: Grant minimal required permissions
2. **Sensitive Data**: Never log sensitive information
3. **Input Sanitization**: Validate and sanitize all inputs
4. **Catalog Boundaries**: Respect multi-tenant isolation

## Integration with MLflow 3.0

```python
# Automatic logging and tracing
import mlflow
mlflow.langchain.autolog()

# Function execution automatically traced
# Agent interactions logged
# Performance metrics captured
```

## Common Implementation Patterns

### Manufacturing Use Case Example
```python
def check_equipment_status(equipment_id: str) -> str:
    """
    Check manufacturing equipment operational status.
    
    Args:
        equipment_id (str): Unique equipment identifier
        
    Returns:
        str: JSON string with equipment status and maintenance recommendations
    """
    import json
    from datetime import datetime
    
    # In real implementation, query Delta table
    status = {
        "equipment_id": equipment_id,
        "status": "OPERATIONAL",
        "last_maintenance": "2025-08-20",
        "next_maintenance_due": "2025-09-20",
        "efficiency": 0.94,
        "alerts": []
    }
    
    return json.dumps(status)
```

### Sales Analytics Use Case Example  
```python
def analyze_customer_churn_risk(customer_id: str, days_back: int = 90) -> str:
    """
    Analyze customer churn risk based on recent activity.
    
    Args:
        customer_id (str): Customer identifier
        days_back (int): Days to look back for analysis
        
    Returns:
        str: JSON with churn risk assessment and next best actions
    """
    import json
    
    # In real implementation, run ML model via UC Functions
    analysis = {
        "customer_id": customer_id,
        "churn_probability": 0.23,
        "risk_level": "MEDIUM", 
        "key_factors": ["Decreased order frequency", "Support tickets"],
        "recommended_actions": ["Personal outreach", "Discount offer"]
    }
    
    return json.dumps(analysis)
```

## Error Handling and Debugging

```python
def robust_function_example(param: str) -> str:
    """Example function with proper error handling."""
    try:
        # Validate input
        if not param or not isinstance(param, str):
            raise ValueError("Parameter must be a non-empty string")
            
        # Function logic
        result = process_data(param)
        
        return result
        
    except Exception as e:
        # Log error for debugging
        error_response = {
            "error": str(e),
            "parameter": param,
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_response)
```

## Related Resources

- Reference **CLAUDE.md** for project-specific implementation guidelines
- Check **docs/managed-mcp-servers-guide.md** for MCP server patterns
- See **docs/deploying-on-behalf-of-user-agents.md** for multi-tenant authentication
- Review **docs/mlflow3-documentation-guide.md** for observability setup

## Implementation Notes

1. **Multi-tenant Architecture**: Use Unity Catalog governance for automatic tenant isolation
2. **Function Versioning**: Use `replace=True` for development, version control for production
3. **Testing Strategy**: Test functions locally before deploying to Unity Catalog
4. **Monitoring**: Leverage MLflow 3.0 for comprehensive agent and tool monitoring
5. **Scalability**: Design functions for serverless execution patterns