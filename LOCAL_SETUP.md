# Local Development Setup - Databricks MFG MCP Examples

## ✅ Setup Complete!

Your local environment is now configured for Databricks development with:
- **OAuth authentication** via CLI profile (`aws-apps`)
- **Databricks Connect** with serverless compute
- **MLflow** tracking to workspace and Unity Catalog
- **Environment variables** in `.env` (safe, non-secret defaults)

## Quick Verification

Run the smoke tests to verify your setup:

```bash
# Test Databricks Connect
python test_databricks_connect.py

# Test MLflow configuration
python test_mlflow.py
```

## Configuration Details

### Authentication
- Using OAuth profile: `aws-apps`
- Workspace: `https://e2-demo-field-eng.cloud.databricks.com/`
- User: `alex.miller@databricks.com`

### Environment Variables (.env)
```bash
DATABRICKS_CONFIG_PROFILE=aws-apps           # CLI profile name
DATABRICKS_HOST=https://e2-demo-field-eng.cloud.databricks.com/
DATABRICKS_SERVERLESS_COMPUTE_ID=auto       # Serverless compute enabled
UC_DEFAULT_CATALOG=users                    # Unity Catalog defaults
UC_DEFAULT_SCHEMA=alex_miller
MLFLOW_TRACKING_URI=databricks              # MLflow to workspace
MLFLOW_REGISTRY_URI=databricks-uc           # Unity Catalog models
MLFLOW_EXPERIMENT_NAME=/Users/alex.miller@databricks.com/mfg-mcp-agent
```

### Installed Packages
- `databricks-connect` (16.1.6) - Spark development on remote clusters
- `mlflow[databricks]` (3.3.1) - ML lifecycle management
- `databricks-sdk` (0.64.0) - Databricks API client
- `python-dotenv` (1.1.1) - Environment variable management

## Using in Your Code

### Databricks Connect Example
```python
import os
from dotenv import load_dotenv
from databricks.connect import DatabricksSession

load_dotenv()

# Create Spark session with serverless compute
spark = DatabricksSession.builder \
    .profile(os.getenv("DATABRICKS_CONFIG_PROFILE")) \
    .serverless(True) \
    .getOrCreate()

# Use Unity Catalog
catalog = os.getenv("UC_DEFAULT_CATALOG")
schema = os.getenv("UC_DEFAULT_SCHEMA")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# Your Spark code here
df = spark.range(1, 100)
df.show()
```

### MLflow Example
```python
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

# Log a run
with mlflow.start_run():
    mlflow.log_param("model_type", "mcp_agent")
    mlflow.log_metric("accuracy", 0.95)
    # Your ML code here
```

## Troubleshooting

### Authentication Issues
```bash
# Re-authenticate with OAuth
databricks auth login --profile aws-apps

# Verify authentication
databricks current-user me --profile aws-apps
```

### Version Compatibility
Ensure Databricks Connect version matches or is lower than your cluster's DBR version:
```bash
# Check installed version
pip show databricks-connect

# Upgrade if needed
pip install -U databricks-connect
```

### Profile Not Found
If you get a "profile not found" error:
1. Check `~/.databrickscfg` exists
2. Verify profile name matches `.env` setting
3. Re-run `databricks auth login --profile aws-apps`

## Next Steps

1. **Deploy Delta tables**: Set up sample manufacturing data
2. **Configure MCP servers**: Set up Genie, Vector Search, UC Functions
3. **Develop agents**: Build and test MCP-enabled agents
4. **Deploy to Model Serving**: Production deployment

## References

- [Databricks Connect Documentation](https://docs.databricks.com/en/dev-tools/databricks-connect/index.html)
- [MLflow on Databricks](https://docs.databricks.com/en/mlflow/index.html)
- [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
- [MCP Documentation](./docs/mcp/)