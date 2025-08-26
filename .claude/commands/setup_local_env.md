---
description: Setup .env + Databricks CLI OAuth + Databricks Connect (local IDE)
argument-hint: [profile-name]
---

## Why this matters
Use OAuth user-to-machine (U2M) with a CLI configuration profile to avoid storing long-lived tokens in .env while enabling local development against Databricks compute[10]. Databricks Connect lets local IDEs run Spark code on remote Databricks clusters or serverless compute when versions are compatible[2]. Prefer profile-based auth for Connect and only set environment variables that are safe to commit locally without secrets[9].

## 1) Create .env template (edit and commit only non-secret defaults)
Add this to a project-level .env file and fill values per environment; do not place secrets here if organization policy forbids it[10].

# Workspace & profile  
DATABRICKS_CONFIG_PROFILE=${ARGUMENTS:-DEV}     # CLI profile name
DATABRICKS_HOST=                                # e.g., https://<workspace-host>

# Databricks Serverless Compute is enabled by default
DATABRICKS_SERVERLESS_COMPUTE_ID=auto

# Unity Catalog defaults for development
UC_DEFAULT_CATALOG=users
UC_DEFAULT_SCHEMA=alex_miller

# MLflow: track to workspace and register in Unity Catalog
MLFLOW_TRACKING_URI=databricks
MLFLOW_REGISTRY_URI=databricks-uc
MLFLOW_EXPERIMENT_NAME=/Users/alex.miller@databricks.com/mfg-mcp-agent

## 2) Authenticate with Databricks CLI (OAuth profile)
- Run: databricks auth login --host "$DATABRICKS_HOST" and name the profile (e.g., DEV)[10].
- Verify: databricks auth profiles and databricks current-user me to confirm the profile is valid and points to the intended workspace[7].
- Keep credentials in the CLI profile (in ~/.databrickscfg) and reference it via DATABRICKS_CONFIG_PROFILE to avoid storing tokens in the repo[10].

## 3) Install local packages (pin to your runtime compatibility)
- Python: pip install -U "databricks-connect" "mlflow[databricks]" "databricks-sdk"  # add project libs as needed[2].
- Match Databricks Connect package to the Databricks Runtime running on your target compute; upgrade or align versions using the support matrix guidance[9].

## 4) Configure Databricks Connect (profile-based)
- Preferred: use profile + cluster_id so credentials flow from the CLI profile without hardcoding tokens[9].
- Example code (see smoke test command for a full snippet): from databricks.connect import DatabricksSession; spark = DatabricksSession.builder.remote(profile=os.getenv("DATABRICKS_CONFIG_PROFILE"), cluster_id=os.getenv("DATABRICKS_CLUSTER_ID")).getOrCreate()[9].
- Ensure the cluster runtime version is equal to or above the Databricks Connect package version per compatibility rules[9].

## 5) Configure MLflow for local runs
- In local code, set mlflow.set_tracking_uri("databricks") to log to the workspace and mlflow.set_registry_uri("databricks-uc") to use Unity Catalog Models where supported[8][11].
- Provide UC catalog/schema defaults from .env to route artifacts and tables consistently during development[11].

## 6) IDE integration notes
- Point the IDE’s interpreter to the environment where databricks-connect and mlflow are installed and ensure the .env file is loaded (e.g., IDE env file support or direnv)[2].
- For team consistency, document the expected profile name, default catalog/schema, and the target cluster class in docs/runbooks/deploy-runbook.md[2].

## Output
- A filled .env with safe, non-secret defaults and a validated CLI profile ready for Databricks Connect and MLflow local development[10].
