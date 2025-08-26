# Deploying On-Behalf-of-User (OBO) Agents with Databricks MCP and Genie

This guide documents how to deploy Databricks agents using on-behalf-of-user (OBO) authentication, specifically with Genie Spaces as Managed Context Protocol (MCP) servers. It covers foundational concepts, recommended patterns, code examples, and best practices for OBO with Genie and MCP, ensuring secure and fine-grained access control aligned with Unity Catalog governance.

---

## Table of Contents

- [Overview](#overview)  
- [OBO Authentication: Concepts & Flow](#obo-authentication-concepts--flow)  
- [How to Set Up Genie Space as an MCP Server](#how-to-set-up-genie-space-as-an-mcp-server)  
- [UserAuthPolicy vs SystemAuthPolicy: How and When to Use](#userauthpolicy-vs-systemauthpolicy-how-and-when-to-use)  
- [Code Example: Deploying an OBO Agent with Genie/MCP](#code-example-deploying-an-obo-agent-with-geniemcp)  
- [Pattern for OBO Client Initialization](#pattern-for-obo-client-initialization)  
- [Best Practices & Pitfalls](#best-practices--pitfalls)  
- [Summary Table](#summary-table)  
- [References & Further Reading](#references--further-reading)

---

## Overview

Databricks supports deploying AI agents that interact with Databricks resources on behalf of the invoking user (OBO). This is crucial for:

- Applying the end user's Unity Catalog permissions and data governance policies automatically  
- Keeping audit and compliance controls tight  
- Supporting collaborative use cases without granting system-level or broadened access

When using Genie Spaces as MCP servers, OBO lets each agent invocation see only what the invoking user can, including tables, dashboards, and data masked by Unity Catalog.

---

## OBO Authentication: Concepts & Flow

- **OBO (On-Behalf-of-User) Authentication**: Enables your agent to make API/resource calls using the permissions and identity of the currently invoking Databricks user.  
- **System Authentication**: The agent acts as itself, using its own service principal or system identity.  
- Scopes control which APIs the agent can call as OBO.  
- You must opt in to OBO at the workspace level (admin action; currently in Beta).

---

## How to Set Up Genie Space as an MCP Server

1. **Authenticate to Databricks Workspace**

```shell
databricks auth login --host https://<your-workspace-hostname>
```

2. **Install Required Libraries**

```shell
pip install -U "mcp>=1.9" "databricks-sdk[openai]" "mlflow>=3.1.0" "databricks-agents>=1.0.0" "databricks-mcp"
```

3. **Prepare Genie MCP Server Endpoint**

Ensure you have your Genie Space ID and Databricks host.

```python
from databricks.sdk import WorkspaceClient

workspace_client = WorkspaceClient(profile="<your-databricks-cli-profile>")
host = workspace_client.config.host
genie_space_id = "<your-genie-space-id>"  # Replace this

MANAGED_MCP_SERVER_URLS = [
    f"{host}/api/2.0/mcp/genie/{genie_space_id}"
]
```

4. **Connect and Test Genie MCP Server**

```python
from databricks_mcp import DatabricksMCPClient

mcp_client = DatabricksMCPClient(
    server_url=MANAGED_MCP_SERVER_URLS[0], 
    workspace_client=workspace_client
)
tools = mcp_client.list_tools()
print(f"Discovered tools {tools} from MCP server {MANAGED_MCP_SERVER_URLS[0]}")
```

---

## UserAuthPolicy vs SystemAuthPolicy: How and When to Use

| Policy Name | What to declare? | Access Identity | Typical for... |
| :---- | :---- | :---- | :---- |
| SystemAuthPolicy | Resource objects | Agent/system principal | LLM endpoints, general agent use |
| UserAuthPolicy | List of API scopes (strings) | End user | Genie, dashboards, fine-grained OBO |

**Key rule:**

- *SystemAuthPolicy*: Only include resources you intend the agent to access as its *own* identity.  
- *UserAuthPolicy*: Specify ONLY the API scopes required for resources your agent will call as the *user* (e.g., `"dashboards.genie"` for Genie Spaces).

### Common Pitfalls

- Do **not** double-nest lists in `SystemAuthPolicy`: `SystemAuthPolicy(resources=[resources])` is incorrect; use `SystemAuthPolicy(resources=resources)`  
- Do **not** include resource objects like `DatabricksGenieSpace` in `UserAuthPolicy`; use API scopes instead.  
- Only initialize user-authenticated clients inside the `predict` (or serving) method, never in agent `__init__`.

---

## Code Example: Deploying an OBO Agent with Genie/MCP

```python
import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.models.auth_policy import UserAuthPolicy, SystemAuthPolicy, AuthPolicy

LLM_ENDPOINT_NAME = "<llm-endpoint-name>"

# System resources - only LLM Endpoint in this example
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)
    # Add more if any system-only resources are needed
]

system_auth_policy = SystemAuthPolicy(resources=resources)

user_auth_policy = UserAuthPolicy(
    api_scopes=[
        "dashboards.genie",  # Genie OBO
        # Add more if needed for other OBO resources
    ]
)

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        python_model="agent.py",
        auth_policy=AuthPolicy(
            system_auth_policy=system_auth_policy,
            user_auth_policy=user_auth_policy
        ),
        pip_requirements=[
            "databricks-mcp",
            "mlflow>=3.1.0"
            # etc...
        ]
    )
```

**Explanation:**

- Only the LLM endpoint is listed as a system resource.  
- `dashboards.genie` (for Genie Space OBO) is specified as a scope in `UserAuthPolicy`.  
- Resources for Genie/MCP are *not* included in `SystemAuthPolicy` if you want everything OBO.

---

## Pattern for OBO Client Initialization

Always create the OBO-aware (user-credentialed) client **inside** the serving method—**not** at agent/module initialization. Example for a Databricks ChatAgent:

```python
from mlflow.pyfunc import ChatAgent
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials
from databricks_langchain.genie import GenieAgent

class MyGenieChatAgent(ChatAgent):
    def predict(self, messages, context=None, custom_inputs=None):
        user_client = WorkspaceClient(
            credentials_strategy=ModelServingUserCredentials()
        )
        genie_agent = GenieAgent(
            genie_space_id="<genie_space_id>",
            genie_agent_name="Genie",
            description="Genie Agent",
            client=user_client  # This enforces OBO!
        )
        # Now make Genie/MCP queries as the user
        # ...process response and return
```

---

## Best Practices & Pitfalls

- **Minimize Scopes**: Only request the exact scopes you need for user OBO.  
- **Separation of Context**: All OBO-aware resource handling must happen inside request/serving context (e.g., `predict`), not in persistent state.  
- **Avoid Caching User Resources**: Do not hold references to user-auth clients or tokens between requests.  
- **Testing**: Validate that your agent can only access resources permitted for the invoking Databricks user.  
- **Consent/Opt-in**: OBO features may require workspace admin opt-in; always confirm it’s enabled.  
- **Documentation**: Document in your agent registry/docs which scopes and resources are accessed OBO vs. system, for compliance and auditing.

---

## Summary Table

| Scenario | Where to Declare | How to Use |
| :---- | :---- | :---- |
| System-level (agent identity) resource access | SystemAuthPolicy | List resources (e.g., LLM endpoints) |
| OBO resource (user identity) access | UserAuthPolicy | List required API scopes (e.g., "dashboards.genie") |
| Resource object references for OBO | *Not required* | Use scopes instead |
| Instantiate user-auth client for OBO | Runtime (in `predict`) | Use `WorkspaceClient(credentials_strategy=ModelServingUserCredentials())` |

---

## References & Further Reading

- Log and register AI agents | Databricks Documentation  
- Agent On Behalf of Users \- User Guide  
- Use Databricks managed MCP servers  
- Unity Catalog docs for governance details  
- Best practices in Databricks agent and OBO security configuration

---

This guide should help you securely and efficiently deploy OBO-capable agents with Genie and MCP on Databricks. Adapt and expand code snippets to your organization's deployment workflow as needed.  
