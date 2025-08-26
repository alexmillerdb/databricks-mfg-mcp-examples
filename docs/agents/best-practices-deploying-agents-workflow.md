# Systematic Process Flow & Best Practices for Developing, Evaluating, Registering, and Deploying Databricks Agents with MLflow 3

This guide distills actionable best practices and a clear step-by-step process for building robust, production-ready GenAI and assistant agents using the Databricks Agents ecosystem, MLflow 3, the ResponsesAgent interface, and Model Serving. Use these stages to drive reliable development, tight evaluation, secure deployment, and traceable observability.

---

## 1\. Author Your Agent – Structure and Patterns

**Use the recommended interface:**

- **ResponsesAgent:** The modern, recommended interface for all new agent projects. Supports multi-turn conversations, multi-agent systems, streaming, tool-calling, and attachment-capable apps.
- **Legacy ChatAgent:** Still supported but not recommended for new projects. Use ResponsesAgent for all new development.
- Agents must process message history and return output(s) in the correct schema (see schema docs).

**Best Practices:**

- Maintain statelessness (no class-level or global state).  
- Use unique message IDs (`uuid.uuid4()`), especially for streaming.  
- Place any user-specific resource initialization (e.g., OBO-aware clients) **inside** the `predict` method for serving isolation.  
- Support streaming (`predict_stream`) for better UX if possible.

**Example (ResponsesAgent):**

```python
from uuid import uuid4
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)

class EchoAgent(ResponsesAgent):
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # Get the last user message
        last_message = None
        for msg in request.input:
            if msg.role == 'user':
                last_message = msg
        
        # Create response
        output_item = self.create_text_output_item(
            text=f"Echo: {last_message.content}",
            id=str(uuid4())
        )
        
        return ResponsesAgentResponse(output=[output_item])
```

---

## 2\. Instrument for Tracing, Feedback, and Observability

**Tracing:**  
Instrument your agent (and any supporting frameworks e.g. LangChain, OpenAI, DSPy) for full observability.

```python
import mlflow

# Enable automatic logging for frameworks
mlflow.openai.autolog()         # or mlflow.langchain.autolog()

@mlflow.trace
def predict_fn(...):
    # For custom flows
    ...
```

**Collect feedback:**

- End users: Use `mlflow.log_feedback(...)` to programmatically log satisfaction, ratings, or comments to traces.  
- Experts: Launch the MLflow Review App for domain review and curation.

---

## 3\. Develop and Curate Evaluation Datasets

**Golden datasets:**  
Construct datasets from real user interactions, bug cases, and key scenarios.

```python
eval_dataset = [
    {
        "inputs": {
            "input": [{"role": "user", "content": "What's the weather in Paris?"}]
        }, 
        "expectations": {"expected_response": "sunny"}
    }
    # Extend with more samples
]
```

- Use MLflow UI or API to curate and manage these datasets for regression checks.

---

## 4\. Define and Register Scorers (LLM Judges or Custom)

- Use built-in LLM-based scorers: `Correctness`, `Safety`, `Groundedness`, etc.  
- For business/domain checks, write your own via `@mlflow.genai.scorers.scorer`.

```python
from mlflow.genai.scorers import Correctness, Safety

scorers = [Correctness(), Safety()]
```

- For nuanced checks, use Guidelines-based scorers (pass/fail rules in plain language).

---

## 5\. Systematically Run Evaluation

Evaluate your agent with scorers and golden datasets, both offline and on live traffic traces:

```python
results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=your_predict_fn,       # or deployed endpoint
    scorers=scorers
)
```

- Inspect trace-level rationales, aggregate metrics, and flag/gather regressions.  
- Tune prompts, code, or retrieval logic based on results.  
- Recycle user and reviewer feedback into your evaluation set.

---

## 6\. Register and Version Agent & Prompts

**App versioning:**

- Use MLflow’s `LoggedModel` to tie code, config, and prompt versions to every evaluation and deployment.  
- Capture Git SHA or manual tags for reproducibility.

**Prompt registry:**

- Register each prompt version via `mlflow.genai.register_prompt`.

---

## 7\. Log, Register, and Deploy Agent to Model Serving

**MLflow Logging:**

- Log with `mlflow.pyfunc.log_model`, specifying the interface, code, requirements, and resource/auth dependencies.

```python
import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.models.auth_policy import SystemAuthPolicy, UserAuthPolicy, AuthPolicy

resources = [DatabricksServingEndpoint(endpoint_name="<llm-endpoint>")]
system_auth_policy = SystemAuthPolicy(resources=resources)
user_auth_policy = UserAuthPolicy(
    api_scopes=["dashboards.genie"]  # Example for Genie OBO
)

with mlflow.start_run():
    logged_info = mlflow.pyfunc.log_model(
        python_model="agent.py",
        auth_policy=AuthPolicy(
            system_auth_policy=system_auth_policy,
            user_auth_policy=user_auth_policy
        ),
        pip_requirements=["mlflow", "databricks-agents"]
    )
```

**Databricks Deployment:**

- Register the agent to Unity Catalog.  
- Deploy via `databricks.agents.deploy()`.

```python
from databricks import agents

deployment = agents.deploy("<uc_model_name>", <uc_model_version>)
print(deployment.query_endpoint)
```

- Your endpoint now supports REST, UI playground, and full production monitoring.

---

## 8\. Production Monitoring & Continuous Improvement

- Enable automatic scoring of live traces with all development-time scorers.  
- Use the MLflow UI for dashboards, trend analysis, and anomaly detection.  
- Routinely update curation datasets and re-run regressive tests on new app versions.

---

## 9\. Access, Security, and Auth Policy Best Practices

- **OBO (On-behalf-of-user):** Always move user-authenticated client/resource setup into the `predict` method; specify minimum necessary API scopes.  
- **SystemAuthPolicy:** List resources for system-level access only.  
- **UserAuthPolicy:** Use for all user-governed, fine-grained access (e.g., Genie, Vector Search).  
- **Separate eval from prod:** Never store user data or transient state in production agent class/module-level state.

---

## Example End-to-End Process Flow

1. **Author agent** with correct interface/schema.  
2. **Instrument** with MLflow tracing/autologging.  
3. **Develop/capture golden data** for eval.  
4. **Register scorers** (LLM-based or code-based).  
5. **Evaluate** agent's quality (on code & live traffic).  
6. **Log/track** app+prompt version.  
7. **Register & deploy** agent with explicit resource/auth policies.  
8. **Monitor production** with the same scorers.  
9. **Collect & use feedback** for iterative improvement.  
10. **Repeat**—integrate improvements, retest, redeploy, monitor.

---

**Following this process ensures your GenAI agents are not only quickly shipped, but reproducible, observable, rigorously evaluated, governance-ready, and continuously improvable—hallmarks of enterprise-grade AI on Databricks.**  
