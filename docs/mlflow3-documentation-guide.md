# MLflow 3 for GenAI Apps — Comprehensive Documentation Guide

This resource provides a single, end-to-end reference for building, observing, evaluating, improving, and operationalizing GenAI applications using MLflow 3\. It covers setup, tracing, evaluation and scorers, user and expert feedback, app and prompt versioning, and deployment best practices—from local development to production monitoring.

---

## Table of Contents

- [1\. Introduction to MLflow 3 for GenAI](#1-introduction-to-mlflow-3-for-genai)  
- [2\. Environment Setup](#2-environment-setup)  
- [3\. Tracing and Observability](#3-tracing-and-observability)  
- [4\. GenAI Evaluation with Scorers & Judges](#4-genai-evaluation-with-scorers--judges)  
- [5\. Collecting Feedback: End User and Expert Labels](#5-collecting-feedback-end-user-and-expert-labels)  
- [6\. Versioning & Tracking: Applications and Prompts](#6-versioning--tracking-applications-and-prompts)  
- [7\. App Development Pattern with MLflow 3](#7-app-development-pattern-with-mlflow-3)  
- [8\. Production Monitoring & Continuous Improvement](#8-production-monitoring--continuous-improvement)  
- [9\. Best Practices, Limitations, Migration](#9-best-practices-limitations-migration)  
- [10\. References & Further Reading](#10-references--further-reading)

---

## 1\. Introduction to MLflow 3 for GenAI

MLflow 3 is a unified, production-grade platform for managing AI lifecycle—now redesigned to natively support GenAI and agent workloads:

- **Comprehensive observability:** Production-grade tracing integrated with 20+ GenAI frameworks, exposing every prompt, retrieval, tool call, and more.  
- **Automated quality evaluation:** Built-in and custom LLM-based scorers (judges) for correctness, groundedness, safety, relevance, etc., applicable online/offline.  
- **Integrated feedback & review:** APIs and UI for end-user and domain expert feedback. Feedback is attached to traces for continuous improvement.  
- **Prompt registry & app versioning:** First-class lifecycle management and lineage for code, prompts, configs, and data. All assets are tracked and governed.  
- **Seamless app development:** Consistency across local IDEs, Databricks notebooks, and production.  
- **Enterprise security and governance** with Unity Catalog, and open-source interoperability.

Learn more: \[MLflow 3 for GenAI | Databricks Documentation\]

---

## 2\. Environment Setup

### Prerequisites

- **Python**: 3.10 or newer recommended  
- **Databricks Workspace** (for managed experience), or use open-source MLflow 3 for local/other infra

### Installation

For Databricks-managed or local development:

```shell
pip install --upgrade "mlflow[databricks]>=3.1"
```

For tracing popular GenAI frameworks—no extra code required for tracing (e.g., OpenAI, LangChain, DSPy, LlamaIndex, etc.)

Optional for enhanced features and integration:

```shell
pip install databricks-agents
pip install openai  # or your LLM SDK
```

### Initial Experiment Setup

```python
import mlflow

# Specify your Databricks workspace URI if running locally
mlflow.set_tracking_uri("databricks")  # or use env var MLFLOW_TRACKING_URI=databricks

# Set the target experiment (create if not exists)
mlflow.set_experiment("/Users/<your_user>/genai-app-experiment")
```

---

## 3\. Tracing and Observability

### Why Tracing?

GenAI observability is critical for debugging, understanding quality, ensuring regulatory compliance, and rapidly improving your app. MLflow Tracing captures every step, from inputs through outputs—including all intermediate operations, LLM calls, retrievals, and tool usages.

### Enabling Tracing — Automatic Instrumentation

MLflow 3 supports one-line instrumentation for 20+ frameworks. Example with OpenAI and LangChain:

```python
import mlflow

# Trace all OpenAI calls
mlflow.openai.autolog()
# Trace all LangChain operations
mlflow.langchain.autolog()

# Your existing GenAI code stays unchanged
client = openai.OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "What is MLflow?"}]
)
```

The system logs each request and its execution structure as a rich trace. No changes needed in your main app logic.

**Manual tracing with decorator/context manager:**

```python
@mlflow.trace
def custom_pipeline(input):
    # arbitrary Python logic here, for custom or advanced tracing
    ...
```

See the trace in notebooks (inline), in the MLflow Experiment UI, or via API (`mlflow.search_traces`).

### Trace Content

Each trace shows:

- All request/response metadata (inputs, outputs)  
- Execution call tree (with span durations, errors, latencies, token usage, costs)  
- All tool calls, retrieval steps, and user/system metadata  
- Linked user and LLM assessment feedback

---

## 4\. GenAI Evaluation with Scorers & Judges

### Overview

Traditional metrics aren't enough for GenAI's open language space. MLflow 3 introduces LLM-based and code-based "scorers"—standard and custom quality checks applied online or in offline regression tests.

#### Common built-in scorers (LLM judges):

- **Correctness** (matches expected answer)  
- **Groundedness** (is response based on facts/context? avoids hallucination)  
- **Safety** (no harmful/inappropriate content)  
- **RelevanceToQuery** (does the response answer the question)  
- **RetrievalRelevance / Sufficiency** (for RAG apps: were correct docs retrieved, and enough of them?)  
- **Guidelines adherence** (evaluate by flexible natural language criteria)

#### Custom code-based scorers:

Any deterministic or learned metric—e.g., domain-specific checks, quality indicators.

### How to Evaluate

#### Step 1: Define Your Predict Function

Instrument your app’s entrypoint using `@mlflow.trace`:

```python
@mlflow.trace
def my_app(input: str) -> dict:
    # app logic — returns dict, e.g., {"response": ...}
    ...
```

#### Step 2: Create an Evaluation Dataset

Can be a list of dicts, Pandas/Spark DataFrame, or saved as a managed MLflow EvaluationDataset:

```python
eval_dataset = [
    {
        "inputs": {"input": "What is the capital of France?"}, 
        "expectations": {"expected_response": "Paris"}
    }
    # ...more test examples
]
```

#### Step 3: Specify Scorers

```python
from mlflow.genai.scorers import Correctness, Safety, Guidelines

guideline = Guidelines(
    name="conciseness",
    guidelines="The response must be concise and clear."
)
scorers = [Correctness(), Safety(), guideline]
```

#### Step 4: Run Evaluation

```python
import mlflow

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=my_app,
    scorers=scorers
)
traces = mlflow.search_traces(run_id=results.run_id)
```

Outputs:

- Trace-level feedback for every scorer  
- Aggregate metrics  
- Explanations (rationales) per example/failure

All evaluation results are linked to the code version (see “versioning” below).

**Custom scorers:** Define custom business logic; wrap with `@scorer` decorator.

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer
def domain_specific(outputs, expectations):
    condition = ...  # business logic
    return Feedback(value=condition, rationale="Details on why")
```

See: \[Scorers | Databricks Documentation\]

### Guidelines-based LLM Scorers

- Write natural language pass/fail rules as guidelines.  
- Use the built-in `Guidelines` or `ExpectationGuidelines` scorers for global or per-row checks, or customize using `meets_guidelines` for arbitrary trace context.  
- \[Guidelines-based LLM scorers | Databricks Documentation\]

---

## 5\. Collecting Feedback: End User and Expert Labels

### End-User Feedback Collection

Attach feedback from actual users to traces. Built-in support for thumbs up/down, ratings, comments, etc. Programmatically:

```python
mlflow.log_feedback(
    trace_id=<trace_id>,
    name="user_satisfaction",
    value=True,  # or False/"positive"/"negative", etc.
    comment="Great answer!"
)
```

- All feedback is visible in the UI attached to traces and is queryable for debugging, regression detection, or dataset curation.

### Expert/Domain-Labeling via Review App

- Use the MLflow Review App to launch UI sessions for expert annotators.  
- Experts can annotate all, or just the problematic traces with structured schemas (ratings, corrected answers, guidelines, etc).  
- Feedback is auto-synced to traces for curation and improved scorer alignment.

---

## 6\. Versioning & Tracking: Applications and Prompts

MLflow 3 introduces “LoggedModels”, a new abstraction for tracking every version of your app—including code commit, config, and prompt dependencies.

### Track App Versions (Git-integrated or manually)

```python
import subprocess

git_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]
).decode("ascii").strip()[:8]
model_name = f"my_genai_app-{git_commit}"
active_model = mlflow.set_active_model(name=model_name)
```

All traces/evaluation results are automatically linked to this version—enabling per-version debugging & comparison.

### Prompt Versioning & Lineage

- Register and version prompts in the MLflow Prompt Registry.  
- Link prompt versions to app versions automatically.

Also supports A/B prompt experiments, rollback, discovery, and DSPy optimizers.

```python
from mlflow.genai import register_prompt

prompt = register_prompt(
    name="company_support_prompt",
    template="You are a helpful assistant. Answer: {{question}}",
    commit_message="Initial support prompt"
)
```

### Compare performance across code/prompt/version changes with full lineage:

- Analysis and evaluations can be filtered/compared by model and prompt version.

---

## 7\. App Development Pattern with MLflow 3

1. **Develop and instrument app**: Use MLflow’s tracing and versioning.  
2. **Create evaluation dataset(s)**: Using real production traces or synthetic/enumerated tests.  
3. **Define (and tune) scorers/judges**: Built-in for standard quality, custom for domain-specific needs.  
4. **Evaluate and debug**: Use the UI or API; inspect failures, rationales, feedback.  
5. **Collect feedback**: Loop in users and experts for edge-case and recurring issues.  
6. **Iterate**: Update code, prompt, or config; re-run evaluation. Confirm regressions are fixed.  
7. **Register and release**: Launch as a new app version; optionally set up deployment or CI/CD promotion checks.  
8. **Monitor in production**: Set up scheduled or continuous monitoring with the same scorers as in development.

— **Sample Dev-Production Loop Diagram**:

- Production app serves and logs traces  
- User feedback flows into traces  
- Automated monitoring runs LLM-judge and custom scorers on traces  
- Evaluate quality and gather expert feedback  
- Curate and tune evaluation datasets  
- Evaluate candidate new versions  
- Compare, deploy, and repeat

---

## 8\. Production Monitoring & Continuous Improvement

Monitoring is not just “metrics dashboarding”—with MLflow 3, you enable proactive, ongoing quality control:

- **Real-time monitoring:** All production traces can be automatically scored in near real-time using your offline-developed scorers.  
- **Continuous benchmarking:** Failures and feedback loop into datasets for future regression/prevention.  
- **Alerting and dashboards:** Dashboards built on MLflow trace/metric data surface emerging issues, usage, and model drift.

### Monitoring Setup

- **Managed Serving (Databricks Model Serving):** Monitoring is set up by default; schedule any scorers (LLM and custom) on a sample or all traffic.  
- **External/Custom Serving and Apps:** Use MLflow Tracing SDK to log to Databricks; monitoring jobs periodically replay scorers for quality metrics.

```python
from mlflow.genai.scorers import Safety, RelevanceToQuery

monitor = mlflow.genai.create_monitor(
    name="customer_support_monitor",
    endpoint="endpoints:/my-prod-endpoint",
    scorers=[Safety(), RelevanceToQuery()],
    sampling_rate=0.1  # 10% sample
)
```

All production and offline traces share the same data model—so you use the same tools and framework everywhere.

---

## 9\. Best Practices, Limitations, Migration

### Best Practices

- **Write Once, Use Everywhere:** Build scorers for dev, then reuse seamlessly in production.  
- **Minimal, Human-aligned Metrics:** Start with built-in safety, groundedness, relevance, and add only your necessary custom checks; consistently tune to match expert feedback.  
- **Use real production data for evaluation:** Curation of regression/evaluation test sets from actual user failures provides the best error-checking and prevents regressions.  
- **Track everything per version:** Use MLflow’s versioning features to always know which code/prompt/dataset generated a given result.  
- **Secure and govern your AI:** Use Unity Catalog for data, model, and prompt governance; enforce access controls and compliance.

### Limitations & Notes

- Some advanced monitoring and review app features require Databricks-managed MLflow.  
- Migration from MLflow 2/AgentEval: There are major usability improvements in MLflow 3, and the new APIs are intentionally similar but require careful mapping for metrics, scorers, and schema.  
- While core APIs are open-source, certain LLM judges and production monitoring at scale are Databricks-only features.
