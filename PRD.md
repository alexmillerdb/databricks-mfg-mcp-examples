Here’s a detailed **Product Requirements Document (PRD)** in Markdown format. I’ve included the full problem framing, architecture (with the diagram we built), use cases, and references to the Databricks documentation you asked to be included.

---

# PRD: Multi-Tenant Supply Chain & Sales Agentic System on Databricks

## Overview

This PRD defines the requirements for building an **end-to-end agentic system on Databricks** leveraging **Model Context Protocol (MCP)**, **Databricks Agent Framework**, and **Unity Catalog (UC)** governance.

The system is a **multi-tenant Supply Chain & Logistics platform** that serves both supply chain managers and sales representatives across many companies (tenants). It demonstrates how **MCP-based agent orchestration** provides secure, governed access to tools and data while ensuring scalability and personalization.

---

## Problem Statement

* Customers want to standardize on **MCP** (both Databricks-managed and custom servers) to connect AI agents with enterprise tools and data.
* **Unity Catalog Functions (UC Functions)** are already used for tool-calling agents, but MCP provides additional **standardization, interoperability, and governance**.
* It is critical to understand **when to use MCP vs UC Functions** and to showcase real-world **multi-tenant enterprise use cases**.

---

## Goals

* Build a **demo platform** for **Supply Chain Optimization** and **Sales Enablement** using Databricks MCP and Agents.
* Show **multi-tenant isolation** of customer/department data via **Unity Catalog** and **on-behalf-of-user (OBO) authentication**.
* Demonstrate **Databricks Managed MCP** servers (Vector Search, Genie, UC Functions) and a **Custom MCP server** hosted on Databricks Apps.
* Integrate **MLflow 3.0** for monitoring, trace logging, and evaluation.
* Provide a **user-facing Databricks App UI** where business personas interact with the system.

---

## Use Cases

### 1. Supply Chain Optimization

**Persona:** Supply Chain Manager

* Query inventory and shipments (via Genie Spaces MCP).
* Run predictive models for part shortages (via UC Functions MCP).
* Retrieve historical incident reports (via Vector Search MCP).
* Receive actionable recommendations (e.g., expedite orders, reallocate stock).

### 2. Predictive Maintenance (Optional Extension)

**Persona:** Maintenance Engineer

* Diagnose anomalies in IoT telemetry via a **Custom MCP Server**.
* Run diagnostic models (via UC Functions MCP).
* Retrieve manuals and past failure reports (via Vector Search).
* Log maintenance tickets automatically (via Custom MCP).

### 3. Sales/Customer Insights

**Persona:** Sales Representative

* Query customer usage and sales KPIs (via Genie MCP).
* Run churn or upsell risk models (UC Functions).
* Retrieve relevant past proposals or support tickets (Vector Search).
* Summarize and generate next-best-action recommendations.

---

## Architecture

### High-Level Flow

1. **User → Databricks App (UI)**

   * Multi-tenant login (OAuth/SSO).
   * OBO token passed downstream.

2. **App → Agent Orchestrator (Databricks Model Serving)**

   * LLM agent orchestrates queries.
   * Uses Databricks Agent Framework with MCP tool-calling.
   * MLflow 3 logs traces, latencies, and evaluations.

3. **Agent → MCP Layer**

   * **Managed MCP Servers**:

     * **Vector Search** – RAG over SOPs, tickets, manuals.
     * **Genie Spaces** – NL→SQL analytics on inventory, sales.
     * **Unity Catalog Functions** – custom models/actions (predict shortage, churn risk, create order).
   * **Custom MCP Servers** (on Databricks Apps):

     * IoT/sensor APIs, ticketing systems, external SaaS connectors.

4. **MCP Servers → Unity Catalog & Lakehouse Data**

   * Delta tables: inventory, shipments, telemetry, CRM, tickets.
   * Lakebase (Postgres): session state, personalization.
   * UC enforces multi-tenant governance with catalogs, schemas, and row-level security.

5. **Response → User**

   * Agent synthesizes structured tool outputs into a natural language business recommendation.

---

## Architecture Diagram

![Architecture Diagram](sandbox:/mnt/data/mcp_supplychain_sales_arch.png)

*(SVG available for slides: [Download](sandbox:/mnt/data/mcp_supplychain_sales_arch.svg))*

---

## Key Requirements

### Functional

* Multi-tenant support via Unity Catalog and OBO.
* Support for multiple personas (Supply Chain Manager, Sales Rep).
* Integration of managed MCP (Genie, Vector Search, UC Functions).
* Custom MCP server for external domain tools.
* User interaction via Databricks Apps UI.
* MLflow 3.0 integration for monitoring and evaluation.

### Non-Functional

* Scalability: All components on Databricks Serverless.
* Security: End-to-end OBO auth, UC-enforced governance.
* Observability: Traces, metrics, and evaluation with MLflow 3.
* Extensibility: Future personalization with Lakebase.

---

## When to Use MCP vs UC Functions

* **UC Functions Only**: Simple, tightly-coupled use cases inside Databricks notebooks or small agents.
* **MCP**: Standardized, reusable, interoperable tool calling across multiple agents, apps, and external systems. MCP allows decoupling of tool definitions from agent code and ensures consistent interfaces  .
* **Managed MCP Servers**: Instant access to Genie, Vector Search, UC Functions without infrastructure overhead  .
* **Custom MCP Servers**: Specialized logic, external APIs, IoT, or ticketing workflows .

---

## References

* **Databricks MCP Overview**
  [Databricks Docs – MCP](https://docs.databricks.com/aws/en/generative-ai/mcp/)

* **Databricks Managed MCP Servers**
  [Managed MCP](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp)
  [Local IDE: Build an agent with managed MCP servers](https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp#local-ide-build-an-agent-with-managed-mcp-servers)

* **Custom MCP Server on Databricks**
  [Custom MCP](https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp)

* **On-Behalf-Of (OBO) User Authentication**
  [Agents OBO Example](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/agents-obo-example.html)
  [Documentation Page](https://docs.databricks.com/aws/en/generative-ai/agent-framework/authenticate-on-behalf-of-user)

* **Mosaic AI Agent Framework: Deploy MCP tool-calling LangGraph Agent**
  [Author and Deploy MCP tool-calling LangGraph Agent](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-mcp-tool-calling-agent.html)


* **MLflow 3 Code and API Docs**
  [MLflow 3 demo GitHub Repo](https://github.com/databricks-solutions/mlflow-demo)
  [MLflow GenAI API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html)
