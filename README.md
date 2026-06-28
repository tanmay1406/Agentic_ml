# SynapseAI

### Autonomous Multi-Agent AutoML Framework

> **An LLM-powered multi-agent AutoML system that autonomously performs Exploratory Data Analysis, Data Preprocessing, Feature Engineering, Model Selection, Evaluation, and Report Generation using specialized AI agents.**

---

## Overview

SynapseAI is an experimental **Agentic AI** framework that automates the end-to-end machine learning workflow through collaboration between multiple specialized AI agents.

Instead of relying on a monolithic AutoML pipeline, SynapseAI adopts a **Blackboard Architecture**, where every agent communicates through a shared `AgentState`. Each agent focuses on a single responsibility while the Supervisor orchestrates the complete workflow.

The framework uses an LLM (currently **Ollama**) to dynamically generate Python code, executes the generated code inside a secure sandbox, evaluates the results, and iteratively improves the pipeline until a satisfactory model is obtained.

---

## Architecture

```
                        User
                          │
                          ▼
                 ┌────────────────┐
                 │   Supervisor   │
                 └───────┬────────┘
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
      ▼                  ▼                  ▼
  EDA Agent        Prep Agent       Feature Agent
      │                  │                  │
      └──────────────────┼──────────────────┘
                         ▼
                  Model Training Agent
                         │
                         ▼
                    Critic Agent
                         │
                         ▼
                    Report Agent
                         │
                         ▼
                    Final Report
```

---

## Features

* Multi-Agent AutoML workflow
* Blackboard-based shared memory (`AgentState`)
* LLM-generated Python code
* Secure sandboxed code execution
* Automatic EDA
* Intelligent preprocessing
* Automatic feature engineering
* Multiple model training
* Automatic model selection
* Leaderboard generation
* Overfitting detection
* Automatic retry mechanism
* Markdown and JSON report generation
* Execution history and decision logging
* Modular agent architecture
* Easy to extend with new agents

---

## Project Structure

```
agentic-automl/
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── supervisor.py
│   ├── eda_agent.py
│   ├── prep_agent.py
│   ├── feature_agent.py
│   ├── model_agent.py
│   ├── critic_agent.py
│   └── report_agent.py
│
├── core/
│   ├── state.py
│   ├── sandbox.py
│   ├── llm_client.py
│   ├── guardrails.py
│   └── prompts.py
│
├── workspace/
│   ├── data/
│   ├── models/
│   └── reports/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Workflow

### 1. Supervisor

* Orchestrates the complete workflow
* Decides which agent runs next
* Handles retries
* Prevents infinite loops

---

### 2. EDA Agent

* Dataset inspection
* Missing value detection
* Duplicate detection
* Feature type analysis
* Problem type detection
* Class distribution analysis
* Statistical summaries

---

### 3. Data Preparation Agent

* Missing value handling
* Encoding categorical variables
* Scaling numerical features
* Outlier treatment
* Dataset validation
* Clean dataset generation

---

### 4. Feature Engineering Agent

* Feature selection
* Constant feature removal
* Correlation analysis
* Datetime feature extraction
* Interaction feature generation
* Polynomial features
* Dataset optimization

---

### 5. Model Agent

Automatically trains multiple machine learning models including:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost
* Support Vector Machine
* K-Nearest Neighbors
* Neural Networks (MLP)

The agent automatically:

* Detects classification/regression
* Chooses evaluation metrics
* Performs train/test split
* Evaluates candidate models
* Saves trained models
* Generates a leaderboard

---

### 6. Critic Agent

Reviews generated models by checking:

* Overfitting
* Underfitting
* Leaderboard consistency
* Missing artifacts
* Invalid metrics
* Model quality

If required, the Critic requests another training iteration.

---

### 7. Report Agent

Generates:

* Markdown report
* JSON summary
* Executive summary
* Training history
* Leaderboard
* Final best model summary

---

## Agent Communication

Every agent communicates through a centralized **AgentState**.

```
Agent
   │
Read State
   │
Generate Code
   │
Execute Code
   │
Update State
   │
Return to Supervisor
```

This architecture allows agents to remain completely independent while sharing a single source of truth.

---

## Sandbox

All LLM-generated code executes inside a secure sandbox.

Features include:

* Syntax validation
* Import whitelist
* Forbidden import detection
* Runtime isolation
* Timeout protection
* Temporary script generation
* Execution logging
* Automatic cleanup

---

## Security

Generated code is validated before execution.

Current protection includes:

* Forbidden imports
* Dangerous function detection
* Import whitelist
* AST validation
* Runtime timeout
* Workspace isolation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/synapseai.git

cd synapseai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install and start Ollama:

```bash
ollama pull qwen3:8b
ollama serve
```

---

## Running

```bash
python main.py \
    --dataset workspace/data/raw/iris.csv \
    --target species
```

Or specify another model:

```bash
python main.py \
    --dataset workspace/data/raw/iris.csv \
    --target species \
    --model qwen3:8b
```

---

## Outputs

After execution the framework generates:

```
workspace/

├── data/
│   └── processed/
│
├── models/
│   ├── best_model.pkl
│   ├── leaderboard.csv
│   └── metrics.json
│
├── reports/
│   ├── report.md
│   └── summary.json
│
└── state.json
```

---

## Current Status

### Implemented

* Multi-agent orchestration
* Shared AgentState
* Supervisor
* EDA Agent
* Preprocessing Agent
* Feature Engineering Agent
* Model Training Agent
* Critic Agent
* Report Agent
* Ollama integration
* Sandbox execution
* Execution logging
* Retry mechanism

### Planned

* Hyperparameter Optimization
* RAG-powered dataset understanding
* Multi-modal datasets
* Distributed execution
* Human-in-the-loop feedback
* Multi-LLM support (OpenAI, Gemini, Claude)
* Docker deployment
* Web dashboard
* API server

---

## Tech Stack

* Python
* Ollama
* Pydantic
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* LightGBM
* CatBoost
* Matplotlib
* Plotly

---

## Design Principles

* Modular
* Extensible
* Agent-based
* Secure
* Reproducible
* Explainable
* Production-oriented

---

## Disclaimer

SynapseAI is an experimental research project exploring the use of Large Language Models as autonomous machine learning engineers. Generated code should be reviewed before deployment in production environments.

---

## License

MIT License

---

## Author

**Tanmay Gupta**

Built as a research project exploring autonomous multi-agent systems for machine learning.
