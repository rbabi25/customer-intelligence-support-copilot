# Customer Intelligence & Support Copilot

## Project Overview

The **Customer Intelligence & Support Copilot** is an end-to-end data analytics and AI project built around a synthetic grocery-delivery business inspired by Swiggy Instamart.

The project demonstrates how customer, transaction, and support-ticket data can be transformed into actionable insights using data engineering, machine learning, privacy protection, retrieval-augmented generation (RAG), and tool-based automation.

The objective is to build a reproducible, locally executable solution without paid APIs or external LLM calls.

## Business Problem

A grocery-delivery business needs to:

- Organize and validate customer and transaction data.
- Identify meaningful customer segments.
- Predict customers who may become inactive.
- Classify and route customer support tickets.
- Protect personally identifiable information (PII).
- Answer policy questions using relevant policy documents.
- Support refund-related decisions through controlled tools and business rules.

## Project Architecture

The project is divided into four parts.

### Part A — Data Foundation

**Objective:** Build a reliable foundation for analytics.

Key activities:
- Generate and inspect synthetic data.
- Clean and standardize datasets.
- Handle missing values, duplicates, and inconsistent data.
- Validate data quality and business rules.
- Store structured data in SQLite.
- Query the database using SQL.

Expected outcome: Clean, validated, queryable data for downstream analysis.

### Part B — Customer Intelligence

**Objective:** Understand customer behavior and identify potential inactivity.

Key activities:
- Create customer-level analytical features.
- Segment customers using suitable clustering or segmentation techniques.
- Build a customer lapse prediction model.
- Evaluate model performance using appropriate metrics.
- Save reusable model artifacts where required.

Expected outcome: Customer segments and data-driven lapse-risk predictions.

### Part C — Support Ticket Triage

**Objective:** Improve support-ticket handling while protecting customer privacy.

Key activities:
- Detect and mask PII before processing ticket text.
- Clean and standardize support tickets.
- Classify tickets into relevant categories.
- Route tickets based on category, priority, or business rules.
- Validate predictions and handle uncertain classifications safely.

Expected outcome: A privacy-aware ticket triage workflow.

### Part D — GenAI Support Copilot

**Objective:** Build a policy-grounded assistant with controlled tool usage.

Key activities:
- Prepare policy documents for retrieval.
- Retrieve relevant policy passages for customer questions.
- Generate policy-grounded answers using the supplied offline `mock_llm.py`.
- Implement controlled SQL tools for data questions.
- Build a refund agent that follows defined eligibility rules.
- Validate tool inputs and outputs and handle errors safely.

Expected outcome: A reproducible support copilot that uses policy evidence and controlled business logic.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python 3.11+ |
| Data manipulation | pandas, NumPy |
| Machine learning | scikit-learn |
| Model persistence | joblib |
| Database | SQLite 3.25+ |
| Querying | SQL |
| GenAI workflow | Supplied offline `mock_llm.py` |
| Version control | Git and GitHub |

## Repository Structure

```text
customer-intelligence-support-copilot/
├── data/
│   ├── raw/
│   ├── processed/
│   └── database/
├── part_a_data_foundation/
│   ├── data_cleaning.py
│   ├── data_validation.py
│   ├── database.py
│   └── analysis.sql
├── part_b_customer_intelligence/
│   ├── customer_features.py
│   ├── segmentation.py
│   ├── lapse_prediction.py
│   └── evaluation.py
├── part_c_ticket_triage/
│   ├── pii_masking.py
│   ├── ticket_cleaning.py
│   ├── classification.py
│   └── routing.py
├── part_d_genai_copilot/
│   ├── policy_documents/
│   ├── retrieval.py
│   ├── policy_assistant.py
│   ├── sql_tools.py
│   └── refund_agent.py
├── models/
├── outputs/
├── generate_data.py
├── mock_llm.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: This is the planned structure. Update filenames and outputs to match the final implementation and the supplied project brief.*

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/rbabi25/customer-intelligence-support-copilot.git
cd customer-intelligence-support-copilot
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify the environment

```bash
python --version
python -c "import sqlite3; print(sqlite3.sqlite_version)"
python -c "import pandas, numpy, sklearn, joblib; print('Dependencies installed successfully')"
```

Use Python 3.11 or newer and SQLite 3.25 or newer.

## Execution Workflow

1. Run the supplied data generator.
2. Inspect the generated datasets and their schemas.
3. Implement Part A cleaning, validation, SQLite storage, and SQL analysis.
4. Build customer features, segmentation, and lapse prediction in Part B.
5. Implement PII masking, ticket classification, and routing in Part C.
6. Build policy retrieval, the policy assistant, SQL tools, and refund agent in Part D.
7. Run the required validations and record the results.
8. Document the final workflow, outputs, and instructions for reproduction.

The exact commands for each part will be documented once the scripts are implemented.

## Reproducibility and Safety

- Use synthetic data for development and demonstration.
- Keep the supplied `generate_data.py` and `mock_llm.py` files unmodified.
- Use the offline mock LLM rather than paid APIs or external LLM calls.
- Keep personally identifiable information protected during ticket processing.
- Validate inputs and outputs before executing business actions.
- Use explicit refund eligibility rules rather than allowing an LLM to make unsupported financial decisions.
- Document dependencies and execution steps.
- Do not commit credentials, API keys, virtual environments, or unintended sensitive data.

## Evaluation

The project will be validated against the supplied requirements for all four parts:

- **Part A:** Data quality, validation, database storage, and SQL queries.
- **Part B:** Customer features, segmentation, lapse prediction, and model evaluation.
- **Part C:** PII masking, ticket classification, and routing.
- **Part D:** Policy retrieval, grounded answers, SQL tool usage, and controlled refund-agent behavior.

Final metrics, sample outputs, and screenshots should be added after the implementation has been executed and verified.

## Project Status

**In progress.** Implementation and validation will be completed incrementally across Parts A–D.

## License

Add a license if you intend to distribute or reuse the project publicly.
