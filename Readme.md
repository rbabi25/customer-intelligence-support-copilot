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

**Expected outcome:** Clean, validated, queryable data for downstream analysis.

### Part B — Customer Intelligence

**Objective:** Understand customer behavior and identify potential inactivity.

Key activities:

- Create customer-level analytical features.
- Segment customers using suitable clustering or segmentation techniques.
- Build a customer lapse prediction model.
- Evaluate model performance using appropriate metrics.
- Save reusable model artifacts where required.

**Expected outcome:** Customer segments and data-driven lapse-risk predictions.

### Part C — Support Ticket Triage

**Objective:** Improve support-ticket handling while protecting customer privacy.

Key activities:

- Detect and mask PII before processing ticket text.
- Clean and standardize support tickets.
- Classify tickets into relevant categories.
- Route tickets based on category, priority, or business rules.
- Validate predictions and handle uncertain classifications safely.

**Expected outcome:** A privacy-aware ticket triage workflow.

### Part D — GenAI Support Copilot

**Objective:** Build a policy-grounded assistant with controlled tool usage.

Key activities:

- Prepare policy documents for retrieval.
- Retrieve relevant policy passages for customer questions.
- Generate policy-grounded answers using the supplied offline `mock_llm.py`.
- Evaluate retrieval quality, answer grounding, and fallback behavior.
- Implement controlled SQL and business-logic tools.
- Build a ReAct refund agent.
- Enforce execution limits, including token budgets, timeouts, and maximum iterations.
- Implement checkpointing and resume functionality.
- Compare ReAct with a plan-and-execute design.

**Expected outcome:** A reproducible support copilot that uses policy evidence and controlled business logic to handle refund-related requests safely.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python 3.11+ |
| Data manipulation | pandas, NumPy |
| Machine learning | scikit-learn |
| Model persistence | joblib |
| Database | SQLite |
| Querying | SQL |
| GenAI workflow | Supplied offline `mock_llm.py` |
| Retrieval | TF-IDF, cosine similarity |
| Agent architecture | ReAct |
| Checkpoint persistence | JSON |
| Logging | JSONL |
| Version control | Git and GitHub |

## Repository Structure

```text
customer-intelligence-support-copilot/
├── data/
│   ├── raw/
│   ├── clean/
│   ├── processed/
│   ├── policies/
│   └── instamart.db
├── src/
│   └── refund_agent.py
├── outputs/
│   ├── policy_chunks.json
│   ├── agent_runs.jsonl
│   ├── refund_ledger.jsonl
│   └── state/
│       └── <run_id>.json
├── mock_llm.py
├── generate_data.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The structure above highlights the main project and Part D artifacts. Retain any additional Part A, B, and C scripts and folders present in the actual repository.*

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/rbabi25/customer-intelligence-support-copilot.git
cd customer-intelligence-support-copilot
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

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

Use Python 3.11 or newer.

## Execution Workflow

1. Run the supplied data generator.
2. Inspect the generated datasets and their schemas.
3. Implement Part A cleaning, validation, SQLite storage, and SQL analysis.
4. Build customer features, segmentation, and lapse prediction in Part B.
5. Implement PII masking, ticket classification, and routing in Part C.
6. Build policy retrieval, the policy assistant, SQL tools, and refund agent in Part D.
7. Run the required validations and record the results.
8. Document the final workflow, outputs, and instructions for reproduction.

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

The project is validated against the supplied requirements for all four parts:

- **Part A:** Data quality, validation, database storage, and SQL queries.
- **Part B:** Customer features, segmentation, lapse prediction, and model evaluation.
- **Part C:** PII masking, ticket classification, and routing.
- **Part D:** Policy retrieval, grounded answers, tool usage, execution limits, checkpointing, and refund-agent behavior.

# Part C — Support Ticket Triage

## Guardrails and PII Protection

Processed 164 support tickets. Four empty tickets were skipped, and four tickets exceeding the 400-token input threshold were flagged without making model calls.

Before classification, PII was redacted using regular expressions:

- 55 email addresses
- 91 phone numbers
- 132 order IDs

A second validation confirmed that no matching PII remained in the redacted ticket text. Only the redacted ticket text was passed as the user message to the LLM.

## Prompt Comparison

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 14,236 | 24,532 |
| Tokens per classified ticket | 96.84 | 166.88 |

Few-shot prompting improved classification accuracy by 14.29 percentage points. It required approximately 72.3% more tokens, demonstrating the trade-off between prompt cost and classification quality.

Zero-shot errors were concentrated in fraud and technical tickets being classified as billing. The labelled examples in the few-shot prompt helped resolve these classification errors.

## Output Validation

The pipeline supports three classification outcomes:

- `classified_json`: Valid JSON containing an allowed category.
- `classified_keyword`: Invalid JSON where exactly one allowed category keyword could be extracted.
- `flagged_invalid_output`: Output that could not be safely classified and therefore requires human review.

Across the 156 eligible tickets, 135 were accepted through JSON validation, 12 through keyword fallback, and 9 were flagged for human review.

## Retry Handling

Each model call supports up to three retries using exponential backoff with jitter. All 19 transient errors in each full run were recovered successfully. No permanent errors occurred.

## Token Budget

A zero-shot demonstration used a 6,000-token budget. Once the accumulated token usage reached the threshold, subsequent tickets were marked `budget_exhausted` and were not sent to the model.

The final reported usage was 6,022 tokens because the budget was checked before each call; the last permitted response took usage beyond the threshold.

The final few-shot run used a 100,000-token budget and successfully processed all eligible tickets.

## Final Routing

The final output contains 164 tickets:

- Fraud tickets → `security_team`, HIGH priority.
- Premium customers → `premium_desk`, HIGH priority.
- Other valid classifications → `support_queue`, NORMAL priority.
- Empty, long, invalid-output, and budget-exhausted tickets → `human_review`, NORMAL priority.

The final few-shot classification accuracy was 100% over the 147 classified tickets. The remaining tickets were deliberately excluded from automated classification and retained for human review.

# Part D — GenAI Support Copilot

## Task 1 — Policy Document Chunking

The five policy documents are loaded from `data/policies/` and divided into smaller chunks for retrieval.

The implementation uses:

- Chunk size: 40 words
- Chunk overlap: 10 words
- Output: `outputs/policy_chunks.json`

The chunks retain their source-document information so that retrieved passages can be traced back to the original policy.

## Task 2 — Policy Retrieval

A TF-IDF vectorizer converts policy chunks and user questions into numerical representations.

Cosine similarity is used to rank relevant chunks. The retriever returns the top three passages for each query.

This provides a lightweight, locally executable retrieval mechanism without requiring an embedding API or external vector database.

## Task 3 — Policy-Grounded Answering

The policy assistant retrieves relevant passages before generating an answer using the supplied offline `mock_llm.py`.

The retrieved policy context is provided to the model so that answers are grounded in the available policy documents.

If the required information is not available in the retrieved context, the assistant uses a fallback response rather than inventing policy information.

## Task 4 — Retrieval and Grounding Evaluation

The evaluation uses answerable and unanswerable policy questions.

The following aspects are checked:

- Whether the expected policy document is retrieved.
- Whether the expected fact appears in the generated answer.
- Whether unanswerable questions receive the correct fallback response.

The evaluation reported:

| Metric | Result |
|---|---:|
| Answerable questions | 12 |
| Expected document retrieved | 12/12 |
| Expected fact in answer | 12/12 |
| Unanswerable questions | 2 |
| Correct fallback answers | 2/2 |

A zero-overlap experiment is also performed to evaluate the effect of chunk overlap on retrieval quality. At least one question fails in the zero-overlap experiment, demonstrating the importance of chunk boundaries and contextual overlap.

*Add the exact zero-overlap chunk count and evaluation result from your execution output here.*

## Task 5 — Tool Implementation

The refund agent uses five controlled tools:

| Tool | Purpose |
|---|---|
| `get_order` | Retrieves order information from SQLite. |
| `search_policy` | Retrieves relevant policy passages. |
| `calculate_refund` | Calculates refund eligibility and amount using business rules. |
| `issue_refund` | Records an eligible refund in the refund ledger. |
| `escalate_to_human` | Records cases requiring human approval. |

Refund calculations are determined by explicit business rules rather than unrestricted LLM decisions.

For example, completed orders with a delivery time exceeding 45 minutes are eligible for a full refund under the implemented rule. Orders that do not meet the eligibility criteria are not automatically refunded.

High-value cases requiring approval are escalated to a human.

The refund ledger includes an idempotency check to prevent duplicate refund issuance for the same order.

## Task 6 — ReAct Refund Agent

The ReAct agent follows an iterative reasoning and tool-execution workflow:

1. Send the current conversation and available tool schemas to the model.
2. Interpret the model's response.
3. Execute the selected tool.
4. Append the tool observation to the message history.
5. Continue reasoning until a final answer or execution limit is reached.

The agent is evaluated using seven scenarios.

### Seven-scenario evaluation

| Scenario | Expected outcome | Expected amount | Result |
|---|---|---:|---|
| S1 | Refund issued | ₹128.25 | REFUND |
| S2 | Refund issued | ₹135.00 | REFUND |
| S3 | Escalation to human | ₹1,999.00 | ESCALATED |
| S4 | Refund denied | ₹0.00 | DENIED |
| S5 | Refund denied | ₹0.00 | DENIED |
| S6 | Order not found | ₹0.00 | NOT_FOUND |
| S7 | Maximum iteration fallback | — | FALLBACK_MAX_ITERATIONS |

The agent must produce the expected outcome for each scenario. S7 is deliberately tested with the default `token_limit=6000` and must terminate with `FALLBACK_MAX_ITERATIONS`.

*Verify the final scenario results against `outputs/agent_runs.jsonl` before submission.*

## Task 7 — Execution Limits and Failure Handling

The agent includes execution safeguards to prevent uncontrolled model and tool execution.

The following limits are implemented:

- Token budget to restrict accumulated token usage.
- Time limit to stop execution when the allowed duration is exceeded.
- Maximum iteration limit to prevent infinite agent loops.

Two separate demonstrations validate the fallback behavior.

| Demonstration | Expected final status |
|---|---|
| Token limit exceeded | `FALLBACK_TOKEN_LIMIT` |
| Timeout reached | `FALLBACK_TIMEOUT` |

The maximum iteration demonstration is also validated through S7:

| Demonstration | Expected final status |
|---|---|
| Maximum iterations reached | `FALLBACK_MAX_ITERATIONS` |

The agent returns explicit fallback statuses instead of continuing execution after a configured limit is reached.

## Task 8 — Checkpointing and Resume

The refund agent supports checkpointing so that an interrupted execution can continue without repeating previously completed tool executions.

After every successful tool execution and after appending its observation to the message history, the agent saves a checkpoint under:

```text
outputs/state/<run_id>.json
```

Each checkpoint stores:

- `run_id`
- `scenario_id`
- Complete message history
- `tokens_used`
- Step number
- Tools already executed
- Execution state required to continue pending tool calls

The `resume_agent(run_id)` function loads the saved checkpoint and continues from the next pending tool execution.

### S1 crash-and-resume demonstration

The S1 scenario is used to demonstrate recovery.

The agent is deliberately interrupted immediately after its second tool execution, after the corresponding checkpoint has been saved.

**Tools executed before the simulated crash:**

1. `get_order`
2. `search_policy`

The agent is then resumed using `resume_agent(run_id)`.

**Tools executed after resuming:**

3. `calculate_refund`
4. `issue_refund`

### Final result

| Validation | Result |
|---|---|
| Tools executed before crash | 2 |
| Tools executed after resume | 2 |
| Total tool executions | 4 |
| Repeated tool executions | 0 |
| Final status | `REFUND` |
| Final refund amount | ₹128.25 |

**Complete execution sequence:**

```text
get_order
search_policy
calculate_refund
issue_refund
```

The resumed execution produces the expected refund result without rerunning the first two tools.

## Task 9 — ReAct versus Plan-and-Execute

The ReAct agent makes multiple model calls because it alternates between reasoning, executing tools, and observing their results. In our evaluation, S1 required 5 model calls, S5 required 3, and S6 required 2. This increases model-call overhead and can increase total token consumption because previous conversation messages and tool observations are included in subsequent prompts.

A plan-and-execute agent would ask the model to generate the complete sequence of steps initially and execute that plan. It would generally require fewer model calls when execution proceeds as expected, potentially reducing latency and token overhead. However, replanning would be necessary if an unexpected result occurred.

For example, if `get_order` returns `NOT_FOUND`, or an order is cancelled, a previously generated plan might contain inappropriate subsequent actions. ReAct can use the observation immediately to decide whether to stop, refuse the refund, or escalate the case. Plan-and-execute must detect the unexpected result and request a revised plan before proceeding.

For this refund workflow, I would choose **ReAct**. Although it may require more model calls, refund decisions depend on real-time order information and policy checks. Its ability to adapt to tool observations makes it safer and more reliable, particularly when issuing financial refunds.

## Part D — Output Artifacts

The main output artifacts are:

| Artifact | Purpose |
|---|---|
| `outputs/policy_chunks.json` | Stores the processed policy chunks. |
| `outputs/agent_runs.jsonl` | Stores agent execution records. |
| `outputs/refund_ledger.jsonl` | Records issued refunds. |
| `outputs/state/<run_id>.json` | Stores execution checkpoints for resume. |

Each agent run is recorded in `agent_runs.jsonl` with:

- `scenario_id`
- `status`
- `final_answer`
- `llm_calls`
- `tokens_used`
- Every executed step

The refund ledger records issued and escalated actions. Checkpoint files preserve the state needed to resume interrupted runs.

## Project Status

**Parts A–D:** Implementation and validation are documented in their respective sections. Final submission should include the executed scripts, generated output artifacts, and verified evaluation results.

## License

Add a license if you intend to distribute or reuse the project publicly.

## Retention List — Evaluation Limitation

The retention list ranks Premium customers from the test set using their predicted lapse probabilities.

The Premium classification is derived from the complete observation window, including September 2025. Therefore, this classification is used only for post-hoc business selection and not as a model feature.

The retention list is illustrative and should not be interpreted as a leakage-free evaluation of the model's performance on Premium customers.

The lapse prediction model itself uses historical features computed exclusively from orders before September 1, 2025.