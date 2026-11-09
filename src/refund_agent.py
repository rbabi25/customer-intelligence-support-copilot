"""
Instamart Part D — Policy Assistant and Refund Agent.

Tasks 1-4: Policy chunking, retrieval, answering and evaluation.
Task 5: Tool implementations and tool schemas.
Task 6: ReAct agent and seven-scenario evaluation.
Task 7: Token, timeout and iteration-limit handling.
Task 8: Checkpointing, crash recovery and resume.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import sys
import time
import uuid

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ================================================================
# PROJECT PATHS AND CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

POLICY_DIR = PROJECT_ROOT / "data" / "policies"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"

REFUND_LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
ESCALATION_LEDGER_PATH = OUTPUT_DIR / "escalation_ledger.jsonl"
RUN_LOG_PATH = OUTPUT_DIR / "agent_runs.jsonl"
STATE_DIR = OUTPUT_DIR / "state"

POLICY_EVAL_PATH = RAW_DIR / "policy_eval.csv"
SCENARIOS_PATH = RAW_DIR / "agent_scenarios.csv"

CHUNK_SIZE = 40
OVERLAP = 10
TOP_K = 3

DEFAULT_MAX_ITERATIONS = 6
DEFAULT_TOKEN_LIMIT = 6000
DEFAULT_TIME_LIMIT_SECONDS = 10

sys.path.insert(0, str(PROJECT_ROOT))
import mock_llm


# ================================================================
# SHARED UTILITIES
# ================================================================

def timestamp():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, record):
    """Append one JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
        )


def read_jsonl(path):
    """Read valid JSONL records, ignoring blank lines."""
    if not path.exists():
        return []

    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

    return records


def write_json(path, data):
    """Atomically write JSON data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)

    temporary_path.replace(path)


def append_run_once(record):
    """
    Append a run only once.

    This prevents a resumed or already-completed run from creating
    duplicate records in agent_runs.jsonl.
    """
    existing = read_jsonl(RUN_LOG_PATH)
    run_id = record["run_id"]

    if any(item.get("run_id") == run_id for item in existing):
        return

    append_jsonl(RUN_LOG_PATH, record)


# ================================================================
# TASK 1 — POLICY CHUNKING
# ================================================================

def chunk_policy_documents(overlap=OVERLAP):
    """
    Read exactly five policy documents and split them into word chunks.

    Each chunk contains at most 40 words.
    The next chunk starts 40 - overlap words after the previous start.
    """

    if not isinstance(overlap, int):
        raise TypeError("overlap must be an integer")

    if not 0 <= overlap < CHUNK_SIZE:
        raise ValueError(
            "overlap must be between 0 and CHUNK_SIZE - 1"
        )

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory not found: {POLICY_DIR}"
        )

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    if len(policy_files) != 5:
        raise ValueError(
            f"Expected exactly 5 policy files, found {len(policy_files)}"
        )

    chunks = []

    for file_path in policy_files:
        words = file_path.read_text(
            encoding="utf-8"
        ).split()

        start = 0

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))

            chunks.append({
                "source": file_path.name,
                "text": " ".join(words[start:end]),
                "start_word": start,
                "end_word": end - 1,
            })

            if end >= len(words):
                break

            start += CHUNK_SIZE - overlap

    return chunks


def save_policy_chunks(overlap=OVERLAP, output_name="policy_chunks.json"):
    """Create and save policy chunks."""
    chunks = chunk_policy_documents(overlap=overlap)

    output_path = OUTPUT_DIR / output_name
    write_json(output_path, chunks)

    return chunks


def load_policy_chunks(path=None):
    """Load saved policy chunks."""
    path = path or (OUTPUT_DIR / "policy_chunks.json")

    if not path.exists():
        raise FileNotFoundError(
            f"Policy chunks not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("The policy chunk file is empty.")

    return chunks


# ================================================================
# TASK 2 — TF-IDF RETRIEVAL
# ================================================================

def build_retriever(chunks):
    """Build an independent TF-IDF retriever for a chunk collection."""

    if not chunks:
        raise ValueError(
            "Cannot build a retriever from an empty chunk collection."
        )

    texts = [item["text"] for item in chunks]

    local_vectorizer = TfidfVectorizer()
    local_vectors = local_vectorizer.fit_transform(texts)

    def local_retrieve(query, k=TOP_K):
        if query is None or not str(query).strip():
            return []

        k = max(0, int(k))

        if k == 0:
            return []

        query_vector = local_vectorizer.transform([str(query)])

        scores = cosine_similarity(
            query_vector,
            local_vectors
        )[0]

        indices = scores.argsort()[::-1][:k]

        return [
            {
                "source": chunks[index]["source"],
                "score": float(scores[index]),
                "text": chunks[index]["text"],
            }
            for index in indices
        ]

    return local_retrieve


# Build the normal-overlap retriever at module load.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not (OUTPUT_DIR / "policy_chunks.json").exists():
    save_policy_chunks(overlap=OVERLAP)

policy_chunks = load_policy_chunks()
retrieve = build_retriever(policy_chunks)


# ================================================================
# TASK 3 — GROUNDED POLICY ANSWERING
# ================================================================

def answer(question, retrieve_fn=None):
    """
    Answer using only retrieved policy context.

    The mock LLM is explicitly instructed to use the fallback
    when the answer is not present in the supplied context.
    """

    if retrieve_fn is None:
        retrieve_fn = retrieve

    results = retrieve_fn(question, k=TOP_K)

    system_message = {
        "role": "system",
        "content": (
            "You are a POLICY ASSISTANT. "
            "Answer the user's question using only the supplied policy "
            "context. Do not use external knowledge or make assumptions. "
            "If the supplied context does not contain the answer, "
            "respond with a sentence beginning exactly with "
            "'I could not find'."
        ),
    }

    context_parts = [
        f"[{item['source']}]\n{item['text']}"
        for item in results
    ]

    user_message = {
        "role": "user",
        "content": (
            "CONTEXT:\n"
            + "\n\n".join(context_parts)
            + "\n\nQUESTION:\n"
            + str(question)
        ),
    }

    response = mock_llm.chat([
        system_message,
        user_message,
    ])

    model_answer = (
        response.choices[0].message.content or ""
    ).strip()

    sources = list(dict.fromkeys(
        item["source"] for item in results
    ))

    return {
        "answer": model_answer,
        "sources": sources,
    }


# ================================================================
# TASK 4 — POLICY ASSISTANT EVALUATION
# ================================================================

def evaluate_policy_assistant(
    retrieve_fn,
    label,
    output_name=None,
):
    """
    Evaluate:
      1. Expected document retrieval.
      2. Expected fact in generated answer.
      3. Correct fallback for unanswerable questions.

    Save the complete row-level evaluation to CSV.
    """

    if not POLICY_EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {POLICY_EVAL_PATH}"
        )

    evaluation = pd.read_csv(POLICY_EVAL_PATH)

    required_columns = {
        "question",
        "expected_doc",
        "expected_fact",
    }

    missing = required_columns.difference(evaluation.columns)

    if missing:
        raise ValueError(
            f"policy_eval.csv is missing columns: {sorted(missing)}"
        )

    results = []

    for _, row in evaluation.iterrows():
        question = str(row["question"])
        expected_doc = str(row["expected_doc"]).strip()

        expected_fact = row["expected_fact"]

        retrieved = retrieve_fn(question, k=TOP_K)

        retrieved_sources = [
            item["source"] for item in retrieved
        ]

        response = answer(
            question,
            retrieve_fn=retrieve_fn,
        )

        model_answer = response["answer"]

        if expected_doc == "NONE":
            retrieval_hit = None
            grounding_hit = None
            fallback_hit = model_answer.startswith(
                "I could not find"
            )
        else:
            retrieval_hit = expected_doc in retrieved_sources

            grounding_hit = (
                isinstance(expected_fact, str)
                and expected_fact.strip().lower()
                in model_answer.lower()
            )

            fallback_hit = None

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answer": model_answer,
            "retrieved_sources": ";".join(retrieved_sources),
            "retrieval_hit": retrieval_hit,
            "grounding_hit": grounding_hit,
            "fallback_hit": fallback_hit,
        })

    results_df = pd.DataFrame(results)

    answerable = results_df[
        results_df["expected_doc"] != "NONE"
    ]

    unanswerable = results_df[
        results_df["expected_doc"] == "NONE"
    ]

    retrieval_hits = int(
        answerable["retrieval_hit"].fillna(False).sum()
    )

    grounding_hits = int(
        answerable["grounding_hit"].fillna(False).sum()
    )

    fallback_hits = int(
        unanswerable["fallback_hit"].fillna(False).sum()
    )

    retrieval_failures = len(answerable) - retrieval_hits
    grounding_failures = len(answerable) - grounding_hits
    fallback_failures = len(unanswerable) - fallback_hits

    if output_name is None:
        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        output_name = f"policy_eval_{safe_label}.csv"

    output_path = OUTPUT_DIR / output_name
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print(f"TASK 4 — POLICY EVALUATION: {label}")
    print("=" * 65)

    print("Total questions:", len(results_df))
    print("Answerable questions:", len(answerable))
    print(
        f"Expected document retrieved: "
        f"{retrieval_hits}/{len(answerable)}"
    )
    print(
        f"Expected fact in answer: "
        f"{grounding_hits}/{len(answerable)}"
    )
    print("Unanswerable questions:", len(unanswerable))
    print(
        f"Correct fallback answers: "
        f"{fallback_hits}/{len(unanswerable)}"
    )

    print("\nFailure counts")
    print("Retrieval failures:", retrieval_failures)
    print("Grounding failures:", grounding_failures)
    print("Fallback failures:", fallback_failures)

    failed_rows = results_df[
        (
            (results_df["expected_doc"] != "NONE")
            & (
                (~results_df["retrieval_hit"].fillna(False))
                | (~results_df["grounding_hit"].fillna(False))
            )
        )
        |
        (
            (results_df["expected_doc"] == "NONE")
            & (~results_df["fallback_hit"].fillna(False))
        )
    ]

    if not failed_rows.empty:
        print("\nFAILED QUESTIONS")

        for _, failure in failed_rows.iterrows():
            print("\nQuestion:", failure["question"])
            print("Expected document:", failure["expected_doc"])
            print("Expected fact:", failure["expected_fact"])
            print("Retrieved:", failure["retrieved_sources"])
            print("Answer:", failure["answer"])

    print("\nEvaluation CSV saved:", output_path)

    summary = {
        "label": label,
        "total_questions": len(results_df),
        "answerable_questions": len(answerable),
        "retrieval_hits": retrieval_hits,
        "grounding_hits": grounding_hits,
        "unanswerable_questions": len(unanswerable),
        "fallback_hits": fallback_hits,
        "retrieval_failures": retrieval_failures,
        "grounding_failures": grounding_failures,
        "fallback_failures": fallback_failures,
        "failed_questions": len(failed_rows),
        "output_path": str(output_path),
    }

    return results_df, summary


def run_task4_evaluations():
    """Run Task 4 using normal overlap and zero overlap."""

    print("\n" + "=" * 65)
    print("TASK 4 — NORMAL OVERLAP EVALUATION")
    print("=" * 65)

    normal_chunks = chunk_policy_documents(
        overlap=OVERLAP
    )

    normal_retriever = build_retriever(normal_chunks)

    normal_df, normal_summary = evaluate_policy_assistant(
        normal_retriever,
        label="NORMAL OVERLAP (10)",
        output_name="policy_eval_normal_overlap.csv",
    )

    print("\nNormal-overlap chunk count:", len(normal_chunks))

    print("\n" + "=" * 65)
    print("TASK 4 — ZERO OVERLAP EVALUATION")
    print("=" * 65)

    zero_chunks = chunk_policy_documents(overlap=0)
    zero_retriever = build_retriever(zero_chunks)

    zero_df, zero_summary = evaluate_policy_assistant(
        zero_retriever,
        label="ZERO OVERLAP (0)",
        output_name="policy_eval_zero_overlap.csv",
    )

    print("\nZero-overlap chunk count:", len(zero_chunks))

    print("\n" + "=" * 65)
    print("TASK 4 — COMPARISON")
    print("=" * 65)

    print(
        "Normal-overlap chunks:",
        len(normal_chunks),
    )
    print(
        "Zero-overlap chunks:",
        len(zero_chunks),
    )

    print(
        "Normal-overlap retrieval:",
        f"{normal_summary['retrieval_hits']}/"
        f"{normal_summary['answerable_questions']}",
    )

    print(
        "Zero-overlap retrieval:",
        f"{zero_summary['retrieval_hits']}/"
        f"{zero_summary['answerable_questions']}",
    )

    print(
        "Normal-overlap grounding:",
        f"{normal_summary['grounding_hits']}/"
        f"{normal_summary['answerable_questions']}",
    )

    print(
        "Zero-overlap grounding:",
        f"{zero_summary['grounding_hits']}/"
        f"{zero_summary['answerable_questions']}",
    )

    return {
        "normal": normal_summary,
        "zero_overlap": zero_summary,
    }


# ================================================================
# TASK 5 — TOOL IMPLEMENTATIONS
# ================================================================

def get_order(order_id):
    """Retrieve order details from SQLite."""

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                order_id,
                status,
                order_amount,
                delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return {
                "error": f"Order {order_id} not found."
            }

        order = dict(row)

        try:
            delivery = order["delivery_minutes"]

            order["delivery_minutes"] = (
                int(float(delivery))
                if delivery is not None
                else None
            )
        except (ValueError, TypeError):
            order["delivery_minutes"] = None

        order["order_amount"] = float(
            order["order_amount"]
        )

        return order

    finally:
        conn.close()


def search_policy(query):
    """Retrieve the top three policy chunks."""
    return {
        "results": retrieve(query, k=TOP_K)
    }


def calculate_refund(order_amount, delivery_minutes):
    """Calculate refund eligibility based on the 45-minute rule."""

    if delivery_minutes is None:
        return {
            "error": (
                "Delivery time is unavailable. "
                "Cannot calculate refund."
            )
        }

    order_amount = float(order_amount)
    delivery_minutes = int(delivery_minutes)

    refund_amount = (
        order_amount
        if delivery_minutes > 45
        else 0.0
    )

    return {
        "order_amount": order_amount,
        "delivery_minutes": delivery_minutes,
        "refund_inr": round(refund_amount, 2),
    }


def _ledger_has_order(path, order_id):
    """Check whether an order already has a ledger entry."""
    for record in read_jsonl(path):
        if record.get("order_id") == order_id:
            return record

    return None


def issue_refund(order_id, refund_inr):
    """Record a refund once per order."""

    refund_inr = round(float(refund_inr), 2)

    if refund_inr <= 0:
        return {
            "error": "Refund amount must be greater than zero."
        }

    existing = _ledger_has_order(
        REFUND_LEDGER_PATH,
        order_id,
    )

    if existing is not None:
        return {
            "success": True,
            "order_id": order_id,
            "refund_inr": round(
                float(existing["refund_inr"]), 2
            ),
            "already_recorded": True,
            "message": (
                f"Refund already recorded for {order_id}."
            ),
        }

    record = {
        "timestamp": timestamp(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": refund_inr,
    }

    append_jsonl(REFUND_LEDGER_PATH, record)

    return {
        "success": True,
        "order_id": order_id,
        "refund_inr": refund_inr,
        "already_recorded": False,
        "message": (
            f"Refund of INR {refund_inr:.2f} "
            f"recorded for {order_id}."
        ),
    }


def escalate_to_human(order_id, refund_inr):
    """Record a human escalation once per order."""

    refund_inr = round(float(refund_inr), 2)

    if refund_inr <= 1000:
        return {
            "error": (
                "Human escalation is only required "
                "for refunds above INR 1000."
            )
        }

    existing = _ledger_has_order(
        ESCALATION_LEDGER_PATH,
        order_id,
    )

    if existing is not None:
        return {
            "success": True,
            "order_id": order_id,
            "refund_inr": round(
                float(existing["refund_inr"]), 2
            ),
            "already_recorded": True,
            "message": (
                f"Escalation already recorded for {order_id}."
            ),
        }

    record = {
        "timestamp": timestamp(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": refund_inr,
        "reason": (
            "Refund exceeds INR 1000 and requires "
            "human support lead approval."
        ),
    }

    append_jsonl(ESCALATION_LEDGER_PATH, record)

    return {
        "success": True,
        "order_id": order_id,
        "refund_inr": refund_inr,
        "already_recorded": False,
        "message": "Refund escalated to human support.",
    }


# ================================================================
# TOOL SCHEMAS
# ================================================================

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": (
                "Retrieve order status, amount and delivery time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "Search policy documents for relevant rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": (
                "Calculate the eligible refund from "
                "order amount and delivery time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {
                        "type": ["integer", "null"]
                    },
                },
                "required": [
                    "order_amount",
                    "delivery_minutes",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": (
                "Record an approved refund in the refund ledger."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": [
                    "order_id",
                    "refund_inr",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Escalate a refund above INR 1000 "
                "for human support lead approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": [
                    "order_id",
                    "refund_inr",
                ],
            },
        },
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}
# ================================================================
# TASK 6 — REACT AGENT, CHECKPOINTING AND RUN LOGGING
# ================================================================

class SimulatedCrash(RuntimeError):
    """Used only to test checkpoint recovery."""


def checkpoint_path(run_id):
    return STATE_DIR / f"{run_id}.json"

def save_checkpoint(state):
    """Save agent state safely, including on Windows."""

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    path = checkpoint_path(state["run_id"])

    temporary_path = STATE_DIR / (
        f"{state['run_id']}_{uuid.uuid4().hex}.tmp"
    )

    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(
                state,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            file.flush()

        # Retry transient Windows file-lock errors.
        last_error = None

        for attempt in range(5):
            try:
                temporary_path.replace(path)
                return path

            except PermissionError as exc:
                last_error = exc
                time.sleep(0.2 * (attempt + 1))

        raise last_error

    finally:
        # Remove any temporary file left behind.
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass

def load_checkpoint(run_id):
    """Load and validate a saved agent checkpoint."""
    path = checkpoint_path(run_id)

    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run_id={run_id}"
        )

    with open(path, "r", encoding="utf-8") as file:
        state = json.load(file)

    required_fields = {
        "run_id",
        "scenario_id",
        "messages",
        "tokens_used",
        "step_number",
        "tools_executed",
        "steps",
        "llm_calls",
        "iteration",
        "pending_tool_calls",
        "pending_tool_index",
        "status",
        "final_answer",
        "elapsed_seconds",
        "max_iterations",
        "token_limit",
        "time_limit_seconds",
        "run_logged",
    }

    missing = required_fields.difference(state)

    if missing:
        raise ValueError(
            f"Checkpoint is missing fields: {sorted(missing)}"
        )

    if not isinstance(state["messages"], list):
        raise ValueError("Checkpoint messages must be a list.")

    if not isinstance(state["tools_executed"], list):
        raise ValueError(
            "Checkpoint tools_executed must be a list."
        )

    if not isinstance(state["steps"], list):
        raise ValueError("Checkpoint steps must be a list.")

    return state


def normalize_tool_call(tool_call):
    """
    Convert a mock-LLM tool call into a JSON-serializable dictionary.

    Supports the tool-call object shape used by mock_llm.py.
    """

    arguments = tool_call.arguments

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if not isinstance(arguments, dict):
        raise ValueError(
            "Tool arguments must be a dictionary."
        )

    return {
        "id": str(tool_call.id),
        "name": str(tool_call.name),
        "arguments": arguments,
    }


def result_from_state(state):
    """Return the public result structure for an agent run."""
    return {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": state["status"],
        "final_answer": state["final_answer"],
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"],
        "elapsed_seconds": round(
            float(state["elapsed_seconds"]), 4
        ),
    }


def log_agent_run(state):
    """
    Persist the final run result to agent_runs.jsonl.

    This is called for normal runs, limit demonstrations and resumed runs.
    """

    if state["status"] == "RUNNING":
        return

    if state.get("run_logged", False):
        return

    record = {
        "timestamp": timestamp(),
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": state["status"],
        "final_answer": state["final_answer"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "elapsed_seconds": round(
            float(state["elapsed_seconds"]), 4
        ),
        "step_number": state["step_number"],
        "tools_executed": state["tools_executed"],
        "steps": state["steps"],
    }

    append_run_once(record)

    state["run_logged"] = True
    save_checkpoint(state)


def finish_agent(state, status, final_answer, started_at):
    """Set terminal state, save it, log it, and return the result."""

    state["status"] = status
    state["final_answer"] = final_answer

    state["elapsed_seconds"] += (
        time.monotonic() - started_at
    )

    save_checkpoint(state)
    log_agent_run(state)

    return result_from_state(state)


def classify_final_status(final_answer):
    """Map mock-LLM final-answer prefixes to scenario statuses."""

    if final_answer.startswith("REFUND:"):
        return "REFUND"

    if final_answer.startswith("ESCALATE:"):
        return "ESCALATE"

    if final_answer.startswith("DENIED:"):
        return "DENIED"

    if final_answer.startswith("NOT_FOUND:"):
        return "NOT_FOUND"

    if final_answer.startswith("NEEDS_INFO:"):
        return "NEEDS_INFO"

    return "COMPLETED"


def run_agent_state(state, crash_after_tool=None):
    """
    Continue an agent from its current state.

    Every successful tool execution is checkpointed before the next
    operation. The pending-tool index is advanced before saving, so
    resume will not execute an already-completed tool again.
    """

    started_at = time.monotonic()
    tools_executed_this_call = 0

    def elapsed_now():
        return (
            float(state["elapsed_seconds"])
            + time.monotonic()
            - started_at
        )

    while state["iteration"] < state["max_iterations"]:

        # --------------------------------------------------------
        # Process pending tool calls from the previous LLM response.
        # --------------------------------------------------------

        pending = state["pending_tool_calls"]

        while state["pending_tool_index"] < len(pending):

            if elapsed_now() >= state["time_limit_seconds"]:
                return finish_agent(
                    state,
                    "FALLBACK_TIMEOUT",
                    (
                        "Request timed out. "
                        "Please contact human support."
                    ),
                    started_at,
                )

            tool_call = pending[
                state["pending_tool_index"]
            ]

            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]

            if tool_name not in TOOL_MAP:
                raise ValueError(
                    f"Unknown tool requested by model: {tool_name}"
                )

            if not isinstance(arguments, dict):
                raise ValueError(
                    f"Invalid arguments for tool {tool_name}"
                )

            # Execute one tool.
            tool_result = TOOL_MAP[tool_name](**arguments)

            # Record the result.
            step = {
                "iteration": state["iteration"],
                "tool": tool_name,
                "arguments": arguments,
                "result": tool_result,
            }

            state["steps"].append(step)
            state["step_number"] += 1

            state["tools_executed"].append({
                "step_number": state["step_number"],
                "tool": tool_name,
                "arguments": arguments,
                "result": tool_result,
            })

            # Append the tool response to the conversation.
            state["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": json.dumps(
                    tool_result,
                    ensure_ascii=False,
                ),
            })

            # Advance the index BEFORE saving the checkpoint.
            # This is critical for preventing duplicate execution.
            state["pending_tool_index"] += 1

            save_checkpoint(state)

            tools_executed_this_call += 1

            # Simulate a process crash after the requested number
            # of successful tool executions.
            if (
                crash_after_tool is not None
                and tools_executed_this_call >= crash_after_tool
            ):
                raise SimulatedCrash(
                    "Simulated crash after "
                    f"{tools_executed_this_call} tool execution(s). "
                    "Checkpoint saved."
                )

        # All calls from the previous assistant message are complete.
        state["pending_tool_calls"] = []
        state["pending_tool_index"] = 0

        if elapsed_now() >= state["time_limit_seconds"]:
            return finish_agent(
                state,
                "FALLBACK_TIMEOUT",
                (
                    "Request timed out. "
                    "Please contact human support."
                ),
                started_at,
            )

        # --------------------------------------------------------
        # Ask the model for its next action.
        # --------------------------------------------------------

        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC,
        )

        state["llm_calls"] += 1

        response_tokens = int(
            response.usage.total_tokens or 0
        )

        state["tokens_used"] += response_tokens

        # Check the cumulative token count after each LLM call.
        if state["tokens_used"] > state["token_limit"]:
            return finish_agent(
                state,
                "FALLBACK_TOKEN_LIMIT",
                (
                    "Token limit exceeded. "
                    "Please contact human support."
                ),
                started_at,
            )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # --------------------------------------------------------
        # The model returned a final answer.
        # --------------------------------------------------------

        if not tool_calls:
            final_answer = (
                message.content or ""
            ).strip()

            status = classify_final_status(final_answer)

            return finish_agent(
                state,
                status,
                final_answer,
                started_at,
            )

        # --------------------------------------------------------
        # The model requested one or more tools.
        # --------------------------------------------------------

        normalized_calls = [
            normalize_tool_call(call)
            for call in tool_calls
        ]

        assistant_tool_calls = []

        for call in normalized_calls:
            assistant_tool_calls.append({
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call["arguments"],
                        ensure_ascii=False,
                    ),
                },
            })

        state["messages"].append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": assistant_tool_calls,
        })

        state["pending_tool_calls"] = normalized_calls
        state["pending_tool_index"] = 0

        # One iteration means one LLM action/response cycle.
        state["iteration"] += 1

        save_checkpoint(state)

    # ------------------------------------------------------------
    # Iteration limit reached.
    # ------------------------------------------------------------

    return finish_agent(
        state,
        "FALLBACK_MAX_ITERATIONS",
        (
            "Maximum iterations reached. "
            "Please contact human support."
        ),
        started_at,
    )


def run_agent(
    scenario,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    token_limit=DEFAULT_TOKEN_LIMIT,
    time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
    run_id=None,
    scenario_id=None,
    crash_after_tool=None,
):
    """Start a new ReAct agent run."""

    max_iterations = int(max_iterations)
    token_limit = int(token_limit)
    time_limit_seconds = float(time_limit_seconds)

    if max_iterations < 0:
        raise ValueError(
            "max_iterations cannot be negative."
        )

    if token_limit < 0:
        raise ValueError(
            "token_limit cannot be negative."
        )

    if time_limit_seconds < 0:
        raise ValueError(
            "time_limit_seconds cannot be negative."
        )

    if crash_after_tool is not None:
        crash_after_tool = int(crash_after_tool)

        if crash_after_tool <= 0:
            raise ValueError(
                "crash_after_tool must be greater than zero."
            )

    run_id = run_id or str(uuid.uuid4())

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id or "ad_hoc",

        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AGENT. Use the available tools to "
                    "complete the user's request. Follow the policies "
                    "and never issue refunds for ineligible orders."
                ),
            },
            {
                "role": "user",
                "content": str(scenario),
            },
        ],

        "tokens_used": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "llm_calls": 0,
        "iteration": 0,

        "pending_tool_calls": [],
        "pending_tool_index": 0,

        "status": "RUNNING",
        "final_answer": "",

        "elapsed_seconds": 0.0,

        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,

        "run_logged": False,
    }

    # Save the initial state so that even a run interrupted before
    # its first tool call has a recoverable checkpoint.
    save_checkpoint(state)

    return run_agent_state(
        state,
        crash_after_tool=crash_after_tool,
    )


def resume_agent(run_id, crash_after_tool=None):
    """Resume a saved run without repeating completed tool calls."""

    state = load_checkpoint(run_id)

    if state["status"] != "RUNNING":
        return result_from_state(state)

    # If a completed result was saved before logging, ensure it is logged.
    if state["status"] != "RUNNING":
        log_agent_run(state)
        return result_from_state(state)

    return run_agent_state(
        state,
        crash_after_tool=crash_after_tool,
    )


# ================================================================
# TASK 6 — SEVEN-SCENARIO EVALUATION
# ================================================================

def extract_refund_amount(result):
    """Extract the refund/escalation amount from executed tool results."""

    for step in result["steps"]:
        if step["tool"] in (
            "issue_refund",
            "escalate_to_human",
        ):
            amount = step["result"].get("refund_inr")

            if amount is not None:
                return round(float(amount), 2)

    return None


def run_scenario_evaluation():
    """Run and evaluate all seven supplied agent scenarios."""

    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(
            f"Scenario CSV not found: {SCENARIOS_PATH}"
        )

    scenarios = pd.read_csv(SCENARIOS_PATH)

    required_columns = {
        "scenario_id",
        "customer_id",
        "message",
        "expected_outcome",
        "expected_refund_inr",
    }

    missing = required_columns.difference(scenarios.columns)

    if missing:
        raise ValueError(
            f"agent_scenarios.csv is missing columns: "
            f"{sorted(missing)}"
        )

    all_results = []

    print("\n" + "=" * 65)
    print("TASK 6 — SEVEN-SCENARIO AGENT EVALUATION")
    print("=" * 65)

    for _, row in scenarios.iterrows():

        scenario_id = str(row["scenario_id"])
        message = str(row["message"])
        expected = str(row["expected_outcome"]).strip()

        expected_amount = round(
            float(row["expected_refund_inr"]),
            2,
        )

        print("\n" + "-" * 60)
        print(f"Testing {scenario_id}")
        print("-" * 60)

        result = run_agent(
            message,
            scenario_id=scenario_id,
        )

        actual = result["status"]
        actual_amount = extract_refund_amount(result)

        # --------------------------------------------------------
        # Outcome validation.
        # --------------------------------------------------------

        if expected == "FALLBACK":
            # S7 must specifically hit the iteration limit.
            outcome_passed = (
                actual == "FALLBACK_MAX_ITERATIONS"
            )
        else:
            outcome_passed = actual == expected

        # --------------------------------------------------------
        # Refund amount validation.
        # --------------------------------------------------------

        if expected in ("REFUND", "ESCALATE"):
            amount_passed = (
                actual_amount is not None
                and abs(actual_amount - expected_amount) < 0.01
            )
        else:
            amount_passed = (
                actual_amount is None
                or abs(actual_amount) < 0.01
            )

        passed = outcome_passed and amount_passed

        record = {
            "timestamp": timestamp(),
            "scenario_id": scenario_id,
            "run_id": result["run_id"],
            "expected_outcome": expected,
            "actual_outcome": actual,
            "status": actual,
            "expected_refund_inr": expected_amount,
            "actual_refund_inr": actual_amount,
            "outcome_passed": outcome_passed,
            "amount_passed": amount_passed,
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "elapsed_seconds": result["elapsed_seconds"],
            "steps": result["steps"],
            "tools_executed": result["tools_executed"],
            "passed": passed,
        }

        all_results.append(record)

        # run_agent already logs the full execution record.
        # This scenario record adds expected-vs-actual evaluation.
        append_jsonl(
            OUTPUT_DIR / "scenario_evaluation.jsonl",
            record,
        )

        print("Expected outcome:", expected)
        print("Actual outcome:", actual)
        print("Expected refund:", expected_amount)
        print("Actual refund:", actual_amount)
        print("Outcome passed:", outcome_passed)
        print("Amount passed:", amount_passed)
        print("Overall:", "PASSED" if passed else "FAILED")
        print("LLM calls:", result["llm_calls"])
        print("Tokens:", result["tokens_used"])
        print(
            "Tool sequence:",
            [step["tool"] for step in result["steps"]],
        )

    passed_count = sum(
        item["passed"] for item in all_results
    )

    failed_count = len(all_results) - passed_count

    print("\n" + "=" * 65)
    print("TASK 6 — SCENARIO SUMMARY")
    print("=" * 65)

    print(
        f"Passed: {passed_count}/{len(all_results)}"
    )
    print(
        f"Failed: {failed_count}/{len(all_results)}"
    )

    print("Run log:", RUN_LOG_PATH)
    print(
        "Scenario evaluation:",
        OUTPUT_DIR / "scenario_evaluation.jsonl",
    )

    return all_results


# ================================================================
# TASK 7 — TOKEN, TIMEOUT AND ITERATION LIMIT TESTS
# ================================================================

def test_agent_limits():
    """Demonstrate all three agent execution limits."""

    print("\n" + "=" * 65)
    print("TASK 7 — AGENT LIMIT TESTS")
    print("=" * 65)

    test_message = (
        "Please process a refund for order ORD-100048."
    )

    # ------------------------------------------------------------
    # 1. Token limit.
    # ------------------------------------------------------------

    print("\nTOKEN LIMIT TEST")

    token_result = run_agent(
        test_message,
        scenario_id="TASK7_TOKEN_LIMIT",
        token_limit=200,
    )

    print("Status:", token_result["status"])
    print("Tokens used:", token_result["tokens_used"])
    print("LLM calls:", token_result["llm_calls"])
    print("Tool executions:", len(token_result["steps"]))

    assert (
        token_result["status"] == "FALLBACK_TOKEN_LIMIT"
    ), (
        "Token-limit demonstration did not produce "
        "FALLBACK_TOKEN_LIMIT."
    )

    assert token_result["tokens_used"] > 200, (
        "Token count did not exceed the configured limit."
    )

    # ------------------------------------------------------------
    # 2. Timeout.
    # ------------------------------------------------------------

    print("\nTIMEOUT TEST")

    timeout_result = run_agent(
        test_message,
        scenario_id="TASK7_TIMEOUT",
        time_limit_seconds=0,
    )

    print("Status:", timeout_result["status"])
    print("LLM calls:", timeout_result["llm_calls"])
    print("Tool executions:", len(timeout_result["steps"]))

    assert (
        timeout_result["status"] == "FALLBACK_TIMEOUT"
    ), (
        "Timeout demonstration did not produce "
        "FALLBACK_TIMEOUT."
    )

    assert timeout_result["llm_calls"] == 0, (
        "The zero-second timeout should stop before the LLM call."
    )

    assert len(timeout_result["steps"]) == 0, (
        "The zero-second timeout should stop before tool execution."
    )

    # ------------------------------------------------------------
    # 3. Maximum iterations.
    # ------------------------------------------------------------

    print("\nMAXIMUM ITERATIONS TEST")

    iteration_result = run_agent(
        test_message,
        scenario_id="TASK7_MAX_ITERATIONS",
        max_iterations=0,
    )

    print("Status:", iteration_result["status"])
    print("LLM calls:", iteration_result["llm_calls"])
    print("Tool executions:", len(iteration_result["steps"]))

    assert (
        iteration_result["status"] == "FALLBACK_MAX_ITERATIONS"
    ), (
        "Iteration demonstration did not produce "
        "FALLBACK_MAX_ITERATIONS."
    )

    assert iteration_result["llm_calls"] == 0, (
        "A zero-iteration run must not call the LLM."
    )

    assert len(iteration_result["steps"]) == 0, (
        "A zero-iteration run must not execute tools."
    )

    print("\nALL TASK 7 LIMIT CHECKS PASSED")

    return {
        "token_limit": token_result,
        "timeout": timeout_result,
        "max_iterations": iteration_result,
    }


# ================================================================
# TASK 8 — CHECKPOINT AND RESUME TEST
# ================================================================

def test_checkpoint_resume_s1():
    """
    Simulate S1 crashing after its second successful tool execution.

    Resume from the saved checkpoint and verify:
      - The first two tools are not executed again.
      - The final sequence contains exactly four tools.
      - The final status is REFUND.
      - The refund amount matches the scenario CSV.
      - Tool step numbers are unique and sequential.
    """

    scenario_id = "S1"

    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(
            f"Scenario CSV not found: {SCENARIOS_PATH}"
        )

    scenarios = pd.read_csv(SCENARIOS_PATH)

    matching = scenarios[
        scenarios["scenario_id"].astype(str) == scenario_id
    ]

    if matching.empty:
        raise ValueError(
            f"{scenario_id} not found in agent_scenarios.csv"
        )

    row = matching.iloc[0]

    scenario_text = str(row["message"])
    expected_amount = round(
        float(row["expected_refund_inr"]),
        2,
    )

    run_id = (
        f"task8_{scenario_id}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    print("\n" + "=" * 65)
    print("TASK 8 — CHECKPOINT AND RESUME TEST")
    print("=" * 65)

    # ------------------------------------------------------------
    # Start S1 and simulate a crash after tool 2.
    # ------------------------------------------------------------

    print("\nStarting S1 with simulated crash after tool 2.")

    try:
        run_agent(
            scenario_text,
            run_id=run_id,
            scenario_id=f"TASK8_{scenario_id}",
            crash_after_tool=2,
        )

        raise AssertionError(
            "Simulated crash was not triggered."
        )

    except SimulatedCrash as exc:
        print(exc)

    # ------------------------------------------------------------
    # Inspect checkpoint before resuming.
    # ------------------------------------------------------------

    checkpoint = load_checkpoint(run_id)

    pre_resume_tools = checkpoint["tools_executed"]

    pre_resume_sequence = [
        item["tool"] for item in pre_resume_tools
    ]

    print(
        "Tools checkpointed before crash:",
        len(pre_resume_tools),
    )

    print(
        "Pre-crash tool sequence:",
        pre_resume_sequence,
    )

    assert len(pre_resume_tools) == 2, (
        "Expected exactly two checkpointed tools before crash."
    )

    assert pre_resume_sequence == [
        "get_order",
        "search_policy",
    ], (
        "Unexpected tool sequence before simulated crash: "
        f"{pre_resume_sequence}"
    )

    # ------------------------------------------------------------
    # Resume the saved run.
    # ------------------------------------------------------------

    print("\nResuming from saved checkpoint.")

    resumed = resume_agent(run_id)

    tool_sequence = [
        item["tool"]
        for item in resumed["tools_executed"]
    ]

    step_numbers = [
        item["step_number"]
        for item in resumed["tools_executed"]
    ]

    print("Final status:", resumed["status"])
    print("Final answer:", resumed["final_answer"])
    print("Complete tool sequence:", tool_sequence)
    print("Total tool executions:", len(tool_sequence))
    print("Step numbers:", step_numbers)

    # ------------------------------------------------------------
    # Extract final refund.
    # ------------------------------------------------------------

    refund_amount = None

    for item in resumed["tools_executed"]:
        if item["tool"] == "issue_refund":
            refund_amount = item["result"].get("refund_inr")

    # ------------------------------------------------------------
    # Assertions.
    # ------------------------------------------------------------

    expected_sequence = [
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
    ]

    assert resumed["status"] == "REFUND", (
        f"Expected REFUND, received {resumed['status']}."
    )

    assert tool_sequence == expected_sequence, (
        "Incorrect tool sequence after resume. "
        f"Expected {expected_sequence}, got {tool_sequence}."
    )

    assert len(tool_sequence) == 4, (
        f"Expected 4 tool executions, got {len(tool_sequence)}."
    )

    assert refund_amount is not None, (
        "No refund amount found in issue_refund result."
    )

    assert abs(
        float(refund_amount) - expected_amount
    ) < 0.01, (
        f"Expected refund {expected_amount}, "
        f"received {refund_amount}."
    )

    assert step_numbers == [1, 2, 3, 4], (
        f"Step numbers are not sequential: {step_numbers}"
    )

    assert len(set(step_numbers)) == len(step_numbers), (
        "Duplicate tool step numbers detected."
    )

    # The checkpoint itself must contain the completed result.
    final_checkpoint = load_checkpoint(run_id)

    assert final_checkpoint["status"] == "REFUND"
    assert len(final_checkpoint["tools_executed"]) == 4

    print("\nTASK 8 CHECKPOINT AND RESUME TEST PASSED")
    print("Run ID:", run_id)
    print("Refund amount:", refund_amount)
    print("Checkpoint:", checkpoint_path(run_id))

    return resumed


# ================================================================
# MAIN — RUN TASKS 1 THROUGH 8
# ================================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # TASK 1 — Recreate and save the normal-overlap chunks.
    # ------------------------------------------------------------

    print("\n" + "=" * 65)
    print("TASK 1 — POLICY CHUNKING")
    print("=" * 65)

    saved_chunks = save_policy_chunks(
        overlap=OVERLAP,
    )

    print("Policy files:", len(list(POLICY_DIR.glob("*.txt"))))
    print("Chunk size:", CHUNK_SIZE)
    print("Overlap:", OVERLAP)
    print("Total chunks:", len(saved_chunks))
    print(
        "Saved to:",
        OUTPUT_DIR / "policy_chunks.json",
    )

    # Refresh the global retriever so Tasks 2, 3 and 5 use
    # the latest normal-overlap chunk collection.
    global policy_chunks, retrieve

    policy_chunks = load_policy_chunks()
    retrieve = build_retriever(policy_chunks)

    # ------------------------------------------------------------
    # TASK 2 — Retrieval smoke test.
    # ------------------------------------------------------------

    print("\n" + "=" * 65)
    print("TASK 2 — RETRIEVAL TEST")
    print("=" * 65)

    retrieval_test = retrieve(
        "What is the refund policy for late delivery?",
        k=TOP_K,
    )

    print("Retrieved chunks:", len(retrieval_test))

    for index, item in enumerate(retrieval_test, start=1):
        print(f"\nResult {index}")
        print("Source:", item["source"])
        print("Score:", round(item["score"], 4))
        print("Text:", item["text"])

    assert len(retrieval_test) == TOP_K

    # ------------------------------------------------------------
    # TASK 3 — Grounded answering smoke test.
    # ------------------------------------------------------------

    print("\n" + "=" * 65)
    print("TASK 3 — POLICY ANSWERING TEST")
    print("=" * 65)

    policy_response = answer(
        "What delivery time makes a completed order eligible "
        "for a full refund?"
    )

    print("Answer:", policy_response["answer"])
    print("Sources:", policy_response["sources"])

    # ------------------------------------------------------------
    # TASK 4 — Normal overlap and zero overlap.
    # ------------------------------------------------------------

    task4_results = run_task4_evaluations()

    # ------------------------------------------------------------
    # TASK 5 — Tool registration and smoke tests.
    # ------------------------------------------------------------

    print("\n" + "=" * 65)
    print("TASK 5 — TOOL REGISTRATION CHECK")
    print("=" * 65)

    print("Tools registered:", list(TOOL_MAP.keys()))

    expected_tools = {
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
        "escalate_to_human",
    }

    assert len(TOOLS_SPEC) == 5
    assert set(TOOL_MAP.keys()) == expected_tools

    print("\nTesting get_order.")

    sample_order = get_order("ORD-110093")
    print(sample_order)

    assert "error" not in sample_order
    assert sample_order["delivery_minutes"] == 19

    print("\nTesting search_policy.")

    policy_result = search_policy(
        "refund policy late delivery"
    )

    print(
        "Retrieved chunks:",
        len(policy_result["results"]),
    )

    assert len(policy_result["results"]) == 3

    print("\nTesting calculate_refund.")

    refund_result = calculate_refund(
        127.50,
        60,
    )

    print(refund_result)

    assert refund_result["refund_inr"] == 127.50

    print("\nTASK 5 CHECKPOINT PASSED")

    # ------------------------------------------------------------
    # TASK 6 — Seven scenarios.
    # ------------------------------------------------------------

    scenario_results = run_scenario_evaluation()

    # ------------------------------------------------------------
    # TASK 7 — Token, timeout and iteration limits.
    # ------------------------------------------------------------

    limit_results = test_agent_limits()

    # ------------------------------------------------------------
    # TASK 8 — Crash recovery and resume.
    # ------------------------------------------------------------

    resume_result = test_checkpoint_resume_s1()

    # ------------------------------------------------------------
    # FINAL SUMMARY.
    # ------------------------------------------------------------

    print("\n" + "=" * 65)
    print("PART D — FINAL EXECUTION SUMMARY")
    print("=" * 65)

    print(
        "Task 1 chunks:",
        len(saved_chunks),
    )

    print(
        "Task 4 normal-overlap chunks:",
        len(chunk_policy_documents(overlap=OVERLAP)),
    )

    print(
        "Task 4 zero-overlap chunks:",
        len(chunk_policy_documents(overlap=0)),
    )

    print(
        "Task 6 scenarios passed:",
        sum(item["passed"] for item in scenario_results),
        "/",
        len(scenario_results),
    )

    print(
        "Task 7 limit tests:",
        len(limit_results),
        "/ 3",
    )

    print(
        "Task 8 resume status:",
        resume_result["status"],
    )

    print("\nOutput files:")
    for path in [
        OUTPUT_DIR / "policy_chunks.json",
        OUTPUT_DIR / "policy_eval_normal_overlap.csv",
        OUTPUT_DIR / "policy_eval_zero_overlap.csv",
        OUTPUT_DIR / "scenario_evaluation.jsonl",
        RUN_LOG_PATH,
        REFUND_LEDGER_PATH,
        ESCALATION_LEDGER_PATH,
    ]:
        print(
            f"{path.name}: "
            f"{'EXISTS' if path.exists() else 'NOT CREATED'}"
        )

    print("\nAll requested tasks have finished executing.")


if __name__ == "__main__":
    main()