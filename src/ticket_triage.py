
# ============================================================
# PART C — SUPPORT TICKET TRIAGE PIPELINE
# ============================================================

import sys
import json
import re
import time
import random
from pathlib import Path

import pandas as pd

# ============================================================
# 0. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import mock_llm from project root
sys.path.insert(0, str(PROJECT_ROOT))
import mock_llm

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKET_FILE = RAW_DIR / "tickets.csv"
SEGMENT_FILE = OUTPUT_DIR / "customer_segments.csv"

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

TOKEN_BUDGET = 6000
FINAL_TOKEN_BUDGET = 100000

ALLOWED_CATEGORIES = {
    "billing", "delivery", "technical", "fraud"
}


# ============================================================
# 1. LOAD TICKETS
# ============================================================

tickets = pd.read_csv(TICKET_FILE)

required_columns = {
    "ticket_id", "customer_id", "order_id",
    "text", "true_category"
}

assert required_columns.issubset(tickets.columns)

tickets = tickets.reset_index(drop=True)
tickets["text"] = tickets["text"].fillna("").astype(str)

print("=" * 60)
print("PART C — SUPPORT TICKET TRIAGE")
print("=" * 60)

print("Tickets loaded:", len(tickets))
print("Columns:", tickets.columns.tolist())

assert len(tickets) == 164


# ============================================================
# 2. INPUT GUARDRAILS AND PII REDACTION
# ============================================================

print("\n" + "=" * 60)
print("TASK 1 & 2 — GUARDRAILS AND PII REDACTION")
print("=" * 60)

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")


def redact_pii(text):
    """Redact PII in the exact required order."""

    redacted, email_count = EMAIL_PATTERN.subn(
        "[EMAIL]", text
    )

    redacted, phone_count = PHONE_PATTERN.subn(
        "[PHONE]", redacted
    )

    redacted, order_count = ORDER_PATTERN.subn(
        "[ORDER_ID]", redacted
    )

    return redacted, email_count, phone_count, order_count


redaction_counts = {
    "email": 0,
    "phone": 0,
    "order_id": 0
}

redacted_texts = []

for text in tickets["text"]:

    redacted, emails, phones, orders_count = redact_pii(text)

    redacted_texts.append(redacted)

    redaction_counts["email"] += emails
    redaction_counts["phone"] += phones
    redaction_counts["order_id"] += orders_count

redaction_counts["total"] = sum(redaction_counts.values())

print("\nPII REDACTION SUMMARY")
print(redaction_counts)

assert redaction_counts["email"] == 55
assert redaction_counts["phone"] == 91
assert redaction_counts["order_id"] == 132
assert redaction_counts["total"] == 278

# Verify no PII remains after redaction
remaining_email = sum(
    len(EMAIL_PATTERN.findall(t)) for t in redacted_texts
)
remaining_phone = sum(
    len(PHONE_PATTERN.findall(t)) for t in redacted_texts
)
remaining_order = sum(
    len(ORDER_PATTERN.findall(t)) for t in redacted_texts
)

print("\nREMAINING PII")
print("Emails:", remaining_email)
print("Phones:", remaining_phone)
print("Order IDs:", remaining_order)

assert remaining_email == 0
assert remaining_phone == 0
assert remaining_order == 0

tickets["redacted_text"] = redacted_texts

# Input guardrail counts
empty_count = sum(
    not text.strip() for text in tickets["text"]
)

long_count = sum(
    bool(text.strip())
    and mock_llm.est_tokens(text) > MAX_INPUT_TOKENS
    for text in tickets["text"]
)

print("\nINPUT GUARDRAILS")
print("Empty tickets:", empty_count)
print("Long tickets:", long_count)

assert empty_count == 4
assert long_count == 4

print("TASK 1 AND 2 CHECKPOINTS PASSED")


# ============================================================
# 3. ZERO-SHOT AND FEW-SHOT PROMPTS
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 — PROMPT DESIGN")
print("=" * 60)

ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
- billing
- delivery
- technical
- fraud

Select the most appropriate category based on the ticket.

Return only valid JSON in this exact format:
{"category": "...", "reason": "..."}

Do not include markdown or any additional text.
""".strip()


FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
- billing
- delivery
- technical
- fraud

Use the following examples as guidance.

EXAMPLES:

Example 1:
Ticket: I was charged twice for the same order.
Output: {"category": "billing", "reason": "Duplicate payment"}

Example 2:
Ticket: Someone accessed my account and placed an unauthorized order.
Output: {"category": "fraud", "reason": "Unauthorized account activity"}

Example 3:
Ticket: My application crashes whenever I log in.
Output: {"category": "technical", "reason": "Application login issue"}

Return only valid JSON in this exact format:
{"category": "...", "reason": "..."}

Do not include markdown or any additional text.
""".strip()

assert "CLASSIFY" in ZERO_SHOT_PROMPT
assert "CLASSIFY" in FEW_SHOT_PROMPT
assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
assert "EXAMPLES:" in FEW_SHOT_PROMPT

for category in ALLOWED_CATEGORIES:
    assert category in ZERO_SHOT_PROMPT
    assert category in FEW_SHOT_PROMPT

print("Zero-shot prompt validated.")
print("Few-shot prompt validated.")
print("TASK 3 CHECKPOINT PASSED")


# ============================================================
# 4. RESPONSE VALIDATION
# ============================================================

def validate_response(response_text):
    """
    Valid JSON with allowed category -> classified_json
    Invalid JSON with exactly one category word -> classified_keyword
    Otherwise -> flagged_invalid_output
    """

    try:
        parsed = json.loads(response_text)

        category = parsed.get("category")

        if (
            isinstance(category, str)
            and category.lower() in ALLOWED_CATEGORIES
        ):
            return {
                "status": "classified_json",
                "category": category.lower(),
                "reason": parsed.get("reason", "")
            }

        # Valid JSON with invalid category is rejected.
        return {
            "status": "flagged_invalid_output",
            "category": None,
            "reason": None
        }

    except (json.JSONDecodeError, TypeError):
        pass

    words = re.findall(r"[a-z]+", str(response_text).lower())

    matched = [
        word for word in words
        if word in ALLOWED_CATEGORIES
    ]

    if len(matched) == 1:
        return {
            "status": "classified_keyword",
            "category": matched[0],
            "reason": "Extracted using keyword fallback"
        }

    return {
        "status": "flagged_invalid_output",
        "category": None,
        "reason": None
    }


# ============================================================
# 5. RESILIENT MODEL CALL
# ============================================================

def call_with_retries(
    messages,
    ticket_id,
    running_tokens,
    token_budget,
    stats
):
    """Call mock LLM with maximum three retries."""

    for attempt in range(MAX_RETRIES + 1):

        # Check budget before each actual model call.
        if running_tokens >= token_budget:
            return None, "budget_exhausted"

        try:
            response = mock_llm.chat(messages)
            return response, "success"

        except mock_llm.TransientLLMError:

            stats["transient_errors"] += 1

            if attempt >= MAX_RETRIES:
                return None, "permanent_error"

            wait = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            stats["retries"] += 1

            print(
                f"Retry ticket {ticket_id} | "
                f"retry {attempt} | wait {wait:.5f} seconds"
            )

            time.sleep(wait)

    return None, "permanent_error"


# ============================================================
# 6. FULL TRIAGE PIPELINE
# ============================================================

def run_pipeline(system_prompt, run_name, token_budget=100000):

    print("\n" + "=" * 60)
    print("STARTING:", run_name)
    print("=" * 60)

    # Reset immediately before each independent full run.
    mock_llm.reset()

    results = []
    running_tokens = 0

    stats = {
        "transient_errors": 0,
        "retries": 0
    }

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]

        original_text = ticket["text"]
        redacted_text = ticket["redacted_text"]

        # ----------------------------------------------------
        # A. EMPTY INPUT
        # ----------------------------------------------------

        if not original_text.strip():

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "skipped_empty",
                "category": None,
                "reason": None,
                "tokens_used": 0
            })
            continue

        # ----------------------------------------------------
        # B. LONG INPUT
        # ----------------------------------------------------

        if mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "flagged_long",
                "category": None,
                "reason": None,
                "tokens_used": 0
            })
            continue

        # ----------------------------------------------------
        # C. PRIVACY ASSERTIONS
        # ----------------------------------------------------

        assert EMAIL_PATTERN.search(redacted_text) is None
        assert PHONE_PATTERN.search(redacted_text) is None
        assert ORDER_PATTERN.search(redacted_text) is None

        # User content must be exactly the redacted ticket text.
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": redacted_text
            }
        ]

        assert messages[-1]["content"] == redacted_text

        # ----------------------------------------------------
        # D. MODEL CALL WITH RETRIES
        # ----------------------------------------------------

        response, call_status = call_with_retries(
            messages=messages,
            ticket_id=ticket_id,
            running_tokens=running_tokens,
            token_budget=token_budget,
            stats=stats
        )

        if call_status == "budget_exhausted":

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "budget_exhausted",
                "category": None,
                "reason": None,
                "tokens_used": 0
            })
            continue

        if call_status == "permanent_error":

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": "permanent_error",
                "category": None,
                "reason": None,
                "tokens_used": 0
            })
            continue

        # ----------------------------------------------------
        # E. TOKEN ACCOUNTING
        # ----------------------------------------------------

        response_text = response.choices[0].message.content

        tokens_used = response.usage.total_tokens
        running_tokens += tokens_used

        # ----------------------------------------------------
        # F. RESPONSE VALIDATION
        # ----------------------------------------------------

        validation = validate_response(response_text)

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": validation["status"],
            "category": validation["category"],
            "reason": validation["reason"],
            "tokens_used": tokens_used
        })

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # G. RUN METRICS
    # --------------------------------------------------------

    status_counts = result_df["status"].value_counts()

    classified_df = result_df[
        result_df["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]

    classified_count = len(classified_df)

    actual_categories = (
        tickets.set_index("ticket_id")["true_category"]
        .astype(str).str.lower().str.strip()
    )

    if classified_count:
        actual = classified_df["ticket_id"].map(actual_categories)
        accuracy = (
            actual.values == classified_df["category"].values
        ).mean()
    else:
        accuracy = 0.0

    total_tokens = int(result_df["tokens_used"].sum())

    tokens_per_ticket = (
        total_tokens / classified_count
        if classified_count else 0
    )

    print("\nSTATUS COUNTS")
    print(status_counts)

    print("\nRUN METRICS")
    print("Total tickets:", len(result_df))
    print("Classified tickets:", classified_count)
    print(f"Accuracy: {accuracy:.4f}")
    print("Total tokens:", total_tokens)
    print("Tokens per classified ticket:", round(tokens_per_ticket, 2))

    print("\nRETRY SUMMARY")
    print("Transient errors:", stats["transient_errors"])
    print("Successful retries:", stats["retries"])

    summary = {
        "run_name": run_name,
        "classified": classified_count,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": tokens_per_ticket,
        "status_counts": status_counts,
        "retry_stats": stats
    }

    return result_df, summary


# ============================================================
# 7. ZERO-SHOT AND FEW-SHOT COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("TASK 6 — PROMPT COMPARISON")
print("=" * 60)

# Zero-shot full run
zero_results, zero_summary = run_pipeline(
    ZERO_SHOT_PROMPT,
    "ZERO_SHOT",
    token_budget=float("inf")
)

# Few-shot full run
few_results, few_summary = run_pipeline(
    FEW_SHOT_PROMPT,
    "FEW_SHOT",
    token_budget=float("inf")
)


# Zero-shot confusion table
actual_categories = (
    tickets.set_index("ticket_id")["true_category"]
    .astype(str).str.lower().str.strip()
)

zero_classified = zero_results[
    zero_results["status"].isin([
        "classified_json", "classified_keyword"
    ])
].copy()

zero_classified["true_category"] = (
    zero_classified["ticket_id"].map(actual_categories)
)

confusion_table = pd.crosstab(
    zero_classified["true_category"],
    zero_classified["category"],
    rownames=["Actual"],
    colnames=["Predicted"]
).reindex(
    index=sorted(ALLOWED_CATEGORIES),
    columns=sorted(ALLOWED_CATEGORIES),
    fill_value=0
)

print("\nZERO-SHOT CONFUSION TABLE")
print(confusion_table)


comparison = pd.DataFrame([
    {
        "prompt": "Zero-shot",
        "classified": zero_summary["classified"],
        "accuracy": zero_summary["accuracy"],
        "total_tokens": zero_summary["total_tokens"],
        "tokens_per_ticket": (
            zero_summary["tokens_per_classified_ticket"]
        )
    },
    {
        "prompt": "Few-shot",
        "classified": few_summary["classified"],
        "accuracy": few_summary["accuracy"],
        "total_tokens": few_summary["total_tokens"],
        "tokens_per_ticket": (
            few_summary["tokens_per_classified_ticket"]
        )
    }
])

print("\nZERO-SHOT VS FEW-SHOT")
print(comparison.to_string(index=False))


# Reference checks
for summary in [zero_summary, few_summary]:

    assert summary["classified"] == 147

    assert summary["status_counts"].get(
        "classified_json", 0
    ) == 135

    assert summary["status_counts"].get(
        "classified_keyword", 0
    ) == 12

    assert summary["status_counts"].get(
        "flagged_invalid_output", 0
    ) == 9

    assert summary["retry_stats"]["transient_errors"] == 19
    assert summary["retry_stats"]["retries"] == 19


print("\nTASK 6 CHECKPOINT PASSED")


# ============================================================
# 8. TOKEN BUDGET DEMONSTRATION
# ============================================================

print("\n" + "=" * 60)
print("TASK 7A — TOKEN BUDGET DEMONSTRATION")
print("=" * 60)

demo_budget = TOKEN_BUDGET

budget_results, budget_summary = run_pipeline(
    ZERO_SHOT_PROMPT,
    "ZERO_SHOT_BUDGET_DEMO",
    token_budget=demo_budget
)

# Lower budget automatically if no ticket was exhausted.
while (
    budget_summary["status_counts"].get(
        "budget_exhausted", 0
    ) == 0
    and demo_budget > 1000
):

    demo_budget -= 1000

    print("\nTrying lower budget:", demo_budget)

    budget_results, budget_summary = run_pipeline(
        ZERO_SHOT_PROMPT,
        "ZERO_SHOT_BUDGET_DEMO",
        token_budget=demo_budget
    )

budget_exhausted_count = budget_summary[
    "status_counts"
].get("budget_exhausted", 0)

assert budget_exhausted_count > 0

budget_results.drop(
    columns=["reason"]
).to_csv(
    OUTPUT_DIR / "triage_budget_demo.csv",
    index=False
)

print("\nBudget used:", demo_budget)
print("Budget-exhausted tickets:", budget_exhausted_count)
print("Saved outputs/triage_budget_demo.csv")


# ============================================================
# 9. FINAL FEW-SHOT RUN
# ============================================================

print("\n" + "=" * 60)
print("TASK 7B — FINAL FEW-SHOT RUN")
print("=" * 60)

final_results, final_summary = run_pipeline(
    FEW_SHOT_PROMPT,
    "FINAL_FEW_SHOT",
    token_budget=FINAL_TOKEN_BUDGET
)

assert len(final_results) == 164


# ============================================================
# 10. LOAD CUSTOMER SEGMENTS
# ============================================================

segments = pd.read_csv(SEGMENT_FILE)

required_segment_columns = {
    "customer_id", "segment_name", "is_premium"
}

assert required_segment_columns.issubset(segments.columns)

final_results["customer_id"] = (
    final_results["customer_id"].astype(str)
)

segments["customer_id"] = (
    segments["customer_id"].astype(str)
)

segments = segments[
    ["customer_id", "segment_name", "is_premium"]
].drop_duplicates(subset=["customer_id"])

final_results = final_results.merge(
    segments,
    on="customer_id",
    how="left"
)

final_results["is_premium"] = (
    final_results["is_premium"]
    .fillna(False)
    .astype(bool)
)


# ============================================================
# 11. BUSINESS ROUTING
# ============================================================

def assign_routing(row):

    status = row["status"]
    category = row["category"]
    is_premium = row["is_premium"]

    # Human review takes precedence
    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
        "permanent_error"
    }:
        return pd.Series({
            "route": "human_review",
            "priority": "NORMAL"
        })

    # Fraud takes precedence over Premium
    if category == "fraud":
        return pd.Series({
            "route": "security_team",
            "priority": "HIGH"
        })

    if is_premium:
        return pd.Series({
            "route": "premium_desk",
            "priority": "HIGH"
        })

    return pd.Series({
        "route": "support_queue",
        "priority": "NORMAL"
    })


routing = final_results.apply(assign_routing, axis=1)

final_results[["route", "priority"]] = routing


# ============================================================
# 12. SAVE FINAL OUTPUT
# ============================================================

final_output = final_results[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
].copy()

final_output["category"] = (
    final_output["category"].fillna("")
)

final_output["segment_name"] = (
    final_output["segment_name"].fillna("")
)

final_output.to_csv(
    OUTPUT_DIR / "triage_results.csv",
    index=False
)


# ============================================================
# 13. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TRIAGE RESULTS")
print("=" * 60)

print("\nFinal output shape:", final_output.shape)

print("\nStatus distribution:")
print(final_output["status"].value_counts())

print("\nRouting distribution:")
print(final_output["route"].value_counts())

print("\nPriority distribution:")
print(final_output["priority"].value_counts())

print("\nFinal few-shot accuracy:")
print(f"{final_summary['accuracy'] * 100:.2f}%")

print("\nFinal total tokens:")
print(final_summary["total_tokens"])

print("\nFinal tokens per classified ticket:")
print(
    round(
        final_summary["tokens_per_classified_ticket"],
        2
    )
)

# Exactly 164 rows
assert len(final_output) == 164

# All output columns must be present and non-null
assert not final_output.isnull().any().any()

# Routing validation
for _, row in final_output.iterrows():

    if row["status"] in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
        "permanent_error"
    }:
        assert row["route"] == "human_review"

    elif row["category"] == "fraud":
        assert row["route"] == "security_team"
        assert row["priority"] == "HIGH"

    elif final_results.loc[
        final_results["ticket_id"] == row["ticket_id"],
        "is_premium"
    ].iloc[0]:
        assert row["route"] == "premium_desk"
        assert row["priority"] == "HIGH"

    else:
        assert row["route"] == "support_queue"
        assert row["priority"] == "NORMAL"


print("\nSaved:")
print(OUTPUT_DIR / "triage_results.csv")

print("\nALL PART C CHECKPOINTS PASSED")
