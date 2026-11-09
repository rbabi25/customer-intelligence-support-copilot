
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

AUDIT_OUTPUT_DIR = PROJECT_ROOT / "data" / "audit"

AUDIT_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RAW DATA
# ============================================================

customers = pd.read_csv(
    RAW_DATA_DIR / "customers.csv",
    keep_default_na=False
)

products = pd.read_csv(
    RAW_DATA_DIR / "products.csv",
    keep_default_na=False
)

orders = pd.read_csv(
    RAW_DATA_DIR / "orders.csv",
    keep_default_na=False
)


# ============================================================
# GENERAL AUDIT FUNCTION
# ============================================================

def audit_table(df, table_name):

    print("\n" + "=" * 80)
    print(f"GENERAL AUDIT: {table_name.upper()}")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Shape
    # --------------------------------------------------------

    print("\nShape:")

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # 2. Data types
    # --------------------------------------------------------

    print("\nData Types:")

    print(df.dtypes.to_string())

    # --------------------------------------------------------
    # 3. Missing values
    # --------------------------------------------------------

    missing_count = df.isna().sum()

    missing_percentage = (
        missing_count / len(df) * 100
    ).round(2)

    missing_report = pd.DataFrame({

        "column": df.columns,

        "data_type": [
            str(dtype)
            for dtype in df.dtypes
        ],

        "missing_count": missing_count.values,

        "missing_percentage": missing_percentage.values

    })

    print("\nMissing Values Report:")

    print(
        missing_report.to_string(
            index=False
        )
    )

    # Save the report
    missing_report.to_csv(
        AUDIT_OUTPUT_DIR / f"{table_name}_missing_report.csv",
        index=False
    )

    return missing_report


# ============================================================
# AUDIT ALL THREE TABLES
# ============================================================

customers_missing_report = audit_table(
    customers,
    "customers"
)

products_missing_report = audit_table(
    products,
    "products"
)

orders_missing_report = audit_table(
    orders,
    "orders"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def count_blank_values(series):

    """
    Counts empty strings and whitespace-only strings.

    Does not count actual NaN values.
    """

    return (
        series.astype(str)
        .str.strip()
        .eq("")
        .sum()
    )


def count_non_numeric_values(series):

    """
    Counts values that cannot be converted to numeric.

    Blank strings are treated as non-numeric.
    """

    numeric_values = pd.to_numeric(
        series,
        errors="coerce"
    )

    return numeric_values.isna().sum()


# ============================================================
# ORDERS AUDIT
# ============================================================

print("\n" + "=" * 80)
print("SPECIFIC AUDIT: ORDERS")
print("=" * 80)


# ------------------------------------------------------------
# 1. Raw order rows
# ------------------------------------------------------------

raw_order_rows = len(orders)


# ------------------------------------------------------------
# 2. Exact duplicate rows
# ------------------------------------------------------------

exact_duplicate_rows = orders.duplicated().sum()


# ------------------------------------------------------------
# 3. Pending rows
# ------------------------------------------------------------

pending_rows = (
    orders["status"]
    .str.strip()
    .str.lower()
    .eq("pending")
    .sum()
)


# ------------------------------------------------------------
# 4. Distinct order IDs
# ------------------------------------------------------------

distinct_order_ids = orders["order_id"].nunique()


# ------------------------------------------------------------
# 5. Non-numeric order amounts
# ------------------------------------------------------------

non_numeric_amounts = count_non_numeric_values(
    orders["order_amount"]
)


# ------------------------------------------------------------
# 6. Negative delivery minutes
# ------------------------------------------------------------

delivery_minutes_numeric = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

negative_delivery_minutes = (
    delivery_minutes_numeric.lt(0)
    .sum()
)


# ------------------------------------------------------------
# 7. Blank delivery minutes
# ------------------------------------------------------------

blank_delivery_minutes = count_blank_values(
    orders["delivery_minutes"]
)


# ============================================================
# CUSTOMERS AUDIT
# ============================================================

print("\n" + "=" * 80)
print("SPECIFIC AUDIT: CUSTOMERS")
print("=" * 80)


# ------------------------------------------------------------
# 1. Blank ages
# ------------------------------------------------------------

blank_ages = count_blank_values(
    customers["age"]
)


# ------------------------------------------------------------
# 2. Ages outside 10-100
# ------------------------------------------------------------

age_numeric = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

age_outside_range = (
    age_numeric.lt(10)
    |
    age_numeric.gt(100)
).sum()


# ------------------------------------------------------------
# 3. Distinct raw city spellings
# ------------------------------------------------------------

distinct_raw_cities = customers["city"].nunique()


# ------------------------------------------------------------
# 4. Distinct non-blank raw gender spellings
# ------------------------------------------------------------

non_blank_gender = customers.loc[
    ~customers["gender"]
    .astype(str)
    .str.strip()
    .eq(""),
    "gender"
]

distinct_raw_genders = non_blank_gender.nunique()


# ------------------------------------------------------------
# 5. Blank genders
# ------------------------------------------------------------

blank_genders = count_blank_values(
    customers["gender"]
)


# ============================================================
# COMBINE SPECIFIC AUDIT RESULTS
# ============================================================

audit_results = {

    "Raw orders rows":
        raw_order_rows,

    "Exact duplicate order rows":
        exact_duplicate_rows,

    "Pending order rows":
        pending_rows,

    "Distinct order_id":
        distinct_order_ids,

    "Non-numeric order_amount":
        non_numeric_amounts,

    "Negative delivery_minutes":
        negative_delivery_minutes,

    "Blank delivery_minutes":
        blank_delivery_minutes,

    "Blank customer age":
        blank_ages,

    "Age outside 10-100":
        age_outside_range,

    "Distinct raw city spellings":
        distinct_raw_cities,

    "Distinct non-blank raw gender spellings":
        distinct_raw_genders,

    "Blank gender":
        blank_genders

}


# ============================================================
# DISPLAY AUDIT RESULTS
# ============================================================

audit_summary = pd.DataFrame(
    list(audit_results.items()),
    columns=[
        "Audit Metric",
        "Value"
    ]
)

print("\n" + "=" * 80)
print("SPECIFIC AUDIT SUMMARY")
print("=" * 80)

print(
    audit_summary.to_string(
        index=False
    )
)


# ============================================================
# REFERENCE CHECKPOINTS
# ============================================================

reference_values = {

    "Raw orders rows": 13084,

    "Exact duplicate order rows": 119,

    "Pending order rows": 656,

    "Distinct order_id": 12309,

    "Non-numeric order_amount": 120,

    "Negative delivery_minutes": 106,

    "Blank delivery_minutes": 2541,

    "Blank customer age": 61,

    "Age outside 10-100": 6,

    "Distinct raw city spellings": 24,

    "Distinct non-blank raw gender spellings": 10,

    "Blank gender": 30

}


# ============================================================
# VALIDATE AUDIT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("REFERENCE CHECKPOINT VALIDATION")
print("=" * 80)


validation_results = []

for metric, expected_value in reference_values.items():

    actual_value = audit_results[metric]

    passed = actual_value == expected_value

    validation_results.append({

        "Metric": metric,

        "Actual": actual_value,

        "Expected": expected_value,

        "Status": "PASS" if passed else "FAIL"

    })


validation_df = pd.DataFrame(
    validation_results
)

print(
    validation_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE AUDIT OUTPUTS
# ============================================================

audit_summary.to_csv(
    AUDIT_OUTPUT_DIR / "specific_audit_summary.csv",
    index=False
)

validation_df.to_csv(
    AUDIT_OUTPUT_DIR / "audit_validation.csv",
    index=False
)


# ============================================================
# FINAL RESULT
# ============================================================

if validation_df["Status"].eq("PASS").all():

    print("\nALL RAW AUDIT CHECKPOINTS PASSED")

else:

    failed_metrics = validation_df.loc[
        validation_df["Status"] == "FAIL",
        "Metric"
    ].tolist()

    print("\nRAW AUDIT CHECKPOINTS FAILED")

    print("Failed metrics:")

    for metric in failed_metrics:

        print(f"- {metric}")
