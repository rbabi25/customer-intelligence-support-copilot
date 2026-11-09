
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DATA_DIR = PROJECT_ROOT / "data" / "clean"

CLEAN_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

orders = pd.read_csv(
    RAW_DATA_DIR / "orders.csv",
    keep_default_na=False
)

products = pd.read_csv(
    RAW_DATA_DIR / "products.csv",
    keep_default_na=False
)

print("Raw orders shape:", orders.shape)


# ============================================================
# STEP 1: CONVERT ORDER TIMESTAMP
# ============================================================

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
)

print("\nStep 1: Converted order_ts to datetime")


# ============================================================
# STEP 2: REMOVE EXACT DUPLICATES
# ============================================================

before = len(orders)

orders = orders.drop_duplicates()

after = len(orders)

print("\nStep 2: Removed exact duplicates")

print("Rows before:", before)
print("Rows after:", after)
print("Duplicates removed:", before - after)

assert before - after == 119


# ============================================================
# STEP 3: KEEP LATEST RECORD PER ORDER_ID
# ============================================================

before = len(orders)

orders = (
    orders
    .sort_values("order_ts", kind="mergesort")
    .drop_duplicates(
        subset=["order_id"],
        keep="last"
    )
    .copy()
)

print("\nStep 3: Kept latest record per order_id")

print("Rows before:", before)
print("Rows after:", len(orders))

assert len(orders) == 12309

assert orders["order_id"].nunique() == 12309


# ============================================================
# STEP 4: CLEAN ORDER AMOUNT
# ============================================================

# Convert order_amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

# Convert quantity and discount to numeric
orders["quantity"] = pd.to_numeric(
    orders["quantity"],
    errors="raise"
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"],
    errors="raise"
)

# Merge product price
orders = orders.merge(
    products[["product_id", "price_inr"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Convert product price to numeric
orders["price_inr"] = pd.to_numeric(
    orders["price_inr"],
    errors="raise"
)

# Calculate replacement amount
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

# Fill missing order amounts
missing_amounts = orders["order_amount"].isna().sum()

orders["order_amount"] = (
    orders["order_amount"]
    .fillna(calculated_amount)
)

print("\nStep 4: Cleaned order_amount")

print("Missing amounts before filling:", missing_amounts)
print("Missing amounts after filling:", orders["order_amount"].isna().sum())

# Remove temporary product price column
orders = orders.drop(columns=["price_inr"])


# ============================================================
# STEP 5: CLEAN DELIVERY MINUTES
# ============================================================

# Convert to numeric
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

# Replace negative values with NaN
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = np.nan

print("\nStep 5: Replaced negative delivery times with NaN")


# ============================================================
# STEP 6: IMPUTE COMPLETED ORDERS
# USING PRODUCT CATEGORY MEDIAN
# ============================================================

# Merge product category
orders = orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Select completed orders
completed_mask = orders["status"].eq("completed")

# Calculate category median using completed orders only
category_median = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

# Fill missing delivery times only for completed orders
orders.loc[
    completed_mask,
    "delivery_minutes"
] = (
    orders.loc[
        completed_mask,
        "delivery_minutes"
    ].fillna(category_median)
)

print("\nStep 6: Imputed completed order delivery times")


# ============================================================
# STEP 7: CANCELLED ORDERS
# ============================================================

cancelled_mask = orders["status"].eq("cancelled")

# Explicitly ensure cancelled orders remain missing
orders.loc[
    cancelled_mask,
    "delivery_minutes"
] = np.nan

print("\nStep 7: Cancelled order delivery times left missing")


# ============================================================
# REMOVE TEMPORARY CATEGORY COLUMN
# ============================================================

orders = orders.drop(columns=["category"])


# ============================================================
# STEP 8: VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("ORDER CLEANING VALIDATION")
print("=" * 65)

status_counts = orders["status"].value_counts()

print("\nStatus distribution:")
print(status_counts)

completed_orders = orders[
    orders["status"] == "completed"
]

cancelled_orders = orders[
    orders["status"] == "cancelled"
]

completed_revenue = round(
    completed_orders["order_amount"].sum(),
    2
)

print("\nCompleted revenue:", completed_revenue)

print(
    "Missing order amounts:",
    orders["order_amount"].isna().sum()
)

print(
    "Missing completed delivery times:",
    completed_orders["delivery_minutes"].isna().sum()
)

print(
    "Missing cancelled delivery times:",
    cancelled_orders["delivery_minutes"].isna().sum()
)


# ============================================================
# ASSERTIONS
# ============================================================

assert len(orders) == 12309

assert orders["order_id"].nunique() == 12309

assert status_counts.get("completed", 0) == 11110

assert status_counts.get("cancelled", 0) == 1199

assert status_counts.get("pending", 0) == 0

assert orders["order_amount"].isna().sum() == 0

assert completed_orders["delivery_minutes"].isna().sum() == 0

assert cancelled_orders["delivery_minutes"].isna().sum() == 1199

assert completed_revenue == 2998668.80

assert not orders["delivery_minutes"].dropna().lt(0).any()

print("\nALL ORDER CHECKPOINTS PASSED")


# ============================================================
# STEP 9: FORMAT TIMESTAMP FOR CSV AND SQLITE
# ============================================================

orders["order_ts"] = (
    orders["order_ts"]
    .dt.strftime("%Y-%m-%d %H:%M:%S")
)


# ============================================================
# STEP 10: SAVE CLEAN ORDERS
# ============================================================

orders.to_csv(
    CLEAN_DATA_DIR / "orders.csv",
    index=False
)

print("\nClean orders saved successfully.")

print("Final orders shape:", orders.shape)
