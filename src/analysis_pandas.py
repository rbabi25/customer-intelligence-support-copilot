
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

# --------------------------------------------------
# 2. Load cleaned datasets
# --------------------------------------------------

customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

print("Customers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)

# --------------------------------------------------
# 3. Merge datasets
# --------------------------------------------------

df = orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

df = df.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# --------------------------------------------------
# 4. Filter completed orders
# --------------------------------------------------

completed = df[df["status"] == "completed"].copy()

print("\nCompleted orders:", completed.shape)

# --------------------------------------------------
# 5. Create revenue pivot table
# --------------------------------------------------

revenue_pivot = pd.pivot_table(
    completed,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Total"
)

print("\nREVENUE BY CITY AND CATEGORY")
print(revenue_pivot.round(2))

# --------------------------------------------------
# 6. Validate total revenue
# --------------------------------------------------

total_revenue = completed["order_amount"].sum()

print("\nTotal completed revenue:", round(total_revenue, 2))

assert round(total_revenue, 2) == 2998668.80
assert round(revenue_pivot.loc["Total", "Total"], 2) == 2998668.80

print("\nPIVOT TABLE CHECKPOINT PASSED")

# --------------------------------------------------
# 7. City-wise business performance using groupby
# --------------------------------------------------

city_analysis = (
    completed
    .groupby("city")
    .agg(
        total_revenue=("order_amount", "sum"),
        average_order_value=("order_amount", "mean"),
        total_orders=("order_id", "count")
    )
    .reset_index()
    .sort_values("total_revenue", ascending=False)
)

print("\nCITY-WISE BUSINESS PERFORMANCE")
print(city_analysis.round(2).to_string(index=False))

# --------------------------------------------------
# 8. Validate city-wise results
# --------------------------------------------------

print("\nTotal revenue across cities:",
      round(city_analysis["total_revenue"].sum(), 2))

assert len(city_analysis) == 6
assert round(city_analysis["total_revenue"].sum(), 2) == 2998668.80

print("\nGROUPBY CHECKPOINT PASSED")
# --------------------------------------------------
# 9. Outlier detection using IQR
# --------------------------------------------------

# Consider completed orders only
amounts = completed["order_amount"]

# Calculate quartiles
Q1 = amounts.quantile(0.25)
Q3 = amounts.quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate outlier boundaries
lower_fence = Q1 - (1.5 * IQR)
upper_fence = Q3 + (1.5 * IQR)

# Identify outliers
outliers = completed[
    (completed["order_amount"] < lower_fence) |
    (completed["order_amount"] > upper_fence)
]

lower_outliers = completed[
    completed["order_amount"] < lower_fence
]

upper_outliers = completed[
    completed["order_amount"] > upper_fence
]

print("\nOUTLIER ANALYSIS")
print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Total outliers:", len(outliers))
print("Lower outliers:", len(lower_outliers))
print("Upper outliers:", len(upper_outliers))

# Validation
assert round(Q1, 2) == 64.80
assert round(Q3, 2) == 297.00
assert round(upper_fence, 2) == 645.30
assert len(upper_outliers) == 897
assert len(lower_outliers) == 0

print("\nIQR CHECKPOINT PASSED")
# --------------------------------------------------
# 10. Correlation analysis
# --------------------------------------------------

correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation_matrix = completed[
    correlation_columns
].corr()

print("\nCORRELATION MATRIX")
print(correlation_matrix.round(3))

# Identify the strongest correlation excluding self-correlations
correlation_pairs = (
    correlation_matrix
    .abs()
    .where(
        ~pd.DataFrame(
            [[i == j for j in range(len(correlation_columns))]
             for i in range(len(correlation_columns))],
            index=correlation_columns,
            columns=correlation_columns
        )
    )
    .stack()
    .sort_values(ascending=False)
)

print("\nSTRONGEST CORRELATION PAIRS")
print(correlation_pairs.head())

print("\nCORRELATION CHECKPOINT PASSED")