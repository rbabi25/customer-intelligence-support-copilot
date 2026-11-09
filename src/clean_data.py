
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
# LOAD CUSTOMERS
# ============================================================

customers = pd.read_csv(
    RAW_DATA_DIR / "customers.csv",
    keep_default_na=False
)

print("Raw customers shape:", customers.shape)


# ============================================================
# 1. STANDARDISE CITY
# ============================================================

customers["city"] = (
    customers["city"]
    .str.strip()
    .str.title()
)

print("\nCity counts after standardisation:")

print(
    customers["city"]
    .value_counts()
)


# ============================================================
# 2. STANDARDISE GENDER
# ============================================================

customers["gender"] = (
    customers["gender"]
    .str.strip()
    .str.lower()
)

# Inspect actual raw spellings before mapping
print("\nRaw gender spellings:")

print(
    customers["gender"]
    .value_counts()
)

# Mapping
gender_mapping = {

    "m": "male",
    "male": "male",
    "man": "male",

    "f": "female",
    "female": "female",
    "woman": "female",

    "": "unknown",
    "unknown": "unknown",
    "other": "unknown",
    "prefer not to say": "unknown"

}

customers["gender"] = (
    customers["gender"]
    .map(gender_mapping)
    .fillna("unknown")
)


# ============================================================
# 3. CLEAN AGE
# ============================================================

customers["age"] = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

# Invalid ages become missing
customers.loc[
    ~customers["age"].between(10, 100),
    "age"
] = np.nan

# Calculate city median using transform
city_median_age = (
    customers
    .groupby("city")["age"]
    .transform("median")
)

# Fill all missing ages
customers["age"] = (
    customers["age"]
    .fillna(city_median_age)
)

# Store age as integer
customers["age"] = customers["age"].astype(int)


# ============================================================
# 4. CONVERT SIGNUP DATE
# ============================================================

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)


# ============================================================
# 5. VALIDATION
# ============================================================

expected_city_counts = {

    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184

}

expected_gender_counts = {

    "female": 610,
    "male": 560,
    "unknown": 30

}

assert customers["city"].nunique() == 6

assert customers["city"].value_counts().to_dict() == (
    expected_city_counts
)

assert customers["gender"].value_counts().to_dict() == (
    expected_gender_counts
)

assert customers["age"].isna().sum() == 0

assert customers["age"].between(18, 60).all()

assert pd.api.types.is_datetime64_any_dtype(
    customers["signup_date"]
)

print("\nALL CUSTOMER CHECKPOINTS PASSED")


# ============================================================
# 6. SAVE CLEAN CUSTOMERS
# ============================================================

customers["signup_date"] = (
    customers["signup_date"]
    .dt.strftime("%Y-%m-%d %H:%M:%S")
)

customers.to_csv(
    CLEAN_DATA_DIR / "customers.csv",
    index=False
)

print("\nClean customers saved successfully.")

print("Clean customers shape:", customers.shape)
