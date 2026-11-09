import pandas as pd
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"

conn = sqlite3.connect(DB_PATH)

tables = {
    "customers": "customers.csv",
    "products": "products.csv",
    "orders": "orders.csv"
}

for table_name, filename in tables.items():
    df = pd.read_csv(
        CLEAN_DIR / filename,
        keep_default_na=False
    )

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

    print(f"Loaded {table_name}: {len(df)} rows")

conn.close()

print("\nSQLite database created successfully.")