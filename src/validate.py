import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

expected_counts = {
    "customers": 1200,
    "products": 40,
    "orders": 12309
}

for table, expected in expected_counts.items():
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    actual = cursor.fetchone()[0]

    print(f"{table}: {actual} rows")
    assert actual == expected, f"{table} row count mismatch"

# Verify order timestamp format
cursor.execute("""
    SELECT order_ts
    FROM orders
    LIMIT 5
""")

print("\nSample order timestamps:")
for row in cursor.fetchall():
    print(row[0])

# Verify SQLite tables
cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
""")

print("\nTables in database:")
print([row[0] for row in cursor.fetchall()])

conn.close()

print("\nALL DATABASE CHECKPOINTS PASSED")