
import pandas as pd
import sqlite3
import re
from pathlib import Path

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SQL_FILE = PROJECT_ROOT / "sql" / "queries.sql"
DB_PATH = PROJECT_ROOT / "data" / "instamart.db"


# --------------------------------------------------
# 2. Read SQL queries from queries.sql
# --------------------------------------------------

sql_text = SQL_FILE.read_text(encoding="utf-8")

# Split the SQL file using markers such as:
# -- Q1: Customer spend
# -- Q2: Revenue by city and category

query_list = re.findall(
    r"(?ms)^\s*--\s*Q\d+.*?\n(.*?)(?=^\s*--\s*Q\d+|\Z)",
    sql_text
)

# Remove empty query sections
query_list = [
    query.strip()
    for query in query_list
    if query.strip()
]

print("Number of queries found:", len(query_list))

# Validate query count
assert len(query_list) == 7, (
    f"Expected 7 SQL queries, but found {len(query_list)}"
)


# --------------------------------------------------
# 3. Connect to SQLite database
# --------------------------------------------------

conn = sqlite3.connect(DB_PATH)


# --------------------------------------------------
# 4. Execute queries using pandas
# --------------------------------------------------

try:

    for i, query in enumerate(query_list, start=1):

        print("\n" + "=" * 60)
        print(f"QUERY {i}")
        print("=" * 60)

        # Execute SQL and return result as DataFrame
        result = pd.read_sql_query(query, conn)

        # Display result
        print(result.to_string(index=False))

        # Display number of rows returned
        print(f"\nRows returned: {len(result)}")

finally:

    # Close database connection
    conn.close()


# --------------------------------------------------
# 5. Completion message
# --------------------------------------------------

print("\nAll seven SQL queries executed successfully.")
