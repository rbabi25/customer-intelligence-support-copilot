import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

products = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "products.csv",
    keep_default_na=False
)

products.to_csv(
    PROJECT_ROOT / "data" / "clean" / "products.csv",
    index=False
)

print("Clean products saved successfully.")
print("Products shape:", products.shape)