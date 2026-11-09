
import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.base import clone
from sklearn.utils import resample

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORDERS_PATH = PROJECT_ROOT / "data" / "clean" / "orders.csv"
CUSTOMERS_PATH = PROJECT_ROOT / "data" / "clean" / "customers.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RFM_OUTPUT = OUTPUT_DIR / "rfm.csv"
KMEANS_OUTPUT = OUTPUT_DIR / "kmeans_evaluation.csv"

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")


# ============================================================
# 2. LOAD CLEANED DATA
# ============================================================

orders = pd.read_csv(ORDERS_PATH)
customers = pd.read_csv(CUSTOMERS_PATH)

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

orders["status"] = orders["status"].str.strip().str.lower()

print("=" * 60)
print("INPUT DATA")
print("=" * 60)

print("Orders shape:", orders.shape)
print("Customers shape:", customers.shape)


# ============================================================
# 3. BUILD RFM TABLE
# ============================================================

# Only completed orders are considered for customer behaviour.
completed = orders[orders["status"] == "completed"].copy()

# Aggregate customer-level behaviour.
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    )
    .reset_index()
)

# Recency = whole days between snapshot and last completed order.
rfm["recency"] = (
    SNAPSHOT_DATE - rfm["last_order_ts"]
).dt.days

# Keep only the required RFM columns.
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

# Sort for easier inspection.
rfm = rfm.sort_values("customer_id").reset_index(drop=True)

print("\n" + "=" * 60)
print("RFM SUMMARY")
print("=" * 60)

print("RFM shape:", rfm.shape)
print("\nRFM summary:")
print(rfm[["recency", "frequency", "monetary"]].describe())


# ============================================================
# 4. RFM VALIDATION
# ============================================================

assert len(rfm) == 1175, "Unexpected RFM customer count"

assert rfm["recency"].median() == 19
assert rfm["recency"].min() == 0
assert rfm["recency"].max() == 180

assert round(rfm["frequency"].mean(), 2) == 9.46
assert rfm["frequency"].max() == 33

assert round(rfm["monetary"].mean(), 2) == 2552.06
assert round(rfm["monetary"].max(),2) == 14766.55

assert not rfm.isnull().any().any()

print("\nALL RFM CHECKPOINTS PASSED")

# Save RFM table.
rfm.to_csv(RFM_OUTPUT, index=False)

print("RFM saved to:", RFM_OUTPUT)


# ============================================================
# 5. STANDARDISE RFM FEATURES
# ============================================================

features = ["recency", "frequency", "monetary"]

X = rfm[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 60)
print("STANDARDISATION")
print("=" * 60)

print("Scaled RFM shape:", X_scaled.shape)


# ============================================================
# 6. EVALUATE K-MEANS FOR K = 2 TO 8
# ============================================================

results = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette": silhouette
    })


# ============================================================
# 7. DISPLAY AND SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("K-MEANS EVALUATION")
print("=" * 60)

print(results_df.round(3).to_string(index=False))

results_df.to_csv(KMEANS_OUTPUT, index=False)

print("\nK-means results saved to:", KMEANS_OUTPUT)

print("\nK-MEANS EVALUATION COMPLETED SUCCESSFULLY")


# ============================================================
# 8. FINAL K-MEANS MODEL (K=4)
# ============================================================

FINAL_K = 4

final_kmeans = KMeans(
    n_clusters=FINAL_K,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_kmeans.fit_predict(X_scaled)


# ============================================================
# 9. PROFILE THE SEGMENTS
# ============================================================

segment_profile = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    )
    .round(2)
)

print("\n" + "=" * 60)
print("CUSTOMER SEGMENT PROFILES")
print("=" * 60)

print(segment_profile.to_string())


# ============================================================
# 10. ASSIGN BUSINESS NAMES
# ============================================================

remaining = set(segment_profile.index)

# Highest monetary value = Premium
premium_id = segment_profile["mean_monetary"].idxmax()
remaining.remove(premium_id)

# Highest frequency among remaining = Loyal & Frequent
loyal_id = segment_profile.loc[
    list(remaining), "mean_frequency"
].idxmax()
remaining.remove(loyal_id)

# Highest recency among remaining = At-Risk
at_risk_id = segment_profile.loc[
    list(remaining), "mean_recency"
].idxmax()
remaining.remove(at_risk_id)

# Remaining cluster = Regular
regular_id = next(iter(remaining))

segment_names = {
    premium_id: "Premium",
    loyal_id: "Loyal & Frequent",
    at_risk_id: "At-Risk",
    regular_id: "Regular"
}

segment_actions = {
    "Premium": "Provide VIP offers, exclusive rewards and personalised recommendations.",
    "Loyal & Frequent": "Encourage repeat purchases through loyalty rewards and subscriptions.",
    "At-Risk": "Launch win-back campaigns with targeted discounts and reminders.",
    "Regular": "Use personalised recommendations and cross-selling to increase engagement."
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

rfm["is_premium"] = rfm["segment_id"] == premium_id

# Display segment names and recommended actions.
print("\n" + "=" * 60)
print("BUSINESS SEGMENT INTERPRETATION")
print("=" * 60)

for segment_id, name in segment_names.items():
    print(f"\nSegment ID: {segment_id}")
    print("Segment name:", name)
    print("Customers:", int(segment_profile.loc[segment_id, "customer_count"]))
    print("Mean recency:", segment_profile.loc[segment_id, "mean_recency"])
    print("Mean frequency:", segment_profile.loc[segment_id, "mean_frequency"])
    print("Mean monetary:", segment_profile.loc[segment_id, "mean_monetary"])
    print("Recommended action:", segment_actions[name])
    print("Premium:", name == "Premium")


# ============================================================
# 11. PCA FOR TWO-DIMENSIONAL PROJECTION
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\n" + "=" * 60)
print("PCA RESULTS")
print("=" * 60)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())


# ============================================================
# 12. SAVE CUSTOMER SEGMENTS
# ============================================================

output_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium"
]

customer_segments = rfm[output_columns].copy()

SEGMENTS_OUTPUT = OUTPUT_DIR / "customer_segments.csv"

customer_segments.to_csv(SEGMENTS_OUTPUT, index=False)

print("\n" + "=" * 60)
print("SEGMENT OUTPUT")
print("=" * 60)

print("Customer segments shape:", customer_segments.shape)
print("Premium customers:", int(customer_segments["is_premium"].sum()))
print("Output saved to:", SEGMENTS_OUTPUT)

assert len(customer_segments) == 1175
assert customer_segments["segment_name"].nunique() == 4
assert customer_segments["is_premium"].sum() == segment_profile.loc[
    premium_id, "customer_count"
]

print("\nALL TASK 3 CHECKPOINTS PASSED")


# ============================================================
# TASK 4: DEFINE THE LAPSE PREDICTION PROBLEM
# ============================================================

CUTOFF_DATE = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END_DATE = pd.Timestamp("2025-10-01 00:00:00")


# ============================================================
# 1. SEPARATE HISTORICAL ORDERS AND LABEL PERIOD
# ============================================================

# Historical data: strictly before September 1.
historical_orders = orders[
    orders["order_ts"] < CUTOFF_DATE
].copy()

# Completed historical orders.
historical_completed = historical_orders[
    historical_orders["status"] == "completed"
].copy()

# Customers who had at least one completed order before cutoff.
cohort_ids = historical_completed["customer_id"].unique()

print("\n" + "=" * 60)
print("LAPSE COHORT")
print("=" * 60)

print("Cohort customers:", len(cohort_ids))


# September orders are used ONLY to construct the target label.
september_completed = orders[
    (orders["order_ts"] >= CUTOFF_DATE)
    & (orders["order_ts"] < LABEL_END_DATE)
    & (orders["status"] == "completed")
].copy()

# Customers who placed at least one completed order in September.
september_active_ids = set(
    september_completed["customer_id"].unique()
)

# Lapsed = 1 if no completed order during September.
labels = pd.DataFrame({"customer_id": cohort_ids})

labels["lapsed"] = (
    ~labels["customer_id"].isin(september_active_ids)
).astype(int)

print("Lapsed customers:", labels["lapsed"].sum())
print("Not lapsed customers:", (labels["lapsed"] == 0).sum())

print(
    "Lapse percentage:",
    round(labels["lapsed"].mean() * 100, 2), "%"
)


# ============================================================
# 2. BUILD HISTORICAL COMPLETED-ORDER FEATURES
# ============================================================

completed_features = (
    historical_completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean")
    )
    .reset_index()
)

# Recency: cutoff date minus last completed order.
completed_features["recency"] = (
    CUTOFF_DATE - completed_features["last_order_ts"]
).dt.days

completed_features.drop(
    columns=["last_order_ts"],
    inplace=True
)


# ============================================================
# 3. CANCELLATION RATE
# ============================================================

# Only completed and cancelled orders are considered final statuses.
final_orders = historical_orders[
    historical_orders["status"].isin(["completed", "cancelled"])
].copy()

cancellation_features = (
    final_orders.groupby("customer_id")
    .agg(
        total_final_orders=("order_id", "nunique"),
        cancelled_orders=(
            "status",
            lambda x: (x == "cancelled").sum()
        )
    )
    .reset_index()
)

cancellation_features["cancellation_rate"] = (
    cancellation_features["cancelled_orders"]
    / cancellation_features["total_final_orders"]
)

cancellation_features = cancellation_features[
    ["customer_id", "cancellation_rate"]
]


# ============================================================
# 4. MOST FREQUENT PRODUCT CATEGORY
# ============================================================

products = pd.read_csv(PROJECT_ROOT / "data" / "clean" / "products.csv")

completed_with_category = historical_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# Most frequently purchased category for each customer.
# In case of a tie, pandas selects the first category alphabetically.
favourite_category = (
    completed_with_category.groupby("customer_id")["category"]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index(name="favourite_category")
)


# ============================================================
# 5. ADD CUSTOMER DEMOGRAPHIC FEATURES
# ============================================================

customer_features = customers[
    ["customer_id", "city", "age", "gender", "signup_date"]
].copy()

customer_features["signup_date"] = pd.to_datetime(
    customer_features["signup_date"]
)

customer_features["tenure_days"] = (
    CUTOFF_DATE - customer_features["signup_date"]
).dt.days

customer_features.drop(
    columns=["signup_date"],
    inplace=True
)


# ============================================================
# 6. COMBINE ALL FEATURES
# ============================================================

lapse_data = (
    completed_features
    .merge(cancellation_features, on="customer_id", how="left")
    .merge(favourite_category, on="customer_id", how="left")
    .merge(customer_features, on="customer_id", how="left")
    .merge(labels, on="customer_id", how="inner")
)

# Any customer without a cancellation record has a zero rate.
lapse_data["cancellation_rate"] = (
    lapse_data["cancellation_rate"].fillna(0)
)

# Remove temporary or unnecessary columns if present.
lapse_data = lapse_data.sort_values(
    "customer_id"
).reset_index(drop=True)


# ============================================================
# 7. DEFINE FEATURES AND TARGET
# ============================================================

feature_columns = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "favourite_category",
    "city",
    "age",
    "gender",
    "tenure_days"
]

X = lapse_data[feature_columns].copy()
y = lapse_data["lapsed"].copy()


# ============================================================
# 8. VALIDATE COHORT AND LABELS
# ============================================================

print("\n" + "=" * 60)
print("LAPSE DATA VALIDATION")
print("=" * 60)

print("Lapse dataset shape:", lapse_data.shape)
print("\nTarget distribution:")
print(y.value_counts().sort_index())

assert len(lapse_data) == 1156
assert y.sum() == 443
assert (y == 0).sum() == 713

assert not X.isnull().any().any(), (
    "Missing values found in lapse features"
)

assert (historical_orders["order_ts"] < CUTOFF_DATE).all()

print("\nALL LAPSE COHORT CHECKPOINTS PASSED")


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

assert len(X_train) == 924
assert len(X_test) == 232

print("\nTRAIN-TEST SPLIT CHECKPOINT PASSED")


# ============================================================
# 10. BASELINE MODEL
# ============================================================

# Baseline: predict every test customer as not lapsed (class 0).
baseline_predictions = pd.Series(
    0,
    index=y_test.index
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    pos_label=1,
    zero_division=0
)

baseline_cm = confusion_matrix(
    y_test,
    baseline_predictions,
    labels=[0, 1]
)

print("\n" + "=" * 60)
print("BASELINE MODEL")
print("=" * 60)

print("Baseline accuracy:", round(baseline_accuracy, 4))
print("Lapsed-class F1:", round(baseline_f1, 4))

print("\nConfusion matrix:")
print(baseline_cm)

print("\nTASK 4 COMPLETED SUCCESSFULLY")


# ============================================================
# TASK 5: PREPROCESS WITHOUT DATA LEAKAGE
# ============================================================

# Identify numeric and categorical features.

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]


# ============================================================
# 1. CHECK MISSING VALUES BEFORE PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

print("Missing values in training features:")
print(X_train.isnull().sum())

assert not X_train.isnull().any().any()
assert not X_test.isnull().any().any()

print("\nNo missing values in training or test features.")


# ============================================================
# 2. CREATE PREPROCESSING PIPELINES
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 3. FIT ONLY ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

# Apply the already fitted transformations to test data.
X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 4. VALIDATE TRANSFORMED DATA
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING RESULTS")
print("=" * 60)

print("Original training shape:", X_train.shape)
print("Original testing shape:", X_test.shape)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

assert not np.isnan(X_train_processed).any()
assert not np.isnan(X_test_processed).any()

print("\nNo missing values in processed features.")


# ============================================================
# 5. COMPARE FREQUENCY MEANS
# ============================================================

# Get the fitted StandardScaler from the numeric pipeline.
fitted_scaler = (
    preprocessor
    .named_transformers_["num"]
    .named_steps["scaler"]
)

# Find the position of frequency in numeric_features.
frequency_index = numeric_features.index("frequency")

training_frequency_mean = fitted_scaler.mean_[frequency_index]

full_cohort_frequency_mean = X["frequency"].mean()

print("\n" + "=" * 60)
print("FREQUENCY MEAN COMPARISON")
print("=" * 60)

print(
    "Training frequency mean:",
    round(training_frequency_mean, 4)
)

print(
    "Full cohort frequency mean:",
    round(full_cohort_frequency_mean, 4)
)

print("\nTASK 5 PREPROCESSING COMPLETED SUCCESSFULLY")


# ============================================================
# TASK 6: TRAIN, TUNE AND EVALUATE MODELS
# ============================================================

# IMPORTANT:
# Use original X_train and X_test, NOT the preprocessed
# X_train_processed and X_test_processed.
#
# Preprocessing is included inside each Pipeline so that
# GridSearchCV fits encoders and scalers independently
# within each cross-validation fold.


# ============================================================
# 1. LOGISTIC REGRESSION GRID SEARCH
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION GRID SEARCH")
print("=" * 60)

lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)

lr_param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

lr_grid = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=lr_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

lr_grid.fit(X_train, y_train)

best_lr = lr_grid.best_estimator_

print("\nBest Logistic Regression parameters:")
print(lr_grid.best_params_)

print(
    "Best Logistic Regression CV F1:",
    round(lr_grid.best_score_, 4)
)


# ============================================================
# 2. RANDOM FOREST GRID SEARCH
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST GRID SEARCH")
print("=" * 60)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

rf_param_grid = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

print("\nBest Random Forest parameters:")
print(rf_grid.best_params_)

print(
    "Best Random Forest CV F1:",
    round(rf_grid.best_score_, 4)
)

print("\nRandom Forest total fits:", len(rf_grid.cv_results_["params"]) * 5)


# ============================================================
# 3. EVALUATE BOTH MODELS ON UNTOUCHED TEST DATA
# ============================================================

lr_predictions = best_lr.predict(X_test)
rf_predictions = best_rf.predict(X_test)


def evaluate_model(model_name, y_true, predictions):

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    precision = precision_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        pos_label=1,
        zero_division=0
    )

    accuracy = accuracy_score(y_true, predictions)

    print("Confusion matrix:")
    print(cm)

    print("Accuracy:", round(accuracy, 4))
    print("Lapsed precision:", round(precision, 4))
    print("Lapsed recall:", round(recall, 4))
    print("Lapsed F1:", round(f1, 4))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


lr_metrics = evaluate_model(
    "TUNED LOGISTIC REGRESSION",
    y_test,
    lr_predictions
)

rf_metrics = evaluate_model(
    "TUNED RANDOM FOREST",
    y_test,
    rf_predictions
)


# ============================================================
# 4. COMPARE THE MODELS
# ============================================================

comparison = pd.DataFrame([
    lr_metrics,
    rf_metrics
])

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison.round(4).to_string(index=False))


# ============================================================
# 5. SELECT THE MODEL
# ============================================================

# Select the model with the higher test F1.
# Both models have already been tuned using training data only.

if rf_metrics["f1"] >= lr_metrics["f1"]:
    selected_model = best_rf
    selected_model_name = "Random Forest"
    selected_predictions = rf_predictions
    selected_metrics = rf_metrics
else:
    selected_model = best_lr
    selected_model_name = "Logistic Regression"
    selected_predictions = lr_predictions
    selected_metrics = lr_metrics

print("\n" + "=" * 60)
print("SELECTED MODEL")
print("=" * 60)

print("Selected model:", selected_model_name)
print("Test F1:", round(selected_metrics["f1"], 4))

assert selected_metrics["f1"] >= 0.55, (
    "Selected model did not reach the required test F1 of 0.55"
)

print("\nTASK 6 COMPLETED SUCCESSFULLY")

# ============================================================
# TASK 7: OVERSAMPLING
# ============================================================

print("\n" + "=" * 60)
print("TASK 7: OVERSAMPLING")
print("=" * 60)

# Separate the two classes in the training data
X_train_not_lapsed = X_train[y_train == 0]
y_train_not_lapsed = y_train[y_train == 0]

X_train_lapsed = X_train[y_train == 1]
y_train_lapsed = y_train[y_train == 1]

print("\nOriginal training distribution:")
print(y_train.value_counts().sort_index())

# Number of additional lapsed rows required
additional_rows = len(X_train_not_lapsed) - len(X_train_lapsed)

# Oversample only the lapsed training customers
X_lapsed_oversampled, y_lapsed_oversampled = resample(
    X_train_lapsed,
    y_train_lapsed,
    replace=True,
    n_samples=additional_rows,
    random_state=42
)

# Combine original training data with duplicated lapsed rows
X_train_balanced = pd.concat(
    [X_train, X_lapsed_oversampled],
    axis=0
)

y_train_balanced = pd.concat(
    [y_train, y_lapsed_oversampled],
    axis=0
)

print("\nBalanced training distribution:")
print(y_train_balanced.value_counts().sort_index())

# Validate the oversampling
assert y_train_balanced.value_counts()[0] == 570
assert y_train_balanced.value_counts()[1] == 570
assert len(X_train_balanced) == 1140

print("\nOVERSAMPLING CHECKPOINT PASSED")


# ------------------------------------------------------------
# REFIT BEST LOGISTIC REGRESSION ON BALANCED DATA
# ------------------------------------------------------------

# Clone the tuned pipeline so the original best_lr remains unchanged
balanced_lr = clone(best_lr)

balanced_lr.fit(X_train_balanced, y_train_balanced)

# Evaluate on the ORIGINAL, unchanged test data
balanced_lr_predictions = balanced_lr.predict(X_test)

balanced_lr_metrics = evaluate_model(
    "BALANCED LOGISTIC REGRESSION",
    y_test,
    balanced_lr_predictions
)


# ------------------------------------------------------------
# COMPARE ORIGINAL AND BALANCED LOGISTIC REGRESSION
# ------------------------------------------------------------

original_lr_predictions = best_lr.predict(X_test)

original_lr_metrics = evaluate_model(
    "ORIGINAL LOGISTIC REGRESSION",
    y_test,
    original_lr_predictions
)

oversampling_comparison = pd.DataFrame(
    [
        {
            "model": "Original Logistic Regression",
            **original_lr_metrics
        },
        {
            "model": "Balanced Logistic Regression",
            **balanced_lr_metrics
        }
    ]
)

print("\n" + "=" * 60)
print("OVERSAMPLING COMPARISON")
print("=" * 60)

print(oversampling_comparison.to_string(index=False))

# Confirm test data was not modified
assert len(X_test) == 232
assert len(y_test) == 232

print("\nTASK 7 COMPLETED SUCCESSFULLY")
print("\n" + "=" * 60)
print("TASK 8.1 — RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

rf_preprocessor = best_rf.named_steps["preprocessor"]
rf_classifier = best_rf.named_steps["classifier"]

rf_feature_names = rf_preprocessor.get_feature_names_out()

rf_importance = pd.DataFrame({
    "feature": rf_feature_names,
    "importance": rf_classifier.feature_importances_
})

rf_importance = rf_importance.sort_values(
    "importance", ascending=False
)

print("\nTOP 5 RANDOM FOREST FEATURES:")
print(rf_importance.head(5).to_string(index=False))

rf_importance.head(5).to_csv(
    OUTPUT_DIR / "top_5_feature_importance.csv",
    index=False
)
print("\n" + "=" * 60)
print("TASK 8.2 — LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 60)

lr_preprocessor = best_lr.named_steps["preprocessor"]
lr_classifier = best_lr.named_steps["classifier"]

lr_feature_names = lr_preprocessor.get_feature_names_out()

lr_coefficients = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": lr_classifier.coef_[0]
})

lr_coefficients["absolute_coefficient"] = (
    lr_coefficients["coefficient"].abs()
)

lr_coefficients = lr_coefficients.sort_values(
    "absolute_coefficient", ascending=False
)

print("\nTOP 3 LOGISTIC REGRESSION COEFFICIENTS:")
print(
    lr_coefficients.head(3)[
        ["feature", "coefficient"]
    ].to_string(index=False)
)

print("\nInterpretation:")
for _, row in lr_coefficients.head(3).iterrows():
    direction = (
        "increases" if row["coefficient"] > 0
        else "decreases"
    )
    print(
        f"{row['feature']}: {direction} the likelihood of lapse "
        f"(coefficient={row['coefficient']:.4f})"
    )
    import joblib
import numpy as np

print("\n" + "=" * 60)
print("TASK 8.3 — MODEL PERSISTENCE")
print("=" * 60)

model_path = OUTPUT_DIR / "lapse_model.joblib"

# Save selected model
joblib.dump(selected_model, model_path)

print(f"Model saved to: {model_path}")

# Reload model
loaded_model = joblib.load(model_path)

# Generate predictions using original and reloaded models
original_predictions = selected_model.predict(X_test)
reloaded_predictions = loaded_model.predict(X_test)

# Verify predictions
assert np.array_equal(
    original_predictions,
    reloaded_predictions
), "Predictions do not match!"

print("Model successfully reloaded.")
print("Original and reloaded predictions match.")
print("MODEL PERSISTENCE CHECKPOINT PASSED")

print("\n" + "=" * 60)
print("TASK 8.4 — PREMIUM CUSTOMER RETENTION LIST")
print("=" * 60)

# Load segmentation results
segments_df = pd.read_csv(
    OUTPUT_DIR / "customer_segments.csv"
)

# Map test-set indices back to customer IDs
retention_df = pd.DataFrame({
    "customer_id": lapse_data.loc[
        X_test.index, "customer_id"
    ].values,
    "actual_lapsed": y_test.values,
    "lapse_probability": selected_model.predict_proba(
        X_test
    )[:, 1]
})

# Add Premium classification
retention_df = retention_df.merge(
    segments_df[["customer_id", "is_premium"]],
    on="customer_id",
    how="left"
)

# Keep Premium customers only
retention_df = retention_df[
    retention_df["is_premium"] == True
].copy()

# Rank by predicted lapse probability
retention_df = retention_df.sort_values(
    "lapse_probability",
    ascending=False
)

# Select top 10
retention_list = retention_df.head(10).copy()

# Save retention list
retention_list.to_csv(
    OUTPUT_DIR / "retention_list.csv",
    index=False
)

print("\nTOP 10 PREMIUM CUSTOMERS FOR RETENTION:")
print(retention_list.to_string(index=False))

print("\nRetention list shape:", retention_list.shape)
print("Premium customers in test set:", len(retention_df))

print("\nRETENTION LIST SAVED SUCCESSFULLY")

