"""
preprocess.py
-------------
Feature definitions, data loading, and sklearn ColumnTransformer.

Features used (8 total — aligned across training, prediction, and UI):
  Numerical   : Customer_rating, Cost_of_the_Product, Prior_purchases,
                Discount_offered, Weight_in_gms
  Categorical : Mode_of_Shipment, Product_importance, Warehouse_block

Target: Reached.on.Time_Y.N → on_time_delivery
        Original 1 = delayed  →  flipped: 1 = on time, 0 = delayed
"""

import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ── Feature definitions ───────────────────────────────────────────────────────

NUMERICAL_FEATURES = [
    "Customer_rating",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Discount_offered",
    "Weight_in_gms",
]

CATEGORICAL_FEATURES = [
    "Mode_of_Shipment",
    "Product_importance",
    "Warehouse_block",
]

ALL_FEATURES  = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "on_time_delivery"
_RAW_TARGET   = "Reached.on.Time_Y.N"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV, drop unused columns, flip and rename target.
    Returns a clean DataFrame with ALL_FEATURES + TARGET_COLUMN.
    """
    df = pd.read_csv(filepath)

    # Keep only the features we use + raw target
    keep = ALL_FEATURES + [_RAW_TARGET]
    df = df[[c for c in keep if c in df.columns]]

    # Flip target: original 1 = delayed → 1 = on time
    df[TARGET_COLUMN] = (df[_RAW_TARGET] == 0).astype(int)
    df = df.drop(columns=[_RAW_TARGET])

    print(f"[load_data] {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[load_data] on_time_delivery: {df[TARGET_COLUMN].value_counts().to_dict()}")
    return df


def get_X_y(df: pd.DataFrame):
    """Return (X, y) — features in pipeline-expected order."""
    return df[ALL_FEATURES].copy(), df[TARGET_COLUMN].copy()


# ── Preprocessor ──────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer:
      Numerical  : SimpleImputer(median) → StandardScaler
      Categorical: SimpleImputer(most_frequent) → OneHotEncoder
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERICAL_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    df = load_data(os.path.join("data", "dataset.csv"))
    X, y = get_X_y(df)
    pre = build_preprocessor()
    Xt  = pre.fit_transform(X)
    print(f"Preprocessor output shape: {Xt.shape}")
