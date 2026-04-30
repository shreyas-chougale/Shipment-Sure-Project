"""
train.py
--------
Trains Logistic Regression, Random Forest, and XGBoost inside a full sklearn
Pipeline. Selects the best model by 5-fold ROC-AUC and saves to pipeline.pkl.

Usage:  python src/train.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import build_preprocessor, load_data, get_X_y

DATA_PATH     = os.path.join("data", "dataset.csv")
PIPELINE_PATH = os.path.join("models", "pipeline.pkl")
os.makedirs("models", exist_ok=True)


def _build_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", estimator),
    ])


def _class_ratio(y_train) -> float:
    """Return negative/positive ratio for XGBoost scale_pos_weight."""
    counts = y_train.value_counts()
    return round(counts.get(0, 1) / max(counts.get(1, 1), 1), 4)


CANDIDATES = {
    "logistic_regression": {
        "estimator": LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced"
        ),
        "param_grid": {
            "clf__C":      [0.001, 0.01, 0.1, 1, 10, 100],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "saga"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "param_grid": {
            "clf__n_estimators":      [100, 200, 300],
            "clf__max_depth":         [None, 8, 15, 25],
            "clf__min_samples_split": [2, 5, 10],
            "clf__max_features":      ["sqrt", "log2"],
        },
    },
    "xgboost": {
        "estimator": None,          # set after computing scale_pos_weight
        "param_grid": {
            "clf__n_estimators":  [100, 200, 300],
            "clf__max_depth":     [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__subsample":     [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.9, 1.0],
        },
    },
}


def train():
    print("=" * 58)
    print("  ShipmentSure – Model Training")
    print("=" * 58)

    df = load_data(DATA_PATH)
    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Class balance (train): {y_train.value_counts().to_dict()}")

    # Set XGBoost scale_pos_weight now that we have y_train
    spw = _class_ratio(y_train)
    CANDIDATES["xgboost"]["estimator"] = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        scale_pos_weight=spw,
    )
    print(f"XGBoost scale_pos_weight = {spw}")

    best_score    = -np.inf
    best_pipeline = None
    best_name     = None

    for name, cfg in CANDIDATES.items():
        print(f"\n[train] {name} ...")
        pipe = _build_pipeline(cfg["estimator"])
        gs   = GridSearchCV(
            pipe,
            param_grid=cfg["param_grid"],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        score = gs.best_score_
        print(f"  CV ROC-AUC : {score:.4f}")
        print(f"  Best params: {gs.best_params_}")

        if score > best_score:
            best_score    = score
            best_pipeline = gs.best_estimator_
            best_name     = name

    # ── Evaluate best on test set ─────────────────────────────────────────
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    print(f"\n{'='*58}")
    print(f"  Best Model : {best_name}")
    print(f"  CV ROC-AUC : {best_score:.4f}")
    print(f"  {'─'*52}")
    print(f"  Accuracy   : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision  : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall     : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score   : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC    : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"{'='*58}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Delayed", "On Time"]))

    joblib.dump(best_pipeline, PIPELINE_PATH)
    print(f"[train] Pipeline saved → {PIPELINE_PATH}")
    return best_pipeline


if __name__ == "__main__":
    train()
