"""
evaluate.py
-----------
Loads the saved pipeline, evaluates on the held-out test set, and saves
confusion matrix + ROC curve to reports/figures/.

Usage:  python src/evaluate.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_data, get_X_y

PIPELINE_PATH = os.path.join("models", "pipeline.pkl")
DATA_PATH     = os.path.join("data", "dataset.csv")
FIGURES_DIR   = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_test_set():
    df = load_data(DATA_PATH)
    X, y = get_X_y(df)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def evaluate():
    print("[evaluate] Loading pipeline ...")
    pipeline = joblib.load(PIPELINE_PATH)

    X_test, y_test = load_test_set()
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "="*52)
    print("  EVALUATION RESULTS")
    print("="*52)
    for lbl, val in [
        ("Accuracy",  accuracy_score(y_test, y_pred)),
        ("Precision", precision_score(y_test, y_pred)),
        ("Recall",    recall_score(y_test, y_pred)),
        ("F1-Score",  f1_score(y_test, y_pred)),
        ("ROC-AUC",   roc_auc_score(y_test, y_prob)),
    ]:
        print(f"  {lbl:<12}: {val:.4f}")
    print("="*52)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Delayed", "On Time"]))

    # Confusion matrix
    path1 = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred),
        display_labels=["Delayed", "On Time"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path1, dpi=150); plt.close()
    print(f"Saved → {path1}")

    # ROC curve
    path2 = os.path.join(FIGURES_DIR, "roc_curve.png")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#3A86FF", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path2, dpi=150); plt.close()
    print(f"Saved → {path2}")

    print("\n[evaluate] Done.")


if __name__ == "__main__":
    evaluate()
