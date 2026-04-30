"""
app.py
------
Flask application for ShipmentSure.

Routes
------
GET  /                – Prediction form
POST /predict         – JSON in → JSON out (prediction + probability)
GET  /feature-importance – Returns top feature importances as JSON
GET  /health          – Health check
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np
from flask import Flask, render_template, request, jsonify
from src.predict import load_pipeline, predict as run_predict
from src.preprocess import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load pipeline at startup ─────────────────────────────────────────────────
try:
    _pipeline = load_pipeline(os.path.join(BASE_DIR, "models", "pipeline.pkl"))
    print("[app] Pipeline loaded.")
except FileNotFoundError as exc:
    _pipeline = None
    print(f"[app] WARNING: {exc}")


def _get_feature_importances() -> list:
    """
    Extract top-8 feature importances from the trained classifier.
    Works for RandomForest, XGBoost (feature_importances_) and
    LogisticRegression (coef_).
    Returns list of {feature, importance} dicts, sorted descending.
    """
    if _pipeline is None:
        return []

    pre = _pipeline.named_steps["preprocessor"]
    clf = _pipeline.named_steps["clf"]

    # Reconstruct feature names after OneHotEncoding
    num_names = NUMERICAL_FEATURES[:]
    cat_encoder = pre.named_transformers_["cat"].named_steps["encoder"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = num_names + cat_names

    try:
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            return []

        pairs = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:8]

        total = sum(v for _, v in pairs) or 1
        return [
            {"feature": f, "importance": round(float(v / total) * 100, 1)}
            for f, v in pairs
        ]
    except Exception:
        return []


# Cache importances once
_importances = None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if _pipeline is None:
        return jsonify({"error": "Model not loaded. Run `python src/train.py` first."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    try:
        input_dict = {
            "Customer_rating":     int(data["Customer_rating"]),
            "Cost_of_the_Product": float(data["Cost_of_the_Product"]),
            "Prior_purchases":     int(data["Prior_purchases"]),
            "Discount_offered":    float(data["Discount_offered"]),
            "Weight_in_gms":       float(data["Weight_in_gms"]),
            "Mode_of_Shipment":    str(data["Mode_of_Shipment"]),
            "Product_importance":  str(data["Product_importance"]),
            "Warehouse_block":     str(data["Warehouse_block"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 422

    result = run_predict(_pipeline, input_dict)
    return jsonify(result)


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    global _importances
    if _importances is None:
        _importances = _get_feature_importances()
    return jsonify({"importances": _importances})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _pipeline is not None})


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
