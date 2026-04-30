"""
predict.py
----------
Single-row inference: pass raw input directly to the pipeline.
The pipeline (preprocessor + classifier) handles all encoding and scaling.

Usage:  python src/predict.py
"""

import os
import sys
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import ALL_FEATURES

PIPELINE_PATH = os.path.join("models", "pipeline.pkl")


def load_pipeline(path: str = PIPELINE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No pipeline at '{path}'. Run `python src/train.py` first."
        )
    return joblib.load(path)


def predict(pipeline, input_dict: dict) -> dict:
    """
    Run inference for one shipment.

    Parameters
    ----------
    pipeline   : loaded sklearn Pipeline
    input_dict : dict with the 8 raw feature values

    Returns
    -------
    dict: prediction (int), probability_on_time (float), label (str)
    """
    # Build DataFrame → pass raw input directly to the pipeline
    df = pd.DataFrame([input_dict])[ALL_FEATURES]

    prediction  = int(pipeline.predict(df)[0])
    proba       = pipeline.predict_proba(df)[0]
    prob_ontime = float(proba[1])

    return {
        "prediction":          prediction,
        "probability_on_time": round(prob_ontime, 4),
        "probability_delayed": round(float(proba[0]), 4),
        "label":               "On Time" if prediction == 1 else "Delayed",
    }


if __name__ == "__main__":
    sample = {
        "Customer_rating":     3,
        "Cost_of_the_Product": 200,
        "Prior_purchases":     3,
        "Discount_offered":    10,
        "Weight_in_gms":       2500,
        "Mode_of_Shipment":    "Ship",
        "Product_importance":  "medium",
        "Warehouse_block":     "D",
    }
    pipe   = load_pipeline()
    result = predict(pipe, sample)
    print("\n=== Prediction ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
