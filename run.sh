#!/bin/bash
# ShipmentSure – Local run script
# Works in VS Code terminal, macOS Terminal, or any Linux/macOS shell.

set -e

echo "========================================"
echo "  ShipmentSure: On-Time Delivery Predictor"
echo "========================================"
echo ""

# 1. Install Python dependencies
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt --quiet
echo "      Done."

# 2. Train model if pipeline.pkl is missing
if [ ! -f "models/pipeline.pkl" ]; then
    echo ""
    echo "[2/3] Training models (LR, Random Forest, XGBoost + GridSearchCV)..."
    python src/train.py
    echo "      Pipeline saved to models/pipeline.pkl"
else
    echo "[2/3] Trained pipeline found — skipping training."
fi

# 3. Start Flask app
echo ""
echo "[3/3] Starting Flask server..."
echo ""
echo "  -> Open browser at: http://127.0.0.1:5000"
echo ""
python app/app.py
