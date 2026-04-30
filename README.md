# ShipmentSure
**Predicting On-Time Delivery Using Supplier Data**

E-Commerce Shipping Dataset · scikit-learn Pipeline · Flask

---

## Quick Start (Local)

```bash
# 1. Move into the project directory
cd ShipmentSure

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Train the model (saves models/pipeline.pkl)
python src/train.py

# 5. Start the web app
python app/app.py
# → Open http://127.0.0.1:5000 in your browser
```

Or use the one-command script:

```bash
chmod +x run.sh && ./run.sh
```

---

## Project Structure

```
ShipmentSure/
├── data/
│   └── dataset.csv                 ← E-Commerce Shipping Dataset (10,999 rows)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py               ← Feature definitions, load_data, build_preprocessor
│   ├── train.py                    ← GridSearchCV: LR / RF / XGBoost → pipeline.pkl
│   ├── evaluate.py                 ← Metrics + confusion matrix + ROC curve
│   └── predict.py                  ← Single-row inference
│
├── models/
│   └── pipeline.pkl                ← Auto-generated after `python src/train.py`
│
├── app/
│   ├── app.py                      ← Flask: GET / · POST /predict · GET /health
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── script.js
│
├── notebooks/
│   └── eda.ipynb
│
├── reports/figures/                ← EDA + evaluation plots (auto-generated)
├── requirements.txt
├── run.sh
└── README.md
```

---

## Features Used (8 total)

| Feature | Type | Values |
|---|---|---|
| Mode_of_Shipment | Categorical | Flight, Ship, Road |
| Product_importance | Categorical | low, medium, high |
| Warehouse_block | Categorical | A, B, C, D, F |
| Customer_rating | Numerical | 1 – 5 |
| Cost_of_the_Product | Numerical | $ amount |
| Prior_purchases | Numerical | count |
| Discount_offered | Numerical | 0 – 65 % |
| Weight_in_gms | Numerical | grams |

**Target:** `Reached.on.Time_Y.N` → `on_time_delivery`
- Original: 1 = delayed → Converted: **1 = on time**, 0 = delayed

---

## ML Pipeline

```
Raw CSV
  └── load_data()  (drops unused columns, flips target)
        ↓
  ColumnTransformer
    ├── Numerical:   SimpleImputer(median) → StandardScaler
    └── Categorical: SimpleImputer(mode)   → OneHotEncoder
        ↓
  Best Classifier (auto-selected via GridSearchCV, 5-fold ROC-AUC)
    ├── Logistic Regression  (C, solver tuned)
    ├── Random Forest        (n_estimators, depth, features tuned)
    └── XGBoost              (n_estimators, depth, lr, subsample tuned)
        ↓
  models/pipeline.pkl
```

Class imbalance handled via `class_weight="balanced"` (LR, RF) and
`scale_pos_weight` auto-computed from training data (XGBoost).

---

## Running Steps

### Train
```bash
python src/train.py
```
Prints CV ROC-AUC for each model, selects the best, prints Accuracy /
Precision / Recall / F1 / ROC-AUC on the test set, saves `pipeline.pkl`.

### Evaluate
```bash
python src/evaluate.py
```
Loads saved pipeline, prints metrics, saves confusion matrix + ROC curve
to `reports/figures/`.

### EDA Notebook
```bash
jupyter notebook notebooks/eda.ipynb
```

### Flask Web App
```bash
python app/app.py
# → http://127.0.0.1:5000
```

---

## Flask API

| Route | Method | Description |
|---|---|---|
| `/` | GET | Prediction form |
| `/predict` | POST (JSON) | Run model, return prediction + probabilities |
| `/feature-importance` | GET | Top feature importances from trained model |
| `/health` | GET | Health check |

**POST /predict — request body:**
```json
{
  "Customer_rating": 3,
  "Cost_of_the_Product": 177,
  "Prior_purchases": 3,
  "Discount_offered": 10,
  "Weight_in_gms": 1233,
  "Mode_of_Shipment": "Ship",
  "Product_importance": "medium",
  "Warehouse_block": "D"
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability_on_time": 0.75,
  "probability_delayed": 0.25,
  "label": "On Time"
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.9+ · Flask |
| Frontend | HTML · CSS · Vanilla JS (no frameworks) |
| ML | scikit-learn · XGBoost · joblib |
| Data | pandas · numpy |
| Visualization | matplotlib · seaborn |

---

## Notes

- **No Replit dependency** — runs entirely locally or on any Linux/macOS/Windows machine.
- **No cloud / external API** — fully offline once dependencies are installed.
- `models/pipeline.pkl` is generated locally; it is not committed to version control.
- Python 3.9 or higher is recommended.

---

## License

MIT
