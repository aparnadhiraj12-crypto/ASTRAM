"""
app.py
──────
Flask API that serves predictions from the 3 trained models (IMPROVED VERSION).
This is what forecast.html calls instead of the rule-based JS logic.

Changes vs the original app.py:
  - Computes `cause_speed` (fast/slow) and `time_unspecified` server-side,
    matching the features the improved models were trained on.
  - forecast.html and any other caller do NOT need to change -- the API
    request/response shape is identical. All new feature engineering is
    handled internally here.

Run:
    python app.py
Then it's live at:
    http://localhost:5000/predict   (POST)
    http://localhost:5000/health    (GET, sanity check)
    http://localhost:5000/          (GET, status landing page)

In Codespaces, port 5000 will auto-forward — check the "PORTS" tab.
Remember to set port 5000's visibility to "Public" or fetch() calls
from forecast.html (served on a different port) will fail.
"""

import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow forecast.html (served from a different port) to call this API

MODELS_DIR = "models"

# ── Load everything once at startup, not per-request ──
priority_model = joblib.load(f"{MODELS_DIR}/priority_model.pkl")
priority_le = joblib.load(f"{MODELS_DIR}/priority_label_encoder.pkl")
closure_model = joblib.load(f"{MODELS_DIR}/closure_model.pkl")
resolution_model = joblib.load(f"{MODELS_DIR}/resolution_model.pkl")

with open(f"{MODELS_DIR}/feature_columns.json") as f:
    FEATURE_COLS = json.load(f)

BASE_FEATURES = FEATURE_COLS["all"]
RES_FEATURES = FEATURE_COLS["all"] + ["corridor", "police_station"]

print("✅ All 3 models loaded successfully")

# ── cause_speed lookup, derived from training-data medians ──
# Must match clean_data.py's classification exactly, or inference will
# silently disagree with what the model learned.
SLOW_CAUSES = {"water_logging", "pot_holes", "construction", "road_conditions", "other_rare"}

# Causes the model has never seen get bucketed the same way clean_data.py
# buckets rare training causes, so OneHotEncoder's handle_unknown="ignore"
# doesn't quietly zero out the whole category.
KNOWN_CAUSES = {
    "vehicle_breakdown", "others", "pot_holes", "construction", "water_logging",
    "accident", "tree_fall", "road_conditions", "congestion", "public_event",
    "procession", "vip_movement", "protest", "other_rare"
}


def normalize_str(value: str) -> str:
    return str(value).strip().lower() if value is not None else "unknown"


def derive_cause_speed(cause: str) -> str:
    return "slow" if cause in SLOW_CAUSES else "fast"


def normalize_cause(cause: str) -> str:
    cause = normalize_str(cause)
    return cause if cause in KNOWN_CAUSES else "other_rare"


def build_input(event: dict, feature_list: list) -> pd.DataFrame:
    """Convert incoming JSON into a single-row DataFrame matching training columns."""
    row = {col: event.get(col, "unknown") for col in feature_list}
    return pd.DataFrame([row])


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "TrafficSense ML API",
        "status": "running",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body like:
    {
      "event_type": "unplanned",
      "event_cause": "vehicle_breakdown",
      "zone": "Central Zone 2",
      "veh_type": "bmtc_bus",
      "corridor": "Mysore Road",
      "police_station": "Yelahanka",
      "hour": 21,
      "day_of_week": 5,
      "month": 3
    }
    """
    try:
        event = request.get_json(force=True)

        # Normalize categorical strings the same way clean_data.py did at training time
        event["event_type"] = normalize_str(event.get("event_type"))
        event["event_cause"] = normalize_cause(event.get("event_cause"))
        event["zone"] = normalize_str(event.get("zone"))
        event["veh_type"] = normalize_str(event.get("veh_type"))
        event["corridor"] = normalize_str(event.get("corridor"))
        event["police_station"] = normalize_str(event.get("police_station"))

        # Derive engineered fields the improved model expects
        event["is_weekend"] = 1 if int(event.get("day_of_week", 0)) in [5, 6] else 0
        event["is_peak_hour"] = 1 if int(event.get("hour", 0)) in [5, 6, 7, 19, 20, 21, 22] else 0
        event["cause_speed"] = derive_cause_speed(event["event_cause"])
        event["time_unspecified"] = 0  # API requests always provide an explicit hour

        # --- Priority prediction ---
        base_input = build_input(event, BASE_FEATURES)
        priority_pred = priority_model.predict(base_input)[0]
        priority_label = priority_le.inverse_transform([priority_pred])[0]
        priority_proba = float(priority_model.predict_proba(base_input)[0].max())

        # --- Road closure prediction ---
        closure_pred = closure_model.predict(base_input)[0]
        closure_proba = float(closure_model.predict_proba(base_input)[0].max())

        # --- Resolution time prediction ---
        res_input = build_input(event, RES_FEATURES)
        log_minutes = resolution_model.predict(res_input)[0]
        minutes = float(np.expm1(log_minutes))
        minutes = max(5, min(minutes, 20000))  # sanity clamp

        return jsonify({
            "priority": priority_label,
            "priority_confidence": round(priority_proba * 100, 1),
            "road_closure_required": bool(closure_pred),
            "closure_confidence": round(closure_proba * 100, 1),
            "estimated_resolution_minutes": round(minutes, 1),
            "model_version": "v2.0-xgboost-improved"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
