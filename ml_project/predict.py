"""
predict.py
──────────
Loads the 3 trained models and runs predictions on sample events.
This is the same logic your Flask/FastAPI backend will use to serve
predictions to forecast.html.

Run:
    python predict.py
"""

import json
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = "models"


def load_models():
    priority_model = joblib.load(f"{MODELS_DIR}/priority_model.pkl")
    priority_le = joblib.load(f"{MODELS_DIR}/priority_label_encoder.pkl")
    closure_model = joblib.load(f"{MODELS_DIR}/closure_model.pkl")
    resolution_model = joblib.load(f"{MODELS_DIR}/resolution_model.pkl")

    with open(f"{MODELS_DIR}/feature_columns.json") as f:
        feature_cols = json.load(f)

    return priority_model, priority_le, closure_model, resolution_model, feature_cols


def build_input(event: dict, feature_list: list) -> pd.DataFrame:
    """Convert a raw event dict into a single-row DataFrame matching training columns."""
    row = {col: event.get(col, "unknown") for col in feature_list}
    return pd.DataFrame([row])


def predict_event(event: dict, models, feature_cols):
    priority_model, priority_le, closure_model, resolution_model, _ = models

    # Priority + closure use the base feature set
    base_input = build_input(event, feature_cols["all"])
    priority_pred = priority_model.predict(base_input)[0]
    priority_label = priority_le.inverse_transform([priority_pred])[0]
    priority_proba = priority_model.predict_proba(base_input)[0].max()

    closure_pred = closure_model.predict(base_input)[0]
    closure_proba = closure_model.predict_proba(base_input)[0].max()

    # Resolution model additionally needs corridor + police_station
    res_features = feature_cols["all"] + ["corridor", "police_station"]
    res_input = build_input(event, res_features)
    log_minutes = resolution_model.predict(res_input)[0]
    minutes = float(np.expm1(log_minutes))

    return {
        "priority": priority_label,
        "priority_confidence": round(float(priority_proba) * 100, 1),
        "road_closure_required": bool(closure_pred),
        "closure_confidence": round(float(closure_proba) * 100, 1),
        "estimated_resolution_minutes": round(minutes, 1)
    }


def main():
    models = load_models()
    _, _, _, _, feature_cols = models

    # A few realistic sample events to sanity-check predictions
    sample_events = [
        {
            "event_type": "unplanned",
            "event_cause": "vehicle_breakdown",
            "zone": "Central Zone 2",
            "veh_type": "bmtc_bus",
            "hour": 21,
            "day_of_week": 5,
            "month": 3,
            "is_weekend": 1,
            "is_peak_hour": 1,
            "corridor": "Mysore Road",
            "police_station": "Yelahanka"
        },
        {
            "event_type": "planned",
            "event_cause": "public_event",
            "zone": "South Zone 1",
            "veh_type": "unknown",
            "hour": 18,
            "day_of_week": 6,
            "month": 1,
            "is_weekend": 1,
            "is_peak_hour": 0,
            "corridor": "CBD 1",
            "police_station": "Halasuru Gate"
        },
        {
            "event_type": "unplanned",
            "event_cause": "pot_holes",
            "zone": "North Zone 1",
            "veh_type": "unknown",
            "hour": 10,
            "day_of_week": 2,
            "month": 2,
            "is_weekend": 0,
            "is_peak_hour": 0,
            "corridor": "Tumkur Road",
            "police_station": "Yeshwanthpura"
        }
    ]

    print("=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    for i, event in enumerate(sample_events, 1):
        result = predict_event(event, models, feature_cols)
        print(f"\n--- Event {i}: {event['event_cause']} on {event['corridor']} ---")
        print(f"  Input  : hour={event['hour']}, type={event['event_type']}, zone={event['zone']}")
        print(f"  Output : Priority={result['priority']} ({result['priority_confidence']}% conf)")
        print(f"           Road Closure={result['road_closure_required']} ({result['closure_confidence']}% conf)")
        print(f"           Est. Resolution={result['estimated_resolution_minutes']:.0f} min "
              f"({result['estimated_resolution_minutes']/60:.1f} hrs)")


if __name__ == "__main__":
    main()
