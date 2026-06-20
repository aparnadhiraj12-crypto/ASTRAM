"""
train.py
────────
Trains 3 models on the processed traffic event dataset:

  1. priority_model.pkl       Classifier -> predicts High / Low priority
  2. closure_model.pkl        Classifier -> predicts road closure needed
  3. resolution_model.pkl     Regressor  -> predicts resolution time (minutes)

Each model is an XGBoost model wrapped in a scikit-learn Pipeline that
handles categorical encoding internally, so raw category strings can be
passed straight in at inference time.

Run:
    python train.py
Outputs (in models/):
    priority_model.pkl
    closure_model.pkl
    resolution_model.pkl
    encoders.pkl           (shared label encoder for priority target)
    feature_columns.json   (exact column order expected at inference)
    metrics.json           (evaluation results for all 3 models)
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

DATA_PATH = "data/processed.csv"
MODELS_DIR = "models"

CAT_FEATURES = ["event_type", "event_cause", "zone", "veh_type"]
NUM_FEATURES = ["hour", "day_of_week", "month", "is_weekend", "is_peak_hour"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

RANDOM_STATE = 42

# NOTE on dropped features:
# `corridor` and `police_station` were excluded after an EDA check showed
# corridor alone gives ~99.8% "purity" against the priority label (almost every
# corridor maps deterministically to one class). That's not a real causal
# signal -- it strongly suggests priority is assigned by a business rule tied
# to corridor/station rather than being caused by the event itself. Training
# on it produces a model that has just memorized a lookup table (we saw 99.9%
# accuracy initially, which was the giveaway). Removing it forces the model to
# learn from genuinely predictive event-level signals instead.


def build_preprocessor():
    """One-hot encode categoricals, pass numeric features through untouched."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ],
        remainder="passthrough"
    )


def train_priority_model(df: pd.DataFrame, metrics: dict):
    print("\n" + "=" * 60)
    print("MODEL 1: Priority Classifier (High / Low)")
    print("=" * 60)

    X = df[ALL_FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df["priority"])  # High=0/Low=1 or similar, saved below

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)

    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

    joblib.dump(pipeline, f"{MODELS_DIR}/priority_model.pkl")
    joblib.dump(le, f"{MODELS_DIR}/priority_label_encoder.pkl")

    metrics["priority_model"] = {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "classes": le.classes_.tolist(),
        "report": report
    }
    return pipeline


def train_closure_model(df: pd.DataFrame, metrics: dict):
    print("\n" + "=" * 60)
    print("MODEL 2: Road Closure Classifier (True / False)")
    print("=" * 60)

    X = df[ALL_FEATURES]
    y = df["requires_road_closure"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Class imbalance handling (closures are rare ~8% of events)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=["No Closure", "Closure"], output_dict=True)

    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(classification_report(y_test, preds, target_names=["No Closure", "Closure"]))

    joblib.dump(pipeline, f"{MODELS_DIR}/closure_model.pkl")

    metrics["closure_model"] = {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "report": report
    }
    return pipeline


def train_resolution_model(df: pd.DataFrame, metrics: dict):
    print("\n" + "=" * 60)
    print("MODEL 3: Resolution Time Regressor (minutes)")
    print("=" * 60)
    print("(Uses corridor + police_station too -- location legitimately")
    print(" affects response/resolution speed, unlike priority assignment)")

    # Only rows where resolution time is known
    sub = df.dropna(subset=["resolution_minutes"]).copy()

    res_features = ALL_FEATURES + ["corridor", "police_station"]
    res_cat_features = CAT_FEATURES + ["corridor", "police_station"]

    # Log-transform target: resolution times are heavily right-skewed
    sub["log_resolution"] = np.log1p(sub["resolution_minutes"])

    X = sub[res_features]
    y = sub["log_resolution"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    res_preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), res_cat_features)],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("prep", res_preprocessor),
        ("reg", XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds_log = pipeline.predict(X_test)

    # Convert back to real minutes for interpretable metrics
    preds_min = np.expm1(preds_log)
    actual_min = np.expm1(y_test)

    mae = mean_absolute_error(actual_min, preds_min)
    r2 = r2_score(y_test, preds_log)  # R2 computed in log-space (more stable)

    print(f"MAE (minutes): {mae:.1f}")
    print(f"R² (log-space): {r2:.3f}")

    joblib.dump(pipeline, f"{MODELS_DIR}/resolution_model.pkl")

    metrics["resolution_model"] = {
        "mae_minutes": round(float(mae), 1),
        "r2_log_space": round(float(r2), 4),
        "trained_on_rows": len(sub),
        "features_used": res_features,
        "note": "Target is log1p(resolution_minutes); inverse with expm1() after predict()"
    }
    return pipeline


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded processed data: {df.shape}")

    metrics = {}

    train_priority_model(df, metrics)
    train_closure_model(df, metrics)
    train_resolution_model(df, metrics)

    # Save feature column order — inference must match this exactly
    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump({"categorical": CAT_FEATURES, "numeric": NUM_FEATURES, "all": ALL_FEATURES}, f, indent=2)

    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED & SAVED to models/")
    print("=" * 60)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'report'} for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
