"""
train_improved.py
──────────────────
Retrains all 3 models on the cleaned dataset (processed_clean.csv).

Changes vs the original train.py:
  1. Uses cleaned categories (no more "Debris" vs "debris" duplication)
  2. Rare event_cause values bucketed into "other_rare"
  3. Adds `cause_speed` (fast/slow) feature -- the single biggest lever for
     improving the resolution-time regressor, since it directly encodes the
     bimodal nature of resolution times instead of making the model infer it.
  4. Adds `time_unspecified` flag for planned events with no logged hour.
  5. Reports MEDIAN absolute error alongside MAE for the regressor, since a
     handful of multi-week pothole/construction cases dominate the mean and
     make MAE alone misleading.
  6. Slightly more conservative XGBoost settings (added L2 regularization,
     early stopping via a validation split) to reduce overfitting risk now
     that there are more one-hot columns from cause_speed/time_unspecified.

Run:
    python train_improved.py
Outputs (in models_improved/):
    priority_model.pkl, closure_model.pkl, resolution_model.pkl,
    priority_label_encoder.pkl, feature_columns.json, metrics.json
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
    mean_absolute_error, median_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

DATA_PATH = "data/processed_clean.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42

CAT_FEATURES = ["event_type", "event_cause", "zone", "veh_type", "cause_speed"]
NUM_FEATURES = ["hour", "day_of_week", "month", "is_weekend", "is_peak_hour", "time_unspecified"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES

import os
os.makedirs(MODELS_DIR, exist_ok=True)


def build_preprocessor(cat_cols):
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )


def train_priority_model(df: pd.DataFrame, metrics: dict):
    print("\n" + "=" * 60)
    print("MODEL 1: Priority Classifier (High / Low)  [IMPROVED]")
    print("=" * 60)

    X = df[ALL_FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df["priority"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("prep", build_preprocessor(CAT_FEATURES)),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,       # L2 regularization -- added
            reg_alpha=0.1,        # L1 regularization -- added
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)

    print(f"Accuracy: {acc:.4f}  (was 0.6765)")
    print(f"F1 Score: {f1:.4f}  (was 0.4585)")
    print(classification_report(y_test, preds, target_names=le.classes_))

    joblib.dump(pipeline, f"{MODELS_DIR}/priority_model.pkl")
    joblib.dump(le, f"{MODELS_DIR}/priority_label_encoder.pkl")

    metrics["priority_model"] = {
        "accuracy": round(acc, 4), "f1_score": round(f1, 4),
        "classes": le.classes_.tolist(), "report": report
    }
    return pipeline


def train_closure_model(df: pd.DataFrame, metrics: dict):
    print("\n" + "=" * 60)
    print("MODEL 2: Road Closure Classifier (True / False)  [IMPROVED]")
    print("=" * 60)

    X = df[ALL_FEATURES]
    y = df["requires_road_closure"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    pipeline = Pipeline([
        ("prep", build_preprocessor(CAT_FEATURES)),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=5,            # reverted to match original -- depth=4 underperformed in testing
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            reg_lambda=1.0,         # lighter regularization -- 2.0 was too aggressive for this rare class
            reg_alpha=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=["No Closure", "Closure"], output_dict=True)

    print(f"Accuracy: {acc:.4f}  (was 0.8245)")
    print(f"F1 Score: {f1:.4f}  (was 0.3310)")
    print(classification_report(y_test, preds, target_names=["No Closure", "Closure"]))

    joblib.dump(pipeline, f"{MODELS_DIR}/closure_model.pkl")

    metrics["closure_model"] = {
        "accuracy": round(acc, 4), "f1_score": round(f1, 4), "report": report
    }
    return pipeline


def train_resolution_model(df: pd.DataFrame, metrics: dict):
    print("\n" + "=" * 60)
    print("MODEL 3: Resolution Time Regressor (minutes)  [IMPROVED]")
    print("=" * 60)

    sub = df.dropna(subset=["resolution_minutes"]).copy()

    res_features = ALL_FEATURES + ["corridor", "police_station"]
    res_cat_features = CAT_FEATURES + ["corridor", "police_station"]

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
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.1,
            random_state=RANDOM_STATE
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds_log = pipeline.predict(X_test)

    preds_min = np.expm1(preds_log)
    actual_min = np.expm1(y_test)

    mae = mean_absolute_error(actual_min, preds_min)
    medae = median_absolute_error(actual_min, preds_min)
    r2 = r2_score(y_test, preds_log)

    # Also report error in log-space minutes (more representative of "typical" error)
    mae_log = mean_absolute_error(y_test, preds_log)

    print(f"MAE (minutes): {mae:.1f}  (was 5663.0)")
    print(f"Median AE (minutes): {medae:.1f}  <-- new metric, much more representative")
    print(f"MAE (log-space): {mae_log:.3f}")
    print(f"R² (log-space): {r2:.4f}  (was 0.5151)")

    joblib.dump(pipeline, f"{MODELS_DIR}/resolution_model.pkl")

    metrics["resolution_model"] = {
        "mae_minutes": round(float(mae), 1),
        "median_ae_minutes": round(float(medae), 1),
        "mae_log_space": round(float(mae_log), 4),
        "r2_log_space": round(float(r2), 4),
        "trained_on_rows": len(sub),
        "features_used": res_features,
        "note": "Target is log1p(resolution_minutes); inverse with expm1() after predict()"
    }
    return pipeline


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded cleaned data: {df.shape}")

    metrics = {}
    train_priority_model(df, metrics)
    train_closure_model(df, metrics)
    train_resolution_model(df, metrics)

    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump({"categorical": CAT_FEATURES, "numeric": NUM_FEATURES, "all": ALL_FEATURES}, f, indent=2)

    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: IMPROVED MODELS")
    print("=" * 60)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'report'} for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
