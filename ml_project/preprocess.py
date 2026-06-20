"""
preprocess.py
─────────────
Loads the raw traffic event CSV, engineers features, and builds clean
train-ready datasets for all 3 models:
  1. priority_model      -> classification (High / Low)
  2. road_closure_model  -> classification (True / False)
  3. resolution_model    -> regression (minutes to close)

Run:
    python preprocess.py
Outputs:
    data/processed.csv   -> full cleaned dataset with engineered features
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/traffic_events.csv"
OUT_PATH = "data/processed.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day-of-week, month, weekend flag from start_datetime."""
    df["start_dt"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df["closed_dt"] = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True)

    df["hour"] = df["start_dt"].dt.hour
    df["day_of_week"] = df["start_dt"].dt.dayofweek          # 0=Mon ... 6=Sun
    df["month"] = df["start_dt"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = df["hour"].isin([5, 6, 7, 19, 20, 21, 22]).astype(int)

    return df


def engineer_target_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """Resolution time in minutes = closed_datetime - start_datetime."""
    df["resolution_minutes"] = (
        (df["closed_dt"] - df["start_dt"]).dt.total_seconds() / 60
    )
    # Drop physically impossible values (negative or zero duration)
    df.loc[df["resolution_minutes"] <= 0, "resolution_minutes"] = np.nan
    return df


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing categorical values with a clear 'unknown' marker."""
    cat_cols = [
        "event_type", "event_cause", "corridor", "zone",
        "police_station", "veh_type", "priority"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip()
            df[col] = df[col].replace("", "unknown")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns useful for modeling + targets.
    These are fields a dispatcher would realistically know
    at the moment an event is reported (no data leakage).
    """
    feature_cols = [
        "event_type", "event_cause", "corridor", "zone",
        "police_station", "veh_type",
        "hour", "day_of_week", "month", "is_weekend", "is_peak_hour",
    ]
    target_cols = ["priority", "requires_road_closure", "resolution_minutes"]

    keep = feature_cols + target_cols
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def main():
    df = load_data(RAW_PATH)
    df = engineer_time_features(df)
    df = engineer_target_resolution(df)
    df = clean_categoricals(df)
    df = select_features(df)

    # Drop rows where priority target itself is missing (only ~2 rows)
    df = df.dropna(subset=["priority"])
    df = df[df["priority"].isin(["High", "Low"])]

    print(f"\nFinal processed shape: {df.shape}")
    print(f"Priority distribution:\n{df['priority'].value_counts()}")
    print(f"Road closure distribution:\n{df['requires_road_closure'].value_counts()}")
    print(f"Resolution time available for: {df['resolution_minutes'].notna().sum()} rows")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {OUT_PATH}")


if __name__ == "__main__":
    main()
