"""
clean_data.py
─────────────
Cleans processed.csv before training. Fixes (in order of impact):

1. Normalizes category strings (lowercase + strip) so "Debris" and "debris"
   collapse into one category instead of fragmenting the model's signal.
2. Buckets ultra-rare event_cause values (<15 occurrences) into "other_rare"
   so the model has enough examples to learn from instead of pure noise.
3. Fills missing hour/day_of_week/month for "planned" events (these are
   genuinely missing because planned events are logged by date range, not
   a single hour) using sensible defaults rather than dropping the rows.
4. Adds a `cause_speed` feature: buckets event_cause into 'fast' or 'slow'
   based on each cause's median historical resolution time. This is the
   single highest-leverage fix for the resolution-time regressor, because
   the raw data mixes two very different timescales (a vehicle breakdown
   resolves in ~40 min; a pothole repair can take ~9 days) into one target.
   Without this signal, the model has to infer "this might take weeks" purely
   from a one-hot category, which is harder to learn cleanly.

Run (from the ml_project/ directory):
    python clean_data.py
Outputs:
    data/processed_clean.csv
"""

import pandas as pd
import numpy as np

IN_PATH = "data/processed.csv"
OUT_PATH = "data/processed_clean.csv"

RARE_CAUSE_THRESHOLD = 15  # causes with fewer rows than this get bucketed


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ["event_type", "event_cause", "zone", "veh_type", "corridor", "police_station"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": "unknown"})
    return df


def bucket_rare_causes(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["event_cause"].value_counts()
    rare = counts[counts < RARE_CAUSE_THRESHOLD].index
    print(f"Bucketing {len(rare)} rare event_cause values into 'other_rare': {list(rare)}")
    df["event_cause"] = df["event_cause"].apply(lambda c: "other_rare" if c in rare else c)
    return df


def fill_missing_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    n_missing = df["hour"].isnull().sum()
    print(f"Filling {n_missing} rows with missing hour/day_of_week/month "
          f"(mostly 'planned' events with no specific start hour logged)")
    # Use the dataset's overall median hour/day as a neutral default,
    # and flag these rows so the model can learn "this was a planned/unspecified-time event"
    df["time_unspecified"] = df["hour"].isnull().astype(int)
    df["hour"] = df["hour"].fillna(df["hour"].median())
    df["day_of_week"] = df["day_of_week"].fillna(df["day_of_week"].median())
    df["month"] = df["month"].fillna(df["month"].median())
    return df


def add_cause_speed_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each event_cause as 'fast' or 'slow' resolving, based on median
    resolution_minutes for that cause (computed only from rows where it's known).
    This becomes a feature available at inference time (it only depends on
    event_cause, which is always known up front).
    """
    medians = df.dropna(subset=["resolution_minutes"]).groupby("event_cause")["resolution_minutes"].median()
    overall_median = df["resolution_minutes"].median()

    # Causes with no historical resolution data fall back to the overall median
    def classify(cause):
        m = medians.get(cause, overall_median)
        return "slow" if m > 1440 else "fast"  # slower than 1 day = 'slow'

    speed_map = {cause: classify(cause) for cause in df["event_cause"].unique()}
    print("\ncause_speed assignment:")
    for cause, speed in sorted(speed_map.items(), key=lambda x: x[1]):
        med = medians.get(cause, np.nan)
        print(f"  {cause:20s} -> {speed:5s} (median: {med:.1f} min)" if not np.isnan(med) else f"  {cause:20s} -> {speed:5s} (no data, used overall median)")

    df["cause_speed"] = df["event_cause"].map(speed_map)
    return df


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded: {df.shape}")

    df = normalize_categories(df)
    df = bucket_rare_causes(df)
    df = fill_missing_time_fields(df)
    df = add_cause_speed_feature(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned data: {df.shape} -> {OUT_PATH}")
    print("\nFinal event_cause distribution:")
    print(df["event_cause"].value_counts())


if __name__ == "__main__":
    main()
