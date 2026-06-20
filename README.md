# TrafficSense ML Pipeline

Trains 3 models on Bengaluru traffic event data.

## Setup
```bash
pip install -r requirements.txt
```

## Run order
```bash
python preprocess.py   # cleans data/traffic_events.csv -> data/processed.csv
python train.py        # trains all 3 models -> models/*.pkl
python predict.py       # sanity-check predictions on sample events
```

## Models trained

| Model | Type | Target | Test Accuracy / MAE |
|---|---|---|---|
| `priority_model.pkl` | XGBoost Classifier | High / Low priority | 67.6% accuracy |
| `closure_model.pkl` | XGBoost Classifier | Road closure required | 82.5% accuracy (F1 0.33 on rare "closure" class) |
| `resolution_model.pkl` | XGBoost Regressor | Resolution time (minutes, log-transformed) | MAE ~5,660 min |

## Important notes (read before presenting to judges)

1. **`corridor` and `police_station` are deliberately excluded from the priority
   and closure models.** An early version hit 99.9% accuracy on priority — that
   was a red flag, not a win. Investigation showed `corridor` alone is ~99.8%
   "pure" against the priority label (e.g. every single Bellary Road event is
   labeled High). That means priority looks like it's assigned by a business
   rule tied to location, not actually caused by the event. Training on it
   would just memorize a lookup table, not learn anything generalizable.
   These two columns ARE used for the resolution-time model, where location
   legitimately affects how fast a team can respond.

2. **Road closure accuracy (82.5%) looks better than it is.** Closures are
   rare (~8% of events), so a model that always predicts "no closure" would
   score ~92% accuracy while being useless. Look at the F1 score (0.33) and
   per-class report in `models/metrics.json` for the honest picture — recall
   on the closure class is what actually matters operationally.

3. **Resolution time MAE is large (~94 hours)** because the target itself is
   extremely skewed — pot holes have a 9-day median resolution time vs.
   ~40 minutes for accidents. A single regressor struggles to fit both
   regimes well. This is a known limitation, see "Next steps" below.

## Next steps to improve (good talking points for judges)
- Try a separate "fast resolution" vs "slow resolution" classifier first,
  then a regressor within each bucket
- Add weather/rainfall data — water_logging and pothole resolution times
  likely correlate strongly with monsoon season
- Collect more `requires_road_closure=True` examples or use SMOTE to
  address the class imbalance
- Try ordinal encoding for `event_cause` based on historical severity rank
  instead of one-hot, to reduce dimensionality

## Files
```
data/
  traffic_events.csv   raw input (8,173 events)
  processed.csv         cleaned + feature-engineered
models/
  priority_model.pkl
  priority_label_encoder.pkl
  closure_model.pkl
  resolution_model.pkl
  feature_columns.json  exact feature order expected at inference
  metrics.json           full evaluation report for all 3 models
preprocess.py
train.py
predict.py
```
