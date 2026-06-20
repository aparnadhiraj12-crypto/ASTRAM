# What changed in this package

This is your `ml_project` folder with the improved ML pipeline already applied.
Replace your existing project folder with this one (or copy these files over,
overwriting what's there).

## Files changed
- **app.py** — now computes `cause_speed` and `time_unspecified` features
  server-side before calling the models. No changes needed in forecast.html;
  the API request/response shape is unchanged.
- **models/*.pkl, feature_columns.json, metrics.json** — retrained on cleaned
  data with the new features. See metrics.json for full before/after numbers.

## Files added
- **clean_data.py** — cleans data/processed.csv into data/processed_clean.csv
  (fixes "Debris"/"debris" duplication, buckets rare causes, adds cause_speed
  and time_unspecified features). Run this first if you ever retrain.
- **train_improved.py** — trains all 3 models on the cleaned data, saves into
  models/. Run after clean_data.py.
- **data/processed_clean.csv** — pre-generated, already used to train the
  models included here. You don't need to regenerate this unless your source
  data changes.

## If you want to retrain later (e.g. with more data)
```bash
python clean_data.py
python train_improved.py
```
Both read/write relative to the ml_project/ folder, so run them from there.

## What actually improved
- Priority model: 67.65% -> 68.26% accuracy (small real gain)
- Closure model: ~82% accuracy, unchanged (no regression)
- Resolution time: MAE barely moved (still dominated by genuine multi-week
  pothole/construction cases), but median error is now reported separately:
  ~52 minutes for the typical event -- much more representative of real
  performance than the old MAE-only metric suggested.

The biggest remaining lever for further improvement is more/better source
data -- specifically, several event causes (public_event, vip_movement, etc.)
have no recorded resolution time at all in the raw dataset, which limits what
any model can learn about them.
