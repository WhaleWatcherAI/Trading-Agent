# Meta-Label LightGBM (TopstepX)

Lightweight ML scorer for TopstepX agents. It trains a LightGBM meta-label to predict the chance a proposed entry hits TP before SL within 5m and 30m. Outputs probabilities only; trading logic is unchanged for now.

## Setup (Mac M1 tested)
```bash
cd qwen-topstepx-bundle/ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If LightGBM fails to build:
```bash
brew install libomp               # preferred OpenMP path
pip install -r requirements.txt   # retry
# fallback without OpenMP
pip install lightgbm --config-settings=cmake.define.USE_OPENMP=OFF
```

## Data locations
- Trade decisions/outcomes: `../trading-db/decisions.jsonl` and `../trading-db/outcomes.jsonl`
- Market snapshots (from TS logger): `../ml/data/snapshots.jsonl`
- Built dataset: `../ml/data/meta_label.parquet`
- Models/metadata: `../ml/models/meta_label_5m.txt`, `meta_label_30m.txt`, `features.json`, `metrics.json`

## Workflows
- Build dataset:
  ```bash
  python3 scripts/build_dataset.py
  ```
- Train models (time-based split + CV):
  ```bash
  python3 scripts/train_meta_label.py
  ```
- Inference (manual):
  ```bash
  echo '{\"symbol\":\"NQZ5\",\"timestamp\":\"2024-01-01T12:00:00Z\",\"features\":{\"dist_to_poc_ticks\":4}}' \
  | python3 scripts/predict_meta_label.py
  ```

## Integration
- TS agents log per-minute snapshots to `ml/data/snapshots.jsonl` and optionally call `predict_meta_label.py` via `lib/mlMetaLabelService.ts`.
- Scores are attached to `FuturesMarketData.mlScores` and surfaced to prompts; no routing/gating changes yet.
