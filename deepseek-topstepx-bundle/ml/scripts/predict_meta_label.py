#!/usr/bin/env python3
"""
LightGBM inference for meta-label probabilities.
Reads a single snapshot JSON (stdin or file) with structure:
{ "symbol": "...", "timestamp": "...", "features": { ... } }
Outputs: {"p_win_5m": float, "p_win_30m": float}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "ml" / "models"
MODEL_5M = MODELS_DIR / "meta_label_5m.txt"
MODEL_30M = MODELS_DIR / "meta_label_30m.txt"
FEATURES_JSON = MODELS_DIR / "features.json"


def load_features() -> List[str]:
    if FEATURES_JSON.exists():
        try:
            data = json.loads(FEATURES_JSON.read_text())
            cols = data.get("feature_columns") or []
            return cols
        except Exception:
            pass
    return []


def read_snapshot(path: str | None) -> Dict:
    if path:
        return json.loads(Path(path).read_text())
    # stdin
    payload = sys.stdin.read()
    if not payload.strip():
        raise ValueError("No input provided")
    return json.loads(payload)


def prepare_row(features: List[str], snapshot: Dict) -> np.ndarray:
    feats = snapshot.get("features", {})
    row = []
    for col in features:
        val = feats.get(col, np.nan)
        try:
            row.append(float(val))
        except Exception:
            row.append(np.nan)
    return np.array(row, dtype=float)


def predict(path: str | None) -> Dict[str, float]:
    features = load_features()
    if not features:
        raise RuntimeError("features.json missing or empty; run training first.")

    snap = read_snapshot(path)
    row = prepare_row(features, snap)

    def _predict(model_path: Path) -> float:
        if not model_path.exists():
            return float("nan")
        booster = lgb.Booster(model_file=str(model_path))
        return float(booster.predict(row.reshape(1, -1))[0])

    return {
        "p_win_5m": _predict(MODEL_5M),
        "p_win_30m": _predict(MODEL_30M),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", help="Optional path to snapshot JSON (defaults to stdin)")
    args = parser.parse_args()

    try:
        scores = predict(args.json_path)
    except Exception as exc:
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(1)

    sys.stdout.write(json.dumps(scores))


if __name__ == "__main__":
    main()
