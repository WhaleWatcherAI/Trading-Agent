#!/usr/bin/env python3
"""
Train LightGBM meta-label models for 5m and 30m TP-before-SL probability.
Uses time-based splits to avoid leakage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "ml" / "data" / "meta_label.parquet"
MODELS_DIR = ROOT / "ml" / "models"
FEATURES_JSON = MODELS_DIR / "features.json"
METRICS_JSON = MODELS_DIR / "metrics.json"

MODEL_5M = MODELS_DIR / "meta_label_5m.txt"
MODEL_30M = MODELS_DIR / "meta_label_30m.txt"


def load_features(df: pd.DataFrame) -> List[str]:
    if FEATURES_JSON.exists():
        try:
            data = json.loads(FEATURES_JSON.read_text())
            cols = data.get("feature_columns")
            if cols:
                return [c for c in cols if c in df.columns]
        except Exception:
            pass
    return [c for c in df.columns if c not in ("symbol", "entry_time", "win_5m", "win_30m")]


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    split_idx = max(1, int(len(df_sorted) * (1 - test_frac)))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def calc_pr_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best = 0.5
    best_f1 = -1
    for t in np.linspace(0.1, 0.9, 17):
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best = t
    return best, calc_pr_metrics(y_true, probs, best)


def run_time_cv(df: pd.DataFrame, features: List[str], label: str, splits: List[float]) -> List[float]:
    aucs = []
    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    for frac in splits:
        split_idx = max(1, int(len(df_sorted) * frac))
        train = df_sorted.iloc[:split_idx]
        valid = df_sorted.iloc[split_idx:]
        if len(valid) < 20 or len(train) < 50:
            continue
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=400,
            min_data_in_leaf=25,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(train[features], train[label])
        probs = model.predict_proba(valid[features])[:, 1]
        aucs.append(roc_auc_score(valid[label], probs))
    return aucs


def train_model(df: pd.DataFrame, features: List[str], label: str, model_path: Path) -> Dict[str, float]:
    train_df, test_df = time_split(df, test_frac=0.2)
    if len(train_df) < 50 or len(test_df) < 10:
        raise RuntimeError(f"Dataset too small for label {label} (train {len(train_df)}, test {len(test_df)})")

    X_train = train_df[features]
    y_train = train_df[label]
    X_test = test_df[features]
    y_test = test_df[label]

    # hold out last 20% of train for early stopping
    valid_split = max(1, int(len(train_df) * 0.8))
    X_tr, X_val = X_train.iloc[:valid_split], X_train.iloc[valid_split:]
    y_tr, y_val = y_train.iloc[:valid_split], y_train.iloc[valid_split:]

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 25,
        "verbose": -1,
    }

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    booster.save_model(str(model_path))

    test_probs = booster.predict(X_test, num_iteration=booster.best_iteration)
    auc = roc_auc_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)
    pr_05 = calc_pr_metrics(y_test, test_probs, 0.5)
    best_t, pr_best = best_f1_threshold(y_test, test_probs)
    cv_aucs = run_time_cv(df, features, label, splits=[0.5, 0.6, 0.7, 0.8])

    metrics = {
        "label": label,
        "auc": float(auc),
        "brier": float(brier),
        "precision@0.5": pr_05["precision"],
        "recall@0.5": pr_05["recall"],
        "f1@0.5": pr_05["f1"],
        "best_f1_threshold": float(best_t),
        "precision@best": pr_best["precision"],
        "recall@best": pr_best["recall"],
        "f1@best": pr_best["f1"],
        "cv_auc_mean": float(np.mean(cv_aucs)) if cv_aucs else None,
        "cv_auc_std": float(np.std(cv_aucs)) if cv_aucs else None,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": len(features),
    }
    return metrics


def main():
    if not DATA_PATH.exists():
        print("Dataset not found. Run scripts/build_dataset.py first.")
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    if df.empty:
        print("Dataset empty; aborting.")
        sys.exit(1)

    feature_cols = load_features(df)
    df = df.sort_values("entry_time").reset_index(drop=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_all: Dict[str, Dict[str, float]] = {}

    print(f"Training with {len(df)} rows, {len(feature_cols)} features")
    metrics_all["win_5m"] = train_model(df, feature_cols, "win_5m", MODEL_5M)
    metrics_all["win_30m"] = train_model(df, feature_cols, "win_30m", MODEL_30M)

    METRICS_JSON.write_text(json.dumps(metrics_all, indent=2))
    print("Saved models and metrics to", MODELS_DIR)
    print(json.dumps(metrics_all, indent=2))


if __name__ == "__main__":
    main()
