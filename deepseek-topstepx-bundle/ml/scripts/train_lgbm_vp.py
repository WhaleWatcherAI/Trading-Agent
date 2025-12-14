#!/usr/bin/env python3
"""
Train a LightGBM (or fallback GradientBoosting) classifier to predict 20-bar
direction using volume-profile-style features on 5m SPY data.

Features (per bar):
- Rolling volume profile (lookback=200 bars, rows=100, VA=70%):
    * poc, vah, val
    * dist_to_poc/val/vah (ticks)
    * pos_in_va (relative position 0-1)
    * vaw (vah-val)
- Price/volatility:
    * returns last 3/5/10 bars
    * range/close, body/close, upper/lower wicks
    * ATR(14), range/ATR
    * volume z-score (20)
- Time of day encodings (sin/cos minutes-in-day)

Label: forward 20-bar return > 0 => 1 else 0.

Train/val split: time-based 80/20.
Outputs:
- AUC/accuracy/F1 on test
- Feature importance (if LightGBM)
- Simple horizon-based trade sim: long if p>0.55, short if p<0.45, hold else flat; exit after 20 bars; reports cumulative return.

Usage:
  DATA_PATH=ml/data/spy_5min_data.parquet python3 ml/scripts/train_lgbm_vp.py
"""

import os
import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    HAS_LGB = False


@dataclass
class VPConfig:
    lookback: int = 200
    rows: int = 100
    va_pct: float = 0.70
    tick_size: float = 0.01


def volume_profile(prices: pd.Series, vols: pd.Series, cfg: VPConfig) -> Tuple[float, float, float]:
    """Compute POC/VAH/VAL on a window using proportional allocation."""
    if len(prices) < 20:
        return np.nan, np.nan, np.nan
    lo, hi = prices.min(), prices.max()
    if hi <= lo:
        return np.nan, np.nan, np.nan
    # bins sized by tick or capped by rows
    bins = max(int((hi - lo) / cfg.tick_size) + 1, cfg.rows)
    edges = np.linspace(lo, hi, bins + 1)
    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")
    df = pd.DataFrame({"price": prices, "vol": vols})
    df["bin"] = pd.cut(df["price"], intervals, include_lowest=True)
    agg = df.groupby("bin", observed=False)["vol"].sum().reset_index()
    agg["mid"] = agg["bin"].apply(lambda x: (x.left + x.right) / 2 if pd.notnull(x) else np.nan)
    agg = agg.dropna(subset=["mid"])
    if agg.empty:
        return np.nan, np.nan, np.nan
    poc_row = agg.loc[agg["vol"].idxmax()]
    poc = poc_row["mid"]
    total = agg["vol"].sum()
    target = total * cfg.va_pct
    captured = poc_row["vol"]
    value_area = {agg["mid"].idxmax()}
    while captured < target and len(value_area) < len(agg):
        mn = min(value_area)
        mx = max(value_area)
        up_idx = mx + 1 if mx + 1 < len(agg) else None
        dn_idx = mn - 1 if mn - 1 >= 0 else None
        up_vol = agg.iloc[up_idx]["vol"] if up_idx is not None else -1
        dn_vol = agg.iloc[dn_idx]["vol"] if dn_idx is not None else -1
        if up_vol >= dn_vol and up_idx is not None:
            value_area.add(up_idx)
            captured += up_vol
        elif dn_idx is not None:
            value_area.add(dn_idx)
            captured += dn_vol
        else:
            break
    vah = agg.loc[max(value_area), "mid"]
    val = agg.loc[min(value_area), "mid"]
    return poc, vah, val


def build_features(df: pd.DataFrame, cfg: VPConfig) -> pd.DataFrame:
    df = df.copy()
    closes = df["close"].to_numpy()
    vols = df["volume"].to_numpy()
    poc_list = np.full(len(df), np.nan)
    vah_list = np.full(len(df), np.nan)
    val_list = np.full(len(df), np.nan)
    rows = cfg.rows
    for i in range(len(df)):
        start = max(0, i - cfg.lookback)
        window_close = closes[start : i + 1]
        window_vol = vols[start : i + 1]
        if len(window_close) < 20:
            continue
        lo, hi = window_close.min(), window_close.max()
        if hi <= lo:
            continue
        bins = max(int((hi - lo) / cfg.tick_size) + 1, rows)
        edges = np.linspace(lo, hi, bins + 1)
        hist, _ = np.histogram(window_close, bins=edges, weights=window_vol)
        if hist.sum() == 0:
            continue
        poc_idx = int(np.argmax(hist))
        mids = (edges[:-1] + edges[1:]) / 2
        poc_price = mids[poc_idx]
        target = hist.sum() * cfg.va_pct
        captured = hist[poc_idx]
        value_area = {poc_idx}
        while captured < target and len(value_area) < len(hist):
            mn = min(value_area)
            mx = max(value_area)
            up_idx = mx + 1 if mx + 1 < len(hist) else None
            dn_idx = mn - 1 if mn - 1 >= 0 else None
            up_vol = hist[up_idx] if up_idx is not None else -1
            dn_vol = hist[dn_idx] if dn_idx is not None else -1
            if up_vol >= dn_vol and up_idx is not None:
                value_area.add(up_idx)
                captured += up_vol
            elif dn_idx is not None:
                value_area.add(dn_idx)
                captured += dn_vol
            else:
                break
        vah_price = mids[max(value_area)]
        val_price = mids[min(value_area)]
        poc_list[i] = poc_price
        vah_list[i] = vah_price
        val_list[i] = val_price
    df["poc"] = poc_list
    df["vah"] = vah_list
    df["val"] = val_list
    df["vaw"] = df["vah"] - df["val"]
    df["dist_poc"] = (df["close"] - df["poc"]) / cfg.tick_size
    df["dist_vah"] = (df["vah"] - df["close"]) / cfg.tick_size
    df["dist_val"] = (df["close"] - df["val"]) / cfg.tick_size
    df["pos_in_va"] = (df["close"] - df["val"]) / (df["vaw"] + 1e-8)

    # returns
    for n in [3, 5, 10]:
        df[f"ret_{n}"] = df["close"].pct_change(n)
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range_rel"] = df["range"] / df["close"]
    df["body_rel"] = df["body"] / df["close"]

    # ATR14
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()))
    df["atr14"] = tr.rolling(14, min_periods=1).mean()
    df["range_atr"] = df["range"] / (df["atr14"] + 1e-8)
    # mid-body deviation as regime-ish proxy
    df["mid_dev"] = (df["close"] - (df["high"] + df["low"]) / 2) / (df["atr14"] + 1e-8)

    # volume zscore
    vol_mean = df["volume"].rolling(20, min_periods=1).mean()
    vol_std = df["volume"].rolling(20, min_periods=1).std().replace(0, np.nan)
    df["vol_z"] = (df["volume"] - vol_mean) / (vol_std + 1e-8)
    # longer vol z
    vol_mean2 = df["volume"].rolling(60, min_periods=1).mean()
    vol_std2 = df["volume"].rolling(60, min_periods=1).std().replace(0, np.nan)
    df["vol_z_long"] = (df["volume"] - vol_mean2) / (vol_std2 + 1e-8)

    # time encodings
    ts = df.index.tz_convert("America/New_York") if df.index.tz else df.index
    minutes = ts.hour * 60 + ts.minute
    df["tod_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
    df["tod_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))
    # day-of-week
    df["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)

    # forward label
    horizon = 20
    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = (fwd_ret > 0).astype(int)
    df = df.dropna()
    return df


def gate_backtest(df: pd.DataFrame, prob_col: str, upper: float = 0.55, lower: float = 0.45, horizon: int = 20) -> float:
    """Simple horizon-based strategy: long if p>upper, short if p<lower, flat otherwise."""
    returns = []
    for i in range(len(df) - horizon):
        p = df.iloc[i][prob_col]
        if p > upper:
            ret = (df["close"].iloc[i + horizon] - df["close"].iloc[i]) / df["close"].iloc[i]
        elif p < lower:
            ret = (df["close"].iloc[i] - df["close"].iloc[i + horizon]) / df["close"].iloc[i]
        else:
            ret = 0.0
        returns.append(ret)
    return float(np.nansum(returns))


def main():
    data_path = os.environ.get("DATA_PATH", "ml/data/spy_5min_data.parquet")
    cfg = VPConfig(
        lookback=int(os.environ.get("LOOKBACK", 200)),
        rows=int(os.environ.get("ROWS", 100)),
        va_pct=float(os.environ.get("VA_PCT", 70.0)) / 100.0,
        tick_size=float(os.environ.get("TICK_SIZE", 0.01)),
    )

    df = pd.read_parquet(data_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df_feat = build_features(df, cfg)
    feature_cols = [
        "poc",
        "vah",
        "val",
        "vaw",
        "dist_poc",
        "dist_vah",
        "dist_val",
        "pos_in_va",
        "ret_3",
        "ret_5",
        "ret_10",
        "range_rel",
        "body_rel",
        "upper_wick",
        "lower_wick",
        "range_atr",
        "vol_z",
        "vol_z_long",
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
        "mid_dev",
    ]
    df_feat = df_feat.dropna(subset=feature_cols + ["label"])

    X = df_feat[feature_cols]
    y = df_feat["label"]

    # time-based split (80/20)
    split_idx = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=96,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
        )
    else:
        model = make_pipeline(StandardScaler(), GradientBoostingClassifier())

    model.fit(X_train, y_train)
    if HAS_LGB:
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model[-1], "predict_proba") else model.predict(X_test)
    y_pred = (proba > 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # simple time-based CV (4 folds) on train portion
    if HAS_LGB:
        kf = KFold(n_splits=4, shuffle=False)
        aucs = []
        for tr, va in kf.split(X_train):
            m = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
            )
            m.fit(X_train.iloc[tr], y_train.iloc[tr])
            p = m.predict_proba(X_train.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[va], p))
        print(f"Time CV AUC (train portion): mean {np.mean(aucs):.4f}, std {np.std(aucs):.4f}")

    # feature importance
    if HAS_LGB:
        fi = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
        print("\nTop 10 feature importances:")
        for name, val in fi[:10]:
            print(f"  {name}: {val:.1f}")

    # Gated strategy
    test_df = df_feat.iloc[split_idx:].copy()
    test_df["proba"] = proba
    gated_ret = gate_backtest(test_df, "proba", upper=0.55, lower=0.45, horizon=20)
    print(f"\nGated strategy cumulative return over test (20-bar exits): {gated_ret*100:.2f}%")


if __name__ == "__main__":
    main()
