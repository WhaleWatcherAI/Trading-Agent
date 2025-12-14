#!/usr/bin/env python3
"""
Train a LightGBM (or fallback GradientBoosting) classifier on 5m data to predict
20-bar direction, then gate the volume-profile rule-based strategy on the test slice
with model probabilities. Supports up to N concurrent positions.

Data: 5m parquet with columns open/high/low/close/volume, index as timestamps.

Env overrides:
  DATA_PATH=ml/data/spy_5min_data.parquet
  LOOKBACK=200        # VP rolling bars
  ROWS=100            # VP bins
  VA_PCT=70           # value area percent
  TICK_SIZE=0.01
  TICK_VALUE=0.01
  COMMISSION_PER_SIDE=0
  CONTRACTS=1
  INITIAL_EQUITY=100000
  MAX_POSITIONS=5
  MODEL_UPPER=0.6     # gate long if p>upper
  MODEL_LOWER=0.4     # gate short if p<lower
"""

import os
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
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
    tick_value: float = 0.01
    commission_per_side: float = 0.0
    contracts: int = 1
    initial_equity: float = 100_000.0
    max_positions: int = 5
    model_upper: float = 0.6
    model_lower: float = 0.4
    use_atr_stop: bool = True
    atr_mult: float = 1.5
    max_hold_bars: int = 0  # 0 = none


def build_features(df: pd.DataFrame, cfg: VPConfig):
    df = df.copy()
    closes = df["close"].to_numpy()
    vols = df["volume"].to_numpy()
    poc = np.full(len(df), np.nan)
    vah = np.full(len(df), np.nan)
    val = np.full(len(df), np.nan)
    for i in range(len(df)):
        s = max(0, i - cfg.lookback)
        wc = closes[s : i + 1]
        wv = vols[s : i + 1]
        if len(wc) < 20:
            continue
        lo, hi = wc.min(), wc.max()
        if hi <= lo:
            continue
        bins = max(int((hi - lo) / cfg.tick_size) + 1, cfg.rows)
        edges = np.linspace(lo, hi, bins + 1)
        hist, _ = np.histogram(wc, bins=edges, weights=wv)
        if hist.sum() == 0:
            continue
        mids = (edges[:-1] + edges[1:]) / 2
        poc_idx = int(np.argmax(hist))
        poc_price = mids[poc_idx]
        target = hist.sum() * cfg.va_pct
        captured = hist[poc_idx]
        va = {poc_idx}
        while captured < target and len(va) < len(hist):
            mn, mx = min(va), max(va)
            up = mx + 1 if mx + 1 < len(hist) else None
            dn = mn - 1 if mn - 1 >= 0 else None
            upv = hist[up] if up is not None else -1
            dnv = hist[dn] if dn is not None else -1
            if upv >= dnv and up is not None:
                va.add(up)
                captured += upv
            elif dn is not None:
                va.add(dn)
                captured += dnv
            else:
                break
        vah_price = mids[max(va)]
        val_price = mids[min(va)]
        poc[i] = poc_price
        vah[i] = vah_price
        val[i] = val_price
    df["poc"] = poc
    df["vah"] = vah
    df["val"] = val
    df["vaw"] = df["vah"] - df["val"]
    df["dist_poc"] = (df["close"] - df["poc"]) / cfg.tick_size
    df["dist_vah"] = (df["vah"] - df["close"]) / cfg.tick_size
    df["dist_val"] = (df["close"] - df["val"]) / cfg.tick_size
    df["pos_in_va"] = (df["close"] - df["val"]) / (df["vaw"] + 1e-8)
    for n in [3, 5, 10]:
        df[f"ret_{n}"] = df["close"].pct_change(n)
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range_rel"] = df["range"] / df["close"]
    df["body_rel"] = df["body"] / df["close"]
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()),
    )
    df["atr14"] = tr.rolling(14, min_periods=1).mean()
    df["range_atr"] = df["range"] / (df["atr14"] + 1e-8)
    vol_mean = df["volume"].rolling(20, min_periods=1).mean()
    vol_std = df["volume"].rolling(20, min_periods=1).std().replace(0, np.nan)
    df["vol_z"] = (df["volume"] - vol_mean) / (vol_std + 1e-8)
    ts = df.index.tz_convert("America/New_York") if df.index.tz else df.index
    minutes = ts.hour * 60 + ts.minute
    df["tod_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
    df["tod_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))
    horizon = 20
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = (fwd > 0).astype(int)
    df = df.dropna()
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
        "tod_sin",
        "tod_cos",
    ]
    return df, feature_cols


def gate_backtest(df: pd.DataFrame, cfg: VPConfig) -> dict:
    """Rule-based VP strategy with model gating and max_positions."""
    equity = cfg.initial_equity
    trades: List[dict] = []
    open_positions = []
    for i in range(len(df)):
        price = df["close"].iloc[i]
        poc = df["poc"].iloc[i]
        vah = df["vah"].iloc[i]
        val = df["val"].iloc[i]
        vaw = vah - val
        if np.isnan(poc) or vaw <= 0:
            continue
        proba = df["proba"].iloc[i]
        # Determine signals
        near_val = price <= val + vaw * 0.1 and price >= val
        near_vah = price >= vah - vaw * 0.1 and price <= vah
        cont_long = price > vah * (1 + 0.05)
        cont_short = price < val * (1 - 0.05)

        # Entries (respect gating and max positions)
        if len(open_positions) < cfg.max_positions:
            if near_val and proba > cfg.model_upper:
                open_positions.append(
                    {
                        "side": 1,
                        "type": "mr_long",
                        "entry_price": price,
                        "entry_idx": i,
                        "stop": price - df["atr14"].iloc[i] * cfg.atr_mult if cfg.use_atr_stop else None,
                    }
                )
            elif near_vah and proba < cfg.model_lower:
                open_positions.append(
                    {
                        "side": -1,
                        "type": "mr_short",
                        "entry_price": price,
                        "entry_idx": i,
                        "stop": price + df["atr14"].iloc[i] * cfg.atr_mult if cfg.use_atr_stop else None,
                    }
                )
            elif cont_long and proba > cfg.model_upper:
                open_positions.append(
                    {
                        "side": 1,
                        "type": "bo_long",
                        "entry_price": price,
                        "entry_idx": i,
                        "stop": price - df["atr14"].iloc[i] * cfg.atr_mult if cfg.use_atr_stop else None,
                    }
                )
            elif cont_short and proba < cfg.model_lower:
                open_positions.append(
                    {
                        "side": -1,
                        "type": "bo_short",
                        "entry_price": price,
                        "entry_idx": i,
                        "stop": price + df["atr14"].iloc[i] * cfg.atr_mult if cfg.use_atr_stop else None,
                    }
                )

        # Exits
        still_open = []
        for pos in open_positions:
            exit_reason = None
            if pos["type"].startswith("mr"):
                if pos["side"] == 1 and price >= poc:
                    exit_reason = "target_poc"
                if pos["side"] == -1 and price <= poc:
                    exit_reason = "target_poc"
            if pos["type"] == "bo_long" and price <= vah:
                exit_reason = exit_reason or "reenter_va"
            if pos["type"] == "bo_short" and price >= val:
                exit_reason = exit_reason or "reenter_va"
            if cfg.max_hold_bars > 0 and (i - pos["entry_idx"]) >= cfg.max_hold_bars:
                exit_reason = exit_reason or "time"
            if cfg.use_atr_stop and pos["stop"] is not None:
                if pos["side"] == 1 and price <= pos["stop"]:
                    exit_reason = exit_reason or "atr_stop"
                if pos["side"] == -1 and price >= pos["stop"]:
                    exit_reason = exit_reason or "atr_stop"

            if exit_reason:
                ticks = (price - pos["entry_price"]) / cfg.tick_size
                ticks = ticks if pos["side"] == 1 else -ticks
                gross = ticks * cfg.tick_value * cfg.contracts
                fees = cfg.commission_per_side * 2 * cfg.contracts
                net = gross - fees
                equity += net
                trades.append(
                    {
                        "entry_idx": pos["entry_idx"],
                        "exit_idx": i,
                        "side": "long" if pos["side"] == 1 else "short",
                        "type": pos["type"],
                        "entry_price": pos["entry_price"],
                        "exit_price": price,
                        "ticks": ticks,
                        "gross": gross,
                        "net": net,
                        "reason": exit_reason,
                    }
                )
            else:
                still_open.append(pos)
        open_positions = still_open

    return {
        "equity": equity,
        "trades": trades,
    }


def main():
    cfg = VPConfig(
        lookback=int(os.environ.get("LOOKBACK", 200)),
        rows=int(os.environ.get("ROWS", 100)),
        va_pct=float(os.environ.get("VA_PCT", 70)) / 100.0,
        tick_size=float(os.environ.get("TICK_SIZE", 0.01)),
        tick_value=float(os.environ.get("TICK_VALUE", 0.01)),
        commission_per_side=float(os.environ.get("COMMISSION_PER_SIDE", 0)),
        contracts=int(os.environ.get("CONTRACTS", 1)),
        initial_equity=float(os.environ.get("INITIAL_EQUITY", 100000)),
        max_positions=int(os.environ.get("MAX_POSITIONS", 5)),
        model_upper=float(os.environ.get("MODEL_UPPER", 0.6)),
        model_lower=float(os.environ.get("MODEL_LOWER", 0.4)),
        use_atr_stop=os.environ.get("USE_ATR_STOP", "true").lower() == "true",
        atr_mult=float(os.environ.get("ATR_MULT", 1.5)),
        max_hold_bars=int(os.environ.get("MAX_HOLD_BARS", 0)),
    )
    data_path = os.environ.get("DATA_PATH", "ml/data/spy_5min_data.parquet")
    df = pd.read_parquet(data_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df_feat, feature_cols = build_features(df, cfg)
    df_feat = df_feat.dropna(subset=feature_cols + ["label"])

    split = int(len(df_feat) * 0.8)
    train = df_feat.iloc[:split]
    test = df_feat.iloc[split:]
    X_train, y_train = train[feature_cols], train["label"]
    X_test, y_test = test[feature_cols], test["label"]

    if HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
        )
    else:
        model = make_pipeline(StandardScaler(), GradientBoostingClassifier())

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    test = test.copy()
    test["proba"] = proba

    res = gate_backtest(test, cfg)
    trades = res["trades"]
    wins = sum(1 for t in trades if t["net"] > 0)
    losses = len(trades) - wins
    print(f"Trades: {len(trades)} | Wins: {wins} | Losses: {losses}")
    if trades:
        avg_win = np.mean([t["net"] for t in trades if t["net"] > 0]) if wins else 0
        avg_loss = np.mean([t["net"] for t in trades if t["net"] <= 0]) if losses else 0
        print(f"Avg win: {avg_win:.4f} | Avg loss: {avg_loss:.4f}")
        pf = (np.sum([t["net"] for t in trades if t["net"] > 0]) / max(1e-8, -np.sum([t["net"] for t in trades if t["net"] < 0]))) if losses else float("inf")
        print(f"Profit factor: {pf:.2f}")
    print(f"Final equity: ${res['equity']:.2f} (Start: ${cfg.initial_equity:.2f})")


if __name__ == "__main__":
    main()
