#!/usr/bin/env python3
"""
LightGBM Backtest with Volume Profile + CVD
Predicts next 20 bars direction

Features:
- Volume Profile (POC, VAH, VAL) from 1-second bars
- CVD (Cumulative Volume Delta) from 1-second bars
- CVD EMA (smoothed trend) from 1-second bars
- Price action from 1-second bars aggregated to 1-minute

Usage:
    python lgbm_vp_cvd_backtest.py --input ../data/bars_1s.json --prediction-horizon 20
"""

import argparse
import json
import sys
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def calculate_volume_profile(bars_1s: List[Dict], tick_size: float = 0.25) -> Dict[str, Any]:
    """Calculate Volume Profile from 1-second bars."""
    if not bars_1s:
        return {"poc": 0, "vah": 0, "val": 0, "total_volume": 0}

    price_volume = defaultdict(float)

    for bar in bars_1s:
        prices = [bar['o'], bar['h'], bar['l'], bar['c']]
        volume = bar.get('v', 0) or 0
        vol_per_price = volume / 4.0 if volume > 0 else 0.25

        for price in prices:
            rounded = round(price / tick_size) * tick_size
            price_volume[rounded] += vol_per_price

    if not price_volume:
        return {"poc": 0, "vah": 0, "val": 0, "total_volume": 0}

    sorted_prices = sorted(price_volume.keys())
    volumes = [price_volume[p] for p in sorted_prices]
    total_volume = sum(volumes)

    poc_idx = np.argmax(volumes)
    poc = sorted_prices[poc_idx]

    target_volume = total_volume * 0.7
    current_volume = volumes[poc_idx]

    lower_idx = poc_idx
    upper_idx = poc_idx

    while current_volume < target_volume:
        lower_vol = volumes[lower_idx - 1] if lower_idx > 0 else 0
        upper_vol = volumes[upper_idx + 1] if upper_idx < len(volumes) - 1 else 0

        if lower_vol >= upper_vol and lower_idx > 0:
            lower_idx -= 1
            current_volume += lower_vol
        elif upper_idx < len(volumes) - 1:
            upper_idx += 1
            current_volume += upper_vol
        else:
            break

    val = sorted_prices[lower_idx]
    vah = sorted_prices[upper_idx]

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "total_volume": total_volume,
    }


def calculate_cvd_from_1s(bars_1s: List[Dict]) -> Tuple[List[float], List[float]]:
    """Calculate CVD and CVD EMA from 1-second bars."""
    if not bars_1s:
        return [], []

    cvd = 0.0
    cvd_values = []

    for bar in bars_1s:
        volume = bar.get('v', 0) or 0
        open_price = bar['o']
        close_price = bar['c']
        high_price = bar['h']
        low_price = bar['l']

        bar_range = high_price - low_price
        if bar_range > 0:
            close_position = (close_price - low_price) / bar_range
            delta = volume * (2 * close_position - 1)
        else:
            delta = volume if close_price >= open_price else -volume

        cvd += delta
        cvd_values.append(cvd)

    # EMA of CVD
    ema_period = 20
    cvd_ema = []

    if len(cvd_values) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema = sum(cvd_values[:ema_period]) / ema_period
        cvd_ema = [ema] * ema_period

        for i in range(ema_period, len(cvd_values)):
            ema = (cvd_values[i] * multiplier) + (ema * (1 - multiplier))
            cvd_ema.append(ema)
    else:
        cvd_ema = cvd_values.copy()

    return cvd_values, cvd_ema


def aggregate_to_1min(bars_1s: List[Dict]) -> List[Dict]:
    """Aggregate 1-second bars to 1-minute bars."""
    if not bars_1s:
        return []

    minute_bars = []
    current_minute = None
    current_bar = None

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            dt = ts

        minute_ts = dt.replace(second=0, microsecond=0)

        if current_minute != minute_ts:
            if current_bar:
                minute_bars.append(current_bar)

            current_minute = minute_ts
            current_bar = {
                't': minute_ts.isoformat(),
                'o': bar['o'],
                'h': bar['h'],
                'l': bar['l'],
                'c': bar['c'],
                'v': bar.get('v', 0) or 0,
            }
        else:
            current_bar['h'] = max(current_bar['h'], bar['h'])
            current_bar['l'] = min(current_bar['l'], bar['l'])
            current_bar['c'] = bar['c']
            current_bar['v'] += bar.get('v', 0) or 0

    if current_bar:
        minute_bars.append(current_bar)

    return minute_bars


def prepare_features(
    bars_1s: List[Dict],
    lookback_1s: int = 1800,
    prediction_horizon: int = 20,
    step: int = 60,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Prepare features and labels for LightGBM.

    Returns:
        X: Feature matrix
        y: Labels (1 = up, 0 = down)
        metadata: List of dicts with prediction metadata
    """
    print(f"[LGBM] Preparing features...", file=sys.stderr)

    features = []
    labels = []
    metadata = []

    for i in range(lookback_1s, len(bars_1s) - prediction_horizon * 60, step):
        # Get 1s bars for VP and CVD
        window_1s = bars_1s[i - lookback_1s:i]

        # Calculate VP
        vp = calculate_volume_profile(window_1s)

        # Calculate CVD
        cvd_values, cvd_ema = calculate_cvd_from_1s(window_1s)

        # Aggregate to 1-min for price features
        window_1m = aggregate_to_1min(window_1s)
        if len(window_1m) < 20:
            continue

        # Current price info
        current_price = window_1m[-1]['c']
        current_high = window_1m[-1]['h']
        current_low = window_1m[-1]['l']
        current_vol = window_1m[-1]['v']

        # Price features
        prices = [b['c'] for b in window_1m]
        volumes = [b['v'] for b in window_1m]

        # Returns
        returns_1 = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
        returns_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 and prices[-6] != 0 else 0
        returns_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 and prices[-11] != 0 else 0
        returns_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 and prices[-21] != 0 else 0

        # Volatility (std of returns)
        if len(prices) >= 20:
            price_returns = [(prices[j] - prices[j-1]) / prices[j-1] if prices[j-1] != 0 else 0
                           for j in range(1, len(prices))]
            volatility = np.std(price_returns[-20:]) if len(price_returns) >= 20 else 0
        else:
            volatility = 0

        # Volume features
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

        # VP features
        poc = vp['poc']
        vah = vp['vah']
        val = vp['val']

        # Price relative to VP
        poc_dist = (current_price - poc) / current_price if current_price != 0 else 0
        vah_dist = (current_price - vah) / current_price if current_price != 0 else 0
        val_dist = (current_price - val) / current_price if current_price != 0 else 0

        # In value area?
        in_value = 1 if val <= current_price <= vah else 0
        above_value = 1 if current_price > vah else 0
        below_value = 1 if current_price < val else 0

        # CVD features
        if len(cvd_values) >= 20 and len(cvd_ema) >= 20:
            cvd_now = cvd_values[-1]
            cvd_ema_now = cvd_ema[-1]
            cvd_vs_ema = cvd_now - cvd_ema_now
            cvd_trend = 1 if cvd_now > cvd_ema_now else 0

            # CVD momentum
            cvd_change = cvd_values[-1] - cvd_values[-20]
            cvd_ema_change = cvd_ema[-1] - cvd_ema[-20]
        else:
            cvd_now = cvd_values[-1] if cvd_values else 0
            cvd_ema_now = cvd_ema[-1] if cvd_ema else 0
            cvd_vs_ema = 0
            cvd_trend = 0
            cvd_change = 0
            cvd_ema_change = 0

        # Normalize CVD features
        cvd_vs_ema_norm = cvd_vs_ema / (abs(cvd_ema_now) + 1)
        cvd_change_norm = cvd_change / (abs(cvd_now) + 1)

        # Build feature vector
        feat = [
            returns_1,
            returns_5,
            returns_10,
            returns_20,
            volatility,
            vol_ratio,
            poc_dist,
            vah_dist,
            val_dist,
            in_value,
            above_value,
            below_value,
            cvd_vs_ema_norm,
            cvd_trend,
            cvd_change_norm,
        ]

        # Calculate label (future direction)
        future_1s = bars_1s[i:i + prediction_horizon * 60]
        if len(future_1s) >= prediction_horizon * 60:
            future_1m = aggregate_to_1min(future_1s)
            if len(future_1m) >= prediction_horizon:
                future_price = future_1m[prediction_horizon - 1]['c']
                actual_move = future_price - current_price
                label = 1 if actual_move > 0 else 0

                features.append(feat)
                labels.append(label)
                metadata.append({
                    'timestamp': bars_1s[i-1]['t'],
                    'current_price': current_price,
                    'future_price': future_price,
                    'actual_move': actual_move,
                    'entry_idx': i,  # 1s bar index for trade simulation
                })

    X = np.array(features)
    y = np.array(labels)

    print(f"[LGBM] Prepared {len(X)} samples with {X.shape[1]} features", file=sys.stderr)

    return X, y, metadata


def simulate_trade(
    bars_1s: List[Dict],
    entry_idx: int,
    direction: str,
    tick_size: float = 0.25,
    sl_ticks: int = 4,
    tp_ticks: int = 16,
    max_bars: int = 1200,  # 20 minutes max in 1s bars
) -> Dict[str, Any]:
    """
    Simulate a trade with SL/TP.
    Returns outcome: 'win', 'loss', or 'timeout'
    """
    entry_price = bars_1s[entry_idx]['c']

    if direction == 'up':
        sl_price = entry_price - (sl_ticks * tick_size)
        tp_price = entry_price + (tp_ticks * tick_size)
    else:  # down
        sl_price = entry_price + (sl_ticks * tick_size)
        tp_price = entry_price - (tp_ticks * tick_size)

    # Walk forward through bars
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(bars_1s))):
        bar = bars_1s[i]
        high = bar['h']
        low = bar['l']

        if direction == 'up':
            # Check SL first (conservative)
            if low <= sl_price:
                return {
                    'outcome': 'loss',
                    'exit_price': sl_price,
                    'pnl_ticks': -sl_ticks,
                    'bars_held': i - entry_idx,
                }
            # Check TP
            if high >= tp_price:
                return {
                    'outcome': 'win',
                    'exit_price': tp_price,
                    'pnl_ticks': tp_ticks,
                    'bars_held': i - entry_idx,
                }
        else:  # down
            # Check SL first
            if high >= sl_price:
                return {
                    'outcome': 'loss',
                    'exit_price': sl_price,
                    'pnl_ticks': -sl_ticks,
                    'bars_held': i - entry_idx,
                }
            # Check TP
            if low <= tp_price:
                return {
                    'outcome': 'win',
                    'exit_price': tp_price,
                    'pnl_ticks': tp_ticks,
                    'bars_held': i - entry_idx,
                }

    # Timeout - exit at last price
    exit_price = bars_1s[min(entry_idx + max_bars - 1, len(bars_1s) - 1)]['c']
    pnl = exit_price - entry_price if direction == 'up' else entry_price - exit_price
    pnl_ticks = pnl / tick_size

    return {
        'outcome': 'timeout',
        'exit_price': exit_price,
        'pnl_ticks': pnl_ticks,
        'bars_held': max_bars,
    }


def run_backtest(
    bars_1s: List[Dict],
    lookback_1s: int = 1800,
    prediction_horizon: int = 20,
    step: int = 60,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    sl_ticks: int = 4,
    tp_ticks: int = 16,
    tick_size: float = 0.25,
) -> Dict[str, Any]:
    """Run LightGBM backtest with train/val/holdout split and SL/TP simulation."""

    print(f"[LGBM] Total 1s bars: {len(bars_1s)}", file=sys.stderr)

    # Prepare features
    X, y, metadata = prepare_features(
        bars_1s=bars_1s,
        lookback_1s=lookback_1s,
        prediction_horizon=prediction_horizon,
        step=step,
    )

    if len(X) < 100:
        print("[LGBM] Not enough samples for training", file=sys.stderr)
        return {"error": "Not enough samples"}

    # Time-based split (no shuffling to avoid leakage)
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_holdout, y_holdout = X[val_end:], y[val_end:]
    metadata_holdout = metadata[val_end:]

    print(f"[LGBM] Train: {len(X_train)}, Val: {len(X_val)}, Holdout: {len(X_holdout)}", file=sys.stderr)

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # LightGBM parameters - stronger regularization to prevent early stopping
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Reduced from 31 - simpler trees
        'learning_rate': 0.01,  # Reduced from 0.05 - slower learning
        'feature_fraction': 0.6,  # Reduced from 0.8 - more regularization
        'bagging_fraction': 0.6,  # Reduced from 0.8 - more regularization
        'bagging_freq': 3,
        'min_data_in_leaf': 50,  # New: require more samples per leaf
        'lambda_l1': 0.1,  # L1 regularization
        'lambda_l2': 0.1,  # L2 regularization
        'max_depth': 5,  # Limit tree depth
        'verbose': -1,
        'seed': 42,
    }

    # Train with early stopping
    print("[LGBM] Training model...", file=sys.stderr)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"[LGBM] Best iteration: {model.best_iteration}", file=sys.stderr)

    # Predict on holdout
    print("[LGBM] Evaluating on holdout...", file=sys.stderr)
    y_pred_proba = model.predict(X_holdout, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    correct = np.sum(y_pred == y_holdout)
    total = len(y_holdout)
    accuracy = correct / total if total > 0 else 0

    # Direction breakdown
    up_pred = np.sum(y_pred == 1)
    down_pred = np.sum(y_pred == 0)
    up_correct = np.sum((y_pred == 1) & (y_holdout == 1))
    down_correct = np.sum((y_pred == 0) & (y_holdout == 0))

    # High confidence predictions
    high_conf_mask = (y_pred_proba > 0.6) | (y_pred_proba < 0.4)
    high_conf_correct = np.sum(y_pred[high_conf_mask] == y_holdout[high_conf_mask])
    high_conf_total = np.sum(high_conf_mask)
    high_conf_acc = high_conf_correct / high_conf_total if high_conf_total > 0 else 0

    # Feature importance
    feature_names = [
        'returns_1', 'returns_5', 'returns_10', 'returns_20',
        'volatility', 'vol_ratio',
        'poc_dist', 'vah_dist', 'val_dist',
        'in_value', 'above_value', 'below_value',
        'cvd_vs_ema', 'cvd_trend', 'cvd_change',
    ]
    importance = dict(zip(feature_names, [int(x) for x in model.feature_importance()]))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Simulate trades with SL/TP
    print(f"[LGBM] Simulating trades with {sl_ticks} tick SL / {tp_ticks} tick TP...", file=sys.stderr)

    trades = []
    total_pnl = 0
    wins = 0
    losses = 0
    timeouts = 0

    for i, (prob, pred, actual, meta) in enumerate(zip(y_pred_proba, y_pred, y_holdout, metadata_holdout)):
        direction = 'up' if pred == 1 else 'down'
        entry_idx = meta['entry_idx']

        # Simulate trade
        trade_result = simulate_trade(
            bars_1s=bars_1s,
            entry_idx=entry_idx,
            direction=direction,
            tick_size=tick_size,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
        )

        total_pnl += trade_result['pnl_ticks']

        if trade_result['outcome'] == 'win':
            wins += 1
        elif trade_result['outcome'] == 'loss':
            losses += 1
        else:
            timeouts += 1

        trades.append({
            'timestamp': meta['timestamp'],
            'entry_price': meta['current_price'],
            'direction': direction,
            'probability': float(prob),
            'outcome': trade_result['outcome'],
            'exit_price': trade_result['exit_price'],
            'pnl_ticks': round(trade_result['pnl_ticks'], 2),
            'bars_held': trade_result['bars_held'],
        })

    # Calculate trade stats
    total_trades = wins + losses + timeouts
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    # Required win rate for breakeven with 4:16 (1:4) risk/reward
    # Win * 16 = Loss * 4 => Win / (Win + Loss) = 4 / (16 + 4) = 20%
    breakeven_wr = sl_ticks / (sl_ticks + tp_ticks) * 100

    # Expected value per trade
    ev_per_trade = (win_rate / 100 * tp_ticks) - ((100 - win_rate) / 100 * sl_ticks)

    results = {
        'total_predictions': total,
        'accuracy': round(accuracy * 100, 2),
        'direction_breakdown': {
            'up_predictions': int(up_pred),
            'up_accuracy': round(up_correct / up_pred * 100, 2) if up_pred > 0 else 0,
            'down_predictions': int(down_pred),
            'down_accuracy': round(down_correct / down_pred * 100, 2) if down_pred > 0 else 0,
        },
        'high_confidence': {
            'count': int(high_conf_total),
            'pct_of_total': round(high_conf_total / total * 100, 2) if total > 0 else 0,
            'accuracy': round(high_conf_acc * 100, 2),
        },
        'trade_simulation': {
            'sl_ticks': sl_ticks,
            'tp_ticks': tp_ticks,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'timeouts': timeouts,
            'win_rate': round(win_rate, 2),
            'breakeven_win_rate': round(breakeven_wr, 2),
            'total_pnl_ticks': round(total_pnl, 2),
            'ev_per_trade_ticks': round(ev_per_trade, 4),
        },
        'feature_importance': sorted_importance[:10],
        'model_info': {
            'best_iteration': model.best_iteration,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'holdout_samples': len(X_holdout),
        },
        'trades': trades[:100],  # Only include first 100 for brevity
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="LightGBM Backtest with VP + CVD")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file with 1s bars")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--lookback", type=int, default=1800, help="1s bars for VP/CVD (default: 1800 = 30min)")
    parser.add_argument("--prediction-horizon", type=int, default=20, help="1-min bars to predict (default: 20)")
    parser.add_argument("--step", type=int, default=60, help="Sample every N 1s bars (default: 60 = 1min)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of bars (0=all)")

    args = parser.parse_args()

    # Load data
    print(f"[LGBM] Loading data from {args.input}...", file=sys.stderr)
    with open(args.input, 'r') as f:
        data = json.load(f)

    bars_1s = data.get("bars", [])
    if args.limit > 0:
        bars_1s = bars_1s[:args.limit]

    print(f"[LGBM] Loaded {len(bars_1s)} 1-second bars", file=sys.stderr)

    # Run backtest
    results = run_backtest(
        bars_1s=bars_1s,
        lookback_1s=args.lookback,
        prediction_horizon=args.prediction_horizon,
        step=args.step,
    )

    # Print results
    print("\n" + "=" * 70, file=sys.stderr)
    print(f"LGBM VP+CVD BACKTEST RESULTS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total holdout predictions: {results['total_predictions']}", file=sys.stderr)
    print(f"Direction accuracy: {results['accuracy']}%", file=sys.stderr)

    print(f"\nDirection Breakdown:", file=sys.stderr)
    print(f"  UP: {results['direction_breakdown']['up_predictions']} predictions, {results['direction_breakdown']['up_accuracy']}% accurate", file=sys.stderr)
    print(f"  DOWN: {results['direction_breakdown']['down_predictions']} predictions, {results['direction_breakdown']['down_accuracy']}% accurate", file=sys.stderr)

    ts = results['trade_simulation']
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"TRADE SIMULATION ({ts['sl_ticks']} tick SL / {ts['tp_ticks']} tick TP)", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)
    print(f"Total trades: {ts['total_trades']}", file=sys.stderr)
    print(f"  Wins: {ts['wins']} | Losses: {ts['losses']} | Timeouts: {ts['timeouts']}", file=sys.stderr)
    print(f"  Win rate: {ts['win_rate']}% (breakeven: {ts['breakeven_win_rate']}%)", file=sys.stderr)
    print(f"  Total PnL: {ts['total_pnl_ticks']} ticks", file=sys.stderr)
    print(f"  EV per trade: {ts['ev_per_trade_ticks']} ticks", file=sys.stderr)

    # Determine profitability
    if ts['win_rate'] > ts['breakeven_win_rate']:
        print(f"\n  ✅ PROFITABLE (win rate > breakeven)", file=sys.stderr)
    else:
        print(f"\n  ❌ NOT PROFITABLE (win rate < breakeven)", file=sys.stderr)

    print(f"\nTop 5 Feature Importance:", file=sys.stderr)
    for feat, imp in results['feature_importance'][:5]:
        print(f"  {feat}: {imp}", file=sys.stderr)

    print("=" * 70, file=sys.stderr)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[LGBM] Saved results to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
