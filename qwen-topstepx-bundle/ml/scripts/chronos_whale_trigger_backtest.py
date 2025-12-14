#!/usr/bin/env python3
"""
Chronos Backtest triggered by Whale Flow Events
Uses Chronos time-series prediction at each whale trigger point.

Usage:
    python chronos_whale_trigger_backtest.py
"""

import json
import sys
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

# Lazy load pipeline
_pipeline = None
_model_name = "amazon/chronos-t5-small"


def get_pipeline():
    """Lazy load Chronos pipeline."""
    global _pipeline
    if _pipeline is None:
        from chronos import ChronosPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[Chronos] Loading {_model_name} on {device}...", file=sys.stderr)
        _pipeline = ChronosPipeline.from_pretrained(
            _model_name,
            device_map=device,
            dtype=dtype,
        )
        print(f"[Chronos] Model loaded", file=sys.stderr)

    return _pipeline


def load_1s_bars(filepath: str) -> List[Dict]:
    """Load 1-second bars from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("bars", [])


def load_whale_options_flow(filepath: str) -> pd.DataFrame:
    """Load whale options flow from CSV."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    return df


def aggregate_bars(bars_1s: List[Dict], period_minutes: int) -> List[Dict]:
    """Aggregate 1-second bars to any timeframe."""
    if not bars_1s:
        return []

    period_data = defaultdict(lambda: {'bars': []})
    period_order = []

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00').replace('+00:00', ''))
        else:
            dt = ts

        if period_minutes >= 1440:
            period_ts = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period_minutes >= 60:
            hour = (dt.hour // (period_minutes // 60)) * (period_minutes // 60)
            period_ts = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
        else:
            minute = (dt.minute // period_minutes) * period_minutes
            period_ts = dt.replace(minute=minute, second=0, microsecond=0)

        period_key = period_ts.isoformat()
        if period_key not in period_data:
            period_order.append(period_key)
        period_data[period_key]['bars'].append(bar)

    result = []
    for period_ts in period_order:
        bars = period_data[period_ts]['bars']
        if not bars:
            continue
        result.append({
            't': period_ts,
            'o': bars[0]['o'],
            'h': max(b['h'] for b in bars),
            'l': min(b['l'] for b in bars),
            'c': bars[-1]['c'],
            'v': sum(b.get('v', 0) or 0 for b in bars),
        })
    return result


def calculate_cvd_1min(bars_1s: List[Dict]) -> Tuple[List[Dict], List[float], List[float]]:
    """Calculate CVD from 1s bars, aggregated to 1-min resolution."""
    if not bars_1s:
        return [], [], []

    minute_data = defaultdict(lambda: {'bars': [], 'delta': 0.0})
    minute_order = []

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00').replace('+00:00', ''))
        else:
            dt = ts
        minute_ts = dt.replace(second=0, microsecond=0).isoformat()

        if minute_ts not in minute_data:
            minute_order.append(minute_ts)

        minute_data[minute_ts]['bars'].append(bar)

        volume = bar.get('v', 0) or 0
        bar_range = bar['h'] - bar['l']
        if bar_range > 0:
            close_position = (bar['c'] - bar['l']) / bar_range
            delta = volume * (2 * close_position - 1)
        else:
            delta = volume if bar['c'] >= bar['o'] else -volume
        minute_data[minute_ts]['delta'] += delta

    bars_1min = []
    cvd = 0.0
    cvd_1min = []

    for minute_ts in minute_order:
        data = minute_data[minute_ts]
        bars = data['bars']
        if not bars:
            continue

        bars_1min.append({
            't': minute_ts,
            'o': bars[0]['o'],
            'h': max(b['h'] for b in bars),
            'l': min(b['l'] for b in bars),
            'c': bars[-1]['c'],
            'v': sum(b.get('v', 0) or 0 for b in bars),
        })

        cvd += data['delta']
        cvd_1min.append(cvd)

    ema_period = 20
    cvd_ema_1min = []
    if len(cvd_1min) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema = sum(cvd_1min[:ema_period]) / ema_period
        cvd_ema_1min = [ema] * ema_period
        for i in range(ema_period, len(cvd_1min)):
            ema = (cvd_1min[i] * multiplier) + (ema * (1 - multiplier))
            cvd_ema_1min.append(ema)
    else:
        cvd_ema_1min = cvd_1min.copy()

    return bars_1min, cvd_1min, cvd_ema_1min


def calculate_volume_profile(bars: List[Dict], lookback: int = 30, tick_size: float = 0.25) -> Dict[str, float]:
    """Calculate Volume Profile from bars."""
    if len(bars) < lookback:
        lookback = len(bars)

    recent = bars[-lookback:]
    price_volume = defaultdict(float)

    for bar in recent:
        prices = [bar['o'], bar['h'], bar['l'], bar['c']]
        volume = bar.get('v', 0) or 0
        vol_per_price = volume / 4.0 if volume > 0 else 0.25

        for price in prices:
            rounded = round(price / tick_size) * tick_size
            price_volume[rounded] += vol_per_price

    if not price_volume:
        return {"poc": 0, "vah": 0, "val": 0}

    sorted_prices = sorted(price_volume.keys())
    volumes = [price_volume[p] for p in sorted_prices]
    total_volume = sum(volumes)

    poc_idx = np.argmax(volumes)
    poc = sorted_prices[poc_idx]

    target_volume = total_volume * 0.7
    current_volume = volumes[poc_idx]
    lower_idx, upper_idx = poc_idx, poc_idx

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

    return {
        "poc": poc,
        "vah": sorted_prices[upper_idx],
        "val": sorted_prices[lower_idx],
    }


def find_swing_highs_lows(bars: List[Dict], lookback: int = 10) -> Tuple[List[float], List[float]]:
    """Find recent swing highs and lows for SL placement."""
    if len(bars) < 3:
        return [], []

    swing_highs = []
    swing_lows = []

    recent = bars[-lookback:]
    for i in range(1, len(recent) - 1):
        if recent[i]['h'] > recent[i-1]['h'] and recent[i]['h'] > recent[i+1]['h']:
            swing_highs.append(recent[i]['h'])
        if recent[i]['l'] < recent[i-1]['l'] and recent[i]['l'] < recent[i+1]['l']:
            swing_lows.append(recent[i]['l'])

    return swing_highs, swing_lows


def predict_with_chronos(
    price_series: List[float],
    prediction_length: int = 20,
    num_samples: int = 50,
) -> Dict[str, Any]:
    """Generate Chronos prediction."""
    if len(price_series) < 30:
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "expected_move": 0.0,
        }

    pipeline = get_pipeline()
    context = torch.tensor(price_series[-512:], dtype=torch.float32)

    forecast = pipeline.predict(
        context,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )

    forecast_np = forecast.squeeze(0).cpu().numpy()
    current_price = price_series[-1]
    end_prices = forecast_np[:, -1]

    up_samples = np.sum(end_prices > current_price)
    down_samples = np.sum(end_prices < current_price)
    total = len(end_prices)

    if up_samples > down_samples:
        direction = "up"
        confidence = up_samples / total
    elif down_samples > up_samples:
        direction = "down"
        confidence = down_samples / total
    else:
        direction = "neutral"
        confidence = 0.5

    median_end = float(np.median(end_prices))
    expected_move = median_end - current_price

    return {
        "direction": direction,
        "confidence": float(confidence),
        "expected_move": float(expected_move),
        "forecast_median": float(median_end),
        "forecast_q10": float(np.percentile(end_prices, 10)),
        "forecast_q90": float(np.percentile(end_prices, 90)),
    }


def simulate_trade(
    bars_1min: List[Dict],
    entry_idx: int,
    direction: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    max_bars: int = 60,
) -> Dict[str, Any]:
    """Simulate a trade with fixed SL/TP."""
    pnl = 0.0
    exit_price = entry_price
    outcome = "timeout"
    bars_held = 0
    max_profit = 0.0

    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(bars_1min))):
        bar = bars_1min[i]
        bars_held = i - entry_idx

        if direction == "long":
            current_profit = bar['h'] - entry_price
            max_profit = max(max_profit, current_profit)

            if bar['l'] <= stop_loss:
                exit_price = stop_loss
                pnl = stop_loss - entry_price
                outcome = "loss"
                break
            elif bar['h'] >= target:
                exit_price = target
                pnl = target - entry_price
                outcome = "win"
                break
        else:  # short
            current_profit = entry_price - bar['l']
            max_profit = max(max_profit, current_profit)

            if bar['h'] >= stop_loss:
                exit_price = stop_loss
                pnl = entry_price - stop_loss
                outcome = "loss"
                break
            elif bar['l'] <= target:
                exit_price = target
                pnl = entry_price - target
                outcome = "win"
                break

    if outcome == "timeout":
        final_bar = bars_1min[min(entry_idx + max_bars, len(bars_1min) - 1)]
        exit_price = final_bar['c']
        if direction == "long":
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
        outcome = "win" if pnl > 0 else "loss"

    return {
        "outcome": outcome,
        "exit_price": exit_price,
        "pnl_points": pnl,
        "bars_held": bars_held,
        "max_profit": max_profit,
    }


def run_backtest(
    bars_1s: List[Dict],
    options_df: pd.DataFrame,
    target_date: str,
    premium_threshold: float = 100000,
) -> Tuple[List[Dict], Dict]:
    """Run backtest triggered by whale flow."""

    # Aggregate bars
    print(f"[Backtest] Aggregating {len(bars_1s)} 1s bars...", file=sys.stderr)
    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
    bars_5min = aggregate_bars(bars_1s, 5)
    print(f"[Backtest] Got {len(bars_1min)} 1-min bars, {len(bars_5min)} 5-min bars", file=sys.stderr)

    if len(bars_1min) < 60:
        print("[Backtest] Not enough data", file=sys.stderr)
        return [], {}

    # Build timestamp -> bar index mapping
    ts_to_idx = {}
    for i, bar in enumerate(bars_1min):
        ts_str = bar['t']
        if isinstance(ts_str, str):
            ts_str = ts_str.replace('+00:00', '')
        ts_to_idx[ts_str] = i

    # Filter options by date and premium
    date_filter = options_df['datetime'].dt.strftime('%Y-%m-%d') == target_date
    premium_filter = options_df['premium'] >= premium_threshold
    filtered_options = options_df[date_filter & premium_filter].copy()
    print(f"[Backtest] Found {len(filtered_options)} whale triggers on {target_date} with premium >= ${premium_threshold:,.0f}", file=sys.stderr)

    trades = []
    price_series = [b['c'] for b in bars_1min]

    for _, row in filtered_options.iterrows():
        whale_ts = row['datetime']
        minute_ts = whale_ts.replace(second=0, microsecond=0).isoformat().replace('+00:00', '')

        if minute_ts not in ts_to_idx:
            continue

        bar_idx = ts_to_idx[minute_ts]
        if bar_idx < 30 or bar_idx >= len(bars_1min) - 20:
            continue

        current_bar = bars_1min[bar_idx]
        current_price = current_bar['c']

        # Get CVD trend
        cvd_now = cvd_1min[bar_idx] if bar_idx < len(cvd_1min) else 0
        cvd_ema_now = cvd_ema_1min[bar_idx] if bar_idx < len(cvd_ema_1min) else 0
        cvd_trend = "up" if cvd_now > cvd_ema_now else "down" if cvd_now < cvd_ema_now else "neutral"

        # Volume Profile
        vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

        # Whale flow direction
        whale_sentiment = row.get('bearish_or_bullish', 'neutral')
        whale_direction = "long" if whale_sentiment == "bullish" else "short" if whale_sentiment == "bearish" else None

        if whale_direction is None:
            continue

        # Chronos prediction
        chronos = predict_with_chronos(price_series[:bar_idx+1], prediction_length=20)
        chronos_direction = "long" if chronos["direction"] == "up" else "short" if chronos["direction"] == "down" else "neutral"

        # Decide direction: Chronos prediction takes priority
        if chronos_direction == "neutral":
            direction = whale_direction
        else:
            direction = chronos_direction

        # Calculate SL/TP based on swing points
        swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:bar_idx//5+1], lookback=10)

        if direction == "long":
            if swing_lows:
                stop_loss = min(swing_lows) - 1
            else:
                stop_loss = current_price - 10
            risk = current_price - stop_loss
            target = current_price + (risk * 3)  # 3:1 RR
        else:
            if swing_highs:
                stop_loss = max(swing_highs) + 1
            else:
                stop_loss = current_price + 10
            risk = stop_loss - current_price
            target = current_price - (risk * 3)  # 3:1 RR

        # Simulate trade
        result = simulate_trade(
            bars_1min, bar_idx, direction, current_price, stop_loss, target, max_bars=60
        )

        trade = {
            "timestamp": minute_ts,
            "bar_index": bar_idx,
            "direction": direction,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "target": target,
            "risk_reward": 3.0,
            "whale_sentiment": whale_sentiment,
            "whale_premium": row['premium'],
            "chronos_direction": chronos["direction"],
            "chronos_confidence": chronos["confidence"],
            "chronos_expected_move": chronos["expected_move"],
            "cvd_trend": cvd_trend,
            "poc": vp["poc"],
            "outcome": result["outcome"],
            "exit_price": result["exit_price"],
            "pnl_points": result["pnl_points"],
            "bars_held": result["bars_held"],
            "max_profit": result["max_profit"],
        }
        trades.append(trade)
        print(f"[Trade {len(trades)}] {minute_ts} {direction.upper()} @ {current_price:.2f} | Chronos: {chronos['direction']} ({chronos['confidence']:.1%}) | {result['outcome'].upper()} {result['pnl_points']:+.2f} pts", file=sys.stderr)

    # Analysis
    if trades:
        wins = sum(1 for t in trades if t["outcome"] == "win")
        total_pnl = sum(t["pnl_points"] for t in trades)
        analysis = {
            "total_trades": len(trades),
            "wins": wins,
            "losses": len(trades) - wins,
            "win_rate": round(wins / len(trades) * 100, 1),
            "total_pnl_points": round(total_pnl, 2),
            "avg_pnl_per_trade": round(total_pnl / len(trades), 2),
        }

        # By direction
        longs = [t for t in trades if t["direction"] == "long"]
        shorts = [t for t in trades if t["direction"] == "short"]
        analysis["by_direction"] = {
            "long": {
                "count": len(longs),
                "wins": sum(1 for t in longs if t["outcome"] == "win"),
                "pnl": round(sum(t["pnl_points"] for t in longs), 2),
            },
            "short": {
                "count": len(shorts),
                "wins": sum(1 for t in shorts if t["outcome"] == "win"),
                "pnl": round(sum(t["pnl_points"] for t in shorts), 2),
            },
        }

        # By CVD alignment
        cvd_aligned = [t for t in trades if
                       (t["direction"] == "long" and t["cvd_trend"] == "up") or
                       (t["direction"] == "short" and t["cvd_trend"] == "down")]
        cvd_against = [t for t in trades if
                       (t["direction"] == "long" and t["cvd_trend"] == "down") or
                       (t["direction"] == "short" and t["cvd_trend"] == "up")]
        analysis["by_cvd_alignment"] = {
            "aligned": {
                "count": len(cvd_aligned),
                "wins": sum(1 for t in cvd_aligned if t["outcome"] == "win"),
                "pnl": round(sum(t["pnl_points"] for t in cvd_aligned), 2),
            },
            "against": {
                "count": len(cvd_against),
                "wins": sum(1 for t in cvd_against if t["outcome"] == "win"),
                "pnl": round(sum(t["pnl_points"] for t in cvd_against), 2),
            },
        }

        # By Chronos confidence
        high_conf = [t for t in trades if t["chronos_confidence"] >= 0.6]
        low_conf = [t for t in trades if t["chronos_confidence"] < 0.6]
        analysis["by_chronos_confidence"] = {
            "high_60+": {
                "count": len(high_conf),
                "wins": sum(1 for t in high_conf if t["outcome"] == "win"),
                "pnl": round(sum(t["pnl_points"] for t in high_conf), 2),
            },
            "low_<60": {
                "count": len(low_conf),
                "wins": sum(1 for t in low_conf if t["outcome"] == "win"),
                "pnl": round(sum(t["pnl_points"] for t in low_conf), 2),
            },
        }
    else:
        analysis = {"total_trades": 0}

    return trades, analysis


def main():
    # Configuration - MODIFY THESE
    TARGET_DATE = "2025-12-04"  # Change to test different dates
    PREMIUM_THRESHOLD = 100000  # $100k minimum premium

    # Paths
    bars_1s_path = "data/bars_1s.json"
    options_flow_path = "data/qqq_options_flow_20251205.csv"
    output_path = f"data/chronos_whale_trigger_results_{TARGET_DATE.replace('-', '')}.json"

    print("=" * 70, file=sys.stderr)
    print("CHRONOS + WHALE FLOW TRIGGER BACKTEST", file=sys.stderr)
    print(f"Date: {TARGET_DATE}", file=sys.stderr)
    print(f"Premium threshold: ${PREMIUM_THRESHOLD:,.0f}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Load data
    print(f"[Load] Loading 1s bars from {bars_1s_path}...", file=sys.stderr)
    bars_1s_all = load_1s_bars(bars_1s_path)
    print(f"[Load] Loaded {len(bars_1s_all)} total 1-second bars", file=sys.stderr)

    # Filter to target date
    bars_1s = [b for b in bars_1s_all if TARGET_DATE in b['t']]
    print(f"[Load] Filtered to {len(bars_1s)} bars on {TARGET_DATE}", file=sys.stderr)

    print(f"[Load] Loading whale options flow from {options_flow_path}...", file=sys.stderr)
    options_df = load_whale_options_flow(options_flow_path)
    print(f"[Load] Loaded {len(options_df)} options flow records", file=sys.stderr)

    # Run backtest
    trades, analysis = run_backtest(
        bars_1s=bars_1s,
        options_df=options_df,
        target_date=TARGET_DATE,
        premium_threshold=PREMIUM_THRESHOLD,
    )

    # Print results
    print("\n" + "=" * 70, file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total trades: {analysis.get('total_trades', 0)}", file=sys.stderr)
    if analysis.get('total_trades', 0) > 0:
        print(f"Win rate: {analysis['win_rate']}%", file=sys.stderr)
        print(f"Total P&L: {analysis['total_pnl_points']:+.2f} points", file=sys.stderr)
        print(f"Avg P&L per trade: {analysis['avg_pnl_per_trade']:+.2f} points", file=sys.stderr)

        print(f"\nBy Direction:", file=sys.stderr)
        for dir_name, stats in analysis.get('by_direction', {}).items():
            print(f"  {dir_name.upper()}: {stats['count']} trades, {stats['wins']} wins, {stats['pnl']:+.2f} pts", file=sys.stderr)

        print(f"\nBy CVD Alignment:", file=sys.stderr)
        for align, stats in analysis.get('by_cvd_alignment', {}).items():
            print(f"  {align.upper()}: {stats['count']} trades, {stats['wins']} wins, {stats['pnl']:+.2f} pts", file=sys.stderr)

        print(f"\nBy Chronos Confidence:", file=sys.stderr)
        for conf, stats in analysis.get('by_chronos_confidence', {}).items():
            print(f"  {conf}: {stats['count']} trades, {stats['wins']} wins, {stats['pnl']:+.2f} pts", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Save results
    result = {
        "analysis": analysis,
        "trades": trades,
        "config": {
            "trigger": "whale_flow",
            "model": _model_name,
            "date": TARGET_DATE,
            "premium_threshold": PREMIUM_THRESHOLD,
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Save] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
