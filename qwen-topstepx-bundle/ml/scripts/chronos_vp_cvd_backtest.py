#!/usr/bin/env python3
"""
Chronos Backtest with Volume Profile + CVD
Predicts next 20 bars direction (default) or N bars ahead

Features:
- Volume Profile (POC, VAH, VAL) from 1-second bars
- CVD (Cumulative Volume Delta) from 1-second bars
- CVD EMA (smoothed trend) from 1-second bars
- Price action from 1-second bars aggregated to 1-minute

Usage:
    python chronos_vp_cvd_backtest.py --input ../data/bars_1s.json --prediction-length 20
"""

import argparse
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


def calculate_volume_profile(bars_1s: List[Dict], tick_size: float = 0.25) -> Dict[str, Any]:
    """
    Calculate Volume Profile from 1-second bars.
    Returns POC, VAH, VAL, and volume distribution.
    """
    if not bars_1s:
        return {"poc": 0, "vah": 0, "val": 0, "total_volume": 0}

    # Aggregate volume at each price level (rounded to tick)
    price_volume = defaultdict(float)

    for bar in bars_1s:
        # Distribute volume across OHLC prices (simple approximation)
        prices = [bar['o'], bar['h'], bar['l'], bar['c']]
        volume = bar.get('v', 0) or 0
        vol_per_price = volume / 4.0 if volume > 0 else 0.25  # Default small volume

        for price in prices:
            rounded = round(price / tick_size) * tick_size
            price_volume[rounded] += vol_per_price

    if not price_volume:
        return {"poc": 0, "vah": 0, "val": 0, "total_volume": 0}

    # Sort by price
    sorted_prices = sorted(price_volume.keys())
    volumes = [price_volume[p] for p in sorted_prices]
    total_volume = sum(volumes)

    # POC = price with highest volume
    poc_idx = np.argmax(volumes)
    poc = sorted_prices[poc_idx]

    # Value Area = 70% of volume around POC
    target_volume = total_volume * 0.7
    current_volume = volumes[poc_idx]

    lower_idx = poc_idx
    upper_idx = poc_idx

    while current_volume < target_volume:
        # Expand in direction with more volume
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

    val = sorted_prices[lower_idx]  # Value Area Low
    vah = sorted_prices[upper_idx]  # Value Area High

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "total_volume": total_volume,
        "price_levels": len(sorted_prices),
    }


def calculate_cvd_from_1s(bars_1s: List[Dict]) -> Tuple[List[float], List[float]]:
    """
    Calculate CVD and CVD EMA from 1-second bars.
    Uses bar direction to estimate buy vs sell volume.

    Returns:
        cvd_values: Cumulative Volume Delta series
        cvd_ema: EMA of CVD (smoothed trend)
    """
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

        # Estimate delta based on bar structure
        # If close > open: more buying, delta positive
        # If close < open: more selling, delta negative
        # Weight by position of close within range

        bar_range = high_price - low_price
        if bar_range > 0:
            # Close position in range (0 = at low, 1 = at high)
            close_position = (close_price - low_price) / bar_range
            # Delta estimate: positive if close near high, negative if close near low
            delta = volume * (2 * close_position - 1)
        else:
            # No range, use simple direction
            delta = volume if close_price >= open_price else -volume

        cvd += delta
        cvd_values.append(cvd)

    # Calculate EMA of CVD (20-period EMA)
    ema_period = 20
    cvd_ema = []

    if len(cvd_values) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema = sum(cvd_values[:ema_period]) / ema_period  # Initial SMA
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
        # Parse timestamp and truncate to minute
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
            # Update existing bar
            current_bar['h'] = max(current_bar['h'], bar['h'])
            current_bar['l'] = min(current_bar['l'], bar['l'])
            current_bar['c'] = bar['c']
            current_bar['v'] += bar.get('v', 0) or 0

    if current_bar:
        minute_bars.append(current_bar)

    return minute_bars


def predict_with_features(
    price_series: List[float],
    cvd_series: List[float],
    cvd_ema_series: List[float],
    vp_features: Dict[str, Any],
    current_price: float,
    prediction_length: int = 20,
    num_samples: int = 50,
) -> Dict[str, Any]:
    """
    Generate prediction using price + CVD + VP features.

    Chronos primarily uses univariate time series, but we enhance
    predictions by:
    1. Running Chronos on price
    2. Using CVD/CVD EMA trend to weight direction
    3. Using VP levels to estimate support/resistance
    """
    pipeline = get_pipeline()

    # Price prediction
    if len(price_series) < 20:
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "expected_move": 0.0,
            "cvd_trend": "neutral",
        }

    context = torch.tensor(price_series[-512:], dtype=torch.float32)  # Last 512 for context

    forecast = pipeline.predict(
        context,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )

    forecast_np = forecast.squeeze(0).cpu().numpy()
    end_prices = forecast_np[:, -1]  # Final bar predictions

    # Price direction from Chronos
    up_samples = np.sum(end_prices > current_price)
    down_samples = np.sum(end_prices < current_price)
    total = len(end_prices)

    if up_samples > down_samples:
        price_direction = "up"
        price_confidence = up_samples / total
    elif down_samples > up_samples:
        price_direction = "down"
        price_confidence = down_samples / total
    else:
        price_direction = "neutral"
        price_confidence = 0.5

    # CVD Trend Analysis
    if len(cvd_series) >= 20 and len(cvd_ema_series) >= 20:
        recent_cvd = cvd_series[-20:]
        recent_ema = cvd_ema_series[-20:]

        cvd_now = recent_cvd[-1]
        cvd_ema_now = recent_ema[-1]

        # CVD above EMA = bullish flow, below = bearish
        if cvd_now > cvd_ema_now:
            cvd_trend = "up"
            cvd_strength = min((cvd_now - cvd_ema_now) / (abs(cvd_ema_now) + 1), 1.0)
        elif cvd_now < cvd_ema_now:
            cvd_trend = "down"
            cvd_strength = min((cvd_ema_now - cvd_now) / (abs(cvd_ema_now) + 1), 1.0)
        else:
            cvd_trend = "neutral"
            cvd_strength = 0.0
    else:
        cvd_trend = "neutral"
        cvd_strength = 0.0

    # Volume Profile Analysis
    poc = vp_features.get("poc", current_price)
    vah = vp_features.get("vah", current_price + 1)
    val = vp_features.get("val", current_price - 1)

    # Price position relative to VP
    if current_price > vah:
        vp_position = "above_value"  # Bullish breakout potential
    elif current_price < val:
        vp_position = "below_value"  # Bearish breakdown potential
    else:
        vp_position = "in_value"  # Mean reversion zone

    # Distance to POC (mean reversion target)
    poc_distance = (poc - current_price) / current_price * 100  # percentage

    # Combined confidence
    combined_confidence = price_confidence

    # Boost if CVD agrees with price direction
    if price_direction == cvd_trend:
        combined_confidence = min(combined_confidence + 0.1, 1.0)
    elif cvd_trend != "neutral" and price_direction != "neutral":
        combined_confidence = max(combined_confidence - 0.1, 0.0)

    # Final direction determination
    final_direction = price_direction

    # Strong CVD can override weak price signal
    if cvd_strength > 0.3 and cvd_trend != "neutral" and price_confidence < 0.6:
        final_direction = cvd_trend

    median_end = float(np.median(end_prices))
    expected_move = median_end - current_price

    return {
        "direction": final_direction,
        "confidence": float(combined_confidence),
        "price_direction": price_direction,
        "price_confidence": float(price_confidence),
        "cvd_trend": cvd_trend,
        "cvd_strength": float(cvd_strength),
        "vp_position": vp_position,
        "poc": float(poc),
        "poc_distance_pct": float(poc_distance),
        "expected_move": float(expected_move),
        "expected_move_pct": float((expected_move / current_price) * 100),
        "forecast_median": float(median_end),
        "forecast_q10": float(np.percentile(end_prices, 10)),
        "forecast_q90": float(np.percentile(end_prices, 90)),
    }


def run_backtest(
    bars_1s: List[Dict],
    lookback_1s: int = 1800,  # 30 min of 1s bars for VP/CVD
    prediction_length: int = 20,  # Predict next 20 1-min bars (default)
    step: int = 60,  # Generate prediction every 60 1s bars (1 min)
    confidence_threshold: float = 0.55,
) -> List[Dict[str, Any]]:
    """
    Run backtest with Volume Profile + CVD features.

    Args:
        bars_1s: 1-second bar data
        lookback_1s: Number of 1s bars to use for VP/CVD calculation
        prediction_length: Number of 1-min bars to predict
        step: Generate prediction every N 1s bars
        confidence_threshold: Threshold for pass/fail
    """
    print(f"[Backtest] Total 1s bars: {len(bars_1s)}", file=sys.stderr)

    # Aggregate to 1-minute for price prediction context
    bars_1m = aggregate_to_1min(bars_1s)
    print(f"[Backtest] Aggregated to {len(bars_1m)} 1-min bars", file=sys.stderr)

    if len(bars_1m) < lookback_1s // 60 + prediction_length:
        print("[Backtest] Not enough data for backtest", file=sys.stderr)
        return []

    predictions = []

    print(f"[Backtest] Running predictions (step={step}, pred_len={prediction_length})...", file=sys.stderr)

    for i in range(lookback_1s, len(bars_1s) - prediction_length * 60, step):
        # Get 1s bars for VP and CVD calculation
        window_1s = bars_1s[i - lookback_1s:i]

        # Calculate features from 1s bars
        vp_features = calculate_volume_profile(window_1s)
        cvd_values, cvd_ema = calculate_cvd_from_1s(window_1s)

        # Get 1-minute bars for price series
        window_1m = aggregate_to_1min(window_1s)
        price_series = [bar['c'] for bar in window_1m]
        current_price = price_series[-1] if price_series else 0

        if len(price_series) < 20:
            continue

        # Generate prediction
        pred = predict_with_features(
            price_series=price_series,
            cvd_series=cvd_values,
            cvd_ema_series=cvd_ema,
            vp_features=vp_features,
            current_price=current_price,
            prediction_length=prediction_length,
        )

        pred["timestamp"] = bars_1s[i-1]['t']
        pred["bar_index_1s"] = i - 1
        pred["current_price"] = float(current_price)
        pred["pass"] = pred["confidence"] >= confidence_threshold

        # Calculate actual outcome
        future_1s = bars_1s[i:i + prediction_length * 60]
        if len(future_1s) >= prediction_length * 60:
            future_1m = aggregate_to_1min(future_1s)
            if len(future_1m) >= prediction_length:
                actual_future = future_1m[prediction_length - 1]['c']
                actual_move = actual_future - current_price
                pred["actual_move"] = float(actual_move)
                pred["actual_direction"] = "up" if actual_move > 0 else "down" if actual_move < 0 else "neutral"
                pred["prediction_correct"] = (
                    (pred["direction"] == "up" and actual_move > 0) or
                    (pred["direction"] == "down" and actual_move < 0)
                )

                # Track max favorable/adverse excursion
                future_highs = [b['h'] for b in future_1m[:prediction_length]]
                future_lows = [b['l'] for b in future_1m[:prediction_length]]
                pred["max_favorable"] = float(max(future_highs) - current_price) if pred["direction"] == "up" else float(current_price - min(future_lows))
                pred["max_adverse"] = float(current_price - min(future_lows)) if pred["direction"] == "up" else float(max(future_highs) - current_price)

        predictions.append(pred)

        if len(predictions) % 100 == 0:
            print(f"[Backtest] Processed {len(predictions)} predictions...", file=sys.stderr)

    print(f"[Backtest] Generated {len(predictions)} predictions", file=sys.stderr)
    return predictions


def analyze_backtest(predictions: List[Dict], confidence_threshold: float = 0.55) -> Dict[str, Any]:
    """Analyze backtest results."""
    with_outcomes = [p for p in predictions if "actual_direction" in p]

    if not with_outcomes:
        return {"error": "No predictions with outcomes"}

    total = len(with_outcomes)
    correct = sum(1 for p in with_outcomes if p.get("prediction_correct", False))

    # High confidence
    high_conf = [p for p in with_outcomes if p["confidence"] >= confidence_threshold]
    high_conf_correct = sum(1 for p in high_conf if p.get("prediction_correct", False))

    # CVD agreement analysis
    cvd_agrees = [p for p in with_outcomes if p["direction"] == p["cvd_trend"]]
    cvd_agrees_correct = sum(1 for p in cvd_agrees if p.get("prediction_correct", False))

    cvd_disagrees = [p for p in with_outcomes if p["direction"] != p["cvd_trend"] and p["cvd_trend"] != "neutral"]
    cvd_disagrees_correct = sum(1 for p in cvd_disagrees if p.get("prediction_correct", False))

    # VP position analysis
    in_value = [p for p in with_outcomes if p["vp_position"] == "in_value"]
    in_value_correct = sum(1 for p in in_value if p.get("prediction_correct", False))

    out_of_value = [p for p in with_outcomes if p["vp_position"] != "in_value"]
    out_of_value_correct = sum(1 for p in out_of_value if p.get("prediction_correct", False))

    # Excursion analysis (for risk/reward)
    avg_mfe = np.mean([p.get("max_favorable", 0) for p in with_outcomes if "max_favorable" in p])
    avg_mae = np.mean([p.get("max_adverse", 0) for p in with_outcomes if "max_adverse" in p])

    return {
        "total_predictions": total,
        "overall_accuracy": round(correct / total * 100, 2) if total > 0 else 0,

        "high_confidence": {
            "count": len(high_conf),
            "pct_of_total": round(len(high_conf) / total * 100, 2) if total > 0 else 0,
            "accuracy": round(high_conf_correct / len(high_conf) * 100, 2) if high_conf else 0,
        },

        "cvd_agreement": {
            "agrees_count": len(cvd_agrees),
            "agrees_accuracy": round(cvd_agrees_correct / len(cvd_agrees) * 100, 2) if cvd_agrees else 0,
            "disagrees_count": len(cvd_disagrees),
            "disagrees_accuracy": round(cvd_disagrees_correct / len(cvd_disagrees) * 100, 2) if cvd_disagrees else 0,
        },

        "volume_profile": {
            "in_value_count": len(in_value),
            "in_value_accuracy": round(in_value_correct / len(in_value) * 100, 2) if in_value else 0,
            "out_of_value_count": len(out_of_value),
            "out_of_value_accuracy": round(out_of_value_correct / len(out_of_value) * 100, 2) if out_of_value else 0,
        },

        "excursion": {
            "avg_max_favorable": round(avg_mfe, 4),
            "avg_max_adverse": round(avg_mae, 4),
            "risk_reward_ratio": round(avg_mfe / avg_mae, 2) if avg_mae > 0 else 0,
        },

        "by_direction": {
            "up": {
                "count": len([p for p in with_outcomes if p["direction"] == "up"]),
                "accuracy": round(sum(1 for p in with_outcomes if p["direction"] == "up" and p.get("prediction_correct", False)) /
                           max(len([p for p in with_outcomes if p["direction"] == "up"]), 1) * 100, 2),
            },
            "down": {
                "count": len([p for p in with_outcomes if p["direction"] == "down"]),
                "accuracy": round(sum(1 for p in with_outcomes if p["direction"] == "down" and p.get("prediction_correct", False)) /
                           max(len([p for p in with_outcomes if p["direction"] == "down"]), 1) * 100, 2),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Chronos Backtest with VP + CVD")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file with 1s bars")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--lookback", type=int, default=1800, help="1s bars for VP/CVD (default: 1800 = 30min)")
    parser.add_argument("--prediction-length", type=int, default=20, help="1-min bars to predict (default: 20)")
    parser.add_argument("--step", type=int, default=60, help="Predict every N 1s bars (default: 60 = 1min)")
    parser.add_argument("--confidence", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of bars (0=all)")

    args = parser.parse_args()

    # Load data
    print(f"[Chronos] Loading data from {args.input}...", file=sys.stderr)
    with open(args.input, 'r') as f:
        data = json.load(f)

    bars_1s = data.get("bars", [])
    if args.limit > 0:
        bars_1s = bars_1s[:args.limit]

    print(f"[Chronos] Loaded {len(bars_1s)} 1-second bars", file=sys.stderr)

    if len(bars_1s) < args.lookback + args.prediction_length * 60:
        print(f"Error: Need at least {args.lookback + args.prediction_length * 60} bars", file=sys.stderr)
        sys.exit(1)

    # Run backtest
    predictions = run_backtest(
        bars_1s=bars_1s,
        lookback_1s=args.lookback,
        prediction_length=args.prediction_length,
        step=args.step,
        confidence_threshold=args.confidence,
    )

    # Analyze
    analysis = analyze_backtest(predictions, args.confidence)

    # Print results
    print("\n" + "=" * 70, file=sys.stderr)
    print("CHRONOS VP+CVD BACKTEST RESULTS (Predict Next 20 1-Min Bars)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total predictions: {analysis['total_predictions']}", file=sys.stderr)
    print(f"Overall accuracy: {analysis['overall_accuracy']}%", file=sys.stderr)

    print(f"\nHigh Confidence (>={args.confidence*100:.0f}%):", file=sys.stderr)
    print(f"  Count: {analysis['high_confidence']['count']} ({analysis['high_confidence']['pct_of_total']}%)", file=sys.stderr)
    print(f"  Accuracy: {analysis['high_confidence']['accuracy']}%", file=sys.stderr)

    print(f"\nCVD Agreement Analysis:", file=sys.stderr)
    print(f"  CVD Agrees: {analysis['cvd_agreement']['agrees_count']} predictions, {analysis['cvd_agreement']['agrees_accuracy']}% accurate", file=sys.stderr)
    print(f"  CVD Disagrees: {analysis['cvd_agreement']['disagrees_count']} predictions, {analysis['cvd_agreement']['disagrees_accuracy']}% accurate", file=sys.stderr)

    print(f"\nVolume Profile Analysis:", file=sys.stderr)
    print(f"  In Value Area: {analysis['volume_profile']['in_value_count']} predictions, {analysis['volume_profile']['in_value_accuracy']}% accurate", file=sys.stderr)
    print(f"  Out of Value: {analysis['volume_profile']['out_of_value_count']} predictions, {analysis['volume_profile']['out_of_value_accuracy']}% accurate", file=sys.stderr)

    print(f"\nExcursion Analysis (Risk/Reward):", file=sys.stderr)
    print(f"  Avg Max Favorable: {analysis['excursion']['avg_max_favorable']:.4f}", file=sys.stderr)
    print(f"  Avg Max Adverse: {analysis['excursion']['avg_max_adverse']:.4f}", file=sys.stderr)
    print(f"  Risk/Reward Ratio: {analysis['excursion']['risk_reward_ratio']:.2f}", file=sys.stderr)

    print(f"\nBy Direction:", file=sys.stderr)
    print(f"  UP: {analysis['by_direction']['up']['count']} predictions, {analysis['by_direction']['up']['accuracy']}% accurate", file=sys.stderr)
    print(f"  DOWN: {analysis['by_direction']['down']['count']} predictions, {analysis['by_direction']['down']['accuracy']}% accurate", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Save results
    result = {
        "analysis": analysis,
        "predictions": predictions,
        "config": {
            "lookback_1s": args.lookback,
            "prediction_length": args.prediction_length,
            "step": args.step,
            "confidence_threshold": args.confidence,
            "model": _model_name,
            "total_bars": len(bars_1s),
        },
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[Chronos] Saved results to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
