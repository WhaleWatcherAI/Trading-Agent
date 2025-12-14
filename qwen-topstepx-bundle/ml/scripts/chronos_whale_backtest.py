#!/usr/bin/env python3
"""
Chronos Backtest with CVD/VP/CVD-Trend + Whale Flow
1-day backtest for 12/5/2025 using:
- 1-second bars aggregated to 1-minute for CVD, Volume Profile, CVD trend
- SPY options flow from Unusual Whales for sentiment
- Chronos prediction for 20 bars ahead

Usage:
    python chronos_whale_backtest.py
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
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    return df


def load_whale_lit_flow(filepath: str) -> pd.DataFrame:
    """Load whale lit/stock flow from CSV."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    return df


def aggregate_1s_to_1min(bars_1s: List[Dict]) -> Tuple[List[Dict], List[float], List[float]]:
    """
    Aggregate 1-second bars to 1-minute bars.
    Also calculate CVD from 1-sec deltas, aggregated to 1-min resolution.

    Returns:
        bars_1min: List of 1-minute OHLCV bars
        cvd_1min: CVD values at 1-minute resolution
        cvd_ema_1min: CVD EMA at 1-minute resolution
    """
    if not bars_1s:
        return [], [], []

    # Group by minute
    minute_data = defaultdict(lambda: {
        'bars': [],
        'delta': 0.0,
    })
    minute_order = []

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            dt = ts
        minute_ts = dt.replace(second=0, microsecond=0).isoformat()

        if minute_ts not in minute_data:
            minute_order.append(minute_ts)

        minute_data[minute_ts]['bars'].append(bar)

        # Calculate delta for this 1-sec bar
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

        minute_data[minute_ts]['delta'] += delta

    # Build 1-minute bars and CVD
    bars_1min = []
    cvd = 0.0
    cvd_1min = []

    for minute_ts in minute_order:
        data = minute_data[minute_ts]
        bars = data['bars']

        if not bars:
            continue

        bar_1min = {
            't': minute_ts,
            'o': bars[0]['o'],
            'h': max(b['h'] for b in bars),
            'l': min(b['l'] for b in bars),
            'c': bars[-1]['c'],
            'v': sum(b.get('v', 0) or 0 for b in bars),
        }
        bars_1min.append(bar_1min)

        # Cumulative CVD at 1-min resolution
        cvd += data['delta']
        cvd_1min.append(cvd)

    # Calculate CVD EMA (20-period) at 1-minute resolution
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


def calculate_volume_profile(bars_1min: List[Dict], lookback: int = 30, tick_size: float = 0.25) -> Dict[str, Any]:
    """Calculate Volume Profile from last N 1-minute bars."""
    if len(bars_1min) < lookback:
        lookback = len(bars_1min)

    recent_bars = bars_1min[-lookback:]
    price_volume = defaultdict(float)

    for bar in recent_bars:
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

    # Value Area (70%)
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

    return {
        "poc": poc,
        "vah": sorted_prices[upper_idx],
        "val": sorted_prices[lower_idx],
    }


def get_whale_sentiment(options_df: pd.DataFrame, lit_df: pd.DataFrame, timestamp: datetime, lookback_minutes: int = 30) -> Dict[str, Any]:
    """
    Get whale options flow + lit flow sentiment for a given timestamp.
    Looks back N minutes from the timestamp.
    """
    cutoff = timestamp - timedelta(minutes=lookback_minutes)

    # Options flow sentiment
    mask = (options_df['datetime'] >= cutoff) & (options_df['datetime'] <= timestamp)
    recent_options = options_df[mask]

    bullish_premium = 0
    bearish_premium = 0
    options_count = 0

    if len(recent_options) > 0:
        bullish = recent_options[recent_options['bearish_or_bullish'] == 'bullish']
        bearish = recent_options[recent_options['bearish_or_bullish'] == 'bearish']
        bullish_premium = bullish['premium'].sum() if len(bullish) > 0 else 0
        bearish_premium = bearish['premium'].sum() if len(bearish) > 0 else 0
        options_count = len(recent_options)

    # Lit flow sentiment (large stock block trades)
    lit_mask = (lit_df['datetime'] >= cutoff) & (lit_df['datetime'] <= timestamp)
    recent_lit = lit_df[lit_mask]

    lit_buy_volume = 0
    lit_sell_volume = 0
    lit_count = 0

    if len(recent_lit) > 0:
        # Estimate buy/sell based on price vs nbbo_bid/ask
        for _, row in recent_lit.iterrows():
            price = row.get('price', 0)
            bid = row.get('nbbo_bid', 0)
            ask = row.get('nbbo_ask', 0)
            size = row.get('size', 0)
            premium = row.get('premium', 0)

            # If price >= ask, likely a buy; if price <= bid, likely a sell
            if price >= ask and ask > 0:
                lit_buy_volume += premium
            elif price <= bid and bid > 0:
                lit_sell_volume += premium
            else:
                # Mid price - split evenly or use other heuristics
                lit_buy_volume += premium / 2
                lit_sell_volume += premium / 2

        lit_count = len(recent_lit)

    # Combined sentiment score
    total_options = bullish_premium + bearish_premium
    total_lit = lit_buy_volume + lit_sell_volume

    options_score = (bullish_premium - bearish_premium) / total_options if total_options > 0 else 0
    lit_score = (lit_buy_volume - lit_sell_volume) / total_lit if total_lit > 0 else 0

    # Weight options flow more heavily (70/30)
    if total_options > 0 and total_lit > 0:
        sentiment_score = 0.7 * options_score + 0.3 * lit_score
    elif total_options > 0:
        sentiment_score = options_score
    elif total_lit > 0:
        sentiment_score = lit_score
    else:
        sentiment_score = 0.0

    if sentiment_score > 0.2:
        net_sentiment = "bullish"
    elif sentiment_score < -0.2:
        net_sentiment = "bearish"
    else:
        net_sentiment = "neutral"

    return {
        "bullish_premium": float(bullish_premium),
        "bearish_premium": float(bearish_premium),
        "lit_buy_volume": float(lit_buy_volume),
        "lit_sell_volume": float(lit_sell_volume),
        "net_sentiment": net_sentiment,
        "sentiment_score": float(sentiment_score),
        "options_count": options_count,
        "lit_count": lit_count,
        "flow_count": options_count + lit_count,
    }


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

    return {
        "direction": direction,
        "confidence": float(confidence),
        "expected_move": float(median_end - current_price),
        "forecast_median": float(median_end),
        "forecast_q10": float(np.percentile(end_prices, 10)),
        "forecast_q90": float(np.percentile(end_prices, 90)),
    }


def run_backtest(
    bars_1s: List[Dict],
    options_df: pd.DataFrame,
    lit_df: pd.DataFrame,
    prediction_length: int = 20,
    lookback_1min: int = 60,
    step_1min: int = 1,
    confidence_threshold: float = 0.55,
) -> List[Dict[str, Any]]:
    """Run the backtest."""
    print(f"[Backtest] Aggregating {len(bars_1s)} 1s bars to 1-minute...", file=sys.stderr)
    bars_1min, cvd_1min, cvd_ema_1min = aggregate_1s_to_1min(bars_1s)
    print(f"[Backtest] Got {len(bars_1min)} 1-minute bars", file=sys.stderr)

    if len(bars_1min) < lookback_1min + prediction_length:
        print("[Backtest] Not enough data", file=sys.stderr)
        return []

    predictions = []
    price_series = [b['c'] for b in bars_1min]

    print(f"[Backtest] Running predictions (lookback={lookback_1min}, pred_len={prediction_length})...", file=sys.stderr)

    for i in range(lookback_1min, len(bars_1min) - prediction_length, step_1min):
        current_bar = bars_1min[i - 1]
        current_price = current_bar['c']
        timestamp_str = current_bar['t']

        # Parse timestamp for whale lookup
        if isinstance(timestamp_str, str):
            # Handle timezone-aware ISO format
            ts = datetime.fromisoformat(timestamp_str.replace('+00:00', ''))
        else:
            ts = timestamp_str

        # Get CVD trend at 1-min resolution
        cvd_now = cvd_1min[i - 1] if i - 1 < len(cvd_1min) else 0
        cvd_ema_now = cvd_ema_1min[i - 1] if i - 1 < len(cvd_ema_1min) else 0

        if cvd_now > cvd_ema_now:
            cvd_trend = "up"
        elif cvd_now < cvd_ema_now:
            cvd_trend = "down"
        else:
            cvd_trend = "neutral"

        # Volume Profile from last 30 1-min bars
        vp = calculate_volume_profile(bars_1min[:i], lookback=30)

        # Whale sentiment (options + lit flow)
        whale = get_whale_sentiment(options_df, lit_df, ts, lookback_minutes=30)

        # Chronos prediction
        chronos = predict_with_chronos(
            price_series[:i],
            prediction_length=prediction_length,
        )

        # Combined signal
        final_direction = chronos["direction"]
        combined_confidence = chronos["confidence"]

        # Boost if CVD agrees
        if chronos["direction"] == cvd_trend:
            combined_confidence = min(combined_confidence + 0.1, 1.0)

        # Boost if whale flow agrees
        if chronos["direction"] == "up" and whale["net_sentiment"] == "bullish":
            combined_confidence = min(combined_confidence + 0.1, 1.0)
        elif chronos["direction"] == "down" and whale["net_sentiment"] == "bearish":
            combined_confidence = min(combined_confidence + 0.1, 1.0)

        pred = {
            "timestamp": timestamp_str,
            "bar_index": i - 1,
            "current_price": float(current_price),
            "direction": final_direction,
            "confidence": float(combined_confidence),
            "chronos_direction": chronos["direction"],
            "chronos_confidence": chronos["confidence"],
            "cvd_trend": cvd_trend,
            "cvd": float(cvd_now),
            "cvd_ema": float(cvd_ema_now),
            "poc": float(vp["poc"]),
            "vah": float(vp["vah"]),
            "val": float(vp["val"]),
            "whale_sentiment": whale["net_sentiment"],
            "whale_score": whale["sentiment_score"],
            "whale_bullish_premium": whale["bullish_premium"],
            "whale_bearish_premium": whale["bearish_premium"],
            "expected_move": chronos["expected_move"],
            "pass": combined_confidence >= confidence_threshold,
        }

        # Calculate actual outcome
        if i + prediction_length <= len(bars_1min):
            actual_future = bars_1min[i + prediction_length - 1]['c']
            actual_move = actual_future - current_price
            pred["actual_move"] = float(actual_move)
            pred["actual_direction"] = "up" if actual_move > 0 else "down" if actual_move < 0 else "neutral"
            pred["prediction_correct"] = (
                (pred["direction"] == "up" and actual_move > 0) or
                (pred["direction"] == "down" and actual_move < 0)
            )

        predictions.append(pred)

        if len(predictions) % 50 == 0:
            print(f"[Backtest] Processed {len(predictions)} predictions...", file=sys.stderr)

    print(f"[Backtest] Generated {len(predictions)} predictions", file=sys.stderr)
    return predictions


def analyze_results(predictions: List[Dict], confidence_threshold: float = 0.55) -> Dict[str, Any]:
    """Analyze backtest results."""
    with_outcomes = [p for p in predictions if "actual_direction" in p]

    if not with_outcomes:
        return {"error": "No predictions with outcomes"}

    total = len(with_outcomes)
    correct = sum(1 for p in with_outcomes if p.get("prediction_correct", False))

    # By CVD trend
    cvd_up = [p for p in with_outcomes if p["cvd_trend"] == "up"]
    cvd_up_correct = sum(1 for p in cvd_up if p.get("prediction_correct", False))

    cvd_down = [p for p in with_outcomes if p["cvd_trend"] == "down"]
    cvd_down_correct = sum(1 for p in cvd_down if p.get("prediction_correct", False))

    # By whale sentiment
    whale_bullish = [p for p in with_outcomes if p["whale_sentiment"] == "bullish"]
    whale_bullish_correct = sum(1 for p in whale_bullish if p.get("prediction_correct", False))

    whale_bearish = [p for p in with_outcomes if p["whale_sentiment"] == "bearish"]
    whale_bearish_correct = sum(1 for p in whale_bearish if p.get("prediction_correct", False))

    # CVD up + Long
    cvd_up_long = [p for p in with_outcomes if p["cvd_trend"] == "up" and p["direction"] == "up"]
    cvd_up_long_correct = sum(1 for p in cvd_up_long if p.get("prediction_correct", False))

    # High confidence
    high_conf = [p for p in with_outcomes if p["confidence"] >= confidence_threshold]
    high_conf_correct = sum(1 for p in high_conf if p.get("prediction_correct", False))

    # By direction
    longs = [p for p in with_outcomes if p["direction"] == "up"]
    longs_correct = sum(1 for p in longs if p.get("prediction_correct", False))

    shorts = [p for p in with_outcomes if p["direction"] == "down"]
    shorts_correct = sum(1 for p in shorts if p.get("prediction_correct", False))

    return {
        "total_predictions": total,
        "overall_accuracy": round(correct / total * 100, 2) if total > 0 else 0,

        "high_confidence": {
            "count": len(high_conf),
            "accuracy": round(high_conf_correct / len(high_conf) * 100, 2) if high_conf else 0,
        },

        "by_direction": {
            "long": {
                "count": len(longs),
                "accuracy": round(longs_correct / len(longs) * 100, 2) if longs else 0,
            },
            "short": {
                "count": len(shorts),
                "accuracy": round(shorts_correct / len(shorts) * 100, 2) if shorts else 0,
            },
        },

        "by_cvd_trend": {
            "up": {
                "count": len(cvd_up),
                "accuracy": round(cvd_up_correct / len(cvd_up) * 100, 2) if cvd_up else 0,
            },
            "down": {
                "count": len(cvd_down),
                "accuracy": round(cvd_down_correct / len(cvd_down) * 100, 2) if cvd_down else 0,
            },
        },

        "cvd_up_long": {
            "count": len(cvd_up_long),
            "accuracy": round(cvd_up_long_correct / len(cvd_up_long) * 100, 2) if cvd_up_long else 0,
        },

        "by_whale_sentiment": {
            "bullish": {
                "count": len(whale_bullish),
                "accuracy": round(whale_bullish_correct / len(whale_bullish) * 100, 2) if whale_bullish else 0,
            },
            "bearish": {
                "count": len(whale_bearish),
                "accuracy": round(whale_bearish_correct / len(whale_bearish) * 100, 2) if whale_bearish else 0,
            },
        },
    }


def main():
    # Paths
    bars_1s_path = "data/bars_1s.json"
    options_flow_path = "data/spy_options_flow_20251205.csv"
    lit_flow_path = "data/spy_lit_flow_20251205.csv"
    output_path = "data/chronos_whale_results.json"

    print("=" * 70, file=sys.stderr)
    print("CHRONOS + CVD/VP + WHALE FLOW BACKTEST", file=sys.stderr)
    print("1-Day Test: 12/5/2025 Only", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Load data
    print(f"[Load] Loading 1s bars from {bars_1s_path}...", file=sys.stderr)
    bars_1s_all = load_1s_bars(bars_1s_path)
    print(f"[Load] Loaded {len(bars_1s_all)} total 1-second bars", file=sys.stderr)

    # Filter to only 12/5/2025
    bars_1s = [b for b in bars_1s_all if '2025-12-05' in b['t']]
    print(f"[Load] Filtered to {len(bars_1s)} bars on 12/5/2025", file=sys.stderr)

    print(f"[Load] Loading whale options flow from {options_flow_path}...", file=sys.stderr)
    options_df = load_whale_options_flow(options_flow_path)
    print(f"[Load] Loaded {len(options_df)} options flow records", file=sys.stderr)

    print(f"[Load] Loading whale lit flow from {lit_flow_path}...", file=sys.stderr)
    lit_df = load_whale_lit_flow(lit_flow_path)
    print(f"[Load] Loaded {len(lit_df)} lit flow records", file=sys.stderr)

    # Run backtest
    predictions = run_backtest(
        bars_1s=bars_1s,
        options_df=options_df,
        lit_df=lit_df,
        prediction_length=20,
        lookback_1min=60,
        step_1min=5,  # Every 5 minutes to speed up
        confidence_threshold=0.55,
    )

    # Analyze
    analysis = analyze_results(predictions, 0.55)

    # Print results
    print("\n" + "=" * 70, file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total predictions: {analysis['total_predictions']}", file=sys.stderr)
    print(f"Overall accuracy: {analysis['overall_accuracy']}%", file=sys.stderr)

    print(f"\nHigh Confidence (>=55%):", file=sys.stderr)
    print(f"  Count: {analysis['high_confidence']['count']}", file=sys.stderr)
    print(f"  Accuracy: {analysis['high_confidence']['accuracy']}%", file=sys.stderr)

    print(f"\nBy Direction:", file=sys.stderr)
    print(f"  LONG: {analysis['by_direction']['long']['count']} trades, {analysis['by_direction']['long']['accuracy']}% accurate", file=sys.stderr)
    print(f"  SHORT: {analysis['by_direction']['short']['count']} trades, {analysis['by_direction']['short']['accuracy']}% accurate", file=sys.stderr)

    print(f"\nBy CVD Trend:", file=sys.stderr)
    print(f"  CVD UP: {analysis['by_cvd_trend']['up']['count']} trades, {analysis['by_cvd_trend']['up']['accuracy']}% accurate", file=sys.stderr)
    print(f"  CVD DOWN: {analysis['by_cvd_trend']['down']['count']} trades, {analysis['by_cvd_trend']['down']['accuracy']}% accurate", file=sys.stderr)

    print(f"\nCVD UP + LONG (your best setup):", file=sys.stderr)
    print(f"  Count: {analysis['cvd_up_long']['count']}", file=sys.stderr)
    print(f"  Accuracy: {analysis['cvd_up_long']['accuracy']}%", file=sys.stderr)

    print(f"\nBy Whale Sentiment:", file=sys.stderr)
    print(f"  BULLISH: {analysis['by_whale_sentiment']['bullish']['count']} trades, {analysis['by_whale_sentiment']['bullish']['accuracy']}% accurate", file=sys.stderr)
    print(f"  BEARISH: {analysis['by_whale_sentiment']['bearish']['count']} trades, {analysis['by_whale_sentiment']['bearish']['accuracy']}% accurate", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Save results
    result = {
        "analysis": analysis,
        "predictions": predictions,
        "config": {
            "prediction_length": 20,
            "lookback_1min": 60,
            "model": _model_name,
            "date": "2025-12-05",
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Save] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
