#!/usr/bin/env python3
"""
Chronos Backtest Integration
Analyzes historical price data and generates Chronos predictions
that can be used as a pre-filter in the TypeScript backtester.

Usage:
    # Generate predictions for historical data
    python chronos_backtest.py --input market_data.parquet --output chronos_signals.json

    # Or pipe price data directly
    echo '{"prices": [...], "timestamps": [...]}' | python chronos_backtest.py
"""

import argparse
import json
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
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


def predict_single(
    prices: List[float],
    prediction_length: int = 5,
    num_samples: int = 20,
) -> Dict[str, Any]:
    """Generate a single prediction from price history."""

    if len(prices) < 10:
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "expected_move": 0.0,
            "pass": False,
        }

    pipeline = get_pipeline()
    context = torch.tensor(prices, dtype=torch.float32)

    forecast = pipeline.predict(
        context,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )

    # Shape: (1, num_samples, prediction_length) -> (num_samples, prediction_length)
    forecast_np = forecast.squeeze(0).cpu().numpy()

    current_price = prices[-1]
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
        "expected_move_pct": float((expected_move / current_price) * 100),
        "forecast_median": float(median_end),
        "forecast_q10": float(np.percentile(end_prices, 10)),
        "forecast_q90": float(np.percentile(end_prices, 90)),
    }


def generate_rolling_predictions(
    prices: List[float],
    timestamps: List[str],
    lookback: int = 60,
    prediction_length: int = 5,
    step: int = 1,
    confidence_threshold: float = 0.55,
) -> List[Dict[str, Any]]:
    """
    Generate rolling Chronos predictions over historical data.

    Args:
        prices: Historical price series
        timestamps: Corresponding timestamps
        lookback: Number of bars to use for each prediction
        prediction_length: Forward prediction horizon
        step: Generate prediction every N bars
        confidence_threshold: Threshold for "pass" flag

    Returns:
        List of predictions with timestamps
    """
    predictions = []
    total = len(prices)

    print(f"[Chronos] Generating predictions for {total} bars (lookback={lookback}, step={step})...", file=sys.stderr)

    for i in range(lookback, total, step):
        price_window = prices[i - lookback:i]

        pred = predict_single(price_window, prediction_length)
        pred["timestamp"] = timestamps[i - 1] if i - 1 < len(timestamps) else None
        pred["bar_index"] = i - 1
        pred["current_price"] = float(prices[i - 1])
        pred["pass"] = pred["confidence"] >= confidence_threshold

        # Calculate actual outcome if we have future data
        if i + prediction_length <= total:
            actual_future = prices[i + prediction_length - 1]
            actual_move = actual_future - prices[i - 1]
            pred["actual_move"] = float(actual_move)
            pred["actual_direction"] = "up" if actual_move > 0 else "down" if actual_move < 0 else "neutral"
            pred["prediction_correct"] = (
                (pred["direction"] == "up" and actual_move > 0) or
                (pred["direction"] == "down" and actual_move < 0)
            )

        predictions.append(pred)

        if len(predictions) % 1000 == 0:
            print(f"[Chronos] Processed {len(predictions)} predictions...", file=sys.stderr)

    print(f"[Chronos] Generated {len(predictions)} predictions", file=sys.stderr)
    return predictions


def analyze_predictions(predictions: List[Dict[str, Any]], confidence_threshold: float = 0.55) -> Dict[str, Any]:
    """Analyze prediction accuracy and filter effectiveness."""

    # Only analyze predictions with actual outcomes
    with_outcomes = [p for p in predictions if "actual_direction" in p]

    if not with_outcomes:
        return {"error": "No predictions with actual outcomes"}

    total = len(with_outcomes)
    correct = sum(1 for p in with_outcomes if p.get("prediction_correct", False))

    # High confidence predictions (would pass filter)
    high_conf = [p for p in with_outcomes if p["confidence"] >= confidence_threshold]
    high_conf_correct = sum(1 for p in high_conf if p.get("prediction_correct", False))

    # Low confidence predictions (would be filtered)
    low_conf = [p for p in with_outcomes if p["confidence"] < confidence_threshold]
    low_conf_correct = sum(1 for p in low_conf if p.get("prediction_correct", False))

    # Direction breakdown
    up_preds = [p for p in with_outcomes if p["direction"] == "up"]
    down_preds = [p for p in with_outcomes if p["direction"] == "down"]

    up_correct = sum(1 for p in up_preds if p.get("prediction_correct", False))
    down_correct = sum(1 for p in down_preds if p.get("prediction_correct", False))

    return {
        "total_predictions": total,
        "overall_accuracy": round(correct / total * 100, 2) if total > 0 else 0,

        "high_confidence": {
            "count": len(high_conf),
            "pct_of_total": round(len(high_conf) / total * 100, 2) if total > 0 else 0,
            "accuracy": round(high_conf_correct / len(high_conf) * 100, 2) if high_conf else 0,
            "correct": high_conf_correct,
        },

        "low_confidence_filtered": {
            "count": len(low_conf),
            "pct_of_total": round(len(low_conf) / total * 100, 2) if total > 0 else 0,
            "accuracy": round(low_conf_correct / len(low_conf) * 100, 2) if low_conf else 0,
            "correct": low_conf_correct,
        },

        "by_direction": {
            "up": {
                "count": len(up_preds),
                "accuracy": round(up_correct / len(up_preds) * 100, 2) if up_preds else 0,
            },
            "down": {
                "count": len(down_preds),
                "accuracy": round(down_correct / len(down_preds) * 100, 2) if down_preds else 0,
            },
        },

        "filter_benefit": {
            "description": "Would filtering improve results?",
            "filtered_out_losers": len(low_conf) - low_conf_correct,
            "filtered_out_winners": low_conf_correct,
            "net_trades_avoided": len(low_conf),
        },

        "confidence_threshold": confidence_threshold,
    }


def load_market_data(filepath: str) -> tuple:
    """Load market data from parquet file."""

    df = pd.read_parquet(filepath)

    # Try common column names
    price_col = None
    time_col = None

    for col in ["close", "Close", "price", "Price"]:
        if col in df.columns:
            price_col = col
            break

    for col in ["timestamp", "Timestamp", "time", "Time", "date", "Date"]:
        if col in df.columns:
            time_col = col
            break

    if price_col is None:
        raise ValueError(f"Could not find price column. Available: {list(df.columns)}")

    prices = df[price_col].tolist()

    if time_col:
        timestamps = df[time_col].astype(str).tolist()
    else:
        timestamps = [str(i) for i in range(len(prices))]

    return prices, timestamps


def main():
    parser = argparse.ArgumentParser(description="Chronos Backtest Integration")
    parser.add_argument("--input", "-i", help="Input parquet/csv file with price data")
    parser.add_argument("--output", "-o", help="Output JSON file for predictions")
    parser.add_argument("--lookback", type=int, default=60, help="Bars for prediction context")
    parser.add_argument("--prediction-length", type=int, default=5, help="Forward prediction horizon")
    parser.add_argument("--step", type=int, default=1, help="Generate prediction every N bars")
    parser.add_argument("--confidence", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--analyze-only", action="store_true", help="Only show analysis, don't save predictions")

    args = parser.parse_args()

    # Load data
    if args.input:
        print(f"[Chronos] Loading data from {args.input}...", file=sys.stderr)
        prices, timestamps = load_market_data(args.input)
        print(f"[Chronos] Loaded {len(prices)} price points", file=sys.stderr)
    else:
        # Read from stdin
        input_data = sys.stdin.read().strip()
        if not input_data:
            print("Error: No input provided. Use --input or pipe JSON data.", file=sys.stderr)
            sys.exit(1)

        data = json.loads(input_data)
        prices = data.get("prices", [])
        timestamps = data.get("timestamps", [str(i) for i in range(len(prices))])

    if len(prices) < args.lookback + args.prediction_length:
        print(f"Error: Need at least {args.lookback + args.prediction_length} price points", file=sys.stderr)
        sys.exit(1)

    # Generate predictions
    predictions = generate_rolling_predictions(
        prices=prices,
        timestamps=timestamps,
        lookback=args.lookback,
        prediction_length=args.prediction_length,
        step=args.step,
        confidence_threshold=args.confidence,
    )

    # Analyze
    analysis = analyze_predictions(predictions, args.confidence)

    # Print analysis
    print("\n" + "=" * 60, file=sys.stderr)
    print("CHRONOS BACKTEST ANALYSIS", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Total predictions: {analysis['total_predictions']}", file=sys.stderr)
    print(f"Overall accuracy: {analysis['overall_accuracy']}%", file=sys.stderr)
    print(f"\nHigh confidence (>={args.confidence * 100:.0f}%):", file=sys.stderr)
    print(f"  Count: {analysis['high_confidence']['count']} ({analysis['high_confidence']['pct_of_total']}% of total)", file=sys.stderr)
    print(f"  Accuracy: {analysis['high_confidence']['accuracy']}%", file=sys.stderr)
    print(f"\nLow confidence (filtered):", file=sys.stderr)
    print(f"  Count: {analysis['low_confidence_filtered']['count']} ({analysis['low_confidence_filtered']['pct_of_total']}% of total)", file=sys.stderr)
    print(f"  Accuracy: {analysis['low_confidence_filtered']['accuracy']}%", file=sys.stderr)
    print(f"\nBy direction:", file=sys.stderr)
    print(f"  UP: {analysis['by_direction']['up']['count']} predictions, {analysis['by_direction']['up']['accuracy']}% accurate", file=sys.stderr)
    print(f"  DOWN: {analysis['by_direction']['down']['count']} predictions, {analysis['by_direction']['down']['accuracy']}% accurate", file=sys.stderr)
    print(f"\nFilter benefit:", file=sys.stderr)
    print(f"  Trades avoided: {analysis['filter_benefit']['net_trades_avoided']}", file=sys.stderr)
    print(f"  Losers avoided: {analysis['filter_benefit']['filtered_out_losers']}", file=sys.stderr)
    print(f"  Winners missed: {analysis['filter_benefit']['filtered_out_winners']}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Output
    if not args.analyze_only:
        result = {
            "analysis": analysis,
            "predictions": predictions,
            "config": {
                "lookback": args.lookback,
                "prediction_length": args.prediction_length,
                "step": args.step,
                "confidence_threshold": args.confidence,
                "model": _model_name,
            },
        }

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n[Chronos] Saved predictions to {args.output}", file=sys.stderr)
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
