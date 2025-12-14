#!/usr/bin/env python3
"""
Chronos-2 Pre-Filter for Trading Decisions
Uses Amazon's Chronos-2 foundation model to predict price movement
before passing to the Qwen trading agent.

Usage:
    echo '{"prices": [100.5, 101.2, ...], "symbol": "NQ"}' | python chronos_filter.py

Returns:
    {"pass": true/false, "direction": "up/down/neutral", "confidence": 0.0-1.0, ...}
"""

import json
import sys
import os
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

# Lazy load Chronos to avoid startup overhead when not needed
_pipeline = None
_model_name = "amazon/chronos-t5-small"  # Options: tiny, mini, small, base, large


def get_pipeline():
    """Lazy load the Chronos pipeline."""
    global _pipeline
    if _pipeline is None:
        from chronos import ChronosPipeline

        # Use GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[Chronos] Loading model {_model_name} on {device}...", file=sys.stderr)
        _pipeline = ChronosPipeline.from_pretrained(
            _model_name,
            device_map=device,
            dtype=dtype,  # Updated: torch_dtype -> dtype
        )
        print(f"[Chronos] Model loaded successfully", file=sys.stderr)

    return _pipeline


def predict_direction(
    prices: List[float],
    prediction_length: int = 5,
    num_samples: int = 20,
    confidence_threshold: float = 0.55,
) -> dict:
    """
    Predict price direction using Chronos-2.

    Args:
        prices: Historical price series (at least 30 data points recommended)
        prediction_length: Number of future steps to predict
        num_samples: Number of forecast samples for uncertainty estimation
        confidence_threshold: Minimum confidence to pass filter (0.5-1.0)

    Returns:
        dict with prediction results
    """
    if len(prices) < 10:
        return {
            "pass": False,
            "direction": "neutral",
            "confidence": 0.0,
            "reason": "insufficient_data",
            "min_required": 10,
            "provided": len(prices),
        }

    try:
        pipeline = get_pipeline()

        # Convert to tensor
        context = torch.tensor(prices, dtype=torch.float32)

        # Generate probabilistic forecasts
        forecast = pipeline.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )

        # forecast shape: (batch=1, num_samples, prediction_length)
        # Squeeze batch dimension and convert to numpy
        forecast_np = forecast.squeeze(0).cpu().numpy()  # -> (num_samples, prediction_length)

        # Calculate statistics
        current_price = prices[-1]
        median_forecast = np.median(forecast_np, axis=0)
        mean_forecast = np.mean(forecast_np, axis=0)

        # End price predictions
        end_prices = forecast_np[:, -1]  # All samples at final step
        median_end = np.median(end_prices)

        # Direction voting across samples
        up_samples = np.sum(end_prices > current_price)
        down_samples = np.sum(end_prices < current_price)
        total_samples = len(end_prices)

        # Calculate directional confidence
        if up_samples > down_samples:
            direction = "up"
            confidence = up_samples / total_samples
        elif down_samples > up_samples:
            direction = "down"
            confidence = down_samples / total_samples
        else:
            direction = "neutral"
            confidence = 0.5

        # Calculate expected move (in points/ticks)
        expected_move = median_end - current_price
        expected_move_pct = (expected_move / current_price) * 100

        # Uncertainty: interquartile range
        iqr = np.percentile(end_prices, 75) - np.percentile(end_prices, 25)
        uncertainty = iqr / current_price  # Normalized

        # Pass filter if confidence exceeds threshold
        passes_filter = bool(confidence >= confidence_threshold)

        # Calculate quantiles for range
        q10 = float(np.percentile(end_prices, 10))
        q50 = float(np.percentile(end_prices, 50))
        q90 = float(np.percentile(end_prices, 90))

        return {
            "pass": passes_filter,
            "direction": direction,
            "confidence": float(round(confidence, 3)),
            "expected_move": float(round(expected_move, 2)),
            "expected_move_pct": float(round(expected_move_pct, 4)),
            "current_price": float(current_price),
            "forecast_median": float(round(median_end, 2)),
            "forecast_q10": float(round(q10, 2)),
            "forecast_q90": float(round(q90, 2)),
            "uncertainty": float(round(uncertainty, 4)),
            "prediction_steps": int(prediction_length),
            "samples_up": int(up_samples),
            "samples_down": int(down_samples),
            "model": _model_name,
        }

    except Exception as e:
        return {
            "pass": False,
            "direction": "neutral",
            "confidence": 0.0,
            "reason": "prediction_error",
            "error": str(e),
        }


def main():
    """Read JSON from stdin and output prediction."""
    try:
        # Read input
        input_data = sys.stdin.read().strip()
        if not input_data:
            print(json.dumps({"error": "No input provided"}))
            sys.exit(1)

        data = json.loads(input_data)

        # Extract prices
        prices = data.get("prices", [])
        if not prices and "candles" in data:
            # Support candle format
            prices = [c.get("close", c.get("c", 0)) for c in data["candles"]]

        if not prices:
            print(json.dumps({"error": "No prices provided", "pass": False}))
            sys.exit(1)

        # Get optional parameters
        prediction_length = data.get("prediction_length", 5)
        confidence_threshold = data.get("confidence_threshold", 0.55)
        num_samples = data.get("num_samples", 20)

        # Run prediction
        result = predict_direction(
            prices=prices,
            prediction_length=prediction_length,
            num_samples=num_samples,
            confidence_threshold=confidence_threshold,
        )

        # Add symbol if provided
        if "symbol" in data:
            result["symbol"] = data["symbol"]

        print(json.dumps(result))

    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}", "pass": False}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e), "pass": False}))
        sys.exit(1)


if __name__ == "__main__":
    main()
