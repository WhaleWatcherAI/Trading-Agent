#!/usr/bin/env python3
"""
Quick test for Chronos filter installation and functionality.
"""

import json
import sys
import numpy as np

def test_import():
    """Test that chronos can be imported."""
    print("Testing Chronos import...", end=" ")
    try:
        from chronos import ChronosPipeline
        print("OK")
        return True
    except ImportError as e:
        print(f"FAILED: {e}")
        print("\nInstall with: pip install chronos-forecasting transformers")
        return False

def test_prediction():
    """Test a simple prediction."""
    print("Testing Chronos prediction...", end=" ")
    try:
        import torch
        from chronos import ChronosPipeline

        # Use tiny model for quick test
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            dtype=torch.float32,  # Updated: torch_dtype -> dtype
        )

        # Sample price data (simulated NQ prices)
        prices = [
            21100, 21105, 21110, 21108, 21115,
            21120, 21118, 21125, 21130, 21128,
            21135, 21140, 21138, 21145, 21150,
            21148, 21155, 21160, 21158, 21165,
        ]

        context = torch.tensor(prices, dtype=torch.float32)
        forecast = pipeline.predict(context, prediction_length=5, num_samples=10)

        # forecast shape: (batch=1, num_samples, prediction_length)
        # Squeeze batch and convert to numpy
        forecast_np = forecast.squeeze(0).cpu().numpy()  # -> (num_samples, prediction_length)

        # Get median across samples for each prediction step
        median = np.median(forecast_np, axis=0)  # -> (prediction_length,)
        current = prices[-1]
        predicted_end = float(median[-1])

        direction = "UP" if predicted_end > current else "DOWN"
        move = predicted_end - current

        print("OK")
        print(f"  Current price: {current}")
        print(f"  Predicted (5 steps): {predicted_end:.2f}")
        print(f"  Direction: {direction} ({move:+.2f} points)")
        return True

    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()
        return False

def test_filter_script():
    """Test the actual filter script."""
    print("Testing chronos_filter.py...", end=" ")
    try:
        import subprocess
        import os

        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "chronos_filter.py")

        # Sample input
        input_data = json.dumps({
            "prices": [21100 + i * 5 + (i % 3) for i in range(30)],
            "symbol": "NQ",
            "prediction_length": 3,
            "num_samples": 10,
        })

        result = subprocess.run(
            [sys.executable, script_path],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"FAILED: {result.stderr}")
            return False

        output = json.loads(result.stdout.strip())
        print("OK")
        print(f"  Result: {json.dumps(output, indent=2)}")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Chronos Filter Test Suite")
    print("=" * 50)

    results = []

    results.append(("Import", test_import()))

    if results[-1][1]:
        results.append(("Prediction", test_prediction()))
        results.append(("Filter Script", test_filter_script()))

    print("\n" + "=" * 50)
    print("Results:")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)
