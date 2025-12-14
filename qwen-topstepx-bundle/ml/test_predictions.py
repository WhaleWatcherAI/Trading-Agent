#!/usr/bin/env python3
"""
Test the trained models with proper feature format.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add scripts to path
sys.path.append(str(Path(__file__).resolve().parents[0] / "scripts"))

from predict_advanced import predict_from_snapshot

def test_models():
    """Test the trained models with complete feature set."""

    # Load the parquet file to see what features are expected
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "ml" / "data"

    df = pd.read_parquet(DATA_DIR / "meta_label.parquet")
    print(f"Dataset shape: {df.shape}")
    print(f"Features in dataset: {list(df.columns)}")

    # Get a sample row to see the feature structure
    sample_row = df.iloc[0]

    # Create a complete snapshot with all features
    snapshot = {
        'symbol': 'NQ',
        'timestamp': '2025-11-30T12:00:00Z',
        'features': {
            'rsi': 45.0,
            'macd': 2.5,
            'bb_upper': 16520.0,
            'bb_lower': 16480.0,
            'sma_20': 16500.0,
            'sma_50': 16490.0,
            'volume_ratio': 1.2,
            'volatility': 0.015,
            'dist_to_poc_ticks': 4.0,
            'macd_signal': 2.0,
            'bb_position': 0.5,
            'atr': 25.0,
            'vwap': 16505.0,
            'confidence': 0.65,
            'price': 16500.0
        },
        'position': 0,
        'balance': 100000,
        'unrealized_pnl': 0,
        'recent_actions': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    }

    print("\n" + "="*60)
    print("TESTING ML MODELS")
    print("="*60)

    try:
        # Get predictions
        predictions = predict_from_snapshot(snapshot)

        print("\nPredictions received:")
        print(json.dumps(predictions, indent=2))

        # Print summary
        print("\n" + "="*60)
        print("TRADING SIGNAL SUMMARY")
        print("="*60)

        if 'ensemble_recommendation' in predictions:
            print(f"Recommendation: {predictions['ensemble_recommendation']}")
            print(f"Confidence: {predictions.get('confidence_score', 0):.2%}")

        if 'win_5m_prob' in predictions:
            print(f"\nLightGBM Win Probabilities:")
            print(f"  5-minute: {predictions['win_5m_prob']:.2%}")
            if 'win_30m_prob' in predictions:
                print(f"  30-minute: {predictions['win_30m_prob']:.2%}")

        if 'lstm_buy_prob' in predictions:
            print(f"\nLSTM Predictions:")
            print(f"  Buy: {predictions['lstm_buy_prob']:.2%}")
            print(f"  Sell: {predictions['lstm_sell_prob']:.2%}")
            print(f"  Hold: {predictions['lstm_hold_prob']:.2%}")

        print("\n✅ Models are working!")

    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

    # Test with different market conditions
    print("\n" + "="*60)
    print("TESTING DIFFERENT MARKET CONDITIONS")
    print("="*60)

    # Oversold condition
    snapshot['features']['rsi'] = 25
    predictions = predict_from_snapshot(snapshot)
    print(f"\nOversold (RSI=25): {predictions.get('ensemble_recommendation', 'N/A')}")

    # Overbought condition
    snapshot['features']['rsi'] = 75
    predictions = predict_from_snapshot(snapshot)
    print(f"Overbought (RSI=75): {predictions.get('ensemble_recommendation', 'N/A')}")

    # Strong momentum
    snapshot['features']['rsi'] = 60
    snapshot['features']['macd'] = 10
    predictions = predict_from_snapshot(snapshot)
    print(f"Strong Momentum: {predictions.get('ensemble_recommendation', 'N/A')}")

if __name__ == "__main__":
    test_models()