#!/usr/bin/env python3
"""
Example usage of LSTM and PPO models for trading decisions.
This script demonstrates how to integrate the ML models into your trading system.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

# Add scripts directory to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "ml" / "scripts"))

from predict_advanced import EnsemblePredictor, predict_from_snapshot

def generate_sample_data():
    """Generate sample market data for demonstration."""
    np.random.seed(42)

    # Generate 100 time points of market data
    timestamps = [datetime.utcnow() - timedelta(minutes=i) for i in range(100, 0, -1)]

    # Simulate price movement
    base_price = 16500
    prices = [base_price]
    for _ in range(99):
        change = np.random.normal(0, 10)
        prices.append(prices[-1] + change)

    # Generate technical indicators
    data = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'volume': np.random.randint(500, 2000, 100),
        'rsi': np.random.uniform(30, 70, 100),
        'macd': np.random.normal(0, 5, 100),
        'bb_upper': [p + 20 for p in prices],
        'bb_lower': [p - 20 for p in prices],
        'vwap': [p + np.random.normal(0, 2) for p in prices],
        'dist_to_poc_ticks': np.random.randint(-10, 10, 100)
    })

    return data

def prepare_trading_snapshot(data: pd.DataFrame, current_idx: int, position: int = 0,
                            balance: float = 100000) -> dict:
    """
    Prepare a trading snapshot for ML prediction.

    Args:
        data: Historical market data
        current_idx: Current time index
        position: Current position size
        balance: Current account balance
    """
    # Get current features
    current_data = data.iloc[current_idx]
    features = {
        'close': float(current_data['close']),
        'volume': float(current_data['volume']),
        'rsi': float(current_data['rsi']),
        'macd': float(current_data['macd']),
        'bb_upper': float(current_data['bb_upper']),
        'bb_lower': float(current_data['bb_lower']),
        'vwap': float(current_data['vwap']),
        'dist_to_poc_ticks': float(current_data['dist_to_poc_ticks'])
    }

    # Prepare sequence for LSTM (last 60 data points)
    sequence_start = max(0, current_idx - 59)
    sequence_data = data.iloc[sequence_start:current_idx + 1]

    sequence = []
    for _, row in sequence_data.iterrows():
        sequence.append([
            row['close'], row['volume'], row['rsi'],
            row['macd'], row['vwap']
        ])

    # Calculate unrealized PnL
    if position != 0:
        entry_price = data.iloc[current_idx - 1]['close']  # Simplified
        current_price = current_data['close']
        unrealized_pnl = position * (current_price - entry_price)
    else:
        unrealized_pnl = 0

    snapshot = {
        'symbol': 'NQ',
        'timestamp': current_data['timestamp'].isoformat(),
        'features': features,
        'sequence': sequence,
        'position': position,
        'balance': balance,
        'unrealized_pnl': unrealized_pnl,
        'recent_actions': [0, 1, 0, 2, 0, 0, 1, 0, 0, 0],  # Example action history
        'timestep': current_idx
    }

    return snapshot

def simulate_trading_session():
    """Simulate a trading session using ML predictions."""
    print("="*60)
    print("TRADING SIMULATION WITH LSTM & PPO")
    print("="*60)

    # Generate sample data
    data = generate_sample_data()
    print(f"\nGenerated {len(data)} time points of market data")

    # Initialize trading state
    balance = 100000
    position = 0
    trades = []

    # Create predictor
    predictor = EnsemblePredictor()
    print(f"Loaded models: {list(predictor.models_loaded.keys())}")

    # Simulate trading from index 60 onwards (need history for LSTM)
    for idx in range(60, len(data)):
        # Prepare snapshot
        snapshot = prepare_trading_snapshot(data, idx, position, balance)

        # Get ML predictions
        predictions = predict_from_snapshot(snapshot)

        # Trading logic based on predictions
        action = predictions['ensemble_recommendation']
        confidence = predictions['confidence_score']

        current_price = data.iloc[idx]['close']

        # Execute trades with confidence threshold
        if confidence > 0.6:  # Only trade when confident
            if action == 'BUY' and position < 5:
                # Buy logic
                position += 1
                balance -= current_price
                trades.append({
                    'time': data.iloc[idx]['timestamp'],
                    'action': 'BUY',
                    'price': current_price,
                    'position': position,
                    'confidence': confidence
                })
                print(f"\n[{idx}] BUY at ${current_price:.2f} (confidence: {confidence:.2%})")

            elif action == 'SELL' and position > -5:
                # Sell logic
                position -= 1
                balance += current_price
                trades.append({
                    'time': data.iloc[idx]['timestamp'],
                    'action': 'SELL',
                    'price': current_price,
                    'position': position,
                    'confidence': confidence
                })
                print(f"\n[{idx}] SELL at ${current_price:.2f} (confidence: {confidence:.2%})")

        # Print status every 10 time steps
        if idx % 10 == 0:
            print(f"\n[{idx}] Status: Position={position}, Balance=${balance:.2f}, Price=${current_price:.2f}")
            if 'lstm_expected_movement' in predictions:
                print(f"  LSTM Movement: {predictions['lstm_expected_movement']:.4f}")
            if 'lstm_risk_score' in predictions:
                print(f"  Risk Score: {predictions['lstm_risk_score']:.2%}")

    # Calculate final results
    final_price = data.iloc[-1]['close']
    final_value = balance + position * final_price
    total_return = (final_value - 100000) / 100000

    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Initial Balance: $100,000")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Number of Trades: {len(trades)}")
    print(f"Final Position: {position}")

    # Show trade summary
    if trades:
        print("\nTrade Summary:")
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        print(f"  Buys: {len(buy_trades)}")
        print(f"  Sells: {len(sell_trades)}")
        avg_confidence = np.mean([t['confidence'] for t in trades])
        print(f"  Average Confidence: {avg_confidence:.2%}")

def example_real_time_prediction():
    """Example of making a real-time prediction."""
    print("\n" + "="*60)
    print("REAL-TIME PREDICTION EXAMPLE")
    print("="*60)

    # Create a sample market snapshot
    snapshot = {
        'symbol': 'NQZ5',
        'timestamp': datetime.utcnow().isoformat(),
        'features': {
            'close': 16525.50,
            'volume': 1250,
            'rsi': 48.5,
            'macd': -2.3,
            'bb_upper': 16545.0,
            'bb_lower': 16505.0,
            'vwap': 16523.75,
            'dist_to_poc_ticks': 3
        },
        'position': 2,  # Currently long 2 contracts
        'balance': 99500,
        'unrealized_pnl': 125.50,
        'recent_actions': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    }

    # Get prediction
    predictions = predict_from_snapshot(snapshot)

    # Display results
    print("\nMarket Snapshot:")
    print(f"  Symbol: {snapshot['symbol']}")
    print(f"  Price: ${snapshot['features']['close']:.2f}")
    print(f"  RSI: {snapshot['features']['rsi']:.1f}")
    print(f"  Position: {snapshot['position']}")
    print(f"  Unrealized P&L: ${snapshot['unrealized_pnl']:.2f}")

    print("\nML Predictions:")
    print(f"  Recommendation: {predictions['ensemble_recommendation']}")
    print(f"  Confidence: {predictions['confidence_score']:.2%}")

    if 'win_5m_prob' in predictions:
        print(f"\n  LightGBM 5min Win Prob: {predictions['win_5m_prob']:.2%}")

    if 'lstm_buy_prob' in predictions:
        print(f"\n  LSTM Signals:")
        print(f"    Buy: {predictions['lstm_buy_prob']:.2%}")
        print(f"    Sell: {predictions['lstm_sell_prob']:.2%}")
        print(f"    Hold: {predictions['lstm_hold_prob']:.2%}")

    if 'ppo_buy_prob' in predictions:
        print(f"\n  PPO Actions:")
        print(f"    Buy: {predictions['ppo_buy_prob']:.2%}")
        print(f"    Sell: {predictions['ppo_sell_prob']:.2%}")
        print(f"    Hold: {predictions['ppo_hold_prob']:.2%}")

    # Trading decision
    print("\nTrading Decision:")
    if predictions['confidence_score'] > 0.7:
        print(f"  >> Execute {predictions['ensemble_recommendation']} order")
    elif predictions['confidence_score'] > 0.5:
        print(f"  >> Consider {predictions['ensemble_recommendation']} (moderate confidence)")
    else:
        print(f"  >> Wait for better setup (low confidence)")

def main():
    """Run examples."""
    print("DeepSeek TopStepX Bundle - ML Trading Examples")
    print("="*60)

    # Check if models exist
    models_dir = ROOT / "ml" / "models"
    lstm_exists = (models_dir / "lstm_trading_model.pth").exists()
    ppo_exists = (models_dir / "ppo_trading_agent.pth").exists()

    if not lstm_exists and not ppo_exists:
        print("\nNote: LSTM and PPO models not found.")
        print("Run the following to train them:")
        print("  python3 ml/scripts/train_lstm_model.py")
        print("  python3 ml/scripts/train_ppo_agent.py")
        print("\nRunning demo with LightGBM only...")

    # Run examples
    try:
        # Example 1: Real-time prediction
        example_real_time_prediction()

        # Example 2: Trading simulation
        print("\n")
        simulate_trading_session()

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have trained the models first.")

if __name__ == "__main__":
    main()