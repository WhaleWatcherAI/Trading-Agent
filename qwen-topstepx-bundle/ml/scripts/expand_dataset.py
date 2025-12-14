#!/usr/bin/env python3
"""
Expand the dataset by generating more synthetic trading examples.
This helps when we don't have enough real data for training.
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
TRADING_DB = ROOT / "trading-db"

def expand_dataset():
    """Expand existing dataset with synthetic variations."""

    # Load existing data
    market_data = pd.read_parquet(DATA_DIR / "market_data.parquet")
    existing_decisions = []

    decisions_path = TRADING_DB / "decisions.jsonl"
    if decisions_path.exists():
        with open(decisions_path, 'r') as f:
            for line in f:
                existing_decisions.append(json.loads(line))

    print(f"Found {len(existing_decisions)} existing decisions")
    print(f"Market data shape: {market_data.shape}")

    # Generate more decisions from different parts of the data
    new_decisions = []
    new_outcomes = []
    new_snapshots = []

    # Sample more aggressively from the market data
    for i in range(200, min(len(market_data) - 100, 5000), 50):
        row = market_data.iloc[i]
        prev_row = market_data.iloc[i-1]

        # Generate decisions with various strategies
        for strategy in ['rsi', 'macd', 'bb', 'momentum', 'sma_cross']:
            signal = None
            confidence = 0

            if strategy == 'rsi':
                if row['RSI'] < 35:
                    signal = 'BUY'
                    confidence = (35 - row['RSI']) / 35
                elif row['RSI'] > 65:
                    signal = 'SELL'
                    confidence = (row['RSI'] - 65) / 35

            elif strategy == 'macd':
                if row['MACD'] > row['MACD_signal']:
                    signal = 'BUY'
                    confidence = min(abs(row['MACD'] - row['MACD_signal']) / 5, 1.0)
                else:
                    signal = 'SELL'
                    confidence = min(abs(row['MACD'] - row['MACD_signal']) / 5, 1.0)

            elif strategy == 'bb':
                if row['Close'] < row['BB_lower']:
                    signal = 'BUY'
                    confidence = 0.7
                elif row['Close'] > row['BB_upper']:
                    signal = 'SELL'
                    confidence = 0.7

            elif strategy == 'momentum':
                if i >= 20:
                    momentum = (row['Close'] - market_data.iloc[i-20]['Close']) / market_data.iloc[i-20]['Close']
                    if momentum > 0.005:
                        signal = 'BUY'
                        confidence = min(momentum * 100, 1.0)
                    elif momentum < -0.005:
                        signal = 'SELL'
                        confidence = min(abs(momentum) * 100, 1.0)

            elif strategy == 'sma_cross':
                if row['SMA_20'] > row['SMA_50'] and prev_row['SMA_20'] <= prev_row['SMA_50']:
                    signal = 'BUY'
                    confidence = 0.65
                elif row['SMA_20'] < row['SMA_50'] and prev_row['SMA_20'] >= prev_row['SMA_50']:
                    signal = 'SELL'
                    confidence = 0.65

            if signal and confidence > 0.3:
                decision_id = f"decision_exp_{i}_{strategy}_{datetime.now().timestamp()}"

                # Create decision
                decision = {
                    'id': decision_id,
                    'timestamp': row.name.isoformat() if hasattr(row.name, 'isoformat') else str(row.name),
                    'symbol': 'NQ',
                    'signal': signal,
                    'strategy': strategy,
                    'confidence': float(confidence),
                    'price': float(row['Close']),
                    'features': {
                        'rsi': float(row['RSI']),
                        'macd': float(row['MACD']),
                        'bb_upper': float(row['BB_upper']),
                        'bb_lower': float(row['BB_lower']),
                        'sma_20': float(row['SMA_20']),
                        'sma_50': float(row['SMA_50']),
                        'volume_ratio': float(row.get('Volume_ratio', 1.0)),
                        'volatility': float(row.get('Volatility', 0.01)),
                        'dist_to_poc_ticks': float(row.get('Dist_to_POC_ticks', 0))
                    }
                }

                # Create outcome
                win_prob = 0.45 + (confidence * 0.15)
                win_5m = random.random() < win_prob
                win_30m = random.random() < (win_prob * 0.95)

                if win_5m:
                    pnl_5m = random.uniform(5, 40)
                else:
                    pnl_5m = random.uniform(-25, -5)

                if win_30m:
                    pnl_30m = random.uniform(10, 60)
                else:
                    pnl_30m = random.uniform(-40, -10)

                outcome = {
                    'decisionId': decision_id,
                    'timestamp': decision['timestamp'],
                    'executedTime': decision['timestamp'],
                    'closedTime': (datetime.now() + timedelta(minutes=30)).isoformat(),
                    'entryPrice': float(decision['price']),
                    'exitPrice': float(decision['price'] + (pnl_30m / 4)),
                    'profitLoss': float(pnl_30m),
                    'pnl_5m': float(pnl_5m),
                    'win_5m': bool(win_5m),
                    'win_30m': bool(win_30m),
                    'signal': signal,
                    'strategy': strategy
                }

                # Create snapshot
                snapshot = {
                    'symbol': 'NQ',
                    'timestamp': decision['timestamp'],
                    'features': decision['features']
                }

                new_decisions.append(decision)
                new_outcomes.append(outcome)
                new_snapshots.append(snapshot)

    print(f"Generated {len(new_decisions)} new decisions")

    # Append to existing files
    with open(TRADING_DB / "decisions.jsonl", 'a') as f:
        for decision in new_decisions:
            f.write(json.dumps(decision) + '\n')

    with open(TRADING_DB / "outcomes.jsonl", 'a') as f:
        for outcome in new_outcomes:
            f.write(json.dumps(outcome) + '\n')

    with open(DATA_DIR / "snapshots.jsonl", 'a') as f:
        for snapshot in new_snapshots:
            f.write(json.dumps(snapshot) + '\n')

    # Calculate statistics
    total_decisions = len(existing_decisions) + len(new_decisions)
    wins_5m = sum(1 for o in new_outcomes if o['win_5m'])
    wins_30m = sum(1 for o in new_outcomes if o['win_30m'])

    print(f"\nDataset expanded:")
    print(f"  Total decisions: {total_decisions}")
    print(f"  New win rate (5m): {wins_5m/len(new_outcomes)*100:.1f}%")
    print(f"  New win rate (30m): {wins_30m/len(new_outcomes)*100:.1f}%")

    return total_decisions

if __name__ == "__main__":
    total = expand_dataset()
    print(f"\n✅ Dataset expanded to {total} examples!")