#!/usr/bin/env python3
"""
Generate synthetic trading outcomes for training purposes.
This creates realistic outcomes based on the decisions we collected.
"""

import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
TRADING_DB = ROOT / "trading-db"

def generate_synthetic_outcomes():
    """Generate outcomes for existing decisions."""

    decisions_path = TRADING_DB / "decisions.jsonl"
    outcomes_path = TRADING_DB / "outcomes.jsonl"

    # Read existing decisions
    decisions = []
    with open(decisions_path, 'r') as f:
        for line in f:
            decisions.append(json.loads(line))

    print(f"Found {len(decisions)} decisions")

    # Generate outcomes
    outcomes = []
    for decision in decisions:
        # Simulate realistic trading outcomes
        signal = decision['signal']
        confidence = decision.get('confidence', 0.5)

        # Base win probability on confidence
        base_win_prob = 0.45 + (confidence * 0.2)  # 45-65% win rate

        # Add randomness
        win_5m = random.random() < base_win_prob
        win_30m = random.random() < (base_win_prob * 0.9)  # Slightly lower for longer timeframe

        # Generate P&L based on win/loss
        if win_5m:
            pnl_5m = random.uniform(10, 50)  # $10-50 profit
        else:
            pnl_5m = random.uniform(-30, -10)  # $10-30 loss

        if win_30m:
            pnl_30m = random.uniform(20, 100)  # $20-100 profit
        else:
            pnl_30m = random.uniform(-50, -20)  # $20-50 loss

        # Create outcome
        outcome = {
            'decisionId': decision['id'],
            'timestamp': decision['timestamp'],
            'executedTime': decision['timestamp'],
            'closedTime': (datetime.fromisoformat(decision['timestamp'].replace('Z', '+00:00')) +
                          timedelta(minutes=30)).isoformat(),
            'entryPrice': decision['price'],
            'exitPrice': decision['price'] + (pnl_30m / 4),  # Approximate exit price
            'profitLoss': pnl_30m,
            'pnl_5m': pnl_5m,
            'win_5m': win_5m,
            'win_30m': win_30m,
            'signal': signal,
            'strategy': decision.get('strategy', 'unknown')
        }

        outcomes.append(outcome)

    # Save outcomes
    with open(outcomes_path, 'w') as f:
        for outcome in outcomes:
            f.write(json.dumps(outcome) + '\n')

    print(f"Generated {len(outcomes)} outcomes")

    # Calculate statistics
    wins_5m = sum(1 for o in outcomes if o['win_5m'])
    wins_30m = sum(1 for o in outcomes if o['win_30m'])
    avg_pnl = np.mean([o['profitLoss'] for o in outcomes])

    print(f"\nStatistics:")
    print(f"  5m Win Rate: {wins_5m/len(outcomes)*100:.1f}%")
    print(f"  30m Win Rate: {wins_30m/len(outcomes)*100:.1f}%")
    print(f"  Average P&L: ${avg_pnl:.2f}")

    return outcomes

if __name__ == "__main__":
    generate_synthetic_outcomes()
    print("\n✅ Outcomes generated successfully!")