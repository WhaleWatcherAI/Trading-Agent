#!/usr/bin/env python3
"""
Backtest with Execution XGB Optimization

Runs baseline model + Execution XGB to optimize trade timing.
Waits for favorable L2 conditions before executing (no time limit).
"""

import sys
import os
import json
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from l2_execution_features import L2ExecutionFeatures
from l2_data_loader import L2DataLoader
from no_whale_regime_backtest import (
    load_1s_bars, LongTermLSTM, ShortTermLSTM, RegimeLSTM,
    EMBEDDING_DIM, DEVICE
)
import torch


def load_models(baseline_dir: str, execution_dir: str):
    """Load baseline + execution models."""
    print(f"[Load] Loading models...")

    # Baseline LSTMs
    longterm_lstm = LongTermLSTM(input_dim=5, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    shortterm_lstm = ShortTermLSTM(input_dim=5, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    regime_lstm = RegimeLSTM(input_dim=12, embedding_dim=EMBEDDING_DIM).to(DEVICE)

    longterm_lstm.load_state_dict(torch.load(os.path.join(baseline_dir, 'longterm_lstm.pt'), map_location=DEVICE))
    shortterm_lstm.load_state_dict(torch.load(os.path.join(baseline_dir, 'shortterm_lstm.pt'), map_location=DEVICE))
    regime_lstm.load_state_dict(torch.load(os.path.join(baseline_dir, 'regime_lstm.pt'), map_location=DEVICE))

    longterm_lstm.eval()
    shortterm_lstm.eval()
    regime_lstm.eval()

    # Baseline XGBoost
    stage1_xgb = xgb.Booster()
    stage1_xgb.load_model(os.path.join(baseline_dir, 'stage1_xgb.json'))

    timing_xgb = xgb.Booster()
    timing_xgb.load_model(os.path.join(baseline_dir, 'timing_xgb.json'))

    final_xgb = xgb.Booster()
    final_xgb.load_model(os.path.join(baseline_dir, 'final_xgb.json'))

    # Execution XGB
    execution_xgb = xgb.Booster()
    execution_xgb.load_model(os.path.join(execution_dir, 'execution_xgb.json'))

    print("[Load] All models loaded")
    return (longterm_lstm, shortterm_lstm, regime_lstm,
            stage1_xgb, timing_xgb, final_xgb, execution_xgb)


def main():
    print("=" * 70)
    print("BACKTEST WITH EXECUTION XGB OPTIMIZATION")
    print("Score-based execution (no time limit)")
    print("=" * 70)

    # Config
    bars_path = "/tmp/databento_nq_dec7_12.json"
    baseline_models_dir = "ml/models/baseline_full_year_no_l2"
    execution_models_dir = "ml/models/execution_xgb"
    l2_data_dir = "ml/data/l2"
    output_file = "/tmp/execution_xgb_backtest_results.json"

    execution_threshold = 80  # Execute when score >= 80

    # Load models
    models = load_models(baseline_models_dir, execution_models_dir)
    (longterm_lstm, shortterm_lstm, regime_lstm,
     stage1_xgb, timing_xgb, final_xgb, execution_xgb) = models

    # Load OHLCV data
    print(f"\n[Load] Loading bars from {bars_path}...")
    all_bars = load_1s_bars(bars_path)

    bars_by_date = defaultdict(list)
    for bar in all_bars:
        ts = bar['t']
        if isinstance(ts, str):
            date = ts[:10]
        else:
            date = ts.strftime('%Y-%m-%d')
        bars_by_date[date].append(bar)

    dates = sorted(bars_by_date.keys())
    print(f"[Load] Found {len(dates)} dates: {dates}")

    # Load L2 data
    print(f"\n[Load] Loading L2 snapshots from {l2_data_dir}...")
    l2_loader = L2DataLoader(l2_data_dir)

    # Run backtest
    print("\n[Backtest] Running with Execution XGB optimization...")
    print(f"Execution threshold: {execution_threshold}/100")
    print(f"Max wait: UNLIMITED (wait for good conditions)\n")

    all_results = {}
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0

    for date in dates:
        print(f"\n[{date}] Processing...")

        # Load L2 for this date
        n_snapshots = l2_loader.load_date(date)
        if n_snapshots == 0:
            print(f"  No L2 data, skipping")
            continue

        # TODO: Run baseline model to generate signals
        # For now, just report L2 data loaded
        print(f"  Loaded {n_snapshots:,} L2 snapshots")
        print(f"  Execution optimization: READY")

        # Placeholder results
        all_results[date] = {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
            "execution_improvements": []
        }

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {(total_wins/total_trades*100 if total_trades > 0 else 0):.1f}%")
    print(f"Total PnL: {total_pnl:+.2f} pts (${total_pnl*20:+,.2f})")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Save] Results saved to {output_file}")
    print("\nNOTE: Full baseline signal generation not yet implemented.")
    print("Next step: Integrate baseline model signal generation.")


if __name__ == "__main__":
    main()
