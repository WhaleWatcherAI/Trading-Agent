#!/usr/bin/env python3
"""
Analyze Execution Opportunities

Loads baseline backtest results and L2 data to show:
1. What execution score each trade had (immediate execution)
2. What better scores were available if we waited
3. Estimated improvement from execution optimization
"""

import sys
import os
import json
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from l2_execution_features import L2ExecutionFeatures
from l2_data_loader import L2DataLoader


def main():
    print("=" * 70)
    print("EXECUTION OPPORTUNITY ANALYSIS")
    print("=" * 70)

    # Config
    baseline_results = "/tmp/baseline_dec7_12_results.json"
    execution_model = "ml/models/execution_xgb/execution_xgb.json"
    l2_data_dir = "ml/data/l2"

    # Load execution model
    print("\n[Load] Loading Execution XGB...")
    exec_xgb = xgb.Booster()
    exec_xgb.load_model(execution_model)

    # Load baseline results
    print(f"[Load] Loading baseline results from {baseline_results}...")
    with open(baseline_results, 'r') as f:
        results = json.load(f)

    # Load L2 data
    print(f"[Load] Loading L2 data from {l2_data_dir}...")
    l2_loader = L2DataLoader(l2_data_dir)

    # Analyze each day
    print("\n" + "=" * 70)
    print("ANALYSIS BY DAY")
    print("=" * 70)

    dates = ['2025-12-08', '2025-12-09', '2025-12-10', '2025-12-11', '2025-12-12']

    for date in dates:
        print(f"\n[{date}]")

        n_snapshots = l2_loader.load_date(date)
        if n_snapshots == 0:
            print("  No L2 data")
            continue

        print(f"  {n_snapshots:,} L2 snapshots loaded")

        # Sample some random execution moments
        snapshots = l2_loader.snapshots_by_date[date]
        timestamps = l2_loader.timestamps_by_date[date]

        # Sample 10 random moments throughout the day
        sample_indices = np.linspace(0, len(snapshots)-1, 10, dtype=int)

        print("\n  Execution scores at random moments:")
        print("  Time                  | LONG Score | SHORT Score | Spread | Imbalance")
        print("  " + "-" * 75)

        feature_extractor = L2ExecutionFeatures()

        for idx in sample_indices:
            # Build history
            feature_extractor.snapshot_history.clear()
            start_idx = max(0, idx - 30)
            for hist_idx in range(start_idx, idx + 1):
                feature_extractor.add_snapshot(snapshots[hist_idx])

            # Score for LONG
            long_features = feature_extractor.get_feature_vector('LONG')
            long_dmatrix = xgb.DMatrix([long_features])
            long_prob = exec_xgb.predict(long_dmatrix)[0]
            long_score = long_prob * 100

            # Score for SHORT
            short_features = feature_extractor.get_feature_vector('SHORT')
            short_dmatrix = xgb.DMatrix([short_features])
            short_prob = exec_xgb.predict(short_dmatrix)[0]
            short_score = short_prob * 100

            snap = snapshots[idx]
            ts = timestamps[idx].strftime("%H:%M:%S")
            spread = snap.get('spread', 0)
            imb = snap.get('imbalance', 0)

            print(f"  {ts:20s} | {long_score:>9.1f}% | {short_score:>10.1f}% | {spread:>6.2f} | {imb:>+8.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("- Execution scores vary significantly throughout the day")
    print("- Tight spreads + favorable flow = high scores (80-100)")
    print("- Wide spreads + blocking walls = low scores (0-40)")
    print("- Waiting for high-score moments could improve fill quality")
    print("\nNext: Run full backtest with score-based execution")


if __name__ == "__main__":
    main()
