#!/usr/bin/env python3
"""
Train baseline models on FULL YEAR (hold out Dec 7-12 for L2 testing).
No L2/CNN features - pure baseline for comparison.
"""

import sys
import os
import torch

# Import from the backtest script
sys.path.insert(0, os.path.dirname(__file__))
from no_whale_regime_backtest import (
    load_1s_bars, train_lstm_models, train_stage1_xgboost,
    train_timing_xgboost, build_final_training_data,
    LongTermLSTM, ShortTermLSTM, RegimeLSTM,
    EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DEVICE
)
from collections import defaultdict
import xgboost as xgb
import numpy as np

def main():
    print("=" * 70)
    print("TRAINING FULL YEAR BASELINE MODELS")
    print("Hold out Dec 7-12 for L2 testing")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading full year data...")
    bars_path = "/tmp/databento_nq_2025_full_year.json"
    all_bars = load_1s_bars(bars_path)

    all_bars_1s = defaultdict(list)
    for bar in all_bars:
        ts = bar['t']
        if isinstance(ts, str):
            date = ts[:10]
        else:
            date = ts.strftime('%Y-%m-%d')
        all_bars_1s[date].append(bar)

    dates = sorted(all_bars_1s.keys())
    print(f"Found {len(dates)} dates: {dates[0]} to {dates[-1]}")

    # Hold out Dec 7-12 (where we have L2 data)
    holdout_dates = ['2025-12-07', '2025-12-08', '2025-12-09',
                     '2025-12-10', '2025-12-11', '2025-12-12']

    train_dates = [d for d in dates if d not in holdout_dates]

    print(f"\n[2/5] Training on {len(train_dates)} days (Jan 1 - Dec 6)")
    print(f"Holding out {len(holdout_dates)} days for L2 testing: {holdout_dates}")

    # Train LSTMs
    print(f"\n[3/5] Training LSTM models on {len(train_dates)} days...")
    print(f"Device: {DEVICE}")
    print("This will take ~2-3 hours...")
    longterm_lstm, shortterm_lstm, regime_lstm = train_lstm_models(
        all_bars_1s, train_dates, epochs=10
    )
    print("LSTMs trained!")

    # Train XGBoost models
    print("\n[4/5] Training XGBoost models...")

    # Stage 1
    print("  Training Stage 1 XGBoost...")
    stage1_xgb = train_stage1_xgboost(all_bars_1s, train_dates)

    # Timing
    print("  Training Timing XGBoost...")
    timing_xgb = train_timing_xgboost(all_bars_1s, train_dates, regime_lstm)

    # Final
    print("  Building final training data...")
    X_train, y_train = build_final_training_data(
        all_bars_1s, train_dates,
        stage1_xgb, timing_xgb, longterm_lstm, shortterm_lstm, regime_lstm
    )

    print(f"  Training Final XGBoost on {len(X_train)} samples...")
    final_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    )
    final_xgb.fit(X_train, y_train, verbose=False)
    print("XGBoost models trained!")

    # Save models
    save_dir = "ml/models/baseline_full_year_no_l2"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[5/5] Saving models to {save_dir}/...")

    # Save LSTMs
    torch.save(longterm_lstm.state_dict(), os.path.join(save_dir, 'longterm_lstm.pt'))
    torch.save(shortterm_lstm.state_dict(), os.path.join(save_dir, 'shortterm_lstm.pt'))
    torch.save(regime_lstm.state_dict(), os.path.join(save_dir, 'regime_lstm.pt'))

    # Save XGBoost
    stage1_xgb.save_model(os.path.join(save_dir, 'stage1_xgb.json'))
    timing_xgb.save_model(os.path.join(save_dir, 'timing_xgb.json'))
    final_xgb.save_model(os.path.join(save_dir, 'final_xgb.json'))

    print("\n" + "=" * 70)
    print("DONE! Models saved:")
    print("=" * 70)
    print(f"  {save_dir}/longterm_lstm.pt")
    print(f"  {save_dir}/shortterm_lstm.pt")
    print(f"  {save_dir}/regime_lstm.pt")
    print(f"  {save_dir}/stage1_xgb.json")
    print(f"  {save_dir}/timing_xgb.json")
    print(f"  {save_dir}/final_xgb.json")
    print(f"\nTrained on {len(train_dates)} days (whole year except Dec 7-12)")
    print("Ready for CNN/L2 integration!")
    print("=" * 70)

if __name__ == "__main__":
    main()
