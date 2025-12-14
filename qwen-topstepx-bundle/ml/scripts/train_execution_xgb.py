#!/usr/bin/env python3
"""
Train Execution XGB Model

Trains XGBoost to score L2 execution opportunities.
Uses baseline model signals + L2 snapshots to learn optimal execution timing.
"""

import sys
import os
import json
import numpy as np
import xgboost as xgb
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))
from l2_execution_features import L2ExecutionFeatures
from l2_data_loader import L2DataLoader
from no_whale_regime_backtest import load_1s_bars


def generate_training_data(
    dates: list,
    l2_loader: L2DataLoader
):
    """
    Generate training data by sampling L2 snapshots and labeling based on quality.

    Good execution opportunities = tight spread, favorable flow, no blocking walls
    """

    X_train = []
    y_train = []

    for date in dates:
        print(f"\n[{date}] Generating training samples...")

        # Load L2 snapshots for this date
        n_snapshots = l2_loader.load_date(date)
        if n_snapshots == 0:
            print(f"  No L2 data for {date}, skipping")
            continue

        # Sample snapshots throughout the day
        snapshots = l2_loader.snapshots_by_date[date]
        timestamps = l2_loader.timestamps_by_date[date]

        print(f"  Sampling from {len(snapshots):,} L2 snapshots...")

        # Sample every 1000th snapshot to get ~2400 samples per day
        sample_indices = range(0, len(snapshots), 1000)

        feature_extractor = L2ExecutionFeatures()

        for idx in sample_indices:
            if idx >= len(snapshots):
                break

            # Add recent history
            start_idx = max(0, idx - 50)
            for hist_idx in range(start_idx, idx + 1):
                feature_extractor.add_snapshot(snapshots[hist_idx])

            # Extract features for both directions
            for direction in ['LONG', 'SHORT']:
                features = feature_extractor.get_feature_vector(direction)

                # Label based on ORDER BOOK STRUCTURE
                snap = snapshots[idx]
                bids = snap.get('bids', [])
                asks = snap.get('asks', [])

                if not bids or not asks:
                    continue

                # Get current price and walls
                best_bid = bids[0]['price']
                best_ask = asks[0]['price']
                mid = (best_bid + best_ask) / 2

                # Detect walls (3x average size)
                bid_sizes = [b['size'] for b in bids]
                ask_sizes = [a['size'] for a in asks]
                avg_bid = sum(bid_sizes) / len(bid_sizes)
                avg_ask = sum(ask_sizes) / len(ask_sizes)

                # Find wall locations and distances
                bid_walls = []
                ask_walls = []

                for i, bid in enumerate(bids):
                    if bid['size'] > avg_bid * 3:
                        distance = mid - bid['price']  # Points below current price
                        bid_walls.append({'level': i, 'size': bid['size'], 'distance': distance})

                for i, ask in enumerate(asks):
                    if ask['size'] > avg_ask * 3:
                        distance = ask['price'] - mid  # Points above current price
                        ask_walls.append({'level': i, 'size': ask['size'], 'distance': distance})

                # Absorption indicators
                imbalance = snap.get('imbalance', 0.0)

                # Label based on wall positions and aggression
                if direction == 'LONG':
                    # LONG: Want bid wall support below + aggressive buyers
                    score = 50.0

                    # Nearby bid wall (support) = GOOD
                    if bid_walls:
                        closest_support = min(bid_walls, key=lambda x: x['distance'])
                        if closest_support['distance'] < 10:  # Within 10 points
                            score += 25  # Strong support nearby
                            # Closer = better
                            score += max(0, 10 - closest_support['distance'])

                    # Ask wall above (resistance) = BAD (wait for break)
                    if ask_walls:
                        closest_resistance = min(ask_walls, key=lambda x: x['distance'])
                        if closest_resistance['distance'] < 10:  # Within 10 points
                            score -= 30  # Resistance blocking

                    # Aggressive buyers (positive imbalance) = GOOD
                    if imbalance > 0.2:
                        score += 15
                    elif imbalance < -0.2:
                        score -= 15

                else:  # SHORT
                    # SHORT: Want ask wall resistance above + aggressive sellers
                    score = 50.0

                    # Nearby ask wall (resistance) = GOOD
                    if ask_walls:
                        closest_resistance = min(ask_walls, key=lambda x: x['distance'])
                        if closest_resistance['distance'] < 10:  # Within 10 points
                            score += 25  # Strong resistance nearby
                            score += max(0, 10 - closest_resistance['distance'])

                    # Bid wall below (support) = BAD (wait for break)
                    if bid_walls:
                        closest_support = min(bid_walls, key=lambda x: x['distance'])
                        if closest_support['distance'] < 10:  # Within 10 points
                            score -= 30  # Support blocking

                    # Aggressive sellers (negative imbalance) = GOOD
                    if imbalance < -0.2:
                        score += 15
                    elif imbalance > 0.2:
                        score -= 15

                # Tight spread is always good
                spread = snap.get('spread', 10.0)
                if spread < 2.5:
                    score += 10
                elif spread > 7.0:
                    score -= 10

                # Clip and label
                score = max(0, min(100, score))
                label = 1 if score >= 70 else 0

                X_train.append(features)
                y_train.append(label)

        print(f"  Generated {len(X_train):,} training samples so far")

    return np.array(X_train), np.array(y_train)


def main():
    print("=" * 70)
    print("TRAINING EXECUTION XGB MODEL")
    print("=" * 70)

    # Config
    bars_path = "/tmp/databento_nq_2025_full_year.json"
    models_dir = "ml/models/baseline_full_year_no_l2"
    l2_data_dir = "ml/data/l2"
    output_dir = "ml/models/execution_xgb"
    os.makedirs(output_dir, exist_ok=True)

    # Training dates (Dec 7-12)
    train_dates = [
        '2025-12-07', '2025-12-08', '2025-12-09',
        '2025-12-10', '2025-12-11', '2025-12-12'
    ]

    # Load L2 data
    print(f"\n[Load] Loading L2 data from {l2_data_dir}...")
    l2_loader = L2DataLoader(l2_data_dir)

    # Generate training data
    print("\n[Train] Generating training data from L2 snapshots...")
    X_train, y_train = generate_training_data(train_dates, l2_loader)

    print(f"\n[Train] Training data shape: {X_train.shape}")
    print(f"[Train] Positive samples: {np.sum(y_train)} / {len(y_train)} ({np.mean(y_train)*100:.1f}%)")

    # Train XGBoost
    print("\n[Train] Training Execution XGB...")
    execution_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    execution_xgb.fit(X_train, y_train, verbose=True)

    # Save model
    model_path = os.path.join(output_dir, 'execution_xgb.json')
    execution_xgb.save_model(model_path)
    print(f"\n[Save] Model saved to {model_path}")

    # Feature importance
    print("\n[Info] Top 10 feature importances:")
    feature_names = L2ExecutionFeatures.get_feature_names()
    importances = execution_xgb.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {i+1:2d}. {feature_names[idx]:25s}: {importances[idx]:.4f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
