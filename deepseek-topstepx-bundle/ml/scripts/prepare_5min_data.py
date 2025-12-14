#!/usr/bin/env python3
"""
Prepare proper 5-minute bar data with gap-based predictions.
No leakage, proper holdout, better timeframe.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"

def resample_to_5min(df_1min):
    """Convert 1-minute bars to 5-minute bars."""
    print("📊 Resampling 1-min to 5-min bars...")

    # Ensure index is datetime
    if not isinstance(df_1min.index, pd.DatetimeIndex):
        df_1min.index = pd.to_datetime(df_1min.index)

    # Resample to 5-minute bars
    df_5min = df_1min.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return df_5min

def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    print("📈 Adding technical indicators...")

    # Price-based indicators
    df['returns'] = df['close'].pct_change()

    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_window = 20
    df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
    bb_std = df['close'].rolling(window=bb_window).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(window=14).mean()

    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()

    # Price position within the day's range
    df['hl_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    return df

def create_features_and_targets(df, feature_window=60, prediction_horizon=20, gap=5):
    """
    Create features and targets with proper gap to prevent leakage.

    Args:
        df: DataFrame with 5-minute bars
        feature_window: Number of bars to use for features (60 = 5 hours)
        prediction_horizon: How many bars ahead to predict (20 = 100 minutes)
        gap: Number of bars gap between features and target (5 = 25 minutes)

    Returns:
        features, targets, metadata
    """
    print(f"\n🎯 Creating features and targets:")
    print(f"   Feature window: {feature_window} bars ({feature_window * 5} minutes)")
    print(f"   Gap: {gap} bars ({gap * 5} minutes)")
    print(f"   Prediction horizon: {prediction_horizon} bars ({prediction_horizon * 5} minutes)")

    # Select feature columns
    feature_cols = [
        'returns', 'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_position', 'bb_width', 'volume_ratio', 'volatility',
        'atr', 'hl_ratio'
    ]

    features = []
    targets = []
    metadata = []

    # Create samples
    for i in range(feature_window, len(df) - prediction_horizon - gap):
        # Feature window: [i-feature_window : i]
        feature_data = df.iloc[i-feature_window:i][feature_cols].values

        # Current price (last price in feature window)
        current_price = df.iloc[i-1]['close']

        # Target price (prediction_horizon bars after the gap)
        target_idx = i + gap + prediction_horizon
        if target_idx >= len(df):
            break

        target_price = df.iloc[target_idx]['close']

        # Calculate target (percentage change)
        target_return = (target_price - current_price) / current_price

        # Classification target (buy/sell/hold)
        if target_return > 0.002:  # 0.2% up (more reasonable for 100-min horizon)
            signal = 1  # Buy
        elif target_return < -0.002:  # 0.2% down
            signal = 2  # Sell
        else:
            signal = 0  # Hold

        features.append(feature_data)
        targets.append([signal, target_return * 100])  # Return in percentage

        metadata.append({
            'timestamp': df.index[i],
            'current_price': current_price,
            'target_price': target_price,
            'target_return': target_return
        })

    features = np.array(features)
    targets = np.array(targets)

    print(f"   Created {len(features)} samples")
    print(f"   Feature shape: {features.shape}")
    print(f"   Target shape: {targets.shape}")

    return features, targets, metadata

def split_data(features, targets, metadata, train_pct=0.7, val_pct=0.15, test_pct=0.15):
    """
    Split data into train/validation/test sets (time-based, no shuffle).
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 0.001, "Percentages must sum to 1"

    n = len(features)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    print(f"\n📊 Data splits:")
    print(f"   Train: 0 to {train_end} ({train_pct:.0%})")
    print(f"   Val: {train_end} to {val_end} ({val_pct:.0%})")
    print(f"   Test: {val_end} to {n} ({test_pct:.0%})")

    # Split features
    X_train = features[:train_end]
    X_val = features[train_end:val_end]
    X_test = features[val_end:]

    # Split targets
    y_train = targets[:train_end]
    y_val = targets[train_end:val_end]
    y_test = targets[val_end:]

    # Split metadata
    meta_train = metadata[:train_end]
    meta_val = metadata[train_end:val_end]
    meta_test = metadata[val_end:]

    print(f"\n   Train dates: {meta_train[0]['timestamp']} to {meta_train[-1]['timestamp']}")
    print(f"   Val dates: {meta_val[0]['timestamp']} to {meta_val[-1]['timestamp']}")
    print(f"   Test dates: {meta_test[0]['timestamp']} to {meta_test[-1]['timestamp']}")

    return (X_train, y_train, meta_train), (X_val, y_val, meta_val), (X_test, y_test, meta_test)

def analyze_targets(targets, split_name=""):
    """Analyze the distribution of targets."""
    signals = targets[:, 0].astype(int)
    returns = targets[:, 1]

    print(f"\n📈 {split_name} Target Analysis:")

    # Signal distribution
    unique, counts = np.unique(signals, return_counts=True)
    total = len(signals)

    signal_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    for sig, count in zip(unique, counts):
        print(f"   {signal_map.get(sig, sig)}: {count} ({count/total:.1%})")

    # Return distribution
    print(f"\n   Return stats:")
    print(f"     Mean: {np.mean(returns):.3f}%")
    print(f"     Std: {np.std(returns):.3f}%")
    print(f"     Min: {np.min(returns):.3f}%")
    print(f"     Max: {np.max(returns):.3f}%")
    print(f"     Positive: {np.sum(returns > 0) / len(returns):.1%}")

def main():
    print("\n" + "="*70)
    print("🚀 PREPARING IMPROVED 5-MINUTE DATA")
    print("="*70)

    # Load 1-minute data
    spy_1min_path = DATA_DIR / 'spy_real_data.parquet'
    if not spy_1min_path.exists():
        print("❌ SPY 1-minute data not found!")
        return

    df_1min = pd.read_parquet(spy_1min_path)
    print(f"\n📊 Loaded {len(df_1min)} 1-minute bars")

    # Resample to 5-minute
    df_5min = resample_to_5min(df_1min)
    print(f"📊 Resampled to {len(df_5min)} 5-minute bars")

    # Add indicators
    df_5min = add_technical_indicators(df_5min)

    # Drop NaN values
    df_5min = df_5min.dropna()
    print(f"📊 After adding indicators: {len(df_5min)} bars")

    # Save 5-minute data
    df_5min.to_parquet(DATA_DIR / 'spy_5min_data.parquet')
    print(f"💾 Saved 5-minute data to spy_5min_data.parquet")

    # Create features and targets
    features, targets, metadata = create_features_and_targets(
        df_5min,
        feature_window=60,      # 5 hours of history
        prediction_horizon=20,   # Predict 100 minutes ahead
        gap=5                   # 25-minute gap to prevent leakage
    )

    # Split data
    train_data, val_data, test_data = split_data(
        features, targets, metadata,
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15
    )

    # Analyze each split
    analyze_targets(train_data[1], "Training")
    analyze_targets(val_data[1], "Validation")
    analyze_targets(test_data[1], "Test (Holdout)")

    # Save prepared data
    np.savez(DATA_DIR / 'prepared_5min_data.npz',
             X_train=train_data[0], y_train=train_data[1],
             X_val=val_data[0], y_val=val_data[1],
             X_test=test_data[0], y_test=test_data[1])

    # Save metadata
    metadata_info = {
        'feature_window': 60,
        'prediction_horizon': 20,
        'gap': 5,
        'feature_cols': [
            'returns', 'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_position', 'bb_width', 'volume_ratio', 'volatility',
            'atr', 'hl_ratio'
        ],
        'n_features': 11,
        'train_samples': len(train_data[0]),
        'val_samples': len(val_data[0]),
        'test_samples': len(test_data[0]),
        'total_samples': len(features),
        'train_dates': [str(train_data[2][0]['timestamp']), str(train_data[2][-1]['timestamp'])],
        'val_dates': [str(val_data[2][0]['timestamp']), str(val_data[2][-1]['timestamp'])],
        'test_dates': [str(test_data[2][0]['timestamp']), str(test_data[2][-1]['timestamp'])],
        'prepared_at': datetime.utcnow().isoformat()
    }

    with open(DATA_DIR / 'data_prep_config.json', 'w') as f:
        json.dump(metadata_info, f, indent=2)

    print("\n" + "="*70)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Files created:")
    print(f"   - spy_5min_data.parquet (5-minute bars with indicators)")
    print(f"   - prepared_5min_data.npz (features and targets)")
    print(f"   - data_prep_config.json (metadata)")

    print(f"\n🎯 Setup:")
    print(f"   - 5-minute bars (less noise)")
    print(f"   - 60-bar feature window (5 hours)")
    print(f"   - 5-bar gap (25 minutes - no leakage)")
    print(f"   - 20-bar prediction (100 minutes ahead)")
    print(f"   - Proper train/val/test splits (70/15/15)")

    return metadata_info

if __name__ == "__main__":
    main()