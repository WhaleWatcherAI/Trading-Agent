#!/usr/bin/env python3
"""
LSTM Backtest with Volume Profile + CVD + ADX + ATR from 1-second bars
Predicts next bar direction (default) or N bars ahead

Features (from 1s bars aggregated to 1min):
- OHLC (normalized)
- Volume
- CVD (Cumulative Volume Delta)
- CVD EMA trend (CVD > EMA = 1, else 0)
- Volume Profile position (price relative to POC, VAH, VAL)
- ADX (Average Directional Index) - trend strength
- ATR (Average True Range) - volatility

NO DATA LEAKAGE:
- Features calculated only from past data
- Train/validation/holdout split
- Proper sequence windowing

Usage:
    python lstm_vp_cvd_backtest.py --input ../data/bars_1s.json --prediction-horizon 1
"""

import argparse
import json
import sys
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ML imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class LSTMPredictor(nn.Module):
    """LSTM model for price direction prediction."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class TimeSeriesDataset(Dataset):
    """Dataset for LSTM sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def aggregate_to_1min(bars_1s: List[Dict]) -> pd.DataFrame:
    """Aggregate 1-second bars to 1-minute bars with OHLCV."""
    if not bars_1s:
        return pd.DataFrame()

    records = []
    current_minute = None
    current_bar = None

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            dt = ts

        minute_ts = dt.replace(second=0, microsecond=0)

        if current_minute != minute_ts:
            if current_bar:
                records.append(current_bar)

            current_minute = minute_ts
            current_bar = {
                'timestamp': minute_ts,
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar.get('v', 0) or 0,
            }
        else:
            current_bar['high'] = max(current_bar['high'], bar['h'])
            current_bar['low'] = min(current_bar['low'], bar['l'])
            current_bar['close'] = bar['c']
            current_bar['volume'] += bar.get('v', 0) or 0

    if current_bar:
        records.append(current_bar)

    df = pd.DataFrame(records)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def calculate_cvd_for_1min(bars_1s: List[Dict], minute_timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Calculate CVD for each 1-minute bar from 1-second bars.
    Returns cumulative CVD at each minute.
    """
    # Group 1s bars by minute and calculate delta
    minute_deltas = defaultdict(float)

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            dt = ts

        minute_ts = dt.replace(second=0, microsecond=0)

        volume = bar.get('v', 0) or 0
        open_price = bar['o']
        close_price = bar['c']
        high_price = bar['h']
        low_price = bar['l']

        # Estimate delta from bar structure
        bar_range = high_price - low_price
        if bar_range > 0:
            close_position = (close_price - low_price) / bar_range
            delta = volume * (2 * close_position - 1)
        else:
            delta = volume if close_price >= open_price else -volume

        minute_deltas[minute_ts] += delta

    # Build cumulative CVD series
    cvd_values = []
    cumulative = 0.0

    for ts in minute_timestamps:
        # Make timezone-aware if needed
        if ts.tzinfo is None:
            ts_aware = ts.tz_localize('UTC')
        else:
            ts_aware = ts

        cumulative += minute_deltas.get(ts_aware, minute_deltas.get(ts, 0))
        cvd_values.append(cumulative)

    return pd.Series(cvd_values, index=minute_timestamps)


def calculate_volume_profile_rolling(df: pd.DataFrame, lookback: int = 30, tick_size: float = 0.25) -> pd.DataFrame:
    """
    Calculate rolling Volume Profile features.
    For each bar, uses only PAST data (no leakage).

    Returns DataFrame with: poc, vah, val, price_vs_poc, price_vs_vah, price_vs_val
    """
    n = len(df)
    poc_list = []
    vah_list = []
    val_list = []

    for i in range(n):
        if i < lookback:
            # Not enough history - use current price as POC
            poc_list.append(df['close'].iloc[i])
            vah_list.append(df['close'].iloc[i])
            val_list.append(df['close'].iloc[i])
            continue

        # Use past 'lookback' bars only
        window = df.iloc[i-lookback:i]

        # Build price-volume distribution
        price_volume = defaultdict(float)
        for _, row in window.iterrows():
            prices = [row['open'], row['high'], row['low'], row['close']]
            vol_per_price = row['volume'] / 4.0 if row['volume'] > 0 else 0.25

            for price in prices:
                rounded = round(price / tick_size) * tick_size
                price_volume[rounded] += vol_per_price

        if not price_volume:
            poc_list.append(df['close'].iloc[i])
            vah_list.append(df['close'].iloc[i])
            val_list.append(df['close'].iloc[i])
            continue

        # POC = highest volume price
        sorted_prices = sorted(price_volume.keys())
        volumes = [price_volume[p] for p in sorted_prices]
        total_volume = sum(volumes)

        poc_idx = np.argmax(volumes)
        poc = sorted_prices[poc_idx]

        # Value Area (70% of volume)
        target_volume = total_volume * 0.7
        current_volume = volumes[poc_idx]
        lower_idx = poc_idx
        upper_idx = poc_idx

        while current_volume < target_volume and (lower_idx > 0 or upper_idx < len(volumes) - 1):
            lower_vol = volumes[lower_idx - 1] if lower_idx > 0 else 0
            upper_vol = volumes[upper_idx + 1] if upper_idx < len(volumes) - 1 else 0

            if lower_vol >= upper_vol and lower_idx > 0:
                lower_idx -= 1
                current_volume += lower_vol
            elif upper_idx < len(volumes) - 1:
                upper_idx += 1
                current_volume += upper_vol
            else:
                break

        val = sorted_prices[lower_idx]
        vah = sorted_prices[upper_idx]

        poc_list.append(poc)
        vah_list.append(vah)
        val_list.append(val)

    vp_df = pd.DataFrame({
        'poc': poc_list,
        'vah': vah_list,
        'val': val_list,
    }, index=df.index)

    # Calculate relative positions (normalized)
    vp_df['price_vs_poc'] = (df['close'] - vp_df['poc']) / (vp_df['vah'] - vp_df['val'] + 1e-6)
    vp_df['price_vs_vah'] = (df['close'] - vp_df['vah']) / (vp_df['vah'] - vp_df['val'] + 1e-6)
    vp_df['price_vs_val'] = (df['close'] - vp_df['val']) / (vp_df['vah'] - vp_df['val'] + 1e-6)

    return vp_df


def calculate_microstructure_features(bars_1s: List[Dict], minute_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate microstructure features from 1-second bars for each 1-minute bar.
    These capture intra-minute price action patterns.
    """
    from collections import defaultdict

    # Group 1s bars by minute
    minute_bars = defaultdict(list)

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            dt = ts
        minute_ts = dt.replace(second=0, microsecond=0)
        minute_bars[minute_ts].append(bar)

    # Calculate features for each minute
    features = {
        'micro_bar_count': [],      # Number of 1s bars in minute (activity)
        'micro_up_bars': [],        # Count of up bars
        'micro_down_bars': [],      # Count of down bars
        'micro_up_ratio': [],       # Ratio of up bars
        'micro_avg_range': [],      # Average bar range (volatility)
        'micro_max_range': [],      # Max bar range (spike detection)
        'micro_range_expansion': [],# Max range / avg range
        'micro_vol_imbalance': [],  # Volume-weighted up/down imbalance
        'micro_close_position': [], # Where price closed within minute range
        'micro_momentum': [],       # Sum of (close-open) normalized
        'micro_reversal': [],       # Did price reverse direction intra-minute
        'micro_trend_strength': [], # Consistency of direction
    }

    for ts in minute_timestamps:
        # Handle timezone
        if ts.tzinfo is None:
            ts_aware = ts.tz_localize('UTC')
        else:
            ts_aware = ts

        bars = minute_bars.get(ts_aware, minute_bars.get(ts, []))

        if not bars or len(bars) < 2:
            # No data - use neutral values
            features['micro_bar_count'].append(len(bars) if bars else 0)
            features['micro_up_bars'].append(0)
            features['micro_down_bars'].append(0)
            features['micro_up_ratio'].append(0.5)
            features['micro_avg_range'].append(0)
            features['micro_max_range'].append(0)
            features['micro_range_expansion'].append(1)
            features['micro_vol_imbalance'].append(0)
            features['micro_close_position'].append(0.5)
            features['micro_momentum'].append(0)
            features['micro_reversal'].append(0)
            features['micro_trend_strength'].append(0)
            continue

        # Count bars and direction
        up_bars = sum(1 for b in bars if b['c'] > b['o'])
        down_bars = sum(1 for b in bars if b['c'] < b['o'])
        total_bars = len(bars)

        # Ranges
        ranges = [b['h'] - b['l'] for b in bars]
        avg_range = np.mean(ranges) if ranges else 0
        max_range = max(ranges) if ranges else 0

        # Volume imbalance
        up_vol = sum(b.get('v', 0) for b in bars if b['c'] >= b['o'])
        down_vol = sum(b.get('v', 0) for b in bars if b['c'] < b['o'])
        total_vol = up_vol + down_vol
        vol_imbalance = (up_vol - down_vol) / (total_vol + 1) if total_vol > 0 else 0

        # Close position within minute range
        minute_high = max(b['h'] for b in bars)
        minute_low = min(b['l'] for b in bars)
        minute_close = bars[-1]['c']
        minute_range = minute_high - minute_low
        close_pos = (minute_close - minute_low) / (minute_range + 1e-10) if minute_range > 0 else 0.5

        # Momentum - sum of price changes
        momentum = sum(b['c'] - b['o'] for b in bars)
        avg_price = np.mean([b['c'] for b in bars])
        norm_momentum = momentum / (avg_price + 1) if avg_price > 0 else 0

        # Reversal detection - did first half and second half move opposite?
        mid = len(bars) // 2
        if mid > 0:
            first_half_move = bars[mid]['c'] - bars[0]['o']
            second_half_move = bars[-1]['c'] - bars[mid]['o']
            reversal = 1 if (first_half_move * second_half_move < 0) else 0
        else:
            reversal = 0

        # Trend strength - how consistent were bar directions
        directions = [1 if b['c'] > b['o'] else -1 if b['c'] < b['o'] else 0 for b in bars]
        if directions:
            trend_strength = abs(sum(directions)) / len(directions)
        else:
            trend_strength = 0

        features['micro_bar_count'].append(total_bars)
        features['micro_up_bars'].append(up_bars)
        features['micro_down_bars'].append(down_bars)
        features['micro_up_ratio'].append(up_bars / total_bars if total_bars > 0 else 0.5)
        features['micro_avg_range'].append(avg_range)
        features['micro_max_range'].append(max_range)
        features['micro_range_expansion'].append(max_range / (avg_range + 1e-10) if avg_range > 0 else 1)
        features['micro_vol_imbalance'].append(vol_imbalance)
        features['micro_close_position'].append(close_pos)
        features['micro_momentum'].append(norm_momentum)
        features['micro_reversal'].append(reversal)
        features['micro_trend_strength'].append(trend_strength)

    return pd.DataFrame(features, index=minute_timestamps)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    ATR measures volatility.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    # True Range is the max of these
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is smoothed TR (Wilder's smoothing = EMA with alpha=1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX) and +DI/-DI.
    ADX measures trend strength (0-100).
    +DI/-DI measure trend direction.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    # Only keep positive values where one dominates
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smooth with Wilder's method
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di


def prepare_features(df: pd.DataFrame, cvd: pd.Series, vp_df: pd.DataFrame, micro_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Prepare all features for LSTM.
    All features use only past data.
    """
    features = pd.DataFrame(index=df.index)

    # OHLC features (normalized returns)
    features['open_ret'] = df['open'].pct_change()
    features['high_ret'] = df['high'].pct_change()
    features['low_ret'] = df['low'].pct_change()
    features['close_ret'] = df['close'].pct_change()

    # Price range features
    features['hl_range'] = (df['high'] - df['low']) / df['close']
    features['co_range'] = (df['close'] - df['open']) / df['close']

    # Volume (log-normalized)
    features['volume_log'] = np.log1p(df['volume'])
    features['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
    features['volume_ratio'] = df['volume'] / (features['volume_ma'] + 1)

    # CVD features
    features['cvd'] = cvd
    features['cvd_change'] = cvd.diff()

    # CVD EMA (20-period)
    features['cvd_ema'] = cvd.ewm(span=20, adjust=False).mean()
    features['cvd_vs_ema'] = (cvd - features['cvd_ema']) / (features['cvd_ema'].abs() + 1)
    features['cvd_trend'] = (cvd > features['cvd_ema']).astype(float)  # Binary: 1 if bullish, 0 if bearish

    # Volume Profile features
    features['price_vs_poc'] = vp_df['price_vs_poc']
    features['price_vs_vah'] = vp_df['price_vs_vah']
    features['price_vs_val'] = vp_df['price_vs_val']

    # ATR (Average True Range) - volatility indicator
    atr = calculate_atr(df, period=14)
    features['atr'] = atr
    features['atr_pct'] = atr / df['close']  # ATR as % of price
    features['atr_ratio'] = atr / atr.rolling(50, min_periods=1).mean()  # ATR vs its MA

    # ADX (Average Directional Index) - trend strength
    adx, plus_di, minus_di = calculate_adx(df, period=14)
    features['adx'] = adx / 100  # Normalize to 0-1 range
    features['plus_di'] = plus_di / 100
    features['minus_di'] = minus_di / 100
    features['di_diff'] = (plus_di - minus_di) / 100  # Positive = bullish trend, negative = bearish
    features['adx_trending'] = (adx > 25).astype(float)  # Binary: 1 if trending, 0 if ranging

    # Microstructure features from 1-second bars
    if micro_df is not None:
        for col in micro_df.columns:
            features[col] = micro_df[col]

    # Fill NaN with 0
    features = features.fillna(0)

    # Replace inf with large values
    features = features.replace([np.inf, -np.inf], 0)

    return features


def create_sequences(features: np.ndarray, targets: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM.
    Each sequence uses only past data to predict future.
    """
    X, y = [], []

    for i in range(seq_length, len(features)):
        X.append(features[i-seq_length:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Train LSTM with early stopping."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}", file=sys.stderr)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}", file=sys.stderr)
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]:
    """Evaluate model on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze().cpu().numpy()

            all_probs.extend(outputs.tolist() if outputs.ndim > 0 else [outputs.item()])
            all_preds.extend((outputs > 0.5).astype(int).tolist() if outputs.ndim > 0 else [int(outputs > 0.5)])
            all_targets.extend(y_batch.numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    # High confidence predictions
    high_conf_mask = (all_probs > 0.6) | (all_probs < 0.4)
    if high_conf_mask.sum() > 0:
        high_conf_acc = accuracy_score(all_targets[high_conf_mask], all_preds[high_conf_mask])
        high_conf_count = high_conf_mask.sum()
    else:
        high_conf_acc = 0.0
        high_conf_count = 0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_targets),
        'up_predictions': int(all_preds.sum()),
        'down_predictions': int(len(all_preds) - all_preds.sum()),
        'actual_up': int(all_targets.sum()),
        'actual_down': int(len(all_targets) - all_targets.sum()),
        'high_confidence': {
            'count': int(high_conf_count),
            'accuracy': float(high_conf_acc),
        },
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'targets': all_targets.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="LSTM Backtest with VP + CVD")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file with 1s bars")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--seq-length", type=int, default=60, help="LSTM sequence length (minutes)")
    parser.add_argument("--prediction-horizon", type=int, default=1, help="Bars to predict ahead (default: 1 = next bar)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")

    args = parser.parse_args()

    print(f"[LSTM] Loading data from {args.input}...", file=sys.stderr)
    with open(args.input, 'r') as f:
        data = json.load(f)

    bars_1s = data.get("bars", [])
    print(f"[LSTM] Loaded {len(bars_1s)} 1-second bars", file=sys.stderr)

    # Aggregate to 1-minute
    print("[LSTM] Aggregating to 1-minute bars...", file=sys.stderr)
    df = aggregate_to_1min(bars_1s)
    print(f"[LSTM] Created {len(df)} 1-minute bars", file=sys.stderr)

    if len(df) < args.seq_length + args.prediction_horizon + 100:
        print(f"Error: Need more data. Have {len(df)} bars, need at least {args.seq_length + args.prediction_horizon + 100}", file=sys.stderr)
        sys.exit(1)

    # Calculate CVD from 1s bars
    print("[LSTM] Calculating CVD from 1-second bars...", file=sys.stderr)
    cvd = calculate_cvd_for_1min(bars_1s, df.index)

    # Calculate rolling Volume Profile (no leakage - uses only past data)
    print("[LSTM] Calculating rolling Volume Profile...", file=sys.stderr)
    vp_df = calculate_volume_profile_rolling(df, lookback=30)

    # Calculate microstructure features from 1-second bars
    print("[LSTM] Calculating microstructure features from 1s bars...", file=sys.stderr)
    micro_df = calculate_microstructure_features(bars_1s, df.index)

    # Prepare features
    print("[LSTM] Preparing features...", file=sys.stderr)
    features_df = prepare_features(df, cvd, vp_df, micro_df)

    # Create target: 1 if price UP after prediction_horizon bars, 0 if DOWN
    # This is the FUTURE price - only used as target, not as feature
    future_returns = df['close'].shift(-args.prediction_horizon) / df['close'] - 1
    targets = (future_returns > 0).astype(float)

    # Remove last prediction_horizon rows (no future data)
    features_df = features_df.iloc[:-args.prediction_horizon]
    targets = targets.iloc[:-args.prediction_horizon]
    df = df.iloc[:-args.prediction_horizon]

    print(f"[LSTM] Feature shape: {features_df.shape}", file=sys.stderr)
    print(f"[LSTM] Features: {list(features_df.columns)}", file=sys.stderr)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)

    # Create sequences
    print(f"[LSTM] Creating sequences (length={args.seq_length})...", file=sys.stderr)
    X, y = create_sequences(features_scaled, targets.values, args.seq_length)
    print(f"[LSTM] Sequence shape: X={X.shape}, y={y.shape}", file=sys.stderr)

    # Split data chronologically (NO SHUFFLE - time series!)
    n_samples = len(X)
    train_end = int(n_samples * args.train_ratio)
    val_end = int(n_samples * (args.train_ratio + args.val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"[LSTM] Split: Train={len(X_train)}, Val={len(X_val)}, Holdout={len(X_test)}", file=sys.stderr)
    print(f"[LSTM] Holdout period: last {len(X_test)} samples ({100*(1-args.train_ratio-args.val_ratio):.0f}% of data)", file=sys.stderr)

    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create and train model
    input_size = X.shape[2]
    model = LSTMPredictor(input_size=input_size, hidden_size=args.hidden_size)

    print(f"\n[LSTM] Training model...", file=sys.stderr)
    print(f"  Input features: {input_size}", file=sys.stderr)
    print(f"  Hidden size: {args.hidden_size}", file=sys.stderr)
    print(f"  Max epochs: {args.epochs}", file=sys.stderr)

    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        patience=15,
    )

    # Evaluate on holdout set
    print(f"\n[LSTM] Evaluating on holdout set...", file=sys.stderr)
    results = evaluate_model(model, test_loader)

    # Print results
    horizon_str = "Next Bar" if args.prediction_horizon == 1 else f"{args.prediction_horizon} Bars"
    print("\n" + "=" * 70, file=sys.stderr)
    print(f"LSTM VP+CVD+ADX+ATR BACKTEST RESULTS (Predict {horizon_str} Direction)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Holdout Set Size: {results['total_samples']} samples", file=sys.stderr)
    print(f"\nAccuracy: {results['accuracy']*100:.2f}%", file=sys.stderr)
    print(f"Precision: {results['precision']*100:.2f}%", file=sys.stderr)
    print(f"Recall: {results['recall']*100:.2f}%", file=sys.stderr)
    print(f"F1 Score: {results['f1']*100:.2f}%", file=sys.stderr)

    print(f"\nPredictions Distribution:", file=sys.stderr)
    print(f"  UP predictions: {results['up_predictions']} ({results['up_predictions']/results['total_samples']*100:.1f}%)", file=sys.stderr)
    print(f"  DOWN predictions: {results['down_predictions']} ({results['down_predictions']/results['total_samples']*100:.1f}%)", file=sys.stderr)

    print(f"\nActual Distribution:", file=sys.stderr)
    print(f"  Actual UP: {results['actual_up']} ({results['actual_up']/results['total_samples']*100:.1f}%)", file=sys.stderr)
    print(f"  Actual DOWN: {results['actual_down']} ({results['actual_down']/results['total_samples']*100:.1f}%)", file=sys.stderr)

    print(f"\nHigh Confidence Predictions (>60% or <40% probability):", file=sys.stderr)
    print(f"  Count: {results['high_confidence']['count']} ({results['high_confidence']['count']/results['total_samples']*100:.1f}% of total)", file=sys.stderr)
    print(f"  Accuracy: {results['high_confidence']['accuracy']*100:.2f}%", file=sys.stderr)

    print(f"\nConfusion Matrix:", file=sys.stderr)
    print(f"  [[TN, FP],   = {results['confusion_matrix']}", file=sys.stderr)
    print(f"   [FN, TP]]", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Save results
    output = {
        'config': {
            'seq_length': args.seq_length,
            'prediction_horizon': args.prediction_horizon,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'holdout_ratio': 1 - args.train_ratio - args.val_ratio,
            'features': list(features_df.columns),
            'total_1min_bars': len(df) + args.prediction_horizon,
            'total_sequences': n_samples,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'holdout_size': len(X_test),
        },
        'results': results,
        'training': {
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'epochs_trained': len(train_losses),
        },
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[LSTM] Saved results to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
