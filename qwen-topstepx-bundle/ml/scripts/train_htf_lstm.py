#!/usr/bin/env python3
"""
Long-Term Multi-Timeframe LSTM - TRAINING WITH PROPER SPLITS

Features:
- Proper train/val/test split with holdout period
- No data leakage (scaler fit only on training data, time-series aware split)
- GPU training support
- Predicts 5-day price direction

Splits:
- Train: 70% of data (oldest)
- Validation: 15% of data (middle)
- Test: 15% of data (newest, holdout)
"""

import json
import sys
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# PyTorch for LSTM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# LSTM configuration
DAILY_SEQ_LEN = 60       # 60 days lookback (reduced from 365 due to limited data)
EMBEDDING_DIM = 32
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15

# Data splits (time-series aware)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_bars(filepath: str) -> List[Dict]:
    """Load bars from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('bars', [])


# =============================================================================
# CVD CALCULATION
# =============================================================================

def calculate_cvd_hourly(bars_1h: List[Dict]) -> Tuple[List[float], List[float]]:
    """Calculate Cumulative Volume Delta (CVD) for 1-hour bars."""
    if not bars_1h:
        return [], []

    cvd_values = [0.0]

    for i, bar in enumerate(bars_1h):
        bar_range = bar['h'] - bar['l']

        if bar_range > 0:
            close_position = (bar['c'] - bar['l']) / bar_range
            delta = bar['v'] * (2 * close_position - 1)
        else:
            delta = bar['v'] if bar['c'] >= bar['o'] else -bar['v']

        if i > 0:
            cvd_values.append(cvd_values[-1] + delta)
        else:
            cvd_values[0] = delta

    # Calculate EMA of CVD (20-period)
    cvd_ema = [cvd_values[0]]
    alpha = 2.0 / (20 + 1)

    for i in range(1, len(cvd_values)):
        ema = cvd_values[i] * alpha + cvd_ema[-1] * (1 - alpha)
        cvd_ema.append(ema)

    return cvd_values, cvd_ema


# =============================================================================
# DAILY VOLUME PROFILE
# =============================================================================

def calculate_daily_volume_profile(bars_daily: List[Dict], lookback: int = 30,
                                   tick_size: float = 0.25) -> List[Dict]:
    """Calculate Volume Profile for each daily bar."""
    vp_results = []

    for i in range(len(bars_daily)):
        start_idx = max(0, i - lookback + 1)
        window = bars_daily[start_idx:i+1]

        if len(window) < 2:
            vp_results.append({'poc': 0, 'vah': 0, 'val': 0})
            continue

        all_highs = [b['h'] for b in window]
        all_lows = [b['l'] for b in window]
        price_high = max(all_highs)
        price_low = min(all_lows)

        if price_high <= price_low:
            vp_results.append({'poc': 0, 'vah': 0, 'val': 0})
            continue

        num_bins = max(10, int((price_high - price_low) / tick_size))
        bins = np.linspace(price_low, price_high, num_bins + 1)
        volume_at_price = np.zeros(num_bins)

        for bar in window:
            bar_vol = bar.get('v', 0)
            bar_range = bar['h'] - bar['l']

            if bar_range > 0:
                for j in range(num_bins):
                    bin_low = bins[j]
                    bin_high = bins[j + 1]
                    overlap_low = max(bar['l'], bin_low)
                    overlap_high = min(bar['h'], bin_high)

                    if overlap_high > overlap_low:
                        overlap_pct = (overlap_high - overlap_low) / bar_range
                        volume_at_price[j] += bar_vol * overlap_pct
            else:
                bin_idx = np.searchsorted(bins, bar['c']) - 1
                if 0 <= bin_idx < num_bins:
                    volume_at_price[bin_idx] += bar_vol

        poc_idx = np.argmax(volume_at_price)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        total_volume = np.sum(volume_at_price)
        value_area_volume = total_volume * 0.70

        vah_idx = poc_idx
        val_idx = poc_idx
        accumulated_volume = volume_at_price[poc_idx]

        while accumulated_volume < value_area_volume:
            vol_above = volume_at_price[vah_idx + 1] if vah_idx + 1 < num_bins else 0
            vol_below = volume_at_price[val_idx - 1] if val_idx > 0 else 0

            if vol_above >= vol_below and vah_idx + 1 < num_bins:
                vah_idx += 1
                accumulated_volume += vol_above
            elif val_idx > 0:
                val_idx -= 1
                accumulated_volume += vol_below
            else:
                break

        vah = (bins[vah_idx] + bins[vah_idx + 1]) / 2
        val = (bins[val_idx] + bins[val_idx + 1]) / 2

        vp_results.append({'poc': poc, 'vah': vah, 'val': val})

    return vp_results


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_rsi(bars: List[Dict], period: int = 14) -> List[float]:
    """Calculate RSI."""
    if len(bars) < period + 1:
        return [50.0] * len(bars)

    closes = [b['c'] for b in bars]
    rsi_values = [50.0] * len(bars)

    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(closes)):
        if avg_loss == 0:
            rsi_values[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))

        if i < len(gains):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    return rsi_values


def calculate_macd(bars: List[Dict], fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD and signal line."""
    closes = [b['c'] for b in bars]

    if len(closes) < slow + signal:
        return [0.0] * len(bars), [0.0] * len(bars)

    def ema(data, period):
        result = [data[0]]
        multiplier = 2.0 / (period + 1)
        for i in range(1, len(data)):
            result.append((data[i] - result[-1]) * multiplier + result[-1])
        return result

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(closes))]
    signal_line = ema(macd_line, signal)

    return macd_line, signal_line


def calculate_atr(bars: List[Dict], period: int = 14) -> List[float]:
    """Calculate Average True Range."""
    if len(bars) < period + 1:
        return [0.0] * len(bars)

    atr_values = [0.0] * len(bars)
    tr_values = []

    for i in range(1, len(bars)):
        high = bars[i]['h']
        low = bars[i]['l']
        prev_close = bars[i-1]['c']
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)

        if i >= period:
            if i == period:
                atr = sum(tr_values[:period]) / period
            else:
                atr = (atr_values[i-1] * (period - 1) + tr) / period
            atr_values[i] = atr

    return atr_values


def calculate_bollinger_bands(bars: List[Dict], period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands."""
    if len(bars) < period:
        return [0.0] * len(bars), [0.0] * len(bars), [0.0] * len(bars)

    closes = [b['c'] for b in bars]
    middle = [0.0] * len(bars)
    upper = [0.0] * len(bars)
    lower = [0.0] * len(bars)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        sma = sum(window) / period
        variance = sum((x - sma) ** 2 for x in window) / period
        std = variance ** 0.5

        middle[i] = sma
        upper[i] = sma + std_dev * std
        lower[i] = sma - std_dev * std

    return middle, upper, lower


# =============================================================================
# LSTM MODEL
# =============================================================================

class LongTermMultiTimeframeLSTM(nn.Module):
    """Long-term LSTM that processes 1h, 4h, and daily bars together."""

    def __init__(self, input_size=26, hidden_size=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 output_size=EMBEDDING_DIM, dropout=DROPOUT):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        """Forward pass."""
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        normed = self.layer_norm(last_hidden)
        embedding = self.fc(normed)
        return embedding


class DirectionClassifier(nn.Module):
    """Classification head for price direction prediction."""

    def __init__(self, embedding_dim=EMBEDDING_DIM, dropout=DROPOUT):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, embedding):
        """Forward pass."""
        return self.classifier(embedding)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_longterm_features(
    bars_1h: List[Dict],
    bars_4h: List[Dict],
    bars_daily: List[Dict],
    cvd_1h: List[float],
    cvd_ema_1h: List[float],
    daily_vp: List[Dict],
    num_days: int = DAILY_SEQ_LEN
) -> Optional[np.ndarray]:
    """Extract features for long-term LSTM."""
    if len(bars_daily) < num_days:
        return None

    # Calculate indicators
    rsi_daily = calculate_rsi(bars_daily, period=14)
    macd_daily, macd_signal = calculate_macd(bars_daily)
    atr_daily = calculate_atr(bars_daily, period=14)
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_daily, period=20)

    # Calculate moving averages
    closes_daily = [b['c'] for b in bars_daily]
    ma20 = [0.0] * len(closes_daily)
    ma50 = [0.0] * len(closes_daily)

    for i in range(len(closes_daily)):
        if i >= 20:
            ma20[i] = sum(closes_daily[i-19:i+1]) / 20
        if i >= 50:
            ma50[i] = sum(closes_daily[i-49:i+1]) / 50

    features = []

    # Process last num_days
    for day_idx in range(len(bars_daily) - num_days, len(bars_daily)):
        daily_bar = bars_daily[day_idx]

        # Corresponding 4h and 1h bars (simplified - last bar of the day)
        h4_idx = min((day_idx + 1) * 6 - 1, len(bars_4h) - 1)
        h1_idx = min((day_idx + 1) * 24 - 1, len(bars_1h) - 1)

        h4_bar = bars_4h[h4_idx] if h4_idx < len(bars_4h) else daily_bar
        h1_bar = bars_1h[h1_idx] if h1_idx < len(bars_1h) else daily_bar

        # Normalize by daily close
        daily_close = daily_bar['c']
        if daily_close == 0:
            daily_close = 1

        # Daily OHLCV normalized (5)
        daily_o_norm = (daily_bar['o'] - daily_close) / daily_close
        daily_h_norm = (daily_bar['h'] - daily_close) / daily_close
        daily_l_norm = (daily_bar['l'] - daily_close) / daily_close
        daily_range_norm = (daily_bar['h'] - daily_bar['l']) / daily_close
        daily_body_norm = (daily_bar['c'] - daily_bar['o']) / daily_close

        # Daily Volume Profile (3)
        vp = daily_vp[day_idx] if day_idx < len(daily_vp) else {'poc': 0, 'vah': 0, 'val': 0}
        vp_poc_norm = (daily_close - vp['poc']) / vp['poc'] if vp['poc'] > 0 else 0
        vp_vah_norm = (daily_close - vp['vah']) / vp['vah'] if vp['vah'] > 0 else 0
        vp_val_norm = (daily_close - vp['val']) / vp['val'] if vp['val'] > 0 else 0

        # 4-hour OHLCV normalized (5)
        h4_o_norm = (h4_bar['o'] - daily_close) / daily_close
        h4_h_norm = (h4_bar['h'] - daily_close) / daily_close
        h4_l_norm = (h4_bar['l'] - daily_close) / daily_close
        h4_range_norm = (h4_bar['h'] - h4_bar['l']) / daily_close
        h4_body_norm = (h4_bar['c'] - h4_bar['o']) / daily_close

        # 1-hour OHLCV normalized (5)
        h1_o_norm = (h1_bar['o'] - daily_close) / daily_close
        h1_h_norm = (h1_bar['h'] - daily_close) / daily_close
        h1_l_norm = (h1_bar['l'] - daily_close) / daily_close
        h1_range_norm = (h1_bar['h'] - h1_bar['l']) / daily_close
        h1_body_norm = (h1_bar['c'] - h1_bar['o']) / daily_close

        # 1-hour CVD (2)
        cvd_now = cvd_1h[h1_idx] if h1_idx < len(cvd_1h) else 0
        cvd_ema_now = cvd_ema_1h[h1_idx] if h1_idx < len(cvd_ema_1h) else 0
        cvd_vs_ema = (cvd_now - cvd_ema_now) / max(abs(cvd_ema_now), 1)
        cvd_norm = cvd_now / max(abs(cvd_now), 1)

        # Daily indicators (3)
        rsi_norm = rsi_daily[day_idx] / 100.0
        macd_norm = (macd_daily[day_idx] - macd_signal[day_idx]) / daily_close
        atr_norm = atr_daily[day_idx] / daily_close

        # Bollinger Band position (1)
        if bb_upper[day_idx] > bb_lower[day_idx]:
            bb_position = (daily_close - bb_lower[day_idx]) / (bb_upper[day_idx] - bb_lower[day_idx])
        else:
            bb_position = 0.5

        # Trend features (2)
        close_vs_ma20 = (daily_close - ma20[day_idx]) / ma20[day_idx] if ma20[day_idx] > 0 else 0
        close_vs_ma50 = (daily_close - ma50[day_idx]) / ma50[day_idx] if ma50[day_idx] > 0 else 0

        # Combine all features (26 total)
        feature_vec = [
            daily_o_norm, daily_h_norm, daily_l_norm, daily_range_norm, daily_body_norm,
            vp_poc_norm, vp_vah_norm, vp_val_norm,
            h4_o_norm, h4_h_norm, h4_l_norm, h4_range_norm, h4_body_norm,
            h1_o_norm, h1_h_norm, h1_l_norm, h1_range_norm, h1_body_norm,
            cvd_norm, cvd_vs_ema,
            rsi_norm, macd_norm, atr_norm,
            bb_position,
            close_vs_ma20, close_vs_ma50
        ]

        features.append(feature_vec)

    return np.array(features, dtype=np.float32)


# =============================================================================
# DATASET
# =============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, classifier, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
    """Train the LSTM model with early stopping."""
    model = model.to(DEVICE)
    classifier = classifier.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            embeddings = model(batch_X)
            logits = classifier(embeddings)

            loss = criterion(logits, batch_y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                1.0
            )
            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (predictions == batch_y.unsqueeze(1)).sum().item()
            train_total += batch_y.size(0)

        # Validation
        model.eval()
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                embeddings = model(batch_X)
                logits = classifier(embeddings)

                loss = criterion(logits, batch_y.unsqueeze(1))

                val_loss += loss.item()
                predictions = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (predictions == batch_y.unsqueeze(1)).sum().item()
                val_total += batch_y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.3%}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.3%}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    return history, best_val_loss


def evaluate_model(model, classifier, test_loader):
    """Evaluate model on test set."""
    model.eval()
    classifier.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            embeddings = model(batch_X)
            logits = classifier(embeddings)

            predictions = (torch.sigmoid(logits) > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    accuracy = np.mean(all_predictions == all_targets)

    # Calculate per-class metrics
    true_positives = np.sum((all_predictions == 1) & (all_targets == 1))
    false_positives = np.sum((all_predictions == 1) & (all_targets == 0))
    true_negatives = np.sum((all_predictions == 0) & (all_targets == 0))
    false_negatives = np.sum((all_predictions == 0) & (all_targets == 1))

    print("\n" + "="*70)
    print("TEST SET PERFORMANCE")
    print("="*70)
    print(f"Accuracy: {accuracy:.3%}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Down    Up")
    print(f"Actual Down:    {true_negatives:4d}  {false_positives:4d}")
    print(f"Actual Up:      {false_negatives:4d}  {true_positives:4d}")

    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"\nPrecision (Up): {precision:.3%}")

    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"Recall (Up): {recall:.3%}")

    return accuracy


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("LONG-TERM HTF LSTM TRAINING")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Sequence length: {DAILY_SEQ_LEN} days")
    print(f"Train/Val/Test: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    print("="*70)

    # Paths
    data_dir = "/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data_TEST/longterm"
    output_dir = "/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/models_TEST/longterm"
    os.makedirs(output_dir, exist_ok=True)

    # Load bars
    print("\nLoading bar data...")
    bars_1h = load_bars(os.path.join(data_dir, 'bars_1h_enq_2025_merged.json'))
    bars_4h = load_bars(os.path.join(data_dir, 'bars_4h_enq_2025_merged.json'))
    bars_daily = load_bars(os.path.join(data_dir, 'bars_1d_enq_2025_merged.json'))

    print(f"  1-hour bars: {len(bars_1h)}")
    print(f"  4-hour bars: {len(bars_4h)}")
    print(f"  Daily bars: {len(bars_daily)}")

    if len(bars_daily) < DAILY_SEQ_LEN + 10:
        print(f"\nError: Need at least {DAILY_SEQ_LEN + 10} daily bars, only have {len(bars_daily)}")
        return

    # Calculate CVD for 1-hour bars
    print("\nComputing 1-hour CVD...")
    cvd_1h, cvd_ema_1h = calculate_cvd_hourly(bars_1h)

    # Calculate daily Volume Profile
    print("Computing daily Volume Profile...")
    daily_vp = calculate_daily_volume_profile(bars_daily, lookback=30)

    # Create training samples
    print("\nCreating training samples...")
    closes_daily = [b['c'] for b in bars_daily]

    X_all = []
    y_all = []

    for i in range(DAILY_SEQ_LEN, len(bars_daily) - 5):
        seq_features = extract_longterm_features(
            bars_1h[:i*24], bars_4h[:i*6], bars_daily[:i],
            cvd_1h[:i*24], cvd_ema_1h[:i*24], daily_vp[:i],
            num_days=DAILY_SEQ_LEN
        )

        if seq_features is not None:
            X_all.append(seq_features)

            # Label: 1 if price higher in 5 days, 0 otherwise
            current_price = closes_daily[i]
            future_price = closes_daily[i + 5]
            label = 1.0 if future_price > current_price else 0.0
            y_all.append(label)

    if len(X_all) == 0:
        print("Error: No training samples generated")
        return

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    print(f"  Total samples: {len(X_all)}")
    print(f"  Feature shape: {X_all.shape}")
    print(f"  Positive ratio: {y_all.mean():.3%}")

    # Time-series split (NO SHUFFLING)
    n_samples = len(X_all)
    train_end = int(n_samples * TRAIN_RATIO)
    val_end = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

    X_train = X_all[:train_end]
    y_train = y_all[:train_end]

    X_val = X_all[train_end:val_end]
    y_val = y_all[train_end:val_end]

    X_test = X_all[val_end:]
    y_test = y_all[val_end:]

    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples (oldest)")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples (newest, holdout)")

    # Normalize features (FIT ONLY ON TRAINING DATA - NO LEAKAGE!)
    print("\nNormalizing features...")
    scaler = StandardScaler()

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    # FIT ONLY ON TRAINING DATA
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    print("  ✓ Scaler fit on training data only (no leakage)")

    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    print("\nInitializing LSTM model...")
    model = LongTermMultiTimeframeLSTM(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_size=EMBEDDING_DIM,
        dropout=DROPOUT
    )
    classifier = DirectionClassifier(embedding_dim=EMBEDDING_DIM, dropout=DROPOUT)

    print(f"  LSTM parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # Train model
    print(f"\nTraining for {EPOCHS} epochs...")
    history, best_val_loss = train_model(
        model, classifier, train_loader, val_loader,
        epochs=EPOCHS, patience=PATIENCE
    )

    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET (HOLDOUT)")
    print("="*70)
    test_acc = evaluate_model(model, classifier, test_loader)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'scaler': scaler,
        'history': history,
        'config': {
            'input_size': X_train.shape[2],
            'hidden_size': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'embedding_dim': EMBEDDING_DIM,
            'dropout': DROPOUT,
            'daily_seq_len': DAILY_SEQ_LEN,
        }
    }, os.path.join(output_dir, 'htf_lstm_model.pt'))

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.3%}")
    print(f"\nModel saved to: {output_dir}/htf_lstm_model.pt")


if __name__ == "__main__":
    main()
