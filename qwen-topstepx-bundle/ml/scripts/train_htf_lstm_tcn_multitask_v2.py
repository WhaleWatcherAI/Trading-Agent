#!/usr/bin/env python3
"""
Long-Term Multi-Timeframe LSTM + TCN - MULTI-TASK LEARNING V2
WITH MICROSTRUCTURE FEATURES

Architecture:
- LSTM branch: Sequential processing of multi-timeframe features
- TCN branch: Temporal convolutional network for pattern detection
- Combined embedding fed to 3 prediction heads:
  1. Next day (1d) up/down
  2. Next 4h bar up/down
  3. Next 1h bar up/down

Enhanced Features:
- Daily Volume Profile calculated from 1-second candles (high precision)
- CVD calculated from 1-second candles, aggregated to 5-minute bars
- CVD EMA (20-period) on 5-minute data
- Proper train/val/test split with holdout period
- No data leakage
- Multi-task learning with weighted losses
- GPU training support
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

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# LSTM configuration
DAILY_SEQ_LEN = 60
EMBEDDING_DIM = 64  # Increased for richer representations
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3

# TCN configuration
TCN_CHANNELS = [64, 128, 128, 64]  # Channel progression
KERNEL_SIZE = 3
TCN_DROPOUT = 0.2

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15

# Loss weights for multi-task learning
LOSS_WEIGHTS = {
    '1d': 1.0,   # Daily prediction
    '4h': 0.7,   # 4-hour prediction
    '1h': 0.5    # 1-hour prediction
}

# Data splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DATA LOADING (Same as before)
# =============================================================================

def load_bars(filepath: str) -> List[Dict]:
    """Load bars from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('bars', [])


def load_microstructure_features(data_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load microstructure features calculated from 1-second bars.

    Returns:
        Tuple of (daily_vp_dict, cvd_5min_dict, cvd_ema_5min_dict)
    """
    print("Loading microstructure features...")

    # Load daily volume profile from 1s candles
    vp_path = os.path.join(data_dir, "daily_vp_from_1s.json")
    with open(vp_path, 'r') as f:
        daily_vp = json.load(f)
    print(f"  Loaded {len(daily_vp)} daily volume profiles")

    # Load CVD 5min from 1s candles
    cvd_path = os.path.join(data_dir, "cvd_5min_from_1s.json")
    with open(cvd_path, 'r') as f:
        cvd_5min = json.load(f)
    print(f"  Loaded {len(cvd_5min)} 5-minute CVD values")

    # Load CVD EMA 5min
    cvd_ema_path = os.path.join(data_dir, "cvd_ema_5min_from_1s.json")
    with open(cvd_ema_path, 'r') as f:
        cvd_ema_5min = json.load(f)
    print(f"  Loaded {len(cvd_ema_5min)} 5-minute CVD EMA values")

    return daily_vp, cvd_5min, cvd_ema_5min


def get_daily_vp_for_bar(bar: Dict, daily_vp_dict: Dict) -> Dict:
    """
    Get daily volume profile for a specific bar by matching date.

    Returns:
        Dict with keys: poc, vah, val, total_volume
    """
    # Extract date from bar timestamp
    ts_str = bar.get('timestamp', '')
    try:
        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        date_key = ts.strftime('%Y-%m-%d')

        if date_key in daily_vp_dict:
            return daily_vp_dict[date_key]
    except:
        pass

    return {'poc': 0, 'vah': 0, 'val': 0, 'total_volume': 0}


def get_cvd_5min_for_timestamp(timestamp: str, cvd_5min_dict: Dict, cvd_ema_dict: Dict) -> Tuple[float, float]:
    """
    Get CVD and CVD EMA for a specific timestamp.

    Returns most recent CVD value at or before the given timestamp.

    Returns:
        Tuple of (cvd_value, cvd_ema_value)
    """
    try:
        target_ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        # Find most recent CVD value at or before target timestamp
        best_cvd = 0
        best_cvd_ema = 0
        best_diff = timedelta(days=999999)

        for cvd_ts_str in cvd_5min_dict.keys():
            cvd_ts = datetime.fromisoformat(cvd_ts_str.replace('Z', '+00:00'))

            if cvd_ts <= target_ts:
                diff = target_ts - cvd_ts
                if diff < best_diff:
                    best_diff = diff
                    best_cvd = cvd_5min_dict[cvd_ts_str]
                    best_cvd_ema = cvd_ema_dict.get(cvd_ts_str, 0)

        return best_cvd, best_cvd_ema
    except:
        return 0, 0


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

    cvd_ema = [cvd_values[0]]
    alpha = 2.0 / (20 + 1)

    for i in range(1, len(cvd_values)):
        ema = cvd_values[i] * alpha + cvd_ema[-1] * (1 - alpha)
        cvd_ema.append(ema)

    return cvd_values, cvd_ema


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


def extract_longterm_features(
    bars_1h: List[Dict],
    bars_4h: List[Dict],
    bars_daily: List[Dict],
    cvd_1h: List[float],
    cvd_ema_1h: List[float],
    daily_vp: List[Dict],
    daily_vp_1s: Dict,  # NEW: High-res VP from 1s candles
    cvd_5min: Dict,      # NEW: CVD from 1s candles (5min aggregation)
    cvd_ema_5min: Dict,  # NEW: CVD EMA from 1s candles (5min aggregation)
    num_days: int = DAILY_SEQ_LEN
) -> Optional[np.ndarray]:
    """
    Extract features for long-term LSTM.

    Now includes microstructure features from 1-second candles:
    - High-resolution daily volume profile
    - 5-minute CVD from 1-second data
    - 5-minute CVD EMA from 1-second data
    """
    if len(bars_daily) < num_days:
        return None

    rsi_daily = calculate_rsi(bars_daily, period=14)
    macd_daily, macd_signal = calculate_macd(bars_daily)
    atr_daily = calculate_atr(bars_daily, period=14)
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_daily, period=20)

    closes_daily = [b['c'] for b in bars_daily]
    ma20 = [0.0] * len(closes_daily)
    ma50 = [0.0] * len(closes_daily)

    for i in range(len(closes_daily)):
        if i >= 20:
            ma20[i] = sum(closes_daily[i-19:i+1]) / 20
        if i >= 50:
            ma50[i] = sum(closes_daily[i-49:i+1]) / 50

    features = []

    for day_idx in range(len(bars_daily) - num_days, len(bars_daily)):
        daily_bar = bars_daily[day_idx]

        h4_idx = min((day_idx + 1) * 6 - 1, len(bars_4h) - 1)
        h1_idx = min((day_idx + 1) * 24 - 1, len(bars_1h) - 1)

        h4_bar = bars_4h[h4_idx] if h4_idx < len(bars_4h) else daily_bar
        h1_bar = bars_1h[h1_idx] if h1_idx < len(bars_1h) else daily_bar

        daily_close = daily_bar['c']
        if daily_close == 0:
            daily_close = 1

        # OHLCV features
        daily_o_norm = (daily_bar['o'] - daily_close) / daily_close
        daily_h_norm = (daily_bar['h'] - daily_close) / daily_close
        daily_l_norm = (daily_bar['l'] - daily_close) / daily_close
        daily_range_norm = (daily_bar['h'] - daily_bar['l']) / daily_close
        daily_body_norm = (daily_bar['c'] - daily_bar['o']) / daily_close

        # Original VP features (from daily/hourly bars)
        vp = daily_vp[day_idx] if day_idx < len(daily_vp) else {'poc': 0, 'vah': 0, 'val': 0}
        vp_poc_norm = (daily_close - vp['poc']) / vp['poc'] if vp['poc'] > 0 else 0
        vp_vah_norm = (daily_close - vp['vah']) / vp['vah'] if vp['vah'] > 0 else 0
        vp_val_norm = (daily_close - vp['val']) / vp['val'] if vp['val'] > 0 else 0

        # NEW: High-resolution VP from 1-second candles
        vp_1s = get_daily_vp_for_bar(daily_bar, daily_vp_1s)
        vp_1s_poc_norm = (daily_close - vp_1s['poc']) / vp_1s['poc'] if vp_1s['poc'] > 0 else 0
        vp_1s_vah_norm = (daily_close - vp_1s['vah']) / vp_1s['vah'] if vp_1s['vah'] > 0 else 0
        vp_1s_val_norm = (daily_close - vp_1s['val']) / vp_1s['val'] if vp_1s['val'] > 0 else 0
        vp_1s_volume_norm = vp_1s.get('total_volume', 0) / 1000000.0  # Normalize volume

        # 4h and 1h features
        h4_o_norm = (h4_bar['o'] - daily_close) / daily_close
        h4_h_norm = (h4_bar['h'] - daily_close) / daily_close
        h4_l_norm = (h4_bar['l'] - daily_close) / daily_close
        h4_range_norm = (h4_bar['h'] - h4_bar['l']) / daily_close
        h4_body_norm = (h4_bar['c'] - h4_bar['o']) / daily_close

        h1_o_norm = (h1_bar['o'] - daily_close) / daily_close
        h1_h_norm = (h1_bar['h'] - daily_close) / daily_close
        h1_l_norm = (h1_bar['l'] - daily_close) / daily_close
        h1_range_norm = (h1_bar['h'] - h1_bar['l']) / daily_close
        h1_body_norm = (h1_bar['c'] - h1_bar['o']) / daily_close

        # Original CVD from 1h bars
        cvd_now = cvd_1h[h1_idx] if h1_idx < len(cvd_1h) else 0
        cvd_ema_now = cvd_ema_1h[h1_idx] if h1_idx < len(cvd_ema_1h) else 0
        cvd_vs_ema = (cvd_now - cvd_ema_now) / max(abs(cvd_ema_now), 1)
        cvd_norm = cvd_now / max(abs(cvd_now), 1)

        # NEW: CVD from 1-second candles (5min aggregation)
        h1_timestamp = h1_bar.get('timestamp', '')
        cvd_5min_val, cvd_ema_5min_val = get_cvd_5min_for_timestamp(
            h1_timestamp, cvd_5min, cvd_ema_5min
        )
        cvd_5min_norm = cvd_5min_val / max(abs(cvd_5min_val), 1) if cvd_5min_val != 0 else 0
        cvd_5min_vs_ema = (cvd_5min_val - cvd_ema_5min_val) / max(abs(cvd_ema_5min_val), 1) if cvd_ema_5min_val != 0 else 0

        # Technical indicators
        rsi_norm = rsi_daily[day_idx] / 100.0
        macd_norm = (macd_daily[day_idx] - macd_signal[day_idx]) / daily_close
        atr_norm = atr_daily[day_idx] / daily_close

        if bb_upper[day_idx] > bb_lower[day_idx]:
            bb_position = (daily_close - bb_lower[day_idx]) / (bb_upper[day_idx] - bb_lower[day_idx])
        else:
            bb_position = 0.5

        close_vs_ma20 = (daily_close - ma20[day_idx]) / ma20[day_idx] if ma20[day_idx] > 0 else 0
        close_vs_ma50 = (daily_close - ma50[day_idx]) / ma50[day_idx] if ma50[day_idx] > 0 else 0

        # Combine all features (now 32 features instead of 26)
        feature_vec = [
            # Daily OHLCV (5)
            daily_o_norm, daily_h_norm, daily_l_norm, daily_range_norm, daily_body_norm,
            # Original VP from daily bars (3)
            vp_poc_norm, vp_vah_norm, vp_val_norm,
            # NEW: High-res VP from 1s candles (4)
            vp_1s_poc_norm, vp_1s_vah_norm, vp_1s_val_norm, vp_1s_volume_norm,
            # 4h OHLCV (5)
            h4_o_norm, h4_h_norm, h4_l_norm, h4_range_norm, h4_body_norm,
            # 1h OHLCV (5)
            h1_o_norm, h1_h_norm, h1_l_norm, h1_range_norm, h1_body_norm,
            # Original CVD from 1h bars (2)
            cvd_norm, cvd_vs_ema,
            # NEW: CVD from 1s candles (5min aggregation) (2)
            cvd_5min_norm, cvd_5min_vs_ema,
            # Technical indicators (3)
            rsi_norm, macd_norm, atr_norm,
            # Bollinger position (1)
            bb_position,
            # Trend features (2)
            close_vs_ma20, close_vs_ma50
        ]

        features.append(feature_vec)

    return np.array(features, dtype=np.float32)


# =============================================================================
# TCN (Temporal Convolutional Network)
# =============================================================================

class Chomp1d(nn.Module):
    """Removes padding from the end of a sequence."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    """Single temporal block with causal convolutions."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass."""
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        # Handle size mismatch if any
        if out.size(2) != res.size(2):
            min_len = min(out.size(2), res.size(2))
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]

        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network."""

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, features]
        Returns:
            [batch_size, seq_len, num_channels[-1]]
        """
        # TCN expects [batch, features, seq_len]
        x = x.transpose(1, 2)
        out = self.network(x)
        # Back to [batch, seq_len, features]
        out = out.transpose(1, 2)
        return out


# =============================================================================
# HYBRID LSTM + TCN MODEL
# =============================================================================

class HybridLSTMTCN(nn.Module):
    """Hybrid LSTM + TCN for multi-timeframe prediction."""

    def __init__(self, input_size=26, lstm_hidden=HIDDEN_DIM, lstm_layers=NUM_LAYERS,
                 tcn_channels=TCN_CHANNELS, kernel_size=KERNEL_SIZE,
                 embedding_dim=EMBEDDING_DIM, dropout=DROPOUT):
        super().__init__()

        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)

        # TCN branch
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=TCN_DROPOUT
        )
        self.tcn_norm = nn.LayerNorm(tcn_channels[-1])

        # Fusion layer
        combined_size = lstm_hidden + tcn_channels[-1]
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-task prediction heads
        self.head_1d = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

        self.head_4h = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

        self.head_1h = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, features]

        Returns:
            logits_1d, logits_4h, logits_1h
        """
        # LSTM branch
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_embedding = self.lstm_norm(h_n[-1])  # Last hidden state

        # TCN branch
        tcn_out = self.tcn(x)
        tcn_embedding = self.tcn_norm(tcn_out[:, -1, :])  # Last timestep

        # Fuse embeddings
        combined = torch.cat([lstm_embedding, tcn_embedding], dim=1)
        fused_embedding = self.fusion(combined)

        # Multi-task predictions
        logits_1d = self.head_1d(fused_embedding)
        logits_4h = self.head_4h(fused_embedding)
        logits_1h = self.head_1h(fused_embedding)

        return logits_1d, logits_4h, logits_1h


# =============================================================================
# DATASET
# =============================================================================

class MultiTaskDataset(Dataset):
    """PyTorch dataset for multi-task time series."""

    def __init__(self, X, y_1d, y_4h, y_1h):
        self.X = torch.FloatTensor(X)
        self.y_1d = torch.FloatTensor(y_1d)
        self.y_4h = torch.FloatTensor(y_4h)
        self.y_1h = torch.FloatTensor(y_1h)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_1d[idx], self.y_4h[idx], self.y_1h[idx]


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
    """Train the hybrid model with multi-task loss."""
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc_1d': [], 'train_acc_4h': [], 'train_acc_1h': [],
        'val_acc_1d': [], 'val_acc_4h': [], 'val_acc_1h': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct_1d = train_correct_4h = train_correct_1h = 0
        train_total = 0

        for batch_X, batch_y_1d, batch_y_4h, batch_y_1h in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y_1d = batch_y_1d.to(DEVICE)
            batch_y_4h = batch_y_4h.to(DEVICE)
            batch_y_1h = batch_y_1h.to(DEVICE)

            optimizer.zero_grad()

            logits_1d, logits_4h, logits_1h = model(batch_X)

            # Multi-task loss
            loss_1d = criterion(logits_1d, batch_y_1d.unsqueeze(1))
            loss_4h = criterion(logits_4h, batch_y_4h.unsqueeze(1))
            loss_1h = criterion(logits_1h, batch_y_1h.unsqueeze(1))

            loss = (LOSS_WEIGHTS['1d'] * loss_1d +
                   LOSS_WEIGHTS['4h'] * loss_4h +
                   LOSS_WEIGHTS['1h'] * loss_1h)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracies
            pred_1d = (torch.sigmoid(logits_1d) > 0.5).float()
            pred_4h = (torch.sigmoid(logits_4h) > 0.5).float()
            pred_1h = (torch.sigmoid(logits_1h) > 0.5).float()

            train_correct_1d += (pred_1d == batch_y_1d.unsqueeze(1)).sum().item()
            train_correct_4h += (pred_4h == batch_y_4h.unsqueeze(1)).sum().item()
            train_correct_1h += (pred_1h == batch_y_1h.unsqueeze(1)).sum().item()
            train_total += batch_y_1d.size(0)

        # Validation
        model.eval()
        val_loss = 0
        val_correct_1d = val_correct_4h = val_correct_1h = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y_1d, batch_y_4h, batch_y_1h in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y_1d = batch_y_1d.to(DEVICE)
                batch_y_4h = batch_y_4h.to(DEVICE)
                batch_y_1h = batch_y_1h.to(DEVICE)

                logits_1d, logits_4h, logits_1h = model(batch_X)

                loss_1d = criterion(logits_1d, batch_y_1d.unsqueeze(1))
                loss_4h = criterion(logits_4h, batch_y_4h.unsqueeze(1))
                loss_1h = criterion(logits_1h, batch_y_1h.unsqueeze(1))

                loss = (LOSS_WEIGHTS['1d'] * loss_1d +
                       LOSS_WEIGHTS['4h'] * loss_4h +
                       LOSS_WEIGHTS['1h'] * loss_1h)

                val_loss += loss.item()

                pred_1d = (torch.sigmoid(logits_1d) > 0.5).float()
                pred_4h = (torch.sigmoid(logits_4h) > 0.5).float()
                pred_1h = (torch.sigmoid(logits_1h) > 0.5).float()

                val_correct_1d += (pred_1d == batch_y_1d.unsqueeze(1)).sum().item()
                val_correct_4h += (pred_4h == batch_y_4h.unsqueeze(1)).sum().item()
                val_correct_1h += (pred_1h == batch_y_1h.unsqueeze(1)).sum().item()
                val_total += batch_y_1d.size(0)

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_acc_1d = train_correct_1d / train_total
        train_acc_4h = train_correct_4h / train_total
        train_acc_1h = train_correct_1h / train_total

        val_acc_1d = val_correct_1d / val_total
        val_acc_4h = val_correct_4h / val_total
        val_acc_1h = val_correct_1h / val_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc_1d'].append(train_acc_1d)
        history['train_acc_4h'].append(train_acc_4h)
        history['train_acc_1h'].append(train_acc_1h)
        history['val_acc_1d'].append(val_acc_1d)
        history['val_acc_4h'].append(val_acc_4h)
        history['val_acc_1h'].append(val_acc_1h)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"    1d Acc: {train_acc_1d:.3%}, 4h Acc: {train_acc_4h:.3%}, 1h Acc: {train_acc_1h:.3%}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"    1d Acc: {val_acc_1d:.3%}, 4h Acc: {val_acc_4h:.3%}, 1h Acc: {val_acc_1h:.3%}")

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


def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()

    all_pred_1d = []
    all_pred_4h = []
    all_pred_1h = []
    all_true_1d = []
    all_true_4h = []
    all_true_1h = []

    with torch.no_grad():
        for batch_X, batch_y_1d, batch_y_4h, batch_y_1h in test_loader:
            batch_X = batch_X.to(DEVICE)

            logits_1d, logits_4h, logits_1h = model(batch_X)

            pred_1d = (torch.sigmoid(logits_1d) > 0.5).float()
            pred_4h = (torch.sigmoid(logits_4h) > 0.5).float()
            pred_1h = (torch.sigmoid(logits_1h) > 0.5).float()

            all_pred_1d.extend(pred_1d.cpu().numpy().flatten())
            all_pred_4h.extend(pred_4h.cpu().numpy().flatten())
            all_pred_1h.extend(pred_1h.cpu().numpy().flatten())

            all_true_1d.extend(batch_y_1d.cpu().numpy())
            all_true_4h.extend(batch_y_4h.cpu().numpy())
            all_true_1h.extend(batch_y_1h.cpu().numpy())

    all_pred_1d = np.array(all_pred_1d)
    all_pred_4h = np.array(all_pred_4h)
    all_pred_1h = np.array(all_pred_1h)

    all_true_1d = np.array(all_true_1d)
    all_true_4h = np.array(all_true_4h)
    all_true_1h = np.array(all_true_1h)

    acc_1d = np.mean(all_pred_1d == all_true_1d)
    acc_4h = np.mean(all_pred_4h == all_true_4h)
    acc_1h = np.mean(all_pred_1h == all_true_1h)

    print("\n" + "="*70)
    print("MULTI-TASK TEST SET PERFORMANCE (HOLDOUT)")
    print("="*70)
    print(f"1-Day Prediction Accuracy:  {acc_1d:.3%}")
    print(f"4-Hour Prediction Accuracy: {acc_4h:.3%}")
    print(f"1-Hour Prediction Accuracy: {acc_1h:.3%}")
    print("="*70)

    return acc_1d, acc_4h, acc_1h


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("HYBRID LSTM + TCN MULTI-TASK TRAINING V2")
    print("WITH MICROSTRUCTURE FEATURES FROM 1-SECOND CANDLES")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Sequence length: {DAILY_SEQ_LEN} days")
    print(f"Tasks: 1d, 4h, 1h direction prediction")
    print(f"Loss weights: 1d={LOSS_WEIGHTS['1d']}, 4h={LOSS_WEIGHTS['4h']}, 1h={LOSS_WEIGHTS['1h']}")
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

    # Load microstructure features from 1-second candles
    daily_vp_1s, cvd_5min, cvd_ema_5min = load_microstructure_features(data_dir)

    # Calculate CVD and VP from hourly/daily data (for comparison)
    print("\nComputing traditional features...")
    cvd_1h, cvd_ema_1h = calculate_cvd_hourly(bars_1h)
    daily_vp = calculate_daily_volume_profile(bars_daily, lookback=30)

    # Create multi-task training samples
    print("\nCreating multi-task samples...")

    closes_daily = [b['c'] for b in bars_daily]
    closes_4h = [b['c'] for b in bars_4h]
    closes_1h = [b['c'] for b in bars_1h]

    X_all = []
    y_all_1d = []
    y_all_4h = []
    y_all_1h = []

    for i in range(DAILY_SEQ_LEN, len(bars_daily) - 1):
        seq_features = extract_longterm_features(
            bars_1h[:i*24], bars_4h[:i*6], bars_daily[:i],
            cvd_1h[:i*24], cvd_ema_1h[:i*24], daily_vp[:i],
            daily_vp_1s, cvd_5min, cvd_ema_5min,  # NEW: Microstructure features
            num_days=DAILY_SEQ_LEN
        )

        if seq_features is not None:
            X_all.append(seq_features)

            # Label 1d: next day up/down
            current_price_1d = closes_daily[i]
            future_price_1d = closes_daily[i + 1]
            label_1d = 1.0 if future_price_1d > current_price_1d else 0.0
            y_all_1d.append(label_1d)

            # Label 4h: next 4h bar up/down
            h4_idx = min(i * 6, len(closes_4h) - 1)
            next_h4_idx = min(h4_idx + 1, len(closes_4h) - 1)
            current_price_4h = closes_4h[h4_idx]
            future_price_4h = closes_4h[next_h4_idx]
            label_4h = 1.0 if future_price_4h > current_price_4h else 0.0
            y_all_4h.append(label_4h)

            # Label 1h: next 1h bar up/down
            h1_idx = min(i * 24, len(closes_1h) - 1)
            next_h1_idx = min(h1_idx + 1, len(closes_1h) - 1)
            current_price_1h = closes_1h[h1_idx]
            future_price_1h = closes_1h[next_h1_idx]
            label_1h = 1.0 if future_price_1h > current_price_1h else 0.0
            y_all_1h.append(label_1h)

    if len(X_all) == 0:
        print("Error: No training samples generated")
        return

    X_all = np.array(X_all)
    y_all_1d = np.array(y_all_1d)
    y_all_4h = np.array(y_all_4h)
    y_all_1h = np.array(y_all_1h)

    print(f"  Total samples: {len(X_all)}")
    print(f"  Feature shape: {X_all.shape}")
    print(f"  1d positive ratio: {y_all_1d.mean():.3%}")
    print(f"  4h positive ratio: {y_all_4h.mean():.3%}")
    print(f"  1h positive ratio: {y_all_1h.mean():.3%}")

    # Time-series split
    n_samples = len(X_all)
    train_end = int(n_samples * TRAIN_RATIO)
    val_end = int(n_samples * (TRAIN_RATIO + VAL_RATIO))

    X_train = X_all[:train_end]
    y_train_1d = y_all_1d[:train_end]
    y_train_4h = y_all_4h[:train_end]
    y_train_1h = y_all_1h[:train_end]

    X_val = X_all[train_end:val_end]
    y_val_1d = y_all_1d[train_end:val_end]
    y_val_4h = y_all_4h[train_end:val_end]
    y_val_1h = y_all_1h[train_end:val_end]

    X_test = X_all[val_end:]
    y_test_1d = y_all_1d[val_end:]
    y_test_4h = y_all_4h[val_end:]
    y_test_1h = y_all_1h[val_end:]

    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples (holdout)")

    # Normalize (FIT ONLY ON TRAINING DATA)
    print("\nNormalizing features...")
    scaler = StandardScaler()

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    print("  ✓ Scaler fit on training data only (no leakage)")

    # Create datasets
    train_dataset = MultiTaskDataset(X_train_scaled, y_train_1d, y_train_4h, y_train_1h)
    val_dataset = MultiTaskDataset(X_val_scaled, y_val_1d, y_val_4h, y_val_1h)
    test_dataset = MultiTaskDataset(X_test_scaled, y_test_1d, y_test_4h, y_test_1h)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    print("\nInitializing Hybrid LSTM + TCN model...")
    model = HybridLSTMTCN(
        input_size=X_train.shape[2],
        lstm_hidden=HIDDEN_DIM,
        lstm_layers=NUM_LAYERS,
        tcn_channels=TCN_CHANNELS,
        kernel_size=KERNEL_SIZE,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT
    )

    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print(f"\nTraining for {EPOCHS} epochs...")
    history, best_val_loss = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, patience=PATIENCE
    )

    # Evaluate on test set
    acc_1d, acc_4h, acc_1h = evaluate_model(model, test_loader)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'history': history,
        'config': {
            'input_size': X_train.shape[2],
            'lstm_hidden': HIDDEN_DIM,
            'lstm_layers': NUM_LAYERS,
            'tcn_channels': TCN_CHANNELS,
            'kernel_size': KERNEL_SIZE,
            'embedding_dim': EMBEDDING_DIM,
            'dropout': DROPOUT,
            'daily_seq_len': DAILY_SEQ_LEN,
        },
        'test_metrics': {
            'acc_1d': acc_1d,
            'acc_4h': acc_4h,
            'acc_1h': acc_1h
        }
    }, os.path.join(output_dir, 'htf_lstm_tcn_multitask_v2_microstructure.pt'))

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Test Accuracy - 1d: {acc_1d:.3%}, 4h: {acc_4h:.3%}, 1h: {acc_1h:.3%}")
    print(f"\nModel saved to: {output_dir}/htf_lstm_tcn_multitask_v2_microstructure.pt")


if __name__ == "__main__":
    main()
