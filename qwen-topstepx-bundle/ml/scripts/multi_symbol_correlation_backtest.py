#!/usr/bin/env python3
"""
Multi-Symbol Correlation LSTM Backtest - EQUAL TREATMENT VERSION

Each symbol gets its own full pipeline:
1. Stage 1 XGBoost - Direction prediction (per symbol)
2. Long-term LSTM - 5-min context (per symbol)
3. Short-term LSTM - 1-second microstructure (per symbol)
4. Regime LSTM - 1-min bars with indicators (per symbol)

SHARED: Cross-Symbol Correlation LSTM
- Takes 5min and 1min OHLC, Volume Profile, CVD from ALL symbols
- Learns correlations, lead/lag relationships, divergences
- Learns conditional patterns like "gold inversely correlates with ES when X"
- Outputs 48-dim embedding shared across all symbol decisions

Final (per symbol): XGBoost combines:
- That symbol's own features + embeddings
- Shared correlation embedding
- Makes independent TAKE/HOLD decision for each symbol

This allows trading all symbols simultaneously with shared correlation intelligence.
"""

import json
import sys
import os
import argparse
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

HOLD_BARS = 20  # Trade timeout in 1-min bars

# LSTM configuration
LONGTERM_SEQ_LEN = 72   # 6 hours of 5-min bars
SHORTTERM_SEQ_LEN = 120 # 2 minutes of 1-second bars
REGIME_SEQ_LEN = 60     # 1 hour of 1-min bars
CORRELATION_SEQ_LEN = 30  # 30 bars for cross-symbol correlation

EMBEDDING_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 2

# Cross-symbol correlation LSTM
CORRELATION_EMBEDDING_DIM = 48  # Larger embedding for multi-symbol data

MIN_CVD_DIVERGENCE = 0.15
MIN_BARS_BETWEEN_TRADES = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Symbol configuration
SYMBOLS = ['MNQ', 'MES', 'MGC']
POINT_VALUES = {'MNQ': 2, 'MES': 5, 'MGC': 10, 'NQ': 20}


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

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


def calculate_bollinger_bands(bars: List[Dict], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """Calculate Bollinger Bands."""
    if len(bars) < period:
        return [0.0] * len(bars), [0.0] * len(bars), [0.0] * len(bars)

    closes = [b['c'] for b in bars]
    middle = [0.0] * len(bars)
    upper = [0.0] * len(bars)
    lower = [0.0] * len(bars)

    for i in range(period - 1, len(bars)):
        window = closes[i - period + 1:i + 1]
        sma = sum(window) / period
        std = np.std(window)

        middle[i] = sma
        upper[i] = sma + std_dev * std
        lower[i] = sma - std_dev * std

    return middle, upper, lower


def calculate_adx(bars: List[Dict], period: int = 14) -> List[float]:
    """Calculate Average Directional Index."""
    if len(bars) < period * 2:
        return [0.0] * len(bars)

    adx_values = [0.0] * len(bars)
    plus_dm = []
    minus_dm = []
    tr_values = []

    for i in range(1, len(bars)):
        high = bars[i]['h']
        low = bars[i]['l']
        prev_high = bars[i-1]['h']
        prev_low = bars[i-1]['l']
        prev_close = bars[i-1]['c']

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)

    if len(tr_values) < period:
        return adx_values

    smooth_tr = sum(tr_values[:period])
    smooth_plus_dm = sum(plus_dm[:period])
    smooth_minus_dm = sum(minus_dm[:period])

    dx_values = []

    for i in range(period, len(tr_values)):
        smooth_tr = smooth_tr - (smooth_tr / period) + tr_values[i]
        smooth_plus_dm = smooth_plus_dm - (smooth_plus_dm / period) + plus_dm[i]
        smooth_minus_dm = smooth_minus_dm - (smooth_minus_dm / period) + minus_dm[i]

        if smooth_tr > 0:
            plus_di = 100 * smooth_plus_dm / smooth_tr
            minus_di = 100 * smooth_minus_dm / smooth_tr
        else:
            plus_di = minus_di = 0

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if plus_di + minus_di > 0 else 0
        dx_values.append(dx)

        bar_idx = i + 1
        if len(dx_values) >= period:
            if len(dx_values) == period:
                adx = sum(dx_values) / period
            else:
                adx = (adx_values[bar_idx - 1] * (period - 1) + dx) / period
            adx_values[bar_idx] = adx

    return adx_values


def calculate_rsi(bars: List[Dict], period: int = 14) -> List[float]:
    """Calculate RSI."""
    if len(bars) < period + 1:
        return [50.0] * len(bars)

    rsi_values = [50.0] * len(bars)
    gains = []
    losses = []

    for i in range(1, len(bars)):
        change = bars[i]['c'] - bars[i-1]['c']
        gains.append(max(0, change))
        losses.append(max(0, -change))

        if i >= period:
            if i == period:
                avg_gain = sum(gains[:period]) / period
                avg_loss = sum(losses[:period]) / period
            else:
                avg_gain = (avg_gain * (period - 1) + gains[-1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[-1]) / period

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))
            else:
                rsi_values[i] = 100 if avg_gain > 0 else 50

    return rsi_values


def calculate_choppiness_index(bars: List[Dict], period: int = 14) -> List[float]:
    """Calculate Choppiness Index."""
    if len(bars) < period + 1:
        return [50.0] * len(bars)

    chop_values = [50.0] * len(bars)

    for i in range(period, len(bars)):
        tr_sum = 0
        highest_high = bars[i]['h']
        lowest_low = bars[i]['l']

        for j in range(i - period + 1, i + 1):
            high = bars[j]['h']
            low = bars[j]['l']
            prev_close = bars[j-1]['c'] if j > 0 else bars[j]['o']

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_sum += tr

            highest_high = max(highest_high, high)
            lowest_low = min(lowest_low, low)

        hl_range = highest_high - lowest_low
        if hl_range > 0 and tr_sum > 0:
            chop = 100 * np.log10(tr_sum / hl_range) / np.log10(period)
            chop_values[i] = min(100, max(0, chop))

    return chop_values


# ============================================================================
# DATA PROCESSING
# ============================================================================

def aggregate_bars(bars_1s: List[Dict], period_minutes: int) -> List[Dict]:
    """Aggregate 1-second bars to higher timeframe."""
    if not bars_1s:
        return []

    period_seconds = period_minutes * 60
    period_data = defaultdict(lambda: {'bars': []})
    period_order = []

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            ts = ts.replace('Z', '+00:00')
            dt = datetime.fromisoformat(ts.replace('+00:00', ''))
        else:
            dt = ts

        total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        period_start = (total_seconds // period_seconds) * period_seconds
        period_ts = dt.replace(hour=period_start // 3600, minute=(period_start % 3600) // 60, second=0, microsecond=0)
        period_key = period_ts.isoformat()

        if period_key not in period_data:
            period_order.append(period_key)

        period_data[period_key]['bars'].append(bar)

    result = []
    for period_ts in period_order:
        bars = period_data[period_ts]['bars']
        if not bars:
            continue

        result.append({
            't': period_ts,
            'o': bars[0]['o'],
            'h': max(b['h'] for b in bars),
            'l': min(b['l'] for b in bars),
            'c': bars[-1]['c'],
            'v': sum(b.get('v', 0) or 0 for b in bars),
        })

    return result


def calculate_cvd_1min(bars_1s: List[Dict]) -> Tuple[List[Dict], List[float], List[float]]:
    """Calculate 1-min bars with CVD."""
    bars_1min = aggregate_bars(bars_1s, 1)

    cvd = 0
    cvd_values = []
    cvd_ema_values = []
    ema_period = 14
    ema_mult = 2 / (ema_period + 1)

    for bar in bars_1min:
        volume = bar.get('v', 0) or 0
        bar_range = bar['h'] - bar['l']

        if bar_range > 0:
            close_position = (bar['c'] - bar['l']) / bar_range
            buy_vol = volume * close_position
            sell_vol = volume * (1 - close_position)
        else:
            if bar['c'] >= bar['o']:
                buy_vol, sell_vol = volume, 0
            else:
                buy_vol, sell_vol = 0, volume

        delta = buy_vol - sell_vol
        cvd += delta
        cvd_values.append(cvd)

        if len(cvd_ema_values) == 0:
            cvd_ema_values.append(cvd)
        else:
            ema = cvd * ema_mult + cvd_ema_values[-1] * (1 - ema_mult)
            cvd_ema_values.append(ema)

    return bars_1min, cvd_values, cvd_ema_values


def calculate_volume_profile(bars: List[Dict], lookback: int = 30, tick_size: float = 0.25) -> Dict[str, float]:
    """Calculate Volume Profile."""
    if len(bars) < lookback:
        lookback = len(bars)

    recent = bars[-lookback:]
    price_volume = defaultdict(float)

    for bar in recent:
        prices = [bar['o'], bar['h'], bar['l'], bar['c']]
        volume = bar.get('v', 0) or 0
        vol_per_price = volume / 4.0 if volume > 0 else 0.25

        for price in prices:
            rounded = round(price / tick_size) * tick_size
            price_volume[rounded] += vol_per_price

    if not price_volume:
        return {"poc": 0, "vah": 0, "val": 0}

    sorted_prices = sorted(price_volume.keys())
    volumes = [price_volume[p] for p in sorted_prices]
    total_volume = sum(volumes)

    poc_idx = np.argmax(volumes)
    poc = sorted_prices[poc_idx]

    target_volume = total_volume * 0.7
    current_volume = volumes[poc_idx]
    lower_idx, upper_idx = poc_idx, poc_idx

    while current_volume < target_volume:
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

    return {
        "poc": poc,
        "vah": sorted_prices[upper_idx],
        "val": sorted_prices[lower_idx],
    }


# ============================================================================
# LSTM MODELS
# ============================================================================

class LongTermLSTM(nn.Module):
    """Long-term LSTM for 6-hour context."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, embedding_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        embedding = self.fc(last_hidden)
        return self.layer_norm(embedding)


class ShortTermLSTM(nn.Module):
    """Short-term LSTM for microstructure."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, embedding_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        embedding = self.fc(last_hidden)
        return self.layer_norm(embedding)


class RegimeLSTM(nn.Module):
    """Regime LSTM for market regime detection."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, embedding_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        embedding = self.fc(last_hidden)
        return self.layer_norm(embedding)


class CrossSymbolCorrelationLSTM(nn.Module):
    """
    NEW: Cross-Symbol Correlation LSTM

    Takes data from multiple symbols (MNQ, MES, MGC) and learns:
    - Cross-symbol correlations
    - Lead/lag relationships
    - Divergences between correlated instruments
    - Market-wide momentum/sentiment

    Input per timestep (per symbol):
    - 5min OHLC normalized (4 features)
    - 1min OHLC normalized (4 features)
    - Volume Profile position (POC, VAH, VAL relative to price) (3 features)
    - CVD and CVD momentum (2 features)
    - Price momentum (1 feature)
    Total: 14 features per symbol x 3 symbols = 42 features per timestep
    """
    def __init__(self, num_symbols: int = 3, features_per_symbol: int = 14,
                 hidden_dim: int = 96, num_layers: int = 2, embedding_dim: int = 48):
        super().__init__()

        input_dim = num_symbols * features_per_symbol

        # Main LSTM for temporal patterns
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)

        # Attention layer to focus on important cross-symbol relationships
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Output projection
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, seq_len, num_symbols * features_per_symbol)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim)

        # Self-attention over time steps
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Combine last LSTM hidden state with attention-weighted representation
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        attn_pooled = attn_out.mean(dim=1)  # (batch, hidden_dim)

        combined = torch.cat([last_hidden, attn_pooled], dim=-1)  # (batch, hidden_dim * 2)

        # Project to embedding
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        embedding = self.fc2(x)

        return self.layer_norm(embedding)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_cross_symbol_features(
    all_symbol_data: Dict[str, Dict],  # symbol -> {bars_1min, bars_5min, cvd_1min, cvd_ema_1min, vp}
    bar_idx: int,
    seq_len: int = 30,
) -> Optional[np.ndarray]:
    """
    Extract cross-symbol features for correlation LSTM.

    For each timestep, extracts features from all symbols:
    - 5min OHLC (normalized)
    - 1min OHLC (normalized)
    - VP position (POC, VAH, VAL relative to price)
    - CVD and CVD momentum
    - Price momentum
    """
    symbols = list(all_symbol_data.keys())
    if len(symbols) < 1:
        return None

    # Check if all symbols have enough data
    for sym in symbols:
        data = all_symbol_data[sym]
        if bar_idx < seq_len or bar_idx >= len(data['bars_1min']):
            return None

    features_per_timestep = []

    for t in range(bar_idx - seq_len, bar_idx):
        timestep_features = []

        for sym in symbols:
            data = all_symbol_data[sym]
            bars_1min = data['bars_1min']
            bars_5min = data['bars_5min']
            cvd_1min = data['cvd_1min']
            cvd_ema_1min = data['cvd_ema_1min']

            if t >= len(bars_1min):
                continue

            bar_1min = bars_1min[t]
            close = bar_1min['c']
            if close == 0:
                close = 1

            # 1min OHLC normalized
            o_1min = (bar_1min['o'] - close) / close
            h_1min = (bar_1min['h'] - close) / close
            l_1min = (bar_1min['l'] - close) / close
            range_1min = (bar_1min['h'] - bar_1min['l']) / close

            # 5min OHLC (use corresponding 5min bar)
            idx_5min = t // 5
            if idx_5min < len(bars_5min):
                bar_5min = bars_5min[idx_5min]
                o_5min = (bar_5min['o'] - close) / close
                h_5min = (bar_5min['h'] - close) / close
                l_5min = (bar_5min['l'] - close) / close
                range_5min = (bar_5min['h'] - bar_5min['l']) / close
            else:
                o_5min = h_5min = l_5min = range_5min = 0

            # Volume Profile position
            vp = calculate_volume_profile(bars_1min[:t+1], lookback=30)
            poc_dist = (close - vp['poc']) / close if vp['poc'] > 0 else 0
            vah_dist = (close - vp['vah']) / close if vp['vah'] > 0 else 0
            val_dist = (close - vp['val']) / close if vp['val'] > 0 else 0

            # CVD features
            cvd_now = cvd_1min[t] if t < len(cvd_1min) else 0
            cvd_ema_now = cvd_ema_1min[t] if t < len(cvd_ema_1min) else 0
            cvd_momentum = (cvd_now - cvd_ema_now) / max(abs(cvd_ema_now), 1)

            cvd_5_ago = cvd_1min[t - 5] if t >= 5 and t - 5 < len(cvd_1min) else cvd_now
            cvd_change = (cvd_now - cvd_5_ago) / max(abs(cvd_5_ago), 1) if cvd_5_ago != 0 else 0

            # Price momentum
            price_5_ago = bars_1min[t - 5]['c'] if t >= 5 else close
            price_momentum = (close - price_5_ago) / price_5_ago if price_5_ago > 0 else 0

            # Combine features for this symbol (14 features)
            symbol_features = [
                o_5min, h_5min, l_5min, range_5min,  # 5min OHLC (4)
                o_1min, h_1min, l_1min, range_1min,  # 1min OHLC (4)
                poc_dist, vah_dist, val_dist,         # VP position (3)
                cvd_momentum, cvd_change,             # CVD (2)
                price_momentum,                        # Momentum (1)
            ]

            timestep_features.extend(symbol_features)

        # Pad if fewer symbols
        while len(timestep_features) < 3 * 14:  # 3 symbols * 14 features
            timestep_features.extend([0] * 14)

        features_per_timestep.append(timestep_features)

    return np.array(features_per_timestep, dtype=np.float32)


def extract_longterm_sequence(
    bars_5min: List[Dict],
    cvd_1min: List[float],
    bar_idx_1min: int,
    seq_len: int = 72,
) -> Optional[np.ndarray]:
    """Extract long-term sequence for LSTM."""
    bar_idx_5min = bar_idx_1min // 5
    if bar_idx_5min < seq_len:
        return None

    features = []
    for i in range(bar_idx_5min - seq_len, bar_idx_5min):
        bar = bars_5min[i]
        close = bar['c'] if bar['c'] != 0 else 1

        o_norm = (bar['o'] - close) / close
        h_norm = (bar['h'] - close) / close
        l_norm = (bar['l'] - close) / close
        range_norm = (bar['h'] - bar['l']) / close
        body_norm = (bar['c'] - bar['o']) / close
        vol_log = np.log1p(bar.get('v', 0) or 0) / 10

        cvd_start_idx = i * 5
        cvd_end_idx = min((i + 1) * 5, len(cvd_1min))
        if cvd_end_idx > cvd_start_idx and cvd_end_idx <= len(cvd_1min):
            cvd_change = (cvd_1min[cvd_end_idx - 1] - cvd_1min[cvd_start_idx]) / max(abs(cvd_1min[cvd_end_idx - 1]), 1)
        else:
            cvd_change = 0

        features.append([o_norm, h_norm, l_norm, range_norm, body_norm, vol_log, cvd_change])

    return np.array(features, dtype=np.float32)


def extract_shortterm_sequence(
    bars_1s: List[Dict],
    bar_idx_1s: int,
    seq_len: int = 120,
) -> Optional[np.ndarray]:
    """Extract short-term sequence for LSTM."""
    if bar_idx_1s < seq_len:
        return None

    features = []
    for i in range(bar_idx_1s - seq_len, bar_idx_1s):
        bar = bars_1s[i]
        close = bar['c'] if bar['c'] != 0 else 1

        o_norm = (bar['o'] - close) / close
        h_norm = (bar['h'] - close) / close
        l_norm = (bar['l'] - close) / close
        range_norm = (bar['h'] - bar['l']) / close
        body_norm = (bar['c'] - bar['o']) / close
        vol_log = np.log1p(bar.get('v', 0) or 0) / 10

        # Microstructure features
        spread = (bar['h'] - bar['l']) / close
        wick_upper = (bar['h'] - max(bar['o'], bar['c'])) / close
        wick_lower = (min(bar['o'], bar['c']) - bar['l']) / close

        features.append([o_norm, h_norm, l_norm, range_norm, body_norm, vol_log, spread, wick_upper, wick_lower])

    return np.array(features, dtype=np.float32)


def extract_regime_sequence(
    bars_1min: List[Dict],
    cvd_1min: List[float],
    cvd_ema_1min: List[float],
    atr: List[float],
    bb_middle: List[float],
    bb_upper: List[float],
    bb_lower: List[float],
    adx: List[float],
    chop: List[float],
    rsi: List[float],
    bar_idx: int,
    vp: Dict[str, float],
    seq_len: int = 60,
) -> Optional[np.ndarray]:
    """Extract regime sequence for LSTM."""
    if bar_idx < seq_len:
        return None

    features = []
    for i in range(bar_idx - seq_len, bar_idx):
        bar = bars_1min[i]
        close = bar['c'] if bar['c'] != 0 else 1

        # OHLCV
        o_norm = (bar['o'] - close) / close
        h_norm = (bar['h'] - close) / close
        l_norm = (bar['l'] - close) / close
        range_norm = (bar['h'] - bar['l']) / close
        body_norm = (bar['c'] - bar['o']) / close
        vol_log = np.log1p(bar.get('v', 0) or 0) / 10

        # CVD
        cvd_now = cvd_1min[i] if i < len(cvd_1min) else 0
        cvd_ema_now = cvd_ema_1min[i] if i < len(cvd_ema_1min) else 0
        cvd_momentum = (cvd_now - cvd_ema_now) / max(abs(cvd_ema_now), 1)

        # Volume Profile
        poc_dist = (close - vp['poc']) / close if vp['poc'] > 0 else 0
        vah_dist = (close - vp['vah']) / close if vp['vah'] > 0 else 0
        val_dist = (close - vp['val']) / close if vp['val'] > 0 else 0

        # Technical indicators
        atr_norm = atr[i] / close if i < len(atr) and close > 0 else 0

        bb_mid = bb_middle[i] if i < len(bb_middle) else close
        bb_up = bb_upper[i] if i < len(bb_upper) else close
        bb_low = bb_lower[i] if i < len(bb_lower) else close
        bb_width = (bb_up - bb_low) / close if close > 0 else 0
        bb_position = (close - bb_low) / (bb_up - bb_low) if bb_up > bb_low else 0.5

        adx_norm = adx[i] / 100.0 if i < len(adx) else 0.25
        chop_norm = (chop[i] - 50) / 50.0 if i < len(chop) else 0
        rsi_norm = (rsi[i] - 50) / 50.0 if i < len(rsi) else 0

        features.append([
            o_norm, h_norm, l_norm, range_norm, body_norm, vol_log,
            cvd_momentum, poc_dist, vah_dist, val_dist,
            atr_norm, bb_width, bb_position, adx_norm, chop_norm, rsi_norm,
            0, 0  # Padding to 18 features
        ])

    return np.array(features, dtype=np.float32)


def extract_xgb_features(
    bars_1min: List[Dict],
    bars_5min: List[Dict],
    bars_15min: List[Dict],
    bars_1h: List[Dict],
    cvd_1min: List[float],
    cvd_ema_1min: List[float],
    bar_idx: int,
    vp: Dict[str, float],
    trigger_sentiment: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Extract XGBoost features."""
    if bar_idx < 20:
        return None

    bar = bars_1min[bar_idx]
    close = bar['c']
    if close == 0:
        return None

    # Price features
    returns_1 = (close - bars_1min[bar_idx - 1]['c']) / bars_1min[bar_idx - 1]['c'] if bars_1min[bar_idx - 1]['c'] > 0 else 0
    returns_5 = (close - bars_1min[bar_idx - 5]['c']) / bars_1min[bar_idx - 5]['c'] if bar_idx >= 5 and bars_1min[bar_idx - 5]['c'] > 0 else 0
    returns_10 = (close - bars_1min[bar_idx - 10]['c']) / bars_1min[bar_idx - 10]['c'] if bar_idx >= 10 and bars_1min[bar_idx - 10]['c'] > 0 else 0

    # Range features
    range_1 = (bar['h'] - bar['l']) / close

    high_10 = max(b['h'] for b in bars_1min[max(0, bar_idx - 10):bar_idx + 1])
    low_10 = min(b['l'] for b in bars_1min[max(0, bar_idx - 10):bar_idx + 1])
    range_10 = (high_10 - low_10) / close

    # Position in range
    pos_in_range = (close - low_10) / (high_10 - low_10) if high_10 > low_10 else 0.5

    # CVD features
    cvd_now = cvd_1min[bar_idx] if bar_idx < len(cvd_1min) else 0
    cvd_ema_now = cvd_ema_1min[bar_idx] if bar_idx < len(cvd_ema_1min) else 0
    cvd_momentum = (cvd_now - cvd_ema_now) / max(abs(cvd_ema_now), 1)

    cvd_5_ago = cvd_1min[bar_idx - 5] if bar_idx >= 5 else cvd_now
    cvd_change_5 = (cvd_now - cvd_5_ago) / max(abs(cvd_5_ago), 1) if cvd_5_ago != 0 else 0

    # VP features
    poc_dist = (close - vp['poc']) / close if vp['poc'] > 0 else 0
    vah_dist = (close - vp['vah']) / close if vp['vah'] > 0 else 0
    val_dist = (close - vp['val']) / close if vp['val'] > 0 else 0

    # Trigger sentiment
    trigger_bullish = 1 if trigger_sentiment == 'bullish' else 0
    trigger_bearish = 1 if trigger_sentiment == 'bearish' else 0

    return {
        'returns_1': returns_1,
        'returns_5': returns_5,
        'returns_10': returns_10,
        'range_1': range_1,
        'range_10': range_10,
        'pos_in_range': pos_in_range,
        'cvd_momentum': cvd_momentum,
        'cvd_change_5': cvd_change_5,
        'poc_dist': poc_dist,
        'vah_dist': vah_dist,
        'val_dist': val_dist,
        'trigger_bullish': trigger_bullish,
        'trigger_bearish': trigger_bearish,
    }


# ============================================================================
# TRADE TRIGGER
# ============================================================================

def detect_trade_trigger(
    bars_1min: List[Dict],
    cvd_1min: List[float],
    cvd_ema_1min: List[float],
    vp: Dict[str, float],
    bar_idx: int,
) -> Optional[str]:
    """Detect trade trigger based on CVD divergence and VP levels."""
    if bar_idx < 10:
        return None

    current_bar = bars_1min[bar_idx]
    current_price = current_bar['c']

    cvd_now = cvd_1min[bar_idx] if bar_idx < len(cvd_1min) else 0
    cvd_ema_now = cvd_ema_1min[bar_idx] if bar_idx < len(cvd_ema_1min) else 0

    price_5_ago = bars_1min[bar_idx - 5]['c']
    price_change = (current_price - price_5_ago) / price_5_ago if price_5_ago > 0 else 0

    cvd_5_ago = cvd_1min[bar_idx - 5] if bar_idx >= 5 else 0
    cvd_change = cvd_now - cvd_5_ago
    cvd_range = max(abs(cvd_1min[bar_idx - 10]), 1) if bar_idx >= 10 else 1
    cvd_change_norm = cvd_change / cvd_range

    cvd_vs_ema = cvd_now - cvd_ema_now
    cvd_ema_divergence = cvd_vs_ema / max(abs(cvd_ema_now), 1)

    price_vs_poc = (current_price - vp['poc']) / vp['poc'] if vp['poc'] > 0 else 0
    price_vs_val = (current_price - vp['val']) / vp['val'] if vp['val'] > 0 else 0
    price_vs_vah = (current_price - vp['vah']) / vp['vah'] if vp['vah'] > 0 else 0

    # Bullish conditions
    bullish_divergence = price_change < -0.001 and cvd_change_norm > MIN_CVD_DIVERGENCE
    bullish_cvd_strength = cvd_ema_divergence > 0.2 and price_vs_poc < 0.001
    bullish_vp_support = price_vs_val < 0.002 and cvd_change_norm > 0

    if bullish_divergence or bullish_cvd_strength or bullish_vp_support:
        return 'bullish'

    # Bearish conditions
    bearish_divergence = price_change > 0.001 and cvd_change_norm < -MIN_CVD_DIVERGENCE
    bearish_cvd_strength = cvd_ema_divergence < -0.2 and price_vs_poc > -0.001
    bearish_vp_resistance = price_vs_vah > -0.002 and cvd_change_norm < 0

    if bearish_divergence or bearish_cvd_strength or bearish_vp_resistance:
        return 'bearish'

    return None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_symbol_data(bars_path: str) -> Tuple[str, Dict[str, List[Dict]]]:
    """Load bar data from JSON file and organize by date."""
    print(f"Loading {bars_path}...", file=sys.stderr)

    with open(bars_path, 'r') as f:
        data = json.load(f)

    symbol = data.get('symbol', 'UNKNOWN')
    bars = data.get('bars', [])

    print(f"  Symbol: {symbol}, Total bars: {len(bars)}", file=sys.stderr)

    # Organize by date
    bars_by_date = defaultdict(list)
    for bar in bars:
        ts = bar['t']
        if isinstance(ts, str):
            date_str = ts[:10]
        else:
            date_str = ts.strftime('%Y-%m-%d')
        bars_by_date[date_str].append(bar)

    print(f"  Days of data: {len(bars_by_date)}", file=sys.stderr)

    return symbol, dict(bars_by_date)


# ============================================================================
# TRAINING
# ============================================================================

def train_lstms(
    all_bars_1s: Dict[str, List[Dict]],
    train_dates: List[str],
    epochs: int = 30,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Train the three LSTMs: long-term, short-term, regime."""
    print("Training LSTMs...", file=sys.stderr)

    longterm_seqs, longterm_labels = [], []
    shortterm_seqs, shortterm_labels = [], []
    regime_seqs, regime_labels = [], []

    for date in train_dates:
        if date not in all_bars_1s:
            continue

        bars_1s = all_bars_1s[date]
        bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
        bars_5min = aggregate_bars(bars_1s, 5)

        if len(bars_1min) < 100:
            continue

        # Calculate indicators
        atr_1min = calculate_atr(bars_1min, period=14)
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_1min, period=20)
        adx_1min = calculate_adx(bars_1min, period=14)
        chop_1min = calculate_choppiness_index(bars_1min, period=14)
        rsi_1min = calculate_rsi(bars_1min, period=14)

        # Long-term sequences
        for i in range(72, len(bars_5min) - 5, 10):
            seq = extract_longterm_sequence(bars_5min, cvd_1min, i * 5, seq_len=72)
            if seq is not None:
                future_price = bars_5min[min(i + 5, len(bars_5min) - 1)]['c']
                current_price = bars_5min[i]['c']
                label = 1 if future_price > current_price else 0
                longterm_seqs.append(seq)
                longterm_labels.append(label)

        # Short-term sequences
        for i in range(120, len(bars_1s) - 60, 60):
            seq = extract_shortterm_sequence(bars_1s, i, seq_len=120)
            if seq is not None:
                future_price = bars_1s[min(i + 60, len(bars_1s) - 1)]['c']
                current_price = bars_1s[i]['c']
                label = 1 if future_price > current_price else 0
                shortterm_seqs.append(seq)
                shortterm_labels.append(label)

        # Regime sequences
        for i in range(REGIME_SEQ_LEN, len(bars_1min) - HOLD_BARS, 5):
            vp = calculate_volume_profile(bars_1min[:i+1], lookback=30)
            seq = extract_regime_sequence(
                bars_1min, cvd_1min, cvd_ema_1min,
                atr_1min, bb_middle, bb_upper, bb_lower,
                adx_1min, chop_1min, rsi_1min,
                i, vp, seq_len=REGIME_SEQ_LEN
            )
            if seq is not None:
                future_idx = min(i + HOLD_BARS, len(bars_1min) - 1)
                current_price = bars_1min[i]['c']
                future_high = max(b['h'] for b in bars_1min[i:future_idx+1])
                future_low = min(b['l'] for b in bars_1min[i:future_idx+1])
                price_range = (future_high - future_low) / current_price
                label = 1 if price_range > 0.002 else 0
                regime_seqs.append(seq)
                regime_labels.append(label)

    # Initialize models
    longterm_lstm = LongTermLSTM(input_dim=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    shortterm_lstm = ShortTermLSTM(input_dim=9, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    regime_lstm = RegimeLSTM(input_dim=18, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)

    # Train long-term LSTM
    if len(longterm_seqs) > 100:
        X_lt = torch.tensor(np.array(longterm_seqs), dtype=torch.float32).to(DEVICE)
        y_lt = torch.tensor(np.array(longterm_labels), dtype=torch.long).to(DEVICE)

        classifier = nn.Linear(EMBEDDING_DIM, 2).to(DEVICE)
        optimizer = torch.optim.Adam(list(longterm_lstm.parameters()) + list(classifier.parameters()), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        longterm_lstm.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = longterm_lstm(X_lt)
            logits = classifier(embeddings)
            loss = criterion(logits, y_lt)
            loss.backward()
            optimizer.step()

    # Train short-term LSTM
    if len(shortterm_seqs) > 100:
        X_st = torch.tensor(np.array(shortterm_seqs), dtype=torch.float32).to(DEVICE)
        y_st = torch.tensor(np.array(shortterm_labels), dtype=torch.long).to(DEVICE)

        classifier = nn.Linear(EMBEDDING_DIM, 2).to(DEVICE)
        optimizer = torch.optim.Adam(list(shortterm_lstm.parameters()) + list(classifier.parameters()), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        shortterm_lstm.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = shortterm_lstm(X_st)
            logits = classifier(embeddings)
            loss = criterion(logits, y_st)
            loss.backward()
            optimizer.step()

    # Train regime LSTM
    if len(regime_seqs) > 100:
        X_reg = torch.tensor(np.array(regime_seqs), dtype=torch.float32).to(DEVICE)
        y_reg = torch.tensor(np.array(regime_labels), dtype=torch.long).to(DEVICE)

        classifier = nn.Linear(EMBEDDING_DIM, 2).to(DEVICE)
        optimizer = torch.optim.Adam(list(regime_lstm.parameters()) + list(classifier.parameters()), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        regime_lstm.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = regime_lstm(X_reg)
            logits = classifier(embeddings)
            loss = criterion(logits, y_reg)
            loss.backward()
            optimizer.step()

    return longterm_lstm, shortterm_lstm, regime_lstm


def train_cross_symbol_lstm(
    all_symbol_bars: Dict[str, Dict[str, List[Dict]]],  # symbol -> date -> bars
    train_dates: List[str],
    epochs: int = 30,
) -> nn.Module:
    """
    Train Cross-Symbol Correlation LSTM.

    Learns correlations and patterns across multiple symbols.
    """
    print("Training Cross-Symbol Correlation LSTM...", file=sys.stderr)

    correlation_seqs = []
    correlation_labels = []

    symbols = list(all_symbol_bars.keys())

    for date in train_dates:
        # Check if all symbols have data for this date
        all_have_data = all(date in all_symbol_bars[sym] for sym in symbols)
        if not all_have_data:
            continue

        # Process each symbol's data
        symbol_data = {}
        min_bars = float('inf')

        for sym in symbols:
            bars_1s = all_symbol_bars[sym][date]
            bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
            bars_5min = aggregate_bars(bars_1s, 5)

            symbol_data[sym] = {
                'bars_1s': bars_1s,
                'bars_1min': bars_1min,
                'bars_5min': bars_5min,
                'cvd_1min': cvd_1min,
                'cvd_ema_1min': cvd_ema_1min,
            }
            min_bars = min(min_bars, len(bars_1min))

        if min_bars < CORRELATION_SEQ_LEN + HOLD_BARS:
            continue

        # Extract cross-symbol sequences
        for i in range(CORRELATION_SEQ_LEN, min_bars - HOLD_BARS, 5):
            seq = extract_cross_symbol_features(symbol_data, i, seq_len=CORRELATION_SEQ_LEN)

            if seq is not None:
                # Label: Is there a tradeable opportunity across any symbol?
                # Check if any symbol has significant move
                max_opportunity = 0
                for sym in symbols:
                    bars_1min = symbol_data[sym]['bars_1min']
                    current_price = bars_1min[i]['c']
                    future_idx = min(i + HOLD_BARS, len(bars_1min) - 1)
                    future_high = max(b['h'] for b in bars_1min[i:future_idx+1])
                    future_low = min(b['l'] for b in bars_1min[i:future_idx+1])

                    long_potential = (future_high - current_price) / current_price
                    short_potential = (current_price - future_low) / current_price
                    max_opportunity = max(max_opportunity, long_potential, short_potential)

                label = 1 if max_opportunity > 0.002 else 0  # 0.2% opportunity
                correlation_seqs.append(seq)
                correlation_labels.append(label)

    # Initialize model
    correlation_lstm = CrossSymbolCorrelationLSTM(
        num_symbols=len(symbols),
        features_per_symbol=14,
        hidden_dim=96,
        num_layers=2,
        embedding_dim=CORRELATION_EMBEDDING_DIM
    ).to(DEVICE)

    if len(correlation_seqs) > 50:
        print(f"  Training on {len(correlation_seqs)} cross-symbol sequences...", file=sys.stderr)

        X_corr = torch.tensor(np.array(correlation_seqs), dtype=torch.float32).to(DEVICE)
        y_corr = torch.tensor(np.array(correlation_labels), dtype=torch.long).to(DEVICE)

        classifier = nn.Linear(CORRELATION_EMBEDDING_DIM, 2).to(DEVICE)
        optimizer = torch.optim.Adam(list(correlation_lstm.parameters()) + list(classifier.parameters()), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        correlation_lstm.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = correlation_lstm(X_corr)
            logits = classifier(embeddings)
            loss = criterion(logits, y_corr)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}, Loss: {loss.item():.4f}", file=sys.stderr)
    else:
        print(f"  Warning: Only {len(correlation_seqs)} sequences, skipping training", file=sys.stderr)

    return correlation_lstm


def train_stage1_xgboost(
    all_bars_1s: Dict[str, List[Dict]],
    train_dates: List[str],
) -> xgb.XGBClassifier:
    """Train Stage 1 XGBoost for direction prediction."""
    X_all = []
    y_all = []

    for date in train_dates:
        if date not in all_bars_1s:
            continue

        bars_1s = all_bars_1s[date]
        bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
        bars_5min = aggregate_bars(bars_1s, 5)
        bars_15min = aggregate_bars(bars_1s, 15)
        bars_1h = aggregate_bars(bars_1s, 60)

        if len(bars_1min) < 100:
            continue

        for i in range(50, len(bars_1min) - HOLD_BARS, 5):
            vp = calculate_volume_profile(bars_1min[:i+1], lookback=30)
            features = extract_xgb_features(
                bars_1min, bars_5min, bars_15min, bars_1h,
                cvd_1min, cvd_ema_1min, i, vp
            )

            if features is None:
                continue

            current_price = bars_1min[i]['c']
            future_price = bars_1min[min(i + HOLD_BARS, len(bars_1min) - 1)]['c']
            label = 1 if future_price > current_price else 0

            X_all.append(list(features.values()))
            y_all.append(label)

    if len(X_all) < 50:
        return None

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(np.array(X_all), np.array(y_all), verbose=False)
    return model


def train_final_xgboost_with_correlation(
    all_bars_1s: Dict[str, List[Dict]],
    all_symbol_bars: Dict[str, Dict[str, List[Dict]]],
    train_dates: List[str],
    stage1_xgb: xgb.XGBClassifier,
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
    regime_lstm: nn.Module,
    correlation_lstm: nn.Module,
    primary_symbol: str,
) -> xgb.XGBClassifier:
    """
    Train Final XGBoost that combines:
    - Stage 1 XGBoost prediction
    - Long-term LSTM embedding
    - Short-term LSTM embedding
    - Regime LSTM embedding
    - NEW: Cross-Symbol Correlation LSTM embedding
    """
    print("  Training Final XGBoost with correlation embeddings...", file=sys.stderr)

    X_all = []
    y_all = []

    longterm_lstm.eval()
    shortterm_lstm.eval()
    regime_lstm.eval()
    correlation_lstm.eval()

    symbols = list(all_symbol_bars.keys())

    for date in train_dates:
        if date not in all_bars_1s:
            continue

        # Check if all symbols have data
        all_have_data = all(date in all_symbol_bars.get(sym, {}) for sym in symbols)

        bars_1s = all_bars_1s[date]
        bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
        bars_5min = aggregate_bars(bars_1s, 5)
        bars_15min = aggregate_bars(bars_1s, 15)
        bars_1h = aggregate_bars(bars_1s, 60)

        if len(bars_1min) < 100:
            continue

        # Calculate indicators
        atr_1min = calculate_atr(bars_1min, period=14)
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_1min, period=20)
        adx_1min = calculate_adx(bars_1min, period=14)
        chop_1min = calculate_choppiness_index(bars_1min, period=14)
        rsi_1min = calculate_rsi(bars_1min, period=14)

        # Prepare cross-symbol data if available
        symbol_data = {}
        if all_have_data:
            for sym in symbols:
                sym_bars_1s = all_symbol_bars[sym][date]
                sym_bars_1min, sym_cvd_1min, sym_cvd_ema_1min = calculate_cvd_1min(sym_bars_1s)
                sym_bars_5min = aggregate_bars(sym_bars_1s, 5)

                symbol_data[sym] = {
                    'bars_1s': sym_bars_1s,
                    'bars_1min': sym_bars_1min,
                    'bars_5min': sym_bars_5min,
                    'cvd_1min': sym_cvd_1min,
                    'cvd_ema_1min': sym_cvd_ema_1min,
                }

        # Build bar -> 1s index mapping
        bars_1s_ts_to_idx = {}
        for i, bar in enumerate(bars_1s):
            ts = bar['t']
            if isinstance(ts, str):
                ts = ts.replace('+00:00', '')
            bars_1s_ts_to_idx[ts] = i

        for bar_idx in range(max(72, REGIME_SEQ_LEN, CORRELATION_SEQ_LEN), len(bars_1min) - HOLD_BARS, 3):
            vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

            trigger = detect_trade_trigger(bars_1min, cvd_1min, cvd_ema_1min, vp, bar_idx)
            if trigger is None:
                continue

            current_bar = bars_1min[bar_idx]
            current_bar_ts = current_bar['t'].replace('+00:00', '') if isinstance(current_bar['t'], str) else str(current_bar['t'])
            current_price = current_bar['c']

            xgb_features = extract_xgb_features(
                bars_1min, bars_5min, bars_15min, bars_1h,
                cvd_1min, cvd_ema_1min, bar_idx, vp,
                trigger_sentiment=trigger
            )
            if xgb_features is None:
                continue

            # Stage 1 prediction
            X_stage1 = np.array(list(xgb_features.values())).reshape(1, -1)
            stage1_pred = stage1_xgb.predict(X_stage1)[0]
            stage1_proba = stage1_xgb.predict_proba(X_stage1)[0]
            stage1_confidence = stage1_proba[stage1_pred]

            # Long-term embedding
            longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, bar_idx, seq_len=72)
            if longterm_seq is None:
                continue

            # Short-term embedding
            bar_1s_idx = None
            for ts_key, idx in bars_1s_ts_to_idx.items():
                if ts_key.startswith(current_bar_ts[:16]):
                    bar_1s_idx = idx
                    break

            if bar_1s_idx is None or bar_1s_idx < 120:
                continue

            shortterm_seq = extract_shortterm_sequence(bars_1s, bar_1s_idx, seq_len=120)
            if shortterm_seq is None:
                continue

            # Regime embedding
            regime_seq = extract_regime_sequence(
                bars_1min, cvd_1min, cvd_ema_1min,
                atr_1min, bb_middle, bb_upper, bb_lower,
                adx_1min, chop_1min, rsi_1min,
                bar_idx, vp, seq_len=REGIME_SEQ_LEN
            )
            if regime_seq is None:
                continue

            # Cross-symbol correlation embedding
            if all_have_data and symbol_data:
                correlation_seq = extract_cross_symbol_features(symbol_data, bar_idx, seq_len=CORRELATION_SEQ_LEN)
            else:
                correlation_seq = None

            with torch.no_grad():
                longterm_tensor = torch.tensor(longterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                shortterm_tensor = torch.tensor(shortterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                regime_tensor = torch.tensor(regime_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                longterm_emb = longterm_lstm(longterm_tensor).cpu().numpy().flatten()
                shortterm_emb = shortterm_lstm(shortterm_tensor).cpu().numpy().flatten()
                regime_emb = regime_lstm(regime_tensor).cpu().numpy().flatten()

                if correlation_seq is not None:
                    correlation_tensor = torch.tensor(correlation_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    correlation_emb = correlation_lstm(correlation_tensor).cpu().numpy().flatten()
                else:
                    correlation_emb = np.zeros(CORRELATION_EMBEDDING_DIM)

            # Combine all features
            final_features = np.concatenate([
                np.array([stage1_pred, stage1_confidence]),  # 2 dims
                longterm_emb,      # 32 dims
                shortterm_emb,     # 32 dims
                regime_emb,        # 32 dims
                correlation_emb,   # 48 dims (NEW!)
            ])

            # Label
            future_price = bars_1min[min(bar_idx + HOLD_BARS, len(bars_1min) - 1)]['c']
            if stage1_pred == 1:  # Long
                profit = future_price - current_price
            else:  # Short
                profit = current_price - future_price

            label = 1 if profit > 0 else 0

            X_all.append(final_features)
            y_all.append(label)

    if len(X_all) < 50:
        return None

    print(f"  Training Final XGBoost on {len(X_all)} samples (label balance: {100*sum(y_all)/len(y_all):.2f}% positive)...", file=sys.stderr)

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(np.array(X_all), np.array(y_all), verbose=False)
    return model


# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(
    primary_bars_1s: Dict[str, List[Dict]],
    all_symbol_bars: Dict[str, Dict[str, List[Dict]]],
    test_dates: List[str],
    stage1_xgb: xgb.XGBClassifier,
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
    regime_lstm: nn.Module,
    correlation_lstm: nn.Module,
    final_xgb: xgb.XGBClassifier,
    primary_symbol: str,
) -> List[Dict]:
    """Run backtest on test dates."""
    trades = []

    longterm_lstm.eval()
    shortterm_lstm.eval()
    regime_lstm.eval()
    correlation_lstm.eval()

    symbols = list(all_symbol_bars.keys())

    for date in test_dates:
        if date not in primary_bars_1s:
            continue

        all_have_data = all(date in all_symbol_bars.get(sym, {}) for sym in symbols)

        bars_1s = primary_bars_1s[date]
        bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
        bars_5min = aggregate_bars(bars_1s, 5)
        bars_15min = aggregate_bars(bars_1s, 15)
        bars_1h = aggregate_bars(bars_1s, 60)

        if len(bars_1min) < 100:
            continue

        # Calculate indicators
        atr_1min = calculate_atr(bars_1min, period=14)
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_1min, period=20)
        adx_1min = calculate_adx(bars_1min, period=14)
        chop_1min = calculate_choppiness_index(bars_1min, period=14)
        rsi_1min = calculate_rsi(bars_1min, period=14)

        # Prepare cross-symbol data
        symbol_data = {}
        if all_have_data:
            for sym in symbols:
                sym_bars_1s = all_symbol_bars[sym][date]
                sym_bars_1min, sym_cvd_1min, sym_cvd_ema_1min = calculate_cvd_1min(sym_bars_1s)
                sym_bars_5min = aggregate_bars(sym_bars_1s, 5)

                symbol_data[sym] = {
                    'bars_1s': sym_bars_1s,
                    'bars_1min': sym_bars_1min,
                    'bars_5min': sym_bars_5min,
                    'cvd_1min': sym_cvd_1min,
                    'cvd_ema_1min': sym_cvd_ema_1min,
                }

        # Build index mapping
        bars_1s_ts_to_idx = {}
        for i, bar in enumerate(bars_1s):
            ts = bar['t']
            if isinstance(ts, str):
                ts = ts.replace('+00:00', '')
            bars_1s_ts_to_idx[ts] = i

        last_trade_bar = -MIN_BARS_BETWEEN_TRADES

        for bar_idx in range(max(72, REGIME_SEQ_LEN, CORRELATION_SEQ_LEN), len(bars_1min) - HOLD_BARS):
            if bar_idx - last_trade_bar < MIN_BARS_BETWEEN_TRADES:
                continue

            vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)
            trigger = detect_trade_trigger(bars_1min, cvd_1min, cvd_ema_1min, vp, bar_idx)

            if trigger is None:
                continue

            current_bar = bars_1min[bar_idx]
            current_bar_ts = current_bar['t'].replace('+00:00', '') if isinstance(current_bar['t'], str) else str(current_bar['t'])
            current_price = current_bar['c']

            xgb_features = extract_xgb_features(
                bars_1min, bars_5min, bars_15min, bars_1h,
                cvd_1min, cvd_ema_1min, bar_idx, vp,
                trigger_sentiment=trigger
            )
            if xgb_features is None:
                continue

            # Stage 1
            X_stage1 = np.array(list(xgb_features.values())).reshape(1, -1)
            stage1_pred = stage1_xgb.predict(X_stage1)[0]
            stage1_proba = stage1_xgb.predict_proba(X_stage1)[0]
            stage1_confidence = stage1_proba[stage1_pred]

            # Get embeddings
            longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, bar_idx, seq_len=72)
            if longterm_seq is None:
                continue

            bar_1s_idx = None
            for ts_key, idx in bars_1s_ts_to_idx.items():
                if ts_key.startswith(current_bar_ts[:16]):
                    bar_1s_idx = idx
                    break

            if bar_1s_idx is None or bar_1s_idx < 120:
                continue

            shortterm_seq = extract_shortterm_sequence(bars_1s, bar_1s_idx, seq_len=120)
            if shortterm_seq is None:
                continue

            regime_seq = extract_regime_sequence(
                bars_1min, cvd_1min, cvd_ema_1min,
                atr_1min, bb_middle, bb_upper, bb_lower,
                adx_1min, chop_1min, rsi_1min,
                bar_idx, vp, seq_len=REGIME_SEQ_LEN
            )
            if regime_seq is None:
                continue

            # Cross-symbol correlation embedding
            if all_have_data and symbol_data:
                correlation_seq = extract_cross_symbol_features(symbol_data, bar_idx, seq_len=CORRELATION_SEQ_LEN)
            else:
                correlation_seq = None

            with torch.no_grad():
                longterm_tensor = torch.tensor(longterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                shortterm_tensor = torch.tensor(shortterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                regime_tensor = torch.tensor(regime_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                longterm_emb = longterm_lstm(longterm_tensor).cpu().numpy().flatten()
                shortterm_emb = shortterm_lstm(shortterm_tensor).cpu().numpy().flatten()
                regime_emb = regime_lstm(regime_tensor).cpu().numpy().flatten()

                if correlation_seq is not None:
                    correlation_tensor = torch.tensor(correlation_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    correlation_emb = correlation_lstm(correlation_tensor).cpu().numpy().flatten()
                else:
                    correlation_emb = np.zeros(CORRELATION_EMBEDDING_DIM)

            # Final decision
            final_features = np.concatenate([
                np.array([stage1_pred, stage1_confidence]),
                longterm_emb,
                shortterm_emb,
                regime_emb,
                correlation_emb,
            ]).reshape(1, -1)

            final_pred = final_xgb.predict(final_features)[0]
            final_proba = final_xgb.predict_proba(final_features)[0]

            if final_pred == 0:  # HOLD
                continue

            # Execute trade
            direction = "long" if stage1_pred == 1 else "short"
            entry_price = current_price

            # Simulate trade
            exit_idx = min(bar_idx + HOLD_BARS, len(bars_1min) - 1)
            exit_price = bars_1min[exit_idx]['c']

            if direction == "long":
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            trades.append({
                'date': date,
                'entry_bar': bar_idx,
                'exit_bar': exit_idx,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl / entry_price * 100,
                'stage1_conf': stage1_confidence,
                'final_conf': final_proba[1],
                'had_correlation_data': all_have_data,
            })

            last_trade_bar = bar_idx

    return trades


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Symbol Correlation LSTM Backtest - Equal Treatment')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbol bars files (all treated equally)')
    parser.add_argument('--output', default='data/multi_symbol_equal_results.json', help='Output file')
    args = parser.parse_args()

    print(f"\n=== Multi-Symbol Correlation LSTM Backtest (EQUAL TREATMENT) ===", file=sys.stderr)
    print(f"Architecture:", file=sys.stderr)
    print(f"  Per-Symbol: Stage1 XGB -> LSTMs (3) -> embeddings (EACH symbol)", file=sys.stderr)
    print(f"  Shared: Cross-Symbol Correlation LSTM -> 48-dim embedding", file=sys.stderr)
    print(f"  Per-Symbol Final XGB: symbol_features + correlation_emb -> TAKE/HOLD", file=sys.stderr)
    print(f"  Trades ALL symbols independently using shared correlation intelligence", file=sys.stderr)

    # Load all symbol data equally
    all_symbol_bars = {}
    symbol_names = []

    for path in args.symbols:
        sym, bars = load_symbol_data(path)
        # Normalize symbol name (strip contract suffix like Z5)
        base_sym = sym.rstrip('0123456789').rstrip('HMUZ') if len(sym) > 3 else sym
        all_symbol_bars[base_sym] = bars
        symbol_names.append(base_sym)

    print(f"\nSymbols loaded (equal treatment): {symbol_names}", file=sys.stderr)

    # Get dates where ALL symbols have data
    common_dates = None
    for sym, bars in all_symbol_bars.items():
        sym_dates = set(bars.keys())
        if common_dates is None:
            common_dates = sym_dates
        else:
            common_dates = common_dates.intersection(sym_dates)

    all_dates = sorted(common_dates)
    print(f"Common dates across all symbols: {len(all_dates)}", file=sys.stderr)

    # Walk-forward validation
    min_train_days = 3
    all_trades = []  # All trades across all symbols
    trades_by_symbol = {sym: [] for sym in symbol_names}

    for test_idx in range(min_train_days, len(all_dates)):
        train_dates = all_dates[:test_idx]
        test_date = all_dates[test_idx]

        print(f"\n[{test_idx - min_train_days + 1}/{len(all_dates) - min_train_days}] Testing {test_date} (training on {len(train_dates)} days)", file=sys.stderr)

        # Train per-symbol pipelines (Stage1 XGB + 3 LSTMs for EACH symbol)
        symbol_models = {}
        for sym in symbol_names:
            print(f"  Training {sym} pipeline...", file=sys.stderr)
            sym_bars = all_symbol_bars[sym]

            # Train this symbol's models
            longterm_lstm, shortterm_lstm, regime_lstm = train_lstms(sym_bars, train_dates)
            stage1_xgb = train_stage1_xgboost(sym_bars, train_dates)

            if stage1_xgb is None:
                print(f"    Skipping {sym} - not enough training data", file=sys.stderr)
                continue

            symbol_models[sym] = {
                'stage1_xgb': stage1_xgb,
                'longterm_lstm': longterm_lstm,
                'shortterm_lstm': shortterm_lstm,
                'regime_lstm': regime_lstm,
            }

        if len(symbol_models) == 0:
            print(f"  Skipping day - no symbols have enough data", file=sys.stderr)
            continue

        # Train SHARED correlation LSTM (uses ALL symbols)
        print(f"  Training shared Correlation LSTM...", file=sys.stderr)
        correlation_lstm = train_cross_symbol_lstm(all_symbol_bars, train_dates)

        # Train per-symbol Final XGBoost (symbol features + correlation embedding)
        for sym in symbol_models.keys():
            print(f"  Training {sym} Final XGBoost...", file=sys.stderr)
            final_xgb = train_final_xgboost_with_correlation(
                all_symbol_bars[sym], all_symbol_bars, train_dates,
                symbol_models[sym]['stage1_xgb'],
                symbol_models[sym]['longterm_lstm'],
                symbol_models[sym]['shortterm_lstm'],
                symbol_models[sym]['regime_lstm'],
                correlation_lstm,
                sym
            )
            symbol_models[sym]['final_xgb'] = final_xgb

        # Run backtest on ALL symbols
        day_trades = []
        for sym in symbol_models.keys():
            if symbol_models[sym].get('final_xgb') is None:
                continue

            trades = run_backtest(
                all_symbol_bars[sym], all_symbol_bars, [test_date],
                symbol_models[sym]['stage1_xgb'],
                symbol_models[sym]['longterm_lstm'],
                symbol_models[sym]['shortterm_lstm'],
                symbol_models[sym]['regime_lstm'],
                correlation_lstm,
                symbol_models[sym]['final_xgb'],
                sym
            )

            for t in trades:
                t['symbol'] = sym
            day_trades.extend(trades)
            trades_by_symbol[sym].extend(trades)

        all_trades.extend(day_trades)

        # Day summary
        day_pnl_by_sym = {}
        for sym in symbol_names:
            sym_trades = [t for t in day_trades if t.get('symbol') == sym]
            sym_pnl = sum(t['pnl'] for t in sym_trades)
            sym_wins = sum(1 for t in sym_trades if t['pnl'] > 0)
            day_pnl_by_sym[sym] = (len(sym_trades), sym_wins, sym_pnl)

        print(f"  Day results:", file=sys.stderr)
        for sym, (total, wins, pnl) in day_pnl_by_sym.items():
            if total > 0:
                pv = POINT_VALUES.get(sym, 5)
                print(f"    {sym}: {total} trades, {wins} wins, {pnl:+.2f} pts (${pnl * pv:+,.0f})", file=sys.stderr)

    # Final Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"=== FINAL RESULTS (EQUAL TREATMENT) ===", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Symbols: {symbol_names}", file=sys.stderr)
    print(f"Total trades across all symbols: {len(all_trades)}", file=sys.stderr)

    total_pnl_dollars = 0
    results_by_symbol = {}

    for sym in symbol_names:
        sym_trades = trades_by_symbol[sym]
        if not sym_trades:
            continue

        wins = sum(1 for t in sym_trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in sym_trades)
        win_rate = 100 * wins / len(sym_trades)
        pv = POINT_VALUES.get(sym, 5)
        pnl_dollars = total_pnl * pv

        total_pnl_dollars += pnl_dollars

        results_by_symbol[sym] = {
            'trades': len(sym_trades),
            'wins': wins,
            'win_rate': win_rate,
            'pnl_points': total_pnl,
            'point_value': pv,
            'pnl_dollars': pnl_dollars,
        }

        print(f"\n{sym}:", file=sys.stderr)
        print(f"  Trades: {len(sym_trades)}", file=sys.stderr)
        print(f"  Win Rate: {win_rate:.1f}%", file=sys.stderr)
        print(f"  P&L: {total_pnl:+.2f} pts (${pnl_dollars:+,.0f} @ ${pv}/pt)", file=sys.stderr)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"COMBINED P&L: ${total_pnl_dollars:+,.0f}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Save results
    output = {
        'architecture': 'equal_treatment_with_correlation_lstm',
        'symbols': symbol_names,
        'total_trades': len(all_trades),
        'total_pnl_dollars': total_pnl_dollars,
        'results_by_symbol': results_by_symbol,
        'trades': all_trades,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
