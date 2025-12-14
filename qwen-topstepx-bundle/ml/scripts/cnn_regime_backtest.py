#!/usr/bin/env python3
"""
CNN + Regime LSTM XGBoost Backtest

Enhanced version with CNN feature extractor for live-updating candles:
- Rolling 5-minute window (4 completed + 1 live-updating candle)
- Live candle updates every 1 second with 1s OHLC and CVD
- CNN extracts spatial patterns from the candle grid
- CNN features feed into all XGBoost models

Architecture:
1. NEW: CNN Feature Extractor - 5-min rolling window updated every second
2. Stage 1 XGBoost - Direction prediction (with CNN features)
3. Long-term LSTM - 5-min context (72 bars = 6 hours)
4. Short-term LSTM - 1-second microstructure (120 bars = 2 minutes)
5. Regime LSTM - 1-min bars with indicators (60 bars = 1 hour)
6. Timing XGBoost - Uses regime + CNN to decide IF to trade
7. Final XGBoost - Makes final TAKE/HOLD decision (with CNN features)
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
import xgboost as xgb

# PyTorch for LSTM and CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Trade timeout in 1-min bars
HOLD_BARS = 20

# LSTM configuration
LONGTERM_SEQ_LEN = 72  # 6 hours of 5-min bars
SHORTTERM_SEQ_LEN = 120  # 2 minutes of 1-second bars
REGIME_SEQ_LEN = 60  # 1 hour of 1-min bars
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 2

# CNN configuration
CNN_WINDOW_SECONDS = 300  # 5 minutes = 300 seconds
CNN_EMBEDDING_DIM = 16  # CNN output embedding size

# Trade trigger settings
MIN_CVD_DIVERGENCE = 0.15
MIN_BARS_BETWEEN_TRADES = 5

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# CNN MODEL FOR LIVE CANDLE EXTRACTION
# =============================================================================

class LiveCandleCNN(nn.Module):
    """
    CNN for extracting features from rolling 5-minute window.

    Input shape: (batch, channels, height, width)
    - channels = 6 (O, H, L, C, V, CVD for each 1-second bar)
    - height = 5 (5 candles: 4 completed 1min + 1 live-updating)
    - width = 60 (60 seconds per candle slot)

    The live candle is built up second-by-second from 1s bars.
    """
    def __init__(self, embedding_dim: int = 16):
        super().__init__()

        # Input: (batch, 6, 5, 60) - 6 channels, 5 candles, 60 seconds each
        self.conv1 = nn.Conv2d(6, 32, kernel_size=(2, 5), stride=(1, 2), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(2, 3), stride=(1, 1), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(64)

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))

        # Final FC layers
        self.fc1 = nn.Linear(64 * 4, 32)
        self.fc2 = nn.Linear(32, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, 6, 5, 60)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)

        return x


def build_cnn_input(
    bars_1s: List[Dict],
    bar_1s_idx: int,
    window_seconds: int = 300,
) -> Optional[np.ndarray]:
    """
    Build CNN input from 1-second bars.

    Creates a 5-candle window:
    - 4 completed 1-minute candles
    - 1 live-updating candle built from remaining 1s bars

    Output shape: (6, 5, 60) - 6 features, 5 candles, 60 seconds each
    Features: O, H, L, C, V, CVD (normalized)
    """
    if bar_1s_idx < window_seconds:
        return None

    # Get 5 minutes of 1s bars ending at current index
    start_idx = bar_1s_idx - window_seconds
    window_bars = bars_1s[start_idx:bar_1s_idx]

    if len(window_bars) < window_seconds // 2:  # Need at least half the data
        return None

    # Reference price for normalization
    ref_price = window_bars[-1]['c'] if window_bars else 1.0
    if ref_price == 0:
        ref_price = 1.0

    # Build 5 candle slots (each 60 seconds)
    # Shape: (6, 5, 60) - 6 features, 5 candles, 60 seconds
    candle_data = np.zeros((6, 5, 60), dtype=np.float32)

    # Calculate CVD for each bar
    cvd_cumulative = 0.0
    bar_cvds = []
    for bar in window_bars:
        vol = bar.get('v', 0) or 0
        bar_range = bar['h'] - bar['l']
        if bar_range > 0:
            close_pos = (bar['c'] - bar['l']) / bar_range
            delta = vol * (2 * close_pos - 1)
        else:
            delta = vol if bar['c'] >= bar['o'] else -vol
        cvd_cumulative += delta
        bar_cvds.append(cvd_cumulative)

    # Normalize CVD
    cvd_max = max(abs(cvd_cumulative), 1)

    # Fill each candle slot
    for candle_idx in range(5):
        candle_start = candle_idx * 60
        candle_end = min((candle_idx + 1) * 60, len(window_bars))

        for sec_offset, bar_offset in enumerate(range(candle_start, candle_end)):
            if bar_offset >= len(window_bars):
                # Pad with last known values
                if sec_offset > 0:
                    for feat_idx in range(6):
                        candle_data[feat_idx, candle_idx, sec_offset] = candle_data[feat_idx, candle_idx, sec_offset - 1]
                continue

            bar = window_bars[bar_offset]

            # Normalize OHLC relative to reference price
            o_norm = (bar['o'] - ref_price) / ref_price * 100  # Scale to percentage
            h_norm = (bar['h'] - ref_price) / ref_price * 100
            l_norm = (bar['l'] - ref_price) / ref_price * 100
            c_norm = (bar['c'] - ref_price) / ref_price * 100

            # Normalize volume (log scale)
            vol = bar.get('v', 0) or 0
            v_norm = np.log1p(vol) / 10.0

            # Normalized CVD
            cvd_norm = bar_cvds[bar_offset] / cvd_max

            candle_data[0, candle_idx, sec_offset] = o_norm
            candle_data[1, candle_idx, sec_offset] = h_norm
            candle_data[2, candle_idx, sec_offset] = l_norm
            candle_data[3, candle_idx, sec_offset] = c_norm
            candle_data[4, candle_idx, sec_offset] = v_norm
            candle_data[5, candle_idx, sec_offset] = cvd_norm

    return candle_data


# =============================================================================
# TECHNICAL INDICATORS (1-min timeframe)
# =============================================================================

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
    """Calculate Bollinger Bands (middle, upper, lower)."""
    if len(bars) < period:
        return [0.0] * len(bars), [0.0] * len(bars), [0.0] * len(bars)

    closes = [b['c'] for b in bars]
    middle = [0.0] * len(bars)
    upper = [0.0] * len(bars)
    lower = [0.0] * len(bars)

    for i in range(period - 1, len(bars)):
        window = closes[i - period + 1:i + 1]
        sma = sum(window) / period
        std = (sum((x - sma) ** 2 for x in window) / period) ** 0.5

        middle[i] = sma
        upper[i] = sma + std_dev * std
        lower[i] = sma - std_dev * std

    return middle, upper, lower


def calculate_adx(bars: List[Dict], period: int = 14) -> List[float]:
    """Calculate Average Directional Index."""
    if len(bars) < period * 2:
        return [25.0] * len(bars)

    adx_values = [25.0] * len(bars)

    plus_dm = []
    minus_dm = []
    tr_values = []

    for i in range(1, len(bars)):
        high = bars[i]['h']
        low = bars[i]['l']
        prev_high = bars[i-1]['h']
        prev_low = bars[i-1]['l']
        prev_close = bars[i-1]['c']

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)

    if len(tr_values) < period:
        return adx_values

    smoothed_tr = sum(tr_values[:period])
    smoothed_plus_dm = sum(plus_dm[:period])
    smoothed_minus_dm = sum(minus_dm[:period])

    dx_values = []

    for i in range(period - 1, len(tr_values)):
        if i > period - 1:
            smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr_values[i]
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]

        plus_di = 100 * smoothed_plus_dm / smoothed_tr if smoothed_tr > 0 else 0
        minus_di = 100 * smoothed_minus_dm / smoothed_tr if smoothed_tr > 0 else 0

        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
        dx_values.append(dx)

        if len(dx_values) >= period:
            adx = sum(dx_values[-period:]) / period
            adx_values[i + 1] = adx

    return adx_values


def calculate_choppiness_index(bars: List[Dict], period: int = 14) -> List[float]:
    """Calculate Choppiness Index."""
    if len(bars) < period + 1:
        return [50.0] * len(bars)

    chop_values = [50.0] * len(bars)

    for i in range(period, len(bars)):
        tr_sum = 0.0
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
            chop_values[i] = max(0, min(100, chop))

    return chop_values


def calculate_rsi(bars: List[Dict], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index."""
    if len(bars) < period + 1:
        return [50.0] * len(bars)

    rsi_values = [50.0] * len(bars)

    gains = []
    losses = []

    for i in range(1, len(bars)):
        change = bars[i]['c'] - bars[i-1]['c']
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    if len(gains) < period:
        return rsi_values

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        rsi_values[i + 1] = rsi

    return rsi_values


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_1s_bars(filepath: str) -> List[Dict]:
    """Load 1-second bars from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("bars", [])


# =============================================================================
# BAR AGGREGATION AND CVD
# =============================================================================

def calculate_cvd_1min(bars_1s: List[Dict]) -> Tuple[List[Dict], List[float], List[float]]:
    """Calculate CVD from 1s bars, aggregated to 1-min resolution."""
    if not bars_1s:
        return [], [], []

    minute_data = defaultdict(lambda: {'bars': [], 'delta': 0.0})
    minute_order = []

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00').replace('+00:00', ''))
        else:
            dt = ts
        minute_ts = dt.replace(second=0, microsecond=0).isoformat()

        if minute_ts not in minute_data:
            minute_order.append(minute_ts)

        minute_data[minute_ts]['bars'].append(bar)

        volume = bar.get('v', 0) or 0
        bar_range = bar['h'] - bar['l']
        if bar_range > 0:
            close_position = (bar['c'] - bar['l']) / bar_range
            delta = volume * (2 * close_position - 1)
        else:
            delta = volume if bar['c'] >= bar['o'] else -volume
        minute_data[minute_ts]['delta'] += delta

    bars_1min = []
    cvd = 0.0
    cvd_1min = []

    for minute_ts in minute_order:
        data = minute_data[minute_ts]
        bars = data['bars']
        if not bars:
            continue

        bars_1min.append({
            't': minute_ts,
            'o': bars[0]['o'],
            'h': max(b['h'] for b in bars),
            'l': min(b['l'] for b in bars),
            'c': bars[-1]['c'],
            'v': sum(b.get('v', 0) or 0 for b in bars),
        })

        cvd += data['delta']
        cvd_1min.append(cvd)

    # EMA
    ema_period = 20
    if len(cvd_1min) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema = sum(cvd_1min[:ema_period]) / ema_period
        cvd_ema_1min = [ema] * ema_period
        for i in range(ema_period, len(cvd_1min)):
            ema = (cvd_1min[i] * multiplier) + (ema * (1 - multiplier))
            cvd_ema_1min.append(ema)
    else:
        cvd_ema_1min = cvd_1min.copy()

    return bars_1min, cvd_1min, cvd_ema_1min


def aggregate_bars(bars_1s: List[Dict], period_minutes: int) -> List[Dict]:
    """Aggregate 1-second bars to any timeframe."""
    if not bars_1s:
        return []

    period_data = defaultdict(lambda: {'bars': []})
    period_order = []

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00').replace('+00:00', ''))
        else:
            dt = ts

        if period_minutes >= 60:
            hour = (dt.hour // (period_minutes // 60)) * (period_minutes // 60)
            period_ts = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
        else:
            minute = (dt.minute // period_minutes) * period_minutes
            period_ts = dt.replace(minute=minute, second=0, microsecond=0)

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


def calculate_footprint_candles(bars_1s: List[Dict], period_seconds: int = 60) -> List[Dict]:
    """Calculate footprint candle data."""
    if not bars_1s:
        return []

    period_data = defaultdict(lambda: {'bars': [], 'buy_volume': 0, 'sell_volume': 0, 'price_volume': defaultdict(lambda: {'buy': 0, 'sell': 0})})
    period_order = []

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00').replace('+00:00', ''))
        else:
            dt = ts

        period_seconds_val = (dt.second // period_seconds) * period_seconds
        period_ts = dt.replace(second=period_seconds_val, microsecond=0).isoformat()

        if period_ts not in period_data:
            period_order.append(period_ts)

        period_data[period_ts]['bars'].append(bar)

        vol = bar.get('v', 0) or 0
        bar_range = bar['h'] - bar['l']
        if bar_range > 0:
            close_pos = (bar['c'] - bar['l']) / bar_range
            buy_vol = vol * close_pos
            sell_vol = vol * (1 - close_pos)
        else:
            if bar['c'] >= bar['o']:
                buy_vol, sell_vol = vol, 0
            else:
                buy_vol, sell_vol = 0, vol

        period_data[period_ts]['buy_volume'] += buy_vol
        period_data[period_ts]['sell_volume'] += sell_vol

    result = []
    for period_ts in period_order:
        data = period_data[period_ts]
        bars = data['bars']
        if not bars:
            continue

        total_vol = data['buy_volume'] + data['sell_volume']
        delta = data['buy_volume'] - data['sell_volume']

        result.append({
            't': period_ts,
            'o': bars[0]['o'],
            'h': max(b['h'] for b in bars),
            'l': min(b['l'] for b in bars),
            'c': bars[-1]['c'],
            'v': sum(b.get('v', 0) or 0 for b in bars),
            'buy_volume': data['buy_volume'],
            'sell_volume': data['sell_volume'],
            'delta': delta,
            'delta_percent': delta / total_vol if total_vol > 0 else 0,
        })

    return result


def calculate_volume_profile(bars: List[Dict], lookback: int = 30, tick_size: float = 0.25) -> Dict[str, float]:
    """Calculate Volume Profile from bars."""
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


def find_swing_highs_lows(bars: List[Dict], lookback: int = 10) -> Tuple[List[float], List[float]]:
    """Find recent swing highs and lows."""
    if len(bars) < 3:
        return [], []

    swing_highs = []
    swing_lows = []

    recent = bars[-lookback:]
    for i in range(1, len(recent) - 1):
        if recent[i]['h'] > recent[i-1]['h'] and recent[i]['h'] > recent[i+1]['h']:
            swing_highs.append(recent[i]['h'])
        if recent[i]['l'] < recent[i-1]['l'] and recent[i]['l'] < recent[i+1]['l']:
            swing_lows.append(recent[i]['l'])

    return swing_highs, swing_lows


# =============================================================================
# TRADE TRIGGER DETECTION
# =============================================================================

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

    cvd_5_ago = cvd_1min[bar_idx - 5] if bar_idx >= 5 and bar_idx - 5 < len(cvd_1min) else 0
    cvd_change = cvd_now - cvd_5_ago

    cvd_range = max(abs(cvd_1min[bar_idx - 10]), 1) if bar_idx >= 10 else 1
    cvd_change_norm = cvd_change / cvd_range

    cvd_vs_ema = cvd_now - cvd_ema_now
    cvd_ema_divergence = cvd_vs_ema / max(abs(cvd_ema_now), 1)

    price_vs_poc = (current_price - vp['poc']) / vp['poc'] if vp['poc'] > 0 else 0
    price_vs_val = (current_price - vp['val']) / vp['val'] if vp['val'] > 0 else 0
    price_vs_vah = (current_price - vp['vah']) / vp['vah'] if vp['vah'] > 0 else 0

    bullish_divergence = price_change < -0.001 and cvd_change_norm > MIN_CVD_DIVERGENCE
    bullish_cvd_strength = cvd_ema_divergence > 0.2 and price_vs_poc < 0.001
    bullish_vp_support = price_vs_val < 0.002 and cvd_change_norm > 0

    if bullish_divergence or bullish_cvd_strength or bullish_vp_support:
        return 'bullish'

    bearish_divergence = price_change > 0.001 and cvd_change_norm < -MIN_CVD_DIVERGENCE
    bearish_cvd_strength = cvd_ema_divergence < -0.2 and price_vs_poc > -0.001
    bearish_vp_resistance = price_vs_vah > -0.002 and cvd_change_norm < 0

    if bearish_divergence or bearish_cvd_strength or bearish_vp_resistance:
        return 'bearish'

    return None


# =============================================================================
# LSTM MODELS
# =============================================================================

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
        embedding = self.layer_norm(embedding)
        return embedding


class ShortTermLSTM(nn.Module):
    """Short-term LSTM for 2-minute microstructure."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, embedding_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        embedding = self.fc(last_hidden)
        embedding = self.layer_norm(embedding)
        return embedding


class RegimeLSTM(nn.Module):
    """Regime LSTM for market regime/volatility detection."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, embedding_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        embedding = self.fc(last_hidden)
        embedding = self.layer_norm(embedding)
        return embedding


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

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
        close = bar['c']
        if close == 0:
            close = 1

        o_norm = (bar['o'] - close) / close
        h_norm = (bar['h'] - close) / close
        l_norm = (bar['l'] - close) / close
        range_norm = (bar['h'] - bar['l']) / close
        body_norm = (bar['c'] - bar['o']) / close
        vol = bar.get('v', 0) or 0
        vol_log = np.log1p(vol) / 10

        cvd_start_idx = i * 5
        cvd_end_idx = min((i + 1) * 5, len(cvd_1min))
        if cvd_end_idx > cvd_start_idx and cvd_end_idx <= len(cvd_1min):
            cvd_change = (cvd_1min[cvd_end_idx - 1] - cvd_1min[cvd_start_idx]) / max(abs(cvd_1min[cvd_end_idx - 1]), 1)
        else:
            cvd_change = 0

        features.append([
            o_norm, h_norm, l_norm, range_norm, body_norm,
            vol_log, cvd_change
        ])

    return np.array(features, dtype=np.float32)


def extract_shortterm_sequence(
    bars_1s: List[Dict],
    footprint_1min: List[Dict],
    bar_idx_1s: int,
    seq_len: int = 120,
) -> Optional[np.ndarray]:
    """Extract short-term sequence for LSTM."""
    if bar_idx_1s < seq_len:
        return None

    features = []
    for i in range(bar_idx_1s - seq_len, bar_idx_1s):
        bar = bars_1s[i]
        close = bar['c']
        if close == 0:
            close = 1

        o_norm = (bar['o'] - close) / close
        h_norm = (bar['h'] - close) / close
        l_norm = (bar['l'] - close) / close
        range_norm = (bar['h'] - bar['l']) / close
        body_norm = (bar['c'] - bar['o']) / close
        vol = bar.get('v', 0) or 0
        vol_log = np.log1p(vol) / 5

        bar_range = bar['h'] - bar['l']
        if bar_range > 0:
            close_position = (bar['c'] - bar['l']) / bar_range
            delta = vol * (2 * close_position - 1)
        else:
            delta = vol if bar['c'] >= bar['o'] else -vol
        delta_norm = delta / max(vol, 1)

        bar_ts = bar['t']
        if isinstance(bar_ts, str):
            bar_dt = datetime.fromisoformat(bar_ts.replace('Z', '+00:00').replace('+00:00', ''))
        else:
            bar_dt = bar_ts
        minute_ts = bar_dt.replace(second=0, microsecond=0).isoformat()

        fp_data = next((fp for fp in footprint_1min if fp['t'] == minute_ts), None)
        if fp_data:
            fp_delta_pct = fp_data.get('delta_percent', 0)
            fp_imbalance = (fp_data.get('buy_volume', 0) - fp_data.get('sell_volume', 0)) / max(fp_data.get('v', 1), 1)
        else:
            fp_delta_pct = 0
            fp_imbalance = 0

        features.append([
            o_norm, h_norm, l_norm, range_norm, body_norm,
            vol_log, delta_norm, fp_delta_pct, fp_imbalance
        ])

    return np.array(features, dtype=np.float32)


def extract_regime_sequence(
    bars_1min: List[Dict],
    cvd_1min: List[float],
    cvd_ema_1min: List[float],
    atr_1min: List[float],
    bb_middle: List[float],
    bb_upper: List[float],
    bb_lower: List[float],
    adx_1min: List[float],
    chop_1min: List[float],
    rsi_1min: List[float],
    bar_idx: int,
    vp: Dict[str, float],
    seq_len: int = 60,
) -> Optional[np.ndarray]:
    """Extract regime sequence for Regime LSTM."""
    if bar_idx < seq_len:
        return None

    features = []

    vol_lookback = min(bar_idx, 100)
    avg_vol = np.mean([bars_1min[j].get('v', 0) or 0 for j in range(bar_idx - vol_lookback, bar_idx)]) or 1

    for i in range(bar_idx - seq_len, bar_idx):
        bar = bars_1min[i]
        close = bar['c']
        if close == 0:
            close = 1

        o_norm = (bar['o'] - close) / close
        h_norm = (bar['h'] - close) / close
        l_norm = (bar['l'] - close) / close
        range_val = bar['h'] - bar['l']
        range_norm = range_val / close
        body_norm = (bar['c'] - bar['o']) / close

        vol = bar.get('v', 0) or 0
        vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
        vol_log = np.log1p(vol_ratio)

        price_vs_poc = (close - vp['poc']) / vp['poc'] if vp['poc'] > 0 else 0
        price_vs_vah = (close - vp['vah']) / vp['vah'] if vp['vah'] > 0 else 0
        price_vs_val = (close - vp['val']) / vp['val'] if vp['val'] > 0 else 0

        cvd_val = cvd_1min[i] if i < len(cvd_1min) else 0
        cvd_ema_val = cvd_ema_1min[i] if i < len(cvd_ema_1min) else 0
        cvd_slice = cvd_1min[max(0, i-20):i+1]
        cvd_range = max(max(abs(v) for v in cvd_slice), 1) if cvd_slice and i > 0 and i < len(cvd_1min) else 1
        cvd_norm = cvd_val / cvd_range
        cvd_vs_ema = (cvd_val - cvd_ema_val) / cvd_range

        atr_val = atr_1min[i] if i < len(atr_1min) else 0
        atr_norm = atr_val / close if close > 0 else 0

        bb_mid = bb_middle[i] if i < len(bb_middle) else close
        bb_up = bb_upper[i] if i < len(bb_upper) else close + 1
        bb_low = bb_lower[i] if i < len(bb_lower) else close - 1
        bb_width = bb_up - bb_low
        bb_width_norm = bb_width / close if close > 0 else 0

        if bb_width > 0:
            bb_position = (close - bb_low) / bb_width
        else:
            bb_position = 0.5
        bb_position = max(0, min(1, bb_position))

        adx_val = adx_1min[i] if i < len(adx_1min) else 25
        adx_norm = adx_val / 100.0

        chop_val = chop_1min[i] if i < len(chop_1min) else 50
        chop_norm = chop_val / 100.0

        rsi_val = rsi_1min[i] if i < len(rsi_1min) else 50
        rsi_norm = (rsi_val - 50) / 50.0

        body_range_ratio = abs(bar['c'] - bar['o']) / range_val if range_val > 0 else 0

        features.append([
            o_norm, h_norm, l_norm, range_norm, body_norm,
            price_vs_poc, price_vs_vah, price_vs_val,
            cvd_norm, cvd_vs_ema,
            atr_norm,
            bb_position, bb_width_norm,
            adx_norm,
            chop_norm,
            rsi_norm,
            vol_log,
            body_range_ratio,
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
    vp: Dict,
    trigger_sentiment: str = None,
    cnn_features: np.ndarray = None,  # NEW: CNN features
) -> Optional[Dict[str, float]]:
    """Extract XGBoost features including CNN embedding."""
    if bar_idx < 30:
        return None

    current_bar = bars_1min[bar_idx]
    current_price = current_bar['c']

    ts_str = current_bar['t']
    if isinstance(ts_str, str):
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00').replace('+00:00', ''))
    else:
        dt = ts_str

    hour = dt.hour
    minute = dt.minute
    minutes_since_open = (hour - 9) * 60 + minute - 30

    time_sin = np.sin(2 * np.pi * minutes_since_open / 390)
    time_cos = np.cos(2 * np.pi * minutes_since_open / 390)
    is_open_30min = 1 if minutes_since_open < 30 else 0
    is_morning = 1 if 30 <= minutes_since_open < 120 else 0
    is_lunch = 1 if 120 <= minutes_since_open < 210 else 0
    is_afternoon = 1 if 210 <= minutes_since_open < 330 else 0
    is_close_30min = 1 if minutes_since_open >= 360 else 0

    cvd_now = cvd_1min[bar_idx] if bar_idx < len(cvd_1min) else 0
    cvd_ema_now = cvd_ema_1min[bar_idx] if bar_idx < len(cvd_ema_1min) else 0
    cvd_vs_ema = cvd_now - cvd_ema_now
    cvd_trend = 1 if cvd_now > cvd_ema_now else -1 if cvd_now < cvd_ema_now else 0
    cvd_slope_5 = (cvd_1min[bar_idx] - cvd_1min[bar_idx-5]) if bar_idx >= 5 else 0
    cvd_slope_10 = (cvd_1min[bar_idx] - cvd_1min[bar_idx-10]) if bar_idx >= 10 else 0
    cvd_slope_20 = (cvd_1min[bar_idx] - cvd_1min[bar_idx-20]) if bar_idx >= 20 else 0

    price_vs_poc = (current_price - vp['poc']) / vp['poc'] if vp['poc'] > 0 else 0
    price_vs_vah = (current_price - vp['vah']) / vp['vah'] if vp['vah'] > 0 else 0
    price_vs_val = (current_price - vp['val']) / vp['val'] if vp['val'] > 0 else 0
    in_value_area = 1 if vp['val'] <= current_price <= vp['vah'] else 0
    above_poc = 1 if current_price > vp['poc'] else 0

    recent_1m = bars_1min[bar_idx-4:bar_idx+1]
    bullish_1m = sum(1 for b in recent_1m if b['c'] > b['o']) / 5
    body_size_1m = np.mean([abs(b['c'] - b['o']) for b in recent_1m])
    range_1m = np.mean([b['h'] - b['l'] for b in recent_1m])
    close_position_1m = np.mean([(b['c'] - b['l']) / (b['h'] - b['l']) if b['h'] > b['l'] else 0.5 for b in recent_1m])

    idx_5m = bar_idx // 5
    if idx_5m >= 3 and idx_5m < len(bars_5min):
        recent_5m = bars_5min[idx_5m-2:idx_5m+1]
        bullish_5m = sum(1 for b in recent_5m if b['c'] > b['o']) / 3
        body_size_5m = np.mean([abs(b['c'] - b['o']) for b in recent_5m])
        close_position_5m = np.mean([(b['c'] - b['l']) / (b['h'] - b['l']) if b['h'] > b['l'] else 0.5 for b in recent_5m])
    else:
        bullish_5m = 0.5
        body_size_5m = body_size_1m
        close_position_5m = 0.5

    idx_15m = bar_idx // 15
    if idx_15m >= 2 and idx_15m < len(bars_15min):
        recent_15m = bars_15min[idx_15m-1:idx_15m+1]
        bullish_15m = sum(1 for b in recent_15m if b['c'] > b['o']) / 2
        close_position_15m = np.mean([(b['c'] - b['l']) / (b['h'] - b['l']) if b['h'] > b['l'] else 0.5 for b in recent_15m])
    else:
        bullish_15m = 0.5
        close_position_15m = 0.5

    idx_1h = bar_idx // 60
    if idx_1h >= 1 and idx_1h < len(bars_1h):
        recent_1h = bars_1h[idx_1h-1:idx_1h+1]
        bullish_1h = sum(1 for b in recent_1h if b['c'] > b['o']) / 2
        close_position_1h = np.mean([(b['c'] - b['l']) / (b['h'] - b['l']) if b['h'] > b['l'] else 0.5 for b in recent_1h])
    else:
        bullish_1h = 0.5
        close_position_1h = 0.5

    trigger_bullish = 1 if trigger_sentiment == 'bullish' else 0
    trigger_bearish = 1 if trigger_sentiment == 'bearish' else 0

    price_change_5 = (current_price - bars_1min[bar_idx-5]['c']) / bars_1min[bar_idx-5]['c'] if bar_idx >= 5 else 0
    price_change_10 = (current_price - bars_1min[bar_idx-10]['c']) / bars_1min[bar_idx-10]['c'] if bar_idx >= 10 else 0

    features = {
        'time_sin': time_sin,
        'time_cos': time_cos,
        'is_open_30min': is_open_30min,
        'is_morning': is_morning,
        'is_lunch': is_lunch,
        'is_afternoon': is_afternoon,
        'is_close_30min': is_close_30min,
        'cvd_vs_ema': cvd_vs_ema,
        'cvd_trend': cvd_trend,
        'cvd_slope_5': cvd_slope_5,
        'cvd_slope_10': cvd_slope_10,
        'cvd_slope_20': cvd_slope_20,
        'price_vs_poc': price_vs_poc,
        'price_vs_vah': price_vs_vah,
        'price_vs_val': price_vs_val,
        'in_value_area': in_value_area,
        'above_poc': above_poc,
        'bullish_1m': bullish_1m,
        'body_size_1m': body_size_1m,
        'range_1m': range_1m,
        'close_position_1m': close_position_1m,
        'bullish_5m': bullish_5m,
        'body_size_5m': body_size_5m,
        'close_position_5m': close_position_5m,
        'bullish_15m': bullish_15m,
        'close_position_15m': close_position_15m,
        'bullish_1h': bullish_1h,
        'close_position_1h': close_position_1h,
        'trigger_bullish': trigger_bullish,
        'trigger_bearish': trigger_bearish,
        'price_change_5': price_change_5,
        'price_change_10': price_change_10,
    }

    # Add CNN features if available
    if cnn_features is not None:
        for i, val in enumerate(cnn_features):
            features[f'cnn_{i}'] = float(val)

    return features


# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate_trade(
    bars_1min: List[Dict],
    entry_bar_idx: int,
    direction: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    max_bars: int = 20,
) -> Dict[str, Any]:
    """Simulate a trade and return outcome."""
    exit_price = entry_price
    exit_reason = "timeout"
    outcome = "loss"
    bars_held = 0

    for i in range(1, min(max_bars + 1, len(bars_1min) - entry_bar_idx)):
        bar = bars_1min[entry_bar_idx + i]
        bars_held = i

        if direction == "long":
            if bar['l'] <= stop_loss:
                exit_price = stop_loss
                exit_reason = "stop_loss"
                outcome = "loss"
                break
            if bar['h'] >= target:
                exit_price = target
                exit_reason = "target"
                outcome = "win"
                break
        else:
            if bar['h'] >= stop_loss:
                exit_price = stop_loss
                exit_reason = "stop_loss"
                outcome = "loss"
                break
            if bar['l'] <= target:
                exit_price = target
                exit_reason = "target"
                outcome = "win"
                break

        exit_price = bar['c']

    if exit_reason == "timeout":
        if direction == "long":
            outcome = "win" if exit_price > entry_price else "loss"
        else:
            outcome = "win" if exit_price < entry_price else "loss"

    if direction == "long":
        pnl_points = exit_price - entry_price
    else:
        pnl_points = entry_price - exit_price

    return {
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "outcome": outcome,
        "pnl_points": pnl_points,
        "bars_held": bars_held,
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_cnn(
    all_bars_1s: Dict[str, List[Dict]],
    train_dates: List[str],
    epochs: int = 10,
) -> LiveCandleCNN:
    """Train CNN on 5-minute rolling windows to predict short-term direction."""
    print("  Training CNN...", file=sys.stderr)

    cnn = LiveCandleCNN(embedding_dim=CNN_EMBEDDING_DIM).to(DEVICE)
    classifier = nn.Linear(CNN_EMBEDDING_DIM, 2).to(DEVICE)

    X_all = []
    y_all = []

    for date in train_dates:
        if date not in all_bars_1s:
            continue

        bars_1s = all_bars_1s[date]
        if len(bars_1s) < CNN_WINDOW_SECONDS + 60:  # Need enough data
            continue

        # Sample every 60 bars (1 per minute) to keep training manageable
        for bar_idx in range(CNN_WINDOW_SECONDS, len(bars_1s) - 60, 60):
            cnn_input = build_cnn_input(bars_1s, bar_idx, CNN_WINDOW_SECONDS)
            if cnn_input is None:
                continue

            # Label: price direction in next 60 seconds
            current_price = bars_1s[bar_idx - 1]['c']
            future_price = bars_1s[min(bar_idx + 60, len(bars_1s) - 1)]['c']
            label = 1 if future_price > current_price else 0

            X_all.append(cnn_input)
            y_all.append(label)

    if len(X_all) < 100:
        print("  Insufficient data for CNN training", file=sys.stderr)
        return cnn

    X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(np.array(y_all), dtype=torch.long).to(DEVICE)

    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(classifier.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    cnn.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = cnn(X_tensor)
        logits = classifier(embeddings)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 3 == 0:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y_tensor).float().mean()
                print(f"    Epoch {epoch}: loss={loss.item():.4f}, acc={acc.item():.3f}", file=sys.stderr)

    return cnn


def train_lstms(
    all_bars_1s: Dict[str, List[Dict]],
    train_dates: List[str],
    epochs: int = 10,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Train all three LSTMs."""
    print("  Training LSTMs...", file=sys.stderr)

    longterm_seqs = []
    longterm_labels = []
    shortterm_seqs = []
    shortterm_labels = []
    regime_seqs = []
    regime_labels = []

    for date in train_dates:
        if date not in all_bars_1s:
            continue

        bars_1s = all_bars_1s[date]
        bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
        bars_5min = aggregate_bars(bars_1s, 5)
        footprint_1min = calculate_footprint_candles(bars_1s, period_seconds=60)

        if len(bars_1min) < 100:
            continue

        atr_1min = calculate_atr(bars_1min, period=14)
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_1min, period=20)
        adx_1min = calculate_adx(bars_1min, period=14)
        chop_1min = calculate_choppiness_index(bars_1min, period=14)
        rsi_1min = calculate_rsi(bars_1min, period=14)

        # Long-term sequences (5-min bars)
        for i in range(LONGTERM_SEQ_LEN, len(bars_5min) - 12, 6):
            seq = extract_longterm_sequence(bars_5min, cvd_1min, i * 5, seq_len=LONGTERM_SEQ_LEN)
            if seq is None:
                continue

            current_price = bars_5min[i]['c']
            future_price = bars_5min[min(i + 12, len(bars_5min) - 1)]['c']
            label = 1 if future_price > current_price else 0

            longterm_seqs.append(seq)
            longterm_labels.append(label)

        # Short-term sequences (1s bars)
        bars_1s_ts_to_idx = {}
        for i, bar in enumerate(bars_1s):
            ts = bar['t']
            if isinstance(ts, str):
                ts = ts.replace('+00:00', '')
            bars_1s_ts_to_idx[ts] = i

        for i in range(200, len(bars_1s) - 60, 60):
            seq = extract_shortterm_sequence(bars_1s, footprint_1min, i, seq_len=SHORTTERM_SEQ_LEN)
            if seq is None:
                continue

            current_price = bars_1s[i]['c']
            future_price = bars_1s[min(i + 60, len(bars_1s) - 1)]['c']
            label = 1 if future_price > current_price else 0

            shortterm_seqs.append(seq)
            shortterm_labels.append(label)

        # Regime sequences (1-min bars)
        for i in range(REGIME_SEQ_LEN, len(bars_1min) - HOLD_BARS, 5):
            vp = calculate_volume_profile(bars_1min[:i+1], lookback=30)
            seq = extract_regime_sequence(
                bars_1min, cvd_1min, cvd_ema_1min,
                atr_1min, bb_middle, bb_upper, bb_lower,
                adx_1min, chop_1min, rsi_1min,
                i, vp, seq_len=REGIME_SEQ_LEN
            )
            if seq is None:
                continue

            current_price = bars_1min[i]['c']
            future_idx = min(i + HOLD_BARS, len(bars_1min) - 1)
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


def train_stage1_xgboost(
    all_bars_1s: Dict[str, List[Dict]],
    train_dates: List[str],
    cnn: nn.Module,
) -> xgb.XGBClassifier:
    """Train Stage 1 XGBoost with CNN features."""
    print("  Training Stage 1 XGBoost (with CNN)...", file=sys.stderr)

    X_all = []
    y_all = []

    cnn.eval()

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

        # Build 1s to 1min index mapping
        bars_1s_ts_to_idx = {}
        for i, bar in enumerate(bars_1s):
            ts = bar['t']
            if isinstance(ts, str):
                ts = ts.replace('+00:00', '')
            bars_1s_ts_to_idx[ts] = i

        for i in range(50, len(bars_1min) - HOLD_BARS, 5):
            vp = calculate_volume_profile(bars_1min[:i+1], lookback=30)

            # Find corresponding 1s bar index
            current_bar = bars_1min[i]
            current_bar_ts = current_bar['t'].replace('+00:00', '') if isinstance(current_bar['t'], str) else current_bar['t']

            bar_1s_idx = None
            for ts_key, idx in bars_1s_ts_to_idx.items():
                if ts_key.startswith(current_bar_ts[:16]):
                    bar_1s_idx = idx
                    break

            # Get CNN features
            cnn_features = None
            if bar_1s_idx is not None and bar_1s_idx >= CNN_WINDOW_SECONDS:
                cnn_input = build_cnn_input(bars_1s, bar_1s_idx, CNN_WINDOW_SECONDS)
                if cnn_input is not None:
                    with torch.no_grad():
                        cnn_tensor = torch.tensor(cnn_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        cnn_features = cnn(cnn_tensor).cpu().numpy().flatten()

            features = extract_xgb_features(
                bars_1min, bars_5min, bars_15min, bars_1h,
                cvd_1min, cvd_ema_1min, i, vp,
                trigger_sentiment=None,
                cnn_features=cnn_features
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


def train_timing_xgboost(
    all_bars_1s: Dict[str, List[Dict]],
    train_dates: List[str],
    regime_lstm: nn.Module,
    cnn: nn.Module,
) -> xgb.XGBClassifier:
    """Train Timing XGBoost with regime + CNN embeddings."""
    print("  Training Timing XGBoost (with CNN)...", file=sys.stderr)

    X_all = []
    y_all = []

    regime_lstm.eval()
    cnn.eval()

    for date in train_dates:
        if date not in all_bars_1s:
            continue

        bars_1s = all_bars_1s[date]
        bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)

        if len(bars_1min) < 100:
            continue

        atr_1min = calculate_atr(bars_1min, period=14)
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_1min, period=20)
        adx_1min = calculate_adx(bars_1min, period=14)
        chop_1min = calculate_choppiness_index(bars_1min, period=14)
        rsi_1min = calculate_rsi(bars_1min, period=14)

        # Build 1s to 1min index mapping
        bars_1s_ts_to_idx = {}
        for i, bar in enumerate(bars_1s):
            ts = bar['t']
            if isinstance(ts, str):
                ts = ts.replace('+00:00', '')
            bars_1s_ts_to_idx[ts] = i

        for i in range(max(60, REGIME_SEQ_LEN), len(bars_1min) - HOLD_BARS, 5):
            vp = calculate_volume_profile(bars_1min[:i+1], lookback=30)

            regime_seq = extract_regime_sequence(
                bars_1min, cvd_1min, cvd_ema_1min,
                atr_1min, bb_middle, bb_upper, bb_lower,
                adx_1min, chop_1min, rsi_1min,
                i, vp, seq_len=REGIME_SEQ_LEN
            )
            if regime_seq is None:
                continue

            # Find corresponding 1s bar index
            current_bar = bars_1min[i]
            current_bar_ts = current_bar['t'].replace('+00:00', '') if isinstance(current_bar['t'], str) else current_bar['t']
            current_price = current_bar['c']

            bar_1s_idx = None
            for ts_key, idx in bars_1s_ts_to_idx.items():
                if ts_key.startswith(current_bar_ts[:16]):
                    bar_1s_idx = idx
                    break

            # Get CNN features
            cnn_emb = np.zeros(CNN_EMBEDDING_DIM)
            if bar_1s_idx is not None and bar_1s_idx >= CNN_WINDOW_SECONDS:
                cnn_input = build_cnn_input(bars_1s, bar_1s_idx, CNN_WINDOW_SECONDS)
                if cnn_input is not None:
                    with torch.no_grad():
                        cnn_tensor = torch.tensor(cnn_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        cnn_emb = cnn(cnn_tensor).cpu().numpy().flatten()

            with torch.no_grad():
                regime_tensor = torch.tensor(regime_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                regime_emb = regime_lstm(regime_tensor).cpu().numpy().flatten()

            current_atr = atr_1min[i] if i < len(atr_1min) else 0
            current_adx = adx_1min[i] if i < len(adx_1min) else 25
            current_chop = chop_1min[i] if i < len(chop_1min) else 50
            current_rsi = rsi_1min[i] if i < len(rsi_1min) else 50
            current_bb_up = bb_upper[i] if i < len(bb_upper) else current_price
            current_bb_low = bb_lower[i] if i < len(bb_lower) else current_price

            bb_width = (current_bb_up - current_bb_low) / current_price if current_price > 0 else 0
            bb_position = (current_price - current_bb_low) / (current_bb_up - current_bb_low) if current_bb_up > current_bb_low else 0.5
            atr_norm = current_atr / current_price if current_price > 0 else 0

            timing_features = np.concatenate([
                regime_emb,
                cnn_emb,  # Add CNN embedding
                np.array([
                    atr_norm,
                    current_adx / 100.0,
                    current_chop / 100.0,
                    (current_rsi - 50) / 50.0,
                    bb_width,
                    bb_position,
                ])
            ])

            # Label: is this a good time to trade?
            future_idx = min(i + HOLD_BARS, len(bars_1min) - 1)
            future_high = max(b['h'] for b in bars_1min[i:future_idx+1])
            future_low = min(b['l'] for b in bars_1min[i:future_idx+1])

            long_potential = (future_high - current_price) / current_price
            short_potential = (current_price - future_low) / current_price
            best_potential = max(long_potential, short_potential)

            label = 1 if best_potential > 0.0015 else 0

            X_all.append(timing_features)
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


def build_final_training_data(
    all_bars_1s: Dict[str, List[Dict]],
    train_dates: List[str],
    stage1_xgb: xgb.XGBClassifier,
    timing_xgb: xgb.XGBClassifier,
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
    regime_lstm: nn.Module,
    cnn: nn.Module,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build training data for final XGBoost with all embeddings including CNN."""
    X_all = []
    y_all = []

    longterm_lstm.eval()
    shortterm_lstm.eval()
    regime_lstm.eval()
    cnn.eval()

    for date in train_dates:
        if date not in all_bars_1s:
            continue

        bars_1s = all_bars_1s[date]
        bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
        bars_5min = aggregate_bars(bars_1s, 5)
        bars_15min = aggregate_bars(bars_1s, 15)
        bars_1h = aggregate_bars(bars_1s, 60)
        footprint_1min = calculate_footprint_candles(bars_1s, period_seconds=60)

        if len(bars_1min) < 100:
            continue

        atr_1min = calculate_atr(bars_1min, period=14)
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_1min, period=20)
        adx_1min = calculate_adx(bars_1min, period=14)
        chop_1min = calculate_choppiness_index(bars_1min, period=14)
        rsi_1min = calculate_rsi(bars_1min, period=14)

        bars_1s_ts_to_idx = {}
        for i, bar in enumerate(bars_1s):
            ts = bar['t']
            if isinstance(ts, str):
                ts = ts.replace('+00:00', '')
            bars_1s_ts_to_idx[ts] = i

        for bar_idx in range(max(72, REGIME_SEQ_LEN), len(bars_1min) - HOLD_BARS, 3):
            vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

            trigger = detect_trade_trigger(bars_1min, cvd_1min, cvd_ema_1min, vp, bar_idx)
            if trigger is None:
                continue

            current_bar = bars_1min[bar_idx]
            current_bar_ts = current_bar['t'].replace('+00:00', '') if isinstance(current_bar['t'], str) else current_bar['t']
            current_price = current_bar['c']

            # Find 1s index
            bar_1s_idx = None
            for ts_key, idx in bars_1s_ts_to_idx.items():
                if ts_key.startswith(current_bar_ts[:16]):
                    bar_1s_idx = idx
                    break

            if bar_1s_idx is None or bar_1s_idx < max(120, CNN_WINDOW_SECONDS):
                continue

            # Get CNN features
            cnn_input = build_cnn_input(bars_1s, bar_1s_idx, CNN_WINDOW_SECONDS)
            if cnn_input is None:
                continue

            with torch.no_grad():
                cnn_tensor = torch.tensor(cnn_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                cnn_emb = cnn(cnn_tensor).cpu().numpy().flatten()

            xgb_features = extract_xgb_features(
                bars_1min, bars_5min, bars_15min, bars_1h,
                cvd_1min, cvd_ema_1min, bar_idx, vp,
                trigger_sentiment=trigger,
                cnn_features=cnn_emb
            )
            if xgb_features is None:
                continue

            # Stage 1 prediction
            X_stage1 = np.array(list(xgb_features.values())).reshape(1, -1)
            stage1_pred = stage1_xgb.predict(X_stage1)[0]
            stage1_proba = stage1_xgb.predict_proba(X_stage1)[0]
            stage1_direction = "long" if stage1_pred == 1 else "short"
            stage1_confidence = stage1_proba[stage1_pred]

            # Get all LSTM embeddings
            longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, bar_idx, seq_len=72)
            if longterm_seq is None:
                continue

            shortterm_seq = extract_shortterm_sequence(bars_1s, footprint_1min, bar_1s_idx, seq_len=120)
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

            with torch.no_grad():
                longterm_tensor = torch.tensor(longterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                shortterm_tensor = torch.tensor(shortterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                regime_tensor = torch.tensor(regime_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                longterm_emb = longterm_lstm(longterm_tensor).cpu().numpy().flatten()
                shortterm_emb = shortterm_lstm(shortterm_tensor).cpu().numpy().flatten()
                regime_emb = regime_lstm(regime_tensor).cpu().numpy().flatten()

            # Get timing XGBoost prediction
            current_atr = atr_1min[bar_idx] if bar_idx < len(atr_1min) else 0
            current_adx = adx_1min[bar_idx] if bar_idx < len(adx_1min) else 25
            current_chop = chop_1min[bar_idx] if bar_idx < len(chop_1min) else 50
            current_rsi = rsi_1min[bar_idx] if bar_idx < len(rsi_1min) else 50
            current_bb_up = bb_upper[bar_idx] if bar_idx < len(bb_upper) else current_price
            current_bb_low = bb_lower[bar_idx] if bar_idx < len(bb_lower) else current_price

            bb_width = (current_bb_up - current_bb_low) / current_price if current_price > 0 else 0
            bb_position = (current_price - current_bb_low) / (current_bb_up - current_bb_low) if current_bb_up > current_bb_low else 0.5
            atr_norm = current_atr / current_price if current_price > 0 else 0

            timing_features = np.concatenate([
                regime_emb,
                cnn_emb,
                np.array([atr_norm, current_adx / 100.0, current_chop / 100.0, (current_rsi - 50) / 50.0, bb_width, bb_position])
            ])

            timing_pred = timing_xgb.predict(timing_features.reshape(1, -1))[0]
            timing_proba = timing_xgb.predict_proba(timing_features.reshape(1, -1))[0]
            timing_confidence = timing_proba[1]

            # Calculate SL/TP
            swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:bar_idx//5+1], lookback=10)

            if stage1_direction == "long":
                valid_lows = [sl for sl in swing_lows if sl < current_price] if swing_lows else []
                stop_loss = min(valid_lows) - 1 if valid_lows else current_price - 10
                risk = current_price - stop_loss
                target = current_price + (risk * 3)
            else:
                valid_highs = [sh for sh in swing_highs if sh > current_price] if swing_highs else []
                stop_loss = max(valid_highs) + 1 if valid_highs else current_price + 10
                risk = stop_loss - current_price
                target = current_price - (risk * 3)

            result = simulate_trade(bars_1min, bar_idx, stage1_direction, current_price, stop_loss, target, max_bars=HOLD_BARS)

            # Combined features with all embeddings including CNN
            combined = np.concatenate([
                np.array(list(xgb_features.values())),
                longterm_emb,
                shortterm_emb,
                regime_emb,
                cnn_emb,
                np.array([stage1_confidence, timing_confidence])
            ])

            min_pnl_threshold = 2.0
            if result['pnl_points'] > min_pnl_threshold:
                label = 1
            else:
                label = 0

            X_all.append(combined)
            y_all.append(label)

    return np.array(X_all), np.array(y_all)


# =============================================================================
# SINGLE DAY BACKTEST
# =============================================================================

def run_single_day(
    bars_1s: List[Dict],
    target_date: str,
    stage1_xgb: xgb.XGBClassifier,
    timing_xgb: xgb.XGBClassifier,
    final_xgb: xgb.XGBClassifier,
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
    regime_lstm: nn.Module,
    cnn: nn.Module,
    starting_equity: float = 0.0,
) -> Tuple[List[Dict], Dict]:
    """Run backtest for a single day with CNN features."""

    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
    bars_5min = aggregate_bars(bars_1s, 5)
    bars_15min = aggregate_bars(bars_1s, 15)
    bars_1h = aggregate_bars(bars_1s, 60)
    footprint_1min = calculate_footprint_candles(bars_1s, period_seconds=60)

    if len(bars_1min) < 100:
        return [], {"max_intraday_dd_points": 0, "max_intraday_dd_from_peak_points": 0}, starting_equity, starting_equity

    atr_1min = calculate_atr(bars_1min, period=14)
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(bars_1min, period=20)
    adx_1min = calculate_adx(bars_1min, period=14)
    chop_1min = calculate_choppiness_index(bars_1min, period=14)
    rsi_1min = calculate_rsi(bars_1min, period=14)

    bars_1s_ts_to_idx = {}
    for i, bar in enumerate(bars_1s):
        ts = bar['t']
        if isinstance(ts, str):
            ts = ts.replace('+00:00', '')
        bars_1s_ts_to_idx[ts] = i

    trades = []
    current_position_exit_bar = 0

    longterm_lstm.eval()
    shortterm_lstm.eval()
    regime_lstm.eval()
    cnn.eval()

    intraday_pnl = 0.0
    intraday_peak_pnl = 0.0
    max_intraday_dd = 0.0
    max_intraday_dd_from_peak = 0.0

    cumulative_equity = starting_equity
    peak_equity = starting_equity

    for bar_idx in range(max(72, REGIME_SEQ_LEN), len(bars_1min) - HOLD_BARS):
        if bar_idx < current_position_exit_bar:
            continue

        if trades and bar_idx - trades[-1].get('entry_bar_idx', 0) < MIN_BARS_BETWEEN_TRADES:
            continue

        current_bar = bars_1min[bar_idx]
        current_bar_ts = current_bar['t'].replace('+00:00', '') if isinstance(current_bar['t'], str) else current_bar['t']
        current_price = current_bar['c']

        vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

        trigger = detect_trade_trigger(bars_1min, cvd_1min, cvd_ema_1min, vp, bar_idx)
        if trigger is None:
            continue

        # Find 1s index
        bar_1s_idx = None
        for ts_key, idx in bars_1s_ts_to_idx.items():
            if ts_key.startswith(current_bar_ts[:16]):
                bar_1s_idx = idx
                break

        if bar_1s_idx is None or bar_1s_idx < max(120, CNN_WINDOW_SECONDS):
            continue

        # Get CNN features
        cnn_input = build_cnn_input(bars_1s, bar_1s_idx, CNN_WINDOW_SECONDS)
        if cnn_input is None:
            continue

        with torch.no_grad():
            cnn_tensor = torch.tensor(cnn_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            cnn_emb = cnn(cnn_tensor).cpu().numpy().flatten()

        xgb_features = extract_xgb_features(
            bars_1min, bars_5min, bars_15min, bars_1h,
            cvd_1min, cvd_ema_1min, bar_idx, vp,
            trigger_sentiment=trigger,
            cnn_features=cnn_emb
        )
        if xgb_features is None:
            continue

        # Stage 1 prediction
        X_stage1 = np.array(list(xgb_features.values())).reshape(1, -1)
        stage1_pred = stage1_xgb.predict(X_stage1)[0]
        stage1_proba = stage1_xgb.predict_proba(X_stage1)[0]
        stage1_direction = "long" if stage1_pred == 1 else "short"
        stage1_confidence = stage1_proba[stage1_pred]

        # Get all LSTM embeddings
        longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, bar_idx, seq_len=72)
        if longterm_seq is None:
            continue

        shortterm_seq = extract_shortterm_sequence(bars_1s, footprint_1min, bar_1s_idx, seq_len=120)
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

        with torch.no_grad():
            longterm_tensor = torch.tensor(longterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            shortterm_tensor = torch.tensor(shortterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            regime_tensor = torch.tensor(regime_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            longterm_emb = longterm_lstm(longterm_tensor).cpu().numpy().flatten()
            shortterm_emb = shortterm_lstm(shortterm_tensor).cpu().numpy().flatten()
            regime_emb = regime_lstm(regime_tensor).cpu().numpy().flatten()

        # Get timing XGBoost prediction
        current_atr = atr_1min[bar_idx] if bar_idx < len(atr_1min) else 0
        current_adx = adx_1min[bar_idx] if bar_idx < len(adx_1min) else 25
        current_chop = chop_1min[bar_idx] if bar_idx < len(chop_1min) else 50
        current_rsi = rsi_1min[bar_idx] if bar_idx < len(rsi_1min) else 50
        current_bb_up = bb_upper[bar_idx] if bar_idx < len(bb_upper) else current_price
        current_bb_low = bb_lower[bar_idx] if bar_idx < len(bb_lower) else current_price

        bb_width = (current_bb_up - current_bb_low) / current_price if current_price > 0 else 0
        bb_position = (current_price - current_bb_low) / (current_bb_up - current_bb_low) if current_bb_up > current_bb_low else 0.5
        atr_norm = current_atr / current_price if current_price > 0 else 0

        timing_features = np.concatenate([
            regime_emb,
            cnn_emb,
            np.array([atr_norm, current_adx / 100.0, current_chop / 100.0, (current_rsi - 50) / 50.0, bb_width, bb_position])
        ])

        timing_pred = timing_xgb.predict(timing_features.reshape(1, -1))[0]
        timing_proba = timing_xgb.predict_proba(timing_features.reshape(1, -1))[0]
        timing_confidence = timing_proba[1]

        if timing_pred != 1:
            continue

        # Calculate SL/TP
        swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:bar_idx//5+1], lookback=10)

        if stage1_direction == "long":
            valid_lows = [sl for sl in swing_lows if sl < current_price] if swing_lows else []
            stop_loss = min(valid_lows) - 1 if valid_lows else current_price - 10
            risk = current_price - stop_loss
            target = current_price + (risk * 3)
        else:
            valid_highs = [sh for sh in swing_highs if sh > current_price] if swing_highs else []
            stop_loss = max(valid_highs) + 1 if valid_highs else current_price + 10
            risk = stop_loss - current_price
            target = current_price - (risk * 3)

        # Combined features for final XGBoost
        combined = np.concatenate([
            np.array(list(xgb_features.values())),
            longterm_emb,
            shortterm_emb,
            regime_emb,
            cnn_emb,
            np.array([stage1_confidence, timing_confidence])
        ])

        X_pred = combined.reshape(1, -1)
        pred = final_xgb.predict(X_pred)[0]
        proba = final_xgb.predict_proba(X_pred)[0]

        if pred != 1:
            continue

        final_score = proba[1]

        result = simulate_trade(
            bars_1min,
            bar_idx,
            stage1_direction,
            current_price,
            stop_loss,
            target,
            max_bars=HOLD_BARS
        )

        current_position_exit_bar = bar_idx + result["bars_held"]

        trade_pnl = result["pnl_points"]
        intraday_pnl += trade_pnl
        cumulative_equity += trade_pnl

        if intraday_pnl > intraday_peak_pnl:
            intraday_peak_pnl = intraday_pnl
        if cumulative_equity > peak_equity:
            peak_equity = cumulative_equity

        dd_from_day_start = -intraday_pnl if intraday_pnl < 0 else 0
        dd_from_intraday_peak = intraday_peak_pnl - intraday_pnl

        if dd_from_day_start > max_intraday_dd:
            max_intraday_dd = dd_from_day_start
        if dd_from_intraday_peak > max_intraday_dd_from_peak:
            max_intraday_dd_from_peak = dd_from_intraday_peak

        trade = {
            "date": target_date,
            "timestamp": current_bar_ts[:19],
            "direction": stage1_direction,
            "entry_price": current_price,
            "entry_bar_idx": bar_idx,
            "stage1_confidence": float(stage1_confidence),
            "timing_confidence": float(timing_confidence),
            "final_score": float(final_score),
            "trigger": trigger,
            "outcome": result["outcome"],
            "pnl_points": result["pnl_points"],
            "exit_reason": result["exit_reason"],
            "cumulative_intraday_pnl": round(intraday_pnl, 2),
            "cumulative_equity": round(cumulative_equity, 2),
        }
        trades.append(trade)

    if trades:
        wins = sum(1 for t in trades if t["outcome"] == "win")
        total_pnl = sum(t["pnl_points"] for t in trades)
        analysis = {
            "date": target_date,
            "total_trades": len(trades),
            "wins": wins,
            "losses": len(trades) - wins,
            "win_rate": round(wins / len(trades) * 100, 1),
            "total_pnl_points": round(total_pnl, 2),
            "max_intraday_dd_points": round(max_intraday_dd, 2),
            "max_intraday_dd_from_peak_points": round(max_intraday_dd_from_peak, 2),
            "ending_equity": round(cumulative_equity, 2),
            "peak_equity": round(peak_equity, 2),
        }
    else:
        analysis = {
            "date": target_date,
            "total_trades": 0,
            "max_intraday_dd_points": 0,
            "max_intraday_dd_from_peak_points": 0,
            "ending_equity": round(starting_equity, 2),
            "peak_equity": round(starting_equity, 2),
        }

    return trades, analysis, cumulative_equity, peak_equity


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CNN + Regime LSTM XGBoost Backtest')
    parser.add_argument('--bars', required=True, help='Path to 1s bars JSON')
    parser.add_argument('--output', default='data/cnn_regime_results.json', help='Path to output JSON')
    parser.add_argument('--point-value', type=float, default=2.0,
                        help='Dollar value per point (NQ=$20, MNQ=$2, MES=$5, MGC=$10)')
    args = parser.parse_args()

    POINT_VALUE = args.point_value

    print("=" * 70, file=sys.stderr)
    print("CNN + REGIME LSTM XGBOOST BACKTEST", file=sys.stderr)
    print("NEW: CNN Feature Extractor with rolling 5-min window", file=sys.stderr)
    print("     (4 completed + 1 live-updating candle, updated every 1s)", file=sys.stderr)
    print("Architecture: CNN -> Stage1 XGB -> LSTMs (3) -> Timing XGB -> Final XGB", file=sys.stderr)
    print(f"Device: {DEVICE}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    print("[Load] Loading bars data...", file=sys.stderr)
    all_bars = load_1s_bars(args.bars)

    all_bars_1s = defaultdict(list)
    for bar in all_bars:
        ts = bar['t']
        if isinstance(ts, str):
            date = ts[:10]
        else:
            date = ts.strftime('%Y-%m-%d')
        all_bars_1s[date].append(bar)

    dates = sorted(all_bars_1s.keys())
    print(f"[Load] Found bars for dates: {dates[:5]}...{dates[-3:]}" if len(dates) > 8 else f"[Load] Found bars for dates: {dates}", file=sys.stderr)

    all_trades = []
    all_analyses = []

    cumulative_equity = 0.0
    peak_equity = 0.0

    # Walk-forward: start testing after min_train_days (same as no_whale_regime)
    min_train_days = 3
    train_size = min_train_days

    for test_idx in range(train_size, len(dates)):
        train_dates = dates[:test_idx]
        test_date = dates[test_idx]

        print(f"\n[Walk-Forward] Test: {test_date} | Train: {train_dates[0]} to {train_dates[-1]}", file=sys.stderr)

        # Train CNN
        cnn = train_cnn(all_bars_1s, train_dates, epochs=10)

        # Train LSTMs
        longterm_lstm, shortterm_lstm, regime_lstm = train_lstms(all_bars_1s, train_dates, epochs=10)

        # Train Stage 1 XGBoost with CNN features
        stage1_xgb = train_stage1_xgboost(all_bars_1s, train_dates, cnn)
        if stage1_xgb is None:
            print(f"  Skipping {test_date} - insufficient training data for Stage1", file=sys.stderr)
            continue

        # Train Timing XGBoost with CNN features
        timing_xgb = train_timing_xgboost(all_bars_1s, train_dates, regime_lstm, cnn)
        if timing_xgb is None:
            print(f"  Skipping {test_date} - insufficient training data for Timing", file=sys.stderr)
            continue

        # Build final training data
        X_final, y_final = build_final_training_data(
            all_bars_1s, train_dates,
            stage1_xgb, timing_xgb,
            longterm_lstm, shortterm_lstm, regime_lstm, cnn
        )

        if len(X_final) < 50:
            print(f"  Skipping {test_date} - insufficient final training samples", file=sys.stderr)
            continue

        final_xgb = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        final_xgb.fit(X_final, y_final, verbose=False)

        # Run test day
        trades, analysis, cumulative_equity, peak_equity = run_single_day(
            all_bars_1s[test_date],
            test_date,
            stage1_xgb, timing_xgb, final_xgb,
            longterm_lstm, shortterm_lstm, regime_lstm, cnn,
            starting_equity=cumulative_equity
        )

        all_trades.extend(trades)
        all_analyses.append(analysis)

        if trades:
            day_pnl = sum(t['pnl_points'] for t in trades)
            day_pnl_dollars = day_pnl * POINT_VALUE
            print(f"  {test_date}: {len(trades)} trades, {analysis.get('win_rate', 0):.1f}% WR, {day_pnl:.2f} pts (${day_pnl_dollars:.2f})", file=sys.stderr)
        else:
            print(f"  {test_date}: No trades", file=sys.stderr)

    # Summary
    print("\n" + "=" * 70, file=sys.stderr)
    print("BACKTEST COMPLETE", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    if all_trades:
        total_trades = len(all_trades)
        wins = sum(1 for t in all_trades if t['outcome'] == 'win')
        losses = total_trades - wins
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        total_pnl_points = sum(t['pnl_points'] for t in all_trades)
        total_pnl_dollars = total_pnl_points * POINT_VALUE

        avg_win = np.mean([t['pnl_points'] for t in all_trades if t['outcome'] == 'win']) if wins > 0 else 0
        avg_loss = np.mean([t['pnl_points'] for t in all_trades if t['outcome'] == 'loss']) if losses > 0 else 0

        print(f"Total Trades: {total_trades}", file=sys.stderr)
        print(f"Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%", file=sys.stderr)
        print(f"Total P&L: {total_pnl_points:.2f} points (${total_pnl_dollars:.2f})", file=sys.stderr)
        print(f"Avg Win: {avg_win:.2f} pts | Avg Loss: {avg_loss:.2f} pts", file=sys.stderr)
        print(f"Peak Equity: {peak_equity:.2f} pts", file=sys.stderr)
    else:
        print("No trades executed", file=sys.stderr)

    # Save results
    output = {
        "summary": {
            "total_trades": len(all_trades),
            "total_pnl_points": sum(t['pnl_points'] for t in all_trades) if all_trades else 0,
            "total_pnl_dollars": sum(t['pnl_points'] for t in all_trades) * POINT_VALUE if all_trades else 0,
            "win_rate": wins / len(all_trades) * 100 if all_trades else 0,
            "point_value": POINT_VALUE,
        },
        "daily_analyses": all_analyses,
        "trades": all_trades,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
