#!/usr/bin/env python3
"""
No-Whale XGBoost + LSTM Ensemble Backtest

This version uses ONLY TopstepX data (price bars, CVD, Volume Profile).
NO Unusual Whales data (options flow, IV rank, OI changes).

Architecture:
1. Stage 1 XGBoost scans every bar for potential trade setups based on price/CVD/VP
2. Both LSTMs process context (long-term 5min, short-term 1s)
3. Final XGBoost decides whether to take the trade

Trigger: Instead of whale flow, we use CVD divergence + VP levels as trade triggers.
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

# PyTorch for LSTM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Trade timeout in 1-min bars
HOLD_BARS = 20

# LSTM configuration
LONGTERM_SEQ_LEN = 72  # 6 hours of 5-min bars
SHORTTERM_SEQ_LEN = 120  # 2 minutes of 1-second bars
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 2

# Trade trigger settings (replaces whale flow)
MIN_CVD_DIVERGENCE = 0.15  # Minimum CVD vs price divergence to trigger
MIN_BARS_BETWEEN_TRADES = 5  # Don't enter too quickly after a trade

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        period_seconds_val = period_seconds
        total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        period_start = (total_seconds // period_seconds_val) * period_seconds_val
        period_ts = dt.replace(hour=period_start // 3600, minute=(period_start % 3600) // 60, second=period_start % 60, microsecond=0)
        period_key = period_ts.isoformat()

        if period_key not in period_data:
            period_order.append(period_key)

        period_data[period_key]['bars'].append(bar)

        volume = bar.get('v', 0) or 0
        bar_range = bar['h'] - bar['l']
        if bar_range > 0:
            close_position = (bar['c'] - bar['l']) / bar_range
            buy_vol = volume * close_position
            sell_vol = volume * (1 - close_position)
        else:
            if bar['c'] >= bar['o']:
                buy_vol = volume
                sell_vol = 0
            else:
                buy_vol = 0
                sell_vol = volume

        period_data[period_key]['buy_volume'] += buy_vol
        period_data[period_key]['sell_volume'] += sell_vol

        tick_size = 0.25
        price_level = round(bar['c'] / tick_size) * tick_size
        period_data[period_key]['price_volume'][price_level]['buy'] += buy_vol
        period_data[period_key]['price_volume'][price_level]['sell'] += sell_vol

    result = []
    for period_ts in period_order:
        data = period_data[period_ts]
        bars = data['bars']
        if not bars:
            continue

        buy_vol = data['buy_volume']
        sell_vol = data['sell_volume']
        total_vol = buy_vol + sell_vol
        delta = buy_vol - sell_vol

        price_volume = data['price_volume']
        if price_volume:
            poc_price = max(price_volume.keys(), key=lambda p: price_volume[p]['buy'] + price_volume[p]['sell'])
            poc_imbalance = (price_volume[poc_price]['buy'] - price_volume[poc_price]['sell']) / max(price_volume[poc_price]['buy'] + price_volume[poc_price]['sell'], 1)
        else:
            poc_price = bars[-1]['c']
            poc_imbalance = 0

        result.append({
            't': period_ts,
            'delta_pct': delta / total_vol if total_vol > 0 else 0,
            'poc_imbalance': poc_imbalance,
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
# TRADE TRIGGER DETECTION (Replaces Whale Flow)
# =============================================================================

def detect_trade_trigger(
    bars_1min: List[Dict],
    cvd_1min: List[float],
    cvd_ema_1min: List[float],
    vp: Dict[str, float],
    bar_idx: int,
) -> Optional[str]:
    """
    Detect trade trigger based on CVD divergence and VP levels.
    Returns 'bullish', 'bearish', or None.

    This replaces whale flow as the trigger mechanism.
    """
    if bar_idx < 10:
        return None

    current_bar = bars_1min[bar_idx]
    current_price = current_bar['c']

    # Get CVD data
    cvd_now = cvd_1min[bar_idx] if bar_idx < len(cvd_1min) else 0
    cvd_ema_now = cvd_ema_1min[bar_idx] if bar_idx < len(cvd_ema_1min) else 0

    # Calculate price change over last 5 bars
    price_5_ago = bars_1min[bar_idx - 5]['c']
    price_change = (current_price - price_5_ago) / price_5_ago if price_5_ago > 0 else 0

    # Calculate CVD change over last 5 bars
    cvd_5_ago = cvd_1min[bar_idx - 5] if bar_idx >= 5 and bar_idx - 5 < len(cvd_1min) else 0
    cvd_change = cvd_now - cvd_5_ago

    # Normalize CVD change
    cvd_range = max(abs(cvd_1min[bar_idx - 10]), 1) if bar_idx >= 10 else 1
    cvd_change_norm = cvd_change / cvd_range

    # CVD vs EMA divergence
    cvd_vs_ema = cvd_now - cvd_ema_now
    cvd_ema_divergence = cvd_vs_ema / max(abs(cvd_ema_now), 1)

    # VP levels
    price_vs_poc = (current_price - vp['poc']) / vp['poc'] if vp['poc'] > 0 else 0
    price_vs_val = (current_price - vp['val']) / vp['val'] if vp['val'] > 0 else 0
    price_vs_vah = (current_price - vp['vah']) / vp['vah'] if vp['vah'] > 0 else 0

    # BULLISH trigger conditions:
    # 1. Price dropping but CVD rising (bullish divergence) + near VAL
    # 2. Strong CVD above EMA + price at/below POC
    bullish_divergence = price_change < -0.001 and cvd_change_norm > MIN_CVD_DIVERGENCE
    bullish_cvd_strength = cvd_ema_divergence > 0.2 and price_vs_poc < 0.001
    bullish_vp_support = price_vs_val < 0.002 and cvd_change_norm > 0

    if bullish_divergence or bullish_cvd_strength or bullish_vp_support:
        return 'bullish'

    # BEARISH trigger conditions:
    # 1. Price rising but CVD falling (bearish divergence) + near VAH
    # 2. Strong CVD below EMA + price at/above POC
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


# =============================================================================
# FEATURE EXTRACTION (No Whale Data)
# =============================================================================

def extract_longterm_sequence(
    bars_5min: List[Dict],
    cvd_1min: List[float],
    bar_idx_1min: int,
    seq_len: int = 72,
) -> Optional[np.ndarray]:
    """Extract long-term sequence for LSTM (no whale features)."""
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
            fp_delta_pct = fp_data.get('delta_pct', 0)
            fp_poc_imbalance = fp_data.get('poc_imbalance', 0)
        else:
            fp_delta_pct = 0
            fp_poc_imbalance = 0

        features.append([
            o_norm, h_norm, l_norm, range_norm, body_norm,
            vol_log, delta_norm,
            fp_delta_pct, fp_poc_imbalance
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
    trigger_sentiment: str = None,  # From detect_trade_trigger
) -> Optional[Dict[str, float]]:
    """Extract XGBoost features (no whale/IV/OI features)."""
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
        recent_1h = bars_1h[idx_1h]
        bullish_1h = 1 if recent_1h['c'] > recent_1h['o'] else 0
        close_position_1h = (recent_1h['c'] - recent_1h['l']) / (recent_1h['h'] - recent_1h['l']) if recent_1h['h'] > recent_1h['l'] else 0.5
    else:
        bullish_1h = 0.5
        close_position_1h = 0.5

    # Trigger sentiment (replaces whale sentiment)
    trigger_bullish = 1 if trigger_sentiment == 'bullish' else 0
    trigger_bearish = 1 if trigger_sentiment == 'bearish' else 0

    price_change_5 = (current_price - bars_1min[bar_idx-5]['c']) / bars_1min[bar_idx-5]['c'] if bar_idx >= 5 else 0
    price_change_10 = (current_price - bars_1min[bar_idx-10]['c']) / bars_1min[bar_idx-10]['c'] if bar_idx >= 10 else 0
    price_change_20 = (current_price - bars_1min[bar_idx-20]['c']) / bars_1min[bar_idx-20]['c'] if bar_idx >= 20 else 0

    recent_high = max(b['h'] for b in bars_1min[bar_idx-20:bar_idx+1])
    recent_low = min(b['l'] for b in bars_1min[bar_idx-20:bar_idx+1])
    price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5

    # Volume features
    vol_now = current_bar.get('v', 0) or 0
    avg_vol_20 = np.mean([b.get('v', 0) or 0 for b in bars_1min[max(0, bar_idx-20):bar_idx+1]])
    vol_ratio = vol_now / avg_vol_20 if avg_vol_20 > 0 else 1.0

    return {
        'time_sin': time_sin,
        'time_cos': time_cos,
        'is_open_30min': is_open_30min,
        'is_morning': is_morning,
        'is_lunch': is_lunch,
        'is_afternoon': is_afternoon,
        'is_close_30min': is_close_30min,
        'cvd_trend': cvd_trend,
        'cvd_vs_ema': cvd_vs_ema,
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
        'price_change_20': price_change_20,
        'price_position': price_position,
        'vol_ratio': vol_ratio,
    }


# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate_trade(
    bars_1min: List[Dict],
    entry_idx: int,
    direction: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    max_bars: int = 60,
) -> Dict[str, Any]:
    """Simulate a trade with fixed SL/TP."""
    pnl = 0.0
    exit_price = entry_price
    outcome = "timeout"
    bars_held = 0
    exit_reason = "timeout"

    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(bars_1min))):
        bar = bars_1min[i]
        bars_held = i - entry_idx

        if direction == "long":
            if bar['l'] <= stop_loss:
                exit_price = stop_loss
                pnl = stop_loss - entry_price
                outcome = "loss"
                exit_reason = "sl"
                break
            elif bar['h'] >= target:
                exit_price = target
                pnl = target - entry_price
                outcome = "win"
                exit_reason = "tp"
                break
        else:
            if bar['h'] >= stop_loss:
                exit_price = stop_loss
                pnl = entry_price - stop_loss
                outcome = "loss"
                exit_reason = "sl"
                break
            elif bar['l'] <= target:
                exit_price = target
                pnl = entry_price - target
                outcome = "win"
                exit_reason = "tp"
                break

    if outcome == "timeout":
        final_bar = bars_1min[min(entry_idx + max_bars, len(bars_1min) - 1)]
        exit_price = final_bar['c']
        if direction == "long":
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
        outcome = "win" if pnl > 0 else "loss"

    return {
        "outcome": outcome,
        "exit_price": exit_price,
        "pnl_points": pnl,
        "bars_held": bars_held,
        "exit_reason": exit_reason,
    }


# =============================================================================
# TRAINING DATA BUILDERS
# =============================================================================

def train_lstm_models(
    all_bars_1s: Dict[str, List[Dict]],
    train_dates: List[str],
    epochs: int = 10,
) -> Tuple[nn.Module, nn.Module]:
    """Train LSTM models (no whale data)."""
    print("  Training LSTM models...", file=sys.stderr)

    longterm_seqs = []
    longterm_labels = []
    shortterm_seqs = []
    shortterm_labels = []

    for date in train_dates:
        if date not in all_bars_1s:
            continue

        bars_1s = all_bars_1s[date]
        bars_1min, cvd_1min, _ = calculate_cvd_1min(bars_1s)
        bars_5min = aggregate_bars(bars_1s, 5)
        footprint_1min = calculate_footprint_candles(bars_1s, period_seconds=60)

        if len(bars_1min) < 100:
            continue

        for i in range(72, len(bars_5min) - 5, 10):
            seq = extract_longterm_sequence(bars_5min, cvd_1min, i * 5, seq_len=72)
            if seq is not None:
                future_price = bars_5min[i + 5]['c'] if i + 5 < len(bars_5min) else bars_5min[i]['c']
                current_price = bars_5min[i]['c']
                label = 1 if future_price > current_price else 0
                longterm_seqs.append(seq)
                longterm_labels.append(label)

        for i in range(120, len(bars_1s) - 60, 60):
            seq = extract_shortterm_sequence(bars_1s, footprint_1min, i, seq_len=120)
            if seq is not None:
                future_idx = min(i + 60, len(bars_1s) - 1)
                future_price = bars_1s[future_idx]['c']
                current_price = bars_1s[i]['c']
                label = 1 if future_price > current_price else 0
                shortterm_seqs.append(seq)
                shortterm_labels.append(label)

    # 7 input features for longterm (removed whale features)
    longterm_lstm = LongTermLSTM(input_dim=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    shortterm_lstm = ShortTermLSTM(input_dim=9, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)

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

    return longterm_lstm, shortterm_lstm


def train_stage1_xgboost(
    all_bars_1s: Dict[str, List[Dict]],
    train_dates: List[str],
) -> xgb.XGBClassifier:
    """Train Stage 1 XGBoost for direction prediction (no whale data)."""
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
                cvd_1min, cvd_ema_1min, i, vp,
                trigger_sentiment=None
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


def build_final_training_data(
    all_bars_1s: Dict[str, List[Dict]],
    train_dates: List[str],
    stage1_xgb: xgb.XGBClassifier,
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training data for final XGBoost.
    Uses CVD/VP triggers instead of whale flow.
    """
    X_all = []
    y_all = []

    longterm_lstm.eval()
    shortterm_lstm.eval()

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

        # Build bar -> 1s index mapping
        bars_1s_ts_to_idx = {}
        for i, bar in enumerate(bars_1s):
            ts = bar['t']
            if isinstance(ts, str):
                ts = ts.replace('+00:00', '')
            bars_1s_ts_to_idx[ts] = i

        # Scan all bars for triggers
        for bar_idx in range(72, len(bars_1min) - HOLD_BARS, 3):  # Every 3 bars
            vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

            # Detect trigger
            trigger = detect_trade_trigger(bars_1min, cvd_1min, cvd_ema_1min, vp, bar_idx)
            if trigger is None:
                continue

            current_bar = bars_1min[bar_idx]
            current_bar_ts = current_bar['t'].replace('+00:00', '') if isinstance(current_bar['t'], str) else current_bar['t']
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
            stage1_direction = "long" if stage1_pred == 1 else "short"
            stage1_confidence = stage1_proba[stage1_pred]

            # Get LSTM embeddings
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

            shortterm_seq = extract_shortterm_sequence(bars_1s, footprint_1min, bar_1s_idx, seq_len=120)
            if shortterm_seq is None:
                continue

            with torch.no_grad():
                longterm_tensor = torch.tensor(longterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                shortterm_tensor = torch.tensor(shortterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                longterm_emb = longterm_lstm(longterm_tensor).cpu().numpy().flatten()
                shortterm_emb = shortterm_lstm(shortterm_tensor).cpu().numpy().flatten()

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

            # Simulate trade
            result = simulate_trade(bars_1min, bar_idx, stage1_direction, current_price, stop_loss, target, max_bars=HOLD_BARS)

            # Combined features
            combined = np.concatenate([
                np.array(list(xgb_features.values())),
                longterm_emb,
                shortterm_emb,
                np.array([stage1_confidence])
            ])

            # Label: 1 if profitable (take trade), 0 if not (HOLD)
            min_pnl_threshold = 2.0
            if result['pnl_points'] > min_pnl_threshold:
                label = 1  # TAKE THIS TRADE
            else:
                label = 0  # HOLD / SKIP

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
    final_xgb: xgb.XGBClassifier,
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
    starting_equity: float = 0.0,
) -> Tuple[List[Dict], Dict]:
    """Run backtest for a single day using CVD/VP triggers."""

    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
    bars_5min = aggregate_bars(bars_1s, 5)
    bars_15min = aggregate_bars(bars_1s, 15)
    bars_1h = aggregate_bars(bars_1s, 60)
    footprint_1min = calculate_footprint_candles(bars_1s, period_seconds=60)

    if len(bars_1min) < 100:
        return [], {"max_intraday_dd_points": 0, "max_intraday_dd_from_peak_points": 0}, starting_equity, starting_equity

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

    # Drawdown tracking
    intraday_pnl = 0.0
    intraday_peak_pnl = 0.0
    max_intraday_dd = 0.0
    max_intraday_dd_from_peak = 0.0

    cumulative_equity = starting_equity
    peak_equity = starting_equity

    # Process bar by bar
    for bar_idx in range(72, len(bars_1min) - HOLD_BARS):
        # Skip if we're in a position
        if bar_idx < current_position_exit_bar:
            continue

        # Minimum bars between trades
        if trades and bar_idx - trades[-1].get('entry_bar_idx', 0) < MIN_BARS_BETWEEN_TRADES:
            continue

        current_bar = bars_1min[bar_idx]
        current_bar_ts = current_bar['t'].replace('+00:00', '') if isinstance(current_bar['t'], str) else current_bar['t']
        current_price = current_bar['c']

        vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

        # Detect trigger (replaces whale flow)
        trigger = detect_trade_trigger(bars_1min, cvd_1min, cvd_ema_1min, vp, bar_idx)
        if trigger is None:
            continue

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
        stage1_direction = "long" if stage1_pred == 1 else "short"
        stage1_confidence = stage1_proba[stage1_pred]

        # Get LSTM embeddings
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

        shortterm_seq = extract_shortterm_sequence(bars_1s, footprint_1min, bar_1s_idx, seq_len=120)
        if shortterm_seq is None:
            continue

        with torch.no_grad():
            longterm_tensor = torch.tensor(longterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            shortterm_tensor = torch.tensor(shortterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            longterm_emb = longterm_lstm(longterm_tensor).cpu().numpy().flatten()
            shortterm_emb = shortterm_lstm(shortterm_tensor).cpu().numpy().flatten()

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
            np.array([stage1_confidence])
        ])

        X_pred = combined.reshape(1, -1)
        pred = final_xgb.predict(X_pred)[0]
        proba = final_xgb.predict_proba(X_pred)[0]

        # Only take trade if final model says TAKE (pred=1)
        if pred != 1:
            continue

        final_score = proba[1]

        # Execute trade
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

        # Update P&L and drawdowns
        trade_pnl = result["pnl_points"]
        intraday_pnl += trade_pnl
        cumulative_equity += trade_pnl

        # Update intraday peak and drawdowns
        if intraday_pnl > intraday_peak_pnl:
            intraday_peak_pnl = intraday_pnl
        if cumulative_equity > peak_equity:
            peak_equity = cumulative_equity

        # Calculate drawdowns after this trade
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
            "final_score": float(final_score),
            "trigger": trigger,
            "outcome": result["outcome"],
            "pnl_points": result["pnl_points"],
            "exit_reason": result["exit_reason"],
            "cumulative_intraday_pnl": round(intraday_pnl, 2),
            "cumulative_equity": round(cumulative_equity, 2),
        }
        trades.append(trade)

    # Analysis
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
    parser = argparse.ArgumentParser(description='No-Whale XGBoost + LSTM Backtest (TopstepX Data Only)')
    parser.add_argument('--bars', required=True, help='Path to 1s bars JSON')
    args = parser.parse_args()

    print("=" * 70, file=sys.stderr)
    print("NO-WHALE XGBOOST + LSTM ENSEMBLE BACKTEST", file=sys.stderr)
    print("Using ONLY TopstepX data (OHLCV, CVD, Volume Profile)", file=sys.stderr)
    print("NO Unusual Whales data (no options flow, no IV rank, no OI changes)", file=sys.stderr)
    print(f"Device: {DEVICE}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Load all data
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
    print(f"[Load] Found bars for dates: {dates}", file=sys.stderr)

    # Walk-forward backtest
    all_trades = []
    all_analyses = []

    # Track cumulative equity and drawdowns across all days
    cumulative_equity = 0.0
    overall_peak_equity = 0.0
    max_dd_from_overall_peak = 0.0

    for i, test_date in enumerate(dates):
        train_dates = dates[:i]

        if len(train_dates) < 2:
            print(f"\n[Day {i+1}] {test_date}: Not enough training days", file=sys.stderr)
            continue

        print(f"\n[Day {i+1}] {test_date}: Training on {train_dates}", file=sys.stderr)

        # Train Stage 1 XGBoost
        print("  Training Stage 1 XGBoost...", file=sys.stderr)
        stage1_xgb = train_stage1_xgboost(all_bars_1s, train_dates)
        if stage1_xgb is None:
            print("  Not enough data for Stage 1", file=sys.stderr)
            continue

        # Train LSTM models
        longterm_lstm, shortterm_lstm = train_lstm_models(all_bars_1s, train_dates, epochs=10)

        # Build training data for final XGBoost
        X_train, y_train = build_final_training_data(
            all_bars_1s, train_dates,
            stage1_xgb, longterm_lstm, shortterm_lstm
        )

        if len(X_train) < 50:
            print(f"  Not enough training data for final model ({len(X_train)} samples)", file=sys.stderr)
            continue

        print(f"  Training Final XGBoost on {len(X_train)} samples (label balance: {np.mean(y_train):.2%} positive)...", file=sys.stderr)

        # Train final XGBoost (binary: TAKE or HOLD)
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

        # Test on current day
        trades, analysis, new_equity, new_peak = run_single_day(
            all_bars_1s[test_date],
            test_date,
            stage1_xgb,
            final_xgb,
            longterm_lstm,
            shortterm_lstm,
            starting_equity=cumulative_equity
        )

        # Update cumulative tracking
        cumulative_equity = new_equity
        if new_peak > overall_peak_equity:
            overall_peak_equity = new_peak

        # Track max drawdown from overall peak
        dd_from_peak = overall_peak_equity - cumulative_equity
        if dd_from_peak > max_dd_from_overall_peak:
            max_dd_from_overall_peak = dd_from_peak

        all_trades.extend(trades)
        all_analyses.append(analysis)

        if analysis.get('total_trades', 0) > 0:
            dd_info = f", DD: -{analysis['max_intraday_dd_from_peak_points']:.2f} pts"
            print(f"  Results: {analysis['total_trades']} trades, {analysis['win_rate']}% WR, {analysis['total_pnl_points']:+.2f} pts{dd_info}", file=sys.stderr)
        else:
            print(f"  No trades", file=sys.stderr)

    # Overall summary
    print("\n" + "=" * 70, file=sys.stderr)
    print("OVERALL RESULTS (NO WHALE - TopstepX Data Only)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    total_trades = len(all_trades)
    if total_trades > 0:
        total_wins = sum(1 for t in all_trades if t['outcome'] == 'win')
        total_pnl = sum(t['pnl_points'] for t in all_trades)

        print(f"Total trades: {total_trades}", file=sys.stderr)
        print(f"Overall win rate: {total_wins/total_trades*100:.1f}%", file=sys.stderr)
        print(f"Total P&L: {total_pnl:+.2f} points (${total_pnl * 20:+,.0f} @ $20/pt)", file=sys.stderr)
        print(f"Avg P&L per trade: {total_pnl/total_trades:+.2f} points", file=sys.stderr)

        # Drawdown summary
        max_intraday_dd = max(a.get('max_intraday_dd_points', 0) for a in all_analyses) if all_analyses else 0
        max_intraday_dd_from_peak = max(a.get('max_intraday_dd_from_peak_points', 0) for a in all_analyses) if all_analyses else 0

        print(f"\nDRAWDOWN METRICS:", file=sys.stderr)
        print(f"  Max intraday DD (from day start): -{max_intraday_dd:.2f} pts (${max_intraday_dd * 20:,.0f})", file=sys.stderr)
        print(f"  Max intraday DD (from intraday peak): -{max_intraday_dd_from_peak:.2f} pts (${max_intraday_dd_from_peak * 20:,.0f})", file=sys.stderr)
        print(f"  Max DD from overall peak: -{max_dd_from_overall_peak:.2f} pts (${max_dd_from_overall_peak * 20:,.0f})", file=sys.stderr)
        print(f"  Peak equity: {overall_peak_equity:+.2f} pts (${overall_peak_equity * 20:+,.0f})", file=sys.stderr)
        print(f"  Final equity: {cumulative_equity:+.2f} pts (${cumulative_equity * 20:+,.0f})", file=sys.stderr)

        print("\nBy Day:", file=sys.stderr)
        for analysis in all_analyses:
            if analysis.get('total_trades', 0) > 0:
                dd_str = f"DD: -{analysis.get('max_intraday_dd_from_peak_points', 0):.1f}"
                print(f"  {analysis['date']}: {analysis['total_trades']} trades, {analysis['win_rate']}% WR, {analysis['total_pnl_points']:+.2f} pts, {dd_str}", file=sys.stderr)

        longs = sum(1 for t in all_trades if t['direction'] == 'long')
        shorts = sum(1 for t in all_trades if t['direction'] == 'short')
        print(f"\nDecision breakdown: {longs} LONG, {shorts} SHORT", file=sys.stderr)

        # Trigger breakdown
        bullish_triggers = sum(1 for t in all_trades if t['trigger'] == 'bullish')
        bearish_triggers = sum(1 for t in all_trades if t['trigger'] == 'bearish')
        print(f"Trigger breakdown: {bullish_triggers} bullish, {bearish_triggers} bearish", file=sys.stderr)

    print("=" * 70, file=sys.stderr)

    # Save results
    result = {
        "summary": {
            "total_trades": total_trades,
            "total_wins": sum(1 for t in all_trades if t['outcome'] == 'win') if total_trades > 0 else 0,
            "win_rate": round(sum(1 for t in all_trades if t['outcome'] == 'win') / total_trades * 100, 1) if total_trades > 0 else 0,
            "total_pnl_points": round(sum(t['pnl_points'] for t in all_trades), 2) if total_trades > 0 else 0,
            "max_intraday_dd_from_day_start_points": round(max(a.get('max_intraday_dd_points', 0) for a in all_analyses), 2) if all_analyses else 0,
            "max_intraday_dd_from_peak_points": round(max(a.get('max_intraday_dd_from_peak_points', 0) for a in all_analyses), 2) if all_analyses else 0,
            "max_dd_from_overall_peak_points": round(max_dd_from_overall_peak, 2),
            "peak_equity_points": round(overall_peak_equity, 2),
            "final_equity_points": round(cumulative_equity, 2),
        },
        "by_day": all_analyses,
        "trades": all_trades,
    }

    output_path = "data/no_whale_xgb_results.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[Save] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
