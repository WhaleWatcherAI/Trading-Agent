#!/usr/bin/env python3
"""
Whale LSTM First Backtest

Architecture:
1. LSTMs run on EVERY bar creating embeddings (not triggered by whales)
2. XGBoost predicts direction on every bar using LSTM embeddings + features
3. Final XGBoost decides WHEN to trade based on:
   - LSTM embeddings
   - Direction confidence
   - Whale flow features (as input features, not triggers)
   - Market regime features

This creates a much larger candidate pool since we evaluate every bar,
not just bars with whale trades.
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

# Evaluation frequency (not every bar - too slow)
EVAL_EVERY_N_BARS = 1  # Evaluate every bar (can increase for speed)

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


def load_options_flow(filepath: str, symbol: str = 'QQQ') -> pd.DataFrame:
    """Load options flow from CSV."""
    df = pd.read_csv(filepath)
    df = df[df['underlying_symbol'] == symbol].copy()

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        df['datetime'] = pd.to_datetime(df['executed_at'], format='mixed', utc=True)

    if 'bearish_or_bullish' not in df.columns:
        def classify_option_trade(row):
            side = row.get('side', '')
            option_type = row.get('option_type', '')
            if side == 'ask':
                return 'bullish' if option_type == 'call' else 'bearish'
            elif side == 'bid':
                return 'bearish' if option_type == 'call' else 'bullish'
            return 'neutral'
        df['bearish_or_bullish'] = df.apply(classify_option_trade, axis=1)

    if 'date' not in df.columns:
        df['date'] = df['datetime'].dt.strftime('%m/%d/%Y')
    if 'time' not in df.columns:
        df['time'] = df['datetime'].dt.strftime('%I:%M:%S %p')

    df['flow_type'] = 'options'
    return df


def load_iv_rank(filepath: str, target_date: str) -> Dict[str, float]:
    """Load IV rank data."""
    if not os.path.exists(filepath):
        return {'iv_rank_1y': 0.0, 'iv_percentile_1y': 0.0, 'iv_rank_1m': 0.0, 'iv_percentile_1m': 0.0, 'volatility': 0.0}
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    target = pd.to_datetime(target_date)
    prev_day_data = df[df['date'] < target].sort_values('date').tail(1)

    if len(prev_day_data) == 0:
        return {'iv_rank_1y': 0.0, 'iv_percentile_1y': 0.0, 'iv_rank_1m': 0.0, 'iv_percentile_1m': 0.0, 'volatility': 0.0}

    row = prev_day_data.iloc[0]
    return {
        'iv_rank_1y': float(row.get('iv_rank_1y', 0)),
        'iv_percentile_1y': float(row.get('iv_percentile_1y', 0)),
        'iv_rank_1m': float(row.get('iv_rank_1m', 0)),
        'iv_percentile_1m': float(row.get('iv_percentile_1m', 0)),
        'volatility': float(row.get('volatility', 0)),
    }


def load_oi_changes(filepath: str, symbol: str = 'QQQ') -> Dict[str, float]:
    """Load OI changes."""
    if not os.path.exists(filepath):
        return {
            'oi_net_call_put': 0.0, 'oi_total_change': 0.0,
            'oi_call_ratio': 0.5, 'oi_bullish_flow_ratio': 0.5, 'oi_large_positions': 0,
        }
    df = pd.read_csv(filepath)
    df = df[df['underlying_symbol'] == symbol].copy()

    if len(df) == 0:
        return {
            'oi_net_call_put': 0.0, 'oi_total_change': 0.0,
            'oi_call_ratio': 0.5, 'oi_bullish_flow_ratio': 0.5, 'oi_large_positions': 0,
        }

    def is_call(opt_sym):
        return 'C0' in opt_sym or 'C00' in opt_sym

    df['is_call'] = df['option_symbol'].apply(is_call)
    calls = df[df['is_call']]
    puts = df[~df['is_call']]

    call_oi_change = calls['oi_diff_plain'].sum()
    put_oi_change = puts['oi_diff_plain'].sum()
    total_oi_change = call_oi_change + put_oi_change
    net_call_put = call_oi_change - put_oi_change
    call_ratio = call_oi_change / total_oi_change if total_oi_change > 0 else 0.5

    total_ask_vol = df['prev_ask_volume'].sum()
    total_bid_vol = df['prev_bid_volume'].sum()
    total_vol = total_ask_vol + total_bid_vol
    bullish_flow_ratio = total_ask_vol / total_vol if total_vol > 0 else 0.5
    large_positions = len(df[df['oi_diff_plain'].abs() > 5000])

    return {
        'oi_net_call_put': net_call_put / 1e6,
        'oi_total_change': total_oi_change / 1e6,
        'oi_call_ratio': call_ratio,
        'oi_bullish_flow_ratio': bullish_flow_ratio,
        'oi_large_positions': large_positions,
    }


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
# FEATURE EXTRACTION
# =============================================================================

def extract_longterm_sequence(
    bars_5min: List[Dict],
    cvd_1min: List[float],
    whale_trades: List[Dict],
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

        bar_ts = bar['t']
        if isinstance(bar_ts, str):
            bar_dt = datetime.fromisoformat(bar_ts.replace('Z', '+00:00').replace('+00:00', ''))
        else:
            bar_dt = bar_ts

        whale_count = 0
        whale_bullish = 0
        whale_bearish = 0
        for wt in whale_trades:
            wt_dt = wt['datetime']
            if hasattr(wt_dt, 'tz_localize'):
                wt_dt = wt_dt.tz_localize(None)
            time_diff = (wt_dt - bar_dt).total_seconds()
            if 0 <= time_diff < 300:
                whale_count += 1
                if wt.get('bearish_or_bullish') == 'bullish':
                    whale_bullish += 1
                elif wt.get('bearish_or_bullish') == 'bearish':
                    whale_bearish += 1

        whale_sentiment = (whale_bullish - whale_bearish) / max(whale_count, 1)

        features.append([
            o_norm, h_norm, l_norm, range_norm, body_norm,
            vol_log, cvd_change,
            whale_count / 10, whale_sentiment
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


def get_whale_flow_features(
    bar_idx: int,
    bars_1min: List[Dict],
    whale_trades: List[Dict],
    options_premium_threshold: float,
) -> Dict[str, float]:
    """Get whale flow features for the current bar (rolling window)."""
    current_bar = bars_1min[bar_idx]
    current_ts = current_bar['t']
    if isinstance(current_ts, str):
        current_dt = datetime.fromisoformat(current_ts.replace('Z', '+00:00').replace('+00:00', ''))
    else:
        current_dt = current_ts

    # Count whale trades in last 5, 15, 30 minutes
    whale_5min_bullish = 0
    whale_5min_bearish = 0
    whale_15min_bullish = 0
    whale_15min_bearish = 0
    whale_30min_bullish = 0
    whale_30min_bearish = 0
    whale_5min_premium = 0
    whale_15min_premium = 0
    whale_30min_premium = 0

    for wt in whale_trades:
        wt_dt = wt['datetime']
        if hasattr(wt_dt, 'tz_localize'):
            wt_dt = wt_dt.tz_localize(None)

        time_diff = (current_dt - wt_dt).total_seconds()
        if time_diff < 0:
            continue  # Future trade

        premium = wt.get('premium', 0)
        sentiment = wt.get('bearish_or_bullish', 'neutral')

        if time_diff <= 300:  # 5 min
            whale_5min_premium += premium
            if sentiment == 'bullish':
                whale_5min_bullish += 1
            elif sentiment == 'bearish':
                whale_5min_bearish += 1

        if time_diff <= 900:  # 15 min
            whale_15min_premium += premium
            if sentiment == 'bullish':
                whale_15min_bullish += 1
            elif sentiment == 'bearish':
                whale_15min_bearish += 1

        if time_diff <= 1800:  # 30 min
            whale_30min_premium += premium
            if sentiment == 'bullish':
                whale_30min_bullish += 1
            elif sentiment == 'bearish':
                whale_30min_bearish += 1

    # Calculate ratios
    whale_5min_total = whale_5min_bullish + whale_5min_bearish
    whale_15min_total = whale_15min_bullish + whale_15min_bearish
    whale_30min_total = whale_30min_bullish + whale_30min_bearish

    return {
        'whale_5min_count': whale_5min_total,
        'whale_5min_sentiment': (whale_5min_bullish - whale_5min_bearish) / max(whale_5min_total, 1),
        'whale_5min_premium': whale_5min_premium / 1e6,
        'whale_15min_count': whale_15min_total,
        'whale_15min_sentiment': (whale_15min_bullish - whale_15min_bearish) / max(whale_15min_total, 1),
        'whale_15min_premium': whale_15min_premium / 1e6,
        'whale_30min_count': whale_30min_total,
        'whale_30min_sentiment': (whale_30min_bullish - whale_30min_bearish) / max(whale_30min_total, 1),
        'whale_30min_premium': whale_30min_premium / 1e6,
        'whale_has_recent': 1 if whale_5min_total > 0 else 0,
    }


def extract_xgb_features(
    bars_1min: List[Dict],
    bars_5min: List[Dict],
    bars_15min: List[Dict],
    bars_1h: List[Dict],
    cvd_1min: List[float],
    cvd_ema_1min: List[float],
    bar_idx: int,
    vp: Dict,
    iv_rank: Dict[str, float] = None,
    oi_changes: Dict[str, float] = None,
) -> Optional[Dict[str, float]]:
    """Extract XGBoost features (without whale sentiment - that's separate now)."""
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

    price_change_5 = (current_price - bars_1min[bar_idx-5]['c']) / bars_1min[bar_idx-5]['c'] if bar_idx >= 5 else 0
    price_change_10 = (current_price - bars_1min[bar_idx-10]['c']) / bars_1min[bar_idx-10]['c'] if bar_idx >= 10 else 0
    price_change_20 = (current_price - bars_1min[bar_idx-20]['c']) / bars_1min[bar_idx-20]['c'] if bar_idx >= 20 else 0

    recent_high = max(b['h'] for b in bars_1min[bar_idx-20:bar_idx+1])
    recent_low = min(b['l'] for b in bars_1min[bar_idx-20:bar_idx+1])
    price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5

    if iv_rank is not None:
        iv_rank_1y = iv_rank.get('iv_rank_1y', 0) / 100.0
        iv_percentile_1y = iv_rank.get('iv_percentile_1y', 0) / 100.0
        iv_rank_1m = iv_rank.get('iv_rank_1m', 0) / 100.0
        iv_percentile_1m = iv_rank.get('iv_percentile_1m', 0) / 100.0
        volatility = iv_rank.get('volatility', 0)
    else:
        iv_rank_1y = 0.0
        iv_percentile_1y = 0.0
        iv_rank_1m = 0.0
        iv_percentile_1m = 0.0
        volatility = 0.0

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
        'price_change_5': price_change_5,
        'price_change_10': price_change_10,
        'price_change_20': price_change_20,
        'price_position': price_position,
        'iv_rank_1y': iv_rank_1y,
        'iv_percentile_1y': iv_percentile_1y,
        'iv_rank_1m': iv_rank_1m,
        'iv_percentile_1m': iv_percentile_1m,
        'volatility': volatility,
        'oi_net_call_put': oi_changes.get('oi_net_call_put', 0.0) if oi_changes else 0.0,
        'oi_total_change': oi_changes.get('oi_total_change', 0.0) if oi_changes else 0.0,
        'oi_call_ratio': oi_changes.get('oi_call_ratio', 0.5) if oi_changes else 0.5,
        'oi_bullish_flow_ratio': oi_changes.get('oi_bullish_flow_ratio', 0.5) if oi_changes else 0.5,
        'oi_large_positions': oi_changes.get('oi_large_positions', 0) if oi_changes else 0,
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
    all_options: Dict[str, pd.DataFrame],
    train_dates: List[str],
    options_premium_threshold: float,
    epochs: int = 10,
) -> Tuple[nn.Module, nn.Module]:
    """Train LSTM models."""
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

        whale_trades = []
        if date in all_options:
            opt_df = all_options[date]
            opt_date_filter = opt_df['datetime'].dt.strftime('%Y-%m-%d') == date
            opt_premium_filter = opt_df['premium'] >= options_premium_threshold
            filtered = opt_df[opt_date_filter & opt_premium_filter]
            whale_trades = filtered.to_dict('records')

        for i in range(72, len(bars_5min) - 5, 10):
            seq = extract_longterm_sequence(bars_5min, cvd_1min, whale_trades, i * 5, seq_len=72)
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

    longterm_lstm = LongTermLSTM(input_dim=9, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)
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


def train_direction_xgboost(
    all_bars_1s: Dict[str, List[Dict]],
    all_options: Dict[str, pd.DataFrame],
    all_iv_ranks: Dict[str, Dict],
    all_oi_changes: Dict[str, Dict],
    train_dates: List[str],
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
    options_premium_threshold: float,
) -> xgb.XGBClassifier:
    """Train direction XGBoost using LSTM embeddings."""
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

        iv_rank = all_iv_ranks.get(date, None)
        oi_changes = all_oi_changes.get(date, None)

        whale_trades = []
        if date in all_options:
            opt_df = all_options[date]
            filtered = opt_df[opt_df['premium'] >= options_premium_threshold]
            whale_trades = filtered.to_dict('records')

        # Build 1s index mapping
        bars_1s_ts_to_idx = {}
        for i, bar in enumerate(bars_1s):
            ts = bar['t']
            if isinstance(ts, str):
                ts = ts.replace('+00:00', '')
            bars_1s_ts_to_idx[ts] = i

        # Sample every N bars for training efficiency
        for bar_idx in range(72, len(bars_1min) - HOLD_BARS, 5):
            vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)
            features = extract_xgb_features(
                bars_1min, bars_5min, bars_15min, bars_1h,
                cvd_1min, cvd_ema_1min, bar_idx, vp,
                iv_rank, oi_changes
            )

            if features is None:
                continue

            # Get LSTM embeddings
            longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, whale_trades, bar_idx, seq_len=72)
            if longterm_seq is None:
                continue

            # Find corresponding 1s bar
            current_bar = bars_1min[bar_idx]
            minute_str = current_bar['t'][:16] if isinstance(current_bar['t'], str) else current_bar['t'].strftime('%Y-%m-%dT%H:%M')

            bar_1s_idx = None
            for ts_key, idx in bars_1s_ts_to_idx.items():
                if ts_key.startswith(minute_str):
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

            # Get whale flow features
            whale_features = get_whale_flow_features(bar_idx, bars_1min, whale_trades, options_premium_threshold)

            # Combine all features
            combined = np.concatenate([
                np.array(list(features.values())),
                longterm_emb,
                shortterm_emb,
                np.array(list(whale_features.values()))
            ])

            current_price = bars_1min[bar_idx]['c']
            future_price = bars_1min[min(bar_idx + HOLD_BARS, len(bars_1min) - 1)]['c']
            label = 1 if future_price > current_price else 0

            X_all.append(combined)
            y_all.append(label)

    if len(X_all) < 50:
        return None

    print(f"  Training Direction XGBoost on {len(X_all)} samples...", file=sys.stderr)
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
    all_options: Dict[str, pd.DataFrame],
    all_iv_ranks: Dict[str, Dict],
    all_oi_changes: Dict[str, Dict],
    train_dates: List[str],
    options_premium_threshold: float,
    direction_xgb: xgb.XGBClassifier,
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build training data for final XGBoost (when to trade)."""
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

        iv_rank = all_iv_ranks.get(date, None)
        oi_changes = all_oi_changes.get(date, None)

        whale_trades = []
        if date in all_options:
            opt_df = all_options[date]
            filtered = opt_df[opt_df['premium'] >= options_premium_threshold]
            whale_trades = filtered.to_dict('records')

        bars_1s_ts_to_idx = {}
        for i, bar in enumerate(bars_1s):
            ts = bar['t']
            if isinstance(ts, str):
                ts = ts.replace('+00:00', '')
            bars_1s_ts_to_idx[ts] = i

        # Sample bars for training
        for bar_idx in range(72, len(bars_1min) - HOLD_BARS, 3):
            vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)
            features = extract_xgb_features(
                bars_1min, bars_5min, bars_15min, bars_1h,
                cvd_1min, cvd_ema_1min, bar_idx, vp,
                iv_rank, oi_changes
            )

            if features is None:
                continue

            longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, whale_trades, bar_idx, seq_len=72)
            if longterm_seq is None:
                continue

            current_bar = bars_1min[bar_idx]
            minute_str = current_bar['t'][:16] if isinstance(current_bar['t'], str) else current_bar['t'].strftime('%Y-%m-%dT%H:%M')

            bar_1s_idx = None
            for ts_key, idx in bars_1s_ts_to_idx.items():
                if ts_key.startswith(minute_str):
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

            whale_features = get_whale_flow_features(bar_idx, bars_1min, whale_trades, options_premium_threshold)

            # Get direction prediction
            combined_dir = np.concatenate([
                np.array(list(features.values())),
                longterm_emb,
                shortterm_emb,
                np.array(list(whale_features.values()))
            ])

            dir_pred = direction_xgb.predict(combined_dir.reshape(1, -1))[0]
            dir_proba = direction_xgb.predict_proba(combined_dir.reshape(1, -1))[0]
            direction = "long" if dir_pred == 1 else "short"
            dir_confidence = dir_proba[dir_pred]

            current_price = bars_1min[bar_idx]['c']

            # Calculate SL/TP
            swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:bar_idx//5+1], lookback=10)

            if direction == "long":
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
            result = simulate_trade(bars_1min, bar_idx, direction, current_price, stop_loss, target, max_bars=HOLD_BARS)

            # Final features include direction confidence
            combined_final = np.concatenate([
                combined_dir,
                np.array([dir_confidence])
            ])

            # Label: profitable trade = 1, else 0
            min_pnl_threshold = 2.0
            if result['pnl_points'] > min_pnl_threshold:
                label = 1
            else:
                label = 0

            X_all.append(combined_final)
            y_all.append(label)

    return np.array(X_all), np.array(y_all)


# =============================================================================
# SINGLE DAY BACKTEST
# =============================================================================

def run_single_day(
    bars_1s: List[Dict],
    options_df: Optional[pd.DataFrame],
    target_date: str,
    direction_xgb: xgb.XGBClassifier,
    final_xgb: xgb.XGBClassifier,
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
    iv_rank: Dict[str, float],
    oi_changes: Dict[str, float],
    options_premium_threshold: float,
) -> Tuple[List[Dict], Dict]:
    """Run backtest for a single day - evaluating every bar."""

    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
    bars_5min = aggregate_bars(bars_1s, 5)
    bars_15min = aggregate_bars(bars_1s, 15)
    bars_1h = aggregate_bars(bars_1s, 60)
    footprint_1min = calculate_footprint_candles(bars_1s, period_seconds=60)

    if len(bars_1min) < 100:
        return [], {}

    bars_1s_ts_to_idx = {}
    for i, bar in enumerate(bars_1s):
        ts = bar['t']
        if isinstance(ts, str):
            ts = ts.replace('+00:00', '')
        bars_1s_ts_to_idx[ts] = i

    whale_trades = []
    if options_df is not None:
        opt_date_filter = options_df['datetime'].dt.strftime('%Y-%m-%d') == target_date
        opt_premium_filter = options_df['premium'] >= options_premium_threshold
        filtered = options_df[opt_date_filter & opt_premium_filter]
        whale_trades = filtered.to_dict('records')

    trades = []
    current_position_exit_bar = 0

    longterm_lstm.eval()
    shortterm_lstm.eval()

    # Evaluate every bar
    for bar_idx in range(72, len(bars_1min) - HOLD_BARS, EVAL_EVERY_N_BARS):
        # Skip if we're in a position
        if bar_idx < current_position_exit_bar:
            continue

        current_bar = bars_1min[bar_idx]
        current_bar_ts = current_bar['t'].replace('+00:00', '') if isinstance(current_bar['t'], str) else current_bar['t']
        current_price = current_bar['c']

        vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)
        features = extract_xgb_features(
            bars_1min, bars_5min, bars_15min, bars_1h,
            cvd_1min, cvd_ema_1min, bar_idx, vp,
            iv_rank, oi_changes
        )

        if features is None:
            continue

        longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, whale_trades, bar_idx, seq_len=72)
        if longterm_seq is None:
            continue

        minute_str = current_bar_ts[:16]
        bar_1s_idx = None
        for ts_key, idx in bars_1s_ts_to_idx.items():
            if ts_key.startswith(minute_str):
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

        whale_features = get_whale_flow_features(bar_idx, bars_1min, whale_trades, options_premium_threshold)

        # Direction prediction
        combined_dir = np.concatenate([
            np.array(list(features.values())),
            longterm_emb,
            shortterm_emb,
            np.array(list(whale_features.values()))
        ])

        dir_pred = direction_xgb.predict(combined_dir.reshape(1, -1))[0]
        dir_proba = direction_xgb.predict_proba(combined_dir.reshape(1, -1))[0]
        direction = "long" if dir_pred == 1 else "short"
        dir_confidence = dir_proba[dir_pred]

        # Final prediction (should we trade?)
        combined_final = np.concatenate([combined_dir, np.array([dir_confidence])])
        final_pred = final_xgb.predict(combined_final.reshape(1, -1))[0]
        final_proba = final_xgb.predict_proba(combined_final.reshape(1, -1))[0]

        # Only trade if final model says yes
        if final_pred != 1:
            continue

        final_score = final_proba[1]

        # Calculate SL/TP
        swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:bar_idx//5+1], lookback=10)

        if direction == "long":
            valid_lows = [sl for sl in swing_lows if sl < current_price] if swing_lows else []
            stop_loss = min(valid_lows) - 1 if valid_lows else current_price - 10
            risk = current_price - stop_loss
            target = current_price + (risk * 3)
        else:
            valid_highs = [sh for sh in swing_highs if sh > current_price] if swing_highs else []
            stop_loss = max(valid_highs) + 1 if valid_highs else current_price + 10
            risk = stop_loss - current_price
            target = current_price - (risk * 3)

        # Execute trade
        result = simulate_trade(bars_1min, bar_idx, direction, current_price, stop_loss, target, max_bars=HOLD_BARS)

        current_position_exit_bar = bar_idx + result["bars_held"]

        trade = {
            "date": target_date,
            "timestamp": current_bar_ts[:19],
            "direction": direction,
            "entry_price": current_price,
            "dir_confidence": float(dir_confidence),
            "final_score": float(final_score),
            "whale_5min_count": whale_features['whale_5min_count'],
            "whale_5min_sentiment": whale_features['whale_5min_sentiment'],
            "outcome": result["outcome"],
            "pnl_points": result["pnl_points"],
            "exit_reason": result["exit_reason"],
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
        }
    else:
        analysis = {"date": target_date, "total_trades": 0}

    return trades, analysis


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Whale LSTM First Backtest')
    parser.add_argument('--bars', required=True, help='Path to 1s bars JSON')
    parser.add_argument('--options-dir', default='data/options_flow', help='Directory with options flow CSVs')
    parser.add_argument('--options-file', default=None, help='Single options flow CSV file')
    parser.add_argument('--ivrank', default='data/iv_rank_history.csv', help='Path to IV rank CSV')
    parser.add_argument('--oi-dir', default='data/oi_changes', help='Directory with OI changes CSVs')
    parser.add_argument('--options-premium', type=float, default=322040, help='Options premium threshold')
    args = parser.parse_args()

    print("=" * 70, file=sys.stderr)
    print("WHALE LSTM FIRST BACKTEST", file=sys.stderr)
    print("(LSTMs on all bars → Direction XGB → Final XGB decides when to trade)", file=sys.stderr)
    print(f"Options premium threshold: ${args.options_premium:,.0f}", file=sys.stderr)
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

    # Load options flow
    print("[Load] Loading options flow...", file=sys.stderr)
    all_options = {}

    if args.options_file and os.path.exists(args.options_file):
        print(f"  Loading from {args.options_file}...", file=sys.stderr)
        full_df = load_options_flow(args.options_file, symbol='QQQ')
        for date in dates:
            date_filter = full_df['datetime'].dt.strftime('%Y-%m-%d') == date
            df = full_df[date_filter].copy()
            whale_count = len(df[df['premium'] >= args.options_premium])
            if len(df) > 0:
                print(f"  {date}: {len(df)} QQQ options ({whale_count} whale)", file=sys.stderr)
                all_options[date] = df
    else:
        for date in dates:
            date_formatted = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
            possible_files = [
                os.path.join(args.options_dir, f'qqq_options_flow_{date_formatted}.csv'),
                os.path.join(args.options_dir, f'bot-eod-report-{date_formatted}.csv'),
            ]
            for options_file in possible_files:
                if os.path.exists(options_file):
                    df = load_options_flow(options_file, symbol='QQQ')
                    whale_count = len(df[df['premium'] >= args.options_premium])
                    if len(df) > 0:
                        print(f"  {date}: {len(df)} QQQ options ({whale_count} whale)", file=sys.stderr)
                        all_options[date] = df
                    break

    # Load IV rank
    print("[Load] Loading IV rank data...", file=sys.stderr)
    all_iv_ranks = {}
    for date in dates:
        all_iv_ranks[date] = load_iv_rank(args.ivrank, date)

    # Load OI changes
    print("[Load] Loading OI changes data...", file=sys.stderr)
    all_oi_changes = {}
    for date in dates:
        prev_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        for d in [prev_date, date]:
            oi_file = os.path.join(args.oi_dir, f'qqq_oi_changes_{d}.csv')
            if os.path.exists(oi_file):
                all_oi_changes[date] = load_oi_changes(oi_file, symbol='QQQ')
                break

    # Walk-forward backtest
    all_trades = []
    all_analyses = []

    for i, test_date in enumerate(dates):
        train_dates = dates[:i]

        if len(train_dates) < 2:
            print(f"\n[Day {i+1}] {test_date}: Not enough training days", file=sys.stderr)
            continue

        print(f"\n[Day {i+1}] {test_date}: Training on {train_dates}", file=sys.stderr)

        # Train LSTM models FIRST
        longterm_lstm, shortterm_lstm = train_lstm_models(
            all_bars_1s, all_options, train_dates, args.options_premium, epochs=10
        )

        # Train Direction XGBoost (uses LSTM embeddings)
        direction_xgb = train_direction_xgboost(
            all_bars_1s, all_options, all_iv_ranks, all_oi_changes,
            train_dates, longterm_lstm, shortterm_lstm, args.options_premium
        )
        if direction_xgb is None:
            print("  Not enough data for Direction XGBoost", file=sys.stderr)
            continue

        # Build training data for final XGBoost
        X_train, y_train = build_final_training_data(
            all_bars_1s, all_options, all_iv_ranks, all_oi_changes,
            train_dates, args.options_premium,
            direction_xgb, longterm_lstm, shortterm_lstm
        )

        if len(X_train) < 50:
            print(f"  Not enough training data for final model ({len(X_train)} samples)", file=sys.stderr)
            continue

        print(f"  Training Final XGBoost on {len(X_train)} samples (label balance: {np.mean(y_train):.2%} positive)...", file=sys.stderr)

        # Train final XGBoost
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
        trades, analysis = run_single_day(
            all_bars_1s[test_date],
            all_options.get(test_date, None),
            test_date,
            direction_xgb,
            final_xgb,
            longterm_lstm,
            shortterm_lstm,
            all_iv_ranks.get(test_date, None),
            all_oi_changes.get(test_date, None),
            args.options_premium
        )

        all_trades.extend(trades)
        all_analyses.append(analysis)

        if analysis.get('total_trades', 0) > 0:
            print(f"  Results: {analysis['total_trades']} trades, {analysis['win_rate']}% WR, {analysis['total_pnl_points']:+.2f} pts", file=sys.stderr)
        else:
            print(f"  No trades", file=sys.stderr)

    # Overall summary
    print("\n" + "=" * 70, file=sys.stderr)
    print("OVERALL RESULTS (WHALE LSTM FIRST)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    total_trades = len(all_trades)
    if total_trades > 0:
        total_wins = sum(1 for t in all_trades if t['outcome'] == 'win')
        total_pnl = sum(t['pnl_points'] for t in all_trades)

        print(f"Total trades: {total_trades}", file=sys.stderr)
        print(f"Overall win rate: {total_wins/total_trades*100:.1f}%", file=sys.stderr)
        print(f"Total P&L: {total_pnl:+.2f} points (${total_pnl * 20:+,.0f} @ $20/pt)", file=sys.stderr)
        print(f"Avg P&L per trade: {total_pnl/total_trades:+.2f} points", file=sys.stderr)

        print("\nBy Day:", file=sys.stderr)
        for analysis in all_analyses:
            if analysis.get('total_trades', 0) > 0:
                print(f"  {analysis['date']}: {analysis['total_trades']} trades, {analysis['win_rate']}% WR, {analysis['total_pnl_points']:+.2f} pts", file=sys.stderr)

        longs = sum(1 for t in all_trades if t['direction'] == 'long')
        shorts = sum(1 for t in all_trades if t['direction'] == 'short')
        print(f"\nDecision breakdown: {longs} LONG, {shorts} SHORT", file=sys.stderr)

    print("=" * 70, file=sys.stderr)

    # Save results
    result = {
        "summary": {
            "total_trades": total_trades,
            "total_wins": sum(1 for t in all_trades if t['outcome'] == 'win') if total_trades > 0 else 0,
            "win_rate": round(sum(1 for t in all_trades if t['outcome'] == 'win') / total_trades * 100, 1) if total_trades > 0 else 0,
            "total_pnl_points": round(sum(t['pnl_points'] for t in all_trades), 2) if total_trades > 0 else 0,
        },
        "by_day": all_analyses,
        "trades": all_trades,
    }

    output_path = "data/whale_lstm_first_results.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[Save] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
