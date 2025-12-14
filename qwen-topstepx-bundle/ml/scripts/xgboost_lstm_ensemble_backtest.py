#!/usr/bin/env python3
"""
Multi-Stage XGBoost + LSTM Ensemble Backtest

Architecture:
1. XGBoost Trigger - Fires on whale options flow (>$322k premium)
2. Long-term LSTM - 6-hour context (whale trades, CVD, 5m/15m/1h candles) -> 32-dim embedding
3. Short-term LSTM - 2-minute microstructure (1s/1m bars, footprint candles) -> 32-dim embedding
4. Final XGBoost - Combines embeddings + original features -> LONG/SHORT/HOLD decision
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
LONGTERM_SEQ_LEN = 360  # 6 hours of 1-min bars
SHORTTERM_SEQ_LEN = 120  # 2 minutes of 1-second bars
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 2

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DATA LOADING FUNCTIONS (from original script)
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
    """
    Calculate footprint candle data - volume at price levels within each candle.
    Returns bid/ask imbalance, delta, and volume profile per candle.
    """
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

        # Round to period
        period_seconds_val = period_seconds
        total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        period_start = (total_seconds // period_seconds_val) * period_seconds_val
        period_ts = dt.replace(hour=period_start // 3600, minute=(period_start % 3600) // 60, second=period_start % 60, microsecond=0)
        period_key = period_ts.isoformat()

        if period_key not in period_data:
            period_order.append(period_key)

        period_data[period_key]['bars'].append(bar)

        # Estimate buy/sell volume from close position
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

        # Add to price level (rounded to tick)
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

        # Find POC (Point of Control) - price with most volume
        price_volume = data['price_volume']
        if price_volume:
            poc_price = max(price_volume.keys(), key=lambda p: price_volume[p]['buy'] + price_volume[p]['sell'])
            poc_imbalance = (price_volume[poc_price]['buy'] - price_volume[poc_price]['sell']) / max(price_volume[poc_price]['buy'] + price_volume[poc_price]['sell'], 1)
        else:
            poc_price = bars[-1]['c']
            poc_imbalance = 0

        result.append({
            't': period_ts,
            'o': bars[0]['o'],
            'h': max(b['h'] for b in bars),
            'l': min(b['l'] for b in bars),
            'c': bars[-1]['c'],
            'v': total_vol,
            'delta': delta,
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
            'delta_pct': delta / total_vol if total_vol > 0 else 0,
            'poc_price': poc_price,
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
    """
    Long-term LSTM for 6-hour context.
    Inputs: 5min candles (OHLCV), CVD, whale trade events, hourly structure
    Output: 32-dim embedding
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, embedding_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        embedding = self.fc(last_hidden)
        embedding = self.layer_norm(embedding)
        return embedding


class ShortTermLSTM(nn.Module):
    """
    Short-term LSTM for 2-minute microstructure.
    Inputs: 1-second bars, 1-min bars, footprint candle data
    Output: 32-dim embedding
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, embedding_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
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
    seq_len: int = 72,  # 72 * 5min = 6 hours
) -> Optional[np.ndarray]:
    """
    Extract long-term sequence for LSTM.
    Features per 5-min bar:
    - OHLCV normalized
    - CVD change
    - Whale trade count/sentiment in this period
    - Price momentum
    """
    bar_idx_5min = bar_idx_1min // 5
    if bar_idx_5min < seq_len:
        return None

    features = []
    for i in range(bar_idx_5min - seq_len, bar_idx_5min):
        bar = bars_5min[i]

        # Normalize OHLCV relative to close
        close = bar['c']
        if close == 0:
            close = 1

        # Price features
        o_norm = (bar['o'] - close) / close
        h_norm = (bar['h'] - close) / close
        l_norm = (bar['l'] - close) / close
        range_norm = (bar['h'] - bar['l']) / close
        body_norm = (bar['c'] - bar['o']) / close

        # Volume (log normalized)
        vol = bar.get('v', 0) or 0
        vol_log = np.log1p(vol) / 10  # Normalize

        # CVD change over this 5-min period
        cvd_start_idx = i * 5
        cvd_end_idx = min((i + 1) * 5, len(cvd_1min))
        if cvd_end_idx > cvd_start_idx and cvd_end_idx <= len(cvd_1min):
            cvd_change = (cvd_1min[cvd_end_idx - 1] - cvd_1min[cvd_start_idx]) / max(abs(cvd_1min[cvd_end_idx - 1]), 1)
        else:
            cvd_change = 0

        # Whale trades in this period
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
            if 0 <= time_diff < 300:  # Within this 5-min bar
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
    seq_len: int = 120,  # 120 seconds = 2 minutes
) -> Optional[np.ndarray]:
    """
    Extract short-term sequence for LSTM.
    Features per 1-second bar:
    - OHLCV normalized
    - Delta (buy - sell estimate)
    - Microstructure features
    """
    if bar_idx_1s < seq_len:
        return None

    features = []
    for i in range(bar_idx_1s - seq_len, bar_idx_1s):
        bar = bars_1s[i]

        close = bar['c']
        if close == 0:
            close = 1

        # Price features
        o_norm = (bar['o'] - close) / close
        h_norm = (bar['h'] - close) / close
        l_norm = (bar['l'] - close) / close
        range_norm = (bar['h'] - bar['l']) / close
        body_norm = (bar['c'] - bar['o']) / close

        # Volume
        vol = bar.get('v', 0) or 0
        vol_log = np.log1p(vol) / 5

        # Estimate delta (buy - sell)
        bar_range = bar['h'] - bar['l']
        if bar_range > 0:
            close_position = (bar['c'] - bar['l']) / bar_range
            delta = vol * (2 * close_position - 1)
        else:
            delta = vol if bar['c'] >= bar['o'] else -vol
        delta_norm = delta / max(vol, 1)

        # Footprint candle data (find matching minute)
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
    whale_sentiment: str = None,
    iv_rank: Dict[str, float] = None,
    oi_changes: Dict[str, float] = None,
) -> Optional[Dict[str, float]]:
    """Extract original XGBoost features."""
    if bar_idx < 30:
        return None

    current_bar = bars_1min[bar_idx]
    current_price = current_bar['c']

    # Time features
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

    # CVD features
    cvd_now = cvd_1min[bar_idx] if bar_idx < len(cvd_1min) else 0
    cvd_ema_now = cvd_ema_1min[bar_idx] if bar_idx < len(cvd_ema_1min) else 0
    cvd_vs_ema = cvd_now - cvd_ema_now
    cvd_trend = 1 if cvd_now > cvd_ema_now else -1 if cvd_now < cvd_ema_now else 0
    cvd_slope_5 = (cvd_1min[bar_idx] - cvd_1min[bar_idx-5]) if bar_idx >= 5 else 0
    cvd_slope_10 = (cvd_1min[bar_idx] - cvd_1min[bar_idx-10]) if bar_idx >= 10 else 0

    # Volume Profile features
    price_vs_poc = (current_price - vp['poc']) / vp['poc'] if vp['poc'] > 0 else 0
    price_vs_vah = (current_price - vp['vah']) / vp['vah'] if vp['vah'] > 0 else 0
    price_vs_val = (current_price - vp['val']) / vp['val'] if vp['val'] > 0 else 0
    in_value_area = 1 if vp['val'] <= current_price <= vp['vah'] else 0
    above_poc = 1 if current_price > vp['poc'] else 0

    # 1-min candle features
    recent_1m = bars_1min[bar_idx-4:bar_idx+1]
    bullish_1m = sum(1 for b in recent_1m if b['c'] > b['o']) / 5
    body_size_1m = np.mean([abs(b['c'] - b['o']) for b in recent_1m])
    range_1m = np.mean([b['h'] - b['l'] for b in recent_1m])
    close_position_1m = np.mean([(b['c'] - b['l']) / (b['h'] - b['l']) if b['h'] > b['l'] else 0.5 for b in recent_1m])

    # 5-min candle features
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

    # 15-min candle features
    idx_15m = bar_idx // 15
    if idx_15m >= 2 and idx_15m < len(bars_15min):
        recent_15m = bars_15min[idx_15m-1:idx_15m+1]
        bullish_15m = sum(1 for b in recent_15m if b['c'] > b['o']) / 2
        close_position_15m = np.mean([(b['c'] - b['l']) / (b['h'] - b['l']) if b['h'] > b['l'] else 0.5 for b in recent_15m])
    else:
        bullish_15m = 0.5
        close_position_15m = 0.5

    # 1-hour candle features
    idx_1h = bar_idx // 60
    if idx_1h >= 1 and idx_1h < len(bars_1h):
        recent_1h = bars_1h[idx_1h]
        bullish_1h = 1 if recent_1h['c'] > recent_1h['o'] else 0
        close_position_1h = (recent_1h['c'] - recent_1h['l']) / (recent_1h['h'] - recent_1h['l']) if recent_1h['h'] > recent_1h['l'] else 0.5
    else:
        bullish_1h = 0.5
        close_position_1h = 0.5

    # Whale sentiment
    whale_bullish = 1 if whale_sentiment == 'bullish' else 0
    whale_bearish = 1 if whale_sentiment == 'bearish' else 0

    # Price momentum
    price_change_5 = (current_price - bars_1min[bar_idx-5]['c']) / bars_1min[bar_idx-5]['c'] if bar_idx >= 5 else 0
    price_change_10 = (current_price - bars_1min[bar_idx-10]['c']) / bars_1min[bar_idx-10]['c'] if bar_idx >= 10 else 0
    price_change_20 = (current_price - bars_1min[bar_idx-20]['c']) / bars_1min[bar_idx-20]['c'] if bar_idx >= 20 else 0

    # Recent high/low position
    recent_high = max(b['h'] for b in bars_1min[bar_idx-20:bar_idx+1])
    recent_low = min(b['l'] for b in bars_1min[bar_idx-20:bar_idx+1])
    price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5

    # IV rank features
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
        'whale_bullish': whale_bullish,
        'whale_bearish': whale_bearish,
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

def build_training_data_with_lstm(
    all_bars_1s: Dict[str, List[Dict]],
    all_options: Dict[str, pd.DataFrame],
    all_iv_ranks: Dict[str, Dict],
    all_oi_changes: Dict[str, Dict],
    train_dates: List[str],
    options_premium_threshold: float,
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training data for final XGBoost decision model.
    Combines: XGBoost features + LSTM embeddings
    Labels: 0=HOLD, 1=LONG, 2=SHORT
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

        iv_rank = all_iv_ranks.get(date, None)
        oi_changes = all_oi_changes.get(date, None)

        # Get whale trades for this day
        whale_trades = []
        if date in all_options:
            opt_df = all_options[date]
            opt_date_filter = opt_df['datetime'].dt.strftime('%Y-%m-%d') == date
            opt_premium_filter = opt_df['premium'] >= options_premium_threshold
            filtered = opt_df[opt_date_filter & opt_premium_filter]
            whale_trades = filtered.to_dict('records')

        # Map 1s bars to indices
        bars_1s_ts_to_idx = {}
        for i, bar in enumerate(bars_1s):
            ts = bar['t']
            if isinstance(ts, str):
                ts = ts.replace('+00:00', '')
            bars_1s_ts_to_idx[ts] = i

        # Sample every 5 bars for training
        for i in range(max(72, 50), len(bars_1min) - HOLD_BARS, 5):
            vp = calculate_volume_profile(bars_1min[:i+1], lookback=30)
            xgb_features = extract_xgb_features(
                bars_1min, bars_5min, bars_15min, bars_1h,
                cvd_1min, cvd_ema_1min, i, vp,
                whale_sentiment=None, iv_rank=iv_rank, oi_changes=oi_changes
            )

            if xgb_features is None:
                continue

            # Get LSTM embeddings
            longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, whale_trades, i, seq_len=72)
            if longterm_seq is None:
                continue

            # Find corresponding 1s bar index
            minute_ts = bars_1min[i]['t']
            if isinstance(minute_ts, str):
                minute_ts = minute_ts.replace('+00:00', '')
            # Find 1s bar at this minute
            bar_1s_idx = None
            for ts_key, idx in bars_1s_ts_to_idx.items():
                if ts_key.startswith(minute_ts[:16]):  # Match up to minute
                    bar_1s_idx = idx
                    break

            if bar_1s_idx is None or bar_1s_idx < 120:
                continue

            shortterm_seq = extract_shortterm_sequence(bars_1s, footprint_1min, bar_1s_idx, seq_len=120)
            if shortterm_seq is None:
                continue

            # Get embeddings
            with torch.no_grad():
                longterm_tensor = torch.tensor(longterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                shortterm_tensor = torch.tensor(shortterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                longterm_emb = longterm_lstm(longterm_tensor).cpu().numpy().flatten()
                shortterm_emb = shortterm_lstm(shortterm_tensor).cpu().numpy().flatten()

            # Combine features
            xgb_feat_array = np.array(list(xgb_features.values()))
            combined = np.concatenate([xgb_feat_array, longterm_emb, shortterm_emb])

            # Calculate label: simulate trade in both directions
            current_price = bars_1min[i]['c']
            swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:i//5+1], lookback=10)

            # Long trade
            valid_lows = [sl for sl in swing_lows if sl < current_price] if swing_lows else []
            if valid_lows:
                long_sl = min(valid_lows) - 1
            else:
                long_sl = current_price - 10
            long_risk = current_price - long_sl
            long_tp = current_price + (long_risk * 3)
            long_result = simulate_trade(bars_1min, i, "long", current_price, long_sl, long_tp, max_bars=HOLD_BARS)

            # Short trade
            valid_highs = [sh for sh in swing_highs if sh > current_price] if swing_highs else []
            if valid_highs:
                short_sl = max(valid_highs) + 1
            else:
                short_sl = current_price + 10
            short_risk = short_sl - current_price
            short_tp = current_price - (short_risk * 3)
            short_result = simulate_trade(bars_1min, i, "short", current_price, short_sl, short_tp, max_bars=HOLD_BARS)

            # Determine best action
            long_pnl = long_result['pnl_points']
            short_pnl = short_result['pnl_points']

            # Label: 0=HOLD (both lose or small gain), 1=LONG (best), 2=SHORT (best)
            min_pnl_threshold = 2.0  # Need at least 2 points to be worthwhile

            if long_pnl > short_pnl and long_pnl > min_pnl_threshold:
                label = 1  # LONG
            elif short_pnl > long_pnl and short_pnl > min_pnl_threshold:
                label = 2  # SHORT
            else:
                label = 0  # HOLD

            X_all.append(combined)
            y_all.append(label)

    return np.array(X_all), np.array(y_all)


def train_lstm_models(
    all_bars_1s: Dict[str, List[Dict]],
    all_options: Dict[str, pd.DataFrame],
    train_dates: List[str],
    options_premium_threshold: float,
    epochs: int = 10,
) -> Tuple[nn.Module, nn.Module]:
    """
    Train LSTM models using self-supervised learning on price prediction.
    """
    print("  Training LSTM models...", file=sys.stderr)

    # Collect sequences for training
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

        # Get whale trades
        whale_trades = []
        if date in all_options:
            opt_df = all_options[date]
            opt_date_filter = opt_df['datetime'].dt.strftime('%Y-%m-%d') == date
            opt_premium_filter = opt_df['premium'] >= options_premium_threshold
            filtered = opt_df[opt_date_filter & opt_premium_filter]
            whale_trades = filtered.to_dict('records')

        # Long-term sequences (5-min bars)
        for i in range(72, len(bars_5min) - 5, 10):
            seq = extract_longterm_sequence(bars_5min, cvd_1min, whale_trades, i * 5, seq_len=72)
            if seq is not None:
                # Label: next 5-min bar direction
                future_price = bars_5min[i + 5]['c'] if i + 5 < len(bars_5min) else bars_5min[i]['c']
                current_price = bars_5min[i]['c']
                label = 1 if future_price > current_price else 0
                longterm_seqs.append(seq)
                longterm_labels.append(label)

        # Short-term sequences (1-second bars)
        for i in range(120, len(bars_1s) - 60, 60):
            seq = extract_shortterm_sequence(bars_1s, footprint_1min, i, seq_len=120)
            if seq is not None:
                # Label: next minute direction
                future_idx = min(i + 60, len(bars_1s) - 1)
                future_price = bars_1s[future_idx]['c']
                current_price = bars_1s[i]['c']
                label = 1 if future_price > current_price else 0
                shortterm_seqs.append(seq)
                shortterm_labels.append(label)

    # Create models
    longterm_lstm = LongTermLSTM(input_dim=9, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    shortterm_lstm = ShortTermLSTM(input_dim=9, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM).to(DEVICE)

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

    return longterm_lstm, shortterm_lstm


# =============================================================================
# SINGLE DAY BACKTEST
# =============================================================================

def run_single_day(
    bars_1s: List[Dict],
    options_df: pd.DataFrame,
    target_date: str,
    final_xgb_model: Any,
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
    iv_rank: Dict[str, float],
    oi_changes: Dict[str, float],
    options_premium_threshold: float,
) -> Tuple[List[Dict], Dict]:
    """Run backtest for a single day using the ensemble model."""

    # Aggregate bars
    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
    bars_5min = aggregate_bars(bars_1s, 5)
    bars_15min = aggregate_bars(bars_1s, 15)
    bars_1h = aggregate_bars(bars_1s, 60)
    footprint_1min = calculate_footprint_candles(bars_1s, period_seconds=60)

    if len(bars_1min) < 100:
        return [], {}

    # Build timestamp mappings
    ts_to_idx = {}
    for i, bar in enumerate(bars_1min):
        ts_str = bar['t'].replace('+00:00', '') if isinstance(bar['t'], str) else bar['t']
        ts_to_idx[ts_str] = i

    bars_1s_ts_to_idx = {}
    for i, bar in enumerate(bars_1s):
        ts = bar['t']
        if isinstance(ts, str):
            ts = ts.replace('+00:00', '')
        bars_1s_ts_to_idx[ts] = i

    # Filter whale options
    opt_date_filter = options_df['datetime'].dt.strftime('%Y-%m-%d') == target_date
    opt_premium_filter = options_df['premium'] >= options_premium_threshold
    filtered_options = options_df[opt_date_filter & opt_premium_filter].copy()

    # Get whale trades list for long-term LSTM
    whale_trades = filtered_options.to_dict('records')

    trades = []
    current_position_exit_bar = 0

    longterm_lstm.eval()
    shortterm_lstm.eval()

    for _, row in filtered_options.iterrows():
        whale_ts = row['datetime']
        minute_ts = whale_ts.replace(second=0, microsecond=0)
        if hasattr(minute_ts, 'tz_localize'):
            minute_ts = minute_ts.tz_localize(None)
        minute_str = minute_ts.strftime('%Y-%m-%dT%H:%M:%S')

        if minute_str not in ts_to_idx:
            continue

        bar_idx = ts_to_idx[minute_str]
        if bar_idx < 72 or bar_idx >= len(bars_1min) - HOLD_BARS:
            continue

        if bar_idx < current_position_exit_bar:
            continue

        current_bar = bars_1min[bar_idx]
        current_price = current_bar['c']

        # Volume Profile
        vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

        # Whale sentiment
        whale_sentiment = row.get('bearish_or_bullish', 'neutral')

        # Extract XGBoost features
        xgb_features = extract_xgb_features(
            bars_1min, bars_5min, bars_15min, bars_1h,
            cvd_1min, cvd_ema_1min, bar_idx, vp,
            whale_sentiment, iv_rank, oi_changes
        )
        if xgb_features is None:
            continue

        # Extract LSTM sequences
        longterm_seq = extract_longterm_sequence(bars_5min, cvd_1min, whale_trades, bar_idx, seq_len=72)
        if longterm_seq is None:
            continue

        # Find 1s bar index
        bar_1s_idx = None
        for ts_key, idx in bars_1s_ts_to_idx.items():
            if ts_key.startswith(minute_str[:16]):
                bar_1s_idx = idx
                break

        if bar_1s_idx is None or bar_1s_idx < 120:
            continue

        shortterm_seq = extract_shortterm_sequence(bars_1s, footprint_1min, bar_1s_idx, seq_len=120)
        if shortterm_seq is None:
            continue

        # Get LSTM embeddings
        with torch.no_grad():
            longterm_tensor = torch.tensor(longterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            shortterm_tensor = torch.tensor(shortterm_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            longterm_emb = longterm_lstm(longterm_tensor).cpu().numpy().flatten()
            shortterm_emb = shortterm_lstm(shortterm_tensor).cpu().numpy().flatten()

        # Combine features
        xgb_feat_array = np.array(list(xgb_features.values()))
        combined = np.concatenate([xgb_feat_array, longterm_emb, shortterm_emb])

        # Final XGBoost prediction: 0=HOLD, 1=LONG, 2=SHORT
        X_pred = combined.reshape(1, -1)
        prediction = final_xgb_model.predict(X_pred)[0]
        proba = final_xgb_model.predict_proba(X_pred)[0]

        # Skip if HOLD
        if prediction == 0:
            continue

        direction = "long" if prediction == 1 else "short"
        confidence = proba[prediction]

        # Calculate SL/TP (3:1 R:R)
        swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:bar_idx//5+1], lookback=10)

        if direction == "long":
            valid_lows = [sl for sl in swing_lows if sl < current_price] if swing_lows else []
            if valid_lows:
                stop_loss = min(valid_lows) - 1
            else:
                stop_loss = current_price - 10
            risk = current_price - stop_loss
            target = current_price + (risk * 3)
        else:
            valid_highs = [sh for sh in swing_highs if sh > current_price] if swing_highs else []
            if valid_highs:
                stop_loss = max(valid_highs) + 1
            else:
                stop_loss = current_price + 10
            risk = stop_loss - current_price
            target = current_price - (risk * 3)

        # Simulate trade
        result = simulate_trade(bars_1min, bar_idx, direction, current_price, stop_loss, target, max_bars=HOLD_BARS)

        current_position_exit_bar = bar_idx + result["bars_held"]

        trade = {
            "date": target_date,
            "timestamp": minute_str,
            "direction": direction,
            "entry_price": current_price,
            "xgb_decision": int(prediction),
            "xgb_confidence": float(confidence),
            "whale_sentiment": whale_sentiment,
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
    parser = argparse.ArgumentParser(description='Multi-Stage XGBoost + LSTM Ensemble Backtest')
    parser.add_argument('--bars', required=True, help='Path to 1s bars JSON')
    parser.add_argument('--options-dir', default='data/options_flow', help='Directory with options flow CSVs')
    parser.add_argument('--options-file', default=None, help='Single options flow CSV file (overrides options-dir)')
    parser.add_argument('--ivrank', default='data/iv_rank_history.csv', help='Path to IV rank CSV')
    parser.add_argument('--oi-dir', default='data/oi_changes', help='Directory with OI changes CSVs')
    parser.add_argument('--options-premium', type=float, default=322040, help='Options premium threshold')
    args = parser.parse_args()

    print("=" * 70, file=sys.stderr)
    print("MULTI-STAGE XGBOOST + LSTM ENSEMBLE BACKTEST", file=sys.stderr)
    print(f"Options premium threshold: ${args.options_premium:,.0f}", file=sys.stderr)
    print(f"Device: {DEVICE}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Load all data
    print("[Load] Loading bars data...", file=sys.stderr)
    all_bars = load_1s_bars(args.bars)

    # Group bars by date
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
        # Load from single unfiltered file
        print(f"  Loading from {args.options_file}...", file=sys.stderr)
        full_df = load_options_flow(args.options_file, symbol='QQQ')
        # Split by date
        for date in dates:
            date_filter = full_df['datetime'].dt.strftime('%Y-%m-%d') == date
            df = full_df[date_filter].copy()
            whale_count = len(df[df['premium'] >= args.options_premium])
            if whale_count > 0:
                print(f"  {date}: {len(df)} QQQ options records ({whale_count} whale)", file=sys.stderr)
                all_options[date] = df
    else:
        # Load from individual files (try multiple naming patterns)
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
                    if whale_count > 0:
                        print(f"  {date}: {len(df)} QQQ options records ({whale_count} whale)", file=sys.stderr)
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

        # Train LSTM models
        longterm_lstm, shortterm_lstm = train_lstm_models(
            all_bars_1s, all_options, train_dates, args.options_premium, epochs=10
        )

        # Build training data with LSTM embeddings
        X_train, y_train = build_training_data_with_lstm(
            all_bars_1s, all_options, all_iv_ranks, all_oi_changes,
            train_dates, args.options_premium,
            longterm_lstm, shortterm_lstm
        )

        if len(X_train) < 50:
            print(f"  Not enough training data ({len(X_train)} samples)", file=sys.stderr)
            continue

        print(f"  Training final XGBoost on {len(X_train)} samples...", file=sys.stderr)

        # Train final XGBoost (3-class: HOLD, LONG, SHORT)
        final_xgb = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob',
            num_class=3,
        )
        final_xgb.fit(X_train, y_train, verbose=False)

        # Test on current day
        if test_date not in all_options:
            print(f"  No options data for {test_date}", file=sys.stderr)
            continue

        trades, analysis = run_single_day(
            all_bars_1s[test_date],
            all_options[test_date],
            test_date,
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
            print(f"  Results: {analysis['total_trades']} trades, {analysis['win_rate']}% win rate, {analysis['total_pnl_points']:+.2f} pts", file=sys.stderr)
        else:
            print(f"  No trades", file=sys.stderr)

    # Overall summary
    print("\n" + "=" * 70, file=sys.stderr)
    print("OVERALL RESULTS", file=sys.stderr)
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

        # Decision breakdown
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

    output_path = "data/xgboost_lstm_ensemble_results.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[Save] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
