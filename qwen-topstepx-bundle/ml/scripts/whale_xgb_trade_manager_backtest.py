#!/usr/bin/env python3
"""
Whale XGB First + XGBoost Trade Manager

Architecture:
1. Entry decision: Same as whale_xgb_first (XGBoost → LSTM → Final XGBoost)
2. Trade management: A separate XGBoost decides at each bar what to do:
   - HOLD: Keep position as is
   - TIGHTEN_SL: Move stop loss tighter (reduce risk)
   - TRAIL_BE: Trail stop to breakeven
   - TAKE_PARTIAL: Close 50% of position, lock in profits
   - SCALE_IN: Add to position (if in profit, max 2x)
   - RAISE_TP: Raise take profit target (let winners run)
   - CLOSE: Close entire position immediately

The trade manager is trained on labeled data from optimal hindsight decisions.
"""

import json
import sys
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from enum import IntEnum
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

# PyTorch for LSTM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Trade management actions
class TradeAction(IntEnum):
    HOLD = 0           # Keep position as is
    TIGHTEN_SL = 1     # Move SL tighter (50% of current risk)
    TRAIL_BE = 2       # Move SL to breakeven
    TAKE_PARTIAL = 3   # Close 50% of position
    SCALE_IN = 4       # Add 50% more to position
    RAISE_TP = 5       # Raise TP by 50%
    CLOSE = 6          # Close entire position now

# Trade timeout in 1-min bars
HOLD_BARS = 30

# LSTM configuration
LONGTERM_SEQ_LEN = 72
SHORTTERM_SEQ_LEN = 120
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
NUM_LAYERS = 2

# Candidate pool settings
MAX_PENDING_CANDIDATES = 10
CANDIDATE_STALE_BARS = 5

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DATA LOADING FUNCTIONS (same as whale_xgb_first)
# =============================================================================

def load_1s_bars(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("bars", [])


def load_options_flow(filepath: str, symbol: str = 'QQQ') -> pd.DataFrame:
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

        bar_1min = {
            't': minute_ts,
            'o': bars[0]['o'],
            'h': max(b['h'] for b in bars),
            'l': min(b['l'] for b in bars),
            'c': bars[-1]['c'],
            'v': sum(b.get('v', 0) or 0 for b in bars),
        }
        bars_1min.append(bar_1min)
        cvd += data['delta']
        cvd_1min.append(cvd)

    # EMA of CVD
    cvd_ema = []
    alpha = 2 / (20 + 1)
    ema = cvd_1min[0] if cvd_1min else 0
    for val in cvd_1min:
        ema = alpha * val + (1 - alpha) * ema
        cvd_ema.append(ema)

    return bars_1min, cvd_1min, cvd_ema


def aggregate_to_5min(bars_1min: List[Dict]) -> List[Dict]:
    if not bars_1min:
        return []
    bars_5min = []
    for i in range(0, len(bars_1min), 5):
        chunk = bars_1min[i:i+5]
        if chunk:
            bar = {
                't': chunk[0]['t'],
                'o': chunk[0]['o'],
                'h': max(b['h'] for b in chunk),
                'l': min(b['l'] for b in chunk),
                'c': chunk[-1]['c'],
                'v': sum(b.get('v', 0) or 0 for b in chunk),
            }
            bars_5min.append(bar)
    return bars_5min


def aggregate_to_15min(bars_1min: List[Dict]) -> List[Dict]:
    if not bars_1min:
        return []
    bars_15min = []
    for i in range(0, len(bars_1min), 15):
        chunk = bars_1min[i:i+15]
        if chunk:
            bar = {
                't': chunk[0]['t'],
                'o': chunk[0]['o'],
                'h': max(b['h'] for b in chunk),
                'l': min(b['l'] for b in chunk),
                'c': chunk[-1]['c'],
                'v': sum(b.get('v', 0) or 0 for b in chunk),
            }
            bars_15min.append(bar)
    return bars_15min


def aggregate_to_1h(bars_1min: List[Dict]) -> List[Dict]:
    if not bars_1min:
        return []
    bars_1h = []
    for i in range(0, len(bars_1min), 60):
        chunk = bars_1min[i:i+60]
        if chunk:
            bar = {
                't': chunk[0]['t'],
                'o': chunk[0]['o'],
                'h': max(b['h'] for b in chunk),
                'l': min(b['l'] for b in chunk),
                'c': chunk[-1]['c'],
                'v': sum(b.get('v', 0) or 0 for b in chunk),
            }
            bars_1h.append(bar)
    return bars_1h


def calculate_volume_profile(bars_1min: List[Dict], num_levels: int = 50) -> Dict[str, float]:
    if not bars_1min:
        return {'poc': 0, 'vah': 0, 'val': 0, 'total_volume': 0}

    prices = []
    volumes = []
    for bar in bars_1min:
        mid = (bar['h'] + bar['l']) / 2
        vol = bar.get('v', 0) or 0
        prices.append(mid)
        volumes.append(vol)

    if not prices or sum(volumes) == 0:
        return {'poc': 0, 'vah': 0, 'val': 0, 'total_volume': 0}

    min_price = min(prices)
    max_price = max(prices)
    if max_price == min_price:
        return {'poc': min_price, 'vah': min_price, 'val': min_price, 'total_volume': sum(volumes)}

    level_size = (max_price - min_price) / num_levels
    volume_at_price = defaultdict(float)

    for price, vol in zip(prices, volumes):
        level = int((price - min_price) / level_size)
        level = min(level, num_levels - 1)
        level_price = min_price + (level + 0.5) * level_size
        volume_at_price[level_price] += vol

    poc = max(volume_at_price.keys(), key=lambda x: volume_at_price[x])
    total_vol = sum(volume_at_price.values())

    sorted_levels = sorted(volume_at_price.keys())
    cumulative = 0
    val = sorted_levels[0]
    vah = sorted_levels[-1]

    for level in sorted_levels:
        cumulative += volume_at_price[level]
        if cumulative >= total_vol * 0.3 and val == sorted_levels[0]:
            val = level
        if cumulative >= total_vol * 0.7:
            vah = level
            break

    return {'poc': poc, 'vah': vah, 'val': val, 'total_volume': total_vol}


# =============================================================================
# LSTM MODELS (same as before)
# =============================================================================

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, embedding_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class LSTMDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], labels: List[int]):
        self.sequences = [torch.FloatTensor(s) for s in sequences]
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def prepare_longterm_sequence(bars_5min: List[Dict], end_idx: int, seq_len: int = LONGTERM_SEQ_LEN) -> np.ndarray:
    start_idx = max(0, end_idx - seq_len + 1)
    bars_slice = bars_5min[start_idx:end_idx + 1]

    features = []
    for bar in bars_slice:
        bar_range = bar['h'] - bar['l']
        close_pos = (bar['c'] - bar['l']) / bar_range if bar_range > 0 else 0.5
        body_size = abs(bar['c'] - bar['o'])
        features.append([
            bar['o'], bar['h'], bar['l'], bar['c'],
            bar.get('v', 0) or 0,
            close_pos, body_size, bar_range,
        ])

    features = np.array(features, dtype=np.float32)

    if len(features) < seq_len:
        padding = np.zeros((seq_len - len(features), features.shape[1]), dtype=np.float32)
        features = np.vstack([padding, features])

    return features


def prepare_shortterm_sequence(bars_1s: List[Dict], end_idx: int, seq_len: int = SHORTTERM_SEQ_LEN) -> np.ndarray:
    start_idx = max(0, end_idx - seq_len + 1)
    bars_slice = bars_1s[start_idx:end_idx + 1]

    features = []
    for bar in bars_slice:
        bar_range = bar['h'] - bar['l']
        close_pos = (bar['c'] - bar['l']) / bar_range if bar_range > 0 else 0.5
        features.append([
            bar['o'], bar['h'], bar['l'], bar['c'],
            bar.get('v', 0) or 0,
            close_pos,
        ])

    features = np.array(features, dtype=np.float32)

    if len(features) < seq_len:
        padding = np.zeros((seq_len - len(features), features.shape[1]), dtype=np.float32)
        features = np.vstack([padding, features])

    return features


def train_lstm(model: nn.Module, sequences: List[np.ndarray], labels: List[int], epochs: int = 10):
    if len(sequences) < 10:
        return

    dataset = LSTMDataset(sequences, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            # For embedding extraction, we add a classification head temporarily
            embeddings = model(batch_x)
            # Simple classifier for training
            logits = torch.zeros(embeddings.size(0), 2, device=DEVICE)
            logits[:, 1] = embeddings.mean(dim=1)
            logits[:, 0] = -embeddings.mean(dim=1)

            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()


# =============================================================================
# TRADE MANAGER FEATURES
# =============================================================================

def get_trade_manager_features(
    bars_1min: List[Dict],
    bars_5min: List[Dict],
    bar_idx: int,
    direction: str,
    entry_price: float,
    current_sl: float,
    current_tp: float,
    bars_in_trade: int,
    current_size: float,
    realized_pnl: float,
    cvd_1min: List[float],
    cvd_ema_1min: List[float],
    vp: Dict[str, float],
) -> Dict[str, float]:
    """
    Extract features for trade manager decision.
    These features capture the current trade state and market conditions.
    """
    if bar_idx >= len(bars_1min):
        return {}

    current_bar = bars_1min[bar_idx]
    current_price = current_bar['c']

    # Trade state features
    if direction == "long":
        unrealized_pnl = current_price - entry_price
        pnl_vs_sl = current_price - current_sl  # Distance to stop loss
        pnl_vs_tp = current_tp - current_price  # Distance to take profit
        risk = entry_price - current_sl
        reward = current_tp - entry_price
    else:
        unrealized_pnl = entry_price - current_price
        pnl_vs_sl = current_sl - current_price
        pnl_vs_tp = current_price - current_tp
        risk = current_sl - entry_price
        reward = entry_price - current_tp

    # Normalize by ATR (use recent range as proxy)
    recent_bars = bars_1min[max(0, bar_idx-20):bar_idx+1]
    atr = np.mean([b['h'] - b['l'] for b in recent_bars]) if recent_bars else 1.0
    atr = max(atr, 0.01)

    # Trade progress
    pnl_ratio = unrealized_pnl / risk if risk != 0 else 0  # How far towards SL (negative) or TP (positive)
    time_progress = bars_in_trade / HOLD_BARS  # How much time has elapsed

    # Is trade profitable?
    in_profit = 1 if unrealized_pnl > 0 else 0
    if direction == "long":
        profit_locked = 1 if current_sl > entry_price else 0
    else:
        profit_locked = 1 if current_sl < entry_price else 0

    # CVD analysis
    cvd_now = cvd_1min[bar_idx] if bar_idx < len(cvd_1min) else 0
    cvd_ema_now = cvd_ema_1min[bar_idx] if bar_idx < len(cvd_ema_1min) else 0
    cvd_trend = 1 if cvd_now > cvd_ema_now else -1 if cvd_now < cvd_ema_now else 0
    cvd_with_trade = cvd_trend if direction == "long" else -cvd_trend  # Is CVD in trade direction?

    # Price momentum
    if bar_idx >= 5:
        price_change_5 = (current_price - bars_1min[bar_idx-5]['c']) / atr
        momentum_with_trade = price_change_5 if direction == "long" else -price_change_5
    else:
        price_change_5 = 0
        momentum_with_trade = 0

    # Candle patterns
    recent_candles = bars_1min[max(0, bar_idx-5):bar_idx+1]
    bullish_candles = sum(1 for b in recent_candles if b['c'] > b['o']) / len(recent_candles)
    candle_direction = bullish_candles if direction == "long" else (1 - bullish_candles)

    # Volume surge
    recent_volume = np.mean([b.get('v', 0) or 0 for b in recent_candles])
    avg_volume = np.mean([b.get('v', 0) or 0 for b in bars_1min[max(0, bar_idx-60):bar_idx+1]])
    volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1.0

    # Price vs volume profile
    price_vs_poc = (current_price - vp['poc']) / atr if vp['poc'] > 0 else 0
    in_value_area = 1 if vp['val'] <= current_price <= vp['vah'] else 0

    # Volatility (range expansion)
    current_range = current_bar['h'] - current_bar['l']
    range_vs_atr = current_range / atr

    # Time of day
    ts_str = current_bar['t']
    if isinstance(ts_str, str):
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00').replace('+00:00', ''))
    else:
        dt = ts_str
    hour = dt.hour
    minute = dt.minute
    minutes_since_open = (hour - 9) * 60 + minute - 30
    is_close_30min = 1 if minutes_since_open >= 360 else 0

    return {
        # Trade state
        'unrealized_pnl_atr': unrealized_pnl / atr,
        'pnl_ratio': pnl_ratio,
        'risk_atr': risk / atr,
        'reward_atr': reward / atr,
        'time_progress': time_progress,
        'in_profit': in_profit,
        'profit_locked': profit_locked,
        'current_size': current_size,
        'realized_pnl_atr': realized_pnl / atr,

        # Market state
        'cvd_with_trade': cvd_with_trade,
        'momentum_with_trade': momentum_with_trade,
        'candle_direction': candle_direction,
        'volume_surge': volume_surge,
        'price_vs_poc_atr': price_vs_poc,
        'in_value_area': in_value_area,
        'range_vs_atr': range_vs_atr,
        'is_close_30min': is_close_30min,

        # Direction encoding
        'is_long': 1 if direction == "long" else 0,
    }


def label_optimal_action(
    bars_1min: List[Dict],
    bar_idx: int,
    direction: str,
    entry_price: float,
    current_sl: float,
    current_tp: float,
    future_bars: int = 10,
) -> int:
    """
    Label the optimal action based on hindsight.
    This is used to train the trade manager.
    """
    if bar_idx + future_bars >= len(bars_1min):
        return TradeAction.HOLD

    current_price = bars_1min[bar_idx]['c']

    # Look at future price action
    future_high = max(b['h'] for b in bars_1min[bar_idx:bar_idx+future_bars+1])
    future_low = min(b['l'] for b in bars_1min[bar_idx:bar_idx+future_bars+1])
    future_close = bars_1min[bar_idx+future_bars]['c']

    if direction == "long":
        unrealized_pnl = current_price - entry_price
        max_future_pnl = future_high - entry_price
        min_future_pnl = future_low - entry_price
        final_pnl = future_close - entry_price
        risk = entry_price - current_sl
    else:
        unrealized_pnl = entry_price - current_price
        max_future_pnl = entry_price - future_low
        min_future_pnl = entry_price - future_high
        final_pnl = entry_price - future_close
        risk = current_sl - entry_price

    # Decision rules based on hindsight
    # 1. If price will hit SL, we should CLOSE now
    if direction == "long" and future_low <= current_sl:
        return TradeAction.CLOSE
    if direction == "short" and future_high >= current_sl:
        return TradeAction.CLOSE

    # 2. If we're in profit and will give it all back, TAKE_PARTIAL or TRAIL_BE
    if unrealized_pnl > 0:
        if min_future_pnl < unrealized_pnl * 0.3:  # Will lose 70% of profits
            if unrealized_pnl > risk * 0.5:  # Decent profit
                return TradeAction.TRAIL_BE
            return TradeAction.TIGHTEN_SL

    # 3. If price will go much higher, RAISE_TP or SCALE_IN
    if max_future_pnl > unrealized_pnl + risk * 1.5:  # Much more upside
        if unrealized_pnl > risk * 0.3:  # Already in profit
            if np.random.random() < 0.5:
                return TradeAction.SCALE_IN
            return TradeAction.RAISE_TP

    # 4. If close to TP and will reverse, CLOSE
    target_distance = current_tp - current_price if direction == "long" else current_price - current_tp
    if target_distance < risk * 0.3 and final_pnl < unrealized_pnl:
        return TradeAction.CLOSE

    # 5. Default: HOLD
    return TradeAction.HOLD


# =============================================================================
# STAGE 1 XGBoost FEATURE EXTRACTION (same as whale_xgb_first)
# =============================================================================

def extract_stage1_features(
    bars_1min: List[Dict],
    bars_5min: List[Dict],
    bars_15min: List[Dict],
    bars_1h: List[Dict],
    bar_idx: int,
    whale_sentiment: str,
    cvd_1min: List[float],
    cvd_ema_1min: List[float],
    vp: Dict[str, float],
    iv_rank: Optional[Dict[str, float]] = None,
    oi_changes: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Extract features for Stage 1 XGBoost direction prediction."""
    if bar_idx < 20 or bar_idx >= len(bars_1min):
        return {}

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

    whale_bullish = 1 if whale_sentiment == 'bullish' else 0
    whale_bearish = 1 if whale_sentiment == 'bearish' else 0

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
# MANAGED TRADE SIMULATION
# =============================================================================

def simulate_managed_trade(
    bars_1min: List[Dict],
    bars_5min: List[Dict],
    entry_idx: int,
    direction: str,
    entry_price: float,
    initial_sl: float,
    initial_tp: float,
    trade_manager: Optional[xgb.XGBClassifier],
    cvd_1min: List[float],
    cvd_ema_1min: List[float],
    vp: Dict[str, float],
    max_bars: int = HOLD_BARS,
    collect_training_data: bool = False,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Simulate a trade with XGBoost trade manager making decisions at each bar.
    Returns trade result and training data for the trade manager.
    """
    current_sl = initial_sl
    current_tp = initial_tp
    current_size = 1.0  # Normalized position size
    realized_pnl = 0.0
    current_price = entry_price

    training_samples = []
    actions_taken = []

    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(bars_1min))):
        bar = bars_1min[i]
        bars_in_trade = i - entry_idx
        prev_price = current_price
        current_price = bar['c']

        # Check if SL or TP hit first (using bar high/low)
        if direction == "long":
            if bar['l'] <= current_sl:
                # Stop loss hit
                pnl = (current_sl - entry_price) * current_size + realized_pnl
                return {
                    "outcome": "loss" if pnl < 0 else "win",
                    "exit_price": current_sl,
                    "pnl_points": round(pnl, 2),
                    "bars_held": bars_in_trade,
                    "exit_reason": "sl",
                    "actions_taken": actions_taken,
                    "final_size": current_size,
                }, training_samples

            if bar['h'] >= current_tp:
                # Take profit hit
                pnl = (current_tp - entry_price) * current_size + realized_pnl
                return {
                    "outcome": "win",
                    "exit_price": current_tp,
                    "pnl_points": round(pnl, 2),
                    "bars_held": bars_in_trade,
                    "exit_reason": "tp",
                    "actions_taken": actions_taken,
                    "final_size": current_size,
                }, training_samples
        else:  # short
            if bar['h'] >= current_sl:
                pnl = (entry_price - current_sl) * current_size + realized_pnl
                return {
                    "outcome": "loss" if pnl < 0 else "win",
                    "exit_price": current_sl,
                    "pnl_points": round(pnl, 2),
                    "bars_held": bars_in_trade,
                    "exit_reason": "sl",
                    "actions_taken": actions_taken,
                    "final_size": current_size,
                }, training_samples

            if bar['l'] <= current_tp:
                pnl = (entry_price - current_tp) * current_size + realized_pnl
                return {
                    "outcome": "win",
                    "exit_price": current_tp,
                    "pnl_points": round(pnl, 2),
                    "bars_held": bars_in_trade,
                    "exit_reason": "tp",
                    "actions_taken": actions_taken,
                    "final_size": current_size,
                }, training_samples

        # Get trade manager features
        tm_features = get_trade_manager_features(
            bars_1min, bars_5min, i, direction, entry_price,
            current_sl, current_tp, bars_in_trade, current_size,
            realized_pnl, cvd_1min, cvd_ema_1min, vp
        )

        if not tm_features:
            continue

        # Collect training data if requested
        if collect_training_data:
            optimal_action = label_optimal_action(
                bars_1min, i, direction, entry_price, current_sl, current_tp
            )
            training_samples.append({
                'features': tm_features,
                'label': optimal_action,
            })

        # Get action from trade manager (or use rule-based if no model)
        if trade_manager is not None:
            feature_names = sorted(tm_features.keys())
            X = np.array([[tm_features[f] for f in feature_names]])
            action = trade_manager.predict(X)[0]
        else:
            # Rule-based fallback: simple trailing stop
            unrealized = (current_price - entry_price) if direction == "long" else (entry_price - current_price)
            risk = abs(entry_price - initial_sl)
            if unrealized > risk * 1.5:
                action = TradeAction.TRAIL_BE
            elif unrealized < -risk * 0.5:
                action = TradeAction.TIGHTEN_SL
            else:
                action = TradeAction.HOLD

        # Execute action
        if action == TradeAction.HOLD:
            pass  # No change

        elif action == TradeAction.TIGHTEN_SL:
            # Move SL 50% closer to entry
            if direction == "long":
                new_sl = entry_price - (entry_price - current_sl) * 0.5
                if new_sl > current_sl:
                    current_sl = new_sl
                    actions_taken.append(('TIGHTEN_SL', bars_in_trade, current_sl))
            else:
                new_sl = entry_price + (current_sl - entry_price) * 0.5
                if new_sl < current_sl:
                    current_sl = new_sl
                    actions_taken.append(('TIGHTEN_SL', bars_in_trade, current_sl))

        elif action == TradeAction.TRAIL_BE:
            # Move SL to breakeven (entry price)
            if direction == "long" and current_price > entry_price:
                if entry_price > current_sl:
                    current_sl = entry_price
                    actions_taken.append(('TRAIL_BE', bars_in_trade, current_sl))
            elif direction == "short" and current_price < entry_price:
                if entry_price < current_sl:
                    current_sl = entry_price
                    actions_taken.append(('TRAIL_BE', bars_in_trade, current_sl))

        elif action == TradeAction.TAKE_PARTIAL:
            # Close 50% of position
            if current_size > 0.5:
                close_size = current_size * 0.5
                if direction == "long":
                    partial_pnl = (current_price - entry_price) * close_size
                else:
                    partial_pnl = (entry_price - current_price) * close_size
                realized_pnl += partial_pnl
                current_size -= close_size
                actions_taken.append(('TAKE_PARTIAL', bars_in_trade, partial_pnl))

        elif action == TradeAction.SCALE_IN:
            # Add 50% to position (max 2x)
            if current_size < 2.0:
                add_size = min(0.5, 2.0 - current_size)
                current_size += add_size
                actions_taken.append(('SCALE_IN', bars_in_trade, current_size))

        elif action == TradeAction.RAISE_TP:
            # Raise TP by 50%
            if direction == "long":
                tp_distance = current_tp - entry_price
                new_tp = current_tp + tp_distance * 0.5
                current_tp = new_tp
            else:
                tp_distance = entry_price - current_tp
                new_tp = current_tp - tp_distance * 0.5
                current_tp = new_tp
            actions_taken.append(('RAISE_TP', bars_in_trade, current_tp))

        elif action == TradeAction.CLOSE:
            # Close entire position immediately
            if direction == "long":
                pnl = (current_price - entry_price) * current_size + realized_pnl
            else:
                pnl = (entry_price - current_price) * current_size + realized_pnl
            return {
                "outcome": "win" if pnl > 0 else "loss",
                "exit_price": current_price,
                "pnl_points": round(pnl, 2),
                "bars_held": bars_in_trade,
                "exit_reason": "managed_close",
                "actions_taken": actions_taken,
                "final_size": current_size,
            }, training_samples

    # Timeout - close at market
    final_bar = bars_1min[min(entry_idx + max_bars, len(bars_1min) - 1)]
    exit_price = final_bar['c']
    if direction == "long":
        pnl = (exit_price - entry_price) * current_size + realized_pnl
    else:
        pnl = (entry_price - exit_price) * current_size + realized_pnl

    return {
        "outcome": "win" if pnl > 0 else "loss",
        "exit_price": exit_price,
        "pnl_points": round(pnl, 2),
        "bars_held": max_bars,
        "exit_reason": "timeout",
        "actions_taken": actions_taken,
        "final_size": current_size,
    }, training_samples


# =============================================================================
# CALCULATE SWING HIGH/LOW FOR STOP LOSS
# =============================================================================

def find_swing_stop(bars_1min: List[Dict], bar_idx: int, direction: str, lookback: int = 10) -> float:
    """Find swing high/low for stop loss placement."""
    recent_bars = bars_1min[max(0, bar_idx - lookback):bar_idx + 1]
    if not recent_bars:
        return bars_1min[bar_idx]['c']

    if direction == "long":
        return min(b['l'] for b in recent_bars)
    else:
        return max(b['h'] for b in recent_bars)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_stage1_xgb(features_list: List[Dict], labels: List[int]) -> xgb.XGBClassifier:
    """Train Stage 1 XGBoost for direction prediction."""
    if not features_list:
        return None

    feature_names = sorted(features_list[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in features_list])
    y = np.array(labels)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    )
    model.fit(X, y)
    return model


def train_final_xgb(features_list: List[Dict], labels: List[int]) -> xgb.XGBClassifier:
    """Train Final XGBoost for trade selection."""
    if not features_list:
        return None

    feature_names = sorted(features_list[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in features_list])
    y = np.array(labels)

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    )
    model.fit(X, y)
    return model


def train_trade_manager_xgb(training_samples: List[Dict]) -> xgb.XGBClassifier:
    """Train trade manager XGBoost for trade management decisions."""
    if len(training_samples) < 50:
        return None

    feature_names = sorted(training_samples[0]['features'].keys())
    X = np.array([[s['features'][k] for k in feature_names] for s in training_samples])
    y = np.array([s['label'] for s in training_samples])

    # Ensure all classes are represented by adding synthetic samples if needed
    unique_classes = set(y)
    num_classes = len(TradeAction)

    # Add dummy samples for missing classes (will be overweighted by real data)
    for c in range(num_classes):
        if c not in unique_classes:
            # Add a synthetic sample with mean features
            mean_features = X.mean(axis=0)
            X = np.vstack([X, mean_features])
            y = np.append(y, c)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=num_classes,
        objective='multi:softmax',
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X, y)
    return model


# =============================================================================
# MAIN BACKTEST FUNCTION
# =============================================================================

def run_single_day_with_trade_manager(
    bars_1s: List[Dict],
    options_df: pd.DataFrame,
    target_date: str,
    stage1_xgb: xgb.XGBClassifier,
    final_xgb: xgb.XGBClassifier,
    trade_manager_xgb: Optional[xgb.XGBClassifier],
    longterm_lstm: nn.Module,
    shortterm_lstm: nn.Module,
    iv_rank: Dict[str, float],
    oi_changes: Dict[str, float],
    options_premium_threshold: float,
    starting_equity: float = 0.0,
    collect_tm_training: bool = False,
) -> Tuple[List[Dict], Dict, float, float, List[Dict]]:
    """Run backtest for a single day with trade manager."""

    # Filter bars for target date
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    day_bars = [b for b in bars_1s if b['t'].startswith(target_date)]

    if len(day_bars) < 1000:
        return [], {"date": target_date, "total_trades": 0}, starting_equity, starting_equity, []

    # Build aggregated bars
    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(day_bars)
    bars_5min = aggregate_to_5min(bars_1min)
    bars_15min = aggregate_to_15min(bars_1min)
    bars_1h = aggregate_to_1h(bars_1min)
    vp = calculate_volume_profile(bars_1min)

    # Get whale trades for this day
    day_options = options_df[options_df['datetime'].dt.strftime('%Y-%m-%d') == target_date].copy()
    if len(day_options) == 0:
        return [], {"date": target_date, "total_trades": 0, "note": "no options data"}, starting_equity, starting_equity, []

    whale_trades = day_options[day_options['premium'] >= options_premium_threshold].copy()
    whale_trades = whale_trades.sort_values('datetime')

    trades = []
    tm_training_samples = []
    in_position = False
    position_end_idx = 0

    # Drawdown tracking
    intraday_pnl = 0.0
    intraday_peak_pnl = 0.0
    max_intraday_dd = 0.0
    max_intraday_dd_from_peak = 0.0

    cumulative_equity = starting_equity
    peak_equity = starting_equity

    candidate_pool = []

    for _, whale in whale_trades.iterrows():
        whale_time = whale['datetime']
        if whale_time.tzinfo is not None:
            whale_time = whale_time.replace(tzinfo=None)

        # Find corresponding bar index
        bar_idx = None
        for i, bar in enumerate(bars_1min):
            bar_time = datetime.fromisoformat(bar['t'].replace('Z', '').replace('+00:00', ''))
            if bar_time >= whale_time:
                bar_idx = i
                break

        if bar_idx is None or bar_idx < 30:
            continue

        # Skip if in position
        if in_position and bar_idx < position_end_idx:
            continue
        in_position = False

        whale_sentiment = whale['bearish_or_bullish']

        # Stage 1: Generate candidate with XGBoost
        features = extract_stage1_features(
            bars_1min, bars_5min, bars_15min, bars_1h, bar_idx,
            whale_sentiment, cvd_1min, cvd_ema_1min, vp, iv_rank, oi_changes
        )

        if not features:
            continue

        feature_names = sorted(features.keys())
        X = np.array([[features[k] for k in feature_names]])
        stage1_pred = stage1_xgb.predict_proba(X)[0]
        direction = "long" if stage1_pred[1] > 0.5 else "short"
        confidence = max(stage1_pred)

        # Add to candidate pool
        candidate = {
            'bar_idx': bar_idx,
            'direction': direction,
            'confidence': confidence,
            'features': features,
            'whale_sentiment': whale_sentiment,
            'entry_price': bars_1min[bar_idx]['c'],
        }

        # Manage pool
        candidate_pool = [c for c in candidate_pool if bar_idx - c['bar_idx'] < CANDIDATE_STALE_BARS]
        candidate_pool.append(candidate)
        if len(candidate_pool) > MAX_PENDING_CANDIDATES:
            candidate_pool = sorted(candidate_pool, key=lambda x: x['confidence'], reverse=True)[:MAX_PENDING_CANDIDATES]

        # Get LSTM embeddings for all candidates
        longterm_lstm.eval()
        shortterm_lstm.eval()

        best_candidate = None
        best_score = 0.0

        for cand in candidate_pool:
            c_bar_idx = cand['bar_idx']
            idx_5m = c_bar_idx // 5

            lt_seq = prepare_longterm_sequence(bars_5min, idx_5m)
            st_idx = min(c_bar_idx * 60, len(day_bars) - 1)
            st_seq = prepare_shortterm_sequence(day_bars, st_idx)

            with torch.no_grad():
                lt_tensor = torch.FloatTensor(lt_seq).unsqueeze(0).to(DEVICE)
                st_tensor = torch.FloatTensor(st_seq).unsqueeze(0).to(DEVICE)
                lt_emb = longterm_lstm(lt_tensor).cpu().numpy()[0]
                st_emb = shortterm_lstm(st_tensor).cpu().numpy()[0]

            # Build final features
            final_features = cand['features'].copy()
            for i, val in enumerate(lt_emb):
                final_features[f'lt_emb_{i}'] = val
            for i, val in enumerate(st_emb):
                final_features[f'st_emb_{i}'] = val
            final_features['stage1_confidence'] = cand['confidence']
            final_features['candidates_in_pool'] = len(candidate_pool)

            # Score with final XGBoost
            final_names = sorted(final_features.keys())
            X_final = np.array([[final_features[k] for k in final_names]])
            final_pred = final_xgb.predict_proba(X_final)[0]
            score = final_pred[1]

            if score > best_score:
                best_score = score
                best_candidate = cand
                best_candidate['final_features'] = final_features
                best_candidate['final_score'] = score

        # Take trade if score > threshold
        if best_candidate and best_score > 0.55:
            c = best_candidate
            entry_idx = c['bar_idx']
            entry_price = c['entry_price']
            direction = c['direction']

            # Calculate SL/TP
            swing_stop = find_swing_stop(bars_1min, entry_idx, direction)
            if direction == "long":
                sl = swing_stop - 1.0
                risk = entry_price - sl
                tp = entry_price + risk * 3.0
            else:
                sl = swing_stop + 1.0
                risk = sl - entry_price
                tp = entry_price - risk * 3.0

            # Simulate managed trade
            result, tm_samples = simulate_managed_trade(
                bars_1min, bars_5min, entry_idx, direction, entry_price,
                sl, tp, trade_manager_xgb, cvd_1min, cvd_ema_1min, vp,
                max_bars=HOLD_BARS, collect_training_data=collect_tm_training
            )

            if collect_tm_training:
                tm_training_samples.extend(tm_samples)

            trade = {
                "date": target_date,
                "entry_time": bars_1min[entry_idx]['t'],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": result["exit_price"],
                "stop_loss": sl,
                "target": tp,
                "pnl_points": result["pnl_points"],
                "outcome": result["outcome"],
                "exit_reason": result["exit_reason"],
                "bars_held": result["bars_held"],
                "final_score": c['final_score'],
                "candidates_in_pool": len(candidate_pool),
                "actions_taken": len(result.get("actions_taken", [])),
                "final_size": result.get("final_size", 1.0),
            }
            trades.append(trade)

            # Update P&L and drawdowns
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

            in_position = True
            position_end_idx = entry_idx + result["bars_held"]
            candidate_pool = []

    # Summary
    if not trades:
        return [], {"date": target_date, "total_trades": 0}, cumulative_equity, peak_equity, tm_training_samples

    wins = sum(1 for t in trades if t['outcome'] == 'win')
    total_pnl = sum(t['pnl_points'] for t in trades)
    total_actions = sum(t['actions_taken'] for t in trades)

    analysis = {
        "date": target_date,
        "total_trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": round(wins / len(trades) * 100, 1),
        "total_pnl_points": round(total_pnl, 2),
        "avg_candidates_per_trade": round(np.mean([t['candidates_in_pool'] for t in trades]), 1),
        "max_intraday_dd_points": round(max_intraday_dd, 2),
        "max_intraday_dd_from_peak_points": round(max_intraday_dd_from_peak, 2),
        "total_tm_actions": total_actions,
        "ending_equity": round(cumulative_equity, 2),
        "peak_equity": round(peak_equity, 2),
    }

    return trades, analysis, cumulative_equity, peak_equity, tm_training_samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bars', default='data/bars_1s_nq.json')
    parser.add_argument('--options-dir', default='data/options_flow')
    parser.add_argument('--ivrank', default='data/iv_rank_history.csv')
    parser.add_argument('--oi-dir', default='data/oi_changes')
    parser.add_argument('--output', default='data/whale_xgb_trade_manager_results.json')
    args = parser.parse_args()

    # Options premium threshold for whale trades
    options_premium_threshold = 322040

    print("=" * 70, file=sys.stderr)
    print("WHALE XGB + TRADE MANAGER BACKTEST", file=sys.stderr)
    print(f"Options premium threshold: ${options_premium_threshold:,}", file=sys.stderr)
    print(f"Device: {DEVICE}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Load bars
    print("[Load] Loading bars data...", file=sys.stderr)
    bars_1s = load_1s_bars(args.bars)
    dates = sorted(set(b['t'][:10] for b in bars_1s))
    print(f"[Load] Found bars for dates: {dates}", file=sys.stderr)

    # Load options flow per day
    print("[Load] Loading options flow...", file=sys.stderr)
    options_by_date = {}
    for date in dates:
        date_formatted = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
        # Try different file naming patterns
        possible_files = [
            os.path.join(args.options_dir, f"bot-eod-report-{date_formatted}.csv"),
            os.path.join(args.options_dir, f"qqq_options_flow_{date_formatted}.csv"),
        ]
        for options_file in possible_files:
            if os.path.exists(options_file):
                df = load_options_flow(options_file)
                options_by_date[date] = df
                whale_count = len(df[df['premium'] >= options_premium_threshold])
                print(f"  {date}: {len(df)} QQQ options ({whale_count} whale)", file=sys.stderr)
                break

    # Load IV rank
    print("[Load] Loading IV rank data...", file=sys.stderr)
    iv_rank_by_date = {}
    for date in dates:
        iv_rank_by_date[date] = load_iv_rank(args.ivrank, date)

    # Load OI changes
    print("[Load] Loading OI changes data...", file=sys.stderr)
    oi_by_date = {}
    for date in dates:
        oi_file = os.path.join(args.oi_dir, f"oi_changes_{date}.csv")
        oi_by_date[date] = load_oi_changes(oi_file)

    # Walk-forward backtest
    all_trades = []
    all_analysis = []
    stage1_training_features = []
    stage1_training_labels = []
    final_training_features = []
    final_training_labels = []
    tm_training_samples = []

    # Track cumulative equity and drawdowns
    cumulative_equity = 0.0
    overall_peak_equity = 0.0
    max_dd_from_overall_peak = 0.0

    # Track worst intraday drawdowns
    max_intraday_dd = 0.0
    max_intraday_dd_from_peak = 0.0

    for day_idx, target_date in enumerate(dates):
        print(f"\n[Day {day_idx + 1}] {target_date}: ", end="", file=sys.stderr)

        # Need at least 2 training days
        training_dates = [d for d in dates[:day_idx] if d in options_by_date]
        if len(training_dates) < 2:
            print("Not enough training days", file=sys.stderr)
            continue

        print(f"Training on {training_dates}", file=sys.stderr)

        # Build training data from previous days
        for train_date in training_dates:
            train_options = options_by_date.get(train_date)
            if train_options is None:
                continue

            day_bars = [b for b in bars_1s if b['t'].startswith(train_date)]
            if len(day_bars) < 1000:
                continue

            bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(day_bars)
            bars_5min = aggregate_to_5min(bars_1min)
            bars_15min = aggregate_to_15min(bars_1min)
            bars_1h = aggregate_to_1h(bars_1min)
            vp = calculate_volume_profile(bars_1min)

            whale_trades = train_options[train_options['premium'] >= options_premium_threshold]

            for _, whale in whale_trades.iterrows():
                whale_time = whale['datetime']
                if whale_time.tzinfo is not None:
                    whale_time = whale_time.replace(tzinfo=None)

                bar_idx = None
                for i, bar in enumerate(bars_1min):
                    bar_time = datetime.fromisoformat(bar['t'].replace('Z', '').replace('+00:00', ''))
                    if bar_time >= whale_time:
                        bar_idx = i
                        break

                if bar_idx is None or bar_idx < 30 or bar_idx + 20 >= len(bars_1min):
                    continue

                features = extract_stage1_features(
                    bars_1min, bars_5min, bars_15min, bars_1h, bar_idx,
                    whale['bearish_or_bullish'], cvd_1min, cvd_ema_1min, vp,
                    iv_rank_by_date.get(train_date), oi_by_date.get(train_date)
                )

                if not features:
                    continue

                # Label: did price go up in next 20 bars?
                entry_price = bars_1min[bar_idx]['c']
                future_price = bars_1min[min(bar_idx + 20, len(bars_1min) - 1)]['c']
                label = 1 if future_price > entry_price else 0

                stage1_training_features.append(features)
                stage1_training_labels.append(label)

        if len(stage1_training_features) < 50:
            print("  Not enough training samples", file=sys.stderr)
            continue

        # Train Stage 1 XGBoost
        print("  Training Stage 1 XGBoost...", file=sys.stderr)
        stage1_xgb = train_stage1_xgb(stage1_training_features, stage1_training_labels)

        # Train LSTM models
        print("  Training LSTM models...", file=sys.stderr)
        longterm_lstm = LSTMEncoder(8, HIDDEN_DIM, NUM_LAYERS, EMBEDDING_DIM).to(DEVICE)
        shortterm_lstm = LSTMEncoder(6, HIDDEN_DIM, NUM_LAYERS, EMBEDDING_DIM).to(DEVICE)

        lt_sequences = []
        st_sequences = []
        lstm_labels = []

        for train_date in training_dates:
            day_bars = [b for b in bars_1s if b['t'].startswith(train_date)]
            if len(day_bars) < 1000:
                continue

            bars_1min, _, _ = calculate_cvd_1min(day_bars)
            bars_5min = aggregate_to_5min(bars_1min)

            for i in range(LONGTERM_SEQ_LEN, len(bars_5min) - 5, 10):
                lt_seq = prepare_longterm_sequence(bars_5min, i)
                lt_sequences.append(lt_seq)

                st_idx = min(i * 5 * 60, len(day_bars) - 1)
                st_seq = prepare_shortterm_sequence(day_bars, st_idx)
                st_sequences.append(st_seq)

                future_idx = min(i + 5, len(bars_5min) - 1)
                label = 1 if bars_5min[future_idx]['c'] > bars_5min[i]['c'] else 0
                lstm_labels.append(label)

        if lt_sequences:
            train_lstm(longterm_lstm, lt_sequences, lstm_labels, epochs=5)
            train_lstm(shortterm_lstm, st_sequences, lstm_labels, epochs=5)

        # Build final XGBoost training data
        final_training_features = []
        final_training_labels = []

        longterm_lstm.eval()
        shortterm_lstm.eval()

        for train_date in training_dates:
            train_options = options_by_date.get(train_date)
            if train_options is None:
                continue

            day_bars = [b for b in bars_1s if b['t'].startswith(train_date)]
            if len(day_bars) < 1000:
                continue

            bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(day_bars)
            bars_5min = aggregate_to_5min(bars_1min)
            bars_15min = aggregate_to_15min(bars_1min)
            bars_1h = aggregate_to_1h(bars_1min)
            vp = calculate_volume_profile(bars_1min)

            whale_trades = train_options[train_options['premium'] >= options_premium_threshold]

            for _, whale in whale_trades.iterrows():
                whale_time = whale['datetime']
                if whale_time.tzinfo is not None:
                    whale_time = whale_time.replace(tzinfo=None)

                bar_idx = None
                for i, bar in enumerate(bars_1min):
                    bar_time = datetime.fromisoformat(bar['t'].replace('Z', '').replace('+00:00', ''))
                    if bar_time >= whale_time:
                        bar_idx = i
                        break

                if bar_idx is None or bar_idx < 30 or bar_idx + 20 >= len(bars_1min):
                    continue

                features = extract_stage1_features(
                    bars_1min, bars_5min, bars_15min, bars_1h, bar_idx,
                    whale['bearish_or_bullish'], cvd_1min, cvd_ema_1min, vp,
                    iv_rank_by_date.get(train_date), oi_by_date.get(train_date)
                )

                if not features:
                    continue

                # Get LSTM embeddings
                idx_5m = bar_idx // 5
                lt_seq = prepare_longterm_sequence(bars_5min, idx_5m)
                st_idx = min(bar_idx * 60, len(day_bars) - 1)
                st_seq = prepare_shortterm_sequence(day_bars, st_idx)

                with torch.no_grad():
                    lt_tensor = torch.FloatTensor(lt_seq).unsqueeze(0).to(DEVICE)
                    st_tensor = torch.FloatTensor(st_seq).unsqueeze(0).to(DEVICE)
                    lt_emb = longterm_lstm(lt_tensor).cpu().numpy()[0]
                    st_emb = shortterm_lstm(st_tensor).cpu().numpy()[0]

                final_features = features.copy()
                for i, val in enumerate(lt_emb):
                    final_features[f'lt_emb_{i}'] = val
                for i, val in enumerate(st_emb):
                    final_features[f'st_emb_{i}'] = val
                final_features['stage1_confidence'] = 0.5
                final_features['candidates_in_pool'] = 1

                # Label
                entry_price = bars_1min[bar_idx]['c']
                future_price = bars_1min[min(bar_idx + 20, len(bars_1min) - 1)]['c']
                label = 1 if future_price > entry_price else 0

                final_training_features.append(final_features)
                final_training_labels.append(label)

        if len(final_training_features) < 50:
            print(f"  Not enough training data for final model ({len(final_training_features)} samples)", file=sys.stderr)
            continue

        # Train Final XGBoost
        pos_rate = sum(final_training_labels) / len(final_training_labels) * 100
        print(f"  Training Final XGBoost on {len(final_training_features)} samples (label balance: {pos_rate:.2f}% positive)...", file=sys.stderr)
        final_xgb = train_final_xgb(final_training_features, final_training_labels)

        # Train trade manager if we have enough samples
        trade_manager_xgb = None
        if len(tm_training_samples) >= 100:
            print(f"  Training Trade Manager on {len(tm_training_samples)} samples...", file=sys.stderr)
            trade_manager_xgb = train_trade_manager_xgb(tm_training_samples)

        # Check if we have options for target date
        if target_date not in options_by_date:
            print(f"  No options data for {target_date}", file=sys.stderr)
            continue

        # Run backtest for this day
        trades, analysis, new_equity, new_peak, new_tm_samples = run_single_day_with_trade_manager(
            bars_1s,
            options_by_date[target_date],
            target_date,
            stage1_xgb,
            final_xgb,
            trade_manager_xgb,
            longterm_lstm,
            shortterm_lstm,
            iv_rank_by_date.get(target_date, {}),
            oi_by_date.get(target_date, {}),
            options_premium_threshold,
            starting_equity=cumulative_equity,
            collect_tm_training=True,
        )

        # Collect trade manager training samples
        tm_training_samples.extend(new_tm_samples)

        cumulative_equity = new_equity
        if new_peak > overall_peak_equity:
            overall_peak_equity = new_peak

        dd_from_peak = overall_peak_equity - cumulative_equity
        if dd_from_peak > max_dd_from_overall_peak:
            max_dd_from_overall_peak = dd_from_peak

        if 'max_intraday_dd_points' in analysis:
            if analysis['max_intraday_dd_points'] > max_intraday_dd:
                max_intraday_dd = analysis['max_intraday_dd_points']
        if 'max_intraday_dd_from_peak_points' in analysis:
            if analysis['max_intraday_dd_from_peak_points'] > max_intraday_dd_from_peak:
                max_intraday_dd_from_peak = analysis['max_intraday_dd_from_peak_points']

        if trades:
            all_trades.extend(trades)
            all_analysis.append(analysis)
            tm_actions = analysis.get('total_tm_actions', 0)
            print(f"  Results: {len(trades)} trades, {analysis['win_rate']}% WR, {analysis['total_pnl_points']:+.2f} pts, DD: {analysis.get('max_intraday_dd_from_peak_points', 0):.2f} pts, TM actions: {tm_actions}", file=sys.stderr)
        else:
            print("  No trades", file=sys.stderr)

    # Final summary
    if all_trades:
        total_trades = len(all_trades)
        total_wins = sum(1 for t in all_trades if t['outcome'] == 'win')
        total_pnl = sum(t['pnl_points'] for t in all_trades)
        total_tm_actions = sum(t.get('actions_taken', 0) for t in all_trades)
        avg_pnl = total_pnl / total_trades

        long_trades = [t for t in all_trades if t['direction'] == 'long']
        short_trades = [t for t in all_trades if t['direction'] == 'short']

        print("\n" + "=" * 70, file=sys.stderr)
        print("OVERALL RESULTS (WHALE XGB + TRADE MANAGER)", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"Total trades: {total_trades}", file=sys.stderr)
        print(f"Overall win rate: {total_wins / total_trades * 100:.1f}%", file=sys.stderr)
        print(f"Total P&L: {total_pnl:+.2f} points (${total_pnl * 20:+,.0f} @ $20/pt)", file=sys.stderr)
        print(f"Avg P&L per trade: {avg_pnl:+.2f} points", file=sys.stderr)
        print(f"Total TM actions: {total_tm_actions}", file=sys.stderr)

        print(f"\nDRAWDOWN METRICS:", file=sys.stderr)
        print(f"  Max intraday DD (from day start): -{max_intraday_dd:.2f} pts (${max_intraday_dd * 20:,.0f})", file=sys.stderr)
        print(f"  Max intraday DD (from intraday peak): -{max_intraday_dd_from_peak:.2f} pts (${max_intraday_dd_from_peak * 20:,.0f})", file=sys.stderr)
        print(f"  Max DD from overall peak: -{max_dd_from_overall_peak:.2f} pts (${max_dd_from_overall_peak * 20:,.0f})", file=sys.stderr)
        print(f"  Peak equity: {overall_peak_equity:+.2f} pts (${overall_peak_equity * 20:+,.0f})", file=sys.stderr)
        print(f"  Final equity: {cumulative_equity:+.2f} pts (${cumulative_equity * 20:+,.0f})", file=sys.stderr)

        print(f"\nBy Day:", file=sys.stderr)
        for a in all_analysis:
            if a['total_trades'] > 0:
                tm_acts = a.get('total_tm_actions', 0)
                print(f"  {a['date']}: {a['total_trades']} trades, {a['win_rate']}% WR, {a['total_pnl_points']:+.2f} pts, TM: {tm_acts}", file=sys.stderr)

        print(f"\nDecision breakdown: {len(long_trades)} LONG, {len(short_trades)} SHORT", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        # Save results
        output = {
            "strategy": "whale_xgb_trade_manager",
            "options_premium_threshold": options_premium_threshold,
            "total_trades": total_trades,
            "win_rate": round(total_wins / total_trades * 100, 1),
            "total_pnl_points": round(total_pnl, 2),
            "total_pnl_dollars": round(total_pnl * 20, 2),
            "avg_pnl_per_trade": round(avg_pnl, 2),
            "max_intraday_dd": round(max_intraday_dd, 2),
            "max_dd_from_peak": round(max_dd_from_overall_peak, 2),
            "total_tm_actions": total_tm_actions,
            "trades": all_trades,
            "daily_analysis": all_analysis,
        }

        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n[Save] Results saved to {output_path}", file=sys.stderr)
    else:
        print("\nNo trades executed.", file=sys.stderr)


if __name__ == "__main__":
    main()
