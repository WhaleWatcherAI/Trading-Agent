#!/usr/bin/env python3
"""
Multi-day XGBoost Backtest with Walk-Forward Training
Trains on previous days' data, tests on current day.
Uses options flow and IV rank features.
"""

import json
import sys
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

# Trade timeout in 1-min bars (20 bars = 20 minutes)
HOLD_BARS = 20


def load_1s_bars(filepath: str) -> List[Dict]:
    """Load 1-second bars from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("bars", [])


def load_options_flow(filepath: str, symbol: str = 'QQQ') -> pd.DataFrame:
    """Load options flow from CSV (handles both raw and processed formats)."""
    df = pd.read_csv(filepath)

    # Filter for symbol
    df = df[df['underlying_symbol'] == symbol].copy()

    # Parse datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        df['datetime'] = pd.to_datetime(df['executed_at'], format='mixed', utc=True)

    # Add bearish_or_bullish if not present
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

    # Add date/time columns if not present
    if 'date' not in df.columns:
        df['date'] = df['datetime'].dt.strftime('%m/%d/%Y')
    if 'time' not in df.columns:
        df['time'] = df['datetime'].dt.strftime('%I:%M:%S %p')

    df['flow_type'] = 'options'
    return df


def load_iv_rank(filepath: str, target_date: str) -> Dict[str, float]:
    """Load IV rank data and get previous day's values."""
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
    """Load OI changes and aggregate into sentiment features."""
    df = pd.read_csv(filepath)

    # Filter for symbol
    df = df[df['underlying_symbol'] == symbol].copy()

    if len(df) == 0:
        return {
            'oi_net_call_put': 0.0,
            'oi_total_change': 0.0,
            'oi_call_ratio': 0.5,
            'oi_bullish_flow_ratio': 0.5,
            'oi_large_positions': 0,
        }

    # Determine call vs put from option_symbol (contains C or P before strike)
    def is_call(opt_sym):
        # Option symbols like QQQ260320C00700000 - C for call, P for put
        if 'C0' in opt_sym or 'C00' in opt_sym:
            return True
        return False

    df['is_call'] = df['option_symbol'].apply(is_call)

    # Aggregate
    calls = df[df['is_call']]
    puts = df[~df['is_call']]

    call_oi_change = calls['oi_diff_plain'].sum()
    put_oi_change = puts['oi_diff_plain'].sum()
    total_oi_change = call_oi_change + put_oi_change

    # Net call-put (positive = bullish positioning)
    net_call_put = call_oi_change - put_oi_change

    # Call ratio
    call_ratio = call_oi_change / total_oi_change if total_oi_change > 0 else 0.5

    # Bullish flow ratio (ask volume = buying)
    total_ask_vol = df['prev_ask_volume'].sum()
    total_bid_vol = df['prev_bid_volume'].sum()
    total_vol = total_ask_vol + total_bid_vol
    bullish_flow_ratio = total_ask_vol / total_vol if total_vol > 0 else 0.5

    # Large positions (OI change > 5000)
    large_positions = len(df[df['oi_diff_plain'].abs() > 5000])

    return {
        'oi_net_call_put': net_call_put / 1e6,  # In millions
        'oi_total_change': total_oi_change / 1e6,
        'oi_call_ratio': call_ratio,
        'oi_bullish_flow_ratio': bullish_flow_ratio,
        'oi_large_positions': large_positions,
    }


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
    cvd_ema_1min = []
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
    """Find recent swing highs and lows for SL placement."""
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


def extract_features(
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
) -> Dict[str, float]:
    """Extract features for XGBoost."""
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
        # OI changes (previous day)
        'oi_net_call_put': oi_changes.get('oi_net_call_put', 0.0) if oi_changes else 0.0,
        'oi_total_change': oi_changes.get('oi_total_change', 0.0) if oi_changes else 0.0,
        'oi_call_ratio': oi_changes.get('oi_call_ratio', 0.5) if oi_changes else 0.5,
        'oi_bullish_flow_ratio': oi_changes.get('oi_bullish_flow_ratio', 0.5) if oi_changes else 0.5,
        'oi_large_positions': oi_changes.get('oi_large_positions', 0) if oi_changes else 0,
    }


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


def run_single_day(
    bars_1s: List[Dict],
    options_df: pd.DataFrame,
    target_date: str,
    direction_model: Any,
    tp_model: Any,
    iv_rank: Dict[str, float],
    oi_changes: Dict[str, float],
    options_premium_threshold: float,
    prediction_bars: int = 20,
    tp_threshold: float = 0.0,  # No filter - collect all for analysis
) -> Tuple[List[Dict], Dict]:
    """Run backtest for a single day using pre-trained model."""

    # Aggregate bars
    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
    bars_5min = aggregate_bars(bars_1s, 5)
    bars_15min = aggregate_bars(bars_1s, 15)
    bars_1h = aggregate_bars(bars_1s, 60)

    if len(bars_1min) < 100:
        return [], {}

    # Build timestamp -> bar index mapping
    ts_to_idx = {}
    for i, bar in enumerate(bars_1min):
        ts_str = bar['t'].replace('+00:00', '') if isinstance(bar['t'], str) else bar['t']
        ts_to_idx[ts_str] = i

    # Filter options flow for this date
    opt_date_filter = options_df['datetime'].dt.strftime('%Y-%m-%d') == target_date
    opt_premium_filter = options_df['premium'] >= options_premium_threshold
    filtered_options = options_df[opt_date_filter & opt_premium_filter].copy()

    trades = []
    current_position_exit_bar = 0

    for _, row in filtered_options.iterrows():
        whale_ts = row['datetime']
        minute_ts = whale_ts.replace(second=0, microsecond=0)
        if hasattr(minute_ts, 'tz_localize'):
            minute_ts = minute_ts.tz_localize(None)
        minute_str = minute_ts.strftime('%Y-%m-%dT%H:%M:%S')

        if minute_str not in ts_to_idx:
            continue

        bar_idx = ts_to_idx[minute_str]
        if bar_idx < 50 or bar_idx >= len(bars_1min) - prediction_bars:
            continue

        if bar_idx < current_position_exit_bar:
            continue

        current_bar = bars_1min[bar_idx]
        current_price = current_bar['c']

        # Volume Profile
        vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

        # Whale sentiment
        whale_sentiment = row.get('bearish_or_bullish', 'neutral')

        # Extract features
        features = extract_features(bars_1min, bars_5min, bars_15min, bars_1h,
                                    cvd_1min, cvd_ema_1min, bar_idx, vp, whale_sentiment, iv_rank, oi_changes)
        if features is None:
            continue

        X_pred = np.array([list(features.values())])

        # Model 1: Direction prediction (up/down)
        prob_direction = direction_model.predict_proba(X_pred)[0]
        prob_up = prob_direction[1]
        direction = "long" if prob_up > 0.5 else "short"

        # Model 2: TP hit prediction (will this trade hit take profit?)
        prob_tp = tp_model.predict_proba(X_pred)[0]
        prob_win = prob_tp[1]

        # Skip if TP model says low probability of hitting TP
        if prob_win < tp_threshold:
            continue

        # Calculate SL/TP (3:1 R:R)
        swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:bar_idx//5+1], lookback=10)

        if direction == "long":
            # For long: stop must be BELOW current price
            valid_lows = [sl for sl in swing_lows if sl < current_price] if swing_lows else []
            if valid_lows:
                stop_loss = min(valid_lows) - 1
            else:
                stop_loss = current_price - 10
            risk = current_price - stop_loss
            target = current_price + (risk * 3)
        else:
            # For short: stop must be ABOVE current price
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
            "xgb_prob_up": float(prob_up),
            "xgb_prob_tp": float(prob_win),
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


def build_training_data_direction(
    all_bars_1s: Dict[str, List[Dict]],
    all_options: Dict[str, pd.DataFrame],
    all_iv_ranks: Dict[str, Dict],
    all_oi_changes: Dict[str, Dict],
    train_dates: List[str],
    options_premium_threshold: float,
    prediction_bars: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build training data for direction prediction (up/down)."""

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

        iv_rank = all_iv_ranks.get(date, None)
        oi_changes = all_oi_changes.get(date, None)

        # Sample every 5 bars for training data
        for i in range(50, len(bars_1min) - prediction_bars, 5):
            vp = calculate_volume_profile(bars_1min[:i+1], lookback=30)
            features = extract_features(bars_1min, bars_5min, bars_15min, bars_1h,
                                        cvd_1min, cvd_ema_1min, i, vp, whale_sentiment=None, iv_rank=iv_rank, oi_changes=oi_changes)

            if features is None:
                continue

            future_price = bars_1min[i + prediction_bars]['c']
            current_price = bars_1min[i]['c']
            label = 1 if future_price > current_price else 0

            X_all.append(list(features.values()))
            y_all.append(label)

    return np.array(X_all), np.array(y_all)


def build_training_data_tp(
    all_bars_1s: Dict[str, List[Dict]],
    all_options: Dict[str, pd.DataFrame],
    all_iv_ranks: Dict[str, Dict],
    all_oi_changes: Dict[str, Dict],
    train_dates: List[str],
    options_premium_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build training data for TP hit prediction.

    Given a direction (from direction model), predict if trade hits TP.
    Label: 1 = hit TP (win), 0 = hit SL or timeout (loss)
    """

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

        iv_rank = all_iv_ranks.get(date, None)
        oi_changes = all_oi_changes.get(date, None)

        # Sample every 5 bars (need HOLD_BARS ahead for trade simulation)
        for i in range(50, len(bars_1min) - HOLD_BARS, 5):
            vp = calculate_volume_profile(bars_1min[:i+1], lookback=30)
            features = extract_features(bars_1min, bars_5min, bars_15min, bars_1h,
                                        cvd_1min, cvd_ema_1min, i, vp, whale_sentiment=None, iv_rank=iv_rank, oi_changes=oi_changes)

            if features is None:
                continue

            current_price = bars_1min[i]['c']

            # Direction from CVD (same as direction model would predict)
            cvd_now = cvd_1min[i] if i < len(cvd_1min) else 0
            cvd_ema_now = cvd_ema_1min[i] if i < len(cvd_ema_1min) else 0
            direction = "long" if cvd_now > cvd_ema_now else "short"

            # Calculate SL/TP (3:1 R:R)
            swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:i//5+1], lookback=10)

            if direction == "long":
                # For long: stop must be BELOW current price
                valid_lows = [sl for sl in swing_lows if sl < current_price] if swing_lows else []
                if valid_lows:
                    stop_loss = min(valid_lows) - 1
                else:
                    stop_loss = current_price - 10
                risk = current_price - stop_loss
                target = current_price + (risk * 3)
            else:
                # For short: stop must be ABOVE current price
                valid_highs = [sh for sh in swing_highs if sh > current_price] if swing_highs else []
                if valid_highs:
                    stop_loss = max(valid_highs) + 1
                else:
                    stop_loss = current_price + 10
                risk = stop_loss - current_price
                target = current_price - (risk * 3)

            # Simulate the trade
            result = simulate_trade(bars_1min, i, direction, current_price, stop_loss, target, max_bars=HOLD_BARS)

            # Label: 1 = hit TP, 0 = hit SL or timeout
            label = 1 if result['outcome'] == 'win' else 0

            X_all.append(list(features.values()))
            y_all.append(label)

    return np.array(X_all), np.array(y_all)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bars', default='data/bars_1s_nq.json', help='Path to bars JSON')
    parser.add_argument('--options-dir', default='data/options_flow', help='Directory with options flow CSVs')
    parser.add_argument('--ivrank', default=None, help='Path to IV rank CSV file')
    parser.add_argument('--oi-dir', default=None, help='Directory with OI changes CSVs')
    parser.add_argument('--symbol', default='QQQ', help='ETF symbol')
    parser.add_argument('--options-premium', type=float, default=322040, help='Min options premium threshold')
    args = parser.parse_args()

    print("=" * 70, file=sys.stderr)
    print("MULTI-DAY XGBOOST WALK-FORWARD BACKTEST", file=sys.stderr)
    print(f"Options premium threshold: ${args.options_premium:,.0f}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Load all bars data
    print("[Load] Loading bars data...", file=sys.stderr)
    all_bars_raw = load_1s_bars(args.bars)

    # Group bars by date
    all_bars_1s = defaultdict(list)
    for bar in all_bars_raw:
        date = bar['t'][:10]
        all_bars_1s[date].append(bar)

    available_dates = sorted(all_bars_1s.keys())
    print(f"[Load] Found bars for dates: {available_dates}", file=sys.stderr)

    # Load all options flow
    print("[Load] Loading options flow...", file=sys.stderr)
    all_options = {}
    for date in available_dates:
        # Try different file naming patterns
        for pattern in [f'bot-eod-report-{date}.csv', f'options_flow_{date}.csv']:
            filepath = os.path.join(args.options_dir, pattern)
            if os.path.exists(filepath):
                df = load_options_flow(filepath, args.symbol)
                all_options[date] = df
                print(f"  {date}: {len(df)} {args.symbol} options records", file=sys.stderr)
                break

    # Load IV rank
    all_iv_ranks = {}
    if args.ivrank and os.path.exists(args.ivrank):
        print("[Load] Loading IV rank data...", file=sys.stderr)
        for date in available_dates:
            iv_data = load_iv_rank(args.ivrank, date)
            all_iv_ranks[date] = iv_data
            print(f"  {date}: IV rank 1Y={iv_data['iv_rank_1y']:.1f}, 1M={iv_data['iv_rank_1m']:.1f}", file=sys.stderr)

    # Load OI changes (previous day's data to avoid leakage)
    all_oi_changes = {}
    if args.oi_dir and os.path.exists(args.oi_dir):
        print("[Load] Loading OI changes data (previous day)...", file=sys.stderr)
        # Get all available OI change dates
        oi_files = {}
        for f in os.listdir(args.oi_dir):
            if f.startswith('chain-oi-changes-') and f.endswith('.csv'):
                oi_date = f.replace('chain-oi-changes-', '').replace('.csv', '')
                oi_files[oi_date] = os.path.join(args.oi_dir, f)

        sorted_oi_dates = sorted(oi_files.keys())

        for date in available_dates:
            # Find the most recent OI date before target date (previous day's data)
            prev_oi_dates = [d for d in sorted_oi_dates if d < date]
            if prev_oi_dates:
                prev_oi_date = prev_oi_dates[-1]
                oi_data = load_oi_changes(oi_files[prev_oi_date], args.symbol)
                all_oi_changes[date] = oi_data
                print(f"  {date}: Using OI from {prev_oi_date} - net_call_put={oi_data['oi_net_call_put']:.2f}M, call_ratio={oi_data['oi_call_ratio']:.2%}", file=sys.stderr)

    # Walk-forward: train on previous days, test on current day
    all_trades = []
    all_analyses = []

    for i, test_date in enumerate(available_dates):
        train_dates = available_dates[:i]  # All previous dates

        if len(train_dates) == 0:
            print(f"\n[Day {i+1}] {test_date}: No previous days - using first 60% of day for training", file=sys.stderr)
            train_dates = [test_date]  # Will use within-day split
        else:
            print(f"\n[Day {i+1}] {test_date}: Training on {train_dates}", file=sys.stderr)

        # Build training data for DIRECTION model (up/down)
        X_train_dir, y_train_dir = build_training_data_direction(
            all_bars_1s, all_options, all_iv_ranks, all_oi_changes,
            train_dates, args.options_premium
        )

        if len(X_train_dir) < 50:
            print(f"  Not enough training data ({len(X_train_dir)} samples)", file=sys.stderr)
            continue

        # Build training data for TP model (will it hit take profit?)
        X_train_tp, y_train_tp = build_training_data_tp(
            all_bars_1s, all_options, all_iv_ranks, all_oi_changes,
            train_dates, args.options_premium
        )

        print(f"  Training direction model on {len(X_train_dir)} samples...", file=sys.stderr)
        print(f"  Training TP model on {len(X_train_tp)} samples...", file=sys.stderr)

        # Train DIRECTION model
        direction_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        direction_model.fit(X_train_dir, y_train_dir, verbose=False)

        # Train TP model
        tp_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        tp_model.fit(X_train_tp, y_train_tp, verbose=False)

        # Test on current day
        if test_date not in all_options:
            print(f"  No options data for {test_date}", file=sys.stderr)
            continue

        trades, analysis = run_single_day(
            all_bars_1s[test_date],
            all_options[test_date],
            test_date,
            direction_model,
            tp_model,
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

    print("=" * 70, file=sys.stderr)

    # Save results
    result = {
        "summary": {
            "total_trades": total_trades,
            "total_wins": sum(1 for t in all_trades if t['outcome'] == 'win') if total_trades > 0 else 0,
            "win_rate": round(sum(1 for t in all_trades if t['outcome'] == 'win') / total_trades * 100, 1) if total_trades > 0 else 0,
            "total_pnl_points": round(sum(t['pnl_points'] for t in all_trades), 2) if total_trades > 0 else 0,
        },
        "daily_analysis": all_analyses,
        "trades": all_trades,
    }

    output_path = f"data/xgboost_multiday_{args.symbol}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Save] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
