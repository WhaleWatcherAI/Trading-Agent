#!/usr/bin/env python3
"""
XGBoost Backtest triggered by Whale Flow Events
Uses same data as Qwen: CVD, CVD trend, Volume Profile, Multi-TF candles, Whale flow

Usage:
    python xgboost_whale_trigger_backtest.py
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


def load_1s_bars(filepath: str) -> List[Dict]:
    """Load 1-second bars from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("bars", [])


def load_whale_options_flow(filepath: str) -> pd.DataFrame:
    """Load whale options flow from CSV."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    df['flow_type'] = 'options'
    return df


def load_whale_lit_flow(filepath: str) -> pd.DataFrame:
    """Load whale lit/stock flow from CSV."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    df['flow_type'] = 'lit'

    def classify_trade(row):
        price = row['price']
        bid = row['nbbo_bid']
        ask = row['nbbo_ask']
        if price >= ask:
            return 'bullish'
        elif price <= bid:
            return 'bearish'
        else:
            return 'neutral'

    df['bearish_or_bullish'] = df.apply(classify_trade, axis=1)
    return df


def load_iv_rank(filepath: str, target_date: str) -> Dict[str, float]:
    """Load IV rank data and get previous day's values for the target date."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Get the previous trading day's IV rank
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


def load_darkpool_data(filepath: str, symbol: str, target_date: str) -> pd.DataFrame:
    """Load dark pool EOD data for a specific symbol and aggregate to 1-min bars."""
    df = pd.read_csv(filepath)

    # Filter for symbol
    df = df[df['ticker'] == symbol].copy()
    if len(df) == 0:
        return pd.DataFrame()

    # Parse datetime (mixed formats - some with ms, some without)
    df['datetime'] = pd.to_datetime(df['executed_at'], format='mixed', utc=True)
    df = df[df['datetime'].dt.strftime('%Y-%m-%d') == target_date]

    # Classify as buy/sell based on price vs bid/ask
    def classify_dp_trade(row):
        price = row['price']
        bid = row['nbbo_bid']
        ask = row['nbbo_ask']
        mid = (bid + ask) / 2
        if price >= ask:
            return 'buy'
        elif price <= bid:
            return 'sell'
        elif price > mid:
            return 'buy'
        else:
            return 'sell'

    df['side'] = df.apply(classify_dp_trade, axis=1)
    df['signed_premium'] = df.apply(lambda r: r['premium'] if r['side'] == 'buy' else -r['premium'], axis=1)

    # Aggregate to 1-minute bars
    df['minute'] = df['datetime'].dt.floor('T')

    dp_1min = df.groupby('minute').agg({
        'premium': 'sum',
        'signed_premium': 'sum',
        'size': 'sum',
        'ticker': 'count',  # trade count
    }).rename(columns={'ticker': 'dp_count', 'premium': 'dp_premium', 'size': 'dp_volume'})

    dp_1min['dp_net_premium'] = dp_1min['signed_premium']
    dp_1min = dp_1min.drop(columns=['signed_premium'])

    return dp_1min


def build_dp_context(dp_1min: pd.DataFrame, bars_1min: List[Dict]) -> Dict[str, List[float]]:
    """Build rolling dark pool context features aligned to 1-min bars."""
    if dp_1min.empty:
        n = len(bars_1min)
        return {
            'dp_cumulative_premium': [0.0] * n,
            'dp_cumulative_net': [0.0] * n,
            'dp_rolling_premium_10': [0.0] * n,
            'dp_rolling_net_10': [0.0] * n,
            'dp_rolling_count_10': [0.0] * n,
        }

    # Build timestamp index
    ts_to_dp = {}
    for ts, row in dp_1min.iterrows():
        ts_str = ts.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(ts, 'strftime') else str(ts)
        ts_to_dp[ts_str] = row

    dp_cumulative_premium = []
    dp_cumulative_net = []
    dp_rolling_premium_10 = []
    dp_rolling_net_10 = []
    dp_rolling_count_10 = []

    cum_premium = 0.0
    cum_net = 0.0
    recent_premium = []
    recent_net = []
    recent_count = []

    for bar in bars_1min:
        ts_str = bar['t'].replace('+00:00', '') if isinstance(bar['t'], str) else bar['t'].isoformat()

        if ts_str in ts_to_dp:
            row = ts_to_dp[ts_str]
            cum_premium += row['dp_premium']
            cum_net += row['dp_net_premium']
            recent_premium.append(row['dp_premium'])
            recent_net.append(row['dp_net_premium'])
            recent_count.append(row['dp_count'])
        else:
            recent_premium.append(0.0)
            recent_net.append(0.0)
            recent_count.append(0.0)

        # Keep only last 10
        if len(recent_premium) > 10:
            recent_premium = recent_premium[-10:]
            recent_net = recent_net[-10:]
            recent_count = recent_count[-10:]

        dp_cumulative_premium.append(cum_premium)
        dp_cumulative_net.append(cum_net)
        dp_rolling_premium_10.append(sum(recent_premium))
        dp_rolling_net_10.append(sum(recent_net))
        dp_rolling_count_10.append(sum(recent_count))

    return {
        'dp_cumulative_premium': dp_cumulative_premium,
        'dp_cumulative_net': dp_cumulative_net,
        'dp_rolling_premium_10': dp_rolling_premium_10,
        'dp_rolling_net_10': dp_rolling_net_10,
        'dp_rolling_count_10': dp_rolling_count_10,
    }


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

        if period_minutes >= 1440:
            period_ts = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period_minutes >= 60:
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
    dp_context: Dict[str, List[float]] = None,
) -> Dict[str, float]:
    """
    Extract features using ONLY the same data as Qwen:
    - CVD and CVD trend
    - Volume Profile (POC, VAH, VAL)
    - Multi-timeframe candle data
    - Whale sentiment
    - Time of day
    """
    if bar_idx < 30:
        return None

    current_bar = bars_1min[bar_idx]
    current_price = current_bar['c']

    # Time-of-day features
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
    cvd_vs_ema = cvd_now - cvd_ema_now  # Raw difference
    cvd_trend = 1 if cvd_now > cvd_ema_now else -1 if cvd_now < cvd_ema_now else 0
    cvd_slope_5 = (cvd_1min[bar_idx] - cvd_1min[bar_idx-5]) if bar_idx >= 5 else 0
    cvd_slope_10 = (cvd_1min[bar_idx] - cvd_1min[bar_idx-10]) if bar_idx >= 10 else 0

    # Volume Profile features
    price_vs_poc = (current_price - vp['poc']) / vp['poc'] if vp['poc'] > 0 else 0
    price_vs_vah = (current_price - vp['vah']) / vp['vah'] if vp['vah'] > 0 else 0
    price_vs_val = (current_price - vp['val']) / vp['val'] if vp['val'] > 0 else 0
    in_value_area = 1 if vp['val'] <= current_price <= vp['vah'] else 0
    above_poc = 1 if current_price > vp['poc'] else 0

    # 1-min candle features (last 5 bars)
    recent_1m = bars_1min[bar_idx-4:bar_idx+1]
    bullish_1m = sum(1 for b in recent_1m if b['c'] > b['o']) / 5
    body_size_1m = np.mean([abs(b['c'] - b['o']) for b in recent_1m])
    range_1m = np.mean([b['h'] - b['l'] for b in recent_1m])
    close_position_1m = np.mean([(b['c'] - b['l']) / (b['h'] - b['l']) if b['h'] > b['l'] else 0.5 for b in recent_1m])

    # 5-min candle features (last 3 bars)
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

    # 15-min candle features (last 2 bars)
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

    # Whale sentiment as features
    whale_bullish = 1 if whale_sentiment == 'bullish' else 0
    whale_bearish = 1 if whale_sentiment == 'bearish' else 0

    # Price momentum (simple)
    price_change_5 = (current_price - bars_1min[bar_idx-5]['c']) / bars_1min[bar_idx-5]['c'] if bar_idx >= 5 else 0
    price_change_10 = (current_price - bars_1min[bar_idx-10]['c']) / bars_1min[bar_idx-10]['c'] if bar_idx >= 10 else 0
    price_change_20 = (current_price - bars_1min[bar_idx-20]['c']) / bars_1min[bar_idx-20]['c'] if bar_idx >= 20 else 0

    # Recent high/low position
    recent_high = max(b['h'] for b in bars_1min[bar_idx-20:bar_idx+1])
    recent_low = min(b['l'] for b in bars_1min[bar_idx-20:bar_idx+1])
    price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5

    # IV rank features (previous day's values - constant for the day)
    if iv_rank is not None:
        iv_rank_1y = iv_rank.get('iv_rank_1y', 0) / 100.0  # Normalize to 0-1
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
        # Time features
        'time_sin': time_sin,
        'time_cos': time_cos,
        'is_open_30min': is_open_30min,
        'is_morning': is_morning,
        'is_lunch': is_lunch,
        'is_afternoon': is_afternoon,
        'is_close_30min': is_close_30min,
        # CVD features
        'cvd_trend': cvd_trend,
        'cvd_vs_ema': cvd_vs_ema,
        'cvd_slope_5': cvd_slope_5,
        'cvd_slope_10': cvd_slope_10,
        # Volume Profile features
        'price_vs_poc': price_vs_poc,
        'price_vs_vah': price_vs_vah,
        'price_vs_val': price_vs_val,
        'in_value_area': in_value_area,
        'above_poc': above_poc,
        # 1-min candle features
        'bullish_1m': bullish_1m,
        'body_size_1m': body_size_1m,
        'range_1m': range_1m,
        'close_position_1m': close_position_1m,
        # 5-min candle features
        'bullish_5m': bullish_5m,
        'body_size_5m': body_size_5m,
        'close_position_5m': close_position_5m,
        # 15-min candle features
        'bullish_15m': bullish_15m,
        'close_position_15m': close_position_15m,
        # 1-hour candle features
        'bullish_1h': bullish_1h,
        'close_position_1h': close_position_1h,
        # Whale sentiment
        'whale_bullish': whale_bullish,
        'whale_bearish': whale_bearish,
        # Price momentum
        'price_change_5': price_change_5,
        'price_change_10': price_change_10,
        'price_change_20': price_change_20,
        'price_position': price_position,
        # IV rank features (previous day)
        'iv_rank_1y': iv_rank_1y,
        'iv_percentile_1y': iv_percentile_1y,
        'iv_rank_1m': iv_rank_1m,
        'iv_percentile_1m': iv_percentile_1m,
        'volatility': volatility,
        # Dark pool context features
        'dp_rolling_premium_10': dp_context['dp_rolling_premium_10'][bar_idx] / 1e6 if dp_context else 0.0,  # in millions
        'dp_rolling_net_10': dp_context['dp_rolling_net_10'][bar_idx] / 1e6 if dp_context else 0.0,
        'dp_rolling_count_10': dp_context['dp_rolling_count_10'][bar_idx] if dp_context else 0.0,
        'dp_net_bias': (dp_context['dp_rolling_net_10'][bar_idx] / max(dp_context['dp_rolling_premium_10'][bar_idx], 1)) if dp_context and dp_context['dp_rolling_premium_10'][bar_idx] > 0 else 0.0,
    }


def get_features_for_bar(
    bars_1min: List[Dict],
    bars_5min: List[Dict],
    bars_15min: List[Dict],
    bars_1h: List[Dict],
    cvd_1min: np.ndarray,
    cvd_ema_1min: np.ndarray,
    bar_idx: int,
) -> List[float]:
    """Get feature vector for a specific bar (for in-trade predictions)."""
    if bar_idx < 30 or bar_idx >= len(bars_1min):
        return None

    # Calculate VP for this bar
    vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

    features_dict = extract_features(
        bars_1min, bars_5min, bars_15min, bars_1h,
        list(cvd_1min), list(cvd_ema_1min), bar_idx, vp, whale_sentiment=None
    )

    if features_dict is None:
        return None

    # Return as list in consistent order
    return list(features_dict.values())


def simulate_trade(
    bars_1min: List[Dict],
    entry_idx: int,
    direction: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    max_bars: int = 60,
    model: Any = None,
    get_features_fn: Any = None,
    bars_5min: List[Dict] = None,
    bars_15min: List[Dict] = None,
    bars_1h: List[Dict] = None,
    cvd_1min: np.ndarray = None,
    cvd_ema_1min: np.ndarray = None,
) -> Dict[str, Any]:
    """Simulate a trade with fixed SL/TP and optional XGB flip exit."""
    pnl = 0.0
    exit_price = entry_price
    outcome = "timeout"
    bars_held = 0
    max_profit = 0.0
    exit_reason = "timeout"

    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(bars_1min))):
        bar = bars_1min[i]
        bars_held = i - entry_idx

        if direction == "long":
            current_profit = bar['h'] - entry_price
            max_profit = max(max_profit, current_profit)

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
            current_profit = entry_price - bar['l']
            max_profit = max(max_profit, current_profit)

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

        # Check for XGB direction flip (if model provided)
        if model is not None and get_features_fn is not None and bars_held >= 1:
            try:
                features = get_features_fn(
                    bars_1min, bars_5min, bars_15min, bars_1h,
                    cvd_1min, cvd_ema_1min, i
                )
                prob = model.predict_proba([features])[0]
                prob_up = prob[1]
                new_direction = "long" if prob_up > 0.5 else "short"

                # If direction flipped, close early
                if new_direction != direction:
                    exit_price = bar['c']
                    if direction == "long":
                        pnl = exit_price - entry_price
                    else:
                        pnl = entry_price - exit_price
                    outcome = "win" if pnl > 0 else "loss"
                    exit_reason = "flip"
                    break
            except:
                pass  # If features fail, continue with SL/TP logic

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
        "max_profit": max_profit,
        "exit_reason": exit_reason,
    }


def run_backtest(
    bars_1s: List[Dict],
    options_df: pd.DataFrame,
    lit_df: pd.DataFrame,
    target_date: str,
    options_premium_threshold: float = 100000,
    lit_premium_threshold: float = 2000000,
    prediction_bars: int = 20,
    min_confidence: float = 0.0,
    iv_rank: Dict[str, float] = None,
    dp_context: Dict[str, List[float]] = None,
) -> Tuple[List[Dict], Dict]:
    """Run backtest triggered by whale flow with walk-forward XGBoost."""

    # Aggregate bars to multiple timeframes
    print(f"[Backtest] Aggregating {len(bars_1s)} 1s bars...", file=sys.stderr)
    bars_1min, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
    bars_5min = aggregate_bars(bars_1s, 5)
    bars_15min = aggregate_bars(bars_1s, 15)
    bars_1h = aggregate_bars(bars_1s, 60)
    print(f"[Backtest] Got {len(bars_1min)} 1m, {len(bars_5min)} 5m, {len(bars_15min)} 15m, {len(bars_1h)} 1h bars", file=sys.stderr)

    if len(bars_1min) < 100:
        print("[Backtest] Not enough data", file=sys.stderr)
        return [], {}

    # Build dark pool context if raw DP data was passed (it's a DataFrame)
    if dp_context is not None and isinstance(dp_context, pd.DataFrame):
        dp_context = build_dp_context(dp_context, bars_1min)
        print(f"[Backtest] Built DP context with {len(dp_context.get('dp_cumulative_premium', []))} bars", file=sys.stderr)

    # Build timestamp -> bar index mapping
    ts_to_idx = {}
    for i, bar in enumerate(bars_1min):
        ts_str = bar['t']
        if isinstance(ts_str, str):
            ts_str = ts_str.replace('+00:00', '')
        ts_to_idx[ts_str] = i

    # Filter and combine options + lit flow
    opt_date_filter = options_df['datetime'].dt.strftime('%Y-%m-%d') == target_date
    opt_premium_filter = options_df['premium'] >= options_premium_threshold
    filtered_options = options_df[opt_date_filter & opt_premium_filter].copy()
    print(f"[Backtest] Found {len(filtered_options)} options triggers (>= ${options_premium_threshold:,.0f})", file=sys.stderr)

    lit_date_filter = lit_df['datetime'].dt.strftime('%Y-%m-%d') == target_date
    lit_premium_filter = lit_df['premium'] >= lit_premium_threshold
    lit_valid = lit_df['bearish_or_bullish'].isin(['bullish', 'bearish'])
    filtered_lit = lit_df[lit_date_filter & lit_premium_filter & lit_valid].copy()
    print(f"[Backtest] Found {len(filtered_lit)} lit/stock triggers (>= ${lit_premium_threshold:,.0f})", file=sys.stderr)

    combined_flow = pd.concat([
        filtered_options[['datetime', 'premium', 'bearish_or_bullish', 'flow_type']],
        filtered_lit[['datetime', 'premium', 'bearish_or_bullish', 'flow_type']]
    ]).sort_values('datetime')

    print(f"[Backtest] Total combined triggers: {len(combined_flow)}", file=sys.stderr)

    # Build training data (walk-forward: first 60%)
    print("[Backtest] Building training features...", file=sys.stderr)
    train_end_idx = int(len(bars_1min) * 0.6)

    X_train = []
    y_train = []

    for i in range(50, train_end_idx - prediction_bars):
        vp = calculate_volume_profile(bars_1min[:i+1], lookback=30)
        features = extract_features(bars_1min, bars_5min, bars_15min, bars_1h,
                                    cvd_1min, cvd_ema_1min, i, vp, whale_sentiment=None, iv_rank=iv_rank, dp_context=dp_context)

        if features is None:
            continue

        future_price = bars_1min[i + prediction_bars]['c']
        current_price = bars_1min[i]['c']
        label = 1 if future_price > current_price else 0

        X_train.append(list(features.values()))
        y_train.append(label)

    if len(X_train) < 100:
        print("[Backtest] Not enough training data", file=sys.stderr)
        return [], {}

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"[Backtest] Training XGBoost on {len(X_train)} samples...", file=sys.stderr)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    )
    model.fit(X_train, y_train)

    print("[Backtest] Model trained. Running predictions...", file=sys.stderr)

    trades = []
    current_position_exit_bar = 0  # Track when current position exits

    for _, row in combined_flow.iterrows():
        whale_ts = row['datetime']
        minute_ts = whale_ts.replace(second=0, microsecond=0).isoformat().replace('+00:00', '')

        if minute_ts not in ts_to_idx:
            continue

        bar_idx = ts_to_idx[minute_ts]
        if bar_idx < 50 or bar_idx >= len(bars_1min) - prediction_bars:
            continue

        # Skip if still in a position
        if bar_idx < current_position_exit_bar:
            continue

        current_bar = bars_1min[bar_idx]
        current_price = current_bar['c']

        # CVD trend
        cvd_now = cvd_1min[bar_idx] if bar_idx < len(cvd_1min) else 0
        cvd_ema_now = cvd_ema_1min[bar_idx] if bar_idx < len(cvd_ema_1min) else 0
        cvd_trend = "up" if cvd_now > cvd_ema_now else "down" if cvd_now < cvd_ema_now else "neutral"

        # Volume Profile
        vp = calculate_volume_profile(bars_1min[:bar_idx+1], lookback=30)

        # Whale flow info
        whale_sentiment = row.get('bearish_or_bullish', 'neutral')
        flow_type = row.get('flow_type', 'unknown')

        # Extract features and predict
        features = extract_features(bars_1min, bars_5min, bars_15min, bars_1h,
                                    cvd_1min, cvd_ema_1min, bar_idx, vp, whale_sentiment, iv_rank=iv_rank, dp_context=dp_context)
        if features is None:
            continue

        X_pred = np.array([list(features.values())])
        prob = model.predict_proba(X_pred)[0]

        prob_up = prob[1]
        confidence = max(prob_up, 1 - prob_up)

        # HOLD if confidence below threshold
        if confidence < min_confidence:
            continue  # Skip this trigger - model says HOLD

        # Decide direction
        if prob_up > 0.5:
            direction = "long"
        else:
            direction = "short"

        # Calculate SL/TP
        swing_highs, swing_lows = find_swing_highs_lows(bars_5min[:bar_idx//5+1], lookback=10)

        if direction == "long":
            if swing_lows:
                stop_loss = min(swing_lows) - 1
            else:
                stop_loss = current_price - 10
            risk = current_price - stop_loss
            target = current_price + (risk * 3)
        else:
            if swing_highs:
                stop_loss = max(swing_highs) + 1
            else:
                stop_loss = current_price + 10
            risk = stop_loss - current_price
            target = current_price - (risk * 3)

        # Simulate trade
        result = simulate_trade(
            bars_1min, bar_idx, direction, current_price, stop_loss, target, max_bars=60
        )

        # Update position exit bar (single position mode)
        current_position_exit_bar = bar_idx + result["bars_held"]

        trade = {
            "timestamp": minute_ts,
            "bar_index": bar_idx,
            "direction": direction,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "target": target,
            "risk_reward": 3.0,
            "flow_type": flow_type,
            "whale_sentiment": whale_sentiment,
            "xgb_prob_up": float(prob_up),
            "xgb_confidence": float(confidence),
            "cvd_trend": cvd_trend,
            "poc": vp["poc"],
            "outcome": result["outcome"],
            "exit_price": result["exit_price"],
            "pnl_points": result["pnl_points"],
            "bars_held": result["bars_held"],
            "max_profit": result["max_profit"],
            "exit_reason": result.get("exit_reason", "unknown"),
        }
        trades.append(trade)
        exit_tag = f"[{result.get('exit_reason', '?')}]" if result.get('exit_reason') else ""
        print(f"[Trade {len(trades)}] {minute_ts} {direction.upper()} @ {current_price:.2f} | {flow_type} {whale_sentiment} | XGB: {prob_up:.1%} up | {result['outcome'].upper()} {result['pnl_points']:+.2f} pts {exit_tag}", file=sys.stderr)

    # Analysis
    if trades:
        wins = sum(1 for t in trades if t["outcome"] == "win")
        total_pnl = sum(t["pnl_points"] for t in trades)
        analysis = {
            "total_trades": len(trades),
            "wins": wins,
            "losses": len(trades) - wins,
            "win_rate": round(wins / len(trades) * 100, 1),
            "total_pnl_points": round(total_pnl, 2),
            "avg_pnl_per_trade": round(total_pnl / len(trades), 2),
        }

        # By direction
        longs = [t for t in trades if t["direction"] == "long"]
        shorts = [t for t in trades if t["direction"] == "short"]
        analysis["by_direction"] = {
            "long": {"count": len(longs), "wins": sum(1 for t in longs if t["outcome"] == "win"), "pnl": round(sum(t["pnl_points"] for t in longs), 2)},
            "short": {"count": len(shorts), "wins": sum(1 for t in shorts if t["outcome"] == "win"), "pnl": round(sum(t["pnl_points"] for t in shorts), 2)},
        }

        # By flow type
        opt_trades = [t for t in trades if t["flow_type"] == "options"]
        lit_trades = [t for t in trades if t["flow_type"] == "lit"]
        analysis["by_flow_type"] = {
            "options": {"count": len(opt_trades), "wins": sum(1 for t in opt_trades if t["outcome"] == "win"), "pnl": round(sum(t["pnl_points"] for t in opt_trades), 2)},
            "lit": {"count": len(lit_trades), "wins": sum(1 for t in lit_trades if t["outcome"] == "win"), "pnl": round(sum(t["pnl_points"] for t in lit_trades), 2)},
        }

        # By confidence
        high_conf = [t for t in trades if t["xgb_confidence"] >= 0.6]
        low_conf = [t for t in trades if t["xgb_confidence"] < 0.6]
        analysis["by_confidence"] = {
            "high_60+": {"count": len(high_conf), "wins": sum(1 for t in high_conf if t["outcome"] == "win"), "pnl": round(sum(t["pnl_points"] for t in high_conf), 2)},
            "low_<60": {"count": len(low_conf), "wins": sum(1 for t in low_conf if t["outcome"] == "win"), "pnl": round(sum(t["pnl_points"] for t in low_conf), 2)},
        }

        # By whale agreement
        whale_agree = [t for t in trades if
                       (t["direction"] == "long" and t["whale_sentiment"] == "bullish") or
                       (t["direction"] == "short" and t["whale_sentiment"] == "bearish")]
        whale_disagree = [t for t in trades if
                         (t["direction"] == "long" and t["whale_sentiment"] == "bearish") or
                         (t["direction"] == "short" and t["whale_sentiment"] == "bullish")]
        analysis["by_whale_agreement"] = {
            "agree": {"count": len(whale_agree), "wins": sum(1 for t in whale_agree if t["outcome"] == "win"), "pnl": round(sum(t["pnl_points"] for t in whale_agree), 2)},
            "disagree": {"count": len(whale_disagree), "wins": sum(1 for t in whale_disagree if t["outcome"] == "win"), "pnl": round(sum(t["pnl_points"] for t in whale_disagree), 2)},
        }

        # By time of day
        def get_time_period(ts):
            dt = datetime.fromisoformat(ts)
            hour = dt.hour
            if hour < 10:
                return "open"
            elif hour < 12:
                return "morning"
            elif hour < 14:
                return "lunch"
            elif hour < 16:
                return "afternoon"
            else:
                return "close"

        for period in ["open", "morning", "lunch", "afternoon", "close"]:
            period_trades = [t for t in trades if get_time_period(t["timestamp"]) == period]
            if period_trades:
                analysis[f"time_{period}"] = {
                    "count": len(period_trades),
                    "wins": sum(1 for t in period_trades if t["outcome"] == "win"),
                    "pnl": round(sum(t["pnl_points"] for t in period_trades), 2),
                }
    else:
        analysis = {"total_trades": 0}

    return trades, analysis


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bars', default='data/bars_1s.json', help='Path to bars JSON')
    parser.add_argument('--options', default='data/qqq_options_flow_20251205.csv', help='Options flow CSV')
    parser.add_argument('--lit', default=None, help='Lit flow CSV (optional)')
    parser.add_argument('--date', default='2025-12-05', help='Target date YYYY-MM-DD')
    parser.add_argument('--instrument', default='NQ/QQQ', help='Instrument name for output')
    parser.add_argument('--min-confidence', type=float, default=0.0, help='Min confidence to trade (0-1). Below this = HOLD')
    parser.add_argument('--options-premium', type=float, default=None, help='Min options premium threshold')
    parser.add_argument('--lit-premium', type=float, default=None, help='Min lit/stock premium threshold')
    parser.add_argument('--symbol', default='QQQ', choices=['QQQ', 'SPY'], help='ETF symbol for auto thresholds')
    parser.add_argument('--ivrank', default=None, help='Path to IV rank CSV file')
    parser.add_argument('--darkpool', default=None, help='Path to dark pool EOD CSV file')
    args = parser.parse_args()

    # Prefiltered thresholds by symbol (minimum values from your filtered data)
    THRESHOLDS = {
        'QQQ': {'options': 322040, 'lit': 2004948},
        'SPY': {'options': 325710, 'lit': 701686},
    }

    # Configuration
    TARGET_DATE = args.date
    # Use CLI override or auto-detect from symbol
    OPTIONS_PREMIUM_THRESHOLD = args.options_premium if args.options_premium is not None else THRESHOLDS[args.symbol]['options']
    LIT_PREMIUM_THRESHOLD = args.lit_premium if args.lit_premium is not None else THRESHOLDS[args.symbol]['lit']

    # Paths
    bars_1s_path = args.bars
    options_flow_path = args.options
    lit_flow_path = args.lit
    output_path = f"data/xgboost_{args.instrument.replace('/', '_')}_{TARGET_DATE.replace('-', '')}.json"

    print("=" * 70, file=sys.stderr)
    print("XGBOOST + WHALE FLOW TRIGGER BACKTEST", file=sys.stderr)
    print(f"Features: CVD, VP, Multi-TF Candles, Whale Sentiment, Time-of-Day", file=sys.stderr)
    print(f"Date: {TARGET_DATE}", file=sys.stderr)
    print(f"Options premium: ${OPTIONS_PREMIUM_THRESHOLD:,.0f} | Lit: ${LIT_PREMIUM_THRESHOLD:,.0f}", file=sys.stderr)
    print(f"Min confidence to trade: {args.min_confidence*100:.0f}% (below = HOLD)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Load data
    print(f"[Load] Loading 1s bars from {bars_1s_path}...", file=sys.stderr)
    bars_1s_all = load_1s_bars(bars_1s_path)
    print(f"[Load] Loaded {len(bars_1s_all)} total 1-second bars", file=sys.stderr)

    bars_1s = [b for b in bars_1s_all if TARGET_DATE in b['t']]
    print(f"[Load] Filtered to {len(bars_1s)} bars on {TARGET_DATE}", file=sys.stderr)

    print(f"[Load] Loading whale options flow...", file=sys.stderr)
    options_df = load_whale_options_flow(options_flow_path)
    print(f"[Load] Loaded {len(options_df)} options flow records", file=sys.stderr)

    if lit_flow_path:
        print(f"[Load] Loading whale lit flow...", file=sys.stderr)
        lit_df = load_whale_lit_flow(lit_flow_path)
        print(f"[Load] Loaded {len(lit_df)} lit flow records", file=sys.stderr)
    else:
        print(f"[Load] No lit flow provided", file=sys.stderr)
        lit_df = pd.DataFrame({'datetime': pd.Series(dtype='datetime64[ns]'),
                               'premium': pd.Series(dtype='float64'),
                               'bearish_or_bullish': pd.Series(dtype='str'),
                               'flow_type': pd.Series(dtype='str')})

    # Load IV rank data if provided
    iv_rank_data = None
    if args.ivrank:
        print(f"[Load] Loading IV rank data from {args.ivrank}...", file=sys.stderr)
        iv_rank_data = load_iv_rank(args.ivrank, TARGET_DATE)
        print(f"[Load] IV rank (prev day): 1Y rank={iv_rank_data['iv_rank_1y']:.1f}, 1M rank={iv_rank_data['iv_rank_1m']:.1f}, vol={iv_rank_data['volatility']:.4f}", file=sys.stderr)
    else:
        print(f"[Load] No IV rank file provided - using zeros", file=sys.stderr)

    # Load dark pool data if provided (need bars_1min first for alignment)
    dp_context_data = None
    if args.darkpool:
        print(f"[Load] Loading dark pool data for {args.symbol} from {args.darkpool}...", file=sys.stderr)
        dp_1min = load_darkpool_data(args.darkpool, args.symbol, TARGET_DATE)
        print(f"[Load] Dark pool: {len(dp_1min)} minute bars with DP activity", file=sys.stderr)
        # We'll build the context inside run_backtest after bars are aggregated
        # For now, pass the raw 1min data
        dp_context_data = dp_1min
    else:
        print(f"[Load] No dark pool file provided", file=sys.stderr)

    # Run backtest
    trades, analysis = run_backtest(
        bars_1s=bars_1s,
        options_df=options_df,
        lit_df=lit_df,
        target_date=TARGET_DATE,
        options_premium_threshold=OPTIONS_PREMIUM_THRESHOLD,
        lit_premium_threshold=LIT_PREMIUM_THRESHOLD,
        min_confidence=args.min_confidence,
        iv_rank=iv_rank_data,
        dp_context=dp_context_data,
    )

    # Print results
    print("\n" + "=" * 70, file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total trades: {analysis.get('total_trades', 0)}", file=sys.stderr)
    if analysis.get('total_trades', 0) > 0:
        print(f"Win rate: {analysis['win_rate']}%", file=sys.stderr)
        print(f"Total P&L: {analysis['total_pnl_points']:+.2f} points", file=sys.stderr)
        print(f"Avg P&L per trade: {analysis['avg_pnl_per_trade']:+.2f} points", file=sys.stderr)

        print(f"\nBy Direction:", file=sys.stderr)
        for dir_name, stats in analysis.get('by_direction', {}).items():
            print(f"  {dir_name.upper()}: {stats['count']} trades, {stats['wins']} wins, {stats['pnl']:+.2f} pts", file=sys.stderr)

        print(f"\nBy Flow Type:", file=sys.stderr)
        for flow, stats in analysis.get('by_flow_type', {}).items():
            print(f"  {flow.upper()}: {stats['count']} trades, {stats['wins']} wins, {stats['pnl']:+.2f} pts", file=sys.stderr)

        print(f"\nBy XGBoost Confidence:", file=sys.stderr)
        for conf, stats in analysis.get('by_confidence', {}).items():
            print(f"  {conf}: {stats['count']} trades, {stats['wins']} wins, {stats['pnl']:+.2f} pts", file=sys.stderr)

        print(f"\nBy Whale Agreement:", file=sys.stderr)
        for agree, stats in analysis.get('by_whale_agreement', {}).items():
            print(f"  {agree.upper()}: {stats['count']} trades, {stats['wins']} wins, {stats['pnl']:+.2f} pts", file=sys.stderr)

        print(f"\nBy Time of Day:", file=sys.stderr)
        for key in ["time_open", "time_morning", "time_lunch", "time_afternoon", "time_close"]:
            if key in analysis:
                stats = analysis[key]
                print(f"  {key.replace('time_', '').upper()}: {stats['count']} trades, {stats['wins']} wins, {stats['pnl']:+.2f} pts", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Save results
    result = {
        "analysis": analysis,
        "trades": trades,
        "config": {
            "trigger": "whale_flow_combined",
            "model": "xgboost",
            "features": "cvd, vp, multi_tf_candles, whale_sentiment, time_of_day, iv_rank" if iv_rank_data else "cvd, vp, multi_tf_candles, whale_sentiment, time_of_day",
            "date": TARGET_DATE,
            "options_premium_threshold": OPTIONS_PREMIUM_THRESHOLD,
            "lit_premium_threshold": LIT_PREMIUM_THRESHOLD,
            "iv_rank": iv_rank_data,
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Save] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
