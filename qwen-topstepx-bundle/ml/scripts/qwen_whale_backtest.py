#!/usr/bin/env python3
"""
Qwen Backtest with Multi-Timeframe + CVD/VP + Whale Flow Triggers

Features:
- Multi-TF candles: 1m, 5m, 15m, 1h, 4h, 1d (aggregated from 1s)
- CVD, Volume Profile, CVD Trend from 1-second bars
- Triggered by whale flow events (options at ask, large stock trades)
- SL based on swing highs/lows
- Minimum 1:2 RR

Usage:
    python qwen_whale_backtest.py
"""

import json
import sys
import os
import subprocess
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Ollama endpoint
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")


def load_1s_bars(filepath: str) -> List[Dict]:
    """Load 1-second bars from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get("bars", [])


def load_whale_options_flow(filepath: str) -> pd.DataFrame:
    """Load whale options flow from CSV."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    return df


def load_whale_lit_flow(filepath: str) -> pd.DataFrame:
    """Load whale lit/stock flow from CSV."""
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%Y %I:%M:%S %p')
    return df


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

        # Truncate to period
        if period_minutes >= 1440:  # Daily
            period_ts = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period_minutes >= 60:  # Hourly+
            hour = (dt.hour // (period_minutes // 60)) * (period_minutes // 60)
            period_ts = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
        else:
            minute = (dt.minute // period_minutes) * period_minutes
            period_ts = dt.replace(minute=minute, second=0, microsecond=0)

        period_key = period_ts.isoformat()
        if period_key not in period_data:
            period_order.append(period_key)
        period_data[period_key]['bars'].append(bar)

    # Build aggregated bars
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
    """
    Calculate CVD from 1s bars, aggregated to 1-min resolution.
    Returns: bars_1min, cvd_1min, cvd_ema_1min
    """
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

        # Calculate delta
        volume = bar.get('v', 0) or 0
        bar_range = bar['h'] - bar['l']
        if bar_range > 0:
            close_position = (bar['c'] - bar['l']) / bar_range
            delta = volume * (2 * close_position - 1)
        else:
            delta = volume if bar['c'] >= bar['o'] else -volume
        minute_data[minute_ts]['delta'] += delta

    # Build 1-min bars and CVD
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

    # CVD EMA (20-period)
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

    # Value Area (70%)
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
        # Swing high: higher than neighbors
        if recent[i]['h'] > recent[i-1]['h'] and recent[i]['h'] > recent[i+1]['h']:
            swing_highs.append(recent[i]['h'])
        # Swing low: lower than neighbors
        if recent[i]['l'] < recent[i-1]['l'] and recent[i]['l'] < recent[i+1]['l']:
            swing_lows.append(recent[i]['l'])

    return swing_highs, swing_lows


def get_whale_triggers(options_df: pd.DataFrame, lit_df: pd.DataFrame,
                       timestamp: datetime, lookback_minutes: int = 5) -> List[Dict]:
    """
    Get whale flow triggers within the lookback window.
    Only uses OPTIONS flow (no lit/stock flow).
    """
    cutoff = timestamp - timedelta(minutes=lookback_minutes)
    triggers = []

    # Options flow triggers ONLY
    mask = (options_df['datetime'] >= cutoff) & (options_df['datetime'] <= timestamp)
    recent_options = options_df[mask]

    for _, row in recent_options.iterrows():
        side = row.get('side', '')
        opt_type = row.get('type', '')
        sentiment = row.get('bearish_or_bullish', '')
        premium = row.get('premium', 0)

        # Call at ASK = bullish, Put at ASK = bearish
        # Call at BID = bearish, Put at BID = bullish
        if premium >= 100000:  # $100k+ premium
            if sentiment == 'bullish':
                triggers.append({
                    'type': 'options',
                    'direction': 'long',
                    'premium': premium,
                    'details': f"{opt_type} {side} ${premium:,.0f}",
                    'timestamp': row['datetime'],
                })
            elif sentiment == 'bearish':
                triggers.append({
                    'type': 'options',
                    'direction': 'short',
                    'premium': premium,
                    'details': f"{opt_type} {side} ${premium:,.0f}",
                    'timestamp': row['datetime'],
                })

    return triggers


def build_qwen_prompt(
    current_price: float,
    timestamp: str,
    candles_1m: List[Dict],
    candles_5m: List[Dict],
    candles_15m: List[Dict],
    candles_1h: List[Dict],
    candles_4h: List[Dict],
    candles_1d: List[Dict],
    cvd: float,
    cvd_ema: float,
    cvd_trend: str,
    vp: Dict[str, float],
    swing_highs: List[float],
    swing_lows: List[float],
    triggers: List[Dict],
    tick_size: float = 0.25,
) -> str:
    """Build prompt for Qwen similar to live agent (minus L2)."""

    # Format candles
    def format_candles(candles, n=5):
        if not candles:
            return "N/A"
        return " ".join([
            f"{c['c']:.2f}({'↑' if c['c'] > c['o'] else '↓'})"
            for c in candles[-n:]
        ])

    # HTF bias
    def get_bias(candles):
        if len(candles) < 2:
            return "neutral"
        last, prev = candles[-1], candles[-2]
        if last['c'] > prev['c'] and last['l'] > prev['l']:
            return "bullish"
        elif last['c'] < prev['c'] and last['h'] < prev['h']:
            return "bearish"
        return "neutral"

    bias_1h = get_bias(candles_1h)
    bias_4h = get_bias(candles_4h)
    bias_1d = get_bias(candles_1d)

    # CVD momentum
    cvd_delta = cvd - cvd_ema
    cvd_strength = "strong" if abs(cvd_delta) > abs(cvd_ema) * 0.1 else "moderate" if abs(cvd_delta) > abs(cvd_ema) * 0.05 else "weak"

    # Trigger summary and whale bias
    trigger_summary = "\n".join([
        f"  - {t['type'].upper()} {t['direction'].upper()}: {t['details']}"
        for t in triggers[-5:]
    ]) if triggers else "  None"

    # Calculate whale flow bias
    long_triggers = sum(1 for t in triggers if t['direction'] == 'long')
    short_triggers = sum(1 for t in triggers if t['direction'] == 'short')
    if long_triggers > short_triggers:
        whale_bias = "BULLISH"
        whale_direction = "LONG"
    elif short_triggers > long_triggers:
        whale_bias = "BEARISH"
        whale_direction = "SHORT"
    else:
        whale_bias = "NEUTRAL"
        whale_direction = "EITHER"

    # Swing levels for SL
    recent_swing_high = max(swing_highs[-3:]) if swing_highs else current_price + 2
    recent_swing_low = min(swing_lows[-3:]) if swing_lows else current_price - 2

    prompt = f"""MES @ {current_price:.2f} | {timestamp}

=== 1. CVD TREND (Primary Signal - from 1-second bars) ===
CVD: {cvd:.0f} | EMA: {cvd_ema:.0f} | Delta: {cvd_delta:+.0f}
TREND: {cvd_trend.upper()} ({cvd_strength})
{"→ FAVOR LONG (buyers in control)" if cvd_trend == "up" else "→ FAVOR SHORT (sellers in control)"}

=== 2. HTF BIAS (Multi-Timeframe) ===
1H: {bias_1h.upper()} | 4H: {bias_4h.upper()} | Daily: {bias_1d.upper()}
{"→ HTF supports LONG" if bias_4h == "bullish" else "→ HTF supports SHORT" if bias_4h == "bearish" else "→ HTF mixed"}

=== 3. VOLUME PROFILE ===
POC: {vp['poc']:.2f} | VAH: {vp['vah']:.2f} | VAL: {vp['val']:.2f}
Price vs POC: {"Above" if current_price > vp['poc'] else "Below"} ({abs(current_price - vp['poc']):.2f} pts)

=== 4. WHALE FLOW TRIGGERS (Last 5 min) ===
{trigger_summary}
WHALE BIAS: {whale_bias} → YOU MUST GO {whale_direction}

=== 5. CANDLES ===
1m: {format_candles(candles_1m, 5)}
5m: {format_candles(candles_5m, 4)}
15m: {format_candles(candles_15m, 3)}
1h: {format_candles(candles_1h, 3)}
4h: {format_candles(candles_4h, 2)}
1D: {format_candles(candles_1d, 2)}

=== 6. SWING LEVELS (for Stop Loss) ===
Recent Swing High: {recent_swing_high:.2f}
Recent Swing Low: {recent_swing_low:.2f}

RULES:
- Consider ALL factors: CVD trend, whale options flow, volume profile, HTF bias
- CVD trend shows order flow momentum
- Whale options flow shows institutional sentiment
- Volume profile shows key support/resistance levels
- Use swing high/low for stop loss
- MINIMUM 1:2 Risk/Reward ratio
- Only take HIGH CONVICTION trades (confidence >= 75)
- If signals conflict or edge is unclear, respond HOLD

Respond with JSON only:
{{"decision":"BUY|SELL|HOLD","confidence":0-100,"entryPrice":{current_price:.2f},"stopLoss":number,"target":number,"riskRewardRatio":number,"reasoning":"brief"}}"""

    return prompt


def call_qwen(prompt: str) -> Optional[Dict]:
    """Call Qwen via Ollama and parse response."""
    try:
        import requests

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 256,
                }
            },
            timeout=30,
        )

        if response.status_code != 200:
            return None

        result = response.json()
        content = result.get("response", "")

        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        return None

    except Exception as e:
        print(f"[Qwen] Error: {e}", file=sys.stderr)
        return None


def maintain_rr_stop(
    current_price: float,
    current_stop: float,
    target: float,
    direction: str,
    min_rr: float = 2.0,
    tick_size: float = 0.25,
) -> Optional[float]:
    """
    Maintain minimum Risk:Reward ratio from current price.

    If R:R drops below min_rr (default 1:2), move stop to restore it.
    Smooth trailing - moves naturally with price, never loosens.

    Example:
    - Current price 100, target 102, stop 96
    - Reward = 2 pts, Risk = 4 pts, R:R = 0.5 (1:2) ✓

    - Current price 101.5, target 102, stop 96
    - Reward = 0.5 pts, Risk = 5.5 pts, R:R = 0.09 (terrible!)
    - Need stop where Risk = Reward / 2 = 0.25 pts
    - New stop = 101.5 - 0.25 = 101.25
    """
    is_long = direction == "long"

    # Calculate current distances
    if is_long:
        reward = target - current_price  # Distance to target
        risk = current_price - current_stop  # Distance to stop
    else:
        reward = current_price - target
        risk = current_stop - current_price

    # If already at or past target, no adjustment needed
    if reward <= 0:
        return None

    # If no risk (stop at or beyond current price), no adjustment
    if risk <= 0:
        return None

    # Current R:R ratio (reward / risk)
    current_rr = reward / risk

    # If R:R is already good (>= 1:min_rr), no change needed
    if current_rr >= (1.0 / min_rr):
        return None

    # R:R is bad - calculate new stop to achieve 1:min_rr
    # We want: reward / new_risk = 1 / min_rr
    # So: new_risk = reward * min_rr
    required_risk = reward * min_rr

    # Calculate new stop price
    if is_long:
        new_stop = current_price - required_risk
    else:
        new_stop = current_price + required_risk

    # Round to tick size
    new_stop = round(new_stop / tick_size) * tick_size

    # Only return if it's tighter than current stop
    if is_long and new_stop > current_stop:
        return new_stop
    elif not is_long and new_stop < current_stop:
        return new_stop

    return None


def simulate_trade_fixed(
    entry_price: float,
    stop_loss: float,
    target: float,
    direction: str,
    future_bars: List[Dict],
) -> Dict[str, Any]:
    """Simulate trade with FIXED stop loss - no trailing."""
    is_long = direction == "long"
    max_profit = 0.0

    for i, bar in enumerate(future_bars):
        high = bar['h']
        low = bar['l']

        if is_long:
            max_profit = max(max_profit, high - entry_price)
        else:
            max_profit = max(max_profit, entry_price - low)

        # Check stop loss hit (fixed)
        if is_long and low <= stop_loss:
            pnl = stop_loss - entry_price
            return {
                "outcome": "win" if pnl > 0 else "loss",
                "exit_price": stop_loss,
                "pnl": pnl,
                "bars_held": i + 1,
                "max_profit": max_profit,
                "final_stop": stop_loss,
                "stop_adjustments": 0,
            }
        elif not is_long and high >= stop_loss:
            pnl = entry_price - stop_loss
            return {
                "outcome": "win" if pnl > 0 else "loss",
                "exit_price": stop_loss,
                "pnl": pnl,
                "bars_held": i + 1,
                "max_profit": max_profit,
                "final_stop": stop_loss,
                "stop_adjustments": 0,
            }

        # Check target hit
        if is_long and high >= target:
            return {
                "outcome": "win",
                "exit_price": target,
                "pnl": target - entry_price,
                "bars_held": i + 1,
                "max_profit": max_profit,
                "final_stop": stop_loss,
                "stop_adjustments": 0,
            }
        elif not is_long and low <= target:
            return {
                "outcome": "win",
                "exit_price": target,
                "pnl": entry_price - target,
                "bars_held": i + 1,
                "max_profit": max_profit,
                "final_stop": stop_loss,
                "stop_adjustments": 0,
            }

    # Timeout
    last_close = future_bars[-1]['c'] if future_bars else entry_price
    pnl = (last_close - entry_price) if is_long else (entry_price - last_close)
    return {
        "outcome": "timeout",
        "exit_price": last_close,
        "pnl": pnl,
        "bars_held": len(future_bars),
        "max_profit": max_profit,
        "final_stop": stop_loss,
        "stop_adjustments": 0,
    }


def simulate_trade(
    entry_price: float,
    stop_loss: float,
    target: float,
    direction: str,
    future_bars: List[Dict],
    min_rr: float = 2.0,
    tick_size: float = 0.25,
) -> Dict[str, Any]:
    """
    Simulate trade with automatic R:R maintenance.

    Simple rule: Always maintain at least 1:min_rr risk/reward ratio.
    If R:R drops below threshold, trail stop to restore it.
    Smooth, continuous - no jumps, no breakeven triggers.
    """
    is_long = direction == "long"
    current_stop = stop_loss
    max_profit = 0.0
    stop_adjustments = 0

    for i, bar in enumerate(future_bars):
        high = bar['h']
        low = bar['l']
        close = bar['c']

        # Track max profit for stats
        if is_long:
            max_profit = max(max_profit, high - entry_price)
        else:
            max_profit = max(max_profit, entry_price - low)

        # === R:R MAINTENANCE ===
        # Check at close price - maintain minimum R:R
        new_stop = maintain_rr_stop(
            current_price=close,
            current_stop=current_stop,
            target=target,
            direction=direction,
            min_rr=min_rr,
            tick_size=tick_size,
        )

        if new_stop:
            current_stop = new_stop
            stop_adjustments += 1

        # === CHECK EXIT CONDITIONS ===

        # Check stop loss hit
        if is_long and low <= current_stop:
            pnl = current_stop - entry_price
            outcome = "win" if pnl > 0 else "loss"
            return {
                "outcome": outcome,
                "exit_price": current_stop,
                "pnl": pnl,
                "bars_held": i + 1,
                "max_profit": max_profit,
                "final_stop": current_stop,
                "stop_adjustments": stop_adjustments,
            }
        elif not is_long and high >= current_stop:
            pnl = entry_price - current_stop
            outcome = "win" if pnl > 0 else "loss"
            return {
                "outcome": outcome,
                "exit_price": current_stop,
                "pnl": pnl,
                "bars_held": i + 1,
                "max_profit": max_profit,
                "final_stop": current_stop,
                "stop_adjustments": stop_adjustments,
            }

        # Check target hit
        if is_long and high >= target:
            return {
                "outcome": "win",
                "exit_price": target,
                "pnl": target - entry_price,
                "bars_held": i + 1,
                "max_profit": max_profit,
                "final_stop": current_stop,
                "stop_adjustments": stop_adjustments,
            }
        elif not is_long and low <= target:
            return {
                "outcome": "win",
                "exit_price": target,
                "pnl": entry_price - target,
                "bars_held": i + 1,
                "max_profit": max_profit,
                "final_stop": current_stop,
                "stop_adjustments": stop_adjustments,
            }

    # Timeout - exit at last close
    last_close = future_bars[-1]['c'] if future_bars else entry_price
    if is_long:
        pnl = last_close - entry_price
    else:
        pnl = entry_price - last_close

    return {
        "outcome": "timeout",
        "exit_price": last_close,
        "pnl": pnl,
        "bars_held": len(future_bars),
        "max_profit": max_profit,
        "final_stop": current_stop,
        "stop_adjustments": stop_adjustments,
    }


def run_backtest(
    bars_1s: List[Dict],
    options_df: pd.DataFrame,
    lit_df: pd.DataFrame,
    use_qwen: bool = True,
    max_trades: int = 50,
) -> List[Dict]:
    """Run the backtest."""

    print("[Backtest] Aggregating timeframes...", file=sys.stderr)

    # Aggregate all timeframes
    bars_1m, cvd_1min, cvd_ema_1min = calculate_cvd_1min(bars_1s)
    bars_5m = aggregate_bars(bars_1s, 5)
    bars_15m = aggregate_bars(bars_1s, 15)
    bars_1h = aggregate_bars(bars_1s, 60)
    bars_4h = aggregate_bars(bars_1s, 240)
    bars_1d = aggregate_bars(bars_1s, 1440)

    print(f"[Backtest] 1m: {len(bars_1m)}, 5m: {len(bars_5m)}, 15m: {len(bars_15m)}, 1h: {len(bars_1h)}, 4h: {len(bars_4h)}, 1d: {len(bars_1d)}", file=sys.stderr)

    trades = []
    lookback = 60  # Need 60 1-min bars for context
    max_hold_bars = 60  # Max 60 min hold time

    print(f"[Backtest] Scanning for whale triggers...", file=sys.stderr)

    # Find all whale trigger timestamps
    all_triggers = []
    for i in range(lookback, len(bars_1m) - max_hold_bars):
        bar = bars_1m[i]
        ts_str = bar['t']
        if isinstance(ts_str, str):
            ts = datetime.fromisoformat(ts_str.replace('+00:00', ''))
        else:
            ts = ts_str

        triggers = get_whale_triggers(options_df, lit_df, ts, lookback_minutes=5)
        if triggers:
            all_triggers.append((i, ts, triggers))

    print(f"[Backtest] Found {len(all_triggers)} trigger points", file=sys.stderr)

    if not all_triggers:
        print("[Backtest] No whale triggers found in data", file=sys.stderr)
        return []

    # Sample triggers if too many
    if len(all_triggers) > max_trades:
        import random
        all_triggers = random.sample(all_triggers, max_trades)
        all_triggers.sort(key=lambda x: x[0])

    print(f"[Backtest] Processing {len(all_triggers)} trades...", file=sys.stderr)

    for idx, (bar_idx, ts, triggers) in enumerate(all_triggers):
        current_bar = bars_1m[bar_idx]
        current_price = current_bar['c']

        # Get historical context (no future peeking)
        candles_1m = bars_1m[:bar_idx + 1]

        # Find corresponding indices for other timeframes
        candles_5m = [b for b in bars_5m if b['t'] <= current_bar['t']]
        candles_15m = [b for b in bars_15m if b['t'] <= current_bar['t']]
        candles_1h = [b for b in bars_1h if b['t'] <= current_bar['t']]
        candles_4h = [b for b in bars_4h if b['t'] <= current_bar['t']]
        candles_1d = [b for b in bars_1d if b['t'] <= current_bar['t']]

        # CVD at this point
        cvd = cvd_1min[bar_idx] if bar_idx < len(cvd_1min) else 0
        cvd_ema = cvd_ema_1min[bar_idx] if bar_idx < len(cvd_ema_1min) else 0
        cvd_trend = "up" if cvd > cvd_ema else "down"

        # Volume Profile
        vp = calculate_volume_profile(candles_1m, lookback=30)

        # Swing levels
        swing_highs, swing_lows = find_swing_highs_lows(candles_1m, lookback=20)

        if use_qwen:
            # Build prompt and call Qwen
            prompt = build_qwen_prompt(
                current_price=current_price,
                timestamp=current_bar['t'][-8:] if isinstance(current_bar['t'], str) else str(current_bar['t']),
                candles_1m=candles_1m,
                candles_5m=candles_5m,
                candles_15m=candles_15m,
                candles_1h=candles_1h,
                candles_4h=candles_4h,
                candles_1d=candles_1d,
                cvd=cvd,
                cvd_ema=cvd_ema,
                cvd_trend=cvd_trend,
                vp=vp,
                swing_highs=swing_highs,
                swing_lows=swing_lows,
                triggers=triggers,
            )

            decision = call_qwen(prompt)

            if not decision or decision.get('decision') == 'HOLD':
                continue

            confidence = decision.get('confidence', 0)

            # Filter by confidence >= 75
            if confidence < 75:
                continue

            direction = "long" if decision.get('decision') == 'BUY' else "short"
            stop_loss = decision.get('stopLoss', 0)
            target = decision.get('target', 0)
            reasoning = decision.get('reasoning', '')

        else:
            # Simple rule-based decision (no Qwen)
            # Follow whale direction + CVD agreement
            whale_direction = triggers[0]['direction'] if triggers else None

            if not whale_direction:
                continue

            # Check CVD agreement
            if whale_direction == "long" and cvd_trend != "up":
                continue
            if whale_direction == "short" and cvd_trend != "down":
                continue

            direction = whale_direction

            # Set SL/TP based on swings with 1:2 RR
            if direction == "long":
                stop_loss = min(swing_lows[-3:]) if swing_lows else current_price - 2
                risk = current_price - stop_loss
                target = current_price + (risk * 2)  # 1:2 RR
            else:
                stop_loss = max(swing_highs[-3:]) if swing_highs else current_price + 2
                risk = stop_loss - current_price
                target = current_price - (risk * 2)  # 1:2 RR

            confidence = 70
            reasoning = f"Whale {direction} + CVD {cvd_trend}"

        # Validate trade parameters
        if direction == "long":
            if stop_loss >= current_price or target <= current_price:
                continue
            risk = current_price - stop_loss
            reward = target - current_price
        else:
            if stop_loss <= current_price or target >= current_price:
                continue
            risk = stop_loss - current_price
            reward = current_price - target

        rr = reward / risk if risk > 0 else 0
        if rr < 1.5:  # Require at least 1.5:1 RR
            continue

        # Simulate trade with future bars
        future_bars = bars_1m[bar_idx + 1:bar_idx + 1 + max_hold_bars]
        result = simulate_trade(current_price, stop_loss, target, direction, future_bars)
        result_fixed = simulate_trade_fixed(current_price, stop_loss, target, direction, future_bars)

        # Calculate whale bias from triggers
        long_triggers = sum(1 for t in triggers if t['direction'] == 'long')
        short_triggers = sum(1 for t in triggers if t['direction'] == 'short')
        if long_triggers > short_triggers:
            whale_bias = "bullish"
        elif short_triggers > long_triggers:
            whale_bias = "bearish"
        else:
            whale_bias = "neutral"

        trade = {
            "timestamp": current_bar['t'],
            "bar_index": bar_idx,
            "direction": direction,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "target": target,
            "risk_reward": round(rr, 2),
            "confidence": confidence,
            "reasoning": reasoning,
            "cvd_trend": cvd_trend,
            "whale_bias": whale_bias,
            "triggers": [t['details'] for t in triggers[:3]],
            "outcome": result['outcome'],
            "exit_price": result['exit_price'],
            "pnl_points": round(result['pnl'], 2),
            "bars_held": result['bars_held'],
            "max_profit": round(result.get('max_profit', 0), 2),
            "final_stop": round(result.get('final_stop', stop_loss), 2),
            "stop_adjustments": result.get('stop_adjustments', 0),
            # Fixed stop comparison
            "fixed_outcome": result_fixed['outcome'],
            "fixed_pnl": round(result_fixed['pnl'], 2),
        }
        trades.append(trade)

        if (idx + 1) % 10 == 0:
            print(f"[Backtest] Processed {idx + 1}/{len(all_triggers)} trades...", file=sys.stderr)

    print(f"[Backtest] Completed {len(trades)} trades", file=sys.stderr)
    return trades


def analyze_results(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze backtest results."""
    if not trades:
        return {"error": "No trades"}

    total = len(trades)
    wins = [t for t in trades if t['outcome'] == 'win']
    losses = [t for t in trades if t['outcome'] == 'loss']
    timeouts = [t for t in trades if t['outcome'] == 'timeout']

    total_pnl = sum(t['pnl_points'] for t in trades)

    # By direction
    longs = [t for t in trades if t['direction'] == 'long']
    shorts = [t for t in trades if t['direction'] == 'short']

    long_wins = len([t for t in longs if t['outcome'] == 'win'])
    short_wins = len([t for t in shorts if t['outcome'] == 'win'])

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "timeouts": len(timeouts),
        "win_rate": round(len(wins) / total * 100, 2) if total > 0 else 0,
        "total_pnl_points": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / total, 2) if total > 0 else 0,
        "avg_winner": round(sum(t['pnl_points'] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loser": round(sum(t['pnl_points'] for t in losses) / len(losses), 2) if losses else 0,
        "by_direction": {
            "long": {
                "count": len(longs),
                "wins": long_wins,
                "win_rate": round(long_wins / len(longs) * 100, 2) if longs else 0,
                "pnl": round(sum(t['pnl_points'] for t in longs), 2),
            },
            "short": {
                "count": len(shorts),
                "wins": short_wins,
                "win_rate": round(short_wins / len(shorts) * 100, 2) if shorts else 0,
                "pnl": round(sum(t['pnl_points'] for t in shorts), 2),
            },
        },
    }


def main():
    # Check for --nq, --mnq, --es, --m2k, or --mym flags
    use_nq = "--nq" in sys.argv
    use_mnq = "--mnq" in sys.argv
    use_es = "--es" in sys.argv
    use_m2k = "--m2k" in sys.argv
    use_mym = "--mym" in sys.argv

    if use_mym:
        bars_1s_path = "data/bars_1s.json"
        options_flow_path = "data/dia_options_flow_20251205.csv"
        lit_flow_path = None
        output_path = "data/qwen_mym_results.json"
        symbol = "MYM"
    elif use_mnq:
        bars_1s_path = "data/bars_1s.json"
        options_flow_path = "data/qqq_options_flow_20251205.csv"
        lit_flow_path = None
        output_path = "data/qwen_mnq_results.json"
        symbol = "MNQ"
    elif use_nq:
        bars_1s_path = "data/bars_1s.json"
        options_flow_path = "data/qqq_options_flow_20251205.csv"
        lit_flow_path = None
        output_path = "data/qwen_nq_results.json"
        symbol = "NQ"
    elif use_es:
        bars_1s_path = "data/bars_1s.json"
        options_flow_path = "data/spy_options_flow_20251205.csv"
        lit_flow_path = "data/spy_lit_flow_20251205.csv"
        output_path = "data/qwen_es_results.json"
        symbol = "ES"
    elif use_m2k:
        bars_1s_path = "data/bars_1s.json"
        options_flow_path = "data/iwm_options_flow_20251205.csv"
        lit_flow_path = None
        output_path = "data/qwen_m2k_results.json"
        symbol = "M2K"
    else:
        bars_1s_path = "data/bars_1s.json"
        options_flow_path = "data/spy_options_flow_20251205.csv"
        lit_flow_path = "data/spy_lit_flow_20251205.csv"
        output_path = "data/qwen_whale_results.json"
        symbol = "MES"

    # Check for --no-qwen flag
    use_qwen = "--no-qwen" not in sys.argv

    print("=" * 70, file=sys.stderr)
    print(f"QWEN BACKTEST + WHALE FLOW TRIGGERS ({symbol})", file=sys.stderr)
    print(f"Mode: {'Qwen LLM' if use_qwen else 'Rule-based (no Qwen)'}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Load data
    print(f"[Load] Loading 1s bars...", file=sys.stderr)
    bars_1s_all = load_1s_bars(bars_1s_path)
    bars_1s = [b for b in bars_1s_all if '2025-12-05' in b['t']]
    print(f"[Load] Filtered to {len(bars_1s)} bars on 12/5/2025", file=sys.stderr)

    print(f"[Load] Loading whale flow data...", file=sys.stderr)
    options_df = load_whale_options_flow(options_flow_path)
    if lit_flow_path:
        lit_df = load_whale_lit_flow(lit_flow_path)
    else:
        lit_df = pd.DataFrame()  # Empty for NQ
    print(f"[Load] Options: {len(options_df)}, Lit: {len(lit_df)}", file=sys.stderr)

    # Run backtest
    trades = run_backtest(
        bars_1s=bars_1s,
        options_df=options_df,
        lit_df=lit_df,
        use_qwen=use_qwen,
        max_trades=100,
    )

    # Analyze
    analysis = analyze_results(trades)

    # Print results
    print("\n" + "=" * 70, file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total trades: {analysis.get('total_trades', 0)}", file=sys.stderr)
    print(f"Wins: {analysis.get('wins', 0)} | Losses: {analysis.get('losses', 0)} | Timeouts: {analysis.get('timeouts', 0)}", file=sys.stderr)
    print(f"Win Rate: {analysis.get('win_rate', 0)}%", file=sys.stderr)
    print(f"Total PnL: {analysis.get('total_pnl_points', 0)} points", file=sys.stderr)
    print(f"Avg PnL/Trade: {analysis.get('avg_pnl_per_trade', 0)} points", file=sys.stderr)
    print(f"Avg Winner: {analysis.get('avg_winner', 0)} | Avg Loser: {analysis.get('avg_loser', 0)}", file=sys.stderr)

    if 'by_direction' in analysis:
        print(f"\nLONG: {analysis['by_direction']['long']['count']} trades, {analysis['by_direction']['long']['win_rate']}% win, {analysis['by_direction']['long']['pnl']} pts", file=sys.stderr)
        print(f"SHORT: {analysis['by_direction']['short']['count']} trades, {analysis['by_direction']['short']['win_rate']}% win, {analysis['by_direction']['short']['pnl']} pts", file=sys.stderr)

    # Compare with fixed stops
    if trades and 'fixed_pnl' in trades[0]:
        fixed_wins = len([t for t in trades if t['fixed_outcome'] == 'win'])
        fixed_pnl = sum(t['fixed_pnl'] for t in trades)
        fixed_losses = [t for t in trades if t['fixed_outcome'] == 'loss']
        fixed_winners = [t for t in trades if t['fixed_outcome'] == 'win']

        print(f"\n--- COMPARISON: R:R Trailing vs Fixed Stops ---", file=sys.stderr)
        print(f"R:R TRAILING: {analysis.get('wins', 0)} wins, {analysis.get('win_rate', 0)}% WR, {analysis.get('total_pnl_points', 0):.2f} pts", file=sys.stderr)
        print(f"FIXED STOPS:  {fixed_wins} wins, {round(fixed_wins/len(trades)*100, 2)}% WR, {fixed_pnl:.2f} pts", file=sys.stderr)
        print(f"Avg Winner - Trailing: {analysis.get('avg_winner', 0):.2f} | Fixed: {round(sum(t['fixed_pnl'] for t in fixed_winners)/len(fixed_winners), 2) if fixed_winners else 0}", file=sys.stderr)
        print(f"Avg Loser  - Trailing: {analysis.get('avg_loser', 0):.2f} | Fixed: {round(sum(t['fixed_pnl'] for t in fixed_losses)/len(fixed_losses), 2) if fixed_losses else 0}", file=sys.stderr)
        print(f"Difference: {analysis.get('total_pnl_points', 0) - fixed_pnl:+.2f} pts", file=sys.stderr)

    print("=" * 70, file=sys.stderr)

    # Save
    result = {
        "analysis": analysis,
        "trades": trades,
        "config": {
            "use_qwen": use_qwen,
            "date": "2025-12-05",
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Save] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
