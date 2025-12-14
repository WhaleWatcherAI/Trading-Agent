#!/usr/bin/env python3
"""
Qwen Backtest with CVD Trend Flip Triggers (No Whale Flow)

Triggers Qwen call when CVD trend flips (crosses EMA).
This is a test version to compare against whale-triggered version.

Usage:
    python qwen_cvd_backtest.py [--nq] [--mnq] [--es] [--m2k] [--mym]
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


def build_qwen_prompt_cvd(
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
    cvd_flip_direction: str,  # "bullish" or "bearish"
    vp: Dict[str, float],
    swing_highs: List[float],
    swing_lows: List[float],
    symbol: str = "MES",
    tick_size: float = 0.25,
) -> str:
    """Build prompt for Qwen triggered by CVD flip (no whale flow)."""

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

    # Swing levels for SL
    recent_swing_high = max(swing_highs[-3:]) if swing_highs else current_price + 2
    recent_swing_low = min(swing_lows[-3:]) if swing_lows else current_price - 2

    prompt = f"""{symbol} @ {current_price:.2f} | {timestamp}

=== CVD TREND FLIP DETECTED ===
CVD just crossed {"ABOVE" if cvd_flip_direction == "bullish" else "BELOW"} its EMA!
This is a {cvd_flip_direction.upper()} signal.

=== 1. CVD TREND (Primary Signal - from 1-second bars) ===
CVD: {cvd:.0f} | EMA: {cvd_ema:.0f} | Delta: {cvd_delta:+.0f}
TREND: {cvd_trend.upper()} ({cvd_strength})
{"→ FAVOR LONG (buyers taking control)" if cvd_trend == "up" else "→ FAVOR SHORT (sellers taking control)"}

=== 2. HTF BIAS (Multi-Timeframe) ===
1H: {bias_1h.upper()} | 4H: {bias_4h.upper()} | Daily: {bias_1d.upper()}
{"→ HTF supports LONG" if bias_4h == "bullish" else "→ HTF supports SHORT" if bias_4h == "bearish" else "→ HTF mixed"}

=== 3. VOLUME PROFILE ===
POC: {vp['poc']:.2f} | VAH: {vp['vah']:.2f} | VAL: {vp['val']:.2f}
Price vs POC: {"Above" if current_price > vp['poc'] else "Below"} ({abs(current_price - vp['poc']):.2f} pts)

=== 4. CANDLES ===
1m: {format_candles(candles_1m, 5)}
5m: {format_candles(candles_5m, 4)}
15m: {format_candles(candles_15m, 3)}
1h: {format_candles(candles_1h, 3)}
4h: {format_candles(candles_4h, 2)}
1D: {format_candles(candles_1d, 2)}

=== 5. SWING LEVELS (for Stop Loss) ===
Recent Swing High: {recent_swing_high:.2f}
Recent Swing Low: {recent_swing_low:.2f}

RULES:
- CVD just flipped {cvd_flip_direction.upper()} - this is the PRIMARY signal
- Consider HTF bias and volume profile for confirmation
- Use swing high/low for stop loss
- MINIMUM 1:2 Risk/Reward ratio
- Only take HIGH CONVICTION trades (confidence >= 75)
- If HTF strongly conflicts with CVD flip, respond HOLD

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
    """Maintain minimum Risk:Reward ratio from current price."""
    is_long = direction == "long"

    # Calculate current distances
    if is_long:
        reward = target - current_price
        risk = current_price - current_stop
    else:
        reward = current_price - target
        risk = current_stop - current_price

    if reward <= 0 or risk <= 0:
        return None

    current_rr = reward / risk

    if current_rr >= (1.0 / min_rr):
        return None

    required_risk = reward * min_rr

    if is_long:
        new_stop = current_price - required_risk
    else:
        new_stop = current_price + required_risk

    new_stop = round(new_stop / tick_size) * tick_size

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
    tick_size: float = 0.25,
) -> Dict[str, Any]:
    """Simulate trade execution with R:R maintenance trailing stop."""
    is_long = direction == "long"
    current_stop = stop_loss
    stop_adjustments = 0
    max_profit = 0.0

    for i, bar in enumerate(future_bars):
        high = bar['h']
        low = bar['l']
        close = bar['c']

        # Track max profit
        if is_long:
            max_profit = max(max_profit, high - entry_price)
        else:
            max_profit = max(max_profit, entry_price - low)

        # Check stop loss hit first
        if is_long and low <= current_stop:
            pnl = current_stop - entry_price
            return {
                "outcome": "win" if pnl > 0 else "loss",
                "exit_price": current_stop,
                "pnl": pnl,
                "bars_held": i + 1,
                "max_profit": max_profit,
                "final_stop": current_stop,
                "stop_adjustments": stop_adjustments,
            }
        elif not is_long and high >= current_stop:
            pnl = entry_price - current_stop
            return {
                "outcome": "win" if pnl > 0 else "loss",
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

        # Adjust stop to maintain R:R
        new_stop = maintain_rr_stop(close, current_stop, target, direction, min_rr=2.0, tick_size=tick_size)
        if new_stop:
            current_stop = new_stop
            stop_adjustments += 1

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


def find_cvd_flips(cvd_1min: List[float], cvd_ema_1min: List[float], min_gap_bars: int = 5) -> List[Tuple[int, str]]:
    """
    Find CVD trend flip points (crosses above/below EMA).
    Returns list of (bar_index, direction) where direction is "bullish" or "bearish".
    min_gap_bars: minimum bars between flips to avoid whipsaw
    """
    flips = []
    last_flip_idx = -min_gap_bars

    for i in range(1, len(cvd_1min)):
        if i < len(cvd_ema_1min):
            prev_cvd = cvd_1min[i-1]
            curr_cvd = cvd_1min[i]
            prev_ema = cvd_ema_1min[i-1]
            curr_ema = cvd_ema_1min[i]

            # Bullish flip: CVD crosses above EMA
            if prev_cvd <= prev_ema and curr_cvd > curr_ema:
                if i - last_flip_idx >= min_gap_bars:
                    flips.append((i, "bullish"))
                    last_flip_idx = i

            # Bearish flip: CVD crosses below EMA
            elif prev_cvd >= prev_ema and curr_cvd < curr_ema:
                if i - last_flip_idx >= min_gap_bars:
                    flips.append((i, "bearish"))
                    last_flip_idx = i

    return flips


def run_backtest(
    bars_1s: List[Dict],
    use_qwen: bool = True,
    max_trades: int = 50,
    symbol: str = "MES",
    tick_size: float = 0.25,
) -> List[Dict]:
    """Run the CVD-flip triggered backtest."""

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

    print(f"[Backtest] Finding CVD trend flips...", file=sys.stderr)

    # Find all CVD flip points (no gap filter)
    cvd_flips = find_cvd_flips(cvd_1min, cvd_ema_1min, min_gap_bars=0)

    # Filter to valid range
    valid_flips = [(idx, direction) for idx, direction in cvd_flips
                   if idx >= lookback and idx < len(bars_1m) - max_hold_bars]

    print(f"[Backtest] Found {len(valid_flips)} CVD flip points", file=sys.stderr)

    if not valid_flips:
        print("[Backtest] No CVD flips found in data", file=sys.stderr)
        return []

    # Sample if too many
    if len(valid_flips) > max_trades:
        import random
        valid_flips = random.sample(valid_flips, max_trades)
        valid_flips.sort(key=lambda x: x[0])

    print(f"[Backtest] Processing {len(valid_flips)} trades...", file=sys.stderr)

    for idx, (bar_idx, flip_direction) in enumerate(valid_flips):
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
        vp = calculate_volume_profile(candles_1m, lookback=30, tick_size=tick_size)

        # Swing levels
        swing_highs, swing_lows = find_swing_highs_lows(candles_1m, lookback=20)

        if use_qwen:
            # Build prompt and call Qwen
            prompt = build_qwen_prompt_cvd(
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
                cvd_flip_direction=flip_direction,
                vp=vp,
                swing_highs=swing_highs,
                swing_lows=swing_lows,
                symbol=symbol,
                tick_size=tick_size,
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
            # Simple rule-based: follow CVD flip direction
            direction = "long" if flip_direction == "bullish" else "short"

            # Set SL/TP based on swings with 1:2 RR
            if direction == "long":
                stop_loss = min(swing_lows[-3:]) if swing_lows else current_price - 2
                risk = current_price - stop_loss
                target = current_price + (risk * 2)
            else:
                stop_loss = max(swing_highs[-3:]) if swing_highs else current_price + 2
                risk = stop_loss - current_price
                target = current_price - (risk * 2)

            confidence = 70
            reasoning = f"CVD flip {flip_direction}"

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
        if rr < 1.5:
            continue

        # Simulate trade with future bars
        future_bars = bars_1m[bar_idx + 1:bar_idx + 1 + max_hold_bars]
        result = simulate_trade(current_price, stop_loss, target, direction, future_bars, tick_size)
        result_fixed = simulate_trade_fixed(current_price, stop_loss, target, direction, future_bars)

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
            "cvd_flip": flip_direction,
            "outcome": result["outcome"],
            "exit_price": result["exit_price"],
            "pnl_points": round(result["pnl"], 2),
            "bars_held": result["bars_held"],
            "max_profit": result["max_profit"],
            "final_stop": result["final_stop"],
            "stop_adjustments": result["stop_adjustments"],
            "fixed_outcome": result_fixed["outcome"],
            "fixed_pnl": round(result_fixed["pnl"], 2),
        }

        trades.append(trade)

        # Progress
        outcome_str = f"{result['outcome'].upper()} {result['pnl']:+.2f} pts"
        print(f"[{idx+1}/{len(valid_flips)}] {current_bar['t']} | CVD flip {flip_direction} | {direction.upper()} | {outcome_str}", file=sys.stderr)

    return trades


def analyze_trades(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze trade results."""
    if not trades:
        return {"total_trades": 0}

    wins = [t for t in trades if t['outcome'] == 'win']
    losses = [t for t in trades if t['outcome'] == 'loss']
    timeouts = [t for t in trades if t['outcome'] == 'timeout']

    total_pnl = sum(t['pnl_points'] for t in trades)
    win_pnl = sum(t['pnl_points'] for t in wins) if wins else 0
    loss_pnl = sum(t['pnl_points'] for t in losses) if losses else 0

    # By direction
    longs = [t for t in trades if t['direction'] == 'long']
    shorts = [t for t in trades if t['direction'] == 'short']

    # By CVD flip direction
    bullish_flips = [t for t in trades if t['cvd_flip'] == 'bullish']
    bearish_flips = [t for t in trades if t['cvd_flip'] == 'bearish']

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "timeouts": len(timeouts),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0,
        "total_pnl_points": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / len(trades), 2) if trades else 0,
        "avg_winner": round(win_pnl / len(wins), 2) if wins else 0,
        "avg_loser": round(loss_pnl / len(losses), 2) if losses else 0,
        "by_direction": {
            "long": {
                "count": len(longs),
                "wins": len([t for t in longs if t['outcome'] == 'win']),
                "win_rate": round(len([t for t in longs if t['outcome'] == 'win']) / len(longs) * 100, 2) if longs else 0,
                "pnl": round(sum(t['pnl_points'] for t in longs), 2),
            },
            "short": {
                "count": len(shorts),
                "wins": len([t for t in shorts if t['outcome'] == 'win']),
                "win_rate": round(len([t for t in shorts if t['outcome'] == 'win']) / len(shorts) * 100, 2) if shorts else 0,
                "pnl": round(sum(t['pnl_points'] for t in shorts), 2),
            },
        },
        "by_cvd_flip": {
            "bullish": {
                "count": len(bullish_flips),
                "wins": len([t for t in bullish_flips if t['outcome'] == 'win']),
                "win_rate": round(len([t for t in bullish_flips if t['outcome'] == 'win']) / len(bullish_flips) * 100, 2) if bullish_flips else 0,
                "pnl": round(sum(t['pnl_points'] for t in bullish_flips), 2),
            },
            "bearish": {
                "count": len(bearish_flips),
                "wins": len([t for t in bearish_flips if t['outcome'] == 'win']),
                "win_rate": round(len([t for t in bearish_flips if t['outcome'] == 'win']) / len(bearish_flips) * 100, 2) if bearish_flips else 0,
                "pnl": round(sum(t['pnl_points'] for t in bearish_flips), 2),
            },
        },
    }


def main():
    # Check for instrument flags
    use_nq = "--nq" in sys.argv
    use_mnq = "--mnq" in sys.argv
    use_es = "--es" in sys.argv
    use_m2k = "--m2k" in sys.argv
    use_mym = "--mym" in sys.argv

    if use_nq:
        bars_1s_path = "data/nq_bars_1s.json"
        output_path = "data/qwen_cvd_nq_results.json"
        symbol = "NQ"
        tick_size = 0.25
    elif use_mnq:
        bars_1s_path = "data/mnq_bars_1s.json"
        output_path = "data/qwen_cvd_mnq_results.json"
        symbol = "MNQ"
        tick_size = 0.25
    elif use_es:
        bars_1s_path = "data/es_bars_1s.json"
        output_path = "data/qwen_cvd_es_results.json"
        symbol = "ES"
        tick_size = 0.25
    elif use_m2k:
        bars_1s_path = "data/m2k_bars_1s.json"
        output_path = "data/qwen_cvd_m2k_results.json"
        symbol = "M2K"
        tick_size = 0.10
    elif use_mym:
        bars_1s_path = "data/mym_bars_1s.json"
        output_path = "data/qwen_cvd_mym_results.json"
        symbol = "MYM"
        tick_size = 1.0
    else:
        # Default: MES
        bars_1s_path = "data/bars_1s.json"
        output_path = "data/qwen_cvd_results.json"
        symbol = "MES"
        tick_size = 0.25

    print(f"[CVD Backtest] Loading 1s bars from {bars_1s_path}...", file=sys.stderr)

    if not os.path.exists(bars_1s_path):
        print(f"[Error] File not found: {bars_1s_path}", file=sys.stderr)
        sys.exit(1)

    bars_1s = load_1s_bars(bars_1s_path)
    print(f"[CVD Backtest] Loaded {len(bars_1s)} 1-second bars", file=sys.stderr)

    # Run backtest
    trades = run_backtest(
        bars_1s=bars_1s,
        use_qwen=True,
        max_trades=50,
        symbol=symbol,
        tick_size=tick_size,
    )

    # Analyze
    analysis = analyze_trades(trades)

    # Compare trailing vs fixed
    fixed_pnl = sum(t['fixed_pnl'] for t in trades)
    trailing_pnl = sum(t['pnl_points'] for t in trades)

    print("\n" + "="*60, file=sys.stderr)
    print(f"CVD FLIP BACKTEST RESULTS - {symbol}", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Total Trades: {analysis['total_trades']}", file=sys.stderr)
    print(f"Wins: {analysis['wins']} | Losses: {analysis['losses']} | Timeouts: {analysis['timeouts']}", file=sys.stderr)
    print(f"Win Rate: {analysis['win_rate']}%", file=sys.stderr)
    print(f"Total PnL (Trailing): {analysis['total_pnl_points']:+.2f} pts", file=sys.stderr)
    print(f"Total PnL (Fixed):    {fixed_pnl:+.2f} pts", file=sys.stderr)
    print(f"Avg Winner: {analysis['avg_winner']:+.2f} pts", file=sys.stderr)
    print(f"Avg Loser: {analysis['avg_loser']:+.2f} pts", file=sys.stderr)
    print("-"*60, file=sys.stderr)
    print("By Direction:", file=sys.stderr)
    for dir_name, stats in analysis['by_direction'].items():
        print(f"  {dir_name.upper()}: {stats['count']} trades, {stats['win_rate']}% win, {stats['pnl']:+.2f} pts", file=sys.stderr)
    print("-"*60, file=sys.stderr)
    print("By CVD Flip:", file=sys.stderr)
    for flip_name, stats in analysis['by_cvd_flip'].items():
        print(f"  {flip_name.upper()}: {stats['count']} trades, {stats['win_rate']}% win, {stats['pnl']:+.2f} pts", file=sys.stderr)
    print("="*60, file=sys.stderr)

    # Save results
    results = {
        "analysis": analysis,
        "trades": trades,
        "config": {
            "trigger": "cvd_flip",
            "use_qwen": True,
            "symbol": symbol,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[CVD Backtest] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
