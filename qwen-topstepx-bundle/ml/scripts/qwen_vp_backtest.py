#!/usr/bin/env python3
"""
Qwen Backtest with Volume Profile Structure Triggers

Triggers:
- Price entering value area (from outside) → target POC
- Price leaving value area → target HTF structure
- Price crossing POC → continuation signal

Volume Profile is built WALK-FORWARD from 1-second bars only up to decision point.
No future data peeking.
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


def calculate_volume_profile_incremental(
    bars_1s: List[Dict],
    tick_size: float = 0.25
) -> Dict[str, Any]:
    """
    Calculate Volume Profile from 1-second bars.
    Returns POC, VAH, VAL, and full price-volume distribution.
    """
    if not bars_1s:
        return {"poc": 0, "vah": 0, "val": 0, "distribution": {}}

    price_volume = defaultdict(float)

    for bar in bars_1s:
        volume = bar.get('v', 0) or 0
        if volume == 0:
            volume = 1  # Minimum volume for price levels touched

        # Distribute volume across the bar's range
        high = bar['h']
        low = bar['l']
        close = bar['c']

        # Price levels in this bar
        price_low = round(low / tick_size) * tick_size
        price_high = round(high / tick_size) * tick_size

        if price_high == price_low:
            price_volume[price_low] += volume
        else:
            num_levels = int((price_high - price_low) / tick_size) + 1
            vol_per_level = volume / num_levels
            price = price_low
            while price <= price_high:
                price_volume[round(price, 2)] += vol_per_level
                price += tick_size

    if not price_volume:
        return {"poc": 0, "vah": 0, "val": 0, "distribution": {}}

    sorted_prices = sorted(price_volume.keys())
    volumes = [price_volume[p] for p in sorted_prices]
    total_volume = sum(volumes)

    # POC - highest volume price
    poc_idx = np.argmax(volumes)
    poc = sorted_prices[poc_idx]

    # Value Area (70% of volume)
    target_volume = total_volume * 0.7
    current_volume = volumes[poc_idx]
    lower_idx, upper_idx = poc_idx, poc_idx

    while current_volume < target_volume and (lower_idx > 0 or upper_idx < len(volumes) - 1):
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
        "distribution": dict(price_volume),
        "total_volume": total_volume,
    }


def calculate_cvd_incremental(bars_1s: List[Dict]) -> Tuple[float, float, str]:
    """
    Calculate CVD and trend from 1-second bars.
    Returns: (cvd, cvd_ema, trend)
    """
    if not bars_1s:
        return 0.0, 0.0, "neutral"

    cvd = 0.0
    cvd_values = []

    for bar in bars_1s:
        volume = bar.get('v', 0) or 0
        bar_range = bar['h'] - bar['l']
        if bar_range > 0:
            close_position = (bar['c'] - bar['l']) / bar_range
            delta = volume * (2 * close_position - 1)
        else:
            delta = volume if bar['c'] >= bar['o'] else -volume
        cvd += delta
        cvd_values.append(cvd)

    # EMA of CVD (20 period on 1-min equivalent, so ~1200 1-second bars)
    ema_period = min(1200, len(cvd_values))
    if ema_period > 0:
        cvd_ema = sum(cvd_values[-ema_period:]) / ema_period
    else:
        cvd_ema = cvd

    trend = "up" if cvd > cvd_ema else "down"
    return cvd, cvd_ema, trend


def find_htf_structure(bars_1m: List[Dict], lookback: int = 60) -> Dict[str, float]:
    """Find higher timeframe structure levels from 1-minute bars."""
    if len(bars_1m) < 5:
        return {"swing_high": 0, "swing_low": 0, "session_high": 0, "session_low": 0}

    recent = bars_1m[-lookback:] if len(bars_1m) >= lookback else bars_1m

    session_high = max(b['h'] for b in recent)
    session_low = min(b['l'] for b in recent)

    # Find swing highs/lows
    swing_highs = []
    swing_lows = []

    for i in range(2, len(recent) - 2):
        # Swing high
        if (recent[i]['h'] > recent[i-1]['h'] and recent[i]['h'] > recent[i-2]['h'] and
            recent[i]['h'] > recent[i+1]['h'] and recent[i]['h'] > recent[i+2]['h']):
            swing_highs.append(recent[i]['h'])
        # Swing low
        if (recent[i]['l'] < recent[i-1]['l'] and recent[i]['l'] < recent[i-2]['l'] and
            recent[i]['l'] < recent[i+1]['l'] and recent[i]['l'] < recent[i+2]['l']):
            swing_lows.append(recent[i]['l'])

    return {
        "swing_high": max(swing_highs) if swing_highs else session_high,
        "swing_low": min(swing_lows) if swing_lows else session_low,
        "session_high": session_high,
        "session_low": session_low,
    }


def detect_vp_signal(
    current_price: float,
    prev_price: float,
    vp: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Detect Volume Profile based trading signals.

    Signals:
    - ENTER_VALUE_FROM_ABOVE: Price crossed into value from above VAH
    - ENTER_VALUE_FROM_BELOW: Price crossed into value from below VAL
    - EXIT_VALUE_ABOVE: Price exiting value above VAH
    - EXIT_VALUE_BELOW: Price exiting value below VAL
    - CROSS_POC_UP: Price crossed POC upward
    - CROSS_POC_DOWN: Price crossed POC downward
    """
    poc = vp['poc']
    vah = vp['vah']
    val = vp['val']

    if poc == 0:
        return None

    # Check for value area entry/exit
    prev_in_value = val <= prev_price <= vah
    curr_in_value = val <= current_price <= vah

    # Entering value from above
    if not prev_in_value and prev_price > vah and curr_in_value:
        return {
            "signal": "ENTER_VALUE_FROM_ABOVE",
            "direction": "short",
            "target": poc,
            "reasoning": f"Price entered value area from above VAH ({vah:.2f}), target POC ({poc:.2f})"
        }

    # Entering value from below
    if not prev_in_value and prev_price < val and curr_in_value:
        return {
            "signal": "ENTER_VALUE_FROM_BELOW",
            "direction": "long",
            "target": poc,
            "reasoning": f"Price entered value area from below VAL ({val:.2f}), target POC ({poc:.2f})"
        }

    # Exiting value above
    if prev_in_value and current_price > vah:
        return {
            "signal": "EXIT_VALUE_ABOVE",
            "direction": "long",
            "target": None,  # Will be set to HTF structure
            "reasoning": f"Price exiting value above VAH ({vah:.2f}), bullish breakout"
        }

    # Exiting value below
    if prev_in_value and current_price < val:
        return {
            "signal": "EXIT_VALUE_BELOW",
            "direction": "short",
            "target": None,  # Will be set to HTF structure
            "reasoning": f"Price exiting value below VAL ({val:.2f}), bearish breakdown"
        }

    # No POC cross signals - only trade on value area entries/exits
    return None


def build_qwen_prompt(
    current_price: float,
    timestamp: str,
    signal: Dict[str, Any],
    vp: Dict[str, Any],
    cvd: float,
    cvd_ema: float,
    cvd_trend: str,
    htf_structure: Dict[str, float],
    recent_bars: List[Dict],
    tick_size: float = 0.25,
) -> str:
    """Build prompt for Qwen decision."""

    # Format recent bars
    def format_bars(bars, n=5):
        if not bars:
            return "N/A"
        return " ".join([
            f"{b['c']:.2f}({'↑' if b['c'] > b['o'] else '↓'})"
            for b in bars[-n:]
        ])

    # Suggested target
    if signal['target']:
        suggested_target = signal['target']
    elif signal['direction'] == 'long':
        suggested_target = htf_structure['swing_high']
    else:
        suggested_target = htf_structure['swing_low']

    # Distance calculations
    if signal['direction'] == 'long':
        potential_reward = suggested_target - current_price
        swing_stop = htf_structure['swing_low']
        potential_risk = current_price - swing_stop
    else:
        potential_reward = current_price - suggested_target
        swing_stop = htf_structure['swing_high']
        potential_risk = swing_stop - current_price

    potential_rr = potential_reward / potential_risk if potential_risk > 0 else 0

    # Check if at/near ATH
    at_ath = current_price >= htf_structure['session_high'] * 0.999

    prompt = f"""MES @ {current_price:.2f} | {timestamp}

=== VOLUME PROFILE SIGNAL ===
Signal: {signal['signal']}
Direction: {signal['direction'].upper()}
Reasoning: {signal['reasoning']}

=== VOLUME PROFILE (Built from session 1-second data) ===
POC: {vp['poc']:.2f}
VAH: {vp['vah']:.2f}
VAL: {vp['val']:.2f}
Price vs POC: {"Above" if current_price > vp['poc'] else "Below"} by {abs(current_price - vp['poc']):.2f} pts

=== CVD (from 1-second bars) ===
CVD: {cvd:.0f} | EMA: {cvd_ema:.0f}
Trend: {cvd_trend.upper()} {"(confirms signal)" if (cvd_trend == "up" and signal['direction'] == "long") or (cvd_trend == "down" and signal['direction'] == "short") else "(diverges from signal)"}

=== HTF STRUCTURE ===
Session High: {htf_structure['session_high']:.2f}
Session Low: {htf_structure['session_low']:.2f}
Swing High: {htf_structure['swing_high']:.2f}
Swing Low: {htf_structure['swing_low']:.2f}
{"*** AT/NEAR ALL-TIME HIGH - UNCHARTED TERRITORY ***" if at_ath else ""}

=== TRADE SETUP ===
Suggested Target: {suggested_target:.2f}
Suggested Stop (swing level): {swing_stop:.2f}
Potential R:R: 1:{potential_rr:.2f}

=== RECENT PRICE ACTION ===
{format_bars(recent_bars, 5)}

=== YOUR TASK ===
1. Evaluate if this VP signal has edge
2. If taking the trade, set entry, stop, and target
3. For breakouts to new highs: estimate target from prior move magnitude
4. MINIMUM 1:2 R:R required
5. If CVD diverges from signal or R:R is poor, HOLD

Respond with JSON only:
{{"decision":"BUY|SELL|HOLD","confidence":0-100,"entryPrice":{current_price:.2f},"stopLoss":number,"target":number,"reasoning":"brief"}}"""

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
                "options": {"temperature": 0.3, "num_predict": 256}
            },
            timeout=30,
        )

        if response.status_code != 200:
            return None

        result = response.json()
        content = result.get("response", "")

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
    """Maintain minimum R:R ratio - trail stop if R:R deteriorates."""
    is_long = direction == "long"

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


def simulate_trade(
    entry_price: float,
    stop_loss: float,
    target: float,
    direction: str,
    future_bars: List[Dict],
    min_rr: float = 2.0,
    tick_size: float = 0.25,
) -> Dict[str, Any]:
    """Simulate trade with R:R maintenance."""
    is_long = direction == "long"
    current_stop = stop_loss
    max_profit = 0.0
    stop_adjustments = 0

    for i, bar in enumerate(future_bars):
        high = bar['h']
        low = bar['l']
        close = bar['c']

        if is_long:
            max_profit = max(max_profit, high - entry_price)
        else:
            max_profit = max(max_profit, entry_price - low)

        new_stop = maintain_rr_stop(close, current_stop, target, direction, min_rr, tick_size)
        if new_stop:
            current_stop = new_stop
            stop_adjustments += 1

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

    last_close = future_bars[-1]['c'] if future_bars else entry_price
    pnl = (last_close - entry_price) if is_long else (entry_price - last_close)

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
    max_trades: int = 100,
    min_bars_for_vp: int = 3600,  # 1 hour of 1s bars minimum
) -> List[Dict]:
    """Run backtest with VP structure signals."""

    print("[Backtest] Aggregating to 1-minute bars...", file=sys.stderr)
    bars_1m = aggregate_bars(bars_1s, 1)
    print(f"[Backtest] 1m bars: {len(bars_1m)}", file=sys.stderr)

    trades = []
    max_hold_bars = 60
    last_signal_bar = -10  # Prevent rapid re-signaling

    print(f"[Backtest] Scanning for VP signals...", file=sys.stderr)

    signals_found = 0

    for bar_idx in range(60, len(bars_1m) - max_hold_bars):
        # Only check every minute (at bar boundaries)
        current_bar = bars_1m[bar_idx]
        prev_bar = bars_1m[bar_idx - 1]

        current_price = current_bar['c']
        prev_price = prev_bar['c']

        # Get 1s bars up to this point for VP calculation
        current_ts = current_bar['t']
        bars_1s_to_now = []
        for b in bars_1s:
            if b['t'] <= current_ts:
                bars_1s_to_now.append(b)
            else:
                break

        # Need minimum data for meaningful VP
        if len(bars_1s_to_now) < min_bars_for_vp:
            continue

        # Calculate VP from 1s bars (walk-forward, no peeking)
        vp = calculate_volume_profile_incremental(bars_1s_to_now)

        # Detect VP signal
        signal = detect_vp_signal(current_price, prev_price, vp)

        if not signal:
            continue

        # Prevent rapid re-signaling
        if bar_idx - last_signal_bar < 5:
            continue

        signals_found += 1
        last_signal_bar = bar_idx

        print(f"[Signal] {current_bar['t']} - {signal['signal']} @ {current_price:.2f} (POC:{vp['poc']:.2f} VAH:{vp['vah']:.2f} VAL:{vp['val']:.2f})", file=sys.stderr)

        # Calculate CVD
        cvd, cvd_ema, cvd_trend = calculate_cvd_incremental(bars_1s_to_now)

        # Get HTF structure
        htf_structure = find_htf_structure(bars_1m[:bar_idx + 1])

        # Set target for breakout signals
        if signal['target'] is None:
            if signal['direction'] == 'long':
                # Estimate target from prior move
                recent_range = htf_structure['session_high'] - htf_structure['session_low']
                signal['target'] = current_price + recent_range * 0.5
            else:
                recent_range = htf_structure['session_high'] - htf_structure['session_low']
                signal['target'] = current_price - recent_range * 0.5

        # Recent bars for context
        recent_bars = bars_1m[bar_idx-5:bar_idx+1]

        # Build prompt and call Qwen
        prompt = build_qwen_prompt(
            current_price=current_price,
            timestamp=current_bar['t'][-8:] if isinstance(current_bar['t'], str) else str(current_bar['t']),
            signal=signal,
            vp=vp,
            cvd=cvd,
            cvd_ema=cvd_ema,
            cvd_trend=cvd_trend,
            htf_structure=htf_structure,
            recent_bars=recent_bars,
        )

        decision = call_qwen(prompt)

        if not decision:
            print(f"  -> Qwen: no response", file=sys.stderr)
            continue
        if decision.get('decision') == 'HOLD':
            print(f"  -> Qwen: HOLD - {decision.get('reasoning', 'no reason')}", file=sys.stderr)
            continue

        confidence = decision.get('confidence', 0)
        if confidence < 70:
            print(f"  -> Qwen: low confidence {confidence}", file=sys.stderr)
            continue

        direction = "long" if decision.get('decision') == 'BUY' else "short"
        stop_loss = decision.get('stopLoss', 0)
        target = decision.get('target', 0)
        reasoning = decision.get('reasoning', '')

        # Validate
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

        # Simulate trade
        future_bars = bars_1m[bar_idx + 1:bar_idx + 1 + max_hold_bars]
        result = simulate_trade(current_price, stop_loss, target, direction, future_bars)

        trade = {
            "timestamp": current_bar['t'],
            "bar_index": bar_idx,
            "signal": signal['signal'],
            "direction": direction,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "target": target,
            "risk_reward": round(rr, 2),
            "confidence": confidence,
            "reasoning": reasoning,
            "cvd_trend": cvd_trend,
            "vp_poc": vp['poc'],
            "vp_vah": vp['vah'],
            "vp_val": vp['val'],
            "outcome": result['outcome'],
            "exit_price": result['exit_price'],
            "pnl_points": round(result['pnl'], 2),
            "bars_held": result['bars_held'],
            "max_profit": round(result.get('max_profit', 0), 2),
            "final_stop": round(result.get('final_stop', stop_loss), 2),
            "stop_adjustments": result.get('stop_adjustments', 0),
        }
        trades.append(trade)

        if len(trades) % 10 == 0:
            print(f"[Backtest] {len(trades)} trades taken...", file=sys.stderr)

        if len(trades) >= max_trades:
            break

    print(f"[Backtest] Found {signals_found} VP signals, took {len(trades)} trades", file=sys.stderr)
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

    longs = [t for t in trades if t['direction'] == 'long']
    shorts = [t for t in trades if t['direction'] == 'short']

    # By signal type
    by_signal = {}
    for t in trades:
        sig = t['signal']
        if sig not in by_signal:
            by_signal[sig] = {'count': 0, 'wins': 0, 'pnl': 0}
        by_signal[sig]['count'] += 1
        if t['outcome'] == 'win':
            by_signal[sig]['wins'] += 1
        by_signal[sig]['pnl'] += t['pnl_points']

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
                "wins": len([t for t in longs if t['outcome'] == 'win']),
                "pnl": round(sum(t['pnl_points'] for t in longs), 2),
            },
            "short": {
                "count": len(shorts),
                "wins": len([t for t in shorts if t['outcome'] == 'win']),
                "pnl": round(sum(t['pnl_points'] for t in shorts), 2),
            },
        },
        "by_signal": by_signal,
    }


def main():
    bars_1s_path = "data/bars_1s.json"
    output_path = "data/qwen_vp_results.json"

    print("=" * 70, file=sys.stderr)
    print("QWEN BACKTEST + VOLUME PROFILE STRUCTURE", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    print(f"[Load] Loading 1s bars...", file=sys.stderr)
    bars_1s_all = load_1s_bars(bars_1s_path)
    bars_1s = [b for b in bars_1s_all if '2025-12-05' in b['t']]
    print(f"[Load] Filtered to {len(bars_1s)} bars on 12/5/2025", file=sys.stderr)

    trades = run_backtest(bars_1s=bars_1s, max_trades=50)

    analysis = analyze_results(trades)

    print("\n" + "=" * 70, file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total trades: {analysis.get('total_trades', 0)}", file=sys.stderr)
    print(f"Wins: {analysis.get('wins', 0)} | Losses: {analysis.get('losses', 0)} | Timeouts: {analysis.get('timeouts', 0)}", file=sys.stderr)
    print(f"Win Rate: {analysis.get('win_rate', 0)}%", file=sys.stderr)
    print(f"Total PnL: {analysis.get('total_pnl_points', 0)} points", file=sys.stderr)
    print(f"Avg Winner: {analysis.get('avg_winner', 0)} | Avg Loser: {analysis.get('avg_loser', 0)}", file=sys.stderr)

    if 'by_signal' in analysis:
        print("\nBy Signal Type:", file=sys.stderr)
        for sig, stats in analysis['by_signal'].items():
            wr = round(stats['wins'] / stats['count'] * 100, 1) if stats['count'] > 0 else 0
            print(f"  {sig}: {stats['count']} trades, {wr}% win, {stats['pnl']:.2f} pts", file=sys.stderr)

    print("=" * 70, file=sys.stderr)

    result = {
        "analysis": analysis,
        "trades": trades,
        "config": {"date": "2025-12-05", "trigger": "volume_profile"},
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Save] Results saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
