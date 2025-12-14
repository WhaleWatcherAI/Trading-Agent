#!/usr/bin/env python3
"""
CVD Crossover Backtest - With Vortex Dynamic Stop Loss
Aggregate 1s bars to 1min, then calculate CVD per minute
Trade when CVD crosses above/below its 20-period EMA (20 min lookback)
- Long when CVD crosses above EMA (bullish)
- Short when CVD crosses below EMA (bearish)
- SL at recent swing low (long) / swing high (short)
- TP at recent swing high (long) / swing low (short)
- Dynamic vortex curve: SL tightens aggressively as price approaches TP
"""

import argparse
import json
import sys
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


def find_swing_high(bars: List[Dict], end_idx: int, lookback: int = 20) -> Optional[float]:
    """Find the most recent swing high (local maximum) in the lookback window."""
    if end_idx < 2:
        return None

    start_idx = max(0, end_idx - lookback)
    swing_high = None

    for i in range(start_idx + 1, end_idx):
        if i + 1 >= len(bars):
            break
        # A swing high: higher than both neighbors
        if bars[i]['h'] > bars[i-1]['h'] and bars[i]['h'] > bars[i+1]['h']:
            if swing_high is None or bars[i]['h'] > swing_high:
                swing_high = bars[i]['h']

    # If no swing found, use highest high in window
    if swing_high is None:
        swing_high = max(bars[j]['h'] for j in range(start_idx, end_idx))

    return swing_high


def find_swing_low(bars: List[Dict], end_idx: int, lookback: int = 20) -> Optional[float]:
    """Find the most recent swing low (local minimum) in the lookback window."""
    if end_idx < 2:
        return None

    start_idx = max(0, end_idx - lookback)
    swing_low = None

    for i in range(start_idx + 1, end_idx):
        if i + 1 >= len(bars):
            break
        # A swing low: lower than both neighbors
        if bars[i]['l'] < bars[i-1]['l'] and bars[i]['l'] < bars[i+1]['l']:
            if swing_low is None or bars[i]['l'] < swing_low:
                swing_low = bars[i]['l']

    # If no swing found, use lowest low in window
    if swing_low is None:
        swing_low = min(bars[j]['l'] for j in range(start_idx, end_idx))

    return swing_low


def vortex_dynamic_stop(
    entry_price: float,
    current_price: float,
    initial_stop: float,
    target_price: float,
    side: str,  # 'long' or 'short'
    tick_size: float = 0.25,
    breakeven_trigger_ticks: int = 8,  # Wait for 8 ticks profit before moving SL
) -> float:
    """
    Vortex curve dynamic stop - like the Qwen implementation.

    Phase 1: Wait until price is 8+ ticks in profit
    Phase 2: Move SL to breakeven + 1 tick
    Phase 3: Trail loosely in first half (give room to breathe)
    Phase 4: Tighten aggressively in second half (vortex effect)

    The idea: Why risk $995 profit for $5 more when you're near TP?
    """
    # Calculate profit in ticks
    if side == 'long':
        profit_points = current_price - entry_price
        total_distance = target_price - entry_price
    else:
        profit_points = entry_price - current_price
        total_distance = entry_price - target_price

    profit_ticks = profit_points / tick_size

    # PHASE 1: Not enough profit yet - keep initial stop
    if profit_ticks < breakeven_trigger_ticks:
        return initial_stop

    # PHASE 2: Just hit breakeven trigger - move to BE + 1 tick
    breakeven_plus_buffer = entry_price + tick_size if side == 'long' else entry_price - tick_size

    if total_distance <= 0:
        return breakeven_plus_buffer

    # Calculate progress toward target (0 to 1)
    progress_ratio = max(0, min(1, profit_points / total_distance))

    # PHASE 3 & 4: Vortex trailing curve
    # First half (0-50%): Trail loosely, lock ~10-30% of profit
    # Second half (50-100%): Tighten aggressively, lock 30-92% of profit

    if progress_ratio < 0.5:
        # First half: loose trailing
        # Linear from 10% at start to 30% at midpoint
        lock_ratio = 0.10 + (progress_ratio * 0.40)  # 10% -> 30%
    else:
        # Second half: aggressive tightening (vortex effect)
        # Use sigmoid for smooth acceleration
        # At 50%: lock 30%, at 75%: lock 60%, at 90%: lock 85%, at 95%: lock 92%
        adjusted_progress = (progress_ratio - 0.5) * 2  # Rescale 0.5-1.0 to 0-1

        SIGMOID_MAX = 0.92
        SIGMOID_STEEPNESS = 5.0
        SIGMOID_MIDPOINT = 0.5

        sigmoid_input = SIGMOID_STEEPNESS * (adjusted_progress - SIGMOID_MIDPOINT)
        sigmoid_value = 1 / (1 + math.exp(-sigmoid_input))

        # Scale from 30% to 92%
        lock_ratio = 0.30 + (sigmoid_value * (SIGMOID_MAX - 0.30))

    # Calculate stop based on lock ratio (percentage of total distance to lock)
    if side == 'long':
        new_stop = entry_price + total_distance * lock_ratio
        # Never move stop backwards, and always at least BE + 1 tick
        return max(initial_stop, breakeven_plus_buffer, new_stop)
    else:
        new_stop = entry_price - total_distance * lock_ratio
        # Never move stop backwards (for short, lower is tighter)
        return min(initial_stop, breakeven_plus_buffer, new_stop)


def aggregate_to_1min(bars_1s: List[Dict]) -> List[Dict]:
    """Aggregate 1-second bars to 1-minute bars."""
    if not bars_1s:
        return []

    minute_bars = []
    current_minute = None
    current_bar = None

    for bar in bars_1s:
        ts = bar['t']
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            dt = ts

        minute_ts = dt.replace(second=0, microsecond=0)

        if current_minute != minute_ts:
            if current_bar:
                minute_bars.append(current_bar)

            current_minute = minute_ts
            current_bar = {
                't': minute_ts.isoformat(),
                'o': bar['o'],
                'h': bar['h'],
                'l': bar['l'],
                'c': bar['c'],
                'v': bar.get('v', 0) or 0,
            }
        else:
            current_bar['h'] = max(current_bar['h'], bar['h'])
            current_bar['l'] = min(current_bar['l'], bar['l'])
            current_bar['c'] = bar['c']
            current_bar['v'] += bar.get('v', 0) or 0

    if current_bar:
        minute_bars.append(current_bar)

    return minute_bars


def calculate_cvd_with_ema(bars_1m: List[Dict], ema_period: int = 20) -> tuple:
    """Calculate CVD and its EMA from 1-minute bars."""
    if not bars_1m:
        return [], []

    cvd = 0.0
    cvd_values = []

    for bar in bars_1m:
        volume = bar.get('v', 0) or 0
        open_price = bar['o']
        close_price = bar['c']
        high_price = bar['h']
        low_price = bar['l']

        bar_range = high_price - low_price
        if bar_range > 0:
            close_position = (close_price - low_price) / bar_range
            delta = volume * (2 * close_position - 1)
        else:
            delta = volume if close_price >= open_price else -volume

        cvd += delta
        cvd_values.append(cvd)

    # EMA of CVD
    cvd_ema = []
    if len(cvd_values) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema = sum(cvd_values[:ema_period]) / ema_period
        cvd_ema = [ema] * ema_period

        for i in range(ema_period, len(cvd_values)):
            ema = (cvd_values[i] * multiplier) + (ema * (1 - multiplier))
            cvd_ema.append(ema)
    else:
        cvd_ema = cvd_values.copy()

    return cvd_values, cvd_ema


def run_backtest(
    bars_1s: List[Dict],
    ema_period: int = 20,
    tick_size: float = 0.25,
    swing_lookback: int = 20,
    use_vortex_sl: bool = True,
) -> Dict[str, Any]:
    """
    Run CVD crossover backtest with swing-based SL/TP and vortex dynamic stop.

    - Entry on CVD crossover (above/below EMA)
    - SL at swing low (long) / swing high (short)
    - TP at swing high (long) / swing low (short)
    - Vortex curve dynamically tightens SL as price approaches TP
    """

    # Aggregate to 1-minute bars first
    print(f"[CVD] Aggregating {len(bars_1s)} 1s bars to 1min bars...", file=sys.stderr)
    bars_1m = aggregate_to_1min(bars_1s)
    print(f"[CVD] Got {len(bars_1m)} 1-minute bars", file=sys.stderr)

    print(f"[CVD] Calculating CVD and EMA({ema_period}) on 1-min bars...", file=sys.stderr)
    cvd_values, cvd_ema = calculate_cvd_with_ema(bars_1m, ema_period)

    if len(cvd_values) < ema_period + 10:
        return {"error": "Not enough data"}

    print(f"[CVD] Running vortex SL backtest (swing lookback={swing_lookback})...", file=sys.stderr)

    trades = []
    total_pnl = 0
    total_pnl_ticks = 0
    wins = 0
    losses = 0
    long_trades = 0
    short_trades = 0
    long_pnl = 0
    short_pnl = 0

    # Track exit reasons
    exit_reasons = {'sl': 0, 'tp': 0, 'flip': 0, 'eod': 0}

    # Skip first 30 minutes for warmup
    start_idx = max(ema_period + 1, 30)

    # Position state
    position = None  # 'long', 'short', or None
    entry_price = 0
    entry_idx = 0
    entry_cvd = 0
    initial_sl = 0
    current_sl = 0
    target_price = 0

    prev_above = cvd_values[start_idx] > cvd_ema[start_idx]

    for i in range(start_idx + 1, len(bars_1m)):
        cvd = cvd_values[i]
        ema = cvd_ema[i]
        curr_above = cvd > ema
        bar = bars_1m[i]
        high = bar['h']
        low = bar['l']
        close = bar['c']

        # If in a position, check SL/TP first
        if position is not None:
            exit_price = None
            exit_reason = None

            # Calculate vortex dynamic stop
            if use_vortex_sl:
                current_sl = vortex_dynamic_stop(
                    entry_price=entry_price,
                    current_price=close,
                    initial_stop=initial_sl,
                    target_price=target_price,
                    side=position,
                    tick_size=tick_size,
                )

            if position == 'long':
                # Check if SL hit (low touches stop)
                if low <= current_sl:
                    exit_price = current_sl
                    exit_reason = 'sl'
                # Check if TP hit (high touches target)
                elif high >= target_price:
                    exit_price = target_price
                    exit_reason = 'tp'
            else:  # short
                # Check if SL hit (high touches stop)
                if high >= current_sl:
                    exit_price = current_sl
                    exit_reason = 'sl'
                # Check if TP hit (low touches target)
                elif low <= target_price:
                    exit_price = target_price
                    exit_reason = 'tp'

            # Exit if SL or TP hit
            if exit_price is not None:
                if position == 'long':
                    pnl = exit_price - entry_price
                    pnl_ticks = pnl / tick_size
                    long_pnl += pnl_ticks
                else:
                    pnl = entry_price - exit_price
                    pnl_ticks = pnl / tick_size
                    short_pnl += pnl_ticks

                total_pnl += pnl
                total_pnl_ticks += pnl_ticks

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                exit_reasons[exit_reason] += 1

                trades.append({
                    'entry_time': bars_1m[entry_idx]['t'],
                    'exit_time': bar['t'],
                    'direction': position,
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2),
                    'initial_sl': round(initial_sl, 2),
                    'final_sl': round(current_sl, 2),
                    'target': round(target_price, 2),
                    'pnl': round(pnl, 2),
                    'pnl_ticks': round(pnl_ticks, 2),
                    'bars_held': i - entry_idx,
                    'exit_reason': exit_reason,
                    'entry_cvd': round(entry_cvd, 2),
                    'exit_cvd': round(cvd, 2),
                })

                position = None

        # Check for crossover (signal flip) - can enter new position
        if curr_above != prev_above:
            # If still in position (wasn't stopped out), close on flip
            if position is not None:
                exit_price = close
                exit_reason = 'flip'

                if position == 'long':
                    pnl = exit_price - entry_price
                    pnl_ticks = pnl / tick_size
                    long_pnl += pnl_ticks
                else:
                    pnl = entry_price - exit_price
                    pnl_ticks = pnl / tick_size
                    short_pnl += pnl_ticks

                total_pnl += pnl
                total_pnl_ticks += pnl_ticks

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                exit_reasons[exit_reason] += 1

                trades.append({
                    'entry_time': bars_1m[entry_idx]['t'],
                    'exit_time': bar['t'],
                    'direction': position,
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2),
                    'initial_sl': round(initial_sl, 2),
                    'final_sl': round(current_sl, 2),
                    'target': round(target_price, 2),
                    'pnl': round(pnl, 2),
                    'pnl_ticks': round(pnl_ticks, 2),
                    'bars_held': i - entry_idx,
                    'exit_reason': exit_reason,
                    'entry_cvd': round(entry_cvd, 2),
                    'exit_cvd': round(cvd, 2),
                })

                position = None

            # Open new position
            position = 'long' if curr_above else 'short'
            entry_price = close
            entry_idx = i
            entry_cvd = cvd

            # Find swing levels for SL/TP
            swing_high = find_swing_high(bars_1m, i, swing_lookback)
            swing_low = find_swing_low(bars_1m, i, swing_lookback)

            if position == 'long':
                initial_sl = swing_low - tick_size  # SL just below swing low
                current_sl = initial_sl
                target_price = swing_high + tick_size  # TP just above swing high
                long_trades += 1
            else:
                initial_sl = swing_high + tick_size  # SL just above swing high
                current_sl = initial_sl
                target_price = swing_low - tick_size  # TP just below swing low
                short_trades += 1

        prev_above = curr_above

    # Close final position at end of data
    if position is not None:
        exit_price = bars_1m[-1]['c']
        exit_reason = 'eod'

        if position == 'long':
            pnl = exit_price - entry_price
            pnl_ticks = pnl / tick_size
            long_pnl += pnl_ticks
        else:
            pnl = entry_price - exit_price
            pnl_ticks = pnl / tick_size
            short_pnl += pnl_ticks

        total_pnl += pnl
        total_pnl_ticks += pnl_ticks

        if pnl > 0:
            wins += 1
        else:
            losses += 1

        exit_reasons[exit_reason] += 1

        trades.append({
            'entry_time': bars_1m[entry_idx]['t'],
            'exit_time': bars_1m[-1]['t'],
            'direction': position,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'initial_sl': round(initial_sl, 2),
            'final_sl': round(current_sl, 2),
            'target': round(target_price, 2),
            'pnl': round(pnl, 2),
            'pnl_ticks': round(pnl_ticks, 2),
            'bars_held': len(bars_1m) - 1 - entry_idx,
            'exit_reason': exit_reason,
            'entry_cvd': round(entry_cvd, 2),
            'exit_cvd': round(cvd_values[-1], 2),
        })

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    # Average trade stats
    avg_pnl_ticks = total_pnl_ticks / total_trades if total_trades > 0 else 0
    avg_bars_held = sum(t['bars_held'] for t in trades) / len(trades) if trades else 0

    # Long vs Short breakdown
    long_wr = sum(1 for t in trades if t['direction'] == 'long' and t['pnl'] > 0) / long_trades * 100 if long_trades > 0 else 0
    short_wr = sum(1 for t in trades if t['direction'] == 'short' and t['pnl'] > 0) / short_trades * 100 if short_trades > 0 else 0

    # Calculate average R:R for trades
    avg_rr_ratio = 0
    rr_trades = [t for t in trades if t.get('initial_sl') and t.get('target')]
    if rr_trades:
        rr_ratios = []
        for t in rr_trades:
            if t['direction'] == 'long':
                risk = t['entry_price'] - t['initial_sl']
                reward = t['target'] - t['entry_price']
            else:
                risk = t['initial_sl'] - t['entry_price']
                reward = t['entry_price'] - t['target']
            if risk > 0:
                rr_ratios.append(reward / risk)
        if rr_ratios:
            avg_rr_ratio = sum(rr_ratios) / len(rr_ratios)

    results = {
        'strategy': 'CVD Crossover - Vortex Dynamic SL (Swing TP/SL)',
        'ema_period': ema_period,
        'swing_lookback': swing_lookback,
        'vortex_sl_enabled': use_vortex_sl,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 2),
        'total_pnl_ticks': round(total_pnl_ticks, 2),
        'avg_pnl_per_trade': round(avg_pnl_ticks, 2),
        'avg_bars_held': round(avg_bars_held, 1),
        'avg_rr_ratio': round(avg_rr_ratio, 2),
        'exit_reasons': exit_reasons,
        'direction_breakdown': {
            'long_trades': long_trades,
            'long_win_rate': round(long_wr, 2),
            'long_pnl_ticks': round(long_pnl, 2),
            'short_trades': short_trades,
            'short_win_rate': round(short_wr, 2),
            'short_pnl_ticks': round(short_pnl, 2),
        },
        'trades': trades[:100],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="CVD Crossover Backtest with Vortex Dynamic SL")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file with 1s bars")
    parser.add_argument("--ema-period", type=int, default=20, help="EMA period (default: 20)")
    parser.add_argument("--swing-lookback", type=int, default=20, help="Swing lookback period (default: 20)")
    parser.add_argument("--no-vortex", action="store_true", help="Disable vortex dynamic SL")

    args = parser.parse_args()

    print(f"[CVD] Loading data from {args.input}...", file=sys.stderr)
    with open(args.input, 'r') as f:
        data = json.load(f)

    bars_1s = data.get("bars", [])
    print(f"[CVD] Loaded {len(bars_1s)} 1-second bars", file=sys.stderr)

    results = run_backtest(
        bars_1s=bars_1s,
        ema_period=args.ema_period,
        swing_lookback=args.swing_lookback,
        use_vortex_sl=not args.no_vortex,
    )

    # Print results
    print("\n" + "=" * 70, file=sys.stderr)
    print(f"CVD CROSSOVER BACKTEST - VORTEX DYNAMIC SL", file=sys.stderr)
    print(f"Strategy: Long when CVD > EMA, Short when CVD < EMA", file=sys.stderr)
    print(f"EMA Period: {args.ema_period} | Swing Lookback: {args.swing_lookback}", file=sys.stderr)
    print(f"Vortex Dynamic SL: {'ENABLED' if not args.no_vortex else 'DISABLED'}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Total trades: {results['total_trades']}", file=sys.stderr)
    print(f"  Wins: {results['wins']} | Losses: {results['losses']}", file=sys.stderr)
    print(f"  Win rate: {results['win_rate']}%", file=sys.stderr)
    print(f"  Total PnL: {results['total_pnl_ticks']} ticks", file=sys.stderr)
    print(f"  Avg PnL per trade: {results['avg_pnl_per_trade']} ticks", file=sys.stderr)
    print(f"  Avg bars held: {results['avg_bars_held']} min", file=sys.stderr)
    print(f"  Avg R:R ratio: {results.get('avg_rr_ratio', 'N/A')}", file=sys.stderr)

    print(f"\nExit Reasons:", file=sys.stderr)
    er = results.get('exit_reasons', {})
    print(f"  SL: {er.get('sl', 0)} | TP: {er.get('tp', 0)} | Flip: {er.get('flip', 0)} | EOD: {er.get('eod', 0)}", file=sys.stderr)

    print(f"\nDirection Breakdown:", file=sys.stderr)
    db = results['direction_breakdown']
    print(f"  LONG:  {db['long_trades']} trades, {db['long_win_rate']}% win rate, {db['long_pnl_ticks']} ticks", file=sys.stderr)
    print(f"  SHORT: {db['short_trades']} trades, {db['short_win_rate']}% win rate, {db['short_pnl_ticks']} ticks", file=sys.stderr)

    if results['total_pnl_ticks'] > 0:
        print(f"\n  PROFITABLE", file=sys.stderr)
    else:
        print(f"\n  NOT PROFITABLE", file=sys.stderr)

    print("=" * 70, file=sys.stderr)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
