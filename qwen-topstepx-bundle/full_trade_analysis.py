#!/usr/bin/env python3
"""
Comprehensive trade-by-trade analysis from live logs
Includes: Entry, SL, TP, P&L, WR, DD, SL/TP distances
"""

import json
import re
from datetime import datetime
from collections import defaultdict

# Parse live logs for all trades with full details
def parse_gc_trades():
    """Extract all GC trades from live log"""
    trades = []
    with open('ml/logs/live_trading_gc_20251210.log', 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for SIGNAL lines
        if 'SIGNAL: ' in line and ('SELL @' in line or 'BUY @' in line):
            match = re.search(r'SIGNAL: (\w+) @ ([\d.]+)', line)
            if match:
                direction = match.group(1)
                entry = float(match.group(2))

                # Get timestamp from previous line
                prev_line = lines[i-1] if i > 0 else ''
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', prev_line)
                if not time_match:
                    time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)

                timestamp = time_match.group(1) if time_match else "Unknown"

                # Get SL and TP from next lines
                sl = None
                tp = None
                sl_ticks = None
                tp_ticks = None

                j = i + 1
                while j < min(i + 10, len(lines)):
                    if 'SL:' in lines[j]:
                        sl_match = re.search(r'SL: ([\d.]+) \((-?\d+) ticks\)', lines[j])
                        if sl_match:
                            sl = float(sl_match.group(1))
                            sl_ticks = int(sl_match.group(2))
                    elif 'TP:' in lines[j]:
                        tp_match = re.search(r'TP: ([\d.]+) \((-?\d+) ticks\)', lines[j])
                        if tp_match:
                            tp = float(tp_match.group(1))
                            tp_ticks = int(tp_match.group(2))

                    if sl is not None and tp is not None:
                        break
                    j += 1

                if sl is not None and tp is not None:
                    trades.append({
                        'time': timestamp,
                        'symbol': 'GC',
                        'direction': direction,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'sl_ticks': abs(sl_ticks) if sl_ticks else None,
                        'tp_ticks': abs(tp_ticks) if tp_ticks else None,
                    })
        i += 1

    return trades

def parse_es_trades():
    """Extract all ES trades from live log"""
    trades = []
    with open('ml/logs/live_trading_es_20251210.log', 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        if 'SIGNAL: ' in line and ('SELL @' in line or 'BUY @' in line):
            match = re.search(r'SIGNAL: (\w+) @ ([\d.]+)', line)
            if match:
                direction = match.group(1)
                entry = float(match.group(2))

                prev_line = lines[i-1] if i > 0 else ''
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', prev_line)
                if not time_match:
                    time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)

                timestamp = time_match.group(1) if time_match else "Unknown"

                sl = None
                tp = None
                sl_ticks = None
                tp_ticks = None

                j = i + 1
                while j < min(i + 10, len(lines)):
                    if 'SL:' in lines[j]:
                        sl_match = re.search(r'SL: ([\d.]+) \((-?\d+) ticks\)', lines[j])
                        if sl_match:
                            sl = float(sl_match.group(1))
                            sl_ticks = int(sl_match.group(2))
                    elif 'TP:' in lines[j]:
                        tp_match = re.search(r'TP: ([\d.]+) \((-?\d+) ticks\)', lines[j])
                        if tp_match:
                            tp = float(tp_match.group(1))
                            tp_ticks = int(tp_match.group(2))

                    if sl is not None and tp is not None:
                        break
                    j += 1

                if sl is not None and tp is not None:
                    trades.append({
                        'time': timestamp,
                        'symbol': 'ES',
                        'direction': direction,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'sl_ticks': abs(sl_ticks) if sl_ticks else None,
                        'tp_ticks': abs(tp_ticks) if tp_ticks else None,
                    })
        i += 1

    return trades

def parse_nq_trades():
    """Extract all NQ trades from live log"""
    trades = []
    with open('ml/logs/live_trading_nq_20251210.log', 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        if 'SIGNAL: ' in line and ('SELL @' in line or 'BUY @' in line):
            match = re.search(r'SIGNAL: (\w+) @ ([\d.]+)', line)
            if match:
                direction = match.group(1)
                entry = float(match.group(2))

                prev_line = lines[i-1] if i > 0 else ''
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', prev_line)
                if not time_match:
                    time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)

                timestamp = time_match.group(1) if time_match else "Unknown"

                sl = None
                tp = None
                sl_ticks = None
                tp_ticks = None

                j = i + 1
                while j < min(i + 10, len(lines)):
                    if 'SL:' in lines[j]:
                        sl_match = re.search(r'SL: ([\d.]+) \((-?\d+) ticks\)', lines[j])
                        if sl_match:
                            sl = float(sl_match.group(1))
                            sl_ticks = int(sl_match.group(2))
                    elif 'TP:' in lines[j]:
                        tp_match = re.search(r'TP: ([\d.]+) \((-?\d+) ticks\)', lines[j])
                        if tp_match:
                            tp = float(tp_match.group(1))
                            tp_ticks = int(tp_match.group(2))

                    if sl is not None and tp is not None:
                        break
                    j += 1

                if sl is not None and tp is not None:
                    trades.append({
                        'time': timestamp,
                        'symbol': 'NQ',
                        'direction': direction,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'sl_ticks': abs(sl_ticks) if sl_ticks else None,
                        'tp_ticks': abs(tp_ticks) if tp_ticks else None,
                    })
        i += 1

    return trades

def calculate_stats(trades):
    """Calculate P&L, WR, DD, etc. from live user data"""

    # User-provided P&L data (actual results from trading)
    pnl_data = {
        # GC minis
        ('2025-12-10 14:02:03', 'GC', 'SELL'): -390,
        ('2025-12-10 14:13:03', 'GC', 'SELL'): -400,
        ('2025-12-10 14:19:03', 'GC', 'SELL'): 1200,
        ('2025-12-10 14:34:03', 'GC', 'BUY'): -760,
        ('2025-12-10 14:40:03', 'GC', 'SELL'): 2020,
        ('2025-12-10 14:46:03', 'GC', 'BUY'): -250,
        ('2025-12-10 14:52:03', 'GC', 'SELL'): -980,
        ('2025-12-10 14:59:03', 'GC', 'SELL'): -180,
        ('2025-12-10 15:14:03', 'GC', 'SELL'): -380,
        ('2025-12-10 15:20:03', 'GC', 'SELL'): -440,

        # ES minis
        ('2025-12-10 14:02:03', 'ES', 'SELL'): -350,
        ('2025-12-10 14:12:03', 'ES', 'SELL'): -350,
        ('2025-12-10 15:01:03', 'ES', 'SELL'): -350,
        ('2025-12-10 15:09:03', 'ES', 'BUY'): -350,

        # NQ - many had order rejections due to TP > 1000 ticks
    }

    by_symbol = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0, 'peak': 0, 'dd': 0, 'running': 0})

    for trade in sorted(trades, key=lambda x: x['time']):
        key = (trade['time'], trade['symbol'], trade['direction'])
        pnl = pnl_data.get(key, 0)  # 0 if not found (order rejection)

        stats = by_symbol[trade['symbol']]
        stats['trades'] += 1

        if pnl > 0:
            stats['wins'] += 1
        elif pnl < 0:
            stats['losses'] += 1

        stats['pnl'] += pnl
        stats['running'] += pnl

        if stats['running'] > stats['peak']:
            stats['peak'] = stats['running']

        dd = stats['peak'] - stats['running']
        if dd > stats['dd']:
            stats['dd'] = dd

    return by_symbol

def main():
    print("=" * 100)
    print("LIVE TRADING PERFORMANCE ANALYSIS - Dec 10, 2025")
    print("=" * 100)

    gc_trades = parse_gc_trades()
    es_trades = parse_es_trades()
    nq_trades = parse_nq_trades()

    print(f"\nGC TRADES: {len(gc_trades)} signals")
    print(f"ES TRADES: {len(es_trades)} signals")
    print(f"NQ TRADES: {len(nq_trades)} signals (many had order rejections - SL > 1000 ticks)")
    print(f"TOTAL: {len(gc_trades) + len(es_trades) + len(nq_trades)} signals\n")

    all_trades = gc_trades + es_trades + nq_trades
    stats = calculate_stats(all_trades)

    print("=" * 100)
    print("TRADE DETAILS BY SYMBOL")
    print("=" * 100)

    for symbol in ['GC', 'ES', 'NQ']:
        if symbol not in stats:
            continue

        s = stats[symbol]
        print(f"\n{symbol}:")
        print(f"  Signals: {s['trades']}")
        print(f"  Wins: {s['wins']}, Losses: {s['losses']}")
        if s['trades'] > 0:
            print(f"  Win Rate: {s['wins']/s['trades']*100:.1f}%")
        print(f"  Total P&L: ${s['pnl']:.0f}")
        print(f"  Max DD: ${s['dd']:.0f}")

    print("\n" + "=" * 100)
    print("SL/TP DISTANCE ANALYSIS")
    print("=" * 100)

    for symbol in ['GC', 'ES', 'NQ']:
        symbol_trades = [t for t in all_trades if t['symbol'] == symbol]
        if not symbol_trades:
            continue

        print(f"\n{symbol}:")
        print(f"{'Time':<20} {'Dir':<6} {'Entry':>10} {'SL':>10} {'TP':>10} {'SL Ticks':>12} {'TP Ticks':>12}")
        print("-" * 90)

        for t in symbol_trades[:5]:  # First 5 trades
            time_short = t['time'][11:16]
            print(f"{time_short:<20} {t['direction']:<6} {t['entry']:>10.2f} {t['sl']:>10.2f} {t['tp']:>10.2f} {t['sl_ticks']:>12} {t['tp_ticks']:>12}")

    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    print("""
1. NQ Order Rejections: Many NQ trades had SL > 1000 ticks, causing order rejections
   - Example: 15:13 BUY had SL of 964 ticks (REJECTED)
   - Example: 15:14 BUY had SL of 981 ticks (REJECTED)
   - Example: 15:15 BUY had SL of 995 ticks (borderline - might have passed)

2. TP Can Exceed 1000 Ticks: NQ TPs went up to 2985 ticks (no rejection)

3. GC & ES SL Distances: Consistently 40 ticks (tight stops)
   - Got hit repeatedly as market rallied against predictions

4. Model Prediction Quality:
   - GC: 10 signals, mostly SELL into bull rally
   - ES: 4+ signals, mostly SELL into bull rally
   - NQ: Many signals, mixed directions but many rejected orders prevented execution

5. Timing Model Issue: Approved trades even when direction was wrong
""")

if __name__ == "__main__":
    main()
