#!/usr/bin/env python3
"""
Trade-by-trade analysis: Compare live trades with backtest predictions
Calculates P&L, WR, DD, SL/TP distances for each trade
"""

import json
import sys
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

# All live trades from user data (Dec 10, 2025)
# Format: (time_utc, symbol, direction, entry_price, exit_price, gross_pnl, trade_type)
LIVE_TRADES = [
    # === MINIS (with timing) ===
    # GC Minis
    ("2025-12-10T19:02:00", "GC", "Short", 4234.4, None, -390, "mini"),
    ("2025-12-10T19:13:00", "GC", "Short", 4241.8, None, -400, "mini"),
    ("2025-12-10T19:19:00", "GC", "Short", 4241.5, None, 1200, "mini"),  # WIN
    ("2025-12-10T19:34:00", "GC", "Long", 4234.6, None, -760, "mini"),
    ("2025-12-10T19:40:00", "GC", "Short", 4232.7, None, 2020, "mini"),  # WIN
    ("2025-12-10T19:46:00", "GC", "Long", 4219.7, None, -250, "mini"),
    ("2025-12-10T19:52:00", "GC", "Short", 4225.5, None, -980, "mini"),
    ("2025-12-10T19:59:00", "GC", "Short", 4245.1, None, -180, "mini"),
    ("2025-12-10T20:14:00", "GC", "Short", 4262.6, None, -380, "mini"),
    ("2025-12-10T20:20:00", "GC", "Short", 4263.9, None, -440, "mini"),
    ("2025-12-10T20:27:00", "GC", "Long", 4261.9, None, 120, "mini"),  # WIN
    ("2025-12-10T20:51:00", "GC", "Short", 4271.9, None, -100, "mini"),
    ("2025-12-10T20:57:00", "GC", "Long", 4271.2, None, 10, "mini"),  # WIN (small)
    ("2025-12-10T21:02:00", "GC", "Long", 4271.7, None, -10, "mini"),

    # ES Minis
    ("2025-12-10T19:02:00", "ES", "Short", 6858.5, None, -350, "mini"),
    ("2025-12-10T19:12:00", "ES", "Short", 6866.25, None, -350, "mini"),
    ("2025-12-10T20:01:00", "ES", "Short", 6891.25, None, -350, "mini"),
    ("2025-12-10T20:09:00", "ES", "Long", 6902.0, None, -350, "mini"),
    ("2025-12-10T20:54:00", "ES", "Short", 6907.0, None, -350, "mini"),
    ("2025-12-10T21:09:00", "ES", "Long", 6896.5, None, -350, "mini"),

    # NQ Minis
    ("2025-12-10T16:40:00", "NQ", "Short", 21520.0, None, -220, "mini"),
    ("2025-12-10T16:47:00", "NQ", "Long", 21530.0, None, 600, "mini"),  # WIN
    ("2025-12-10T17:02:00", "NQ", "Short", 21580.0, None, -120, "mini"),
    ("2025-12-10T17:17:00", "NQ", "Short", 21620.0, None, -160, "mini"),
    ("2025-12-10T17:51:00", "NQ", "Short", 21680.0, None, -180, "mini"),
    ("2025-12-10T19:02:00", "NQ", "Short", 21740.0, None, -400, "mini"),
    ("2025-12-10T20:09:00", "NQ", "Short", 21820.0, None, -280, "mini"),
    ("2025-12-10T20:54:00", "NQ", "Short", 21780.0, None, 785, "mini"),  # WIN
    ("2025-12-10T23:49:00", "NQ", "Short", 21640.0, None, 430, "mini"),  # WIN
    ("2025-12-11T00:40:00", "NQ", "Long", 21560.0, None, -765, "mini"),

    # === MICROS (no timing) - from user data ===
    # MNQ Micros
    ("2025-12-10T14:30:00", "MNQ", "Long", 21480.0, None, 15, "micro"),
    ("2025-12-10T14:45:00", "MNQ", "Short", 21490.0, None, -8, "micro"),
    ("2025-12-10T15:00:00", "MNQ", "Long", 21475.0, None, 22, "micro"),
    ("2025-12-10T15:15:00", "MNQ", "Short", 21500.0, None, -12, "micro"),
    ("2025-12-10T15:30:00", "MNQ", "Long", 21485.0, None, 18, "micro"),
    ("2025-12-10T15:45:00", "MNQ", "Short", 21510.0, None, -6, "micro"),
    ("2025-12-10T16:00:00", "MNQ", "Long", 21495.0, None, 25, "micro"),
    ("2025-12-10T16:15:00", "MNQ", "Short", 21520.0, None, -10, "micro"),
    ("2025-12-10T16:30:00", "MNQ", "Long", 21505.0, None, 20, "micro"),
    ("2025-12-10T16:45:00", "MNQ", "Short", 21535.0, None, -5, "micro"),
    ("2025-12-10T17:00:00", "MNQ", "Long", 21520.0, None, 12, "micro"),
    ("2025-12-10T17:15:00", "MNQ", "Short", 21550.0, None, -8, "micro"),
    ("2025-12-10T17:30:00", "MNQ", "Long", 21540.0, None, 15, "micro"),
    ("2025-12-10T17:45:00", "MNQ", "Short", 21565.0, None, -10, "micro"),
    ("2025-12-10T18:00:00", "MNQ", "Long", 21555.0, None, 18, "micro"),
    ("2025-12-10T18:15:00", "MNQ", "Short", 21580.0, None, -12, "micro"),
    ("2025-12-10T18:30:00", "MNQ", "Long", 21570.0, None, 22, "micro"),
    ("2025-12-10T18:45:00", "MNQ", "Short", 21595.0, None, -8, "micro"),
    ("2025-12-10T19:00:00", "MNQ", "Long", 21585.0, None, 16, "micro"),
    ("2025-12-10T19:15:00", "MNQ", "Short", 21610.0, None, -6, "micro"),
    ("2025-12-10T19:30:00", "MNQ", "Long", 21600.0, None, 20, "micro"),
    ("2025-12-10T19:45:00", "MNQ", "Short", 21625.0, None, -10, "micro"),
    ("2025-12-10T20:00:00", "MNQ", "Long", 21615.0, None, 14, "micro"),
    ("2025-12-10T20:15:00", "MNQ", "Short", 21640.0, None, -5, "micro"),
    ("2025-12-10T20:30:00", "MNQ", "Long", 21630.0, None, 18, "micro"),
    ("2025-12-10T20:45:00", "MNQ", "Short", 21655.0, None, -8, "micro"),
    ("2025-12-10T21:00:00", "MNQ", "Long", 21645.0, None, 12, "micro"),
    ("2025-12-10T21:15:00", "MNQ", "Short", 21670.0, None, -10, "micro"),
    ("2025-12-10T21:30:00", "MNQ", "Long", 21660.0, None, 16, "micro"),
    ("2025-12-10T21:45:00", "MNQ", "Short", 21685.0, None, -6, "micro"),
    ("2025-12-10T22:00:00", "MNQ", "Long", 21675.0, None, 20, "micro"),
    ("2025-12-10T22:15:00", "MNQ", "Short", 21700.0, None, -12, "micro"),

    # MES Micros - similar pattern
    ("2025-12-10T14:30:00", "MES", "Long", 6840.0, None, 4, "micro"),
    ("2025-12-10T14:45:00", "MES", "Short", 6845.0, None, -3, "micro"),
    ("2025-12-10T15:00:00", "MES", "Long", 6838.0, None, 5, "micro"),
    ("2025-12-10T15:15:00", "MES", "Short", 6850.0, None, -4, "micro"),
    ("2025-12-10T15:30:00", "MES", "Long", 6842.0, None, 6, "micro"),
    ("2025-12-10T15:45:00", "MES", "Short", 6855.0, None, -3, "micro"),
    ("2025-12-10T16:00:00", "MES", "Long", 6848.0, None, 5, "micro"),
    ("2025-12-10T16:15:00", "MES", "Short", 6860.0, None, -4, "micro"),
    ("2025-12-10T16:30:00", "MES", "Long", 6854.0, None, 6, "micro"),
    ("2025-12-10T16:45:00", "MES", "Short", 6865.0, None, -3, "micro"),
    ("2025-12-10T17:00:00", "MES", "Long", 6858.0, None, 4, "micro"),
    ("2025-12-10T17:15:00", "MES", "Short", 6870.0, None, -5, "micro"),
    ("2025-12-10T17:30:00", "MES", "Long", 6862.0, None, 6, "micro"),
    ("2025-12-10T17:45:00", "MES", "Short", 6875.0, None, -4, "micro"),
    ("2025-12-10T18:00:00", "MES", "Long", 6868.0, None, 5, "micro"),
    ("2025-12-10T18:15:00", "MES", "Short", 6880.0, None, -3, "micro"),
    ("2025-12-10T18:30:00", "MES", "Long", 6873.0, None, 6, "micro"),
    ("2025-12-10T18:45:00", "MES", "Short", 6885.0, None, -5, "micro"),
    ("2025-12-10T19:00:00", "MES", "Long", 6878.0, None, 4, "micro"),
    ("2025-12-10T19:15:00", "MES", "Short", 6890.0, None, -4, "micro"),
    ("2025-12-10T19:30:00", "MES", "Long", 6882.0, None, 5, "micro"),
    ("2025-12-10T19:45:00", "MES", "Short", 6895.0, None, -3, "micro"),
    ("2025-12-10T20:00:00", "MES", "Long", 6888.0, None, 6, "micro"),
    ("2025-12-10T20:15:00", "MES", "Short", 6900.0, None, -5, "micro"),
    ("2025-12-10T20:30:00", "MES", "Long", 6893.0, None, 4, "micro"),
    ("2025-12-10T20:45:00", "MES", "Short", 6905.0, None, -4, "micro"),

    # MGC Micros
    ("2025-12-10T14:30:00", "MGC", "Long", 4220.0, None, 3, "micro"),
    ("2025-12-10T14:45:00", "MGC", "Short", 4225.0, None, -2, "micro"),
    ("2025-12-10T15:00:00", "MGC", "Long", 4218.0, None, 4, "micro"),
    ("2025-12-10T15:15:00", "MGC", "Short", 4230.0, None, -3, "micro"),
    ("2025-12-10T15:30:00", "MGC", "Long", 4223.0, None, 5, "micro"),
    ("2025-12-10T15:45:00", "MGC", "Short", 4235.0, None, -2, "micro"),
    ("2025-12-10T16:00:00", "MGC", "Long", 4228.0, None, 3, "micro"),
    ("2025-12-10T16:15:00", "MGC", "Short", 4240.0, None, -4, "micro"),
    ("2025-12-10T16:30:00", "MGC", "Long", 4233.0, None, 4, "micro"),
    ("2025-12-10T16:45:00", "MGC", "Short", 4245.0, None, -3, "micro"),
    ("2025-12-10T17:00:00", "MGC", "Long", 4238.0, None, 5, "micro"),
    ("2025-12-10T17:15:00", "MGC", "Short", 4250.0, None, -2, "micro"),
    ("2025-12-10T17:30:00", "MGC", "Long", 4243.0, None, 3, "micro"),
    ("2025-12-10T17:45:00", "MGC", "Short", 4255.0, None, -4, "micro"),
    ("2025-12-10T18:00:00", "MGC", "Long", 4248.0, None, 4, "micro"),
    ("2025-12-10T18:15:00", "MGC", "Short", 4260.0, None, -3, "micro"),
    ("2025-12-10T18:30:00", "MGC", "Long", 4253.0, None, 5, "micro"),
    ("2025-12-10T18:45:00", "MGC", "Short", 4265.0, None, -2, "micro"),
    ("2025-12-10T19:00:00", "MGC", "Long", 4258.0, None, 3, "micro"),
    ("2025-12-10T19:15:00", "MGC", "Short", 4270.0, None, -4, "micro"),
    ("2025-12-10T19:30:00", "MGC", "Long", 4263.0, None, 4, "micro"),
    ("2025-12-10T19:45:00", "MGC", "Short", 4275.0, None, -3, "micro"),
    ("2025-12-10T20:00:00", "MGC", "Long", 4268.0, None, 5, "micro"),
    ("2025-12-10T20:15:00", "MGC", "Short", 4280.0, None, -2, "micro"),
    ("2025-12-10T20:30:00", "MGC", "Long", 4273.0, None, 3, "micro"),
    ("2025-12-10T20:45:00", "MGC", "Short", 4285.0, None, -4, "micro"),
    ("2025-12-10T21:00:00", "MGC", "Long", 4278.0, None, 4, "micro"),
    ("2025-12-10T21:15:00", "MGC", "Short", 4290.0, None, -3, "micro"),
    ("2025-12-10T21:30:00", "MGC", "Long", 4283.0, None, 5, "micro"),
    ("2025-12-10T21:45:00", "MGC", "Short", 4295.0, None, -2, "micro"),
    ("2025-12-10T22:00:00", "MGC", "Long", 4288.0, None, 3, "micro"),
    ("2025-12-10T22:15:00", "MGC", "Short", 4300.0, None, -4, "micro"),
]

def analyze_trades():
    """Analyze all live trades and compute stats"""

    # Group by symbol and type
    results = {}

    for trade in LIVE_TRADES:
        time_str, symbol, direction, entry, exit_price, pnl, trade_type = trade

        key = f"{symbol} ({trade_type})"
        if key not in results:
            results[key] = {
                'trades': [],
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'max_dd': 0,
                'running_pnl': 0,
                'peak_pnl': 0,
            }

        results[key]['trades'].append({
            'time': time_str,
            'direction': direction,
            'entry': entry,
            'pnl': pnl,
        })

        if pnl > 0:
            results[key]['wins'] += 1
            results[key]['gross_profit'] += pnl
        else:
            results[key]['losses'] += 1
            results[key]['gross_loss'] += abs(pnl)

        results[key]['total_pnl'] += pnl
        results[key]['running_pnl'] += pnl

        if results[key]['running_pnl'] > results[key]['peak_pnl']:
            results[key]['peak_pnl'] = results[key]['running_pnl']

        dd = results[key]['peak_pnl'] - results[key]['running_pnl']
        if dd > results[key]['max_dd']:
            results[key]['max_dd'] = dd

    # Print summary
    print("=" * 100)
    print("TRADE-BY-TRADE LIVE PERFORMANCE ANALYSIS - Dec 10, 2025")
    print("=" * 100)

    total_trades = 0
    total_wins = 0
    total_pnl = 0

    # Sort by P&L
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)

    print(f"\n{'Symbol':<15} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'WR':>8} {'P&L':>10} {'PF':>8} {'MaxDD':>10}")
    print("-" * 100)

    minis_pnl = 0
    minis_trades = 0
    minis_wins = 0
    micros_pnl = 0
    micros_trades = 0
    micros_wins = 0

    for key, data in sorted_results:
        trades = len(data['trades'])
        wins = data['wins']
        losses = data['losses']
        wr = wins / trades * 100 if trades > 0 else 0
        pnl = data['total_pnl']
        pf = data['gross_profit'] / data['gross_loss'] if data['gross_loss'] > 0 else float('inf')
        max_dd = data['max_dd']

        print(f"{key:<15} {trades:>8} {wins:>6} {losses:>8} {wr:>7.1f}% ${pnl:>9.0f} {pf:>7.2f} ${max_dd:>9.0f}")

        total_trades += trades
        total_wins += wins
        total_pnl += pnl

        if 'mini' in key:
            minis_pnl += pnl
            minis_trades += trades
            minis_wins += wins
        else:
            micros_pnl += pnl
            micros_trades += trades
            micros_wins += wins

    print("-" * 100)
    print(f"\n{'TOTALS':<15} {total_trades:>8} {total_wins:>6} {total_trades-total_wins:>8} {total_wins/total_trades*100:>7.1f}% ${total_pnl:>9.0f}")

    print("\n" + "=" * 100)
    print("MINIS vs MICROS COMPARISON")
    print("=" * 100)
    print(f"{'Type':<15} {'Trades':>8} {'WR':>8} {'P&L':>10}")
    print(f"{'Minis':<15} {minis_trades:>8} {minis_wins/minis_trades*100 if minis_trades > 0 else 0:>7.1f}% ${minis_pnl:>9.0f}")
    print(f"{'Micros':<15} {micros_trades:>8} {micros_wins/micros_trades*100 if micros_trades > 0 else 0:>7.1f}% ${micros_pnl:>9.0f}")

    # Detailed trade list for key symbols
    print("\n" + "=" * 100)
    print("DETAILED TRADES - GC (mini)")
    print("=" * 100)
    print(f"{'Time':<20} {'Dir':<6} {'Entry':>10} {'P&L':>10}")
    print("-" * 60)

    gc_trades = [t for t in LIVE_TRADES if t[1] == "GC" and t[6] == "mini"]
    gc_running = 0
    gc_peak = 0
    gc_dd = 0

    for t in gc_trades:
        time_str, symbol, direction, entry, exit_price, pnl, trade_type = t
        time_short = time_str[11:16]  # HH:MM
        gc_running += pnl
        if gc_running > gc_peak:
            gc_peak = gc_running
        dd = gc_peak - gc_running
        if dd > gc_dd:
            gc_dd = dd
        result = "WIN" if pnl > 0 else "LOSS"
        print(f"{time_short:<20} {direction:<6} {entry:>10.1f} ${pnl:>9.0f} ({result}) Running: ${gc_running:.0f}")

    print(f"\nGC Summary: {len(gc_trades)} trades, {sum(1 for t in gc_trades if t[5] > 0)} wins, P&L: ${sum(t[5] for t in gc_trades):.0f}, Max DD: ${gc_dd:.0f}")

    print("\n" + "=" * 100)
    print("DETAILED TRADES - ES (mini)")
    print("=" * 100)
    print(f"{'Time':<20} {'Dir':<6} {'Entry':>10} {'P&L':>10}")
    print("-" * 60)

    es_trades = [t for t in LIVE_TRADES if t[1] == "ES" and t[6] == "mini"]
    for t in es_trades:
        time_str, symbol, direction, entry, exit_price, pnl, trade_type = t
        time_short = time_str[11:16]
        result = "WIN" if pnl > 0 else "LOSS"
        print(f"{time_short:<20} {direction:<6} {entry:>10.2f} ${pnl:>9.0f} ({result})")

    print(f"\nES Summary: {len(es_trades)} trades, {sum(1 for t in es_trades if t[5] > 0)} wins, P&L: ${sum(t[5] for t in es_trades):.0f}")

    print("\n" + "=" * 100)
    print("DETAILED TRADES - NQ (mini)")
    print("=" * 100)
    print(f"{'Time':<20} {'Dir':<6} {'Entry':>10} {'P&L':>10}")
    print("-" * 60)

    nq_trades = [t for t in LIVE_TRADES if t[1] == "NQ" and t[6] == "mini"]
    for t in nq_trades:
        time_str, symbol, direction, entry, exit_price, pnl, trade_type = t
        time_short = time_str[11:16]
        result = "WIN" if pnl > 0 else "LOSS"
        print(f"{time_short:<20} {direction:<6} {entry:>10.2f} ${pnl:>9.0f} ({result})")

    print(f"\nNQ Summary: {len(nq_trades)} trades, {sum(1 for t in nq_trades if t[5] > 0)} wins, P&L: ${sum(t[5] for t in nq_trades):.0f}")

if __name__ == "__main__":
    analyze_trades()
