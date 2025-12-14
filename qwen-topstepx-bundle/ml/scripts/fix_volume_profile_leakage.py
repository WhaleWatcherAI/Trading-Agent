#!/usr/bin/env python3
"""
FIX VOLUME PROFILE DATA LEAKAGE

The bug: Volume profile was calculated INCLUDING the current bar, causing look-ahead bias.

Fixed approach:
1. For daily VP: Use only PREVIOUS days (not including current day)
2. For 1s VP: Calculate VP from previous days, not current day
3. Ensure all features use only historical data
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

# Paths
DATA_DIR = Path("/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data_TEST/longterm")


def load_bars(filepath: str) -> List[Dict]:
    """Load bars from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('bars', [])


def calculate_daily_volume_profile_NO_LEAKAGE(bars_daily: List[Dict], lookback: int = 30,
                                               tick_size: float = 0.25) -> List[Dict]:
    """
    Calculate Volume Profile for each daily bar - NO LEAKAGE VERSION.

    Key fix: For bar at index i, use ONLY bars from i-lookback to i-1 (NOT including i).

    Args:
        bars_daily: List of daily bars
        lookback: Number of previous days to use
        tick_size: Price tick size for binning

    Returns:
        List of volume profile dicts (one per bar)
    """
    vp_results = []

    for i in range(len(bars_daily)):
        # FIX: Use bars from [i-lookback, i), NOT including current bar i
        start_idx = max(0, i - lookback)
        window = bars_daily[start_idx:i]  # ✅ EXCLUDES bar i

        if len(window) < 2:
            vp_results.append({'poc': 0, 'vah': 0, 'val': 0})
            continue

        all_highs = [b['h'] for b in window]
        all_lows = [b['l'] for b in window]
        price_high = max(all_highs)
        price_low = min(all_lows)

        if price_high <= price_low:
            vp_results.append({'poc': 0, 'vah': 0, 'val': 0})
            continue

        num_bins = max(10, int((price_high - price_low) / tick_size))
        bins = np.linspace(price_low, price_high, num_bins + 1)
        volume_at_price = np.zeros(num_bins)

        for bar in window:
            bar_vol = bar.get('v', 0)
            bar_range = bar['h'] - bar['l']

            if bar_range > 0:
                for j in range(num_bins):
                    bin_low = bins[j]
                    bin_high = bins[j + 1]
                    overlap_low = max(bar['l'], bin_low)
                    overlap_high = min(bar['h'], bin_high)

                    if overlap_high > overlap_low:
                        overlap_pct = (overlap_high - overlap_low) / bar_range
                        volume_at_price[j] += bar_vol * overlap_pct
            else:
                bin_idx = np.searchsorted(bins, bar['c']) - 1
                if 0 <= bin_idx < num_bins:
                    volume_at_price[bin_idx] += bar_vol

        poc_idx = np.argmax(volume_at_price)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        total_volume = np.sum(volume_at_price)
        value_area_volume = total_volume * 0.70

        vah_idx = poc_idx
        val_idx = poc_idx
        accumulated_volume = volume_at_price[poc_idx]

        while accumulated_volume < value_area_volume:
            vol_above = volume_at_price[vah_idx + 1] if vah_idx + 1 < num_bins else 0
            vol_below = volume_at_price[val_idx - 1] if val_idx > 0 else 0

            if vol_above >= vol_below and vah_idx + 1 < num_bins:
                vah_idx += 1
                accumulated_volume += vol_above
            elif val_idx > 0:
                val_idx -= 1
                accumulated_volume += vol_below
            else:
                break

        vah = (bins[vah_idx] + bins[vah_idx + 1]) / 2
        val = (bins[val_idx] + bins[val_idx + 1]) / 2

        vp_results.append({'poc': poc, 'vah': vah, 'val': val})

    return vp_results


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string to datetime."""
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except:
        return datetime.strptime(ts_str.replace('+00:00', ''), '%Y-%m-%dT%H:%M:%S')


def load_1s_bars(filepath: str) -> List[Dict]:
    """Load 1-second bars from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('bars', [])


def group_bars_by_day(bars_1s: List[Dict]) -> Dict[str, List[Dict]]:
    """Group 1-second bars by trading day."""
    bars_by_day = defaultdict(list)

    for bar in bars_1s:
        ts = parse_timestamp(bar['t'])
        date_key = ts.strftime('%Y-%m-%d')
        bars_by_day[date_key].append(bar)

    return dict(bars_by_day)


def calculate_daily_vp_from_1s_NO_LEAKAGE(bars_by_day: Dict[str, List[Dict]],
                                           lookback_days: int = 5,
                                           tick_size: float = 0.25) -> Dict[str, Dict]:
    """
    Calculate high-resolution daily VP from 1s candles - NO LEAKAGE VERSION.

    Key fix: For each day D, calculate VP using only days BEFORE D (not including D).

    Args:
        bars_by_day: Dict mapping date strings to lists of 1s bars
        lookback_days: Number of previous days to include in VP calculation
        tick_size: Price tick size

    Returns:
        Dict mapping date strings to volume profile dicts
    """
    sorted_dates = sorted(bars_by_day.keys())
    vp_results = {}

    for i, current_date in enumerate(sorted_dates):
        # FIX: Use only PREVIOUS days, not current day
        start_idx = max(0, i - lookback_days)
        lookback_dates = sorted_dates[start_idx:i]  # ✅ EXCLUDES current day

        if len(lookback_dates) == 0:
            vp_results[current_date] = {'poc': 0, 'vah': 0, 'val': 0, 'total_volume': 0}
            continue

        # Collect all 1s bars from lookback period
        all_bars = []
        for date in lookback_dates:
            all_bars.extend(bars_by_day[date])

        if len(all_bars) < 2:
            vp_results[current_date] = {'poc': 0, 'vah': 0, 'val': 0, 'total_volume': 0}
            continue

        # Calculate VP from these historical 1s bars
        all_highs = [b['h'] for b in all_bars]
        all_lows = [b['l'] for b in all_bars]
        price_high = max(all_highs)
        price_low = min(all_lows)

        if price_high <= price_low:
            vp_results[current_date] = {'poc': 0, 'vah': 0, 'val': 0, 'total_volume': 0}
            continue

        num_bins = max(10, int((price_high - price_low) / tick_size))
        num_bins = min(num_bins, 10000)
        bins = np.linspace(price_low, price_high, num_bins + 1)
        volume_at_price = np.zeros(num_bins)

        for bar in all_bars:
            bar_vol = bar.get('v', 0)
            bar_range = bar['h'] - bar['l']

            if bar_range > 0:
                for j in range(num_bins):
                    bin_low = bins[j]
                    bin_high = bins[j + 1]
                    overlap_low = max(bar['l'], bin_low)
                    overlap_high = min(bar['h'], bin_high)

                    if overlap_high > overlap_low:
                        overlap_pct = (overlap_high - overlap_low) / bar_range
                        volume_at_price[j] += bar_vol * overlap_pct
            else:
                bin_idx = np.searchsorted(bins, bar['c']) - 1
                if 0 <= bin_idx < num_bins:
                    volume_at_price[bin_idx] += bar_vol

        poc_idx = np.argmax(volume_at_price)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        total_volume = np.sum(volume_at_price)
        value_area_volume = total_volume * 0.70

        vah_idx = poc_idx
        val_idx = poc_idx
        accumulated_volume = volume_at_price[poc_idx]

        while accumulated_volume < value_area_volume:
            vol_above = volume_at_price[vah_idx + 1] if vah_idx + 1 < num_bins else 0
            vol_below = volume_at_price[val_idx - 1] if val_idx > 0 else 0

            if vol_above >= vol_below and vah_idx + 1 < num_bins:
                vah_idx += 1
                accumulated_volume += vol_above
            elif val_idx > 0:
                val_idx -= 1
                accumulated_volume += vol_below
            else:
                break

        vah = (bins[vah_idx] + bins[vah_idx + 1]) / 2
        val = (bins[val_idx] + bins[val_idx + 1]) / 2

        vp_results[current_date] = {
            'poc': poc,
            'vah': vah,
            'val': val,
            'total_volume': int(total_volume)
        }

    return vp_results


def main():
    print("="*70)
    print("FIXING VOLUME PROFILE DATA LEAKAGE")
    print("="*70)
    print("Issue: VP was calculated INCLUDING current bar (look-ahead bias)")
    print("Fix: Use only PREVIOUS bars/days for VP calculation")
    print("="*70)

    # Fix daily VP from daily bars
    print("\n[1/2] Recalculating daily VP from daily bars (NO LEAKAGE)...")
    bars_daily = load_bars(DATA_DIR / 'bars_1d_enq_2025_merged.json')
    print(f"  Loaded {len(bars_daily)} daily bars")

    vp_daily_fixed = calculate_daily_volume_profile_NO_LEAKAGE(bars_daily, lookback=30)
    print(f"  ✓ Calculated {len(vp_daily_fixed)} VP values (no leakage)")

    # Fix high-res VP from 1s candles
    print("\n[2/2] Recalculating high-res VP from 1s candles (NO LEAKAGE)...")
    bars_1s_path = "/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data/bars_1s_enq.json"
    bars_1s = load_1s_bars(bars_1s_path)
    print(f"  Loaded {len(bars_1s)} 1-second bars")

    bars_by_day = group_bars_by_day(bars_1s)
    print(f"  Grouped into {len(bars_by_day)} trading days")

    vp_1s_fixed = calculate_daily_vp_from_1s_NO_LEAKAGE(bars_by_day, lookback_days=5)
    print(f"  ✓ Calculated {len(vp_1s_fixed)} high-res VP values (no leakage)")

    # Save fixed VP data
    output_vp_1s = DATA_DIR / "daily_vp_from_1s_FIXED.json"
    with open(output_vp_1s, 'w') as f:
        json.dump(vp_1s_fixed, f, indent=2)
    print(f"\n✓ Saved fixed high-res VP to: {output_vp_1s.name}")

    # Create verification report
    print("\n" + "="*70)
    print("VERIFICATION: Comparing original vs fixed VP")
    print("="*70)

    # Load original (leaky) VP
    with open(DATA_DIR / "daily_vp_from_1s.json", 'r') as f:
        vp_1s_original = json.load(f)

    # Compare a few examples
    print("\nSample comparison (first 5 days):")
    for i, date in enumerate(sorted(vp_1s_fixed.keys())[:5]):
        orig = vp_1s_original.get(date, {})
        fixed = vp_1s_fixed[date]
        print(f"\n  {date}:")
        print(f"    Original POC: {orig.get('poc', 0):.2f} (INCLUDES current day - LEAKAGE)")
        print(f"    Fixed POC:    {fixed['poc']:.2f} (ONLY previous days - NO LEAKAGE)")
        if orig.get('poc', 0) != 0:
            diff_pct = abs(orig.get('poc', 0) - fixed['poc']) / orig.get('poc', 0) * 100
            print(f"    Difference: {diff_pct:.2f}%")

    print("\n" + "="*70)
    print("✓ FIXED DATA SAVED - Ready to retrain with no leakage!")
    print("="*70)


if __name__ == "__main__":
    main()
