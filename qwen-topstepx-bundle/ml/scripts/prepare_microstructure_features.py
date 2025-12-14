#!/usr/bin/env python3
"""
Calculate microstructure features from 1-second candles:
1. Daily Volume Profile from 1-second data (high precision)
2. CVD from 1-second data aggregated to 5-minute candles
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

# Paths
DATA_DIR = Path("/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data")
OUTPUT_DIR = Path("/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data_TEST/longterm")


def load_1s_bars(filepath: str) -> List[Dict]:
    """Load 1-second bars from JSON file."""
    print(f"Loading 1-second bars from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    bars = data.get('bars', [])
    print(f"  Loaded {len(bars)} 1-second bars")
    return bars


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string to datetime."""
    # Handle both formats: with/without microseconds
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except:
        # Fallback for other formats
        return datetime.strptime(ts_str.replace('+00:00', ''), '%Y-%m-%dT%H:%M:%S')


def group_bars_by_day(bars_1s: List[Dict]) -> Dict[str, List[Dict]]:
    """Group 1-second bars by trading day (date)."""
    bars_by_day = defaultdict(list)

    for bar in bars_1s:
        ts = parse_timestamp(bar['t'])
        date_key = ts.strftime('%Y-%m-%d')
        bars_by_day[date_key].append(bar)

    print(f"  Grouped into {len(bars_by_day)} trading days")
    return dict(bars_by_day)


def calculate_daily_vp_from_1s(bars_1s_day: List[Dict], tick_size: float = 0.25) -> Dict:
    """
    Calculate Volume Profile for a single day from 1-second bars.

    This gives much higher precision than using daily/hourly bars.

    Returns:
        Dict with keys: poc, vah, val, volume_distribution
    """
    if len(bars_1s_day) < 2:
        return {'poc': 0, 'vah': 0, 'val': 0, 'total_volume': 0}

    # Find price range for the day
    all_highs = [b['h'] for b in bars_1s_day]
    all_lows = [b['l'] for b in bars_1s_day]
    price_high = max(all_highs)
    price_low = min(all_lows)

    if price_high <= price_low:
        return {'poc': 0, 'vah': 0, 'val': 0, 'total_volume': 0}

    # Create fine-grained price bins (using tick_size)
    num_bins = max(10, int((price_high - price_low) / tick_size))
    num_bins = min(num_bins, 10000)  # Cap at 10k bins for memory

    bins = np.linspace(price_low, price_high, num_bins + 1)
    volume_at_price = np.zeros(num_bins)

    # Distribute volume from each 1-second bar across price levels
    for bar in bars_1s_day:
        bar_vol = bar.get('v', 0)
        bar_range = bar['h'] - bar['l']

        if bar_range > 0:
            # Distribute volume proportionally across the bar's range
            for j in range(num_bins):
                bin_low = bins[j]
                bin_high = bins[j + 1]

                # Check overlap between bar range and bin
                overlap_low = max(bar['l'], bin_low)
                overlap_high = min(bar['h'], bin_high)

                if overlap_high > overlap_low:
                    overlap_pct = (overlap_high - overlap_low) / bar_range
                    volume_at_price[j] += bar_vol * overlap_pct
        else:
            # Single price bar - put all volume at close price
            bin_idx = np.searchsorted(bins, bar['c']) - 1
            if 0 <= bin_idx < num_bins:
                volume_at_price[bin_idx] += bar_vol

    # Find POC (Point of Control) - price level with most volume
    poc_idx = np.argmax(volume_at_price)
    poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

    # Calculate Value Area (70% of volume)
    total_volume = np.sum(volume_at_price)
    value_area_volume = total_volume * 0.70

    # Expand from POC until we reach 70% volume
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

    return {
        'poc': poc,
        'vah': vah,
        'val': val,
        'total_volume': int(total_volume),
        'num_bins': num_bins
    }


def calculate_cvd_from_1s(bars_1s: List[Dict]) -> Tuple[List[float], List[str]]:
    """
    Calculate Cumulative Volume Delta (CVD) from 1-second bars.

    CVD estimates buying vs selling pressure based on where the close is
    within the bar's range.

    Returns:
        Tuple of (cvd_values, timestamps)
    """
    cvd_values = [0.0]
    timestamps = []

    if not bars_1s:
        return [], []

    timestamps.append(bars_1s[0]['t'])

    for i, bar in enumerate(bars_1s):
        bar_range = bar['h'] - bar['l']

        if bar_range > 0:
            # Close position: 0 = low, 0.5 = mid, 1 = high
            close_position = (bar['c'] - bar['l']) / bar_range

            # Delta estimate: volume weighted by close position
            # Close near high = buying pressure (positive delta)
            # Close near low = selling pressure (negative delta)
            delta = bar['v'] * (2 * close_position - 1)
        else:
            # No range - use close vs open
            delta = bar['v'] if bar['c'] >= bar['o'] else -bar['v']

        # Cumulative sum
        if i > 0:
            cvd_values.append(cvd_values[-1] + delta)
            timestamps.append(bar['t'])
        else:
            cvd_values[0] = delta

    return cvd_values, timestamps


def aggregate_cvd_to_5min(cvd_values: List[float], timestamps: List[str]) -> Dict[str, float]:
    """
    Aggregate 1-second CVD to 5-minute candles.

    Returns:
        Dict mapping 5-minute timestamp to CVD value
    """
    if len(cvd_values) != len(timestamps):
        return {}

    cvd_5min = {}

    for i, (cvd, ts_str) in enumerate(zip(cvd_values, timestamps)):
        ts = parse_timestamp(ts_str)

        # Round down to nearest 5-minute interval
        minute = (ts.minute // 5) * 5
        ts_5min = ts.replace(minute=minute, second=0, microsecond=0)
        ts_5min_key = ts_5min.isoformat()

        # Use the last CVD value in each 5-minute window
        cvd_5min[ts_5min_key] = cvd

    return cvd_5min


def main():
    print("="*70)
    print("CALCULATING MICROSTRUCTURE FEATURES FROM 1-SECOND CANDLES")
    print("="*70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load 1-second bars
    bars_1s = load_1s_bars(DATA_DIR / "bars_1s_enq.json")

    if not bars_1s:
        print("Error: No 1-second bars found")
        return

    print(f"\nDate range: {bars_1s[0]['t']} to {bars_1s[-1]['t']}")

    # Group by day
    print("\nGrouping bars by trading day...")
    bars_by_day = group_bars_by_day(bars_1s)

    # Calculate daily volume profile from 1-second bars
    print("\nCalculating daily volume profiles from 1-second data...")
    daily_vp_results = {}

    for date_key in sorted(bars_by_day.keys()):
        day_bars = bars_by_day[date_key]
        vp = calculate_daily_vp_from_1s(day_bars, tick_size=0.25)
        daily_vp_results[date_key] = vp
        print(f"  {date_key}: POC={vp['poc']:.2f}, VAH={vp['vah']:.2f}, VAL={vp['val']:.2f}, Vol={vp['total_volume']:,}")

    # Save daily VP results
    vp_output = OUTPUT_DIR / "daily_vp_from_1s.json"
    with open(vp_output, 'w') as f:
        json.dump(daily_vp_results, f, indent=2)
    print(f"\n✓ Saved daily volume profiles to {vp_output.name}")

    # Calculate CVD from 1-second bars
    print("\nCalculating CVD from 1-second bars...")
    cvd_values, cvd_timestamps = calculate_cvd_from_1s(bars_1s)
    print(f"  Calculated CVD for {len(cvd_values)} 1-second bars")
    print(f"  CVD range: {min(cvd_values):.0f} to {max(cvd_values):.0f}")

    # Aggregate CVD to 5-minute candles
    print("\nAggregating CVD to 5-minute candles...")
    cvd_5min = aggregate_cvd_to_5min(cvd_values, cvd_timestamps)
    print(f"  Created {len(cvd_5min)} 5-minute CVD candles")

    # Save CVD 5-minute results
    cvd_output = OUTPUT_DIR / "cvd_5min_from_1s.json"
    with open(cvd_output, 'w') as f:
        json.dump(cvd_5min, f, indent=2)
    print(f"✓ Saved 5-minute CVD to {cvd_output.name}")

    # Calculate CVD EMA (20-period) on 5-minute data
    print("\nCalculating CVD EMA (20-period)...")
    sorted_times = sorted(cvd_5min.keys())
    cvd_values_sorted = [cvd_5min[t] for t in sorted_times]

    cvd_ema = [cvd_values_sorted[0]]
    alpha = 2.0 / (20 + 1)

    for i in range(1, len(cvd_values_sorted)):
        ema = cvd_values_sorted[i] * alpha + cvd_ema[-1] * (1 - alpha)
        cvd_ema.append(ema)

    cvd_ema_5min = {sorted_times[i]: cvd_ema[i] for i in range(len(sorted_times))}

    # Save CVD EMA
    cvd_ema_output = OUTPUT_DIR / "cvd_ema_5min_from_1s.json"
    with open(cvd_ema_output, 'w') as f:
        json.dump(cvd_ema_5min, f, indent=2)
    print(f"✓ Saved CVD EMA to {cvd_ema_output.name}")

    # Create summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total 1-second bars processed: {len(bars_1s):,}")
    print(f"Trading days: {len(bars_by_day)}")
    print(f"Daily volume profiles: {len(daily_vp_results)}")
    print(f"5-minute CVD candles: {len(cvd_5min)}")
    print(f"\nOutput files:")
    print(f"  - {vp_output.name}")
    print(f"  - {cvd_output.name}")
    print(f"  - {cvd_ema_output.name}")
    print("="*70)


if __name__ == "__main__":
    main()
