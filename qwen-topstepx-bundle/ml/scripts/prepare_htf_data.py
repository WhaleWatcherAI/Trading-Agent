#!/usr/bin/env python3
"""
Prepare HTF (Higher Time Frame) candle data for LSTM training.

Merges individual contract files (H25, M25, U25, Z25) into unified datasets
for 1d, 4h, and 1h timeframes.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Paths
CANDLES_DIR = Path("/home/costa/Trading-Agent/tmp/topstepx-candles-2025")
OUTPUT_DIR = Path("/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data_TEST/longterm")

# Contract names
CONTRACTS = ["H25", "M25", "U25", "Z25"]
TIMEFRAMES = ["1d", "4h", "1h"]


def load_candles(filepath: str) -> List[Dict]:
    """Load candles from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('bars', [])


def convert_bar_format(bar: Dict) -> Dict:
    """Convert API bar format to LSTM expected format (o, h, l, c, v)."""
    return {
        'timestamp': bar['timestamp'],
        'o': bar['open'],
        'h': bar['high'],
        'l': bar['low'],
        'c': bar['close'],
        'v': bar['volume']
    }


def merge_contracts(timeframe: str) -> List[Dict]:
    """Merge bars from all contracts for a given timeframe."""
    all_bars = []

    for contract in CONTRACTS:
        filename = f"CON_F_US_ENQ_{contract}-{timeframe}.json"
        filepath = CANDLES_DIR / filename

        if not filepath.exists():
            print(f"  Warning: {filename} not found, skipping...")
            continue

        bars = load_candles(str(filepath))
        print(f"  Loaded {len(bars)} bars from {contract}")
        all_bars.extend(bars)

    # Sort by timestamp
    all_bars.sort(key=lambda x: x['timestamp'])

    # Convert format
    converted_bars = [convert_bar_format(bar) for bar in all_bars]

    return converted_bars


def main():
    print("="*70)
    print("PREPARING HTF CANDLE DATA FOR LSTM TRAINING")
    print("="*70)
    print(f"Candles directory: {CANDLES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each timeframe
    for timeframe in TIMEFRAMES:
        print(f"\nProcessing {timeframe} timeframe...")

        merged_bars = merge_contracts(timeframe)

        if not merged_bars:
            print(f"  No bars found for {timeframe}")
            continue

        # Save merged data
        output_file = OUTPUT_DIR / f"bars_{timeframe}_enq_2025_merged.json"
        output_data = {
            "symbol": "ENQ",
            "timeframe": timeframe,
            "bars": merged_bars
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"  ✓ Saved {len(merged_bars)} bars to {output_file.name}")

        # Print date range
        if merged_bars:
            first_date = merged_bars[0]['timestamp']
            last_date = merged_bars[-1]['timestamp']
            print(f"  Date range: {first_date} to {last_date}")

    print("\n" + "="*70)
    print("✓ DATA PREPARATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
