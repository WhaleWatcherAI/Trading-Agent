#!/usr/bin/env python3
"""
Fetch historical bars from Alpaca API for stocks/ETFs
Outputs in the same JSON format as TopstepX bars for compatibility with backtests
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests

load_dotenv()

ALPACA_KEY = os.getenv('ALPACA_KEY_ID') or os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET = os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
ALPACA_BASE = 'https://data.alpaca.markets/v2'

def fetch_bars(symbol: str, start_date: str, end_date: str, timeframe: str = '1Min'):
    """
    Fetch historical bars from Alpaca
    timeframe: 1Sec, 1Min, 5Min, 15Min, 1Hour, 1Day
    """
    headers = {
        'APCA-API-KEY-ID': ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
    }

    all_bars = []
    page_token = None
    page = 0

    while True:
        page += 1
        params = {
            'start': start_date,
            'end': end_date,
            'timeframe': timeframe,
            'limit': 10000,
            'adjustment': 'split',  # Adjust for stock splits
            'feed': 'iex',  # Use IEX feed (free tier)
        }
        if page_token:
            params['page_token'] = page_token

        print(f"Fetching page {page}... (total bars so far: {len(all_bars)})")

        resp = requests.get(
            f'{ALPACA_BASE}/stocks/{symbol}/bars',
            headers=headers,
            params=params
        )

        if resp.status_code != 200:
            print(f"Error: {resp.status_code} - {resp.text}")
            break

        data = resp.json()
        bars = data.get('bars', [])

        if not bars:
            print("No more bars returned")
            break

        all_bars.extend(bars)
        print(f"  Got {len(bars)} bars (total: {len(all_bars)})")

        page_token = data.get('next_page_token')
        if not page_token:
            break

    return all_bars


def main():
    parser = argparse.ArgumentParser(description='Fetch Alpaca historical bars')
    parser.add_argument('--symbol', default='QQQ', help='Symbol to fetch (default: QQQ)')
    parser.add_argument('--start', default='2024-12-08', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-12-08', help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='1Min', help='Timeframe: 1Sec, 1Min, 5Min, etc.')
    parser.add_argument('--output', help='Output file path')
    args = parser.parse_args()

    print(f"\n=== Alpaca Bar Fetcher ===")
    print(f"Symbol: {args.symbol}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Timeframe: {args.timeframe}")
    print(f"API Key: {ALPACA_KEY[:10]}..." if ALPACA_KEY else "API Key: NOT SET")
    print()

    if not ALPACA_KEY or not ALPACA_SECRET:
        print("ERROR: Alpaca API credentials not set in environment")
        return

    # Fetch bars
    raw_bars = fetch_bars(args.symbol, args.start, args.end, args.timeframe)

    if not raw_bars:
        print("No bars fetched!")
        return

    print(f"\nTotal bars fetched: {len(raw_bars)}")

    # Convert to our standard format
    bars = []
    for b in raw_bars:
        bars.append({
            't': b['t'],  # ISO timestamp
            'o': b['o'],  # open
            'h': b['h'],  # high
            'l': b['l'],  # low
            'c': b['c'],  # close
            'v': b['v'],  # volume
        })

    # Sort by timestamp
    bars.sort(key=lambda x: x['t'])

    # Output structure matching TopstepX format
    output = {
        'symbol': args.symbol,
        'unit': 'minute' if 'Min' in args.timeframe else args.timeframe.lower(),
        'unitNumber': int(args.timeframe.replace('Min', '').replace('Sec', '').replace('Hour', '').replace('Day', '') or '1'),
        'startTime': args.start,
        'endTime': args.end,
        'barCount': len(bars),
        'bars': bars,
    }

    # Output path
    if args.output:
        output_path = args.output
    else:
        output_path = f'ml/data/bars_{args.timeframe.lower()}_{args.symbol.lower()}_1yr.json'

    # Save
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Print sample
    print("\nSample bars (first 5):")
    for b in bars[:5]:
        print(f"  {b['t']}: O={b['o']} H={b['h']} L={b['l']} C={b['c']} V={b['v']}")

    print("\nSample bars (last 5):")
    for b in bars[-5:]:
        print(f"  {b['t']}: O={b['o']} H={b['h']} L={b['l']} C={b['c']} V={b['v']}")


if __name__ == '__main__':
    main()
