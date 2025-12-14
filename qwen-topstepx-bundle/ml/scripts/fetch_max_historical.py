#!/usr/bin/env python3
"""
Fetch maximum historical data from TopstepX API.
Attempts to get 30 days of 1-second bars for NQ.
"""
import json
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict
import os

# Configuration
TOPSTEPX_BASE_URL = "https://api.topstepx.com"
TOPSTEPX_CONTRACT_ID = os.getenv("TOPSTEPX_CONTRACT_ID", "CON.F.US.ENQ.Z25")
TOPSTEPX_API_KEY = os.getenv("TOPSTEPX_API_KEY")
OUTPUT_FILE = "/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data/bars_1s_nq.json"

# Try to fetch 30 days back
DAYS_TO_FETCH = 30

def get_auth_headers():
    return {
        "Authorization": f"Bearer {TOPSTEPX_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def fetch_bars(start_time: datetime, end_time: datetime, chunk_minutes: int = 30) -> List[Dict]:
    """Fetch bars from TopstepX API in chunks."""
    all_bars = []
    current_start = start_time

    total_minutes = (end_time - start_time).total_seconds() / 60
    total_chunks = int(total_minutes / chunk_minutes) + 1
    chunk_num = 0

    print(f"\nFetching {total_minutes:.0f} minutes ({total_minutes/60/24:.1f} days) in {total_chunks} chunks...")

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(minutes=chunk_minutes), end_time)
        chunk_num += 1

        payload = {
            "contractId": TOPSTEPX_CONTRACT_ID,
            "live": False,
            "startTime": current_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit": 1,  # Seconds
            "unitNumber": 1,
            "limit": 20000,
            "includePartialBar": True,
        }

        try:
            response = requests.post(
                f"{TOPSTEPX_BASE_URL}/api/History/retrieveBars",
                json=payload,
                headers=get_auth_headers(),
                timeout=60
            )

            data = response.json()
            if data.get('success'):
                bars = data.get('bars') or []
                for bar in bars:
                    all_bars.append({
                        't': bar.get('t') or bar.get('timestamp'),
                        'o': float(bar.get('o') or bar.get('open') or 0),
                        'h': float(bar.get('h') or bar.get('high') or 0),
                        'l': float(bar.get('l') or bar.get('low') or 0),
                        'c': float(bar.get('c') or bar.get('close') or 0),
                        'v': int(bar.get('v') or bar.get('volume') or 0),
                    })
                print(f"  Chunk {chunk_num}/{total_chunks}: {len(bars)} bars ({current_start.strftime('%Y-%m-%d %H:%M')})")
            else:
                print(f"  Chunk {chunk_num} failed: {data.get('errorMessage', 'unknown')}")

        except Exception as e:
            print(f"  Chunk {chunk_num} error: {e}")

        current_start = chunk_end
        time.sleep(0.5)  # Rate limiting

    return all_bars

def main():
    if not TOPSTEPX_API_KEY:
        print("ERROR: TOPSTEPX_API_KEY environment variable not set")
        return 1

    print("=" * 70)
    print("FETCHING MAXIMUM HISTORICAL DATA FROM TOPSTEPX")
    print("=" * 70)
    print(f"Contract: {TOPSTEPX_CONTRACT_ID}")
    print(f"Days to fetch: {DAYS_TO_FETCH}")
    print(f"Output: {OUTPUT_FILE}")

    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=DAYS_TO_FETCH)

    print(f"\nTime range:")
    print(f"  Start: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  End:   {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Fetch bars
    bars = fetch_bars(start_time, end_time)

    if not bars:
        print("\nERROR: No bars fetched!")
        return 1

    print(f"\nTotal bars fetched: {len(bars):,}")

    # Prepare output format
    output = {
        "symbol": "NQZ5",
        "contractId": TOPSTEPX_CONTRACT_ID,
        "unit": 1,
        "unitNumber": 1,
        "startTime": bars[0]['t'] if bars else start_time.isoformat() + 'Z',
        "endTime": bars[-1]['t'] if bars else end_time.isoformat() + 'Z',
        "barCount": len(bars),
        "bars": bars
    }

    # Save to file
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f)

    # Summary by date
    from collections import defaultdict
    dates = defaultdict(int)
    for bar in bars:
        date = bar['t'][:10]
        dates[date] += 1

    print(f"\nBars by date:")
    for date in sorted(dates.keys()):
        print(f"  {date}: {dates[date]:,} bars")

    print(f"\nTotal trading days: {len(dates)}")
    print("Done!")
    return 0

if __name__ == "__main__":
    exit(main())
