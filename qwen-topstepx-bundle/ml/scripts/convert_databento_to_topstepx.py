#!/usr/bin/env python3
"""
Convert Databento data to TopstepX format.
Makes the formats identical so ML models can't tell the difference.
"""
import zipfile
import zstandard as zstd
import json
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, List

def convert_databento_to_topstepx(
    zip_path: str,
    output_path: str,
    start_date: str = None,  # Format: 2025-11-25
    end_date: str = None,
    target_symbol: str = "NQZ5"  # Front month contract we want
):
    """
    Convert Databento format to TopstepX format.

    Databento format:
      {
        "hd": {"ts_event": "2024-12-13T00:00:00.000000000Z", ...},
        "open": "22010.750000000",
        "high": "22011.500000000",
        "low": "22010.750000000",
        "close": "22011.500000000",
        "volume": "6",
        "symbol": "NQZ4"
      }

    TopstepX format:
      {
        "t": "2025-11-27T03:46:48+00:00",
        "o": 25311.75,
        "h": 25311.75,
        "l": 25311.75,
        "c": 25311.75,
        "v": 2
      }
    """
    print(f"Converting Databento data to TopstepX format...")
    print(f"  Target symbol: {target_symbol}")
    if start_date:
        print(f"  Date range: {start_date} to {end_date or 'end'}")

    bars = []
    symbols_seen = set()
    date_counts = defaultdict(int)

    with zipfile.ZipFile(zip_path, 'r') as z:
        compressed_data = z.read('glbx-mdp3-20241213-20251212.ohlcv-1s.json.zst')

        dctx = zstd.ZstdDecompressor()

        buffer = b''
        processed = 0

        with dctx.stream_reader(compressed_data) as reader:
            while True:
                chunk = reader.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break

                buffer += chunk
                lines = buffer.split(b'\n')
                buffer = lines[-1]  # Keep incomplete line

                for line in lines[:-1]:
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                        symbol = record['symbol']
                        symbols_seen.add(symbol)

                        # Filter to target symbol (or closest front month)
                        # NQZ5 = Dec 2025, NQH5 = Mar 2025, etc.
                        if target_symbol and symbol != target_symbol:
                            # For now, skip other symbols
                            # Later we can add logic to roll contracts
                            continue

                        # Parse timestamp
                        ts_str = record['hd']['ts_event']
                        # Convert from "2024-12-13T00:00:00.000000000Z" to "2024-12-13T00:00:00+00:00"
                        ts_clean = ts_str.replace('.000000000Z', '+00:00').replace('Z', '+00:00')

                        date = ts_clean[:10]

                        # Filter by date range if specified
                        if start_date and date < start_date:
                            continue
                        if end_date and date > end_date:
                            continue

                        # Convert to TopstepX format
                        bar = {
                            't': ts_clean,
                            'o': float(record['open']),
                            'h': float(record['high']),
                            'l': float(record['low']),
                            'c': float(record['close']),
                            'v': int(record['volume'])
                        }

                        bars.append(bar)
                        date_counts[date] += 1

                        processed += 1
                        if processed % 100000 == 0:
                            print(f"  Processed {processed:,} records, {len(bars):,} bars kept...", end='\r')

                    except Exception as e:
                        print(f"\nError processing line: {e}")
                        continue

        # Process any remaining buffer
        if buffer.strip():
            try:
                record = json.loads(buffer)
                symbol = record['symbol']
                if not target_symbol or symbol == target_symbol:
                    ts_str = record['hd']['ts_event']
                    ts_clean = ts_str.replace('.000000000Z', '+00:00').replace('Z', '+00:00')
                    date = ts_clean[:10]

                    if (not start_date or date >= start_date) and (not end_date or date <= end_date):
                        bar = {
                            't': ts_clean,
                            'o': float(record['open']),
                            'h': float(record['high']),
                            'l': float(record['low']),
                            'c': float(record['close']),
                            'v': int(record['volume'])
                        }
                        bars.append(bar)
                        date_counts[date] += 1
            except:
                pass

    print(f"\nProcessed {processed:,} total records")
    print(f"Symbols found: {sorted(symbols_seen)}")
    print(f"Kept {len(bars):,} bars for {target_symbol}")

    if not bars:
        print("ERROR: No bars found!")
        return 1

    # Create TopstepX format output
    output = {
        "symbol": target_symbol,
        "contractId": f"CON.F.US.{target_symbol.replace('NQ', 'ENQ')}",  # Databento -> TopstepX contract ID
        "unit": 1,
        "unitNumber": 1,
        "startTime": bars[0]['t'],
        "endTime": bars[-1]['t'],
        "barCount": len(bars),
        "bars": bars
    }

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f)

    print(f"\nSummary:")
    print(f"  Start: {output['startTime']}")
    print(f"  End: {output['endTime']}")
    print(f"  Total bars: {len(bars):,}")
    print(f"\nBars by date:")
    for date in sorted(date_counts.keys())[:10]:
        print(f"    {date}: {date_counts[date]:,} bars")
    if len(date_counts) > 10:
        print(f"    ... ({len(date_counts) - 10} more dates)")

    return 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert Databento to TopstepX format')
    parser.add_argument('--zip', required=True, help='Path to Databento zip file')
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', default='NQZ5', help='Target symbol (default: NQZ5)')

    args = parser.parse_args()

    exit(convert_databento_to_topstepx(
        args.zip,
        args.output,
        args.start_date,
        args.end_date,
        args.symbol
    ))
