#!/usr/bin/env python3
"""
Convert Databento MBP-10 DBN files to Topstep-style L2 snapshots.

Goal: keep backtest/live parity so the models can't tell Databento vs TopstepX.

Output format (matches l2_data_collector snapshots):
{
  "t": ISO timestamp,
  "sym": "nq",                # lower-case symbol key
  "bids": [{"price": 123.0, "size": 5}, ...],  # top 10 (price>0,size>0)
  "asks": [...],
  "spread": 1.25,
  "mid": 123.5,
  "bid_depth": 500,
  "ask_depth": 420,
  "imbalance": 0.0865
}
"""
import argparse
import json
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import databento as db
import pandas as pd


def _row_to_snapshot(row, symbol_key: str) -> Dict:
    """Convert one mbp-10 row to a Topstep-style snapshot dict."""
    bids: List[Dict] = []
    asks: List[Dict] = []
    for i in range(10):
        bp = row.get(f"bid_px_{i:02d}")
        bs = row.get(f"bid_sz_{i:02d}")
        ap = row.get(f"ask_px_{i:02d}")
        az = row.get(f"ask_sz_{i:02d}")
        if isinstance(bp, (int, float)) and bp > 0 and isinstance(bs, (int, float)) and bs > 0:
            bids.append({"price": float(bp), "size": int(bs)})
        if isinstance(ap, (int, float)) and ap > 0 and isinstance(az, (int, float)) and az > 0:
            asks.append({"price": float(ap), "size": int(az)})

    if not bids or not asks:
        return {}

    best_bid = bids[0]["price"]
    best_ask = asks[0]["price"]
    if best_bid <= 0 or best_ask <= 0:
        return {}

    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2
    bid_depth = sum(b["size"] for b in bids)
    ask_depth = sum(a["size"] for a in asks)
    denom = max(bid_depth + ask_depth, 1)
    imbalance = round((bid_depth - ask_depth) / denom, 4)

    ts = row["ts_event"]
    if isinstance(ts, (pd.Timestamp, datetime)):
        ts_iso = ts.isoformat()
    else:
        ts_iso = str(ts)

    return {
        "t": ts_iso,
        "sym": symbol_key,
        "bids": bids,
        "asks": asks,
        "spread": spread,
        "mid": mid,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance": imbalance,
    }


def convert_mbp10_zip(
    zip_path: Path,
    output_dir: Path,
    target_symbol: str,
    symbol_key: str,
    chunk_size: int = 100_000,
) -> Dict[str, int]:
    """
    Stream-convert MBP-10 DBN files inside a Databento zip to Topstep-style snapshots.
    Returns {date: snapshot_count}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = defaultdict(int)

    with zipfile.ZipFile(zip_path, "r") as z:
        dbn_files = [n for n in z.namelist() if n.endswith(".dbn.zst")]
        if not dbn_files:
            raise FileNotFoundError("No *.dbn.zst files found in zip")

        for name in dbn_files:
            with tempfile.TemporaryDirectory() as td:
                local_path = Path(td) / name
                with z.open(name) as src, open(local_path, "wb") as dst:
                    dst.write(src.read())

                store = db.DBNStore.from_file(local_path)
                # Iterate in chunks to avoid OOM
                for chunk in store.to_df(schema="mbp-10", map_symbols=True, count=chunk_size):
                    chunk = chunk[chunk["symbol"] == target_symbol]  # keep only outright
                    if chunk.empty:
                        continue
                    for _, row in chunk.iterrows():
                        snap = _row_to_snapshot(row, symbol_key=symbol_key)
                        if not snap:
                            continue
                        date = snap["t"][:10]
                        counts[date] += 1
                        out_path = output_dir / f"l2_snapshots_{symbol_key}_{date}.json"
                        with open(out_path, "a") as f:
                            f.write(json.dumps(snap) + "\n")

    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description="Convert Databento MBP-10 DBN to Topstep-style snapshots.")
    parser.add_argument("--zip", required=True, help="Path to Databento zip (with .dbn.zst files).")
    parser.add_argument("--output-dir", default="ml/data/l2", help="Directory to write snapshot JSONL files.")
    parser.add_argument("--symbol", default="NQH6", help="Target outright symbol (e.g., NQH6).")
    parser.add_argument(
        "--symbol-key",
        default="nq",
        help="Lowercase symbol key to store in snapshots (match l2_data_collector convention).",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Rows per chunk to avoid OOM.")
    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    print(f"Converting {zip_path} -> {output_dir} for symbol {args.symbol} (sym key: {args.symbol_key})")
    counts = convert_mbp10_zip(zip_path, output_dir, args.symbol, args.symbol_key, args.chunk_size)
    total = sum(counts.values())
    print(f"Done. Wrote {total:,} snapshots across {len(counts)} days.")
    for d, c in sorted(counts.items()):
        print(f"  {d}: {c:,} snapshots")


if __name__ == "__main__":
    main()
