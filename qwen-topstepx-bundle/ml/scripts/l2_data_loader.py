#!/usr/bin/env python3
"""
L2 Order Book Data Loader

Loads Databento MBP-10 snapshots (converted to TopstepX format) and
syncs them with 1-second OHLCV bars for CNN input.
"""

import json
import bisect
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict


class L2DataLoader:
    """
    Load and sync L2 snapshots with 1-second bars.
    """

    def __init__(self, l2_data_dir: str = "ml/data/l2"):
        """
        Args:
            l2_data_dir: Directory containing l2_snapshots_*.json files
        """
        self.l2_data_dir = Path(l2_data_dir)
        self.snapshots_by_date: Dict[str, List[Dict]] = {}
        self.timestamps_by_date: Dict[str, List[datetime]] = {}

    def load_date(self, date: str, symbol: str = "nq") -> int:
        """
        Load L2 snapshots for a specific date.

        Args:
            date: Date string "YYYY-MM-DD"
            symbol: Symbol key (e.g., "nq")

        Returns:
            Number of snapshots loaded
        """
        if date in self.snapshots_by_date:
            return len(self.snapshots_by_date[date])

        file_path = self.l2_data_dir / f"l2_snapshots_{symbol}_{date}.json"
        if not file_path.exists():
            print(f"[L2] No snapshots for {date}")
            return 0

        snapshots = []
        timestamps = []

        print(f"[L2] Loading snapshots for {date}...", end="", flush=True)

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    snap = json.loads(line)
                    # Parse timestamp
                    ts_str = snap.get("t", "")
                    if not ts_str:
                        continue

                    # Handle nanosecond precision: Python only supports microseconds
                    # Truncate "2025-12-08T01:41:16.263201751+00:00" to "2025-12-08T01:41:16.263201+00:00"
                    if '.' in ts_str:
                        parts = ts_str.split('.')
                        if len(parts) == 2:
                            fractional = parts[1]
                            # Find timezone offset
                            tz_start = fractional.find('+') if '+' in fractional else fractional.find('-')
                            if tz_start > 0:
                                frac_part = fractional[:tz_start]
                                tz_part = fractional[tz_start:]
                                # Truncate to 6 digits (microseconds)
                                frac_part = frac_part[:6]
                                ts_str = f"{parts[0]}.{frac_part}{tz_part}"

                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

                    snapshots.append(snap)
                    timestamps.append(ts)

                    if line_num % 100000 == 0:
                        print(f"\r[L2] Loading {date}: {line_num:,} snapshots...", end="", flush=True)

                except Exception as e:
                    if line_num % 100000 == 0:
                        print(f"\n[L2] Warning: Failed to parse line {line_num}: {e}")
                    continue

        self.snapshots_by_date[date] = snapshots
        self.timestamps_by_date[date] = timestamps

        print(f"\r[L2] Loaded {len(snapshots):,} snapshots for {date}     ")
        return len(snapshots)

    def get_snapshot_at_time(
        self,
        date: str,
        target_time: datetime,
        max_age_seconds: int = 1
    ) -> Optional[Dict]:
        """
        Get L2 snapshot closest to target time (within max_age).

        Args:
            date: Date string "YYYY-MM-DD"
            target_time: Target datetime
            max_age_seconds: Maximum age of snapshot to accept

        Returns:
            Snapshot dict or None if none within max_age
        """
        if date not in self.snapshots_by_date:
            return None

        timestamps = self.timestamps_by_date[date]
        snapshots = self.snapshots_by_date[date]

        if not timestamps:
            return None

        # Binary search for closest timestamp
        idx = bisect.bisect_left(timestamps, target_time)

        # Check both idx and idx-1 for closest
        candidates = []
        if idx < len(timestamps):
            candidates.append((idx, abs((timestamps[idx] - target_time).total_seconds())))
        if idx > 0:
            candidates.append((idx-1, abs((timestamps[idx-1] - target_time).total_seconds())))

        if not candidates:
            return None

        # Get closest
        best_idx, age = min(candidates, key=lambda x: x[1])

        if age > max_age_seconds:
            return None

        return snapshots[best_idx]

    def get_snapshot_sequence(
        self,
        date: str,
        end_time: datetime,
        seq_len: int = 30,
        interval_seconds: float = 1.0
    ) -> List[Dict]:
        """
        Get sequence of L2 snapshots leading up to end_time.

        Args:
            date: Date string "YYYY-MM-DD"
            end_time: End datetime
            seq_len: Number of snapshots to retrieve
            interval_seconds: Time between snapshots

        Returns:
            List of snapshots (may be less than seq_len if not enough data)
        """
        if date not in self.snapshots_by_date:
            return []

        sequence = []
        for i in range(seq_len):
            # Go backwards from end_time
            offset = (seq_len - 1 - i) * interval_seconds
            target_time = end_time - timedelta(seconds=offset)

            snap = self.get_snapshot_at_time(date, target_time, max_age_seconds=1)
            if snap:
                sequence.append(snap)
            elif sequence:
                # Pad with last known snapshot
                sequence.append(sequence[-1])
            else:
                # No snapshots yet, create dummy
                sequence.append(self._create_dummy_snapshot())

        return sequence

    def _create_dummy_snapshot(self) -> Dict:
        """Create a dummy snapshot for padding."""
        return {
            "t": "1970-01-01T00:00:00+00:00",
            "sym": "nq",
            "bids": [{"price": 0, "size": 0} for _ in range(10)],
            "asks": [{"price": 0, "size": 0} for _ in range(10)],
            "spread": 0,
            "mid": 0,
            "bid_depth": 0,
            "ask_depth": 0,
            "imbalance": 0,
        }

    def sync_with_bars(
        self,
        bars: List[Dict],
        date: str,
        seq_len: int = 30
    ) -> Dict[int, List[Dict]]:
        """
        Sync L2 snapshots with 1-second OHLCV bars.

        Args:
            bars: List of 1-second bars with 't' timestamp
            date: Date string "YYYY-MM-DD"
            seq_len: Snapshot sequence length for CNN

        Returns:
            Dict mapping bar_idx -> snapshot_sequence (list of 30 snapshots)
        """
        if date not in self.snapshots_by_date:
            print(f"[L2] No snapshots for {date}, loading...")
            self.load_date(date)

        synced = {}

        for bar_idx, bar in enumerate(bars):
            # Parse bar timestamp
            bar_time = datetime.fromisoformat(bar['t'].replace('Z', '+00:00'))

            # Get snapshot sequence ending at this bar
            sequence = self.get_snapshot_sequence(date, bar_time, seq_len=seq_len)

            if sequence:
                synced[bar_idx] = sequence

            if (bar_idx + 1) % 10000 == 0:
                print(f"[L2] Synced {bar_idx + 1:,}/{len(bars):,} bars", end='\r', flush=True)

        print(f"[L2] Synced {len(synced):,}/{len(bars):,} bars with L2 data     ")
        return synced

    def get_stats(self, date: str) -> Dict:
        """Get statistics about loaded L2 data."""
        if date not in self.snapshots_by_date:
            return {}

        snapshots = self.snapshots_by_date[date]
        timestamps = self.timestamps_by_date[date]

        if not snapshots:
            return {}

        spreads = [s.get("spread", 0) for s in snapshots if s.get("spread", 0) > 0]
        imbalances = [s.get("imbalance", 0) for s in snapshots]

        return {
            "count": len(snapshots),
            "start_time": timestamps[0].isoformat(),
            "end_time": timestamps[-1].isoformat(),
            "avg_spread": np.mean(spreads) if spreads else 0,
            "avg_imbalance": np.mean(imbalances),
            "std_imbalance": np.std(imbalances),
        }


def test_l2_loader():
    """Test the L2 data loader."""
    print("=" * 80)
    print("Testing L2 Data Loader")
    print("=" * 80)

    loader = L2DataLoader(l2_data_dir="/home/costa/Trading-Agent/qwen-topstepx-bundle/ml/data/l2")

    # Load a date
    date = "2025-12-07"
    count = loader.load_date(date, symbol="nq")
    print(f"\nLoaded {count:,} snapshots for {date}")

    # Get stats
    stats = loader.get_stats(date)
    print(f"\nStats:")
    print(f"  Start: {stats['start_time']}")
    print(f"  End: {stats['end_time']}")
    print(f"  Avg spread: {stats['avg_spread']:.4f}")
    print(f"  Avg imbalance: {stats['avg_imbalance']:.4f}")

    # Test getting snapshot at specific time
    timestamps = loader.timestamps_by_date[date]
    if timestamps:
        test_time = timestamps[len(timestamps) // 2]
        snap = loader.get_snapshot_at_time(date, test_time)
        print(f"\nSnapshot at {test_time}:")
        print(f"  Spread: {snap['spread']:.2f}")
        print(f"  Mid: {snap['mid']:.2f}")
        print(f"  Imbalance: {snap['imbalance']:.4f}")
        print(f"  Bid depth: {snap['bid_depth']}")
        print(f"  Ask depth: {snap['ask_depth']}")

        # Test sequence
        sequence = loader.get_snapshot_sequence(date, test_time, seq_len=30)
        print(f"\nSequence of {len(sequence)} snapshots retrieved")

    print("\n" + "=" * 80)
    print("✓ L2 Data Loader working")


if __name__ == "__main__":
    test_l2_loader()
