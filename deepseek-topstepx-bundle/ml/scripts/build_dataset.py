#!/usr/bin/env python3
"""
Build a meta-label dataset from trading decisions/outcomes + TS-logged market snapshots.
Labels: win_5m / win_30m (TP before SL within horizon, approximated by P&L within time window).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ml" / "data"
MODELS_DIR = ROOT / "ml" / "models"
SNAPSHOT_PATH = DATA_DIR / "snapshots.jsonl"
DECISIONS_PATH = ROOT / "trading-db" / "decisions.jsonl"
OUTCOMES_PATH = ROOT / "trading-db" / "outcomes.jsonl"
OUTPUT_PARQUET = DATA_DIR / "meta_label.parquet"
FEATURES_JSON = MODELS_DIR / "features.json"


@dataclass
class Snapshot:
    symbol: str
    timestamp: datetime
    features: Dict[str, float]


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _parse_dt(dt_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def load_snapshots() -> Dict[str, List[Snapshot]]:
    raw = _read_jsonl(SNAPSHOT_PATH)
    grouped: Dict[str, List[Snapshot]] = {}
    for row in raw:
        symbol = row.get("symbol")
        ts = _parse_dt(row.get("timestamp"))
        feats = row.get("features") or {}
        if not symbol or not ts:
            continue
        numeric_feats = {
            k: (float(v) if v is not None and not isinstance(v, str) and not math.isnan(float(v)) else np.nan)
            for k, v in feats.items()
            if v is None or isinstance(v, (int, float))
        }
        grouped.setdefault(symbol, []).append(Snapshot(symbol, ts, numeric_feats))

    for sym, snaps in grouped.items():
        grouped[sym] = sorted(snaps, key=lambda s: s.timestamp)
    return grouped


def load_decisions() -> Dict[str, dict]:
    decisions = {}
    for row in _read_jsonl(DECISIONS_PATH):
        if not isinstance(row, dict):
            continue
        if "id" in row:
            decisions[row["id"]] = row
    return decisions


def load_outcomes() -> List[dict]:
    return [row for row in _read_jsonl(OUTCOMES_PATH) if isinstance(row, dict)]


def find_snapshot(symbol: str, ts: datetime, snaps_by_symbol: Dict[str, List[Snapshot]]) -> Optional[Snapshot]:
    snaps = snaps_by_symbol.get(symbol, [])
    if not snaps:
        return None

    # binary search for last snapshot <= ts
    lo, hi = 0, len(snaps) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if snaps[mid].timestamp <= ts:
            best = snaps[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best or snaps[0]


def label_outcome(outcome: dict, decision: dict, horizon_minutes: int) -> Optional[int]:
    exec_time = _parse_dt(outcome.get("executedTime") or decision.get("filledTime") or decision.get("timestamp"))
    close_time = _parse_dt(outcome.get("closedTime") or outcome.get("timestamp"))
    if not exec_time or not close_time:
        return None

    duration_min = (close_time - exec_time).total_seconds() / 60.0
    profit = float(outcome.get("profitLoss") or 0.0)
    # Approximation: classify win if trade closed with profit within horizon
    if duration_min <= horizon_minutes and profit > 0:
        return 1
    return 0


def build_dataset() -> Tuple[pd.DataFrame, List[str]]:
    decisions = load_decisions()
    outcomes = load_outcomes()
    snaps = load_snapshots()

    if not decisions:
        print("No decisions found; aborting.")
        return pd.DataFrame(), []
    if not outcomes:
        print("No outcomes found; aborting.")
        return pd.DataFrame(), []
    if not snaps:
        print("No snapshots found; aborting. Enable TS snapshot logging.")
        return pd.DataFrame(), []

    rows = []
    for outcome in tqdm(outcomes, desc="Linking trades to snapshots"):
        decision_id = outcome.get("decisionId")
        decision = decisions.get(decision_id)
        if not decision:
            continue
        symbol = decision.get("symbol")
        exec_time = _parse_dt(outcome.get("executedTime") or decision.get("filledTime") or decision.get("timestamp"))
        if not symbol or not exec_time:
            continue

        snap = find_snapshot(symbol, exec_time, snaps)
        if not snap:
            continue

        win_5m = label_outcome(outcome, decision, 5)
        win_30m = label_outcome(outcome, decision, 30)
        if win_5m is None or win_30m is None:
            continue

        base = {
            "symbol": symbol,
            "entry_time": exec_time.isoformat(),
            "win_5m": win_5m,
            "win_30m": win_30m,
        }
        row = {**base, **snap.features}
        rows.append(row)

    if not rows:
        print("No matched rows; nothing to save.")
        return pd.DataFrame(), []

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("symbol", "entry_time", "win_5m", "win_30m")]
    df = df.sort_values("entry_time")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    FEATURES_JSON.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_JSON.write_text(json.dumps({"feature_columns": feature_cols, "generated_at": datetime.utcnow().isoformat()}, indent=2))

    print(f"Saved dataset: {OUTPUT_PARQUET} ({len(df)} rows, {len(feature_cols)} features)")
    return df, feature_cols


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df, features = build_dataset()
    if df.empty:
        sys.exit(1)
    print(df.head(3))


if __name__ == "__main__":
    main()
