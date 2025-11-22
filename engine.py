from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple
import json
import os

from config import Settings
from topstep_client import TopstepClient
from features import FeatureEngine
from llm_client import LLMClient
from execution import ExecutionEngine
from storage import save_llm_exchange


def _load_strategy_state_from_log(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Reconstruct strategy_state by replaying strategy_updates from the LLM log.
    """
    state: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return state

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = record.get("response") or {}
            updates = response.get("strategy_updates") or {}
            for tweak in updates.get("strategy_tweaks", []):
                name = tweak.get("name")
                changes = tweak.get("changes") or {}
                if not name:
                    continue
                cfg = state.setdefault(name, {})
                cfg.update(changes)
    return state


def _load_historical_notes(path: str, symbol: str, max_notes: int = 20) -> List[str]:
    """
    Extract a small list of notes_to_future_self from the log for the given symbol.
    """
    if not os.path.exists(path):
        return []

    notes: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("symbol") != symbol:
                continue
            response = record.get("response") or {}
            for note in response.get("notes_to_future_self") or []:
                if isinstance(note, str):
                    notes.append(note)
    return notes[-max_notes:]


def _should_call_llm(
    now: float,
    last_call_time: float | None,
    importance_zones: List[Dict[str, Any]],
    price: float,
    settings: Settings,
) -> bool:
    """
    Decide whether to call the LLM, based on time since last call and
    proximity to high-priority zones.
    """
    if last_call_time is None:
        return True

    default_interval = settings.llm_decision_interval_default_sec
    outer_interval = settings.llm_decision_interval_outer_band_sec
    inner_interval = settings.llm_decision_interval_inner_band_sec

    elapsed = now - last_call_time
    min_interval = default_interval

    for zone in importance_zones:
        if zone.get("priority") != "high":
            continue
        center = zone.get("center_price")
        inner = zone.get("inner_band_ticks", 0)
        outer = zone.get("outer_band_ticks", 0)
        if center is None:
            continue
        distance_ticks = abs(price - center) / max(0.25, zone.get("tick_size", 0.25))
        if distance_ticks <= inner:
            min_interval = min(min_interval, inner_interval)
        elif distance_ticks <= outer:
            min_interval = min(min_interval, outer_interval)

    return elapsed >= min_interval


async def run_agent(settings: Settings) -> None:
    """
    Main agent loop: streams market data, computes features, calls LLM,
    and routes decisions to the execution engine.
    """
    topstep = TopstepClient(settings)
    feature_engine = FeatureEngine()
    llm = LLMClient(settings)

    strategy_state = _load_strategy_state_from_log(settings.llm_log_path)
    historical_notes = _load_historical_notes(settings.llm_log_path, settings.symbol)
    importance_zones: List[Dict[str, Any]] = []

    execution = ExecutionEngine(
        settings=settings,
        topstep_client=topstep,
        account_balance=settings.account_balance,
    )

    last_llm_call: float | None = None

    async for snapshot in topstep.stream_snapshots():
        last_price = snapshot.bar.close

        feat_state = feature_engine.update_features_and_get_state(snapshot)

        now = snapshot.timestamp

        if _should_call_llm(now, last_llm_call, importance_zones, last_price, settings):
            payload: Dict[str, Any] = {
                "mode": "live_decision",
                "symbol": settings.symbol,
                "timestamp": snapshot.timestamp,
                "session": "Unknown",  # you can add a proper session classifier
                "price": last_price,
                "ohlc": {
                    "open": snapshot.bar.open,
                    "high": snapshot.bar.high,
                    "low": snapshot.bar.low,
                    "close": snapshot.bar.close,
                },
                "profiles": feat_state.get("profiles", []),
                "derived_state": feat_state.get("derived_state", {}),
                "orderflow": feat_state.get("orderflow", {}),
                "session_metrics": feat_state.get("session_metrics", {}),
                "open_positions": [
                    {
                        "id": pos.id,
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "size": pos.size,
                        "stop_price": pos.stop_price,
                        "target_price": pos.target_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                    }
                    for pos in execution.positions.values()
                ],
                "importance_zones": importance_zones,
                "recent_performance_summary": {
                    "session_pnl": execution.current_daily_pnl,
                    "session_drawdown": execution.current_daily_drawdown,
                },
                "historical_notes_snippet": historical_notes,
                "strategy_state": strategy_state,
            }

            response = await llm.request_decision(payload)
            save_llm_exchange(settings.symbol, payload, response, settings.llm_log_path)

            # Apply strategy updates (self‑learning)
            updates = response.get("strategy_updates") or {}
            for tweak in updates.get("strategy_tweaks", []):
                name = tweak.get("name")
                changes = tweak.get("changes") or {}
                if not name:
                    continue
                cfg = strategy_state.setdefault(name, {})
                cfg.update(changes)

            importance_zones = response.get("importance_zones") or []
            notes = response.get("notes_to_future_self") or []
            for note in notes:
                if isinstance(note, str):
                    historical_notes.append(note)
            if len(historical_notes) > 20:
                historical_notes = historical_notes[-20:]

            decisions = response.get("trade_decisions") or []
            await execution.apply_trade_decisions(decisions, current_price=last_price, strategy_state=strategy_state)

            last_llm_call = now
