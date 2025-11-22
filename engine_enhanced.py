"""
Enhanced Engine for Self-Learning Trading Agent
Integrates all enhanced components for true self-learning behavior
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple
import json
import os
import time

from config import Settings
from topstep_client import TopstepClient
from features_enhanced import EnhancedFeatureEngine
from llm_client_enhanced import EnhancedLLMClient
from execution_enhanced import EnhancedExecutionEngine
from storage import save_llm_exchange


def _load_strategy_state_from_log(path: str) -> Dict[str, Dict[str, Any]]:
    """Reconstruct strategy_state by replaying strategy_updates from the LLM log"""
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


def _load_historical_notes(path: str, symbol: str, max_notes: int = 30) -> List[str]:
    """Extract historical notes from past self for context"""
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


def _load_performance_history(path: str, symbol: str) -> Dict[str, Any]:
    """Calculate strategy performance from historical trades"""
    if not os.path.exists(path):
        return {}

    strategy_performance = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("symbol") != symbol:
                continue

            response = record.get("response") or {}

            # Track trades per strategy
            for decision in response.get("trade_decisions", []):
                if decision.get("action") == "enter":
                    strategy = decision.get("entry_condition_id", "unknown")
                    if strategy not in strategy_performance:
                        strategy_performance[strategy] = {
                            "trades": 0,
                            "wins": 0,
                            "losses": 0,
                            "total_pnl": 0.0
                        }
                    strategy_performance[strategy]["trades"] += 1

    # Convert to performance metrics
    performance = {}
    for strategy, stats in strategy_performance.items():
        if stats["trades"] > 0:
            performance[strategy] = {
                "win_rate_last_30": stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0,
                "total_trades": stats["trades"],
                "net_pnl": stats["total_pnl"]
            }

    return performance


def _should_call_llm(
    now: float,
    last_call_time: float | None,
    importance_zones: List[Dict[str, Any]],
    price: float,
    settings: Settings,
) -> bool:
    """Decide whether to call the LLM based on time and proximity to zones"""
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
        distance_ticks = abs(price - center) / 0.25  # NQ tick size
        if distance_ticks <= inner:
            min_interval = min(min_interval, inner_interval)
        elif distance_ticks <= outer:
            min_interval = min(min_interval, outer_interval)

    return elapsed >= min_interval


async def run_enhanced_agent(settings: Settings) -> None:
    """
    Main self-learning agent loop with enhanced features
    """
    print("=" * 80)
    print("🧠 SELF-LEARNING FABIO AGENT - ENHANCED VERSION")
    print("=" * 80)
    print(f"Symbol: {settings.symbol}")
    print(f"Mode: {settings.mode}")
    print("Features: POC Cross Tracking | Market Inference | Active Strategy Learning")
    print("=" * 80)

    # Initialize components
    topstep = TopstepClient(settings)
    feature_engine = EnhancedFeatureEngine()
    llm = EnhancedLLMClient(settings)

    # Fetch account info dynamically from TopStep
    print("[Engine] Fetching account balance from TopStep...")
    try:
        account_info = await topstep.get_account_info()
        account_balance = account_info.get("balance", settings.account_balance)
        print(f"[Engine] ✅ Account balance fetched: ${account_balance:,.2f}")
        print(f"[Engine] Account Status: {account_info.get('account_status', 'unknown')}")
        print(f"[Engine] Daily P&L: ${account_info.get('daily_pnl', 0):,.2f}")
    except Exception as e:
        print(f"[Engine] ⚠️  Failed to fetch account info: {e}")
        print(f"[Engine] Using fallback balance from config: ${settings.account_balance:,.2f}")
        account_balance = settings.account_balance

    execution = EnhancedExecutionEngine(
        settings=settings,
        topstep_client=topstep,
        account_balance=account_balance,
    )

    # Initialize execution strategies
    execution.initialize_default_strategies()

    # Store account info for periodic updates
    last_balance_sync = time.time()
    balance_sync_interval = 300  # Sync every 5 minutes

    # Load persisted state
    strategy_state = _load_strategy_state_from_log(settings.llm_log_path)
    if strategy_state:
        # Apply loaded strategy state to execution engine
        for strategy_name, config in strategy_state.items():
            execution.strategy_state[strategy_name] = config
        execution.active_strategies = [
            name for name, cfg in strategy_state.items()
            if cfg.get("enabled", False)
        ]
        print(f"[Engine] Loaded strategy state from log: {list(strategy_state.keys())}")

    historical_notes = _load_historical_notes(settings.llm_log_path, settings.symbol)
    print(f"[Engine] Loaded {len(historical_notes)} historical notes")

    # Load performance history
    performance_history = _load_performance_history(settings.llm_log_path, settings.symbol)

    importance_zones: List[Dict[str, Any]] = []
    last_llm_call: float | None = None

    print("[Engine] Starting market data stream...")

    async for snapshot in topstep.stream_snapshots():
        last_price = snapshot.bar.close
        now = snapshot.timestamp

        # Update enhanced features (with POC tracking, etc.)
        feat_state = feature_engine.update_features_and_get_state(snapshot)

        # Check stops and targets
        execution.check_stops_and_targets(last_price)

        # Periodically sync account balance
        if time.time() - last_balance_sync > balance_sync_interval:
            try:
                account_info = await topstep.get_account_info()
                new_balance = account_info.get("balance", execution.account_balance)
                if new_balance != execution.account_balance:
                    print(f"[Engine] 💰 Balance updated: ${execution.account_balance:,.2f} → ${new_balance:,.2f}")
                    execution.account_balance = new_balance
                    execution.daily_pnl = account_info.get("daily_pnl", execution.daily_pnl)
                last_balance_sync = time.time()
            except Exception as e:
                print(f"[Engine] Failed to sync balance: {e}")

        # Decide if we should call LLM
        if _should_call_llm(now, last_llm_call, importance_zones, last_price, settings):

            # Get current performance from execution engine
            current_performance = execution.get_performance_stats()

            # Merge with historical performance
            all_performance = {**performance_history, **current_performance}

            # Build enhanced payload with raw stats for LLM to infer
            payload: Dict[str, Any] = {
                "mode": "live_decision",
                "symbol": settings.symbol,
                "timestamp": now,
                "price": last_price,

                # Raw market stats - LLM will infer regime
                "market_stats": feat_state.get("market_stats", {}),

                # Volume profile data
                "profiles": feat_state.get("profiles", []),

                # Order flow metrics
                "orderflow": feat_state.get("orderflow", {}),

                # Current positions
                "open_positions": [
                    {
                        "id": pos.id,
                        "side": pos.side,
                        "entry_price": pos.entry_price,
                        "size": pos.size,
                        "stop_price": pos.stop_price,
                        "target_price": pos.target_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "strategy": pos.strategy,
                        "age_seconds": now - pos.entry_time,
                    }
                    for pos in execution.positions.values()
                ],

                # Strategy state for LLM to modify
                "strategy_state": execution.strategy_state,

                # Performance metrics for self-learning
                "strategy_performance": all_performance,

                # Historical context
                "historical_notes_snippet": historical_notes[-20:],

                # Important zones from previous LLM calls
                "importance_zones": importance_zones,

                # Session metrics
                "session_stats": {
                    "daily_pnl": execution.daily_pnl,
                    "trades_today": execution.trades_today,
                    "max_daily_loss": execution.max_daily_loss,
                }
            }

            # Call enhanced LLM with self-learning prompt
            try:
                print(f"[Engine] Calling LLM for decision...")
                llm_response = await llm.request_decision(payload)
                last_llm_call = now

                # Log the exchange
                save_llm_exchange(settings.symbol, payload, llm_response, settings.llm_log_path)

                # Extract market assessment (LLM inferred it)
                market_assessment = llm_response.get("market_assessment", {})
                print(f"[Engine] LLM Assessment: {market_assessment.get('regime', 'unknown')} "
                      f"regime, {market_assessment.get('market_state', 'unknown')} state")
                print(f"  Reasoning: {market_assessment.get('reasoning', 'N/A')}")

                # Apply strategy updates (self-learning!)
                strategy_updates = llm_response.get("strategy_updates", {})
                if strategy_updates:
                    execution.apply_strategy_updates(strategy_updates)
                    print("[Engine] Strategy state updated by LLM")

                # Process trade decisions
                for decision in llm_response.get("trade_decisions", []):
                    result = await execution.process_trade_decision(decision, last_price)
                    print(f"[Engine] Trade decision result: {result}")

                # Update importance zones
                new_zones = llm_response.get("importance_zones", [])
                if new_zones:
                    importance_zones = new_zones
                    print(f"[Engine] Updated {len(importance_zones)} importance zones")

                # Store notes to future self
                notes = llm_response.get("notes_to_future_self", [])
                if notes:
                    print(f"[Engine] Notes to future self: {notes[0][:100]}...")

                # Add trade results to feature engine for performance tracking
                for trade in execution.closed_trades[-10:]:
                    feature_engine.add_trade_result(trade)

            except Exception as e:
                print(f"[Engine] LLM error: {e}")

        # Small delay to avoid overwhelming the system
        await asyncio.sleep(0.1)

    print("[Engine] Market data stream ended")


async def main():
    """Main entry point for enhanced agent"""
    settings = Settings.from_env()

    try:
        await run_enhanced_agent(settings)
    except KeyboardInterrupt:
        print("\n[Engine] Shutdown requested")
    except Exception as e:
        print(f"[Engine] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Engine] Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())