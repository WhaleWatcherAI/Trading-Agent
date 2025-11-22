#!/usr/bin/env python3
"""
Enhanced Fabio Agent with Dashboard Integration
Combines the self-learning engine_enhanced.py with dashboard_bridge.py
"""

from __future__ import annotations

print("[DEBUG] Script starting - imports beginning...", flush=True)

import asyncio
import argparse
from typing import Any, Dict, List, Optional
import os
import json
import time

from config import Settings
from topstep_client import TopstepClient
from features_enhanced import EnhancedFeatureEngine
from llm_client_enhanced import EnhancedLLMClient
from execution_enhanced import EnhancedExecutionEngine
from storage import save_llm_exchange
from dashboard_bridge import DashboardBridge, DashboardPosition, DashboardTrade

print("[DEBUG] All imports complete!", flush=True)


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


async def run_fabio_with_dashboard(settings: Settings, dashboard_url: str = "http://localhost:3337"):
    """
    Run enhanced Fabio agent with dashboard integration
    """
    print("=" * 80, flush=True)
    print("🧠 ENHANCED FABIO AGENT + DASHBOARD", flush=True)
    print("=" * 80, flush=True)
    print(f"Symbol: {settings.symbol}", flush=True)
    print(f"Mode: {settings.mode}", flush=True)
    print(f"Dashboard: {dashboard_url}", flush=True)
    print("Features: Self-Learning | POC Tracking | Market Inference", flush=True)
    print("=" * 80, flush=True)

    # Initialize dashboard bridge
    print("[DEBUG] Creating DashboardBridge...", flush=True)
    bridge = DashboardBridge(dashboard_url)
    print("[DEBUG] Starting bridge connection task...", flush=True)
    bridge_task = asyncio.create_task(bridge.connect())
    print("[DEBUG] Waiting 2 seconds for bridge...", flush=True)
    await asyncio.sleep(2)

    if not bridge.connected:
        print("⚠️  Dashboard not connected, continuing anyway...", flush=True)
    else:
        print("✅ Dashboard bridge connected!", flush=True)

    # Initialize Fabio components
    print("[DEBUG] Creating TopstepClient...", flush=True)
    topstep = TopstepClient(settings)
    print("[DEBUG] Creating EnhancedFeatureEngine...", flush=True)
    feature_engine = EnhancedFeatureEngine()
    print("[DEBUG] Creating EnhancedLLMClient...", flush=True)
    llm = EnhancedLLMClient(settings)
    print("[DEBUG] Creating EnhancedExecutionEngine...", flush=True)
    execution = EnhancedExecutionEngine(
        settings=settings,
        topstep_client=topstep,
        account_balance=settings.account_balance,
    )
    print("[DEBUG] All components created!", flush=True)

    # Initialize execution strategies
    print("[DEBUG] Initializing default strategies...", flush=True)
    execution.initialize_default_strategies()

    # Load persisted state
    print("[DEBUG] Loading strategy state from log...", flush=True)
    strategy_state = _load_strategy_state_from_log(settings.llm_log_path)
    if strategy_state:
        for strategy_name, config in strategy_state.items():
            execution.strategy_state[strategy_name] = config
        execution.active_strategies = [
            name for name, cfg in strategy_state.items()
            if cfg.get("enabled", False)
        ]
        print(f"[Fabio] Loaded strategy state: {list(strategy_state.keys())}")

    historical_notes = _load_historical_notes(settings.llm_log_path, settings.symbol)
    print(f"[Fabio] Loaded {len(historical_notes)} historical notes")

    performance_history = _load_performance_history(settings.llm_log_path, settings.symbol)

    importance_zones: List[Dict[str, Any]] = []
    last_llm_call: float | None = None
    last_dashboard_update = 0.0

    # Send startup logs
    await bridge.send_log("🧠 Enhanced Fabio agent started", "success")
    await bridge.send_log(f"Trading {settings.symbol} in {settings.mode} mode", "info")

    print("[Fabio] Starting market data stream from dashboard...", flush=True)
    print("[DEBUG] Receiving market data via Socket.IO 'bar' events...", flush=True)

    # Main loop - process market data from dashboard
    last_llm_call_time = None
    while True:
        # Wait for latest bar from dashboard
        if bridge.latest_bar is None:
            await asyncio.sleep(0.5)
            continue

        bar_data = bridge.latest_bar
        last_price = bar_data.get('close', 0)
        now = bar_data.get('timestamp', time.time())

        # Create a minimal snapshot for the feature engine
        # (We don't have full market depth from dashboard, but that's OK)
        from topstep_client import OneSecondBar, MarketSnapshot
        bar = OneSecondBar(
            open=bar_data.get('open', last_price),
            high=bar_data.get('high', last_price),
            low=bar_data.get('low', last_price),
            close=last_price,
            volume=bar_data.get('volume', 0),
            timestamp=now,
        )
        snapshot = MarketSnapshot(
            timestamp=now,
            symbol=settings.symbol,
            bar=bar,
            l1=None,  # Not available from dashboard
            l2=[],    # Not available from dashboard
            recent_trades=[],  # Not available from dashboard
        )

        # Update enhanced features
        feat_state = feature_engine.update_features_and_get_state(snapshot)

        # Check stops and targets
        execution.check_stops_and_targets(last_price)

        # Update dashboard periodically (every 1 second)
        if now - last_dashboard_update >= 1.0:
            await _update_dashboard(bridge, execution, last_price, now)
            last_dashboard_update = now

        # Decide if we should call LLM
        if _should_call_llm(now, last_llm_call, importance_zones, last_price, settings):
            await bridge.send_log("🤔 Requesting LLM decision...", "info")

            # Get current performance
            current_performance = execution.get_performance_stats()
            all_performance = {**performance_history, **current_performance}

            # Build enhanced payload
            payload: Dict[str, Any] = {
                "mode": "live_decision",
                "symbol": settings.symbol,
                "timestamp": now,
                "price": last_price,
                "market_stats": feat_state.get("market_stats", {}),
                "profiles": feat_state.get("profiles", []),
                "orderflow": feat_state.get("orderflow", {}),
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
                "strategy_state": execution.strategy_state,
                "strategy_performance": all_performance,
                "historical_notes_snippet": historical_notes[-20:],
                "importance_zones": importance_zones,
                "session_stats": {
                    "daily_pnl": execution.daily_pnl,
                    "trades_today": execution.trades_today,
                    "max_daily_loss": execution.max_daily_loss,
                }
            }

            # Call LLM
            try:
                print(f"[Fabio] Calling LLM (price={last_price:.2f})...")
                llm_response = await llm.request_decision(payload)
                last_llm_call = now

                # Log exchange
                save_llm_exchange(settings.symbol, payload, llm_response, settings.llm_log_path)

                # Extract assessment
                market_assessment = llm_response.get("market_assessment", {})
                regime = market_assessment.get("regime", "unknown")
                state = market_assessment.get("market_state", "unknown")
                reasoning = market_assessment.get("reasoning", "")

                print(f"[Fabio] LLM Assessment: {regime} regime, {state} state")
                print(f"  Reasoning: {reasoning[:100]}...")

                await bridge.send_log(f"📊 LLM: {regime} regime, {state}", "info")
                if reasoning:
                    await bridge.send_log(f"💡 {reasoning[:200]}", "info")

                # Apply strategy updates
                strategy_updates = llm_response.get("strategy_updates", {})
                if strategy_updates:
                    execution.apply_strategy_updates(strategy_updates)
                    print("[Fabio] Strategy updated")
                    await bridge.send_log("🔄 Strategy parameters updated", "success")

                # Process trade decisions
                trade_decisions = llm_response.get("trade_decisions", [])
                if not trade_decisions:
                    # No trade - log the reason why
                    print(f"[Fabio] 💭 No trade taken")
                    print(f"  Reasoning: {reasoning}")
                    await bridge.send_log(f"💭 Waiting: {reasoning[:150]}", "info")
                else:
                    for decision in trade_decisions:
                        result = await execution.process_trade_decision(decision, last_price)
                        print(f"[Fabio] Trade decision: {result}")

                        if result.get("success"):
                            action = decision.get("action", "unknown")
                            await bridge.send_log(f"✅ {action.upper()} executed", "success")

                # Update importance zones
                new_zones = llm_response.get("importance_zones", [])
                if new_zones:
                    importance_zones = new_zones
                    print(f"[Fabio] Updated {len(importance_zones)} zones")

                # Notes to future self
                notes = llm_response.get("notes_to_future_self", [])
                if notes:
                    print(f"[Fabio] Note: {notes[0][:100]}...")

            except Exception as e:
                print(f"[Fabio] LLM error: {e}")
                await bridge.send_log(f"❌ LLM error: {str(e)}", "error")

        await asyncio.sleep(0.1)

    print("[Fabio] Market data stream ended")


async def _update_dashboard(bridge, execution, price, timestamp):
    """Update dashboard with current state"""
    # Build position data
    position = None
    if execution.positions:
        pos = list(execution.positions.values())[0]
        position = DashboardPosition(
            side=pos.side,
            entry_price=pos.entry_price,
            stop_loss=pos.stop_price,
            tp1=pos.target_price,
            tp2=pos.target_price,  # Enhanced execution may have different TP structure
            contracts=pos.size,
            unrealized_pnl=pos.unrealized_pnl,
            entry_pattern=pos.strategy or "Fabio LLM Decision"
        )

    # Build recent trades
    recent_trades = []
    for trade in execution.closed_trades[-10:]:
        recent_trades.append(DashboardTrade(
            side=trade.get("side", "long"),
            pnl=trade.get("pnl", 0),
            exit_reason=trade.get("exit_reason", "unknown"),
            exit_price=trade.get("exit_price", 0)
        ))

    # Calculate stats
    total_trades = len(execution.closed_trades)
    winning_trades = len([t for t in execution.closed_trades if t.get("pnl", 0) > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_pnl = sum(t.get("pnl", 0) for t in execution.closed_trades)

    # Send status update
    await bridge.send_status(
        position=position,
        total_trades=total_trades,
        win_rate=win_rate,
        total_pnl=total_pnl,
        entry_price=position.entry_price if position else price,
        closed_trades=recent_trades
    )


async def main():
    """Main entry point"""
    print("[DEBUG] main() function called!", flush=True)
    parser = argparse.ArgumentParser(description="Enhanced Fabio Agent with Dashboard")
    parser.add_argument("--symbol", default=os.getenv("SYMBOL", "NQZ5"), help="Trading symbol")
    parser.add_argument("--mode", default=os.getenv("MODE", "paper_trading"), help="Trading mode")
    parser.add_argument("--dashboard-url", default="http://localhost:3337", help="Dashboard URL")
    args = parser.parse_args()

    # Create settings
    os.environ["TRADING_SYMBOL"] = args.symbol
    os.environ["TRADING_MODE"] = args.mode
    settings = Settings.from_env()

    try:
        await run_fabio_with_dashboard(settings, args.dashboard_url)
    except KeyboardInterrupt:
        print("\n[Fabio] Shutdown requested")
    except Exception as e:
        print(f"[Fabio] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Fabio] Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
