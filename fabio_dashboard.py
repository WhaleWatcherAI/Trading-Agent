"""
Integration layer between Fabio LLM agent and the NQ ICT dashboard.

This module:
1. Runs the Fabio agent engine
2. Monitors execution state (positions, trades, PnL)
3. Broadcasts updates to the dashboard via Socket.IO bridge

Usage:
    python fabio_dashboard.py --symbol NQZ5 --mode paper_trading
"""

from __future__ import annotations

import asyncio
import argparse
from typing import Optional
from datetime import datetime

from config import Settings
from engine import run_agent
from dashboard_bridge import DashboardBridge, DashboardPosition, DashboardTrade
from execution import ExecutionEngine, Position
from topstep_client import TopstepClient, MarketSnapshot
from features import FeatureEngine
from llm_client import LLMClient
from storage import save_llm_exchange
import os
import json


class FabioDashboardIntegration:
    """
    Integration layer that runs Fabio agent and broadcasts to dashboard.
    """

    def __init__(
        self,
        settings: Settings,
        dashboard_url: str = "http://localhost:3337",
    ):
        self.settings = settings
        self.dashboard_url = dashboard_url
        self.bridge: Optional[DashboardBridge] = None

        # Agent components
        self.topstep: Optional[TopstepClient] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.llm: Optional[LLMClient] = None
        self.execution: Optional[ExecutionEngine] = None

        # State tracking
        self.last_status_update = 0.0
        self.status_update_interval = 1.0  # seconds

    async def run(self):
        """Run the integrated Fabio agent + dashboard"""
        print("=" * 80)
        print("FABIO LLM TRADING AGENT - DASHBOARD INTEGRATION")
        print("=" * 80)
        print(f"Symbol: {self.settings.symbol}")
        print(f"Mode: {self.settings.mode}")
        print(f"Dashboard: {self.dashboard_url}")
        print("=" * 80)

        # Initialize dashboard bridge
        self.bridge = DashboardBridge(self.dashboard_url)

        # Start bridge connection in background
        bridge_task = asyncio.create_task(self.bridge.connect())

        # Wait a moment for connection
        await asyncio.sleep(2)

        if not self.bridge.connected:
            print("⚠️ Warning: Dashboard not connected, continuing anyway...")

        # Initialize agent components (similar to engine.py)
        self.topstep = TopstepClient(self.settings)
        self.feature_engine = FeatureEngine()
        self.llm = LLMClient(self.settings)

        # Load strategy state from log
        strategy_state = self._load_strategy_state()
        historical_notes = self._load_historical_notes()
        importance_zones = []

        self.execution = ExecutionEngine(
            settings=self.settings,
            topstep_client=self.topstep,
            account_balance=self.settings.account_balance,
        )

        last_llm_call: Optional[float] = None

        # Send initial log
        await self._send_log("Fabio agent started", "success")
        await self._send_log(f"Trading {self.settings.symbol} in {self.settings.mode} mode", "info")

        # Start periodic status updates
        status_task = asyncio.create_task(self._periodic_status_update())

        try:
            # Main agent loop (adapted from engine.py)
            async for snapshot in self.topstep.stream_snapshots():
                last_price = snapshot.bar.close

                # Log price updates occasionally
                if int(snapshot.timestamp) % 10 == 0:  # Every 10 seconds
                    await self._send_log(f"Price: {last_price:.2f}", "info")

                feat_state = self.feature_engine.update_features_and_get_state(snapshot)

                now = snapshot.timestamp

                # Check if we should call LLM
                if self._should_call_llm(now, last_llm_call, importance_zones, last_price):
                    await self._send_log("Requesting LLM decision...", "info")

                    payload = self._build_llm_payload(
                        snapshot, feat_state, importance_zones,
                        historical_notes, strategy_state
                    )

                    # Send the prompt to the dashboard BEFORE calling the LLM
                    if self.bridge and self.bridge.connected:
                        await self.bridge.send_llm_prompt(payload, timestamp=now)

                    response = await self.llm.request_decision(payload)
                    save_llm_exchange(
                        self.settings.symbol,
                        payload,
                        response,
                        self.settings.llm_log_path
                    )

                    # Log LLM response
                    await self._send_log(f"LLM response received", "success")

                    # Optionally send the response too
                    if self.bridge and self.bridge.connected and response:
                        # Send response as a separate prompt entry
                        response_data = {
                            "type": "llm_response",
                            "decisions": response.get("trade_decisions", []),
                            "importance_zones": response.get("importance_zones", []),
                            "notes": response.get("notes_to_future_self", []),
                            "strategy_updates": response.get("strategy_updates", {}),
                        }
                        await self.bridge.send_llm_prompt(response_data, timestamp=now)

                    # Apply strategy updates
                    updates = response.get("strategy_updates") or {}
                    for tweak in updates.get("strategy_tweaks", []):
                        name = tweak.get("name")
                        changes = tweak.get("changes") or {}
                        if not name:
                            continue
                        cfg = strategy_state.setdefault(name, {})
                        cfg.update(changes)
                        await self._send_log(f"Strategy updated: {name}", "info")

                    importance_zones = response.get("importance_zones") or []
                    notes = response.get("notes_to_future_self") or []
                    for note in notes:
                        if isinstance(note, str):
                            historical_notes.append(note)
                    if len(historical_notes) > 20:
                        historical_notes = historical_notes[-20:]

                    # Track positions before applying decisions
                    old_positions = set(self.execution.positions.keys())

                    # Apply trade decisions
                    decisions = response.get("trade_decisions") or []
                    if decisions:
                        await self._send_log(f"Applying {len(decisions)} trade decision(s)", "info")

                    await self.execution.apply_trade_decisions(
                        decisions,
                        current_price=last_price,
                        strategy_state=strategy_state
                    )

                    # Check for new positions
                    new_positions = set(self.execution.positions.keys())
                    opened_positions = new_positions - old_positions
                    closed_positions = old_positions - new_positions

                    for pos_id in opened_positions:
                        pos = self.execution.positions[pos_id]
                        await self._send_log(
                            f"Position opened: {pos.side.upper()} {pos.size} @ {pos.entry_price:.2f}",
                            "success"
                        )

                    for pos_id in closed_positions:
                        # Position was closed, log it
                        await self._send_log(f"Position closed: {pos_id}", "info")

                    last_llm_call = now

                    # Force status update after LLM decision
                    await self._update_dashboard_status()

        except KeyboardInterrupt:
            print("\n[FabioDashboard] Shutting down...")
            await self._send_log("Fabio agent shutting down", "warning")
        finally:
            status_task.cancel()
            await self.bridge.disconnect()

    async def _periodic_status_update(self):
        """Periodically update dashboard status"""
        while True:
            try:
                await asyncio.sleep(self.status_update_interval)
                await self._update_dashboard_status()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[FabioDashboard] Error in status update: {e}")

    async def _update_dashboard_status(self):
        """Send current status to dashboard"""
        if not self.bridge or not self.bridge.connected or not self.execution:
            return

        # Get current position (if any)
        position = None
        if self.execution.positions:
            # Get first position (Fabio typically trades one at a time)
            pos_id, pos = next(iter(self.execution.positions.items()))
            position = self._convert_position(pos)

        # Get closed trades
        closed_trades = [
            self._convert_closed_trade(trade)
            for trade in self.execution.closed_trades
        ]

        # Calculate total PnL
        total_pnl = sum(trade.pnl for trade in closed_trades)

        await self.bridge.emit_status(
            position=position,
            closed_trades=closed_trades,
            total_pnl=total_pnl,
            symbol=self.settings.symbol,
            is_trading=True,
        )

    async def _send_log(self, message: str, log_type: str = "info"):
        """Send log message to dashboard"""
        if self.bridge and self.bridge.connected:
            await self.bridge.send_log(message, log_type)
        # Also print to console
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}][Fabio] {message}")

    def _convert_position(self, pos: Position) -> DashboardPosition:
        """Convert execution engine Position to dashboard format"""
        return DashboardPosition(
            side=pos.side,
            entryPrice=pos.entry_price,
            entryTime=datetime.fromtimestamp(pos.entry_time).isoformat() if pos.entry_time else datetime.now().isoformat(),
            stopLoss=pos.stop_price or 0.0,
            targetTP1=pos.target_price or 0.0,
            targetTP2=pos.target_price or 0.0,  # Fabio doesn't have TP2, use same as TP1
            totalQty=int(pos.size),
            remaining=int(pos.size),
            entryPattern="Fabio LLM Decision",
            unrealizedPnL=pos.unrealized_pnl,
        )

    def _convert_closed_trade(self, trade: dict) -> DashboardTrade:
        """Convert closed trade dict to dashboard format"""
        return DashboardTrade(
            tradeId=trade.get("id", ""),
            side=trade.get("side", "long"),
            entryPrice=trade.get("entry_price", 0.0),
            exitPrice=trade.get("exit_price", 0.0),
            entryTime=trade.get("entry_time", ""),
            exitTime=trade.get("exit_time", ""),
            quantity=trade.get("size", 0.0),
            pnl=trade.get("pnl", 0.0),
            exitReason=trade.get("exit_reason", "closed"),
            entryPattern="Fabio LLM",
        )

    def _should_call_llm(
        self,
        now: float,
        last_call_time: Optional[float],
        importance_zones: list,
        price: float,
    ) -> bool:
        """Decide whether to call LLM (copied from engine.py logic)"""
        if last_call_time is None:
            return True

        default_interval = self.settings.llm_decision_interval_default_sec
        outer_interval = self.settings.llm_decision_interval_outer_band_sec
        inner_interval = self.settings.llm_decision_interval_inner_band_sec

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

    def _build_llm_payload(
        self,
        snapshot: MarketSnapshot,
        feat_state: dict,
        importance_zones: list,
        historical_notes: list,
        strategy_state: dict,
    ) -> dict:
        """Build LLM request payload"""
        return {
            "mode": "live_decision",
            "symbol": self.settings.symbol,
            "timestamp": snapshot.timestamp,
            "session": "Unknown",
            "price": snapshot.bar.close,
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
                for pos in self.execution.positions.values()
            ],
            "importance_zones": importance_zones,
            "recent_performance_summary": {
                "session_pnl": self.execution.current_daily_pnl,
                "session_drawdown": self.execution.current_daily_drawdown,
            },
            "historical_notes_snippet": historical_notes,
            "strategy_state": strategy_state,
        }

    def _load_strategy_state(self) -> dict:
        """Load strategy state from log file"""
        state = {}
        path = self.settings.llm_log_path
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

    def _load_historical_notes(self, max_notes: int = 20) -> list:
        """Load historical notes from log file"""
        notes = []
        path = self.settings.llm_log_path
        if not os.path.exists(path):
            return notes

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("symbol") != self.settings.symbol:
                    continue
                response = record.get("response") or {}
                for note in response.get("notes_to_future_self") or []:
                    if isinstance(note, str):
                        notes.append(note)
        return notes[-max_notes:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fabio LLM agent with dashboard integration"
    )
    parser.add_argument(
        "--mode",
        choices=["live_trading", "paper_trading"],
        default=None,
        help="Trading mode (default from env TRADING_MODE)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Instrument symbol (e.g. NQZ5)",
    )
    parser.add_argument(
        "--dashboard-url",
        type=str,
        default="http://localhost:3337",
        help="Dashboard server URL",
    )
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> Settings:
    """Build settings from CLI args and environment"""
    settings = Settings.from_env()
    if args.mode is not None:
        settings.mode = args.mode  # type: ignore[assignment]
    if args.symbol is not None:
        settings.symbol = args.symbol
    return settings


async def main():
    """Main entry point"""
    args = parse_args()
    settings = build_settings(args)

    integration = FabioDashboardIntegration(
        settings=settings,
        dashboard_url=args.dashboard_url,
    )

    await integration.run()


if __name__ == "__main__":
    asyncio.run(main())
