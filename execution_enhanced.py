"""
Enhanced Execution Engine with Active Strategy Management
Strategy updates from LLM actually affect trading behavior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import uuid

from config import Settings
from topstep_client import TopstepClient


@dataclass
class Position:
    """Track an open position"""
    id: str
    side: str  # "long" or "short"
    entry_price: float
    size: float
    stop_price: float
    target_price: float
    entry_time: float
    strategy: str  # Which strategy opened this
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class EnhancedExecutionEngine:
    """
    Enhanced execution engine that respects LLM strategy updates
    """

    settings: Settings
    topstep_client: TopstepClient
    account_balance: float

    # Core state
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_trades: List[Dict[str, Any]] = field(default_factory=list)

    # Strategy state that LLM can modify
    strategy_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_strategies: List[str] = field(default_factory=list)

    # Risk management
    daily_pnl: float = 0.0
    max_daily_loss: float = -1000.0  # Default $1000 max loss
    max_positions: int = 1
    trades_today: int = 0
    max_trades_per_day: int = 10

    # Performance tracking
    strategy_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def initialize_default_strategies(self):
        """Initialize with default strategy configurations"""
        self.strategy_state = {
            "MeanReversionPOC_v2": {
                "enabled": True,
                "risk_fraction": 0.005,
                "max_trades_per_session": 3,
                "min_rr": 2.0,
                "stop_ticks": 20,
                "target_ticks": 40
            },
            "TrendLVN_Pullback_v1": {
                "enabled": True,
                "risk_fraction": 0.007,
                "max_trades_per_session": 2,
                "min_rr": 2.5,
                "stop_ticks": 15,
                "target_ticks": 38
            },
            "AbsorptionFade_v1": {
                "enabled": False,  # Disabled by default
                "risk_fraction": 0.003,
                "max_trades_per_session": 2,
                "min_rr": 1.5,
                "stop_ticks": 10,
                "target_ticks": 15
            }
        }

        # Initialize active strategies
        self.active_strategies = [
            name for name, config in self.strategy_state.items()
            if config.get("enabled", False)
        ]

    def apply_strategy_updates(self, updates: Dict[str, Any]):
        """
        Apply strategy updates from LLM - THIS IS THE KEY SELF-LEARNING MECHANISM
        """
        if not updates:
            return

        # Update active strategies list
        if "active_strategies" in updates:
            self.active_strategies = updates["active_strategies"]
            print(f"[Execution] Active strategies updated: {self.active_strategies}")

        # Apply individual strategy tweaks
        for tweak in updates.get("strategy_tweaks", []):
            strategy_name = tweak.get("name")
            changes = tweak.get("changes", {})
            reason = tweak.get("reason", "No reason provided")

            if strategy_name and changes:
                if strategy_name not in self.strategy_state:
                    self.strategy_state[strategy_name] = {}

                # Apply changes
                self.strategy_state[strategy_name].update(changes)

                # Update active list if enabled flag changed
                if "enabled" in changes:
                    if changes["enabled"] and strategy_name not in self.active_strategies:
                        self.active_strategies.append(strategy_name)
                    elif not changes["enabled"] and strategy_name in self.active_strategies:
                        self.active_strategies.remove(strategy_name)

                print(f"[Execution] Strategy '{strategy_name}' updated: {changes}")
                print(f"  Reason: {reason}")

    def can_trade(self, strategy_name: str) -> tuple[bool, str]:
        """
        Check if we can take a trade based on strategy state and risk rules
        """
        # Check if strategy is enabled
        if strategy_name not in self.active_strategies:
            return False, f"Strategy {strategy_name} is disabled"

        strategy_config = self.strategy_state.get(strategy_name, {})
        if not strategy_config.get("enabled", False):
            return False, f"Strategy {strategy_name} is not enabled"

        # Check daily loss limit
        if self.daily_pnl <= self.max_daily_loss:
            return False, f"Daily loss limit reached: {self.daily_pnl}"

        # Check max positions
        if len(self.positions) >= self.max_positions:
            return False, f"Max positions reached: {len(self.positions)}"

        # Check max trades per day
        if self.trades_today >= self.max_trades_per_day:
            return False, f"Max trades per day reached: {self.trades_today}"

        # Check strategy-specific trade limit
        strategy_trades_today = sum(
            1 for trade in self.closed_trades
            if trade.get("strategy") == strategy_name
            and trade.get("close_time", 0) > time.time() - 86400
        )

        max_strategy_trades = strategy_config.get("max_trades_per_session", 5)
        if strategy_trades_today >= max_strategy_trades:
            return False, f"Strategy {strategy_name} max trades reached: {strategy_trades_today}"

        return True, "OK"

    async def process_trade_decision(
        self,
        decision: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Process a trade decision from the LLM with strategy state enforcement
        """
        action = decision.get("action")
        strategy_name = decision.get("entry_condition_id", "unknown")

        if action == "enter":
            # Check if we can trade with this strategy
            can_trade, reason = self.can_trade(strategy_name)
            if not can_trade:
                return {
                    "status": "rejected",
                    "reason": reason,
                    "strategy": strategy_name
                }

            # Get strategy configuration
            strategy_config = self.strategy_state.get(strategy_name, {})

            # Calculate position size based on strategy risk
            risk_fraction = strategy_config.get("risk_fraction", 0.005)
            stop_price = decision.get("stop_price", current_price - 20)

            risk_amount = self.account_balance * risk_fraction
            stop_distance = abs(current_price - stop_price)
            tick_value = 5.0  # $5 per tick for MNQ

            position_size = max(1, int(risk_amount / (stop_distance * tick_value)))

            # Ensure R:R meets strategy minimum
            target_price = decision.get("target_price", current_price + 40)
            target_distance = abs(target_price - current_price)
            rr_ratio = target_distance / stop_distance if stop_distance > 0 else 0

            min_rr = strategy_config.get("min_rr", 1.5)
            if rr_ratio < min_rr:
                return {
                    "status": "rejected",
                    "reason": f"R:R {rr_ratio:.1f} below minimum {min_rr}",
                    "strategy": strategy_name
                }

            # Create position
            position_id = str(uuid.uuid4())[:8]
            position = Position(
                id=position_id,
                side=decision.get("side", "long"),
                entry_price=current_price,
                size=position_size,
                stop_price=stop_price,
                target_price=target_price,
                entry_time=time.time(),
                strategy=strategy_name
            )

            # Execute order (simplified)
            try:
                # This would actually call TopstepClient
                # order = await self.topstep_client.place_order(...)

                self.positions[position_id] = position
                self.trades_today += 1

                # Track strategy usage
                if strategy_name not in self.strategy_performance:
                    self.strategy_performance[strategy_name] = {
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "total_pnl": 0.0
                    }
                self.strategy_performance[strategy_name]["trades"] += 1

                return {
                    "status": "executed",
                    "position_id": position_id,
                    "strategy": strategy_name,
                    "size": position_size,
                    "entry_price": current_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "risk_amount": risk_amount,
                    "rr_ratio": rr_ratio
                }

            except Exception as e:
                return {
                    "status": "error",
                    "reason": str(e),
                    "strategy": strategy_name
                }

        elif action == "exit":
            # Handle position exit
            for pos_id, pos in list(self.positions.items()):
                self._close_position(pos_id, current_price, "manual_exit")

            return {"status": "positions_closed"}

        elif action == "modify":
            # Handle stop modification (only tightening allowed)
            for pos_id, pos in self.positions.items():
                new_stop = decision.get("stop_price")
                if new_stop:
                    if pos.side == "long" and new_stop > pos.stop_price:
                        pos.stop_price = new_stop
                    elif pos.side == "short" and new_stop < pos.stop_price:
                        pos.stop_price = new_stop

            return {"status": "stops_modified"}

        return {"status": "no_action"}

    def _close_position(self, position_id: str, exit_price: float, reason: str):
        """Close a position and record results"""
        if position_id not in self.positions:
            return

        pos = self.positions[position_id]

        # Calculate PnL
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.size * 5  # $5 per tick MNQ
        else:
            pnl = (pos.entry_price - exit_price) * pos.size * 5

        # Update daily PnL
        self.daily_pnl += pnl

        # Update strategy performance
        strategy = pos.strategy
        if strategy in self.strategy_performance:
            stats = self.strategy_performance[strategy]
            stats["total_pnl"] += pnl
            if pnl > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

        # Record closed trade
        self.closed_trades.append({
            "position_id": position_id,
            "strategy": strategy,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "size": pos.size,
            "pnl": pnl,
            "entry_time": pos.entry_time,
            "close_time": time.time(),
            "reason": reason,
            "rr": (exit_price - pos.entry_price) / (pos.entry_price - pos.stop_price)
                  if pos.side == "long" and pos.entry_price != pos.stop_price
                  else (pos.entry_price - exit_price) / (pos.stop_price - pos.entry_price)
                  if pos.side == "short" and pos.stop_price != pos.entry_price
                  else 0
        })

        # Remove from active positions
        del self.positions[position_id]

        print(f"[Execution] Position {position_id} closed: PnL ${pnl:.2f}, Reason: {reason}")

    def check_stops_and_targets(self, current_price: float):
        """Check if any positions hit stops or targets"""
        for pos_id, pos in list(self.positions.items()):
            if pos.side == "long":
                if current_price <= pos.stop_price:
                    self._close_position(pos_id, pos.stop_price, "stop_hit")
                elif current_price >= pos.target_price:
                    self._close_position(pos_id, pos.target_price, "target_hit")
            else:  # short
                if current_price >= pos.stop_price:
                    self._close_position(pos_id, pos.stop_price, "stop_hit")
                elif current_price <= pos.target_price:
                    self._close_position(pos_id, pos.target_price, "target_hit")

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for each strategy"""
        stats = {}

        for strategy, perf in self.strategy_performance.items():
            total_trades = perf["trades"]
            if total_trades > 0:
                win_rate = perf["wins"] / total_trades
                stats[strategy] = {
                    "win_rate_last_30": win_rate,
                    "total_trades": total_trades,
                    "net_pnl": perf["total_pnl"],
                    "avg_pnl": perf["total_pnl"] / total_trades
                }

        return stats

    def reset_daily_stats(self):
        """Reset daily statistics (call at session start)"""
        self.daily_pnl = 0.0
        self.trades_today = 0
        print("[Execution] Daily stats reset")