from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import Settings
from topstep_client import TopstepClient


@dataclass
class Position:
    id: str
    side: str  # "long" | "short"
    entry_price: float
    size: float
    stop_price: float
    target_price: float
    unrealized_pnl: float = 0.0


@dataclass
class ExecutionEngine:
    """
    Applies LLM trade decisions with risk management and live/paper modes.
    """

    settings: Settings
    topstep_client: TopstepClient
    account_balance: float
    current_daily_pnl: float = 0.0
    current_daily_drawdown: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    session_trade_counts: Dict[str, int] = field(default_factory=dict)

    def _clamped_risk_fraction(self, requested: Optional[float]) -> float:
        base = self.settings.risk_per_trade_fraction
        if requested is not None:
            base = max(self.settings.min_risk_fraction, min(requested, self.settings.max_risk_fraction))
        return base

    def _can_open_new_trade(self) -> bool:
        max_daily_loss = self.settings.max_daily_loss_fraction * self.account_balance
        return self.current_daily_drawdown < max_daily_loss

    async def apply_trade_decisions(self, decisions: List[Dict[str, Any]], current_price: float,
                                    strategy_state: Dict[str, Dict[str, Any]]) -> None:
        """
        Apply a list of trade_decisions from the LLM.
        """
        for decision in decisions:
            action = decision.get("action", "no_action")
            if action == "no_action":
                continue

            side = decision.get("side")
            position_id = decision.get("position_id")
            risk_fraction = self._clamped_risk_fraction(decision.get("risk_fraction"))
            entry_price = current_price
            stop_price = decision.get("stop_price")
            target_price = decision.get("target_price")
            strategy_name = decision.get("strategy_name") or decision.get("entry_condition_id", "default")

            if not self._can_open_new_trade() and action == "enter":
                print("[EXEC] Daily loss limit reached, rejecting new entry.")
                continue

            if action == "enter":
                await self._enter_position(
                    side=side,
                    risk_fraction=risk_fraction,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    strategy_name=strategy_name,
                    strategy_state=strategy_state,
                )
            elif action in ("manage", "exit") and position_id:
                await self._manage_or_exit_position(
                    position_id=position_id,
                    action=action,
                    new_stop=stop_price,
                    new_target=target_price,
                )

    async def _enter_position(
        self,
        side: str,
        risk_fraction: float,
        entry_price: float,
        stop_price: Optional[float],
        target_price: Optional[float],
        strategy_name: str,
        strategy_state: Dict[str, Dict[str, Any]],
    ) -> None:
        cfg = strategy_state.get(strategy_name, {})
        if cfg.get("enabled", True) is False:
            print(f"[EXEC] Strategy {strategy_name} disabled, skipping entry.")
            return

        max_trades = cfg.get("max_trades_per_session")
        count = self.session_trade_counts.get(strategy_name, 0)
        if max_trades is not None and count >= max_trades:
            print(f"[EXEC] Strategy {strategy_name} max trades per session reached, skipping entry.")
            return

        if stop_price is None or entry_price == stop_price:
            print("[EXEC] Invalid stop price, cannot compute size.")
            return

        tick_value = cfg.get("tick_value", 5.0)
        tick_size = cfg.get("tick_size", 0.25)
        risk_per_trade_dollars = self.account_balance * risk_fraction
        risk_per_contract = abs(entry_price - stop_price) / tick_size * tick_value
        if risk_per_contract <= 0:
            print("[EXEC] Non-positive risk per contract, skipping.")
            return
        size = max(1.0, risk_per_trade_dollars / risk_per_contract)

        if self.settings.mode == "live_trading":
            order = await self.topstep_client.place_order(
                side=side,
                qty=size,
                price_instruction="market",
                limit_price=None,
                stop_price=stop_price,
                target_price=target_price,
            )
            order_id = order.get("order_id", f"live-{len(self.positions)+1}")
        else:
            order_id = f"paper-{len(self.positions)+1}"

        pos = Position(
            id=order_id,
            side=side,
            entry_price=entry_price,
            size=size,
            stop_price=stop_price,
            target_price=target_price or entry_price,
        )
        self.positions[order_id] = pos
        self.session_trade_counts[strategy_name] = count + 1

        print(f"[EXEC] Opened {side} position {order_id} size={size:.2f} at {entry_price:.2f}")

    async def _manage_or_exit_position(
        self,
        position_id: str,
        action: str,
        new_stop: Optional[float],
        new_target: Optional[float],
    ) -> None:
        pos = self.positions.get(position_id)
        if not pos:
            print(f"[EXEC] No such position {position_id}")
            return

        if action == "exit":
            await self._close_position(position_id, reason="llm_exit")
            return

        # Manage stop: never widen
        if new_stop is not None:
            if pos.side == "long":
                if new_stop > pos.stop_price:
                    pos.stop_price = new_stop
            else:
                if new_stop < pos.stop_price:
                    pos.stop_price = new_stop

        if new_target is not None:
            pos.target_price = new_target

    async def _close_position(self, position_id: str, reason: str) -> None:
        pos = self.positions.pop(position_id, None)
        if not pos:
            return
        print(f"[EXEC] Closed position {position_id} ({reason})")

