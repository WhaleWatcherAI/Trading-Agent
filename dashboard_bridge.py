"""
Socket.IO bridge to connect Fabio LLM agent to the existing NQ ICT dashboard.

This module provides a bridge that:
1. Monitors the Fabio agent's execution engine for position/trade updates
2. Emits Socket.IO events to the Node.js dashboard server
3. Allows the existing dashboard UI to display Fabio's trades instead of ICT logic

The Node.js server (live-topstepx-nq-ict.ts) continues to provide:
- Chart data (bars, ticks)
- Market depth / orderbook
- Trade data for CVD calculation

This bridge adds:
- Fabio's position updates
- Fabio's PnL and statistics
- Fabio's activity logs
- Fabio's closed trades
"""

from __future__ import annotations

import asyncio
import socketio
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class DashboardPosition:
    """Position data formatted for the dashboard"""
    side: str  # 'long' or 'short'
    entryPrice: float
    entryTime: str
    stopLoss: float
    targetTP1: float
    targetTP2: float
    totalQty: int
    remaining: int
    entryPattern: str
    unrealizedPnL: float


@dataclass
class DashboardTrade:
    """Closed trade data formatted for the dashboard"""
    tradeId: str
    side: str
    entryPrice: float
    exitPrice: float
    entryTime: str
    exitTime: str
    quantity: float
    pnl: float
    exitReason: str
    entryPattern: str


class DashboardBridge:
    """
    Socket.IO client that connects to the Node.js dashboard server
    and emits Fabio agent's trading data.
    """

    def __init__(self, dashboard_url: str = "http://localhost:3337"):
        """
        Initialize the dashboard bridge.

        Args:
            dashboard_url: URL of the Node.js dashboard server
        """
        self.dashboard_url = dashboard_url
        self.sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_delay=1,
            reconnection_delay_max=5,
            reconnection_attempts=0,  # Infinite
        )
        self.connected = False
        self.symbol = "NQZ5"  # Default, can be overridden
        self.latest_bar = None  # Store latest market bar from dashboard
        self.bar_callback = None  # Optional callback for new bars

        # Setup event handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup Socket.IO event handlers"""

        @self.sio.event
        async def connect():
            self.connected = True
            print(f"[DashboardBridge] ✅ Connected to dashboard at {self.dashboard_url}")
            await self._send_log("Fabio agent connected", "success")

        @self.sio.event
        async def disconnect():
            self.connected = False
            print("[DashboardBridge] ❌ Disconnected from dashboard")

        @self.sio.event
        async def connect_error(data):
            print(f"[DashboardBridge] ⚠️ Connection error: {data}")

        @self.sio.on('bar')
        async def on_bar(bar_data):
            """Receive real-time bar data from dashboard"""
            print(f"[DashboardBridge] 📊 Received bar: close={bar_data.get('close', 'N/A')}", flush=True)
            self.latest_bar = bar_data
            # Call callback if registered
            if self.bar_callback:
                await self.bar_callback(bar_data)

    async def connect(self):
        """Connect to the dashboard server"""
        try:
            print(f"[DashboardBridge] Connecting to {self.dashboard_url}...")
            await self.sio.connect(self.dashboard_url)
            # Don't wait here - let connection stay alive in background
            print(f"[DashboardBridge] Connection initiated")
        except Exception as e:
            print(f"[DashboardBridge] Failed to connect: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the dashboard server"""
        if self.connected:
            await self.sio.disconnect()

    async def emit_status(
        self,
        position: Optional[DashboardPosition],
        closed_trades: List[DashboardTrade],
        total_pnl: float,
        symbol: str = "NQZ5",
        is_trading: bool = True,
    ):
        """
        Emit a status update to the dashboard.

        This matches the format expected by the dashboard's 'status' event handler.

        Args:
            position: Current open position (None if no position)
            closed_trades: List of closed trades
            total_pnl: Total PnL across all trades
            symbol: Trading symbol
            is_trading: Whether the agent is actively trading
        """
        if not self.connected:
            return

        # Calculate statistics
        winners = [t for t in closed_trades if t.pnl > 0]
        losers = [t for t in closed_trades if t.pnl <= 0]
        win_rate = (len(winners) / len(closed_trades) * 100) if closed_trades else 0

        status_data = {
            "symbol": symbol,
            "isTrading": is_trading,
            "position": self._format_position(position) if position else None,
            "closedTrades": [self._format_trade(t) for t in closed_trades[-20:]],  # Last 20 trades
            "accountStats": {
                "totalTrades": len(closed_trades),
                "winners": len(winners),
                "losers": len(losers),
                "winRate": round(win_rate, 1),
                "totalPnL": round(total_pnl, 2),
            },
            "timestamp": datetime.now().isoformat(),
        }

        await self.sio.emit("status", status_data)

    async def emit_trade(self, trade: DashboardTrade):
        """
        Emit a closed trade notification to the dashboard.

        Args:
            trade: Closed trade data
        """
        if not self.connected:
            return

        trade_data = self._format_trade(trade)
        await self.sio.emit("trade", trade_data)

        # Also send a log message
        result = "WIN" if trade.pnl > 0 else "LOSS"
        await self._send_log(
            f"Trade closed: {trade.side.upper()} {result} ${trade.pnl:.2f}",
            "success" if trade.pnl > 0 else "error"
        )

    async def send_log(self, message: str, log_type: str = "info"):
        """
        Send a log message to the dashboard.

        Args:
            message: Log message
            log_type: One of 'info', 'success', 'warning', 'error'
        """
        await self._send_log(message, log_type)

    async def send_llm_prompt(self, payload: Dict[str, Any], timestamp: Optional[float] = None):
        """
        Send an LLM prompt/payload to the dashboard for display.

        Args:
            payload: The LLM request payload (will be JSON-formatted in dashboard)
            timestamp: Optional timestamp (defaults to now)
        """
        if not self.connected:
            return

        prompt_data = {
            "payload": payload,
            "timestamp": timestamp or datetime.now().timestamp(),
            "timestamp_str": datetime.now().isoformat(),
        }

        await self.sio.emit("llm_prompt", prompt_data)

    async def _send_log(self, message: str, log_type: str = "info"):
        """Internal method to send log messages"""
        if not self.connected:
            return

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "type": log_type,
        }
        await self.sio.emit("log", log_data)

    def _format_position(self, position: DashboardPosition) -> Dict[str, Any]:
        """Format position data for the dashboard"""
        return {
            "side": position.side,
            "entryPrice": position.entryPrice,
            "entryTime": position.entryTime,
            "stopLoss": position.stopLoss,
            "targetTP1": position.targetTP1,
            "targetTP2": position.targetTP2,
            "totalQty": position.totalQty,
            "remaining": position.remaining,
            "entryPattern": position.entryPattern,
            "unrealizedPnL": position.unrealizedPnL,
            "contracts": position.totalQty,
        }

    def _format_trade(self, trade: DashboardTrade) -> Dict[str, Any]:
        """Format trade data for the dashboard"""
        return {
            "tradeId": trade.tradeId,
            "side": trade.side,
            "entryPrice": trade.entryPrice,
            "exitPrice": trade.exitPrice,
            "entryTime": trade.entryTime,
            "exitTime": trade.exitTime,
            "quantity": trade.quantity,
            "pnl": trade.pnl,
            "exitReason": trade.exitReason,
            "entryPattern": trade.entryPattern,
        }


# Convenience function to create and run the bridge
async def run_dashboard_bridge(
    dashboard_url: str = "http://localhost:3337",
    update_callback: Optional[callable] = None,
):
    """
    Run the dashboard bridge with optional update callback.

    Args:
        dashboard_url: URL of the Node.js dashboard server
        update_callback: Optional async callback function that receives the bridge instance
                        and can send updates. Should be a coroutine that runs continuously.

    Example:
        async def send_updates(bridge):
            while True:
                # Get data from Fabio agent
                position = get_current_position()
                trades = get_closed_trades()
                pnl = calculate_total_pnl()

                # Send to dashboard
                await bridge.emit_status(position, trades, pnl)
                await asyncio.sleep(1)

        await run_dashboard_bridge(update_callback=send_updates)
    """
    bridge = DashboardBridge(dashboard_url)

    try:
        # Start connection in background
        connect_task = asyncio.create_task(bridge.connect())

        # If callback provided, run it
        if update_callback:
            await update_callback(bridge)
        else:
            # Just keep connection alive
            await connect_task

    except KeyboardInterrupt:
        print("\n[DashboardBridge] Shutting down...")
    finally:
        await bridge.disconnect()


if __name__ == "__main__":
    # Example usage
    async def example_updates(bridge: DashboardBridge):
        """Example update loop"""
        await asyncio.sleep(2)  # Wait for connection

        # Example: Send initial status
        await bridge.send_log("Fabio agent initialized", "info")

        # Example: Simulate position update
        position = DashboardPosition(
            side="long",
            entryPrice=21000.00,
            entryTime=datetime.now().isoformat(),
            stopLoss=20996.00,
            targetTP1=21004.00,
            targetTP2=21008.00,
            totalQty=3,
            remaining=3,
            entryPattern="LLM Decision: Bullish momentum",
            unrealizedPnL=0.0,
        )

        await bridge.emit_status(
            position=position,
            closed_trades=[],
            total_pnl=0.0,
            is_trading=True,
        )

        await bridge.send_log("Position opened: LONG @ 21000.00", "success")

        # Keep running
        while True:
            await asyncio.sleep(5)
            # Update unrealized PnL periodically
            position.unrealizedPnL = 50.0  # Example
            await bridge.emit_status(
                position=position,
                closed_trades=[],
                total_pnl=0.0,
                is_trading=True,
            )

    asyncio.run(run_dashboard_bridge(update_callback=example_updates))
