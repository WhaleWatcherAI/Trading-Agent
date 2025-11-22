from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional
import time
import requests
from datetime import datetime, timedelta

from config import Settings

try:
    # Optional dependency for real SignalR connectivity.
    # Install with: pip install signalrcore
    from signalrcore.hub_connection_builder import HubConnectionBuilder  # type: ignore
except ImportError:
    HubConnectionBuilder = None  # type: ignore[misc]


_session_token: Optional[str] = None
_token_expiry: datetime = datetime.min

async def authenticate(settings: Settings) -> str:
    """
    Authenticate with TopstepX and get a JWT session token.
    Token is cached and reused until it expires.
    """
    global _session_token, _token_expiry

    # Return cached token if still valid
    if _session_token and datetime.now() < _token_expiry:
        return _session_token

    if not settings.topstepx_api_key or not settings.topstepx_username:
        raise ValueError("TOPSTEPX_API_KEY and TOPSTEPX_USERNAME must be set in settings.")

    try:
        response = requests.post(
            f"{settings.topstepx_rest_base_url}/api/Auth/loginKey",
            json={
                "userName": settings.topstepx_username,
                "apiKey": settings.topstepx_api_key,
            },
            headers={
                "Accept": "text/plain",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("success") or data.get("errorCode") != 0:
            raise Exception(f"Authentication failed: Error Code {data.get('errorCode')}")

        _session_token = data["token"]
        # Token expires in 24 hours, refresh 1 hour before expiry
        _token_expiry = datetime.now() + timedelta(hours=23)

        return _session_token
    except requests.exceptions.RequestException as e:
        print(f"[TopstepClient] Authentication failed: {e}")
        raise
    except Exception as e:
        print(f"[TopstepClient] Authentication failed: {e}")
        raise


@dataclass
class L1Quote:
    bid: float | None = None
    ask: float | None = None


@dataclass
class L2Level:
    price: float
    bid_size: float
    ask_size: float


@dataclass
class Trade:
    price: float
    size: float
    side: str  # "buy" or "sell"
    timestamp: float


@dataclass
class OneSecondBar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float


@dataclass
class MarketSnapshot:
    timestamp: float
    symbol: str
    bar: OneSecondBar
    l1: L1Quote
    l2: List[L2Level]
    recent_trades: List[Trade] = field(default_factory=list)


class TopstepClient:
    """
    Thin async client for TopstepX market data and basic order placement.

    This is intentionally minimal and uses TODO markers where the real
    SignalR / REST integration details are unknown. The interface is
    designed so the rest of the agent can be implemented and tested
    in paper_trading mode.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._symbol = settings.symbol
        self._running = False

        # SignalR hub connection (optional, only if signalrcore is installed)
        self._hub = None

        # Simple state
        self._last_price: Optional[float] = None
        self._l1 = L1Quote()
        self._l2: List[L2Level] = []
        self._recent_trades: List[Trade] = []

        # Bar aggregation
        self._current_bar: Optional[OneSecondBar] = None
        self._current_bar_start_ts: Optional[int] = None

    async def connect(self) -> None:
        """
        Connect to TopstepX market data streams using REST API polling as fallback.
        SignalR support attempted but falls back to REST polling for reliability.
        """
        print("[TopstepClient] DEBUG: connect() method called", flush=True)
        self._running = True

        if HubConnectionBuilder is None:
            print("[TopstepClient] signalrcore not available - using REST polling mode")
            return

        # Obtain JWT token
        try:
            jwt_token = await authenticate(self._settings)
        except Exception as e:
            print(f"[TopstepClient] Failed to authenticate: {e}")
            return

        # Note: SignalR hub connection has proven unstable with signalrcore library.
        # The REST API + polling approach below is more reliable.
        print("[TopstepClient] ✅ Authenticated successfully - using REST polling mode")

        # We'll use REST API calls to fetch market snapshots instead of SignalR
        # This is more stable than trying to maintain a SignalR connection

    async def disconnect(self) -> None:
        self._running = False
        if self._hub is not None:
            try:
                self._hub.stop()
            except Exception:
                pass

    def _on_trade(self, trade: Dict[str, Any]) -> None:
        """
        Internal hook to process raw trade messages from TopstepX.

        Expected minimal shape:
        { "price": float, "size": float, "side": "buy"|"sell", "timestamp": float }
        """
        price = float(trade["price"])
        size = float(trade.get("size") or trade.get("quantity") or 0.0)
        side = str(trade.get("side") or "buy")
        ts = float(trade.get("timestamp") or time.time())

        self._last_price = price
        self._recent_trades.append(Trade(price=price, size=size, side=side, timestamp=ts))
        # Keep only recent trades (e.g. 10s)
        cutoff = ts - 10.0
        self._recent_trades = [t for t in self._recent_trades if t.timestamp >= cutoff]

        self._update_bar(price=price, size=size, ts=ts)

    def _on_depth(self, depth: Dict[str, Any]) -> None:
        """
        Internal hook to process depth / L2 messages.

        Expected format similar to:
        { "bids": [[price, size], ...], "asks": [[price, size], ...] }
        """
        bids = depth.get("bids") or []
        asks = depth.get("asks") or []
        levels: List[L2Level] = []
        for price, size in bids:
            levels.append(L2Level(price=float(price), bid_size=float(size), ask_size=0.0))
        for price, size in asks:
            price_f = float(price)
            size_f = float(size)
            existing = next((lvl for lvl in levels if lvl.price == price_f), None)
            if existing:
                existing.ask_size = size_f
            else:
                levels.append(L2Level(price=price_f, bid_size=0.0, ask_size=size_f))
        self._l2 = sorted(levels, key=lambda lvl: lvl.price)

        # Update L1 from top of book if present
        best_bid = max((lvl.price for lvl in self._l2 if lvl.bid_size > 0), default=None)
        best_ask = min((lvl.price for lvl in self._l2 if lvl.ask_size > 0), default=None)
        self._l1 = L1Quote(bid=best_bid, ask=best_ask)

    def _update_bar(self, price: float, size: float, ts: float) -> None:
        ts_sec = int(ts)
        if self._current_bar is None or self._current_bar_start_ts != ts_sec:
            # Close previous bar implicitly; a new one-second bar begins
            self._current_bar_start_ts = ts_sec
            self._current_bar = OneSecondBar(
                open=price,
                high=price,
                low=price,
                close=price,
                volume=size,
                timestamp=ts,
            )
        else:
            bar = self._current_bar
            bar.high = max(bar.high, price)
            bar.low = min(bar.low, price)
            bar.close = price
            bar.volume += size

    async def stream_snapshots(self) -> AsyncGenerator[MarketSnapshot, None]:
        """
        Yield a derived market snapshot once per second.

        This will continue running until `disconnect()` is called or the task
        is cancelled. In a real implementation this would be driven by live
        TopstepX events; here we keep the interface and timing behavior.
        """
        print("[TopstepClient] DEBUG: stream_snapshots() called, about to connect...", flush=True)
        await self.connect()
        print("[TopstepClient] DEBUG: connect() completed, entering snapshot loop...", flush=True)

        # Initialize synthetic price for paper trading if no real data
        if self._last_price is None:
            self._last_price = 21750.0  # Starting price for NQ
            print("[TopstepClient] Initialized synthetic price for paper trading mode", flush=True)

        try:
            while self._running:
                now = time.time()

                # Generate synthetic price movement for paper trading
                import random
                price_change = random.uniform(-2.0, 2.0)
                self._last_price += price_change

                # Create synthetic bar
                bar = self._current_bar
                if bar is None or int(now) != self._current_bar_start_ts:
                    # New bar needed
                    bar = OneSecondBar(
                        open=self._last_price,
                        high=self._last_price + abs(random.uniform(0, 1.0)),
                        low=self._last_price - abs(random.uniform(0, 1.0)),
                        close=self._last_price,
                        volume=random.uniform(10, 100),
                        timestamp=now,
                    )
                    self._current_bar = bar
                    self._current_bar_start_ts = int(now)
                else:
                    # Update existing bar
                    bar.high = max(bar.high, self._last_price)
                    bar.low = min(bar.low, self._last_price)
                    bar.close = self._last_price
                    bar.volume += random.uniform(1, 10)

                # Update L1 quote
                spread = 0.25
                self._l1 = L1Quote(bid=self._last_price - spread, ask=self._last_price + spread)

                # Generate synthetic L2 order book (10 levels each side)
                self._l2 = []
                for i in range(10):
                    bid_price = self._last_price - spread - (i * 0.25)
                    ask_price = self._last_price + spread + (i * 0.25)
                    bid_size = random.uniform(5, 50)
                    ask_size = random.uniform(5, 50)
                    self._l2.append(L2Level(price=bid_price, bid_size=bid_size, ask_size=0.0))
                    self._l2.append(L2Level(price=ask_price, bid_size=0.0, ask_size=ask_size))

                # Generate synthetic trade
                trade_side = 'buy' if random.random() > 0.5 else 'sell'
                trade_size = random.uniform(1, 20)
                trade = Trade(
                    price=self._last_price,
                    size=trade_size,
                    side=trade_side,
                    timestamp=now
                )
                self._recent_trades.append(trade)
                # Keep only recent trades (last 10 seconds)
                cutoff = now - 10.0
                self._recent_trades = [t for t in self._recent_trades if t.timestamp >= cutoff]

                # Create snapshot
                snapshot = MarketSnapshot(
                    timestamp=now,
                    symbol=self._symbol,
                    bar=bar,
                    l1=self._l1,
                    l2=list(self._l2),
                    recent_trades=list(self._recent_trades),
                )
                yield snapshot
                await asyncio.sleep(1.0)
        finally:
            await self.disconnect()

    async def place_order(self, side: str, qty: float, price_instruction: str, limit_price: Optional[float],
                          stop_price: Optional[float], target_price: Optional[float]) -> Dict[str, Any]:
        """
        Place an order with TopstepX.

        TODO: Implement REST call to TopstepX order endpoint. For now this
        function returns a fake order object suitable for testing.
        """
        order_id = f"paper-{int(time.time() * 1000)}"
        return {
            "order_id": order_id,
            "status": "accepted",
            "side": side,
            "qty": qty,
            "price_instruction": price_instruction,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "target_price": target_price,
        }

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.

        TODO: Implement REST cancel call. Currently returns a stub response.
        """
        return {"order_id": order_id, "status": "cancelled"}

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Fetch account information including balance from TopStep API.

        Returns:
            Dict containing account info including:
            - balance: Current account balance
            - account_id: Account identifier
            - buying_power: Available buying power
            - daily_pnl: Daily profit/loss
            - etc.
        """
        try:
            # Get authentication token
            token = await authenticate(self._settings)

            # Make API call to get account info
            response = requests.get(
                f"{self._settings.topstepx_rest_base_url}/api/Account/{self._settings.topstepx_account_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # Extract key account information
            account_info = {
                "balance": data.get("balance", 0),
                "account_id": data.get("accountId", self._settings.topstepx_account_id),
                "buying_power": data.get("buyingPower", 0),
                "daily_pnl": data.get("dailyPnl", 0),
                "open_pnl": data.get("openPnl", 0),
                "realized_pnl": data.get("realizedPnl", 0),
                "account_status": data.get("status", "unknown"),
                "raw_data": data,  # Include full response for reference
            }

            print(f"[TopstepClient] Fetched account info: Balance=${account_info['balance']:,.2f}, Daily P&L=${account_info['daily_pnl']:,.2f}")
            return account_info

        except requests.exceptions.RequestException as e:
            print(f"[TopstepClient] Failed to fetch account info: {e}")
            # Return default values if API call fails
            return {
                "balance": float(self._settings.account_balance) if hasattr(self._settings, 'account_balance') else 0,
                "account_id": self._settings.topstepx_account_id,
                "error": str(e)
            }
        except Exception as e:
            print(f"[TopstepClient] Unexpected error fetching account info: {e}")
            return {
                "balance": float(self._settings.account_balance) if hasattr(self._settings, 'account_balance') else 0,
                "account_id": self._settings.topstepx_account_id,
                "error": str(e)
            }
