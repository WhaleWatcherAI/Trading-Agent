"""
Enhanced Feature Engine for Self-Learning Fabio Agent
Adds POC cross tracking, regime-inference stats, and performance metrics
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple, Any, Optional
import math
import statistics
import time
from datetime import datetime, timedelta

from topstep_client import MarketSnapshot, Trade


@dataclass
class POCCrossTracker:
    """Tracks POC crosses over different time windows"""

    crosses: Deque[float] = field(default_factory=deque)  # Timestamps of crosses
    last_price: Optional[float] = None
    last_poc: Optional[float] = None
    time_near_poc_start: Optional[float] = None
    time_near_poc_total: float = 0.0
    poc_tolerance_ticks: int = 3  # Consider "near POC" within 3 ticks

    def update(self, current_price: float, poc_price: float, timestamp: float, tick_size: float = 0.25) -> None:
        """Update POC cross tracking with new price"""

        # Check for POC cross
        if self.last_price is not None and self.last_poc is not None:
            crossed = (
                (self.last_price <= self.last_poc <= current_price) or
                (self.last_price >= self.last_poc >= current_price)
            )
            if crossed:
                self.crosses.append(timestamp)

        # Track time near POC
        distance_ticks = abs(current_price - poc_price) / tick_size
        if distance_ticks <= self.poc_tolerance_ticks:
            if self.time_near_poc_start is None:
                self.time_near_poc_start = timestamp
        else:
            if self.time_near_poc_start is not None:
                self.time_near_poc_total += timestamp - self.time_near_poc_start
                self.time_near_poc_start = None

        self.last_price = current_price
        self.last_poc = poc_price

        # Clean old crosses (keep last hour)
        cutoff = timestamp - 3600
        while self.crosses and self.crosses[0] < cutoff:
            self.crosses.popleft()

    def get_cross_counts(self, timestamp: float) -> Dict[str, int]:
        """Get POC cross counts for different time windows"""
        counts = {
            "count_last_5min": 0,
            "count_last_15min": 0,
            "count_last_30min": 0,
            "session_total": len(self.crosses)
        }

        for cross_time in self.crosses:
            age_sec = timestamp - cross_time
            if age_sec <= 300:  # 5 min
                counts["count_last_5min"] += 1
            if age_sec <= 900:  # 15 min
                counts["count_last_15min"] += 1
            if age_sec <= 1800:  # 30 min
                counts["count_last_30min"] += 1

        return counts

    def get_time_near_poc(self, timestamp: float, window_sec: int = 1800) -> float:
        """Get time spent near POC in last window"""
        current_time_near = 0.0
        if self.time_near_poc_start is not None:
            current_time_near = timestamp - self.time_near_poc_start

        # For now, return total + current (would need windowing for production)
        return self.time_near_poc_total + current_time_near


@dataclass
class SessionRangeTracker:
    """Tracks session ranges and percentiles"""

    daily_ranges: Deque[float] = field(default_factory=deque)  # Last N day ranges
    max_days: int = 10
    current_session_start: Optional[datetime] = None
    session_high: Optional[float] = None
    session_low: Optional[float] = None
    prev_session_high: Optional[float] = None
    prev_session_low: Optional[float] = None
    prev_session_poc: Optional[float] = None

    def update_price(self, price: float, timestamp: float) -> None:
        """Update session high/low with new price"""
        if self.session_high is None or price > self.session_high:
            self.session_high = price
        if self.session_low is None or price < self.session_low:
            self.session_low = price

    def new_session(self, prev_high: float, prev_low: float, prev_poc: float) -> None:
        """Start new session, saving previous session data"""
        if self.session_high is not None and self.session_low is not None:
            session_range = self.session_high - self.session_low
            self.daily_ranges.append(session_range)
            if len(self.daily_ranges) > self.max_days:
                self.daily_ranges.popleft()

        self.prev_session_high = prev_high
        self.prev_session_low = prev_low
        self.prev_session_poc = prev_poc
        self.session_high = None
        self.session_low = None
        self.current_session_start = datetime.now()

    def get_session_percentile(self) -> Optional[int]:
        """Get current session range as percentile of recent days"""
        if not self.daily_ranges or self.session_high is None or self.session_low is None:
            return None

        current_range = self.session_high - self.session_low
        ranges = list(self.daily_ranges)
        ranges.sort()

        # Find percentile
        below_count = sum(1 for r in ranges if r < current_range)
        percentile = int((below_count / len(ranges)) * 100)
        return percentile


@dataclass
class ValueAreaTracker:
    """Tracks time spent in/out of value area"""

    time_in_value: float = 0.0
    time_above_value: float = 0.0
    time_below_value: float = 0.0
    last_update_time: Optional[float] = None
    last_location: Optional[str] = None  # "inside", "above", "below"

    def update(self, price: float, vah: float, val: float, timestamp: float) -> None:
        """Update time spent in different value zones"""
        if price >= val and price <= vah:
            current_location = "inside"
        elif price > vah:
            current_location = "above"
        else:
            current_location = "below"

        if self.last_update_time is not None:
            time_delta = timestamp - self.last_update_time

            if self.last_location == "inside":
                self.time_in_value += time_delta
            elif self.last_location == "above":
                self.time_above_value += time_delta
            elif self.last_location == "below":
                self.time_below_value += time_delta

        self.last_location = current_location
        self.last_update_time = timestamp

    def get_time_stats(self, window_sec: int = 1800) -> Dict[str, float]:
        """Get time spent in each zone for the window"""
        # Simplified - in production would track windowed values
        total = self.time_in_value + self.time_above_value + self.time_below_value
        if total == 0:
            return {
                "time_in_value_sec": 0,
                "time_above_value_sec": 0,
                "time_below_value_sec": 0
            }

        # Return recent window approximation
        return {
            "time_in_value_sec": min(self.time_in_value, window_sec * 0.5),
            "time_above_value_sec": min(self.time_above_value, window_sec * 0.25),
            "time_below_value_sec": min(self.time_below_value, window_sec * 0.25)
        }


@dataclass
class EnhancedFeatureEngine:
    """
    Enhanced feature engine with POC tracking and market stats for self-learning
    """

    # Original parameters
    window_sec: int = 300
    profile_bucket_size: float = 5.0
    big_trade_threshold: float = 20.0
    tick_size: float = 0.25  # NQ tick size

    # Original buffers
    bars: Deque[Dict[str, Any]] = field(default_factory=deque)
    trades: Deque[Trade] = field(default_factory=deque)

    # Enhanced trackers
    poc_tracker: POCCrossTracker = field(default_factory=POCCrossTracker)
    session_tracker: SessionRangeTracker = field(default_factory=SessionRangeTracker)
    value_tracker: ValueAreaTracker = field(default_factory=ValueAreaTracker)

    # CVD tracking
    cvd_value: float = 0.0
    cvd_history: Deque[Tuple[float, float]] = field(default_factory=deque)

    # Performance tracking (for strategy feedback)
    trade_results: Deque[Dict[str, Any]] = field(default_factory=deque)

    def update_features_and_get_state(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Enhanced update with raw stats for LLM to infer market state
        """
        now = snapshot.timestamp
        bar = snapshot.bar

        # Update all trackers
        self.session_tracker.update_price(bar.close, now)
        self._update_bars(bar, now)
        self._update_trades(snapshot.recent_trades, now)

        # Compute volume profile and levels
        profile = self._compute_volume_profile()
        poc, vah, val, lvns = self._compute_profile_levels(profile)

        # Update POC cross tracker
        self.poc_tracker.update(bar.close, poc, now, self.tick_size)
        poc_cross_stats = self.poc_tracker.get_cross_counts(now)

        # Update value area tracker
        self.value_tracker.update(bar.close, vah, val, now)
        value_time_stats = self.value_tracker.get_time_stats(1800)  # 30min window

        # CVD calculations
        cvd_trend = self._update_cvd(snapshot.recent_trades, now)
        cvd_slope_5min = self._calculate_cvd_slope(300)
        cvd_slope_15min = self._calculate_cvd_slope(900)

        # Absorption and exhaustion
        buy_absorption_score, sell_absorption_score = self._compute_absorption_scores()
        buy_exhaustion_score, sell_exhaustion_score = self._compute_exhaustion_scores()

        # Big prints
        big_prints = self._get_recent_big_prints(now)

        # Average bar ranges
        avg_bar_range_5min = self._calculate_avg_bar_range(300)
        avg_bar_range_15min = self._calculate_avg_bar_range(900)

        # Distance calculations
        distance_to_poc_ticks = abs(bar.close - poc) / self.tick_size if poc else 0
        distance_to_vah_ticks = abs(bar.close - vah) / self.tick_size if vah else 0
        distance_to_val_ticks = abs(bar.close - val) / self.tick_size if val else 0

        # Session metrics
        session_high = self.session_tracker.session_high
        session_low = self.session_tracker.session_low
        session_range = (session_high - session_low) if session_high and session_low else 0
        session_range_percentile = self.session_tracker.get_session_percentile()

        # Build the enhanced state with RAW STATS for LLM to infer regime
        state = {
            # Raw market statistics (no labels!)
            "market_stats": {
                # Session data
                "session_high": session_high,
                "session_low": session_low,
                "session_range": session_range,
                "session_range_vs_10day_percentile": session_range_percentile or 50,

                # Previous session
                "prev_day_high": self.session_tracker.prev_session_high,
                "prev_day_low": self.session_tracker.prev_session_low,
                "prev_day_poc": self.session_tracker.prev_session_poc,

                # Distances
                "distance_to_poc_ticks": distance_to_poc_ticks,
                "distance_to_vah_ticks": distance_to_vah_ticks,
                "distance_to_val_ticks": distance_to_val_ticks,

                # POC crosses
                "poc_cross_stats": poc_cross_stats,
                "time_near_poc_last_30min_sec": self.poc_tracker.get_time_near_poc(now, 1800),

                # Time in/out of value
                "time_in_value_sec_last_30min": value_time_stats["time_in_value_sec"],
                "time_above_value_sec_last_30min": value_time_stats["time_above_value_sec"],
                "time_below_value_sec_last_30min": value_time_stats["time_below_value_sec"],

                # CVD
                "cvd": {
                    "value": self.cvd_value,
                    "slope_5min": cvd_slope_5min,
                    "slope_15min": cvd_slope_15min
                },

                # Volatility
                "volatility": {
                    "avg_bar_range_ticks_5min": avg_bar_range_5min,
                    "avg_bar_range_ticks_15min": avg_bar_range_15min
                }
            },

            # Profile levels (factual data)
            "profiles": [
                {
                    "id": "intraday_profile_01",
                    "type": "session",
                    "poc": poc,
                    "vah": vah,
                    "val": val,
                    "lvns": [{"id": f"lvn_{i}", "price": price} for i, price in enumerate(lvns, start=1)]
                }
            ],

            # Order flow scores (let LLM interpret)
            "orderflow": {
                "cvd_value": self.cvd_value,
                "buy_absorption_score": buy_absorption_score,
                "sell_absorption_score": sell_absorption_score,
                "buy_exhaustion_score": buy_exhaustion_score,
                "sell_exhaustion_score": sell_exhaustion_score,
                "big_prints": [
                    {
                        "side": t.side,
                        "price": t.price,
                        "size": t.size,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t.timestamp))
                    }
                    for t in big_prints
                ]
            }
        }

        return state

    def _calculate_cvd_slope(self, window_sec: int) -> float:
        """Calculate CVD slope over time window"""
        if len(self.cvd_history) < 2:
            return 0.0

        cutoff = time.time() - window_sec
        recent = [(t, v) for t, v in self.cvd_history if t >= cutoff]

        if len(recent) < 2:
            return 0.0

        # Simple linear regression for slope
        n = len(recent)
        sum_x = sum(t for t, _ in recent)
        sum_y = sum(v for _, v in recent)
        sum_xy = sum(t * v for t, v in recent)
        sum_x2 = sum(t * t for t, _ in recent)

        denominator = (n * sum_x2 - sum_x * sum_x)
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def _calculate_avg_bar_range(self, window_sec: int) -> float:
        """Calculate average bar range in ticks over window"""
        if not self.bars:
            return 0.0

        cutoff = time.time() - window_sec
        recent_bars = [b for b in self.bars if b["timestamp"] >= cutoff]

        if not recent_bars:
            return 0.0

        ranges = [(b["high"] - b["low"]) / self.tick_size for b in recent_bars]
        return statistics.mean(ranges) if ranges else 0.0

    def _update_bars(self, bar: Any, now: float) -> None:
        """Update bar buffer"""
        self.bars.append({
            "timestamp": now,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })
        cutoff = now - 3600  # Keep 1 hour
        while self.bars and self.bars[0]["timestamp"] < cutoff:
            self.bars.popleft()

    def _update_trades(self, recent_trades: List[Trade], now: float) -> None:
        """Update trades buffer"""
        for t in recent_trades:
            self.trades.append(t)
        cutoff = now - 3600  # Keep 1 hour
        while self.trades and self.trades[0].timestamp < cutoff:
            self.trades.popleft()

    def _update_cvd(self, recent_trades: List[Trade], now: float) -> str:
        """Update CVD and return trend"""
        for trade in recent_trades:
            if trade.side == "buy":
                self.cvd_value += trade.size
            else:
                self.cvd_value -= trade.size

        self.cvd_history.append((now, self.cvd_value))

        # Keep only recent history
        cutoff = now - 3600
        while self.cvd_history and self.cvd_history[0][0] < cutoff:
            self.cvd_history.popleft()

        # Determine trend
        if len(self.cvd_history) >= 2:
            recent_slope = self._calculate_cvd_slope(300)
            if recent_slope > 0.1:
                return "up"
            elif recent_slope < -0.1:
                return "down"
        return "neutral"

    def _compute_volume_profile(self) -> Dict[float, float]:
        """Compute volume profile from recent trades"""
        profile = {}
        for trade in self.trades:
            bucket = math.floor(trade.price / self.profile_bucket_size) * self.profile_bucket_size
            profile[bucket] = profile.get(bucket, 0) + trade.size
        return profile

    def _compute_profile_levels(self, profile: Dict[float, float]) -> Tuple[float, float, float, List[float]]:
        """Compute POC, VAH, VAL, and LVNs from profile"""
        if not profile:
            return 0, 0, 0, []

        # POC is price with highest volume
        poc = max(profile, key=profile.get)

        # Simple VAH/VAL (70% of volume)
        sorted_buckets = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(profile.values())
        target_volume = total_volume * 0.7

        accumulated = 0
        value_prices = []
        for price, vol in sorted_buckets:
            value_prices.append(price)
            accumulated += vol
            if accumulated >= target_volume:
                break

        vah = max(value_prices) if value_prices else poc
        val = min(value_prices) if value_prices else poc

        # LVNs are low volume areas
        lvns = [p for p, v in profile.items()
                if v < statistics.mean(profile.values()) * 0.5][:3]

        return poc, vah, val, lvns

    def _compute_absorption_scores(self) -> Tuple[float, float]:
        """Compute absorption scores"""
        # Simplified implementation
        return 0.0, 0.0

    def _compute_exhaustion_scores(self) -> Tuple[float, float]:
        """Compute exhaustion scores"""
        # Simplified implementation
        return 0.0, 0.0

    def _get_recent_big_prints(self, now: float) -> List[Trade]:
        """Get recent big prints"""
        cutoff = now - 300  # Last 5 min
        return [t for t in self.trades
                if t.timestamp >= cutoff and t.size >= self.big_trade_threshold]

    def add_trade_result(self, result: Dict[str, Any]) -> None:
        """Add a trade result for performance tracking"""
        self.trade_results.append(result)
        # Keep last 100 trades
        if len(self.trade_results) > 100:
            self.trade_results.popleft()

    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance stats per strategy"""
        if not self.trade_results:
            return {}

        strategy_stats = {}
        for result in self.trade_results:
            strategy = result.get("strategy", "unknown")
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "trades": [],
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0
                }

            stats = strategy_stats[strategy]
            stats["trades"].append(result)

            pnl = result.get("pnl", 0)
            stats["total_pnl"] += pnl
            if pnl > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

        # Calculate final metrics
        performance = {}
        for strategy, stats in strategy_stats.items():
            total_trades = len(stats["trades"])
            if total_trades > 0:
                win_rate = stats["wins"] / total_trades
                avg_pnl = stats["total_pnl"] / total_trades

                # Calculate average R:R if available
                avg_rr = 0.0
                rr_values = [t.get("rr", 0) for t in stats["trades"] if "rr" in t]
                if rr_values:
                    avg_rr = statistics.mean(rr_values)

                performance[strategy] = {
                    "win_rate_last_30": win_rate,
                    "avg_rr_last_30": avg_rr,
                    "net_pnl": stats["total_pnl"],
                    "total_trades": total_trades
                }

        return performance