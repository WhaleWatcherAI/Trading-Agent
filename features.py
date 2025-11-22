from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple, Any
import math
import statistics
import time

from topstep_client import MarketSnapshot, Trade


@dataclass
class VolumeProfileBucket:
    price: float
    volume: float = 0.0


def _bucket_price(price: float, bucket_size: float) -> float:
    return math.floor(price / bucket_size) * bucket_size


@dataclass
class FeatureEngine:
    """
    Real-time feature engine for the LLM trader agent.

    Maintains rolling buffers of bars and trades and exposes a single
    update_features_and_get_state() entry point.
    """

    window_sec: int = 300
    profile_bucket_size: float = 5.0  # price units per bucket
    big_trade_threshold: float = 20.0

    bars: Deque[Dict[str, Any]] = field(default_factory=deque)
    trades: Deque[Trade] = field(default_factory=deque)

    # session metrics
    session_high: float | None = None
    session_low: float | None = None

    # CVD
    cvd_value: float = 0.0
    cvd_history: Deque[Tuple[float, float]] = field(default_factory=deque)  # (timestamp, cvd)

    def update_features_and_get_state(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Update rolling buffers with the latest snapshot and return a dictionary
        of derived features suitable for sending to the LLM.
        """
        now = snapshot.timestamp
        bar = snapshot.bar

        self._update_session_range(bar.close)
        self._update_bars(bar, now)
        self._update_trades(snapshot.recent_trades, now)

        profile = self._compute_volume_profile()
        poc, vah, val, lvns = self._compute_profile_levels(profile)

        cvd_trend = self._update_cvd(snapshot.recent_trades, now)

        buy_absorption_score, sell_absorption_score = self._compute_absorption_scores()
        buy_exhaustion_score, sell_exhaustion_score = self._compute_exhaustion_scores()

        big_prints = self._get_recent_big_prints(now)

        market_state_flag, range_condition = self._classify_market_state(poc, vah, val)
        location_vs_value, location_vs_poc = self._classify_location(snapshot.bar.close, poc, vah, val)

        buyers_in_control_score, sellers_in_control_score = self._compute_control_scores(
            cvd_trend, buy_absorption_score, sell_absorption_score,
            buy_exhaustion_score, sell_exhaustion_score, big_prints,
        )

        state = {
            "profiles": [
                {
                    "id": "intraday_profile_01",
                    "type": "session",
                    "poc": poc,
                    "vah": vah,
                    "val": val,
                    "lvns": [{"id": f"lvn_{i}", "price": price} for i, price in enumerate(lvns, start=1)],
                }
            ],
            "derived_state": {
                "market_state_flag": market_state_flag,
                "range_condition": range_condition,
                "location_vs_value": location_vs_value,
                "location_vs_poc": location_vs_poc,
                "buyers_in_control_score": buyers_in_control_score,
                "sellers_in_control_score": sellers_in_control_score,
            },
            "orderflow": {
                "cvd_trend": cvd_trend,
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
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t.timestamp)),
                    }
                    for t in big_prints
                ],
            },
            "session_metrics": {
                "session_high": self.session_high,
                "session_low": self.session_low,
                "session_range": (
                    (self.session_high - self.session_low)
                    if self.session_high is not None and self.session_low is not None
                    else None
                ),
            },
        }

        return state

    # Internal helpers

    def _update_session_range(self, price: float) -> None:
        if self.session_high is None or price > self.session_high:
            self.session_high = price
        if self.session_low is None or price < self.session_low:
            self.session_low = price

    def _update_bars(self, bar: Any, now: float) -> None:
        self.bars.append(
            {
                "timestamp": now,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )
        cutoff = now - self.window_sec
        while self.bars and self.bars[0]["timestamp"] < cutoff:
            self.bars.popleft()

    def _update_trades(self, recent_trades: List[Trade], now: float) -> None:
        for t in recent_trades:
            self.trades.append(t)
        cutoff = now - self.window_sec
        while self.trades and self.trades[0].timestamp < cutoff:
            self.trades.popleft()

    def _compute_volume_profile(self) -> Dict[float, VolumeProfileBucket]:
        profile: Dict[float, VolumeProfileBucket] = {}
        for t in self.trades:
            key = _bucket_price(t.price, self.profile_bucket_size)
            bucket = profile.get(key)
            if not bucket:
                bucket = VolumeProfileBucket(price=key)
                profile[key] = bucket
            bucket.volume += t.size
        return profile

    def _compute_profile_levels(
        self, profile: Dict[float, VolumeProfileBucket]
    ) -> tuple[float | None, float | None, float | None, List[float]]:
        if not profile:
            return None, None, None, []

        buckets = sorted(profile.values(), key=lambda b: b.price)
        volumes = [b.volume for b in buckets]
        total_vol = sum(volumes)
        if total_vol <= 0:
            return None, None, None, []

        poc_bucket = max(buckets, key=lambda b: b.volume)
        poc_price = poc_bucket.price

        # Simple 70% value area around POC: expand outward until volume >= 70%
        target_vol = total_vol * 0.7
        cum_vol = poc_bucket.volume
        left_idx = right_idx = buckets.index(poc_bucket)
        while cum_vol < target_vol and (left_idx > 0 or right_idx < len(buckets) - 1):
            left_vol = buckets[left_idx - 1].volume if left_idx > 0 else 0
            right_vol = buckets[right_idx + 1].volume if right_idx < len(buckets) - 1 else 0
            if right_vol >= left_vol and right_idx < len(buckets) - 1:
                right_idx += 1
                cum_vol += buckets[right_idx].volume
            elif left_idx > 0:
                left_idx -= 1
                cum_vol += buckets[left_idx].volume
            else:
                break
        val_price = buckets[left_idx].price
        vah_price = buckets[right_idx].price

        # LVNs: local minima of volume between neighbors
        lvns: List[float] = []
        for i in range(1, len(buckets) - 1):
            prev_v = buckets[i - 1].volume
            cur_v = buckets[i].volume
            next_v = buckets[i + 1].volume
            if cur_v < prev_v and cur_v < next_v:
                lvns.append(buckets[i].price)

        return poc_price, vah_price, val_price, lvns

    def _update_cvd(self, trades: List[Trade], now: float) -> str:
        for t in trades:
            if t.side == "buy":
                self.cvd_value += t.size
            else:
                self.cvd_value -= t.size
        self.cvd_history.append((now, self.cvd_value))
        cutoff = now - self.window_sec
        while self.cvd_history and self.cvd_history[0][0] < cutoff:
            self.cvd_history.popleft()

        # Compute simple trend from last few points
        if len(self.cvd_history) < 3:
            return "flat"
        times, values = zip(*self.cvd_history)
        first, last = values[0], values[-1]
        diff = last - first
        if abs(diff) < max(10.0, 0.01 * abs(first)):
            return "flat"
        return "up" if diff > 0 else "down"

    def _compute_absorption_scores(self) -> tuple[float, float]:
        """
        Crude absorption: high volume with very small net price change.
        """
        if not self.bars:
            return 0.0, 0.0

        recent = list(self.bars)[-30:]  # last ~30 seconds
        total_vol = sum(b["volume"] for b in recent)
        if total_vol <= 0:
            return 0.0, 0.0

        highs = [b["high"] for b in recent]
        lows = [b["low"] for b in recent]
        price_range = max(highs) - min(lows)
        if price_range <= 0:
            return 0.0, 0.0

        avg_bar_body = statistics.mean(abs(b["close"] - b["open"]) for b in recent)

        # Normalize scores between 0 and 1
        absorption_score = min(1.0, (total_vol / max(1.0, price_range * 10.0)) * (1.0 / (1.0 + avg_bar_body)))
        # Directional split: simple heuristic from last close vs first close
        first_close = recent[0]["close"]
        last_close = recent[-1]["close"]
        if last_close >= first_close:
            return absorption_score, absorption_score * 0.2
        return absorption_score * 0.2, absorption_score

    def _compute_exhaustion_scores(self) -> tuple[float, float]:
        """
        Crude exhaustion: price extension with shrinking volume vs prior window.
        """
        if len(self.bars) < 40:
            return 0.0, 0.0
        recent = list(self.bars)
        prev_window = recent[-40:-20]
        curr_window = recent[-20:]

        prev_vol = sum(b["volume"] for b in prev_window)
        curr_vol = sum(b["volume"] for b in curr_window)
        if prev_vol <= 0:
            return 0.0, 0.0

        volume_ratio = curr_vol / prev_vol

        # Direction from price
        prev_close = prev_window[-1]["close"]
        curr_close = curr_window[-1]["close"]

        if volume_ratio < 0.5:
            if curr_close > prev_close:
                # buying exhaustion at highs
                return 0.0, min(1.0, (0.5 - volume_ratio) * 2.0)
            else:
                # selling exhaustion at lows
                return min(1.0, (0.5 - volume_ratio) * 2.0), 0.0
        return 0.0, 0.0

    def _get_recent_big_prints(self, now: float) -> List[Trade]:
        cutoff = now - 60.0
        return [
            t
            for t in self.trades
            if t.timestamp >= cutoff and t.size >= self.big_trade_threshold
        ]

    def _classify_market_state(
        self, poc: float | None, vah: float | None, val: float | None
    ) -> tuple[str, str]:
        if poc is None or not self.bars:
            return "balanced", "range"
        last = self.bars[-1]["close"]
        distance_from_poc = abs(last - poc)

        if vah is not None and val is not None:
            value_range = vah - val
        else:
            value_range = max(1.0, 10.0)

        if distance_from_poc > value_range * 0.75:
            # out of balance
            if last > poc:
                return "out_of_balance_up", "trend"
            return "out_of_balance_down", "trend"

        # inside value – decide between range vs chop based on volatility
        closes = [b["close"] for b in self.bars]
        if len(closes) >= 10:
            recent_range = max(closes) - min(closes)
            if recent_range < value_range * 0.25:
                return "balanced", "chop"
        return "balanced", "range"

    def _classify_location(
        self, price: float, poc: float | None, vah: float | None, val: float | None
    ) -> tuple[str, str]:
        if poc is None or vah is None or val is None:
            return "unknown", "unknown"

        if price > vah:
            location_vs_value = "above_value"
        elif price < val:
            location_vs_value = "below_value"
        else:
            location_vs_value = "inside_value"

        if price > poc:
            location_vs_poc = "above_poc"
        elif price < poc:
            location_vs_poc = "below_poc"
        else:
            location_vs_poc = "at_poc"

        return location_vs_value, location_vs_poc

    def _compute_control_scores(
        self,
        cvd_trend: str,
        buy_absorption_score: float,
        sell_absorption_score: float,
        buy_exhaustion_score: float,
        sell_exhaustion_score: float,
        big_prints: List[Trade],
    ) -> tuple[float, float]:
        buyers = 0.0
        sellers = 0.0

        if cvd_trend == "up":
            buyers += 0.4
        elif cvd_trend == "down":
            sellers += 0.4

        buyers += max(0.0, 0.3 - buy_exhaustion_score)
        sellers += max(0.0, 0.3 - sell_exhaustion_score)

        buyers += buy_absorption_score * 0.2
        sellers += sell_absorption_score * 0.2

        buy_big = sum(t.size for t in big_prints if t.side == "buy")
        sell_big = sum(t.size for t in big_prints if t.side == "sell")
        total_big = buy_big + sell_big
        if total_big > 0:
            buyers += (buy_big / total_big) * 0.1
            sellers += (sell_big / total_big) * 0.1

        return min(1.0, buyers), min(1.0, sellers)

