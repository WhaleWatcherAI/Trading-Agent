#!/usr/bin/env python3
"""
Rule-based backtest of the Volume Profile strategy (no PPO) using a full
intraday session profile (6:00pm ET → 5:00pm ET next day).

Volume Profile is recomputed each bar using all bars in the current session
(robust per-session profile, not a rolling window).

Rules (single position at a time):
- Entry:
    * Mean reversion long near VAL (price_position <= 0.2)
    * Mean reversion short near VAH (price_position >= 0.8)
    * Continuation long above VAH (price_position > 1.0)
    * Continuation short below VAL (price_position < 0.0)
- Exit:
    * Mean reversion: take profit at POC band (0.45-0.55) or max_hold bars.
    * Continuation: exit on re-entry into value area (0-1) or max_hold bars.
    * Hard stop if price crosses opposite extreme (long below -0.1, short above 1.1).

PnL:
- Tick-accurate PnL with tick_size, tick_value, commissions, contracts.
- Default values suit NQ: tick_size=0.25, tick_value=$5/tick.
- Override via env: TICK_SIZE, TICK_VALUE, COMMISSION_PER_SIDE, CONTRACTS, INITIAL_EQUITY, DATA_PATH.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class VPConfig:
    num_price_levels: int = 120
    value_area_pct: float = 0.70
    max_value_area_iterations: int = 150
    max_hold_bars: int = 30
    tick_size: float = 0.25           # Override via TICK_SIZE env (e.g., 0.25 for NQ)
    tick_value: float = 5.0           # $ per tick (e.g., $5 for NQ)
    commission_per_side: float = 0.0  # $ per side per contract
    contracts: int = 1                # Contracts per trade
    initial_equity: float = 100_000.0 # For return calculation


def _robust_profile(session_df: pd.DataFrame, cfg: VPConfig) -> Tuple[float, float, float, float]:
    """Compute POC/VAH/VAL and price_position using all bars in the current session."""
    if len(session_df) < 20:
        return 0.5, 0.6, 0.4, 0.5

    high = session_df["high"].max()
    low = session_df["low"].min()
    price_range = high - low
    if price_range <= 0:
        return 0.5, 0.6, 0.4, 0.5

    # Price levels and bin edges for proportional volume allocation
    levels = np.linspace(low, high, cfg.num_price_levels)
    edges = np.linspace(low, high, cfg.num_price_levels + 1)
    vp = np.zeros(cfg.num_price_levels)

    for _, bar in session_df.iterrows():
        bar_low, bar_high, bar_vol = bar["low"], bar["high"], bar["volume"]
        if bar_high <= bar_low or bar_vol <= 0:
            continue
        # Contribution proportional to overlap length with each price bin
        overlap_low = np.maximum(edges[:-1], bar_low)
        overlap_high = np.minimum(edges[1:], bar_high)
        overlap = np.maximum(0, overlap_high - overlap_low)
        total_overlap = (bar_high - bar_low)
        if total_overlap <= 0:
            continue
        vp += bar_vol * (overlap / total_overlap)

    if vp.sum() == 0:
        return 0.5, 0.6, 0.4, 0.5

    poc_idx = int(np.argmax(vp))
    poc_price = levels[poc_idx]

    target_volume = vp.sum() * cfg.value_area_pct
    value_area = {poc_idx}
    captured = vp[poc_idx]
    iterations = 0
    while captured < target_volume and iterations < cfg.max_value_area_iterations:
        iterations += 1
        min_idx = min(value_area)
        max_idx = max(value_area)
        up_vol = vp[max_idx + 1] if max_idx < len(vp) - 1 else 0
        down_vol = vp[min_idx - 1] if min_idx > 0 else 0
        if up_vol >= down_vol and max_idx < len(vp) - 1:
            value_area.add(max_idx + 1)
            captured += up_vol
        elif min_idx > 0:
            value_area.add(min_idx - 1)
            captured += down_vol
        else:
            break

    vah_price = levels[max(value_area)]
    val_price = levels[min(value_area)]

    price = session_df["close"].iloc[-1]
    vaw = vah_price - val_price
    if vaw <= 0:
        return 0.5, 0.6, 0.4, 0.5

    if price < val_price:
        price_position = (price - val_price) / vaw
    elif price > vah_price:
        price_position = (price - val_price) / vaw
    else:
        price_position = (price - val_price) / vaw

    poc_norm = (poc_price - low) / price_range
    vah_norm = (vah_price - low) / price_range
    val_norm = (val_price - low) / price_range
    return float(poc_norm), float(vah_norm), float(val_norm), float(price_position)


def backtest(df: pd.DataFrame, cfg: VPConfig):
    equity = cfg.initial_equity
    trades: List[dict] = []
    position = 0  # 0 flat, 1 long, -1 short
    entry_price = 0.0
    entry_idx = 0
    trade_type = ""
    session_id_prev = None
    session_start_idx = 0

    for i in range(len(df)):
        ts_local = df["_ts_local"].iloc[i]
        session_id = ts_local.date() if ts_local.hour >= 18 else (ts_local - pd.Timedelta(days=1)).date()
        if session_id != session_id_prev:
            session_start_idx = i
            session_id_prev = session_id

        session_df = df.iloc[session_start_idx : i + 1]
        poc, vah, val, pos = _robust_profile(session_df, cfg)
        price = df.iloc[i]["close"]

        if position == 0:
            # Entry signals
            if pos < 0:
                position, trade_type = -1, "breakout_short"
                entry_price, entry_idx = price, i
            elif pos > 1:
                position, trade_type = 1, "breakout_long"
                entry_price, entry_idx = price, i
            elif pos <= 0.2:
                position, trade_type = 1, "meanrev_val"
                entry_price, entry_idx = price, i
            elif pos >= 0.8:
                position, trade_type = -1, "meanrev_vah"
                entry_price, entry_idx = price, i
            else:
                continue
        else:
            hold = i - entry_idx
            exit_reason = None
            if trade_type.startswith("meanrev"):
                if 0.45 <= pos <= 0.55:
                    exit_reason = "target_poc"
            if trade_type == "breakout_long" and pos < 1:
                exit_reason = exit_reason or "reenter_value"
            if trade_type == "breakout_short" and pos > 0:
                exit_reason = exit_reason or "reenter_value"
            if position == 1 and pos < -0.1:
                exit_reason = exit_reason or "hard_stop"
            if position == -1 and pos > 1.1:
                exit_reason = exit_reason or "hard_stop"
            if hold >= cfg.max_hold_bars and exit_reason is None:
                exit_reason = "time"

            if exit_reason:
                ticks_gained = (price - entry_price) / cfg.tick_size
                ticks_gained = ticks_gained if position == 1 else -ticks_gained
                gross = ticks_gained * cfg.tick_value * cfg.contracts
                fees = cfg.commission_per_side * 2 * cfg.contracts
                net = gross - fees
                equity += net
                trades.append(
                    {
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "side": "long" if position == 1 else "short",
                        "type": trade_type,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "ticks": ticks_gained,
                        "gross_pnl": gross,
                        "net_pnl": net,
                        "exit_reason": exit_reason,
                    }
                )
                position = 0
                trade_type = ""

    # Close any open position at the end
    if position != 0:
        price = df.iloc[-1]["close"]
        ticks_gained = (price - entry_price) / cfg.tick_size
        ticks_gained = ticks_gained if position == 1 else -ticks_gained
        gross = ticks_gained * cfg.tick_value * cfg.contracts
        fees = cfg.commission_per_side * 2 * cfg.contracts
        net = gross - fees
        equity += net
        trades.append(
            {
                "entry_idx": entry_idx,
                "exit_idx": len(df) - 1,
                "side": "long" if position == 1 else "short",
                "type": trade_type,
                "entry_price": entry_price,
                "exit_price": price,
                "ticks": ticks_gained,
                "gross_pnl": gross,
                "net_pnl": net,
                "exit_reason": "end_of_data",
            }
        )

    pnls = [t["net_pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    print("=== Rule-Based Volume Profile Backtest ===")
    print(f"Bars: {len(df)} | Trades: {len(trades)} | Wins: {wins} | Losses: {losses}")
    print(f"Final equity: ${equity:,.2f}  (Return: {(equity/cfg.initial_equity-1)*100:.2f}%)")
    if trades:
        avg_win = np.mean([p for p in pnls if p > 0]) if wins else 0
        avg_loss = np.mean([p for p in pnls if p <= 0]) if losses else 0
        meanrev = sum(1 for t in trades if t['type'].startswith('meanrev'))
        breakout = sum(1 for t in trades if t['type'].startswith('breakout'))
        print(f"Avg win: ${avg_win:.2f} | Avg loss: ${avg_loss:.2f}")
        print(f"Mean reversion trades: {meanrev} | Breakout trades: {breakout}")
        print(f"Tick size: {cfg.tick_size} | Tick value: ${cfg.tick_value} | Contracts: {cfg.contracts} | Commission/side: ${cfg.commission_per_side}")


def main():
    data_path = Path(os.environ.get("DATA_PATH", "ml/data/spy_5min_data.parquet"))
    cfg = VPConfig(
        tick_size=float(os.environ.get("TICK_SIZE", 0.25)),
        tick_value=float(os.environ.get("TICK_VALUE", 5.0)),
        commission_per_side=float(os.environ.get("COMMISSION_PER_SIDE", 0.0)),
        contracts=int(os.environ.get("CONTRACTS", 1)),
        initial_equity=float(os.environ.get("INITIAL_EQUITY", 100_000.0)),
    )
    df = pd.read_parquet(data_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df["_ts_local"] = df.index.tz_convert("America/New_York")
    print(f"Loaded {len(df)} bars from {data_path} (session: 6pm ET → 5pm ET)")
    backtest(df, cfg)


if __name__ == "__main__":
    main()
