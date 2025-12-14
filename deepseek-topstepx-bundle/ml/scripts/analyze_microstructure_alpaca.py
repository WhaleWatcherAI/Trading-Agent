#!/usr/bin/env python3
"""
Microstructure analyzer for stocks using Alpaca 1-second bars (tick if provided by Alpaca).

- Fetches 1s bars for the last ~90 days (configurable) for a symbol (default SPY).
- Aggregates into 5m HTF candles.
- For each 5m candle, computes:
    * In-candle volume profile at tick-size bins (default $0.01)
    * POC / VAH / VAL (70% value area)
    * Per-price volume and delta proxy (delta = sign of bar change * volume; Alpaca bars lack aggressor side)
    * CVD inside the candle (delta proxy cumulative)
    * Imbalance near candle high/low (top/bottom 20% of range)
    * Simple divergence flag vs candle direction

Note: Alpaca 1s bars don’t include trade aggressor side; delta is approximated from price change sign.
If you have true tick data with side, swap in that data frame (price, volume, side) and bypass fetch.

Env:
  APCA_API_KEY_ID, APCA_API_SECRET_KEY (read from .env if present)
  APCA_API_BASE_URL (optional, defaults to https://data.alpaca.markets)

CLI:
  python3 ml/scripts/analyze_microstructure_alpaca.py --symbol SPY --days 90 --htf 5m --tick-size 0.01
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def fetch_1s_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    base_url: str,
    key: str,
    secret: str,
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Fetch 1-second bars from Alpaca data v2 API with pagination.
    Falls back to 1-minute if 1s unavailable or not supported.
    """
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    timeframe = "1Sec"
    fallback_timeframe = "1Min"
    url = f"{base_url}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 10_000,
        "feed": feed,
    }

    records: List[Dict] = []
    next_token: Optional[str] = None
    while True:
        if next_token:
            params["page_token"] = next_token
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            # If 1Sec not supported, break to fallback
            if timeframe == "1Sec" and resp.status_code == 400:
                records.clear()
                break
            raise RuntimeError(f"Alpaca API error {resp.status_code}: {resp.text}")
        data = resp.json()
        bars = data.get("bars", [])
        if not bars:
            break
        for b in bars:
            records.append(
                {
                    "ts": pd.to_datetime(b["t"]),
                    "open": float(b["o"]),
                    "high": float(b["h"]),
                    "low": float(b["l"]),
                    "close": float(b["c"]),
                    "volume": float(b["v"]),
                }
            )
        next_token = data.get("next_page_token")
        if not next_token:
            break

    df = pd.DataFrame.from_records(records)
    if df.empty and timeframe == "1Sec":
        # Fallback to 1Min
        params["timeframe"] = fallback_timeframe
        next_token = None
        while True:
            if next_token:
                params["page_token"] = next_token
            resp = requests.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"Alpaca API error {resp.status_code}: {resp.text}")
            data = resp.json()
            bars = data.get("bars", [])
            if not bars:
                break
            for b in bars:
                records.append(
                    {
                        "ts": pd.to_datetime(b["t"]),
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low": float(b["l"]),
                        "close": float(b["c"]),
                        "volume": float(b["v"]),
                    }
                )
            next_token = data.get("next_page_token")
            if not next_token:
                break
        df = pd.DataFrame.from_records(records)

    if df.empty:
        raise RuntimeError("No bars returned.")

    df = df.set_index("ts").sort_index()
    return df


def volume_profile(
    prices: pd.Series,
    vols: pd.Series,
    tick_size: float,
    num_bins: int = 400,
    value_area_pct: float = 0.70,
) -> Tuple[float, float, float, pd.DataFrame]:
    """Compute POC, VAH, VAL using proportional allocation into price bins."""
    if prices.empty:
        return float("nan"), float("nan"), float("nan"), pd.DataFrame()
    lo, hi = prices.min(), prices.max()
    if hi <= lo:
        return float("nan"), float("nan"), float("nan"), pd.DataFrame()
    bins = max(int((hi - lo) / tick_size) + 1, num_bins)
    edges = np.linspace(lo, hi, bins + 1)
    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")
    df = pd.DataFrame({"price": prices, "vol": vols})
    df["bin"] = pd.cut(df["price"], intervals, include_lowest=True)
    agg = df.groupby("bin")["vol"].sum().reset_index()
    agg["mid"] = agg["bin"].apply(lambda x: (x.left + x.right) / 2 if pd.notnull(x) else float("nan"))
    agg = agg.dropna(subset=["mid"])
    if agg.empty:
        return float("nan"), float("nan"), float("nan"), pd.DataFrame()
    poc_row = agg.loc[agg["vol"].idxmax()]
    poc = poc_row["mid"]
    total_vol = agg["vol"].sum()
    target = total_vol * value_area_pct
    captured = poc_row["vol"]
    value_area = {agg["mid"].idxmax()}
    # Expand VA iteratively
    while captured < target and len(value_area) < len(agg):
        min_idx = min(value_area)
        max_idx = max(value_area)
        up_idx = max_idx + 1 if max_idx + 1 < len(agg) else None
        dn_idx = min_idx - 1 if min_idx - 1 >= 0 else None
        up_vol = agg.iloc[up_idx]["vol"] if up_idx is not None else -1
        dn_vol = agg.iloc[dn_idx]["vol"] if dn_idx is not None else -1
        if up_vol >= dn_vol and up_idx is not None:
            value_area.add(up_idx)
            captured += up_vol
        elif dn_idx is not None:
            value_area.add(dn_idx)
            captured += dn_vol
        else:
            break
    vah = agg.loc[max(value_area), "mid"]
    val = agg.loc[min(value_area), "mid"]
    return poc, vah, val, agg


def analyze_htf(df_1s: pd.DataFrame, htf: str, tick_size: float) -> pd.DataFrame:
    """Aggregate to HTF and compute microstructure stats per HTF candle."""
    ohlc = df_1s["close"].resample(htf).ohlc()
    vol = df_1s["volume"].resample(htf).sum()
    df_htf = ohlc.join(vol).dropna()
    results = []
    for ts, row in df_htf.iterrows():
        mask = (df_1s.index >= ts) & (df_1s.index < ts + pd.Timedelta(htf))
        sub = df_1s.loc[mask]
        if sub.empty:
            continue
        poc, vah, val, profile = volume_profile(sub["close"], sub["volume"], tick_size)
        rng = row["high"] - row["low"]
        top_band = row["low"] + rng * 0.8
        bot_band = row["low"] + rng * 0.2
        top_vol = sub.loc[sub["close"] >= top_band, "volume"].sum()
        bot_vol = sub.loc[sub["close"] <= bot_band, "volume"].sum()
        # delta proxy: sign of price change * volume
        delta = ((sub["close"].diff().fillna(0).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)) * sub["volume"]).sum()
        cvd = ((sub["close"].diff().fillna(0).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)) * sub["volume"]).cumsum().iloc[-1]
        dir_sign = 1 if row["close"] > row["open"] else -1 if row["close"] < row["open"] else 0
        divergence = (dir_sign != 0 and (delta * dir_sign) < 0)
        results.append(
            {
                "ts": ts,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "poc": poc,
                "vah": vah,
                "val": val,
                "top_vol": top_vol,
                "bot_vol": bot_vol,
                "delta_proxy": delta,
                "cvd_proxy": cvd,
                "divergence": divergence,
            }
        )
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--htf", default="5min", help="5min/15min/1H/etc.")
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--value-area-pct", type=float, default=70.0)
    args = parser.parse_args()

    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    base_url = os.environ.get("APCA_API_BASE_URL", "https://data.alpaca.markets")
    feed = os.environ.get("APCA_DATA_FEED", "iex")
    if not key or not secret:
        print("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in env.", file=sys.stderr)
        sys.exit(1)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    print(f"Fetching {args.symbol} 1s bars from {start} to {end} ...")
    df_1s = fetch_1s_bars(args.symbol, start, end, base_url, key, secret, feed=feed)
    print(f"Fetched {len(df_1s)} bars.")

    df_htf = analyze_htf(df_1s, args.htf, args.tick_size)
    if df_htf.empty:
        print("No HTF results.")
        return

    print("\n=== Sample (last 10 HTF candles) ===")
    print(
        df_htf.tail(10)[
            [
                "ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "poc",
                "vah",
                "val",
                "top_vol",
                "bot_vol",
                "delta_proxy",
                "cvd_proxy",
                "divergence",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
