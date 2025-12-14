#!/usr/bin/env python3
"""
Test Qwen2.5 with trading data snapshot
Fetches latest NQ futures data and asks Qwen for analysis
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import os

# Ollama config
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = "qwen2.5:7b"

def query_ollama(prompt: str) -> str:
    """Send prompt to Ollama and get response"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.3,
        "top_p": 0.9,
    }
    data = json.dumps(payload).encode("utf-8")
    url = urllib.parse.urljoin(OLLAMA_HOST, "/api/generate")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()

    out = json.loads(body.decode("utf-8"))
    return out.get("response", "").strip()


# Sample market snapshot (similar to what Fabio agent sends to DeepSeek)
# This mimics the FuturesMarketData structure
SAMPLE_SNAPSHOT = {
    "symbol": "NQ",
    "timestamp": datetime.now().isoformat(),
    "currentPrice": 21145.50,

    # Recent 5-min candles
    "candles": [
        {"timestamp": "2025-12-01T09:30:00Z", "open": 21120.00, "high": 21135.50, "low": 21115.25, "close": 21130.75, "volume": 12500},
        {"timestamp": "2025-12-01T09:35:00Z", "open": 21130.75, "high": 21148.00, "low": 21128.50, "close": 21142.25, "volume": 8900},
        {"timestamp": "2025-12-01T09:40:00Z", "open": 21142.25, "high": 21155.75, "low": 21140.00, "close": 21145.50, "volume": 7200},
    ],

    # CVD (Cumulative Volume Delta)
    "cvd": {
        "value": 245,
        "trend": "up",
        "ohlc": {"open": 180, "high": 290, "low": 165, "close": 245}
    },

    # Volume Profile
    "volumeProfile": {
        "poc": 21128.50,    # Point of Control
        "vah": 21165.25,    # Value Area High
        "val": 21095.00,    # Value Area Low
        "lvns": [21110.00, 21155.00],  # Low Volume Nodes
        "sessionHigh": 21172.50,
        "sessionLow": 21082.25
    },

    # Order flow - big trades
    "orderFlow": {
        "bigTrades": [
            {"price": 21145.00, "size": 85, "side": "buy", "timestamp": "2025-12-01T09:42:15Z"},
            {"price": 21142.50, "size": 120, "side": "sell", "timestamp": "2025-12-01T09:41:30Z"},
            {"price": 21148.25, "size": 65, "side": "buy", "timestamp": "2025-12-01T09:40:45Z"},
        ]
    },

    # L2 / Microstructure (resting orders)
    "microstructure": {
        "restingLimitOrders": [
            {"price": 21150.00, "restingBid": 45, "restingAsk": 180, "total": 225},
            {"price": 21145.00, "restingBid": 120, "restingAsk": 65, "total": 185},
            {"price": 21140.00, "restingBid": 210, "restingAsk": 30, "total": 240},
            {"price": 21135.00, "restingBid": 150, "restingAsk": 25, "total": 175},
            {"price": 21155.00, "restingBid": 20, "restingAsk": 250, "total": 270},
            {"price": 21160.00, "restingBid": 15, "restingAsk": 320, "total": 335},
        ],
        "largeWhaleTrades": [
            {"price": 21145.00, "size": 85, "side": "buy"},
            {"price": 21142.50, "size": 120, "side": "sell"},
        ]
    },

    # Flow signals
    "flowSignals": {
        "deltaLast1m": 42,
        "deltaLast5m": 185,
        "cvdSlopeShort": 0.8,
        "cvdSlopeLong": 0.3,
        "cvdDivergence": "none"
    },

    # Technical indicators
    "indicators": {
        "rsi14": 58.5,
        "sma9": 21138.25,
        "sma20": 21125.50,
        "atr14": 32.5,
        "bbUpper": 21175.00,
        "bbLower": 21090.00,
        "bbMiddle": 21132.50
    },

    # Market state
    "marketState": "balanced_leaning_bullish",

    # Current position (none)
    "openPosition": None,

    # Account
    "account": {
        "balance": 52450.00,
        "position": 0,
        "unrealizedPnL": 0,
        "realizedPnL": 125.50
    }
}


def format_snapshot_for_llm(snapshot: dict) -> str:
    """Format the snapshot into a concise prompt for the LLM"""

    candles = snapshot["candles"]
    last_candle = candles[-1]
    vp = snapshot["volumeProfile"]
    flow = snapshot["flowSignals"]
    ind = snapshot["indicators"]
    micro = snapshot["microstructure"]

    # Format L2 book
    l2_bids = [(o["price"], o["restingBid"]) for o in micro["restingLimitOrders"] if o["restingBid"] > 50]
    l2_asks = [(o["price"], o["restingAsk"]) for o in micro["restingLimitOrders"] if o["restingAsk"] > 50]
    l2_bids.sort(key=lambda x: -x[0])  # Highest bid first
    l2_asks.sort(key=lambda x: x[0])   # Lowest ask first

    prompt = f"""NQ Futures Analysis - {snapshot['timestamp'][:19]}

PRICE: {snapshot['currentPrice']}
Last candle: O={last_candle['open']} H={last_candle['high']} L={last_candle['low']} C={last_candle['close']} Vol={last_candle['volume']}

VOLUME PROFILE:
- POC: {vp['poc']} (dist: {snapshot['currentPrice'] - vp['poc']:+.2f})
- VAH: {vp['vah']} (dist: {snapshot['currentPrice'] - vp['vah']:+.2f})
- VAL: {vp['val']} (dist: {snapshot['currentPrice'] - vp['val']:+.2f})
- Session: H={vp['sessionHigh']} L={vp['sessionLow']}
- LVNs: {vp['lvns']}

ORDER FLOW:
- CVD: {snapshot['cvd']['value']} ({snapshot['cvd']['trend']})
- Delta 1m: {flow['deltaLast1m']}, 5m: {flow['deltaLast5m']}
- CVD slope: short={flow['cvdSlopeShort']}, long={flow['cvdSlopeLong']}
- Divergence: {flow['cvdDivergence']}

L2 BOOK (significant levels):
- Bids: {', '.join([f'{p}@{s}' for p,s in l2_bids[:3]])}
- Asks: {', '.join([f'{p}@{s}' for p,s in l2_asks[:3]])}

WHALE TRADES:
{chr(10).join([f"- {t['side'].upper()} {t['size']} @ {t['price']}" for t in micro['largeWhaleTrades']])}

INDICATORS:
- RSI(14): {ind['rsi14']}
- SMA9: {ind['sma9']}, SMA20: {ind['sma20']} (SMA9 {'>' if ind['sma9'] > ind['sma20'] else '<'} SMA20)
- BB: Upper={ind['bbUpper']}, Mid={ind['bbMiddle']}, Lower={ind['bbLower']}
- ATR(14): {ind['atr14']}

MARKET STATE: {snapshot['marketState']}
POSITION: {'FLAT' if not snapshot['openPosition'] else snapshot['openPosition']}

Based on this data, provide:
1. DIRECTION: LONG, SHORT, or FLAT
2. ENTRY: specific price
3. STOP LOSS: specific price (explain why)
4. TAKE PROFIT: specific price (explain why)
5. CONFIDENCE: 1-10
6. KEY FACTORS: 2-3 bullet points

Be concise. Use the L2 and volume profile for SL/TP placement."""

    return prompt


def main():
    print("="*60)
    print("QWEN2.5 TRADING ANALYSIS TEST")
    print("="*60)

    prompt = format_snapshot_for_llm(SAMPLE_SNAPSHOT)

    print("\n--- INPUT SNAPSHOT ---")
    print(prompt)
    print("\n--- QUERYING QWEN2.5:7B ---\n")

    try:
        response = query_ollama(prompt)
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  /mnt/wslg/distro/home/costa/ollama-download/bin/ollama serve")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
