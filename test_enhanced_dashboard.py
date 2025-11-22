#!/usr/bin/env python3
"""
Test script for enhanced dashboard features:
- Threshold adjustment controls
- Volume profile toggle
- LLM prompt viewing with timestamps

This sends test LLM prompts and positions to verify the new dashboard features.
"""

import socketio
import time
import json

print("=" * 80)
print("ENHANCED DASHBOARD TEST - LLM Prompts + Threshold Controls")
print("=" * 80)

# Create synchronous client
sio = socketio.Client(reconnection=True)

# Event handlers
@sio.event
def connect():
    print("✅ Connected to dashboard!")
    print()

    # Send test log
    print("1. Sending initialization log...")
    sio.emit("log", {
        "timestamp": time.time(),
        "message": "Enhanced dashboard test - connection successful!",
        "type": "success"
    })

    # Send test LLM prompt
    print("2. Sending test LLM prompt...")
    test_prompt = {
        "mode": "live_decision",
        "symbol": "NQZ5",
        "timestamp": time.time(),
        "session": "RTH",
        "price": 21000.50,
        "ohlc": {
            "open": 21000.00,
            "high": 21005.00,
            "low": 20995.00,
            "close": 21000.50,
        },
        "profiles": [
            {"price": 21000, "buy_volume": 150, "sell_volume": 100},
            {"price": 21001, "buy_volume": 200, "sell_volume": 180},
        ],
        "derived_state": {
            "cvd": 1500,
            "trend": "bullish",
            "volatility": "medium",
        },
        "orderflow": {
            "imbalance": 0.35,
            "absorption_detected": False,
        },
        "open_positions": [],
        "importance_zones": [
            {
                "center_price": 21000,
                "priority": "high",
                "reason": "High volume node + absorption zone",
            }
        ],
        "recent_performance_summary": {
            "session_pnl": 450.00,
            "session_drawdown": -120.00,
        },
        "historical_notes_snippet": [
            "Price tends to bounce at 21000 level",
            "Watch for exhaustion near 21050",
        ],
    }

    sio.emit("llm_prompt", {
        "payload": test_prompt,
        "timestamp": time.time(),
        "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    time.sleep(2)

    # Send test LLM response
    print("3. Sending test LLM response...")
    test_response = {
        "type": "llm_response",
        "decisions": [
            {
                "action": "enter",
                "side": "long",
                "size": 3,
                "price_instruction": "market",
                "stop_price": 20996.00,
                "target_price": 21008.00,
                "reasoning": "Bullish CVD + support at high volume node",
            }
        ],
        "importance_zones": [
            {
                "center_price": 21000,
                "priority": "high",
                "inner_band_ticks": 2,
                "outer_band_ticks": 5,
            }
        ],
        "notes_to_future_self": [
            "Strong buying pressure observed at 21000 level",
            "Consider tightening stop if price moves above 21005",
        ],
        "strategy_updates": {
            "strategy_tweaks": [
                {
                    "name": "absorption_sensitivity",
                    "changes": {"threshold": 50},
                }
            ]
        },
    }

    sio.emit("llm_prompt", {
        "payload": test_response,
        "timestamp": time.time(),
        "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    time.sleep(2)

    # Send test position
    print("4. Sending test position...")
    sio.emit("status", {
        "symbol": "NQZ5",
        "isTrading": True,
        "position": {
            "side": "long",
            "entryPrice": 21000.00,
            "entryTime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stopLoss": 20996.00,
            "targetTP1": 21004.00,
            "targetTP2": 21008.00,
            "totalQty": 3,
            "remaining": 3,
            "unrealizedPnL": 50.00,
            "entryPattern": "Fabio LLM: Bullish CVD + Support",
        },
        "closedTrades": [],
        "accountStats": {
            "totalTrades": 0,
            "winners": 0,
            "losers": 0,
            "winRate": 0,
            "totalPnL": 0,
        },
        "timestamp": time.time()
    })

    print()
    print("✅ Test data sent successfully!")
    print()
    print("=" * 80)
    print("CHECK THE DASHBOARD NOW!")
    print("=" * 80)
    print()
    print("You should see:")
    print("  ✓ Threshold adjustment controls in the legend panel")
    print("    - Absorption (contracts)")
    print("    - Exhaustion push (contracts)")
    print("    - Drop factor (0-1)")
    print()
    print("  ✓ Volume profile toggle checkbox")
    print()
    print("  ✓ LLM Prompts & Decisions panel showing:")
    print("    - Test LLM request with full payload")
    print("    - Test LLM response with decisions and notes")
    print("    - Auto-expand checkbox and Clear button")
    print("    - Click on prompts to expand/collapse them")
    print()
    print("  ✓ Current position: LONG @ 21000.00")
    print()
    print("  ✓ Activity log showing all events")
    print()
    print("=" * 80)

@sio.event
def disconnect():
    print("Disconnected from dashboard")

@sio.event
def connect_error(data):
    print(f"❌ Connection error: {data}")

# Connect
print("Connecting to http://localhost:3337...")
try:
    sio.connect("http://localhost:3337")
    print("Waiting 10 seconds...")
    time.sleep(10)

    print()
    print("=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)
    print("The dashboard should now show all the new features.")
    print("Press Ctrl+C to exit")
    print("=" * 80)

    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")

    sio.disconnect()
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    import traceback
    traceback.print_exc()
