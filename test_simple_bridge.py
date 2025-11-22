#!/usr/bin/env python3
"""
Simple synchronous test for the dashboard bridge.
"""

import socketio
import time

print("=" * 80)
print("SIMPLE DASHBOARD BRIDGE TEST")
print("=" * 80)

# Create synchronous client
sio = socketio.Client(reconnection=True)

# Event handlers
@sio.event
def connect():
    print("✅ Connected to dashboard!")
    print()

    # Send a test log message
    print("Sending test log message...")
    sio.emit("log", {
        "timestamp": time.time(),
        "message": "Python bridge test - connection successful!",
        "type": "success"
    })

    # Send a test status
    print("Sending test status...")
    sio.emit("status", {
        "symbol": "NQZ5",
        "isTrading": True,
        "position": None,
        "closedTrades": [],
        "accountStats": {
            "totalTrades": 0,
            "winners": 0,
            "losers": 0,
            "winRate": 0,
            "totalPnL": 0
        },
        "timestamp": time.time()
    })

    print("✅ Test messages sent successfully!")
    print()
    print("Check the dashboard at http://localhost:3337")
    print("You should see the log message in the Activity Log")
    print()

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
    print("Waiting 5 seconds...")
    time.sleep(5)

    print("=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)

    sio.disconnect()
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    import traceback
    traceback.print_exc()
