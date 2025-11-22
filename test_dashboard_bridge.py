#!/usr/bin/env python3
"""
Test script for the dashboard bridge.

This script connects to the Node.js dashboard server and sends simulated
position and trade updates to verify the integration is working correctly.

Usage:
    # Make sure the Node.js server is running first:
    npx tsx live-topstepx-nq-ict.ts

    # Then run this test:
    python test_dashboard_bridge.py
"""

import asyncio
from datetime import datetime
from dashboard_bridge import DashboardBridge, DashboardPosition, DashboardTrade


async def test_bridge():
    """Test the dashboard bridge with simulated data"""

    print("=" * 80)
    print("DASHBOARD BRIDGE TEST")
    print("=" * 80)
    print("Make sure the Node.js server is running on port 3337")
    print("Then open http://localhost:3337 in your browser")
    print("=" * 80)
    print()

    # Create bridge
    bridge = DashboardBridge(dashboard_url="http://localhost:3337")

    # Connect in background
    connect_task = asyncio.create_task(bridge.connect())

    # Wait for connection
    print("Connecting to dashboard...")
    await asyncio.sleep(3)

    if not bridge.connected:
        print("❌ Failed to connect to dashboard!")
        print("Make sure the Node.js server is running:")
        print("  npx tsx live-topstepx-nq-ict.ts")
        return

    print("✅ Connected to dashboard!")
    print()

    # Test 1: Send initial log
    print("Test 1: Sending log messages...")
    await bridge.send_log("Test bridge initialized", "info")
    await bridge.send_log("Starting simulated trading", "success")
    await asyncio.sleep(1)

    # Test 2: Send status with no position
    print("Test 2: Sending status update (no position)...")
    await bridge.emit_status(
        position=None,
        closed_trades=[],
        total_pnl=0.0,
        symbol="NQZ5",
        is_trading=True,
    )
    await asyncio.sleep(2)

    # Test 3: Open a position
    print("Test 3: Opening a LONG position...")
    await bridge.send_log("Opening LONG position @ 21000.00", "success")

    position = DashboardPosition(
        side="long",
        entryPrice=21000.00,
        entryTime=datetime.now().isoformat(),
        stopLoss=20996.00,
        targetTP1=21004.00,
        targetTP2=21008.00,
        totalQty=3,
        remaining=3,
        entryPattern="Test: Fabio LLM Decision",
        unrealizedPnL=0.0,
    )

    await bridge.emit_status(
        position=position,
        closed_trades=[],
        total_pnl=0.0,
        is_trading=True,
    )
    await asyncio.sleep(2)

    # Test 4: Update position with unrealized PnL
    print("Test 4: Updating position with unrealized P&L...")
    for i in range(5):
        position.unrealizedPnL = 50.0 * (i + 1)
        await bridge.emit_status(
            position=position,
            closed_trades=[],
            total_pnl=0.0,
            is_trading=True,
        )
        await bridge.send_log(f"Price moving in our favor: ${position.unrealizedPnL:.2f}", "info")
        await asyncio.sleep(2)

    # Test 5: Close position with profit
    print("Test 5: Closing position with profit...")
    await bridge.send_log("Take profit hit! Closing position", "success")

    trade1 = DashboardTrade(
        tradeId="test-001",
        side="long",
        entryPrice=21000.00,
        exitPrice=21004.00,
        entryTime=position.entryTime,
        exitTime=datetime.now().isoformat(),
        quantity=3,
        pnl=240.00,  # 3 contracts * 4 points * $20/point
        exitReason="tp1",
        entryPattern="Test: Fabio LLM",
    )

    await bridge.emit_trade(trade1)

    # Update status with no position
    await bridge.emit_status(
        position=None,
        closed_trades=[trade1],
        total_pnl=240.00,
        is_trading=True,
    )
    await asyncio.sleep(3)

    # Test 6: Open SHORT position
    print("Test 6: Opening a SHORT position...")
    await bridge.send_log("Opening SHORT position @ 21000.00", "info")

    position2 = DashboardPosition(
        side="short",
        entryPrice=21000.00,
        entryTime=datetime.now().isoformat(),
        stopLoss=21004.00,
        targetTP1=20996.00,
        targetTP2=20992.00,
        totalQty=3,
        remaining=3,
        entryPattern="Test: Bearish momentum detected",
        unrealizedPnL=0.0,
    )

    await bridge.emit_status(
        position=position2,
        closed_trades=[trade1],
        total_pnl=240.00,
        is_trading=True,
    )
    await asyncio.sleep(2)

    # Test 7: Stop loss hit
    print("Test 7: Stop loss hit...")
    await bridge.send_log("Stop loss hit! Closing position", "error")

    trade2 = DashboardTrade(
        tradeId="test-002",
        side="short",
        entryPrice=21000.00,
        exitPrice=21004.00,
        entryTime=position2.entryTime,
        exitTime=datetime.now().isoformat(),
        quantity=3,
        pnl=-240.00,  # Loss
        exitReason="stop",
        entryPattern="Test: Fabio LLM",
    )

    await bridge.emit_trade(trade2)

    # Update status with both trades
    await bridge.emit_status(
        position=None,
        closed_trades=[trade1, trade2],
        total_pnl=0.00,  # Break even
        is_trading=True,
    )
    await asyncio.sleep(3)

    # Test 8: Multiple winning trades
    print("Test 8: Adding more trades to show statistics...")
    all_trades = [trade1, trade2]

    for i in range(8):
        side = "long" if i % 2 == 0 else "short"
        pnl = 160.0 if i % 3 != 0 else -80.0  # 70% win rate

        trade = DashboardTrade(
            tradeId=f"test-{i+3:03d}",
            side=side,
            entryPrice=21000.00,
            exitPrice=21004.00 if pnl > 0 else 20998.00,
            entryTime=datetime.now().isoformat(),
            exitTime=datetime.now().isoformat(),
            quantity=3,
            pnl=pnl,
            exitReason="tp1" if pnl > 0 else "stop",
            entryPattern="Test: Fabio LLM",
        )
        all_trades.append(trade)

        result = "WIN" if pnl > 0 else "LOSS"
        await bridge.send_log(f"Trade #{i+3}: {side.upper()} {result} ${pnl:.2f}", "success" if pnl > 0 else "warning")

        total_pnl = sum(t.pnl for t in all_trades)
        await bridge.emit_status(
            position=None,
            closed_trades=all_trades,
            total_pnl=total_pnl,
            is_trading=True,
        )
        await asyncio.sleep(1)

    print()
    print("=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)
    print("Check the dashboard at http://localhost:3337")
    print("You should see:")
    print("  ✓ Activity log with all messages")
    print("  ✓ Trade history (10 trades)")
    print("  ✓ Account statistics (70% win rate)")
    print("  ✓ Total P&L displayed")
    print()
    print("Press Ctrl+C to exit")
    print("=" * 80)

    # Keep connection alive
    try:
        await asyncio.sleep(300)  # 5 minutes
    except KeyboardInterrupt:
        print("\nShutting down...")

    await bridge.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(test_bridge())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
