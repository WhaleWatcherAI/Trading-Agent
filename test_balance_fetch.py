#!/usr/bin/env python3
"""
Test script to verify dynamic account balance fetching from TopStep API
"""

import asyncio
import os
from dotenv import load_dotenv
from config import Settings
from topstep_client import TopstepClient

# Load environment variables
load_dotenv()

async def test_balance_fetch():
    """Test fetching account balance from TopStep"""
    print("=" * 60)
    print("TESTING DYNAMIC BALANCE FETCHING FROM TOPSTEP")
    print("=" * 60)

    # Initialize settings
    settings = Settings()

    print(f"\n📋 Configuration:")
    print(f"  Account ID: {settings.topstepx_account_id}")
    print(f"  API Base URL: {settings.topstepx_rest_base_url}")
    print(f"  Static Balance (fallback): ${settings.account_balance:,.2f}")

    # Initialize TopStep client
    topstep = TopstepClient(settings)

    print("\n🔄 Fetching account info from TopStep API...")

    try:
        # Fetch account info
        account_info = await topstep.get_account_info()

        if "error" in account_info:
            print(f"\n⚠️  Error fetching balance: {account_info['error']}")
            print(f"   Using fallback balance: ${account_info['balance']:,.2f}")
        else:
            print(f"\n✅ Successfully fetched account info!")
            print(f"\n💰 Account Details:")
            print(f"  Balance: ${account_info['balance']:,.2f}")
            print(f"  Account ID: {account_info['account_id']}")
            print(f"  Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            print(f"  Daily P&L: ${account_info.get('daily_pnl', 0):,.2f}")
            print(f"  Open P&L: ${account_info.get('open_pnl', 0):,.2f}")
            print(f"  Realized P&L: ${account_info.get('realized_pnl', 0):,.2f}")
            print(f"  Account Status: {account_info.get('account_status', 'unknown')}")

            # Check if balance differs from static config
            if account_info['balance'] != settings.account_balance:
                print(f"\n📊 Balance Comparison:")
                print(f"  Static Config: ${settings.account_balance:,.2f}")
                print(f"  Dynamic API:   ${account_info['balance']:,.2f}")
                print(f"  Difference:    ${account_info['balance'] - settings.account_balance:,.2f}")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    # Test periodic sync simulation
    print("\n📅 Testing periodic sync (simulating 5-minute interval)...")
    print("   This would normally sync every 5 minutes in the live system")

    await asyncio.sleep(2)  # Short delay for demo

    print("\n🔄 Simulating periodic balance sync...")
    try:
        account_info2 = await topstep.get_account_info()
        if "error" not in account_info2:
            print(f"✅ Periodic sync successful - Balance: ${account_info2['balance']:,.2f}")
    except Exception as e:
        print(f"⚠️  Periodic sync failed: {e}")

async def main():
    """Main entry point"""
    await test_balance_fetch()

if __name__ == "__main__":
    asyncio.run(main())