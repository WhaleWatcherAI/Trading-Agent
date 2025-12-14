#!/usr/bin/env python3
"""
L2 Market Depth Data Collector for TopstepX

Collects and saves for all 3 contracts (NQ, ES, GC):
1. Raw L2 updates (every update from websocket) - gzip compressed
2. Periodic snapshots (every 5 seconds, similar to what Qwen agent sees)

Storage estimate (24/7 futures hours):
- Raw updates: ~1-3GB/day (compressed ~0.3-1GB/day)
- Snapshots: ~50-100MB/day
- 2 weeks: ~10-20GB total
"""

import os
import json
import time
import signal
import logging
import gzip
from datetime import datetime
from typing import Dict, Any
from collections import deque
from threading import Thread, Lock
import warnings
warnings.filterwarnings("ignore")

import asyncio
import aiohttp
from aiohttp import ClientSession

# Load environment
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
load_dotenv(env_path)

import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

TOPSTEPX_API_KEY = os.environ.get('TOPSTEPX_API_KEY', '')
TOPSTEPX_USERNAME = os.environ.get('TOPSTEPX_USERNAME', '')
TOPSTEPX_BASE_URL = os.environ.get('TOPSTEPX_BASE_URL', 'https://api.topstepx.com')

# Contracts to collect
CONTRACTS = {
    'nq': 'CON.F.US.ENQ.Z25',
    'es': 'CON.F.US.EP.Z25',
    'gc': 'CON.F.US.GCE.G26',
}

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'l2')
LOGS_DIR = os.path.join(SCRIPT_DIR, '..', 'logs')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Collection settings
SNAPSHOT_INTERVAL_SECONDS = 5
RAW_FLUSH_INTERVAL = 30
COMPRESS_RAW = True

# Global state
session_token = None
token_expiry = 0
running = True
logger = None

# Per-symbol data
symbol_data = {}
for sym in CONTRACTS:
    symbol_data[sym] = {
        'raw_buffer': deque(maxlen=50000),
        'orderbook': {'bids': [], 'asks': [], 'timestamp': None},
        'lock': Lock(),
        'stats': {'raw_updates': 0, 'snapshots': 0, 'bytes': 0}
    }

total_stats = {'start_time': None, 'total_updates': 0}

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    global logger
    log_file = os.path.join(LOGS_DIR, f'l2_collector_{datetime.utcnow().strftime("%Y%m%d")}.log')
    logger = logging.getLogger('l2_collector')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-5s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    return log_file

# =============================================================================
# AUTHENTICATION
# =============================================================================

def authenticate():
    global session_token, token_expiry

    if session_token and time.time() < token_expiry:
        return session_token

    response = requests.post(
        f"{TOPSTEPX_BASE_URL}/api/Auth/loginKey",
        json={"userName": TOPSTEPX_USERNAME, "apiKey": TOPSTEPX_API_KEY},
        timeout=30
    )

    data = response.json()
    if not data.get('success') or data.get('errorCode') != 0:
        raise Exception(f"Authentication failed: {data}")

    session_token = data['token']
    token_expiry = time.time() + 23 * 3600
    return session_token

# =============================================================================
# DATA SAVING
# =============================================================================

def flush_raw_buffer(symbol: str):
    """Flush raw updates buffer to disk."""
    data = symbol_data[symbol]

    with data['lock']:
        updates = list(data['raw_buffer'])
        data['raw_buffer'].clear()

    if not updates:
        return

    date_str = datetime.utcnow().strftime('%Y%m%d')
    file_path = os.path.join(DATA_DIR, f'l2_raw_{symbol}_{date_str}.json.gz')

    try:
        with gzip.open(file_path, 'ab') as f:
            for update in updates:
                line = json.dumps(update) + '\n'
                f.write(line.encode('utf-8'))
                data['stats']['bytes'] += len(line)
    except Exception as e:
        if logger:
            logger.error(f"Error flushing {symbol} buffer: {e}")

def save_snapshot(symbol: str):
    """Save current orderbook snapshot."""
    data = symbol_data[symbol]
    ob = data['orderbook']

    if not ob['timestamp']:
        return

    date_str = datetime.utcnow().strftime('%Y%m%d')
    file_path = os.path.join(DATA_DIR, f'l2_snapshots_{symbol}_{date_str}.json')

    snapshot = {
        't': datetime.utcnow().isoformat(),
        'sym': symbol,
        'bids': ob['bids'][:10],
        'asks': ob['asks'][:10],
    }

    # Calculate derived fields
    if ob['bids'] and ob['asks']:
        best_bid = ob['bids'][0].get('price', 0)
        best_ask = ob['asks'][0].get('price', 0)
        if best_bid > 0 and best_ask > 0:
            snapshot['spread'] = round(best_ask - best_bid, 4)
            snapshot['mid'] = round((best_bid + best_ask) / 2, 4)
            snapshot['bid_depth'] = sum(b.get('size', b.get('qty', 0)) for b in ob['bids'][:10])
            snapshot['ask_depth'] = sum(a.get('size', a.get('qty', 0)) for a in ob['asks'][:10])
            snapshot['imbalance'] = round((snapshot['bid_depth'] - snapshot['ask_depth']) /
                                          max(snapshot['bid_depth'] + snapshot['ask_depth'], 1), 4)

    try:
        with open(file_path, 'a') as f:
            f.write(json.dumps(snapshot) + '\n')
        data['stats']['snapshots'] += 1
    except Exception as e:
        if logger:
            logger.error(f"Error saving {symbol} snapshot: {e}")

# =============================================================================
# WEBSOCKET CONNECTION (SignalR)
# =============================================================================

def get_symbol_for_contract(contract_id: str) -> str:
    """Get symbol key for a contract ID."""
    for sym, cid in CONTRACTS.items():
        if cid == contract_id:
            return sym
    return None

def handle_market_depth(contract_id: str, depth: dict):
    """Handle incoming L2 market depth update."""
    global total_stats

    symbol = get_symbol_for_contract(contract_id)
    if not symbol or not depth:
        return

    data = symbol_data[symbol]
    timestamp = datetime.utcnow().isoformat()

    # Store raw update
    raw_update = {'t': timestamp, 'd': depth}

    with data['lock']:
        data['raw_buffer'].append(raw_update)
        data['stats']['raw_updates'] += 1
        total_stats['total_updates'] += 1

    # Update current orderbook
    if 'bids' in depth:
        data['orderbook']['bids'] = depth['bids']
    if 'asks' in depth:
        data['orderbook']['asks'] = depth['asks']
    data['orderbook']['timestamp'] = timestamp

# =============================================================================
# BACKGROUND THREADS
# =============================================================================

def data_saver_thread():
    """Background thread to save data periodically."""
    last_snapshot = time.time()
    last_flush = time.time()

    while running:
        now = time.time()

        # Save snapshots every N seconds
        if now - last_snapshot >= SNAPSHOT_INTERVAL_SECONDS:
            for symbol in CONTRACTS:
                save_snapshot(symbol)
            last_snapshot = now

        # Flush raw buffers
        if now - last_flush >= RAW_FLUSH_INTERVAL:
            for symbol in CONTRACTS:
                flush_raw_buffer(symbol)
            last_flush = now

        time.sleep(0.5)

    # Final flush
    for symbol in CONTRACTS:
        flush_raw_buffer(symbol)
        save_snapshot(symbol)

def stats_display_thread():
    """Background thread to display stats."""
    while running:
        if total_stats['start_time']:
            uptime = (datetime.utcnow() - total_stats['start_time']).total_seconds() / 60

            total_bytes = sum(symbol_data[s]['stats']['bytes'] for s in CONTRACTS)
            total_snaps = sum(symbol_data[s]['stats']['snapshots'] for s in CONTRACTS)
            mb_written = total_bytes / (1024 * 1024)
            rate = total_stats['total_updates'] / max(uptime * 60, 1)

            status_parts = []
            for sym in CONTRACTS:
                updates = symbol_data[sym]['stats']['raw_updates']
                status_parts.append(f"{sym.upper()}:{updates:,}")

            print(f"\r[L2] {' | '.join(status_parts)} | "
                  f"Snaps:{total_snaps:,} | Size:{mb_written:.1f}MB | "
                  f"Rate:{rate:.1f}/s | Up:{uptime:.0f}m", end='', flush=True)

        time.sleep(5)

# =============================================================================
# SIGNALR CONNECTION
# =============================================================================

async def connect_signalr():
    """Connect to TopstepX SignalR hub and subscribe to market depth."""
    global running

    from signalrcore.hub_connection_builder import HubConnectionBuilder

    token = authenticate()

    # Real-time market data hub is on rtc subdomain, not api
    hub_url = "https://rtc.topstepx.com/hubs/market"

    hub_connection = HubConnectionBuilder()\
        .with_url(hub_url, options={
            "access_token_factory": lambda: token,
            "headers": {"Authorization": f"Bearer {token}"},
            "skip_negotiation": True,  # Direct WebSocket connection
        })\
        .with_automatic_reconnect({
            "type": "raw",
            "keep_alive_interval": 10,
            "reconnect_interval": 5,
        })\
        .build()

    # Register handlers
    hub_connection.on("GatewayMarketDepth", lambda args: handle_market_depth(args[0], args[1]) if len(args) >= 2 else None)
    hub_connection.on("gatewaymarketdepth", lambda args: handle_market_depth(args[0], args[1]) if len(args) >= 2 else None)

    hub_connection.on_open(lambda: logger.info("SignalR connected") if logger else None)
    hub_connection.on_close(lambda: logger.warning("SignalR disconnected") if logger else None)
    hub_connection.on_error(lambda e: logger.error(f"SignalR error: {e}") if logger else None)

    # Start connection
    hub_connection.start()

    # Wait for connection
    await asyncio.sleep(2)

    # Subscribe to all contracts
    for symbol, contract_id in CONTRACTS.items():
        try:
            hub_connection.send("SubscribeContractMarketDepth", [contract_id])
            if logger:
                logger.info(f"Subscribed to {symbol.upper()} ({contract_id}) market depth")
            await asyncio.sleep(0.5)
        except Exception as e:
            if logger:
                logger.error(f"Failed to subscribe {symbol}: {e}")

    return hub_connection

# =============================================================================
# MAIN
# =============================================================================

async def main():
    global running, total_stats

    setup_logging()

    print("=" * 70)
    print("     L2 MARKET DEPTH DATA COLLECTOR - ALL SYMBOLS")
    print("=" * 70)
    print(f"  Contracts:")
    for sym, cid in CONTRACTS.items():
        print(f"    - {sym.upper()}: {cid}")
    print(f"  Snapshot interval: {SNAPSHOT_INTERVAL_SECONDS}s")
    print(f"  Data directory: {DATA_DIR}")
    print("=" * 70)

    # Authenticate
    print("\nAuthenticating with TopstepX...")
    try:
        authenticate()
        print("  ✓ Authentication successful")
    except Exception as e:
        print(f"  ✗ Authentication failed: {e}")
        return

    total_stats['start_time'] = datetime.utcnow()

    # Start background threads
    saver_thread = Thread(target=data_saver_thread, daemon=True)
    saver_thread.start()

    display_thread = Thread(target=stats_display_thread, daemon=True)
    display_thread.start()

    # Connect to SignalR
    print("\nConnecting to market data websocket...")
    try:
        hub = await connect_signalr()
        print("  ✓ Connected and subscribed to all symbols")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("\nFalling back to REST API polling...")
        await poll_market_depth_rest()
        return

    print("\n" + "=" * 70)
    print("  Collecting L2 data 24/7... Press Ctrl+C to stop")
    print("=" * 70 + "\n")

    # Keep running
    while running:
        await asyncio.sleep(1)

        # Re-authenticate if needed
        if time.time() >= token_expiry - 3600:
            try:
                authenticate()
            except:
                pass

    hub.stop()

async def poll_market_depth_rest():
    """Fallback: Poll market depth via REST API."""
    global running

    print("\nUsing REST API polling (1 second interval)...")

    while running:
        token = authenticate()

        for symbol, contract_id in CONTRACTS.items():
            try:
                response = requests.post(
                    f"{TOPSTEPX_BASE_URL}/api/History/retrieveMarketDepth",
                    json={"contractId": contract_id},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get('success') and data.get('depth'):
                        handle_market_depth(contract_id, data['depth'])
            except Exception as e:
                if logger:
                    logger.debug(f"REST poll error for {symbol}: {e}")

        await asyncio.sleep(1)

def signal_handler(signum, frame):
    global running
    print("\n\nShutdown signal received...")
    running = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        # Print final stats
        print("\n\n" + "=" * 70)
        print("  L2 COLLECTOR STOPPED")
        print("=" * 70)
        for sym in CONTRACTS:
            s = symbol_data[sym]['stats']
            print(f"  {sym.upper()}: {s['raw_updates']:,} updates, {s['snapshots']:,} snapshots")
        total_bytes = sum(symbol_data[s]['stats']['bytes'] for s in CONTRACTS)
        print(f"  Total data: {total_bytes / (1024*1024):.2f} MB")
        print("=" * 70)
