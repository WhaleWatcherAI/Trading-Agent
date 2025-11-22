#!/usr/bin/env python3
"""
Minimal Socket.IO Server for Fabio Agent Dashboard
Uses synthetic/demo market data to avoid SignalR connection issues
"""

import asyncio
import os
import math
import time
from aiohttp import web
import socketio

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='aiohttp',
    cors_allowed_origins='*'
)

app = web.Application()
sio.attach(app)

# Global state
trading_active = False
main_task = None
account_balance = 50000.0
daily_pnl = 0.0
trades_today = 0

# Synthetic market data
class SyntheticMarket:
    """Generates realistic synthetic market data for demo/paper trading"""
    def __init__(self, initial_price=24750.0):
        self.price = initial_price
        self.last_update = time.time()
        self.bar_open = initial_price
        self.bar_high = initial_price
        self.bar_low = initial_price
        self.bar_volume = 0
        self.last_trade_time = time.time()

    def update(self):
        """Generate next price tick using random walk"""
        now = time.time()
        # Tick every 100ms approx
        if now - self.last_trade_time < 0.1:
            return None

        self.last_trade_time = now

        # Random walk with slight drift
        change = (hash(int(now * 100)) % 100 - 50) / 10000.0
        self.price += change

        # Update bar
        self.bar_high = max(self.bar_high, self.price)
        self.bar_low = min(self.bar_low, self.price)
        self.bar_volume += 1.0

        # Reset bar every second
        bar_second = int(now)
        if int(self.last_update) != bar_second:
            bar_data = {
                'timestamp': self.last_update,
                'open': self.bar_open,
                'high': self.bar_high,
                'low': self.bar_low,
                'close': self.price,
                'volume': self.bar_volume,
            }
            self.bar_open = self.price
            self.bar_high = self.price
            self.bar_low = self.price
            self.bar_volume = 0
            self.last_update = now
            return bar_data

        return None

market = SyntheticMarket()
bars_history = []

@sio.event
async def connect(sid, environ):
    """Client connected"""
    print(f'[SocketIO] Client connected: {sid}')
    await sio.emit('status', {
        'isTrading': trading_active,
        'accountStats': {
            'balance': account_balance,
            'dailyPnL': daily_pnl,
            'tradesCount': trades_today
        }
    }, room=sid)

@sio.event
async def disconnect(sid):
    """Client disconnected"""
    print(f'[SocketIO] Client disconnected: {sid}')

@sio.event
async def request_chart_history(sid):
    """Client requesting historical chart data"""
    print(f'[SocketIO] Client {sid} requesting chart history')
    # Send last 100 bars
    await sio.emit('chart_history', bars_history[-100:], room=sid)

@sio.event
async def start_trading(sid):
    """Client requested to start trading"""
    global trading_active, main_task

    print(f'[SocketIO] Client {sid} requested to start trading')

    if not trading_active:
        trading_active = True
        await sio.emit('log', {
            'message': 'Starting Fabio Agent...',
            'type': 'info'
        })

        main_task = asyncio.create_task(run_fabio_agent())

        await sio.emit('status', {
            'isTrading': True,
            'accountStats': {
                'balance': account_balance,
                'dailyPnL': daily_pnl,
                'tradesCount': trades_today
            }
        })

@sio.event
async def stop_trading(sid):
    """Client requested to stop trading"""
    global trading_active, main_task

    print(f'[SocketIO] Client {sid} requested to stop trading')

    if trading_active:
        trading_active = False
        await sio.emit('log', {
            'message': 'Stopping Fabio Agent...',
            'type': 'warning'
        })

        if main_task:
            main_task.cancel()
            try:
                await main_task
            except asyncio.CancelledError:
                pass

        await sio.emit('status', {
            'isTrading': False,
            'accountStats': {
                'balance': account_balance,
                'dailyPnL': daily_pnl,
                'tradesCount': trades_today
            }
        })

async def run_fabio_agent():
    """Main Fabio agent trading loop"""
    global account_balance, daily_pnl, trades_today

    try:
        await sio.emit('log', {
            'message': 'Fabio Agent initialized',
            'type': 'success'
        })

        await sio.emit('log', {
            'message': f'Account balance: ${account_balance:,.2f}',
            'type': 'success'
        })

        await sio.emit('log', {
            'message': 'Generating synthetic market data...',
            'type': 'info'
        })

        tick_count = 0
        last_llm_call = time.time()

        while trading_active:
            bar_data = market.update()

            # Emit bar update
            if bar_data:
                bars_history.append(bar_data)
                await sio.emit('bar', bar_data)

            # Emit tick update every 100ms
            await sio.emit('tick', {
                'timestamp': time.time(),
                'price': market.price,
                'bid': market.price - 0.25,
                'ask': market.price + 0.25,
            })

            tick_count += 1

            # Update status every 10 ticks
            if tick_count % 10 == 0:
                await sio.emit('status', {
                    'isTrading': trading_active,
                    'accountStats': {
                        'balance': account_balance,
                        'dailyPnL': daily_pnl,
                        'tradesCount': trades_today
                    }
                })

            # Call "LLM" every 5 seconds
            now = time.time()
            if now - last_llm_call >= 5.0:
                last_llm_call = now

                # Simulate LLM decision-making
                rsi = 50 + 20 * math.sin(tick_count / 50.0)
                bb_distance = abs(math.sin(tick_count / 100.0))

                await sio.emit('log', {
                    'message': f'📊 Market Analysis: RSI={rsi:.1f}, BB_Dist={bb_distance:.2f}',
                    'type': 'info'
                })

                # Simulate trade decision
                if rsi > 55 and bb_distance > 0.5:
                    await sio.emit('log', {
                        'message': '🤖 LLM Decision: SHORT signal detected (overbought)',
                        'type': 'warning'
                    })
                elif rsi < 45 and bb_distance > 0.5:
                    await sio.emit('log', {
                        'message': '🤖 LLM Decision: LONG signal detected (oversold)',
                        'type': 'warning'
                    })
                else:
                    await sio.emit('log', {
                        'message': '🤖 LLM Decision: Hold - waiting for better setup',
                        'type': 'info'
                    })

            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        await sio.emit('log', {
            'message': 'Fabio Agent stopped',
            'type': 'warning'
        })
    except Exception as e:
        await sio.emit('log', {
            'message': f'Fabio Agent error: {str(e)}',
            'type': 'error'
        })
        import traceback
        traceback.print_exc()

# Serve the dashboard HTML
async def index(request):
    """Serve the Fabio Agent dashboard"""
    with open('public/fabio-agent-dashboard.html', 'r') as f:
        html_content = f.read()
    return web.Response(text=html_content, content_type='text/html')

# Static files
app.router.add_static('/public', 'public')
app.router.add_get('/', index)

async def init_app():
    """Initialize the application"""
    print('=' * 80)
    print('🧠 FABIO AGENT - MINIMAL SOCKET.IO SERVER')
    print('=' * 80)
    print('Mode: Synthetic market data (paper trading)')
    print('Dashboard: http://localhost:3337')
    print('=' * 80)
    return app

if __name__ == '__main__':
    app = asyncio.run(init_app())
    web.run_app(app, host='0.0.0.0', port=3337)
