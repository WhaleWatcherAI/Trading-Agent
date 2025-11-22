#!/usr/bin/env python3
"""
Socket.IO Server Bridge for Fabio Agent Dashboard
Connects the Python Fabio agent (engine_enhanced.py) to the web dashboard
"""

import asyncio
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from aiohttp import web
import socketio
import json

from config import Settings
from topstep_client import TopstepClient, MarketSnapshot
from features_enhanced import EnhancedFeatureEngine
from llm_client_enhanced import EnhancedLLMClient
from execution_enhanced import EnhancedExecutionEngine
from storage import save_llm_exchange


# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='aiohttp',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)

app = web.Application()
sio.attach(app)

# Global state
settings = None
topstep_client = None
feature_engine = None
llm_client = None
execution_engine = None
trading_active = False
main_task = None
last_price: float = 0.0
closed_trade_index: int = 0


def _compute_account_stats() -> Dict[str, Any]:
    """Build aggregated account statistics for dashboard/status events."""
    if execution_engine is None:
        base_balance = settings.account_balance if settings else 42000.0
        return {
            "balance": base_balance,
            "dailyPnL": 0.0,
            "tradesCount": 0,
            "totalTrades": 0,
            "winners": 0,
            "losers": 0,
            "winRate": 0.0,
            "totalPnL": 0.0,
        }

    closed = execution_engine.closed_trades
    winners = [t for t in closed if t.get("pnl", 0.0) > 0]
    losers = [t for t in closed if t.get("pnl", 0.0) <= 0]
    total_pnl = sum(t.get("pnl", 0.0) for t in closed) if closed else 0.0
    win_rate = (len(winners) / len(closed) * 100.0) if closed else 0.0

    return {
        "balance": execution_engine.account_balance,
        "dailyPnL": execution_engine.daily_pnl,
        "tradesCount": execution_engine.trades_today,
        "totalTrades": len(closed),
        "winners": len(winners),
        "losers": len(losers),
        "winRate": win_rate,
        "totalPnL": total_pnl,
    }


def _format_position_for_dashboard() -> Optional[Dict[str, Any]]:
    """Convert current open position (if any) into NQ Winner dashboard format."""
    if execution_engine is None or not execution_engine.positions:
        return None

    # Single-position display: take the first active position
    pos = next(iter(execution_engine.positions.values()))

    # Use latest known price for unrealized PnL if available
    if last_price and last_price > 0:
        if pos.side == "long":
            unrealized = (last_price - pos.entry_price) * pos.size * 5.0
        else:
            unrealized = (pos.entry_price - last_price) * pos.size * 5.0
    else:
        unrealized = 0.0

    return {
        "side": pos.side.upper(),
        "contracts": int(pos.size),
        "entryPrice": pos.entry_price,
        "currentPrice": last_price or pos.entry_price,
        "stopPrice": pos.stop_price,
        "targetPrice": pos.target_price,
        "unrealizedPnl": unrealized,
        "entryTime": datetime.fromtimestamp(pos.entry_time).isoformat(),
    }


def _format_closed_trades(limit: int = 20) -> List[Dict[str, Any]]:
    """Format closed trades list for dashboard consumption."""
    if execution_engine is None:
        return []

    trades = execution_engine.closed_trades[-limit:]
    formatted: List[Dict[str, Any]] = []
    for t in trades:
        formatted.append(
            {
                "side": t.get("side", "long").upper(),
                "entryPrice": t.get("entry_price", 0.0),
                "exitPrice": t.get("exit_price", 0.0),
                "entryTime": datetime.fromtimestamp(t.get("entry_time", 0.0)).isoformat()
                if t.get("entry_time") is not None
                else "",
                "exitTime": datetime.fromtimestamp(t.get("close_time", 0.0)).isoformat()
                if t.get("close_time") is not None
                else "",
                "pnl": t.get("pnl", 0.0),
                "exitReason": t.get("reason", "closed"),
            }
        )
    return formatted


@sio.event
async def connect(sid, environ):
    """Client connected to dashboard"""
    print(f'[SocketIO] Client connected: {sid}')

    # Send initial status
    await sio.emit(
        'status',
        {
            'isTrading': trading_active,
            'accountStats': _compute_account_stats(),
            'position': _format_position_for_dashboard(),
            'closedTrades': _format_closed_trades(),
        },
        room=sid,
    )


@sio.event
async def disconnect(sid):
    """Client disconnected"""
    print(f'[SocketIO] Client disconnected: {sid}')


@sio.event
async def request_chart_history(sid):
    """Client requesting historical chart data"""
    print(f'[SocketIO] Client {sid} requesting chart history')

    # For now, send empty history - data will come from live stream
    await sio.emit('chart_history', [], room=sid)


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

        # Start the trading engine
        main_task = asyncio.create_task(run_fabio_agent())

        await sio.emit('status', {
            'isTrading': True,
            'accountStats': _compute_account_stats(),
            'position': _format_position_for_dashboard(),
            'closedTrades': _format_closed_trades(),
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

        # Cancel the trading task
        if main_task:
            main_task.cancel()
            try:
                await main_task
            except asyncio.CancelledError:
                pass

        await sio.emit('status', {
            'isTrading': False,
            'accountStats': _compute_account_stats(),
            'position': _format_position_for_dashboard(),
            'closedTrades': _format_closed_trades(),
        })


@sio.event
async def flatten_positions(sid):
    """Client requested to flatten all open positions at market (using last price)."""
    global execution_engine

    print(f'[SocketIO] Client {sid} requested to flatten positions')

    if execution_engine is None or not execution_engine.positions:
        await sio.emit(
            'log',
            {
                'message': 'No open positions to flatten',
                'type': 'info',
            },
            room=sid,
        )
        return

    # Use last known price as proxy for market exit
    exit_price = last_price or next(iter(execution_engine.positions.values())).entry_price

    # Close all positions
    for pos_id in list(execution_engine.positions.keys()):
        execution_engine._close_position(pos_id, exit_price, 'manual_flatten')

    await sio.emit(
        'log',
        {
            'message': 'All positions flattened manually',
            'type': 'warning',
        },
    )


async def run_fabio_agent():
    """Main Fabio agent trading loop"""
    global topstep_client, feature_engine, llm_client, execution_engine

    try:
        await sio.emit('log', {
            'message': 'Fabio Agent initialized',
            'type': 'success'
        })

        # Initialize components
        topstep_client = TopstepClient(settings)
        feature_engine = EnhancedFeatureEngine()
        llm_client = EnhancedLLMClient(settings)

        # Fetch account balance
        await sio.emit('log', {
            'message': 'Fetching account balance from TopStep...',
            'type': 'info'
        })

        account_info = await topstep_client.get_account_info()
        account_balance = account_info.get('balance', settings.account_balance)

        await sio.emit('log', {
            'message': f'Account balance: ${account_balance:,.2f}',
            'type': 'success'
        })

        execution_engine = EnhancedExecutionEngine(
            settings=settings,
            topstep_client=topstep_client,
            account_balance=account_balance
        )

        execution_engine.initialize_default_strategies()

        # Start streaming market data
        await sio.emit('log', {
            'message': 'Connecting to market data stream...',
            'type': 'info'
        })

        bar_count = 0
        last_llm_call = None
        importance_zones = []

        async for snapshot in topstep_client.stream_snapshots():
            if not trading_active:
                break

            global last_price, closed_trade_index

            bar_count += 1
            last_price = snapshot.bar.close
            now = snapshot.timestamp

            # Update features
            feat_state = feature_engine.update_features_and_get_state(snapshot)

            # Check stops and targets
            execution_engine.check_stops_and_targets(last_price)

            # Emit advanced Fabio features to dashboard
            # 1. Volume Profile Data
            if 'profiles' in feat_state and len(feat_state['profiles']) > 0:
                profile = feat_state['profiles'][0]
                volume_profile = {
                    'poc': profile.get('poc', 0),
                    'vah': profile.get('vah', 0),
                    'val': profile.get('val', 0),
                    'lvns': profile.get('lvns', []),
                    'session_high': feat_state.get('market_stats', {}).get('session_high', last_price),
                    'session_low': feat_state.get('market_stats', {}).get('session_low', last_price),
                }
                await sio.emit('volume_profile', volume_profile)

            # 2. CVD and Order Flow Indicators
            orderflow = feat_state.get('orderflow', {})
            cvd_data = {
                'cvd_value': feature_engine.cvd_value if hasattr(feature_engine, 'cvd_value') else 0,
                'cvd_trend': orderflow.get('cvd_trend', 'neutral'),
                'buy_absorption': orderflow.get('buy_absorption_score', 0),
                'sell_absorption': orderflow.get('sell_absorption_score', 0),
                'buy_exhaustion': orderflow.get('buy_exhaustion_score', 0),
                'sell_exhaustion': orderflow.get('sell_exhaustion_score', 0),
                'big_prints': orderflow.get('big_prints', []),
            }
            await sio.emit('cvd', cvd_data)

            # 3. L2 Order Book Data
            if snapshot.l2:
                l2_data = {
                    'bids': [[level.price, level.bid_size] for level in snapshot.l2 if level.bid_size > 0][:10],
                    'asks': [[level.price, level.ask_size] for level in snapshot.l2 if level.ask_size > 0][:10],
                    'spread': snapshot.l1.ask - snapshot.l1.bid if snapshot.l1.ask and snapshot.l1.bid else 0,
                }
                await sio.emit('l2_data', l2_data)

            # 4. Market State Indicator
            derived_state = feat_state.get('derived_state', {})
            market_state = feat_state.get('market_stats', {})
            market_state_data = {
                'state': market_state.get('market_state_flag', 'balanced'),
                'range_condition': market_state.get('range_condition', 'normal'),
                'location_vs_value': derived_state.get('location_vs_value', 'inside'),
                'location_vs_poc': derived_state.get('location_vs_poc', 'at_poc'),
                'buyers_control': derived_state.get('buyers_in_control_score', 0),
                'sellers_control': derived_state.get('sellers_in_control_score', 0),
                'poc_crosses': market_state.get('poc_cross_counts', {}).get('count_last_5min', 0) if 'poc_cross_counts' in market_state else 0,
                'time_in_value': market_state.get('time_in_value_sec', 0) if 'time_in_value_sec' in market_state else 0,
            }
            await sio.emit('market_state', market_state_data)

            # 5. Absorption/Exhaustion Signals
            if orderflow.get('buy_absorption_score', 0) > 0.7 or orderflow.get('sell_absorption_score', 0) > 0.7:
                await sio.emit('absorption', {
                    'type': 'buy' if orderflow.get('buy_absorption_score', 0) > 0.7 else 'sell',
                    'strength': max(orderflow.get('buy_absorption_score', 0), orderflow.get('sell_absorption_score', 0)),
                    'price': last_price,
                    'timestamp': datetime.fromtimestamp(now).isoformat(),
                })

            if orderflow.get('buy_exhaustion_score', 0) > 0.7 or orderflow.get('sell_exhaustion_score', 0) > 0.7:
                await sio.emit('exhaustion', {
                    'type': 'buy' if orderflow.get('buy_exhaustion_score', 0) > 0.7 else 'sell',
                    'strength': max(orderflow.get('buy_exhaustion_score', 0), orderflow.get('sell_exhaustion_score', 0)),
                    'price': last_price,
                    'timestamp': datetime.fromtimestamp(now).isoformat(),
                })

            # Emit bar update to dashboard every second
            bar_data = {
                'timestamp': datetime.fromtimestamp(snapshot.timestamp).isoformat(),
                'open': snapshot.bar.open,
                'high': snapshot.bar.high,
                'low': snapshot.bar.low,
                'close': snapshot.bar.close,
                'volume': snapshot.bar.volume,
            }

            await sio.emit('bar', bar_data)

            # Emit tick update
            await sio.emit('tick', {
                'timestamp': bar_data['timestamp'],
                'open': snapshot.bar.open,
                'high': snapshot.bar.high,
                'low': snapshot.bar.low,
                'close': snapshot.bar.close,
            })

            # After price/feature updates, emit trade events for any newly closed trades
            if execution_engine is not None:
                new_trades = execution_engine.closed_trades[closed_trade_index:]
                for t in new_trades:
                    trade_payload = {
                        "side": t.get("side", "long").upper(),
                        "entryPrice": t.get("entry_price", 0.0),
                        "exitPrice": t.get("exit_price", 0.0),
                        "entryTime": datetime.fromtimestamp(t.get("entry_time", 0.0)).isoformat()
                        if t.get("entry_time") is not None
                        else "",
                        "exitTime": datetime.fromtimestamp(t.get("close_time", 0.0)).isoformat()
                        if t.get("close_time") is not None
                        else "",
                        "pnl": t.get("pnl", 0.0),
                        "exitReason": t.get("reason", "closed"),
                    }
                    await sio.emit('trade', trade_payload)

                closed_trade_index += len(new_trades)

            # Update status every 10 bars (includes account stats and open position)
            if bar_count % 10 == 0:
                await sio.emit(
                    'status',
                    {
                        'isTrading': trading_active,
                        'accountStats': _compute_account_stats(),
                        'position': _format_position_for_dashboard(),
                        'closedTrades': _format_closed_trades(),
                    },
                )

            # Call LLM periodically (every 30 seconds for now)
            should_call_llm = (
                last_llm_call is None or
                (now - last_llm_call) >= 30
            )

            if should_call_llm:
                await sio.emit('log', {
                    'message': 'Calling LLM for trading decision...',
                    'type': 'info'
                })

                # Get current performance
                current_performance = execution_engine.get_performance_stats()

                # Build payload for LLM
                payload = {
                    'mode': 'live_decision',
                    'symbol': settings.symbol,
                    'timestamp': now,
                    'price': last_price,
                    'market_stats': feat_state.get('market_stats', {}),
                    'profiles': feat_state.get('profiles', []),
                    'orderflow': feat_state.get('orderflow', {}),
                    'open_positions': [
                        {
                            'id': pos.id,
                            'side': pos.side,
                            'entry_price': pos.entry_price,
                            'size': pos.size,
                            'stop_price': pos.stop_price,
                            'target_price': pos.target_price,
                            'unrealized_pnl': pos.unrealized_pnl,
                            'strategy': pos.strategy,
                            'age_seconds': now - pos.entry_time,
                        }
                        for pos in execution_engine.positions.values()
                    ],
                    'strategy_state': execution_engine.strategy_state,
                    'strategy_performance': current_performance,
                    'importance_zones': importance_zones,
                    'session_stats': {
                        'daily_pnl': execution_engine.daily_pnl,
                        'trades_today': execution_engine.trades_today,
                        'max_daily_loss': execution_engine.max_daily_loss,
                    }
                }

                try:
                    llm_response = await llm_client.request_decision(payload)
                    last_llm_call = now

                    # Log the exchange
                    save_llm_exchange(settings.symbol, payload, llm_response, settings.llm_log_path)

                    # Emit LLM decision to dashboard
                    await sio.emit('log', {
                        'message': f'LLM Decision: {llm_response.get("market_assessment", {}).get("regime", "unknown")} regime',
                        'type': 'success'
                    })

                    # Emit detailed LLM decision for the decision viewer
                    await sio.emit('llm_decision', {
                        'timestamp': datetime.fromtimestamp(now).isoformat(),
                        'market_assessment': llm_response.get('market_assessment', {}),
                        'trade_decisions': llm_response.get('trade_decisions', []),
                        'reasoning': llm_response.get('reasoning', ''),
                        'notes': llm_response.get('notes_to_future_self', []),
                        'confidence': llm_response.get('confidence_score', 0),
                        'strategy_updates': llm_response.get('strategy_updates', {}),
                        'importance_zones': llm_response.get('importance_zones', []),
                    })

                    # Apply strategy updates
                    strategy_updates = llm_response.get('strategy_updates', {})
                    if strategy_updates:
                        execution_engine.apply_strategy_updates(strategy_updates)
                        await sio.emit('log', {
                            'message': 'Strategy parameters updated by LLM',
                            'type': 'info'
                        })

                    # Process trade decisions
                    for decision in llm_response.get('trade_decisions', []):
                        result = await execution_engine.process_trade_decision(decision, last_price)
                        await sio.emit('log', {
                            'message': f'Trade decision: {result}',
                            'type': 'info'
                        })

                    # Update importance zones
                    new_zones = llm_response.get('importance_zones', [])
                    if new_zones:
                        importance_zones = new_zones

                except Exception as e:
                    await sio.emit('log', {
                        'message': f'LLM error: {str(e)}',
                        'type': 'error'
                    })

            # Small delay
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
    """Serve the Fabio TopStep dashboard with advanced features"""
    with open('public/fabio-topstep-dashboard.html', 'r') as f:
        html_content = f.read()
    return web.Response(text=html_content, content_type='text/html')


# Static files
app.router.add_static('/public', 'public')
app.router.add_get('/', index)


async def init_app():
    """Initialize the application"""
    global settings

    settings = Settings.from_env()

    print('=' * 80)
    print('🧠 FABIO AGENT - SOCKET.IO DASHBOARD SERVER')
    print('=' * 80)
    print(f'Symbol: {settings.symbol}')
    print(f'Mode: {settings.mode}')
    print(f'Dashboard: http://localhost:3337')
    print('=' * 80)

    return app


if __name__ == '__main__':
    app = asyncio.run(init_app())
    web.run_app(app, host='0.0.0.0', port=3337)
