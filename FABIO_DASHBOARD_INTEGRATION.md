# Fabio LLM Agent + NQ ICT Dashboard Integration

This integration connects your Python Fabio LLM trading agent to the existing NQ ICT dashboard, allowing you to visualize Fabio's trades in real-time using the same UI.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BROWSER                               │
│                 nq-ict-dashboard.html                        │
│           (Chart, Orderbook, Positions, Trades)              │
└────────────────────┬────────────────────────────────────────┘
                     │ Socket.IO (events: status, log, trade)
                     │
┌────────────────────▼────────────────────────────────────────┐
│               Node.js Server                                 │
│         live-topstepx-nq-ict.ts (port 3337)                 │
│                                                              │
│  • Serves dashboard HTML                                     │
│  • Provides chart data (bars, ticks)                         │
│  • Provides market depth / orderbook                         │
│  • Relays Socket.IO events from Python                       │
└────────────────────┬────────────────────────────────────────┘
                     │ Socket.IO client connection
                     │
┌────────────────────▼────────────────────────────────────────┐
│            Python Socket.IO Bridge                           │
│              dashboard_bridge.py                             │
│                                                              │
│  • Emits position updates                                    │
│  • Emits trade notifications                                 │
│  • Emits log messages                                        │
│  • Emits PnL and statistics                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Fabio LLM Agent                                 │
│         fabio_dashboard.py (integrated)                      │
│                                                              │
│  • TopstepClient: market data streaming                      │
│  • FeatureEngine: compute features                           │
│  • LLMClient: OpenAI decision making                         │
│  • ExecutionEngine: trade execution                          │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. `dashboard_bridge.py`
Socket.IO client that connects to the Node.js server and emits Fabio's trading data in the format expected by the dashboard.

**Key Events:**
- `status` - Position, PnL, account stats, closed trades
- `log` - Activity log messages
- `trade` - Individual closed trade notifications

### 2. `fabio_dashboard.py`
Integration layer that runs the Fabio agent and monitors execution state, broadcasting updates to the dashboard via the bridge.

### 3. Node.js Server (existing)
`live-topstepx-nq-ict.ts` continues to:
- Serve the dashboard HTML
- Provide chart data (bars, ticks)
- Provide market depth / orderbook
- Provide trade data for CVD calculation

The ICT strategy logic in the Node server is **not used** - Fabio makes all trading decisions.

### 4. Dashboard (existing)
`public/nq-ict-dashboard.html` displays:
- Real-time price chart with orderbook overlay
- CVD chart
- Current position
- Recent trades
- Activity log
- Account statistics

## Installation

### 1. Install Python Dependencies

```bash
pip install python-socketio[asyncio_client] aiohttp
```

Or add to your `requirements.txt`:
```
python-socketio[asyncio_client]>=5.11.0
aiohttp>=3.9.0
```

### 2. Verify Existing Setup

Make sure you have:
- ✅ Fabio agent Python files (engine.py, topstep_client.py, etc.)
- ✅ Node.js dashboard server (live-topstepx-nq-ict.ts)
- ✅ Dashboard HTML (public/nq-ict-dashboard.html)
- ✅ TopstepX credentials in .env

## Usage

### Step 1: Start the Node.js Dashboard Server

```bash
npx tsx live-topstepx-nq-ict.ts
```

This will:
- Start the HTTP server on port 3337
- Connect to TopstepX for real-time market data
- Serve the dashboard at http://localhost:3337

### Step 2: Start Fabio Agent with Dashboard Integration

```bash
python fabio_dashboard.py --symbol NQZ5 --mode paper_trading
```

Or for live trading:
```bash
python fabio_dashboard.py --symbol NQZ5 --mode live_trading
```

Optional arguments:
- `--dashboard-url` - Dashboard server URL (default: http://localhost:3337)
- `--symbol` - Trading symbol (default: from env TRADING_SYMBOL)
- `--mode` - Trading mode (default: from env TRADING_MODE)

### Step 3: Open the Dashboard

Open your browser to:
```
http://localhost:3337
```

You should see:
- Real-time price chart with orderbook
- CVD chart
- Fabio's positions and trades displayed in real-time
- Activity log showing Fabio's decisions
- Account statistics

## What You'll See

### Activity Log
```
[10:32:15] Fabio agent started
[10:32:15] Trading NQZ5 in paper_trading mode
[10:32:16] Fabio agent connected
[10:32:20] Price: 21000.00
[10:32:25] Requesting LLM decision...
[10:32:28] LLM response received
[10:32:28] Position opened: LONG 3 @ 21000.00
```

### Position Display
```
Entry: $21000.00          Pattern: Fabio LLM Decision
Stop Loss: $20996.00      TP1/TP2: $21004.00 / $21008.00
Contracts: 3              Unrealized: $0.00
```

### Trade History
```
L  +$160  CLOSED @ $21004.00
S  -$80   CLOSED @ $20998.00
L  +$240  CLOSED @ $21008.00
```

## How It Works

1. **Market Data Flow (Node → Dashboard)**
   - TopstepX sends quotes, trades, depth to Node server
   - Node server broadcasts to dashboard via Socket.IO
   - Dashboard renders charts, orderbook, CVD

2. **Trading Logic (Python Fabio Agent)**
   - TopstepClient streams market data to Python
   - FeatureEngine computes features
   - LLMClient (OpenAI) makes trading decisions
   - ExecutionEngine executes trades

3. **Status Updates (Python → Node → Dashboard)**
   - Fabio dashboard integration monitors ExecutionEngine
   - Detects position changes, closed trades, PnL updates
   - Emits Socket.IO events to Node server
   - Node server relays to dashboard
   - Dashboard updates UI in real-time

## Configuration

### Dashboard Port
The Node server runs on port 3337 by default. To change:

1. Set environment variable:
   ```bash
   export TOPSTEPX_NQ_ICT_DASHBOARD_PORT=3338
   ```

2. Update Python command:
   ```bash
   python fabio_dashboard.py --dashboard-url http://localhost:3338
   ```

### Update Frequency
Status updates are sent every 1 second by default. To change, edit `fabio_dashboard.py`:

```python
self.status_update_interval = 2.0  # seconds
```

## Troubleshooting

### Dashboard shows "OFFLINE"
- Check that Node server is running on port 3337
- Check browser console for connection errors
- Verify firewall allows port 3337

### Python can't connect to dashboard
```
[DashboardBridge] Failed to connect: Connection refused
```

**Solution:**
1. Make sure Node server is running first
2. Check the port matches (default 3337)
3. Try: `curl http://localhost:3337/api/strategy/config`

### No position updates shown
- Check that Fabio agent is making trading decisions
- Check Python console for log messages
- Check dashboard Activity Log for connection status
- Verify Socket.IO events in browser DevTools (Network → WS)

### TypeScript/Node errors
If you see SignalR or WebSocket errors in the Node server, make sure:
- TopstepX credentials are valid in .env
- Internet connection is active
- TopstepX API is accessible

## Testing Without Market Data

To test the dashboard integration without live market data:

```bash
python dashboard_bridge.py
```

This runs the example in `dashboard_bridge.py` which:
- Connects to the dashboard
- Sends a simulated position
- Updates unrealized PnL every 5 seconds

## Advanced: Customizing the Dashboard

The dashboard HTML expects these Socket.IO events:

### `status` event
```javascript
{
  symbol: "NQZ5",
  isTrading: true,
  position: {
    side: "long",
    entryPrice: 21000.00,
    stopLoss: 20996.00,
    targetTP1: 21004.00,
    targetTP2: 21008.00,
    totalQty: 3,
    remaining: 3,
    unrealizedPnL: 50.00,
    entryPattern: "Fabio LLM Decision"
  },
  closedTrades: [...],
  accountStats: {
    totalTrades: 10,
    winners: 7,
    losers: 3,
    winRate: 70.0,
    totalPnL: 850.00
  }
}
```

### `log` event
```javascript
{
  timestamp: "2025-01-17T10:32:15.123Z",
  message: "Position opened: LONG 3 @ 21000.00",
  type: "success"  // "info", "success", "warning", "error"
}
```

### `trade` event
```javascript
{
  side: "long",
  entryPrice: 21000.00,
  exitPrice: 21004.00,
  pnl: 160.00,
  exitReason: "tp1",
  entryPattern: "Fabio LLM"
}
```

## Next Steps

1. **Add More Metrics**: Extend `dashboard_bridge.py` to emit additional metrics (Sharpe ratio, max drawdown, etc.)

2. **Real-time Charts**: Modify the dashboard to display Fabio's importance zones or LLM confidence scores

3. **Control Panel**: Add start/stop buttons in the dashboard to control the Python agent

4. **Multi-Symbol**: Run multiple Fabio agents for different symbols, each with its own dashboard

## Support

For issues specific to:
- **Dashboard UI**: Check `public/nq-ict-dashboard.html`
- **Node server**: Check `live-topstepx-nq-ict.ts`
- **Python bridge**: Check `dashboard_bridge.py`
- **Fabio integration**: Check `fabio_dashboard.py`
