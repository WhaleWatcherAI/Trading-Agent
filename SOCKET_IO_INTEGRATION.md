# Socket.IO Integration Patterns - Live Trading Dashboard

## Overview
The trading system uses Socket.IO (v4.6.1) for real-time bidirectional communication between live trading servers and dashboard clients. Each strategy runs on its own port and emits market data, position updates, and trade events to connected dashboards.

---

## Port Configuration & Connection Details

### Multi-Symbol Setup Ports
- **MNQ (NQ Winner Enhanced)**: Port 3333 (env: `TOPSTEPX_NQ_DASHBOARD_PORT`)
- **MES (MES Winner)**: Port 3334 (env: `TOPSTEPX_NQ_DASHBOARD_PORT`)
- **MGC (MGC Winner)**: Port 3335 (env: `TOPSTEPX_NQ_DASHBOARD_PORT`)
- **M6E (M6E Winner)**: Port 3336 (env: `TOPSTEPX_NQ_DASHBOARD_PORT`)
- **MGC PO3**: Port 3006 (env: `TOPSTEPX_MGC_DASHBOARD_PORT`)

### Server Setup
```typescript
import { Server } from 'socket.io';
import http from 'http';

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: '*' }
});

server.listen(DASHBOARD_PORT, () => {
  log(`Dashboard server running on http://localhost:${DASHBOARD_PORT}`);
});
```

---

## Socket Events Reference

### 1. CONNECT EVENT
**Direction**: Client → Server (automatic)

When a dashboard connects to the strategy server.

**Server Handler:**
```typescript
io.on('connection', (socket) => {
  log(`Dashboard client connected: ${socket.id}`);
  // ...sends initial data
});
```

---

### 2. CONFIG Event
**Direction**: Server → Client (on connection)

Sends initial strategy configuration to dashboard.

**Emitted From**: Server-side connection handler
**Event Name**: `config`

**Data Structure:**
```typescript
socket.emit('config', {
  symbol: string,           // e.g., "NQZ5", "MES", "MGC", "M6E"
  bbPeriod: number,         // e.g., 20 (Bollinger Band period)
  bbStdDev: number,         // e.g., 3 (standard deviation)
  rsiPeriod: number,        // e.g., 24
  rsiOversold: number,      // e.g., 30
  rsiOverbought: number     // e.g., 70
});
```

**Recipient Handlers** (in dashboard HTML):
```javascript
// No explicit handler - used by server during initial connection
socket.emit('config', {...})  // Server sends automatically
```

---

### 3. CHARTHISTORY Event
**Direction**: Bidirectional

Initial chart data sent on connection, or requested by client.

**Emitted From**: 
- Server: On connection and when client requests refresh
- Client: To request historical data

**Event Names**: `chartHistory`

**Data Structure:**
```typescript
// Array of ChartData objects
[
  {
    timestamp: string,        // ISO timestamp
    open: number,
    high: number,
    low: number,
    close: number,
    bbUpper?: number,         // Bollinger Band upper
    bbMiddle?: number,        // Bollinger Band middle (keep for compat)
    bbBasis?: number,         // Bollinger Band middle (dashboard expects this)
    bbLower?: number,         // Bollinger Band lower
    rsi?: number,             // Relative Strength Index
    ttmMomentum?: number,     // TTM Squeeze momentum value
    squeeze?: {
      momentum: number,
      squeezeFiring: boolean
    },
    signal?: 'long' | 'short' | null,
    entry?: boolean,
    exit?: boolean
  },
  // ... more bars
]
```

**Server Emission:**
```typescript
io.on('connection', (socket) => {
  const completeData = getCompleteChartData();  // Last 500 bars
  if (completeData.length > 0) {
    socket.emit('chartHistory', completeData);
  }
});

socket.on('chartHistory', () => {
  const data = getCompleteChartData();
  socket.emit('chartHistory', data);
});
```

**Dashboard Handler:**
```javascript
socket.on('chartHistory', (data) => {
  if (data && data.length > 0) {
    updateCharts(data);  // Updates all chart series
  }
});
```

---

### 4. BAR Event
**Direction**: Server → Client (continuous)

Sent whenever a new 1-minute bar completes with all calculated indicators.

**Event Name**: `bar`

**Data Structure:**
```typescript
{
  timestamp: string,        // ISO timestamp of bar close
  open: number,
  high: number,
  low: number,
  close: number,
  bbUpper: number,         // Upper Bollinger Band
  bbBasis: number,         // Middle Bollinger Band (MA)
  bbMiddle: number,        // For compatibility
  bbLower: number,         // Lower Bollinger Band
  rsi: number,             // RSI(24) value
  ttmMomentum: number,     // TTM Squeeze momentum
  squeeze: {
    momentum: number,
    squeezeFiring: boolean  // true = squeeze active
  },
  signal: null             // Set to 'long' or 'short' if signals detected
}
```

**Server Emission:**
```typescript
// After bar completes and indicators calculated
io.emit('bar', chartPoint);
```

**Dashboard Handler:**
```javascript
socket.on('bar', (bar) => {
  // Update candlestick
  if (candleSeries && bar) {
    candleSeries.update({
      time: Math.floor(new Date(bar.timestamp).getTime() / 1000),
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close
    });
  }
  
  // Update Bollinger Bands
  if (bar.bbUpper != null && bbUpperSeries) {
    bbUpperSeries.update({
      time: candleData.time,
      value: bar.bbUpper
    });
  }
  // ... similar for bbLower, bbBasis
  
  // Update RSI
  if (bar.rsi != null && rsiSeries) {
    rsiSeries.update({
      time: candleData.time,
      value: bar.rsi
    });
  }
  
  // Update TTM Squeeze momentum (histogram)
  if (bar.ttmMomentum != null && ttmMomentumSeries) {
    ttmMomentumSeries.update({
      time: candleData.time,
      value: bar.ttmMomentum,
      color: bar.ttmMomentum >= 0 ? '#22c55e' : '#ef4444'
    });
  }
  
  // Update squeeze indicator (yellow dot when ON, green when OFF)
  if (bar.squeeze && ttmSqueezeSeries) {
    ttmSqueezeSeries.update({
      time: candleData.time,
      value: 0,  // Always at zero line
      color: bar.squeeze.squeezeFiring ? '#eab308' : '#22c55e'
    });
  }
});
```

---

### 5. TICK Event
**Direction**: Server → Client (high frequency)

Intra-bar tick updates - sends current bar OHLC without indicators (lighter than bar event).

**Event Name**: `tick`

**Data Structure:**
```typescript
{
  timestamp: string,  // ISO timestamp
  open: number,
  high: number,
  low: number,
  close: number
  // No indicators - lightweight for frequent updates
}
```

**Server Emission:**
```typescript
// On each quote update within the bar
io.emit('tick', {
  timestamp: currentBar.timestamp,
  open: currentBar.open,
  high: currentBar.high,
  low: currentBar.low,
  close: currentBar.close
});
```

**Dashboard Handler:**
```javascript
socket.on('tick', (tick) => {
  // Update the current candle with live price
  if (candleSeries && tick) {
    const candleData = {
      time: Math.floor(new Date(tick.timestamp).getTime() / 1000),
      open: tick.open,
      high: tick.high,
      low: tick.low,
      close: tick.close
    };
    candleSeries.update(candleData);
  }
});
```

---

### 6. STATUS Event
**Direction**: Server → Client (periodic)

Comprehensive status update sent every 30 seconds and when state changes.

**Event Name**: `status`

**Data Structure:**
```typescript
{
  tradingEnabled: boolean,
  position: ActivePosition | null,  // Current open position or null
  pendingSetup: 'long' | 'short' | null,  // Pending setup signal
  realizedPnL: number,              // Realized P&L for the day
  accountStatus: {
    balance: number,
    buyingPower: number,
    dailyPnL: number,
    openPositions: number,
    dailyLossLimit: number,
    isAtRisk: boolean
  },
  lastQuote: number,                // Last quote price
  currentBar: ChartData | null,     // Current bar being formed
  timestamp: string                 // ISO timestamp
}
```

**Position Object** (when there's an open trade):
```typescript
{
  tradeId: string,                  // e.g., "NQ-WINNER-1731456789-1"
  symbol: string,                   // e.g., "NQZ5"
  contractId: string,               // Broker contract ID
  side: 'long' | 'short',
  entryPrice: number,
  entryTime: string,                // ISO timestamp
  stopLoss: number,
  target: number,
  totalQty: number,                 // Number of contracts
  entryRSI: number,                 // RSI at entry
  entryOrderId: string | number,    // Order ID from broker
  stopOrderId?: string | number,
  targetOrderId?: string | number,
  stopFilled: boolean,
  targetFilled: boolean,
  stopLimitPending: boolean,
  monitoringStop: boolean,
  unrealizedPnL?: number,           // Current open P&L
  entryCommission?: number,
  exitCommission?: number
}
```

**Server Emission:**
```typescript
function broadcastDashboardUpdate() {
  const currentPnL = position ? 
    calculatePnL(position.entryPrice, lastQuotePrice, position.side, position.totalQty) 
    : 0;

  io.emit('status', {
    tradingEnabled,
    position: position ? {
      ...position,
      unrealizedPnL: currentPnL,
    } : null,
    pendingSetup: pendingSetup?.side,
    realizedPnL,
    accountStatus,
    lastQuote: lastQuotePrice,
    currentBar,
    timestamp: nowIso(),
  });
}
```

**Update Triggers:**
- Position entry/exit
- Account balance change
- Trading enable/disable
- Every 30 seconds (periodic update)

**Dashboard Handler:**
```javascript
socket.on('status', (data) => {
  updateStatus(data);  // Updates UI elements
  // Updates account balance, P&L, position info, trades list
});
```

---

### 7. TRADE Event
**Direction**: Server → Client (on position close)

Sent when a position is closed/exited.

**Event Name**: `trade`

**Data Structure:**
```typescript
{
  tradeId: string,              // e.g., "NQ-WINNER-1731456789-1"
  side: 'long' | 'short',
  entryPrice: number,
  exitPrice: number,
  entryTime: string,            // ISO timestamp
  exitTime: string,             // ISO timestamp of close
  quantity: number,             // Number of contracts
  pnl: number,                  // Realized P&L in dollars
  exitReason: string            // e.g., "sl_hit", "tp_hit", "manual", "end_of_session"
}
```

**Server Emission:**
```typescript
if (event.type === 'exit') {
  const trade = {
    tradeId: event.tradeId,
    side: event.side,
    entryPrice: position?.entryPrice || event.entryPrice,
    exitPrice: event.exitPrice,
    entryTime: position?.entryTime || event.timestamp,
    exitTime: nowIso(),
    quantity: event.qty,
    pnl: event.pnl,
    exitReason: event.reason,
  };
  io.emit('trade', trade);
}
```

**Dashboard Handler:**
```javascript
socket.on('trade', (trade) => {
  trades.push(trade);
  updateTradeHistory();
  showAlert(
    `Trade closed: ${trade.side} ${trade.pnl >= 0 ? 'WIN' : 'LOSS'} $${trade.pnl.toFixed(2)}`,
    trade.pnl >= 0 ? 'success' : 'error'
  );
});
```

---

### 8. LOG Event
**Direction**: Server → Client (continuous)

Activity log messages for strategy monitoring.

**Event Name**: `log`

**Data Structure:**
```typescript
{
  timestamp: string,            // ISO timestamp
  message: string,              // Log message text
  type: 'info' | 'success' | 'warning' | 'error'
}
```

**Server Emission:**
```typescript
function log(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') {
  // ...
  io.emit('log', { timestamp: nowIso(), message, type });
}
```

**Dashboard Handler:**
```javascript
socket.on('log', (data) => {
  addActivityLog(data.message, data.type || 'info');
});

function addActivityLog(message, type = 'info') {
  const timestamp = new Date().toLocaleTimeString();
  const colors = {
    info: '#94a3b8',
    success: '#22c55e',
    warning: '#eab308',
    error: '#ef4444'
  };
  
  const entry = document.createElement('div');
  entry.style.color = colors[type];
  entry.textContent = `[${timestamp}] ${message}`;
  document.getElementById('activityLog').appendChild(entry);
}
```

---

### 9. ALERT Event
**Direction**: Server → Client (as needed)

Important alert messages to display to user.

**Event Name**: `alert`

**Data Structure:**
```typescript
{
  message: string,
  type: 'success' | 'error' | 'warning' | 'info'
}
```

**Dashboard Handler:**
```javascript
socket.on('alert', (data) => {
  showAlert(data.message, data.type);
});

function showAlert(message, type = 'info') {
  const alert = document.createElement('div');
  alert.className = `alert-box alert-${type}`;
  alert.textContent = message;
  document.body.appendChild(alert);
  
  setTimeout(() => {
    alert.style.animation = 'slideOut 0.3s';
    setTimeout(() => alert.remove(), 300);
  }, 3000);
}
```

---

### 10. DISCONNECT Event
**Direction**: Client → Server (automatic)

When dashboard client disconnects.

**Event Name**: `disconnect`

**Server Handler:**
```typescript
socket.on('disconnect', () => {
  log(`Dashboard client disconnected: ${socket.id}`);
});
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│           LIVE TRADING SERVER (Node.js)                  │
│     (live-topstepx-*.ts on Port 3333-3336)              │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  WebSocket Server (Socket.IO)                           │
│  ├─ Receives market data from TopstepX API              │
│  ├─ Calculates indicators (BB, RSI, TTM)                │
│  ├─ Manages positions and orders                        │
│  └─ Broadcasts updates to all connected dashboards      │
│                                                           │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ WebSocket (Socket.IO)
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│          DASHBOARD CLIENT (Browser)                      │
│   (public/*.html with LightweightCharts)                │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Socket Connection (io())                               │
│  ├─ Receives: config, chartHistory, bar, tick          │
│  ├─ Receives: status, trade, log, alert                │
│  ├─ Sends: chartHistory (request)                      │
│  └─ Displays updates in real-time                      │
│                                                           │
│  Charts & UI                                            │
│  ├─ Main price chart (candlesticks + BB)               │
│  ├─ RSI indicator chart                                │
│  ├─ TTM Squeeze momentum chart                         │
│  ├─ Position monitor                                   │
│  ├─ Trade history table                                │
│  └─ Activity log                                       │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## Multi-Symbol Dashboard Integration

The **multi-symbol-dashboard.html** acts as a master controller that:

1. **Embeds individual dashboards** via iframes on separate ports:
   - MNQ on port 3333
   - MES on port 3334
   - MGC on port 3335
   - M6E on port 3336

2. **Fetches REST API data** from each instance:
   ```javascript
   // Update account info from MNQ (all share same account)
   const response = await fetch('http://localhost:3333/api/status');
   const data = await response.json();
   
   // Fetch from all 4 instances for combined stats
   const responses = await Promise.all([
     fetch('http://localhost:3333/api/status'),
     fetch('http://localhost:3334/api/status'),
     fetch('http://localhost:3335/api/status'),
     fetch('http://localhost:3336/api/status'),
   ]);
   ```

3. **REST API Endpoints Used:**
   - `GET /api/status` - Current status
   - `GET /api/accounts` - Available accounts
   - `POST /api/account/{id}` - Switch account
   - `POST /api/trading/start` - Start trading
   - `POST /api/trading/stop` - Stop trading
   - `POST /api/position/flatten` - Close position
   - `GET /api/chart` - Get chart data

---

## Symbol-Specific Messaging Patterns

### Account/Symbol Context
The system maintains one account that can trade multiple symbols simultaneously. Each live trader:

1. **Resolves its contract ID** at startup
2. **Uses the same account ID** for all positions
3. **Emits symbol-specific data** including symbol in config and position objects

Example message flow:
```
Chart Data for NQZ5:
  config: { symbol: "NQZ5", ... }
  bar: { timestamp: "...", close: 5250.75, ... }
  position: { symbol: "NQZ5", side: "long", ... }
  trade: { tradeId: "NQ-WINNER-...", exitPrice: 5251.50, ... }
```

### Message Isolation
- Each strategy instance has its own Socket.IO server on separate port
- Dashboard clients connect to specific ports for specific symbols
- Multi-symbol dashboard aggregates data via REST API (not WebSocket)
- No cross-symbol messaging - each maintains independent connection

---

## Performance Characteristics

### Update Frequency
- **Tick Events**: High frequency (every quote/price change) - lightweight
- **Bar Events**: Once per minute (when bar closes) - heavy with indicators
- **Status Events**: Every 30 seconds + on state changes
- **Log Events**: As strategy actions occur
- **Trade Events**: Only on position close

### Data Limits
- **Chart History**: Last 500 bars cached (keeps memory bounded)
- **Trade History**: Unlimited (stored in file + memory)
- **Activity Log**: Last 50 entries in UI (older entries removed from DOM)

### Message Size Examples
- Config: ~200 bytes
- Tick: ~150 bytes
- Bar: ~300 bytes (with all indicators)
- Status: ~1.5 KB (with full position + account details)
- Trade: ~400 bytes
- Log: 100-500 bytes

---

## Error Handling & Connection Management

### Dashboard Connection Flow
```javascript
function connect() {
  socket = io();
  
  socket.on('connect', () => {
    console.log('Connected to strategy server');
    socket.emit('chartHistory');  // Request initial data
  });
  
  socket.on('disconnect', () => {
    console.log('Disconnected from server');
    // Browser will auto-reconnect
  });
  
  // All event handlers for incoming data
  socket.on('chartHistory', ...);
  socket.on('config', ...);
  socket.on('status', ...);
  socket.on('bar', ...);
  socket.on('tick', ...);
  socket.on('trade', ...);
  socket.on('log', ...);
  socket.on('alert', ...);
}
```

### Auto-Reconnection
Socket.IO handles reconnection automatically with exponential backoff.

---

## Best Practices

1. **Validate Data**: Charts check for valid numbers before plotting
2. **Prevent Concurrent Updates**: Dashboard uses `isUpdatingCharts` flag
3. **Chart Preservation**: Prevents duplicate chart data loads
4. **Graceful Degradation**: REST API fallback for chart data if WebSocket slow
5. **Memory Management**: Keeps chart history bounded to 500 bars
6. **UI Responsiveness**: Separates heavy bar events from light tick events

