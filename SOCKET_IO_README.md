# Socket.IO Integration Documentation

This folder contains comprehensive documentation of the Socket.IO integration patterns used in the live trading system.

## Files

1. **SOCKET_IO_INTEGRATION.md** (19 KB, 700+ lines)
   - Complete technical reference
   - Detailed event specifications
   - Data structure definitions
   - Code examples for all events
   - Server and client handler implementations
   - Multi-symbol dashboard integration
   - Performance characteristics
   - Best practices and error handling

2. **SOCKET_IO_QUICK_REFERENCE.txt** (5.9 KB, 190 lines)
   - Quick lookup table of events
   - Port mapping
   - Core data structures summary
   - Broadcast function details
   - Performance profile
   - Symbol messaging patterns
   - Important notes for developers

## Quick Overview

### Event Types (10 total)

| Event | Direction | Frequency | Purpose |
|-------|-----------|-----------|---------|
| config | S→C | Once | Strategy configuration |
| chartHistory | Bidirectional | On demand | Historical bar data (500 bars) |
| bar | S→C | 1/minute | Completed bar with indicators |
| tick | S→C | High | Intra-bar quote updates |
| status | S→C | 30s + changes | Account and position status |
| trade | S→C | On exit | Closed position details |
| log | S→C | Continuous | Activity log messages |
| alert | S→C | As needed | Important notifications |
| connect | Automatic | - | Client connection |
| disconnect | Automatic | - | Client disconnection |

### Port Configuration

```
Port 3333: NQ Winner (MNQ)       - live-topstepx-nq-winner-enhanced.ts
Port 3334: MES Winner            - live-topstepx-mes-winner.ts
Port 3335: MGC Winner            - live-topstepx-mgc-winner.ts
Port 3336: M6E Winner            - live-topstepx-m6e-winner.ts
Port 3006: MGC PO3               - live-topstepx-mgc-po3.ts
```

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│           LIVE TRADING SERVER (Node.js)                  │
│   (live-topstepx-*.ts - WebSocket on port 3333-3336)   │
├─────────────────────────────────────────────────────────┤
│ • Receives market data from TopstepX API                │
│ • Calculates indicators (BB, RSI, TTM Squeeze)          │
│ • Manages positions and orders                          │
│ • Broadcasts updates via Socket.IO                      │
└────────────────┬────────────────────────────────────────┘
                 │ WebSocket (Socket.IO v4.6.1)
                 ▼
┌─────────────────────────────────────────────────────────┐
│          DASHBOARD CLIENT (Browser)                      │
│   (public/*.html with LightweightCharts)                │
├─────────────────────────────────────────────────────────┤
│ • Receives real-time updates via Socket.IO              │
│ • Displays price charts and indicators                  │
│ • Shows position and account status                     │
│ • Logs trading activity                                 │
└─────────────────────────────────────────────────────────┘
```

## Key Concepts

### Broadcasting Pattern

Most events use `io.emit()` to broadcast to all connected clients:

```typescript
io.emit('bar', chartPoint);        // Broadcast new bar
io.emit('status', {...});          // Broadcast status update
io.emit('trade', trade);           // Broadcast closed trade
io.emit('log', {...});             // Broadcast activity log
```

### Request-Response Pattern

Chart history uses a request-response pattern:

```typescript
// Client requests
socket.emit('chartHistory');

// Server responds
socket.on('chartHistory', () => {
  socket.emit('chartHistory', data);
});
```

### Broadcast Function

The `broadcastDashboardUpdate()` function at line 716 in each trading file emits the `status` event with the complete system state:

```typescript
function broadcastDashboardUpdate() {
  io.emit('status', {
    tradingEnabled: boolean,
    position: ActivePosition | null,
    accountStatus: {...},
    realizedPnL: number,
    timestamp: string
  });
}
```

**Called when:**
- Position entry/exit
- Trading state changes
- Account balance updates
- Every 30 seconds (periodic)

## Data Structures

### ChartData (bar event)
```typescript
{
  timestamp: string,
  open: number,
  high: number,
  low: number,
  close: number,
  bbUpper: number,
  bbBasis: number,      // Bollinger Band middle
  bbLower: number,
  rsi: number,          // RSI(24)
  ttmMomentum: number,  // TTM Squeeze momentum
  squeeze: {
    momentum: number,
    squeezeFiring: boolean
  }
}
```

### ActivePosition (status.position)
```typescript
{
  tradeId: string,
  symbol: string,
  side: 'long' | 'short',
  entryPrice: number,
  entryTime: string,
  totalQty: number,
  entryRSI: number,
  unrealizedPnL: number,
  entryOrderId: string | number,
  stopOrderId?: string | number,
  targetOrderId?: string | number
}
```

### Trade (trade event)
```typescript
{
  tradeId: string,
  side: 'long' | 'short',
  entryPrice: number,
  exitPrice: number,
  entryTime: string,
  exitTime: string,
  quantity: number,
  pnl: number,
  exitReason: string
}
```

## Multi-Symbol System

### Architecture
- **4 independent strategies** (NQ, MES, MGC, M6E)
- **1 shared account** - all positions under same account ID
- **Separate WebSocket servers** - each strategy on different port
- **Message isolation** - no cross-symbol WebSocket communication

### Master Dashboard
The `public/multi-symbol-dashboard.html` aggregates data by:
1. Embedding individual dashboards as iframes (ports 3333-3336)
2. Fetching account data via REST API
3. Providing unified controls (Start All, Stop All, Flatten All)

### Symbol Context
Every message includes symbol information:
- `config.symbol` - "NQZ5", "MES", "MGC", "M6E"
- `position.symbol` - indicates which symbol for trade
- `tradeId` - includes symbol prefix (e.g., "NQ-WINNER-...")

## Performance Characteristics

### Update Frequency
- **tick**: High frequency (every quote)
- **bar**: Once per minute (on bar close)
- **status**: Every 30 seconds + on state change
- **trade**: Only on position exit
- **log**: Strategy-dependent

### Data Limits
- **Chart history**: Last 500 bars (memory efficient)
- **Activity log**: Last 50 entries in UI (DOM-limited)
- **Trade history**: Unlimited (stored in file + memory)

### Message Sizes (approximate)
- Config: 200 bytes
- Tick: 150 bytes
- Bar: 300 bytes (with all indicators)
- Status: 1,500 bytes (with position + account)
- Trade: 400 bytes
- Log: 100-500 bytes

## Dashboard Event Handlers

### Complete Handler List

```javascript
socket.on('connect', () => {
  // Connected - request initial data
  socket.emit('chartHistory');
});

socket.on('chartHistory', (data) => {
  // Received chart data - plot all bars
  updateCharts(data);
});

socket.on('bar', (bar) => {
  // New bar completed - update all charts
  candleSeries.update(bar);
  bbUpperSeries.update({time: ..., value: bar.bbUpper});
  rsiSeries.update({time: ..., value: bar.rsi});
  ttmMomentumSeries.update({time: ..., value: bar.ttmMomentum});
});

socket.on('tick', (tick) => {
  // Intra-bar price update
  candleSeries.update(tick);
});

socket.on('status', (data) => {
  // Account and position status
  updateStatus(data);
  updatePositionDisplay();
  updateTradeHistory();
});

socket.on('trade', (trade) => {
  // Trade closed
  trades.push(trade);
  updateTradeHistory();
  showAlert(`${trade.side} closed: ${trade.pnl}`);
});

socket.on('log', (data) => {
  // Activity log
  addActivityLog(data.message, data.type);
});

socket.on('alert', (data) => {
  // Important alert
  showAlert(data.message, data.type);
});

socket.on('disconnect', () => {
  // Disconnected (auto-reconnect will trigger)
});
```

## Common Use Cases

### Real-Time Price Updates
```
tick event → candleSeries.update() → chart updates instantly
```

### Indicator Monitoring
```
bar event → separate series for Bollinger Bands, RSI, TTM
Squeeze indicator shows ON/OFF with yellow/green dots
```

### Position Monitoring
```
status event (every 30s) → unrealizedPnL updated
Position display refreshed with current price data
```

### Trade History
```
trade event (on exit) → added to history table
Includes entry/exit price, PnL, and exit reason
```

### Activity Logging
```
log event → timestamped messages with color coding
Useful for debugging and strategy monitoring
```

## Best Practices

1. **Validate Data**: Always check for valid numbers before plotting
2. **Prevent Concurrent Updates**: Use guard flags to prevent race conditions
3. **Chart Preservation**: Prevent duplicate data loads
4. **Graceful Degradation**: Fallback to REST API if WebSocket slow
5. **Memory Management**: Keep chart history bounded
6. **UI Responsiveness**: Separate heavy (bar) from light (tick) events

## Finding Specific Patterns

### To find where events are emitted:
```bash
grep -n "io\.emit\|socket\.emit" live-topstepx-nq-winner-enhanced.ts
```

### To find the broadcast function:
```bash
grep -n "function broadcastDashboardUpdate" live-topstepx-nq-winner-enhanced.ts
```

### To find event handlers in dashboard:
```bash
grep -n "socket\.on" public/nq-winner-dashboard.html
```

## References

- **Socket.IO Documentation**: https://socket.io/docs/v4/
- **LightweightCharts**: https://tradingview.github.io/lightweight-charts/
- **Server Implementation**: live-topstepx-nq-winner-enhanced.ts (lines 2168-2200)
- **Dashboard Implementation**: public/nq-winner-dashboard.html (lines 870-991)

---

For detailed specifications, see **SOCKET_IO_INTEGRATION.md**

For quick lookups, see **SOCKET_IO_QUICK_REFERENCE.txt**
