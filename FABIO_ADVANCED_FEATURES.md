# Fabio Advanced Features - Socket.IO Integration

## Overview
The Fabio Socket.IO server has been enhanced with advanced order flow and volume profile features that emit real-time market microstructure data to the dashboard.

## Advanced Features Now Emitted

### 1. Volume Profile Data (`volume_profile` event)
Emitted every market snapshot with:
- **POC** (Point of Control): Price level with highest volume
- **VAH** (Value Area High): Upper boundary of 70% volume area
- **VAL** (Value Area Low): Lower boundary of 70% volume area
- **LVNs** (Low Volume Nodes): Price levels with minimal trading activity
- **Session High/Low**: Current session's price range

### 2. CVD and Order Flow (`cvd` event)
Real-time order flow analytics:
- **CVD Value**: Cumulative Volume Delta absolute value
- **CVD Trend**: Direction of volume delta (up/down/neutral)
- **Buy/Sell Absorption**: Scores indicating large order absorption (0-1)
- **Buy/Sell Exhaustion**: Scores indicating momentum exhaustion (0-1)
- **Big Prints**: List of recent large trades above threshold

### 3. L2 Order Book (`l2_data` event)
Market depth information:
- **Bids**: Top 10 bid levels [price, size]
- **Asks**: Top 10 ask levels [price, size]
- **Spread**: Current bid-ask spread

### 4. Market State (`market_state` event)
Overall market condition assessment:
- **State**: balanced/imbalanced/breakout
- **Range Condition**: normal/expanding/contracting
- **Location vs Value**: Price position relative to value area
- **Location vs POC**: Price position relative to POC
- **Buyers/Sellers Control**: Control scores (0-1)
- **POC Crosses**: Number of POC crosses in last 5 minutes
- **Time in Value**: Seconds spent in value area

### 5. Absorption/Exhaustion Signals
Special events emitted when thresholds exceeded (>0.7):
- **`absorption` event**: Large order absorption detected
- **`exhaustion` event**: Momentum exhaustion detected
Both include type (buy/sell), strength, price, and timestamp

### 6. LLM Decision Viewer (`llm_decision` event)
Detailed AI decision reasoning:
- **Market Assessment**: LLM's market regime analysis
- **Trade Decisions**: Specific entry/exit recommendations
- **Reasoning**: Detailed explanation of decision logic
- **Notes**: Self-learning notes for future iterations
- **Confidence**: Decision confidence score (0-1)
- **Strategy Updates**: Parameter adjustments made
- **Importance Zones**: Key price levels identified

## How to Use

### Starting the Server
```bash
python fabio_socketio_server.py
```

The server will:
1. Start on port 3337
2. Serve the NQ Winner dashboard HTML
3. Begin emitting all advanced features when trading starts

### Dashboard Connection
The dashboard automatically connects via Socket.IO and subscribes to all events. To display the advanced features, the dashboard HTML needs UI panels for:

1. **Volume Profile Panel**: Chart showing POC, VAH, VAL with volume bars
2. **L2 Order Book**: Depth ladder showing bid/ask levels
3. **CVD Indicator**: Line chart of cumulative volume delta
4. **Market State Badge**: Visual indicator of current market state
5. **LLM Decision Log**: Scrolling view of AI reasoning
6. **Absorption/Exhaustion Alerts**: Pop-up notifications for signals

## Event Flow

```
Market Data → Feature Engine → Socket.IO Server → Dashboard
     ↓              ↓                  ↓              ↓
  Snapshots    Extract Features   Emit Events    Display UI
```

## Socket.IO Events Reference

| Event | Frequency | Description |
|-------|-----------|-------------|
| `volume_profile` | Every snapshot | POC, VAH, VAL, LVNs |
| `cvd` | Every snapshot | CVD value, trend, order flow |
| `l2_data` | When available | Order book depth |
| `market_state` | Every snapshot | Market regime indicators |
| `absorption` | On trigger | Large order absorption alert |
| `exhaustion` | On trigger | Momentum exhaustion alert |
| `llm_decision` | Every 30s | AI trading decision details |
| `bar` | Every second | OHLCV price bar |
| `tick` | Every second | Current price tick |
| `status` | Every 10 bars | Account and position status |
| `trade` | On trade close | Executed trade details |
| `log` | As needed | System messages |

## Testing

To test the advanced features are working:

1. Start the server: `python fabio_socketio_server.py`
2. Open dashboard: http://localhost:3337
3. Click "Start Trading"
4. Open browser developer console
5. Monitor Network tab → WS → Messages to see all events

## Next Steps

The dashboard HTML needs UI components added to visualize these features. The events are being emitted but need corresponding display panels in the frontend.