# TopstepX NQ Winner Live Trading Strategy

## Overview
Live implementation of the NQ Winner mean reversion strategy with bracket orders and advanced stop-loss monitoring.

## Key Features

### 1. **Exact Backtest Strategy Logic**
- Bollinger Bands: 20-period, 3σ
- RSI(24): Oversold 30, Overbought 70
- Two-stage entry: BB touch + RSI extreme → TTM Squeeze trigger
- Stop Loss: 0.01% (0.0001)
- Take Profit: 0.05% (0.0005)

### 2. **Bracket Orders: Market Entry + Limit Exits**
- **Entry**: Market order IOC (immediate fill at best available price)
- **Take Profit**: 0-tick limit IOC immediately placed
- **Stop Loss**: 0-tick limit IOC immediately placed
- All orders placed simultaneously as bracket

### 3. **Stop Limit Monitoring & Fallback**
```typescript
// If stop limit doesn't fill after 1.5 seconds:
1. Cancel unfilled stop limit order
2. Place market IOC order immediately
3. Only converts if stop hasn't filled
```

### 4. **WebSocket Integration**
- **Market Hub**: Real-time quotes for 1-minute bar construction
- **User Hub**: Order fills, position updates, account status
- Automatic reconnection with resubscription

### 5. **Trading Hours Enforcement**
- Futures session hours (Sunday 6pm - Friday 5pm CT)
- Automatic position flattening before session close
- No entries during closed hours

## Configuration (Environment Variables)

```bash
# Contract & Symbol
TOPSTEPX_NQ_LIVE_SYMBOL=NQZ5          # Default: NQZ5
TOPSTEPX_NQ_LIVE_CONTRACT_ID=         # Optional: specific contract ID

# Strategy Parameters (defaults match backtest)
TOPSTEPX_NQ_BB_PERIOD=20              # Bollinger Bands period
TOPSTEPX_NQ_BB_STDDEV=3               # Bollinger Bands std dev
TOPSTEPX_NQ_RSI_PERIOD=24             # RSI period
TOPSTEPX_NQ_RSI_OVERSOLD=30           # RSI oversold threshold
TOPSTEPX_NQ_RSI_OVERBOUGHT=70         # RSI overbought threshold

# Risk Management
TOPSTEPX_NQ_STOP_PERCENT=0.0001       # 0.01% stop loss
TOPSTEPX_NQ_TP_PERCENT=0.0005         # 0.05% take profit
TOPSTEPX_NQ_CONTRACTS=3               # Number of contracts

# Monitoring
TOPSTEPX_NQ_STOP_MONITOR_MS=1500      # Wait 1.5s before converting stop to market
TOPSTEPX_NQ_BACKFILL=100              # Initial bars to load

# Account
TOPSTEPX_ACCOUNT_ID=                  # Your TopstepX account ID
TOPSTEPX_NQ_LIVE_COMMISSION=1.40      # Commission per side (USD)

# API Endpoints
TOPSTEPX_REST_BASE=https://api.topstepx.com
TOPSTEPX_MARKET_HUB_URL=https://rtc.topstepx.com/hubs/market
TOPSTEPX_USER_HUB_URL=https://rtc.topstepx.com/hubs/user

# Logging
TOPSTEPX_NQ_TRADE_LOG=./logs/topstepx-nq-winner-live.jsonl
```

## Running the Strategy

### Prerequisites
```bash
# Install dependencies
npm install

# Set environment variables in .env
cp .env.example .env
# Edit .env with your TopstepX credentials
```

### Start Trading
```bash
# Using tsx
npx tsx live-topstepx-nq-winner.ts

# Or if executable
./live-topstepx-nq-winner.ts
```

### Manual Position Flatten
```bash
# Send SIGUSR2 signal to flatten position
kill -USR2 <process_id>

# Or use Ctrl+C for graceful shutdown
```

## Order Flow Example

### Entry Scenario
```
1. Bar closes at 20,250.00
2. Price touched BB lower band (20,248.50)
3. RSI = 28.5 (oversold)
4. TTM Squeeze ON detected

→ Place bracket order:
  - Entry: Buy 3 @ MARKET (fills immediately, e.g., @ 20,250.50)
  - Stop:  Sell 3 @ 20,248.48 (limit IOC) [0.01% below estimated entry]
  - Target: Sell 3 @ 20,260.63 (limit IOC) [0.05% above estimated entry]

→ All 3 orders placed simultaneously
→ Actual entry price updated when fill event received via WebSocket
```

### Stop Limit Monitoring
```
T+0ms:   Stop limit placed @ 20,247.98
T+100ms:  Price moving, stop not triggered yet → NOT FILLED
T+1500ms: Monitor delay complete → Still NOT FILLED

→ Execute fallback:
  1. Cancel stop limit order
  2. Place market IOC: Sell 3 @ MARKET
  3. Exit logged with "isMarketStop: true"
```

## Differences from Backtest

### ✅ Improvements
1. **Real fills**: No simulated slippage, actual exchange fills
2. **Real-time monitoring**: Active stop limit monitoring
3. **WebSocket fills**: Instant notification of fills via user hub
4. **Dynamic bars**: Quotes update intra-bar for monitoring

### ⚠️ Considerations
1. **Entry Slippage**: Market orders typically fill 0.5-2 ticks away from bar close
2. **Liquidity**: NQ is highly liquid (deep order book, fast fills)
3. **Latency**: 10-50ms typical for order placement
4. **Exit Fills**: Stop/Target limit IOC may not fill in fast-moving markets (monitored)

## Monitoring & Logs

### Console Output
```
[2025-11-11T12:34:56.789Z][NQZ5] LONG setup detected @ 20250.00 (RSI 28.5, awaiting TTM Squeeze trigger)
[2025-11-11T12:35:00.123Z][NQZ5] TTM Squeeze trigger fired - entering LONG @ 20250.00
[2025-11-11T12:35:00.234Z][NQZ5] [BRACKET] Entry Buy MARKET, Stop @ 20247.98, Target @ 20260.13
[2025-11-11T12:35:00.345Z][NQZ5] [BRACKET] Entry market order placed: 12345
[2025-11-11T12:35:00.567Z][NQZ5] [BRACKET] Stop order placed: 12346, Target order placed: 12347
[2025-11-11T12:35:00.789Z][NQZ5] ENTERED LONG MARKET @ ~20250.00 (RSI 28.5, Entry: 12345)
[2025-11-11T12:35:01.012Z][NQZ5] User trade Buy 3@20250.50
[2025-11-11T12:35:01.123Z][NQZ5] Entry filled @ 20250.50 (market order)
[2025-11-11T12:35:02.789Z][NQZ5] [MONITOR] Monitoring stop limit 12346 for 1500ms
[2025-11-11T12:35:45.123Z][NQZ5] User trade Sell 3@20260.50
[2025-11-11T12:35:45.234Z][NQZ5] EXITED LONG @ 20260.50 (target) | PnL: +600.00
```

### Trade Log (JSONL)
```json
{"timestamp":"2025-11-11T12:35:00.789Z","type":"entry","tradeId":"NQ-WINNER-123-1","side":"LONG","price":20250.25,"qty":3,...}
{"timestamp":"2025-11-11T12:35:45.234Z","type":"exit","tradeId":"NQ-WINNER-123-1","reason":"target","pnl":615.00,...}
```

## Safety Features

### 1. **Session Management**
- Auto-flatten 5 minutes before session close
- No new entries outside trading hours
- Weekend gap protection

### 2. **Error Handling**
- Rate limit detection (429 errors)
- Failed orders logged but don't crash
- WebSocket auto-reconnect

### 3. **Position Validation**
- One position at a time
- Prevent duplicate entries
- Fill status tracking via user hub

### 4. **Shutdown Protection**
```bash
SIGINT (Ctrl+C)   → Graceful shutdown with position flatten
SIGTERM           → Graceful shutdown
SIGUSR2           → Flatten position only (keeps running)
```

## Performance Expectations

Based on backtest (20 days):
- **Win Rate**: 75% (21/28 trades)
- **Avg Trade**: $544 net profit
- **Profit Factor**: 4.71
- **Max Drawdown**: $1,802

**Live adjustments**:
- Expect 10-20% lower performance due to real slippage
- Stop limit → market conversion adds ~0.5-1 tick adverse
- Target fills may be slightly less favorable (85% vs 100% passive)

## Troubleshooting

### Stop Limit Not Converting Fast Enough
```bash
# Reduce monitor delay if needed (default 1.5s)
TOPSTEPX_NQ_STOP_MONITOR_MS=1000  # Reduce to 1 second
# Or increase if getting too many market conversions
TOPSTEPX_NQ_STOP_MONITOR_MS=2000  # Increase to 2 seconds
```

### Orders Not Filling
```bash
# Increase from 0-tick to 1-tick limit
# Modify placeBracketEntry() to use entryPrice ± 1 tick
```

### WebSocket Disconnections
```bash
# Check logs for reconnection messages
# SignalR auto-reconnects, but may miss fills during gap
```

### Rate Limits (429)
```bash
# TopstepX has rate limits on order placement
# Strategy already handles this with error logging
# Reduce entry frequency if persistent
```

## Risk Warning

⚠️ **This is live trading with real money**
- Start with 1 contract to validate
- Monitor first 10-20 trades closely
- TopstepX funded accounts have daily loss limits
- Stop limit monitoring adds complexity - test thoroughly

## Next Steps

1. **Paper Trade First**: Run on TopstepX demo account for 100+ trades
2. **Monitor Fills**: Watch stop limit fill rates vs market conversions
3. **Adjust Parameters**: May need wider stops for live market conditions
4. **Scale Up**: Start with 1 contract, scale to 3 after validation

## Support

- TopstepX API Docs: https://docs.topstepx.com
- SignalR Hub: https://github.com/dotnet/aspnetcore/tree/main/src/SignalR
- Strategy Issues: Check trade logs in `./logs/topstepx-nq-winner-live.jsonl`
