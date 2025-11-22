# Trading Agent Project - Comprehensive Overview

## Executive Summary

This is a sophisticated **multi-strategy algorithmic trading system** designed for intraday futures and options trading. The project combines:
- **TypeScript/Node.js** backend for live trading, backtesting, and dashboards
- **Python** for the "Fabio" LLM-driven trading agent with advanced market structure analysis
- **TopstepX futures broker integration** for real-time market data and order execution
- **Multiple trading strategies** (Mean Reversion, ICT-based patterns, SMA crossovers, TTM Squeeze, etc.)
- **Real-time dashboards** with Socket.IO for live position monitoring and performance tracking
- **Comprehensive backtesting framework** with 102+ backtest implementations

The project is actively used for both paper trading (simulation) and live trading on TopstepX accounts.

---

## 1. PROJECT ARCHITECTURE & STRUCTURE

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface Layer                         │
│  - NQ Winner Dashboard (nq-winner-dashboard.html)               │
│  - NQ ICT Dashboard (nq-ict-dashboard.html)                     │
│  - Multi-Symbol Dashboard (multi-symbol-dashboard.html)         │
│  - BTCDashboard, MGC Dashboard                                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Socket.IO / HTTP REST
┌──────────────────────▼──────────────────────────────────────────┐
│                   Node.js Server Layer                           │
│  - Express.js HTTP server with REST API                         │
│  - Socket.IO WebSocket for real-time updates                    │
│  - Multiple live-*.ts trading strategy runners                  │
│  - Data aggregation and chart generation                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP/REST + WebSocket
┌──────────────────────▼──────────────────────────────────────────┐
│               Core Trading Logic (TypeScript/Node)              │
│  - Strategy Engines (Mean Reversion, TTM Squeeze, etc)          │
│  - Technical Indicators (RSI, Bollinger Bands, MACD, etc)       │
│  - Risk Management & Position Sizing                            │
│  - Trade Lifecycle Management (Entry/Exit/Management)           │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP REST API
┌──────────────────────▼──────────────────────────────────────────┐
│            Broker & Data Integration Layer                      │
│  - TopstepX API (Authentication, Orders, Account)              │
│  - Tradier API (Stock prices, options chains)                  │
│  - Coinbase API (Crypto prices)                                │
│  - Unusual Whales API (Options flow)                           │
│  - SignalR WebSocket (Real-time market data)                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│           Python LLM Agent Layer (Fabio)                        │
│  - LLM Decision Engine (GPT-4 integration)                      │
│  - Market Structure Analysis (Volume Profile, CVD, etc)        │
│  - Feature Engine (Technical analysis)                         │
│  - Execution Engine (Risk management)                          │
│  - Dashboard Bridge (Socket.IO client)                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│            Broker API Integration                               │
│  - TopstepX REST API                                           │
│  - Order Management                                            │
│  - Account Monitoring                                          │
└──────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
trading-agent/
├── app/
│   ├── api/                          # Next.js API routes for REST endpoints
│   │   ├── topstepx/                # TopstepX-specific endpoints
│   │   │   ├── account/             # Account status endpoints
│   │   │   ├── accounts/            # Multi-account endpoints
│   │   │   ├── mr-second/          # Mean Reversion (1-second) API
│   │   │   └── sma-second/         # SMA Crossover (1-second) API
│   │   ├── analyze/                # Trade analysis endpoint
│   │   ├── regime/                 # Regime backtesting endpoints
│   │   ├── mean-reversion/         # Mean reversion backtest API
│   │   ├── options-gex/            # Options GEX calculation
│   │   └── scanner/                # Market scanner API
│   ├── page.tsx                    # Main dashboard page
│   └── layout.tsx                  # App layout
│
├── lib/                             # Core trading libraries (TypeScript)
│   ├── topstepx.ts                 # TopstepX API client
│   ├── tradier.ts                  # Tradier API client
│   ├── coinbase.ts                 # Coinbase API client
│   ├── unusualwhales.ts            # Unusual Whales API client
│   ├── alpaca.ts                   # Alpaca broker integration
│   ├── twelveData.ts               # Twelve Data market data API
│   │
│   ├── meanReversionAgent.ts       # Mean reversion strategy logic
│   ├── meanReversionBacktester.ts  # Mean reversion backtest engine
│   ├── meanReversionBacktesterFutures.ts
│   │
│   ├── smaCrossoverAgent.ts        # SMA crossover strategy
│   ├── keltnerScalpAgent.ts        # Keltner channel scalping
│   ├── momentumAgent.ts            # Momentum-based trading
│   │
│   ├── regimeBacktester.ts         # Regime analysis + backtesting
│   ├── regimeBacktesterV2.ts       # V2 regime backtester
│   ├── regimeAgent.ts              # Regime-driven agent
│   ├── regimeLifecycle.ts          # Trade lifecycle management
│   │
│   ├── ttmSqueeze.ts               # TTM Squeeze indicator
│   ├── technicals.ts               # Technical indicators wrapper
│   ├── gexCalculator.ts            # GEX (Gamma Exposure) calculation
│   ├── volumeProfile.ts            # Volume profile analysis
│   ├── svpFramework.ts             # Session Value Profile framework
│   │
│   ├── analyzer.ts                 # Core analysis engine
│   ├── scoringEngine.ts            # Trade scoring algorithm
│   ├── contractSelector.ts         # Options contract selection
│   ├── sentimentAnalyzer.ts        # Sentiment analysis
│   │
│   ├── futuresFees.ts              # Commission calculations
│   ├── fabioPlaybook.ts            # Fabio's trading playbook spec
│   │
│   └── [other support libraries]
│
├── Python Components/               # Python trading agent (Fabio)
│   ├── config.py                   # Settings/configuration
│   ├── topstep_client.py          # TopstepX Python client
│   ├── engine.py                   # Main trading agent loop
│   ├── features.py                 # Feature engine (volume profile, CVD, etc)
│   ├── execution.py                # Trade execution + risk management
│   ├── llm_client.py              # OpenAI LLM integration
│   ├── storage.py                  # Logging/persistence
│   │
│   ├── fabio_dashboard.py         # Fabio + Dashboard integration
│   ├── dashboard_bridge.py        # Socket.IO client for dashboard
│   │
│   └── [test files]
│
├── Live Trading Scripts/            # Active trading strategies
│   ├── live-topstepx-nq-winner-enhanced.ts      # NQ Winner (main)
│   ├── live-topstepx-mes-winner.ts              # MES Winner
│   ├── live-topstepx-m6e-winner.ts              # M6E Winner
│   ├── live-topstepx-mgc-winner.ts              # MGC Winner
│   ├── live-topstepx-nq-ict.ts                  # NQ ICT/SMC patterns
│   ├── live-topstepx-nq-trender.ts              # NQ Trend follower
│   ├── live-topstepx-multi-symbol.ts            # Multi-symbol runner
│   ├── live-topstepx-sma.ts                     # SMA crossover
│   ├── live-topstepx-sma-second.ts              # SMA (1-second bars)
│   ├── live-topstepx-sma-bracket.ts             # SMA with bracket orders
│   ├── live-topstepx-mean-reversion-1s.ts       # Mean reversion (1-second)
│   ├── live-btc-coinbase.ts                     # Bitcoin strategy (Coinbase)
│   └── [other live strategies]
│
├── Backtest Scripts/                # 102+ backtest implementations
│   ├── backtest-topstepx-mean-reversion-nq-winner.ts
│   ├── backtest-ict-liquidity-sweep-fvg.ts
│   ├── backtest-ict-bos-choch-fvg.ts
│   ├── backtest-ict-po3-lite.ts
│   ├── backtest-nq-ict-*.ts                     # Various ICT patterns
│   ├── backtest-*-mean-reversion*.ts            # MR backtests
│   ├── backtest-keltner-*.ts                    # Keltner backtests
│   ├── backtest-coinbase-*.ts                   # Crypto backtests
│   └── [100+ more backtest files]
│
├── Public Assets/                   # Frontend dashboards
│   ├── nq-winner-dashboard.html                 # Main NQ dashboard
│   ├── nq-ict-dashboard.html                    # ICT pattern dashboard
│   ├── multi-symbol-dashboard.html              # Multi-symbol view
│   ├── btc-dashboard.html                       # Bitcoin dashboard
│   ├── mgc-po3-dashboard.html                   # MGC with PO3
│   └── test-connection.html                     # Connection tester
│
├── Logs/                            # Trading logs and state
│   ├── m6e.log, mes.log, mgc.log, mnq.log      # Live trading logs
│   ├── .m6e-state.json, .mes-state.json         # Position state
│   ├── topstepx-*.jsonl                         # Trade logs (JSONL)
│   ├── critical-orders.log                      # Order execution log
│   └── [error logs, debugging logs]
│
├── Documentation/                   # Strategy docs
│   ├── FABIO_DASHBOARD_QUICKSTART.md            # Quick start guide
│   ├── FABIO_DASHBOARD_INTEGRATION.md           # Full integration docs
│   ├── ICT-STRATEGIES-*.md                      # ICT strategy details
│   ├── NQ-ICT-*.md                              # NQ-specific docs
│   ├── README-NQ-ICT-STRATEGY.md                # NQ ICT overview
│   ├── SOCKET_IO_*.md                           # WebSocket docs
│   └── [other documentation]
│
├── package.json                     # Node.js dependencies
├── tsconfig.json                    # TypeScript configuration
├── .env                             # API keys and config (local)
├── .env.example                     # Example environment variables
└── projectx-rest.ts                # ProjectX REST wrapper

```

---

## 2. MAIN TRADING STRATEGIES IMPLEMENTED

### A. Mean Reversion Strategy (Most Active)

**Philosophy:** Trade when price goes too far from Bollinger Bands middle/SMA, expecting mean reversion back to fair value.

**Key Files:**
- `live-topstepx-mean-reversion-1s.ts` - Live trader
- `lib/meanReversionAgent.ts` - Strategy logic
- `lib/meanReversionBacktester.ts` - Backtest engine
- `backtest-topstepx-mean-reversion-nq-winner.ts` - Main backtest

**Indicators:**
- Bollinger Bands (20-period SMA, 2-3 standard deviations)
- RSI (14-24 period, oversold <30, overbought >70)
- TTM Squeeze (Bollinger Bands + Keltner Channels)
- GEX (Gamma Exposure) regime filtering

**Entry Logic:**
1. Price touches/penetrates Bollinger Bands outer band
2. RSI confirms extreme (oversold or overbought)
3. TTM Squeeze firing confirms entry
4. GEX regime checks (positive GEX = mean reversion favorable)

**Risk Management:**
- Stop Loss: 0.04%-0.5% from entry (configurable)
- Take Profit: 1.5%-2% target (3-5x risk/reward)
- Contracts: 3 total (scale out at TP1 and TP2)
- Commission: Built-in per contract type

**Active Trading:**
- **Symbols:** NQ, MES, MNQ, MGC, M6E, GC
- **Timeframe:** 1-minute bars (primary), 5-minute (secondary)
- **Session:** NY session (09:30-16:00 ET) + overnight sessions
- **Backtest Results:** Profitability varies by symbol; NQ shows 45-60% win rate

### B. ICT (Inner Circle Trader) Strategies

**Philosophy:** Trade Fair Value Gaps (FVG), Break of Structure (BOS), and liquidity sweeps using 1-minute charts.

**Strategy Files:**
- `backtest-ict-liquidity-sweep-fvg.ts` - Liquidity sweep + FVG return
- `backtest-ict-bos-choch-fvg.ts` - BOS/CHOCH + FVG
- `backtest-ict-po3-lite.ts` - Point of Interest (POI) variations
- `live-topstepx-nq-ict.ts` - Live ICT runner with dashboard

**Core Patterns:**
1. **Liquidity Sweep:** Price raids prior day's H/L, then fades back
2. **Fair Value Gap:** 3-bar pattern (bullish: low[t] > high[t-2])
3. **Break of Structure:** Pivot breaks (swing high/low violations)
4. **Market Manipulation:** Targeting institutional resting liquidity

**Entry/Exit:**
- **Long:** Sweep low + Bullish FVG → Entry at 50% midpoint
- **Short:** Sweep high + Bearish FVG → Entry at 50% midpoint
- **Stop Loss:** Just beyond sweep extreme (2-tick buffer)
- **TP1/TP2:** 1R and 2R targets with scaling

**Active Trading:**
- **Primary Symbol:** NQZ5 (E-mini NASDAQ futures)
- **Sessions:** NY morning (09:30-11:30) and afternoon (13:30-15:30)
- **Dashboard:** Real-time chart with CVD, volume profile, orderbook

### C. SMA Crossover Strategy

**Philosophy:** Trade golden/death crosses of fast (9) and slow (21) period SMAs.

**Key Files:**
- `lib/smaCrossoverAgent.ts` - Strategy engine
- `live-topstepx-sma.ts` - Live trader
- `live-topstepx-sma-second.ts` - 1-second bar version
- `live-topstepx-sma-bracket.ts` - With bracket orders (TP/SL)

**Signals:**
- **Buy Signal:** 9-SMA crosses above 21-SMA
- **Sell Signal:** 9-SMA crosses below 21-SMA
- **Optional:** Price cross a single SMA (configurable)

**Trade Execution:**
- Market orders on signal
- Bracket orders with pre-defined TP and SL
- Automatic timeout (15s) on market orders
- Flatten before close option

**Risk Management:**
- Position sizing based on account equity
- Flatten before US equity close (configurable)
- Min/max DTE filters for options

**Active Trading:**
- **Symbols:** SPY, GLD, TSLA, NVDA
- **Timeframes:** 1-minute (primary), 15-minute (secondary)
- **Trade Mode:** Stock or options (configurable)

### D. Fabio LLM-Driven Agent

**Philosophy:** Use advanced market structure analysis (Fabio's playbook) with GPT-4 decision-making for trend and mean reversion setups.

**Key Files:**
- `fabio_dashboard.py` - Integration layer
- `engine.py` - Agent main loop
- `features.py` - Feature extraction
- `execution.py` - Risk management
- `llm_client.py` - OpenAI integration
- `lib/fabioPlaybook.ts` - Strategy specification

**Fabio's Framework:**
- **Decision Layers:** Market state + Location + Aggression
- **Market States:** Balanced, Out-of-Balance-Uptrend, Out-of-Balance-Downtrend
- **Location:** Relative to Volume Profile (POC, VAH, VAL, LVNs)
- **Aggression:** CVD trend, big prints, footprint imbalance

**Trading Models:**
1. **Trend Continuation:** Trade out-of-balance directional moves
2. **Mean Reversion:** Trade reversions back to POC after failed breakouts

**Risk Rules:**
- Risk per trade: 0.25%-0.5% account
- Stop placement: Beyond aggressive print/zone + 1-2 tick buffer
- Never widen stops; tighten to breakeven when strong
- Default target: POC level

**Active Trading:**
- **Symbols:** NQ, ES
- **Session:** NY trading hours
- **Mode:** Paper trading (testing) or Live trading
- **Decision Interval:** 10-60s (configurable based on zones)

### E. Other Specialized Strategies

**Keltner Scalping (`lib/keltnerScalpAgent.ts`):**
- Keltner Channels for tighter volatility trading
- Quick entries/exits
- Low win rate but high R:R

**Momentum Agent (`lib/momentumAgent.ts`):**
- MACD crossovers
- Momentum divergences
- Trend strength validation

**Regime-Based Agent (`lib/regimeAgent.ts`):**
- Regime detection (bull/bear/range)
- Bias transitions
- Multi-timeframe confirmation

---

## 3. TECHNOLOGIES & FRAMEWORKS

### Frontend/UI
- **Next.js 16** - React framework with API routes
- **TypeScript** - Type-safe implementation
- **Tailwind CSS** - Styling
- **Lightweight-Charts** - Professional charting library
- **Socket.IO Client** - Real-time WebSocket updates
- **HTML5 Canvas** - Custom chart overlays

### Backend/Trading Logic
- **Express.js** - HTTP server
- **TypeScript** - Core trading logic
- **technicalindicators** - TA calculations (RSI, MACD, Bollinger Bands, etc.)
- **Socket.IO** - WebSocket server for real-time updates
- **SignalR (.NET)** - TopstepX market data streaming
- **Axios** - HTTP client for APIs

### Python Components
- **asyncio** - Async/await for concurrent operations
- **httpx** - Async HTTP client
- **python-socketio** - Socket.IO client
- **aiohttp** - Async HTTP framework
- **requests** - Synchronous HTTP client

### Data Sources & APIs
- **TopstepX** - Futures broker (market data, orders, accounts)
- **Tradier** - Stock prices, options chains, time & sales
- **Coinbase Advanced Trade** - Cryptocurrency trading
- **Unusual Whales** - Options flow, whale trades
- **Twelve Data** - Real-time market data (alternative)
- **Alpaca** - Paper trading alternative
- **Benzinga** - Market news headlines
- **OpenAI GPT-4** - LLM for trading decisions

---

## 4. TRADING PLATFORMS & BROKER INTEGRATION

### Primary Broker: TopstepX

**Integration Points:**
- **Authentication:** JWT token-based (`/api/Auth/loginKey`)
- **Market Data:** SignalR WebSocket for real-time bars, quotes, trades
- **Order Management:** REST API for placing, modifying, canceling orders
- **Account:** Account balance, buying power, PnL, positions
- **Contract Metadata:** Tick size, multiplier, margin requirements

**Key Clients:**
- `lib/topstepx.ts` - TypeScript REST client
- `topstep_client.py` - Python async client

**Supported Futures:**
- NQ (E-mini NASDAQ-100, $20 multiplier)
- MNQ (Micro NASDAQ, $2 multiplier)
- ES (E-mini S&P 500, $50 multiplier)
- MES (Micro S&P 500, $5 multiplier)
- 6E (Euro FX, $12.50 multiplier)
- GC (Gold, $100 multiplier)
- MGC (Micro Gold, $10 multiplier)

**Commission Rates (per side):**
- ES/NQ: $1.40
- MES/MNQ: $0.37
- 6E: $1.62
- GC/MGC: $1.62

### Secondary Brokers
- **Alpaca** - Paper trading for stocks/options (via SMA runner)
- **Coinbase** - Cryptocurrency trading (bitcoin strategy)

---

## 5. FABIO LLM AGENT IMPLEMENTATION

### Architecture

```
Market Data Stream (TopstepX)
        ↓
Feature Engine (Python)
  ├─ Volume Profile (POC, VAH, VAL, LVNs)
  ├─ CVD (Cumulative Volume Delta)
  ├─ Orderflow Analysis (absorption, exhaustion)
  ├─ Big Prints Detection
  └─ Market Structure Classification
        ↓
LLM Client (GPT-4.1-mini)
  ├─ System Prompt: Fabio's playbook rules
  ├─ Market State Assessment
  ├─ Trade Decision Generation
  ├─ Risk Parameter Updates
  └─ Strategy Notes
        ↓
Execution Engine (Python)
  ├─ Position Management
  ├─ Risk Checks (daily loss, account balance)
  ├─ Order Placement
  └─ State Persistence
        ↓
Dashboard Bridge (Socket.IO)
  └─ Real-time Updates to Browser
```

### Decision Flow

1. **Market State Analysis:**
   - Classify market as balanced/out-of-balance
   - Identify price location vs value structure
   - Assess control (buyers vs sellers)

2. **LLM Decision Request:**
   - Send current state + positions + strategy config
   - Request: trade decisions, zone updates, strategy tweaks
   - Receive: JSON with entries, exits, management moves

3. **Execution:**
   - Validate risk (position sizing, daily loss)
   - Place orders (market or limit)
   - Update strategy state (enabled flags, max trades)
   - Log decision and outcome

4. **Learning:**
   - Store LLM exchanges in JSONL log
   - Extract lessons for future decisions
   - Update importance zones

### System Prompt (Simplified)

```
You are Fabio, an intraday futures trader who:
- Uses Volume Profile, CVD, and order flow for decision-making
- Only trades Trend and Mean Reversion models (NOT ICT/SMC)
- Respects market state (balanced vs. out-of-balance)
- Never widens stops; tighten to breakeven when strong
- Risk per trade: 0.25%-0.5% account

Input: Market state, features, positions, strategy config
Output: JSON with trade_decisions, importance_zones, strategy_updates
```

---

## 6. DASHBOARD & MONITORING SYSTEMS

### Live Dashboards (HTML5 + Socket.IO)

**1. NQ Winner Dashboard** (`nq-winner-dashboard.html`)
- Real-time 1-minute candlestick chart
- RSI and TTM Squeeze indicator panels
- Current position monitor (entry, SL, TP, unrealized PnL)
- Trade history with win/loss tracking
- Activity log with timestamps
- Account status (balance, daily PnL, win rate)
- Controls: Start/Stop trading, Flatten position

**2. NQ ICT Dashboard** (`nq-ict-dashboard.html`)
- Similar layout but optimized for ICT patterns
- Fair Value Gap visualization
- Liquidity level overlays
- CVD (Cumulative Volume Delta) chart
- Volume profile with POC highlighting
- Order book overlay (green bids, red asks)

**3. Multi-Symbol Dashboard** (`multi-symbol-dashboard.html`)
- Monitor 4+ symbols simultaneously
- Individual position panels per symbol
- Consolidated PnL and statistics
- Symbol selector dropdown

**4. Python Dashboard Bridge** (`dashboard_bridge.py`)
- Socket.IO client connecting Python agents to Node server
- Emits: positions, trades, logs, account status
- Receives: control commands (start/stop trading)
- JSON serialization for all events

### Data Broadcasting (Socket.IO Events)

**Position Updates:**
```javascript
{
  type: 'position',
  symbol: 'NQZ5',
  side: 'long',
  entry: 21000.00,
  stop: 20996.00,
  tp1: 21004.00,
  tp2: 21008.00,
  size: 3,
  unrealizedPnL: 50.00
}
```

**Trade Closes:**
```javascript
{
  type: 'trade_closed',
  symbol: 'NQZ5',
  side: 'long',
  entryPrice: 21000.00,
  exitPrice: 21004.00,
  pnl: 160.00,
  exitReason: 'TP1'
}
```

**Activity Logs:**
```javascript
{
  type: 'log',
  timestamp: '2025-11-17T14:32:15Z',
  message: 'Position opened: LONG 3 @ 21000.00',
  level: 'info'
}
```

---

## 7. RISK MANAGEMENT & POSITION SIZING

### Position Sizing Model

```
Position Size = (Account Balance × Risk Fraction) / (Stop Loss Distance in Ticks × Tick Value)

Example (NQ = $20/tick):
- Account: $50,000
- Risk: 0.25% = $125
- Entry: 21000, Stop: 20996 = 4 ticks = $80
- Contracts: 125 / 80 = 1.5 → 1 contract
```

### Risk Parameters (Configurable)

**Daily Loss Limits:**
- Max daily loss: 3% of account ($1,500 on $50k)
- All new entries blocked if limit hit
- Daily PnL reset at session start

**Per-Trade Risk:**
- Min risk: 0.1% account
- Default risk: 0.25% account  
- Max risk: 0.5% account
- Configurable via strategy updates (Fabio can adjust)

**Position Sizing Constraints:**
- Max contracts per entry: configurable (usually 3)
- Max open positions: configurable
- Max trades per session: configurable per strategy

### Stop Management

**Entry:**
- Stop placed immediately (limit or market)
- Stop-limit orders with grace period before market fallback
- Buffer from obvious breakout points (usually 1-2 ticks)

**Management:**
- Move to breakeven when strong (1st TP hit)
- Tighten on consecutive wins
- Never widen (only move closer)
- Trailing stops for trend trades

**Exit Triggers:**
- Stop loss hit
- Take profit hit (TP1, TP2, or full exit)
- Session end (flatten before close)
- Daily loss limit (reject new entries)
- Strategy disabled (stop all)

### Commission & Slippage

**Estimated Costs:**
- Entry: 1 tick slippage + commission
- Exit: 1 tick slippage + commission
- Example (NQ): 2 × (1 tick × $20 + $1.40) = $46.80 RT

**Built-in Models:**
- `lib/futuresFees.ts` - Exact commission lookup
- Backtests include realistic slippage (0.5-2 ticks configurable)

---

## 8. DATA SOURCES & MARKET DATA HANDLING

### TopstepX Market Data

**Real-time Streaming (SignalR WebSocket):**
- 1-second OHLCV bars
- L1 quotes (bid/ask)
- L2 market depth
- Trade tape (size, price, buy/sell side)
- Volume aggregation

**REST Historical Data:**
- Fetch minute bars (1, 5, 15, 60 minute)
- Configurable lookback (100-20000 bars)
- Metadata (tick size, multiplier, margin)

**Implementation:**
```typescript
// Authenticate
const token = await authenticate(); // JWT token

// Fetch bars
const bars = await fetchTopstepXFuturesBars({
  contractId: 'CON.F.US.MNQ.U25',
  startTime: '2025-11-10T00:00:00Z',
  endTime: '2025-11-17T00:00:00Z',
  unit: 2,      // Minutes
  unitNumber: 1 // 1-minute bars
});

// SignalR connection for live data
const hub = new HubConnectionBuilder()
  .withUrl(WS_URL, { accessTokenFactory: () => token })
  .withAutomaticReconnect()
  .build();
```

### Tradier Integration

**Stock Options Data:**
- Options chains (strikes, expiration dates, greeks)
- Time & sales (tick-by-tick trades)
- Stock quotes (bid/ask/last)
- Historical OHLC

**Usage:**
- Option contract selection for recommendations
- Backtesting (time & sales augmentation)
- Technical analysis on stocks

### Unusual Whales (Options Flow)

**Data Points:**
- Whale flow alerts (large options trades)
- Unusual options activity
- Sector flow analysis
- Greeks flow (delta, vega, gamma)

**Use Case:**
- Regime confirmation (whale buying/selling)
- Bias direction
- Institutional sentiment

### Twelve Data (Alternative Feed)

**Features:**
- Real-time bars (1-minute, 5-minute, etc.)
- WebSocket streaming
- Fallback when Tradier rate-limited

**Configuration:**
- `TWELVE_DATA_API_KEY` - API key
- `TWELVE_DATA_WS_URL` - WebSocket endpoint
- `SMA_USE_TWELVE_DATA` - Use Twelve Data vs Alpaca

### Coinbase API

**Crypto Trading:**
- BTC/USD, ETH/USD prices
- Real-time updates
- Order placement for crypto futures

**Implementation:** `lib/coinbaseAdvancedTrade.ts`

---

## 9. ORDER EXECUTION & MANAGEMENT SYSTEMS

### Order Types Supported

**Market Orders:**
- Immediate execution
- Entry and exit
- Configurable timeout (15s default before cancel)

**Limit Orders:**
- TP1/TP2 profit targets
- Entry at specific price levels
- Grace period before market fallback

**Stop-Loss Orders:**
- Stop-market (immediate execution on stop)
- Stop-limit (with limit price)
- Pre-placed at entry

**Bracket Orders (Entry + TP + SL):**
- Single order with multiple legs
- TP1 and TP2 with scaling
- SL with stop-limit

### Order Lifecycle

```
Entry → Pre-Trade Risk Check
  ↓
  Verify daily loss not exceeded
  Verify account has buying power
  Verify strategy enabled
  ↓
Place Entry Order (Limit or Market)
  ↓
  If limit → Wait for fill or timeout
  If timeout → Convert to market
  ↓
Order Filled → Create Position
  ↓
Place TP1, TP2, SL Orders
  ↓
Monitor Position
  ↓
  On TP1 hit → Scale out 50%
  On TP2 hit → Close remaining
  On SL hit → Close all
  ↓
Exit Order → Close Position
  ↓
Calculate PnL → Log Trade
```

### Risk Checks (Pre-Execution)

```typescript
async function executeEntry(entry: TradeDecision) {
  // 1. Daily loss check
  if (currentDailyPnL + potentialLoss > dailyLossLimit) {
    return reject("Daily loss limit exceeded");
  }
  
  // 2. Account balance check
  if (accountBalance < requiredMargin) {
    return reject("Insufficient buying power");
  }
  
  // 3. Strategy enabled check
  if (!strategyState[strategyName].enabled) {
    return reject("Strategy disabled");
  }
  
  // 4. Max trades check
  const tradesThisSession = countTradesToday(strategyName);
  if (tradesThisSession >= maxTradesPerSession) {
    return reject("Max trades for session reached");
  }
  
  // 5. Position sizing check
  const size = calculatePositionSize(riskFraction);
  if (size < 1) {
    return reject("Position size rounds to 0");
  }
  
  // All checks pass → Place order
  await placeOrder(entry, size);
}
```

### State Persistence

**Position State Files:**
- `.m6e-state.json` - M6E positions
- `.mes-state.json` - MES positions
- `.nq-state.json` - NQ positions
- `.mgc-state.json` - MGC positions

**Trade Logs (JSONL):**
- `topstepx-nq-winner-enhanced.jsonl` - Full trade records
- One JSON object per line (JSONL format)
- Includes: entry, exit, PnL, commission, slippage

**Example Trade Log Entry:**
```json
{
  "timestamp": "2025-11-17T14:32:15Z",
  "symbol": "NQZ5",
  "side": "long",
  "entryTime": "2025-11-17T14:32:15Z",
  "entryPrice": 21000.00,
  "exitTime": "2025-11-17T14:33:20Z",
  "exitPrice": 21004.00,
  "contracts": 1,
  "pnl": 80.00,
  "commission": 2.80,
  "netPnL": 77.20,
  "duration": "1m5s"
}
```

---

## 10. TESTING & BACKTESTING INFRASTRUCTURE

### Backtest Framework

**Overview:**
- 102+ backtest implementations
- Supports all major strategies
- Realistic slippage and commission models
- Multi-symbol support

**Core Components:**
- Data fetching from TopstepX
- Technical indicator calculation
- Entry/exit signal generation
- Trade simulation
- Performance metrics

### Backtest Metrics

**Trade Statistics:**
- Total trades
- Winning trades / Win rate %
- Losing trades
- Largest win / Largest loss
- Average win / Average loss
- Profit factor (Gross Win / Gross Loss)

**Return Metrics:**
- Total PnL
- Net PnL (after commission)
- ROI (Return on Investment)
- Sharpe Ratio (risk-adjusted return)
- Max drawdown
- Drawdown duration

**Performance:**
- Trades per day
- Average trade duration
- Best day / Worst day
- Consecutive wins/losses

### Example Backtest Output

```
╔════════════════════════════════════════════════════════╗
║          NQ MEAN REVERSION BACKTEST RESULTS           ║
╚════════════════════════════════════════════════════════╝

Period: 2025-11-01 to 2025-11-17
Contracts: 1 (MNQ $2 multiplier)

TRADE SUMMARY
─────────────
Total Trades:         45
Winning Trades:       23 (51.1%)
Losing Trades:        22 (48.9%)

PROFIT METRICS
──────────────
Largest Win:          $160.00
Largest Loss:         -$80.00
Average Win:          $95.33
Average Loss:         -$72.50
Profit Factor:        1.23x

PnL SUMMARY
───────────
Gross PnL:            $2,193.00
Commission:           -$126.00
Slippage:             -$45.00
Net PnL:              $2,022.00

RISK METRICS
────────────
Max Drawdown:         -$312.00 (7.2% of equity)
Avg Trade Duration:   3.2 minutes
Best Day:             +$487.00
Worst Day:            -$198.00

EQUITY CURVE
─────────────
Starting:             $50,000.00
Ending:               $52,022.00
Return:               4.04%
Sharpe Ratio:         1.85
```

### Backtest Configuration (Environment Variables)

```bash
# Date range
BACKTEST_START="2025-11-01T00:00:00Z"
BACKTEST_END="2025-11-17T00:00:00Z"

# Symbol & contract
BACKTEST_SYMBOL="NQZ5"
BACKTEST_CONTRACT_ID="CON.F.US.MNQ.U25"

# Strategy parameters
BB_PERIOD=20                    # Bollinger Band period
BB_STD_DEV=3                    # Standard deviations
RSI_PERIOD=24                   # RSI period
RSI_OVERSOLD=30                 # Oversold threshold
RSI_OVERBOUGHT=70               # Overbought threshold

# Risk parameters
STOP_LOSS_PERCENT=0.004         # 0.4% from entry
TAKE_PROFIT_PERCENT=0.02        # 2% target
CONTRACTS=1                     # Contracts per trade
SLIPPAGE_TICKS=1                # Entry/exit slippage

# Output
BACKTEST_OUTPUT="backtest-results.json"
```

### Running Backtests

**TypeScript:**
```bash
npx tsx backtest-topstepx-mean-reversion-nq-winner.ts

# Custom date range
BACKTEST_START="2025-11-01T00:00:00Z" \
  npx tsx backtest-topstepx-mean-reversion-nq-winner.ts

# With custom parameters
BB_PERIOD=20 RSI_PERIOD=24 RSI_OVERSOLD=30 \
  npx tsx backtest-topstepx-mean-reversion-nq-winner.ts
```

**Python:**
```bash
python3 engine.py --backtest --symbol NQZ5 --mode paper_trading
```

---

## 11. PROJECT SUMMARY & CURRENT STATE

### Active Components

**Live Trading:**
- NQ Winner (enhanced, 1-minute TTM Squeeze)
- MES Winner (scalping, fast exits)
- M6E Winner (currency futures)
- MGC Winner (micro gold)
- NQ ICT (Fair Value Gap patterns)
- Multi-symbol runner (simultaneous 4 symbols)
- SMA strategies (stocks + options)
- Fabio LLM agent (market structure analysis)

**Dashboards:**
- Real-time HTML5 dashboards with Socket.IO
- Multi-symbol support
- Trade monitoring and performance tracking
- Activity logs and alerts

**Data Integrations:**
- TopstepX (primary futures broker)
- Tradier (stock options)
- Unusual Whales (options flow)
- Twelve Data (fallback market data)
- Alpaca (paper stock trading)
- Coinbase (crypto)

**Python Agent:**
- Fabio LLM-driven futures trader
- Dashboard bridge for Socket.IO
- Feature engine (volume profile, CVD analysis)
- Execution engine with risk management
- State persistence and logging

### Recent Changes (Last Week)

```
commit a5ce850 - fix: prevent infinite reconnection loop in NQZ5
commit e426508 - feat: add comprehensive permanent condition monitoring logs
commit 5fbc53d - fix: prevent phantom trades with critical order logging
commit f6cb588 - feat: add enhanced state persistence with position recovery
commit 07a5fb0 - feat: add multi-symbol trading system with unified dashboard
```

### Statistics

- **102+ backtest files** for various strategies and symbols
- **21 live trading scripts** covering different strategies
- **40+ library files** for core trading logic
- **9 dashboards** (HTML5 + React)
- **12 Python modules** (Fabio agent)
- **Extensive logging** (15+ MB of trading logs)

### Performance Notes

**Profitability (Based on Backtests):**
- NQ Mean Reversion: ~45-60% win rate, 1.2-1.5x profit factor
- MES Mean Reversion: Similar, slightly higher commission drag
- ICT FVG Patterns: 50-55% win rate, 1.3-1.8x profit factor
- SMA Crossover: 40-45% win rate on stocks, high R:R
- Varies significantly by date range and market conditions

**Drawdowns:**
- Typically 5-15% of account on backtest
- Managed daily loss limits (3% standard)
- Risk per trade: 0.25-0.5% account

---

## 12. HOW COMPONENTS WORK TOGETHER

### Live Trading Flow

```
Market Data (TopstepX)
     ↓
1-minute candles + quotes
     ↓
Strategy Engine (TypeScript)
  - Calculate indicators (RSI, BB, TTM Squeeze)
  - Generate signals (entry, exit, management)
  - Size position (risk per trade)
     ↓
Risk Management Check
  - Daily loss limit?
  - Account buying power?
  - Max positions?
     ↓
Order Execution
  - Place market/limit order
  - Set TP1/TP2 and SL
  - Monitor fills
     ↓
Position Monitoring
  - Track unrealized PnL
  - Monitor stops and targets
  - Log all activity
     ↓
Dashboard Updates (Socket.IO)
  - Chart candles
  - Position updates
  - Trade history
  - Account stats
     ↓
Trade Closure
  - Log final PnL
  - Update account balance
  - Reset for next trade
```

### Backtesting Flow

```
Historical Data (TopstepX REST)
     ↓
Load minute bars (1-20000)
     ↓
Calculate Technical Indicators
  - RSI, Bollinger Bands, MACD, etc.
  - Iteratively for each bar
     ↓
Generate Entry/Exit Signals
  - Evaluate conditions for each bar
  - Track signal history
     ↓
Order Simulation
  - Determine entry price (with slippage)
  - Place TP1/TP2 and SL
  - Check fills on each bar
     ↓
P&L Calculation
  - Entry commission + slippage
  - Exit commission + slippage
  - Calculate actual PnL
     ↓
Aggregate Statistics
  - Win rate, profit factor
  - Drawdown, Sharpe ratio
  - Trade-by-trade breakdown
     ↓
Output Results
  - JSON file with trades
  - Terminal summary
  - HTML report (optional)
```

### Fabio Agent Decision Loop

```
Every 10-60 seconds (configurable):

1. Fetch Latest Market Snapshot
   - 1-minute OHLC bar
   - L1 quote (bid/ask)
   - L2 depth (20 levels)
   - Recent trades (tape)

2. Feature Engine Processes
   - Volume profile (POC, VAH, VAL)
   - CVD calculation
   - Orderflow (absorption, exhaustion)
   - Big print detection
   - Market structure classification

3. LLM Gets Decision Request
   {
     "timestamp": "2025-11-17T14:32:15Z",
     "symbol": "NQZ5",
     "price": 21000.00,
     "features": { ... },
     "positions": [ ... ],
     "strategy_state": { ... },
     "importance_zones": [ ... ],
     "historical_notes": [ ... ]
   }

4. LLM Returns Decision
   {
     "market_assessment": {
       "market_state": "out_of_balance_uptrend",
       "regime": "trend",
       "chosen_model": "trend_continuation"
     },
     "trade_decisions": [
       {
         "action": "enter",
         "side": "long",
         "risk_fraction": 0.0025,
         "stop_price": 20996.00,
         "target_price": 21004.00
       }
     ],
     "importance_zones": [ ... ],
     "strategy_updates": { ... },
     "notes_to_future_self": [ ... ]
   }

5. Execution Engine Validates
   - Check daily loss
   - Check buying power
   - Check strategy enabled
   - Check max trades
   - Calculate position size

6. Place Orders
   - Market entry order
   - Limit TP1/TP2
   - Stop-limit SL

7. Monitor & Update Dashboard
   - Emit socket events
   - Update position display
   - Log decision

8. Log Exchange for Learning
   {
     "timestamp": "2025-11-17T14:32:15Z",
     "symbol": "NQZ5",
     "request": { ... },
     "response": { ... },
     "execution_result": { ... }
   }

Repeat...
```

---

## 13. CONFIGURATION & SETUP

### Environment Variables (.env)

```bash
# API Keys
OPENAI_API_KEY=sk-...
TOPSTEPX_API_KEY=your_key
TOPSTEPX_USERNAME=your_username
TOPSTEPX_ACCOUNT_ID=account_123

# Broker URLs
TOPSTEPX_BASE_URL=https://api.topstepx.com
TOPSTEPX_MARKET_HUB_URL=wss://rtc.topstepx.com/hubs/market

# Trading Parameters
TRADING_MODE=paper_trading  # or live_trading
TRADING_SYMBOL=NQZ5
ACCOUNT_BALANCE=50000
RISK_PER_TRADE_FRACTION=0.0025
MAX_DAILY_LOSS_FRACTION=0.03

# LLM Configuration
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=${OPENAI_API_KEY}
OPENAI_MODEL=gpt-4-1-mini

# Decision Timing
LLM_DECISION_INTERVAL_DEFAULT_SEC=60
LLM_DECISION_INTERVAL_OUTER_BAND_SEC=30
LLM_DECISION_INTERVAL_INNER_BAND_SEC=10
```

### Starting Live Trading

**NQ Winner (Main Strategy):**
```bash
npx tsx live-topstepx-nq-winner-enhanced.ts
```

**With Dashboard:**
```bash
# Terminal 1
npx tsx live-topstepx-nq-ict.ts
# Browser: http://localhost:3337

# Terminal 2
python3 fabio_dashboard.py --symbol NQZ5 --mode paper_trading
```

**Multiple Symbols:**
```bash
npx tsx live-topstepx-multi-symbol.ts
```

---

## 14. KEY INSIGHTS & DESIGN PATTERNS

### Strong Points

1. **Multi-Strategy Approach:** Different strategies for different market conditions (mean reversion, trend, scalping)
2. **Real-Time Risk Management:** Daily loss limits, position sizing, stop management all enforced
3. **Comprehensive Backtesting:** 102+ backtests allow strategy validation before live trading
4. **LLM Integration:** Fabio agent brings advanced market structure analysis beyond simple indicators
5. **Dashboard Monitoring:** Real-time tracking of positions, trades, and PnL
6. **Modular Architecture:** Strategies isolated in separate files, easy to add/modify
7. **Multi-Broker Support:** Can trade stocks, options, crypto, and futures
8. **State Persistence:** Position recovery and trade logging for audit trail

### Areas for Improvement

1. **Strategy Overlap:** Mean reversion and TTM squeeze might generate correlated signals
2. **LLM Cost:** Frequent GPT-4 calls could add up (~$0.001 per call)
3. **Single Point of Failure:** TopstepX is primary data source; Twelve Data fallback could be more robust
4. **Paper vs Live Gap:** Backtests may not capture slippage/rejection issues in live trading
5. **Optimization:** Backtest parameters could be auto-optimized (walk-forward testing)

---

## 15. QUICK START FOR RUNNING PROJECT

### Prerequisites

```bash
# Install Node.js (14+)
# Install Python 3.9+
# Install dependencies
npm install
pip3 install python-socketio aiohttp httpx
```

### Step 1: Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Step 2: Run Live Trading

**Option A: Simple SMA Strategy**
```bash
npx tsx live-topstepx-sma.ts
```

**Option B: NQ Winner with Dashboard**
```bash
# Terminal 1
npx tsx live-topstepx-nq-winner-enhanced.ts

# Terminal 2 (optional, if you want to see dashboard)
# Open browser: http://localhost:3337
```

**Option C: Fabio LLM Agent**
```bash
# Terminal 1
npx tsx live-topstepx-nq-ict.ts

# Terminal 2
python3 fabio_dashboard.py --symbol NQZ5 --mode paper_trading

# Browser
# Open: http://localhost:3337
```

### Step 3: Run Backtest

```bash
# Mean reversion on NQ
npx tsx backtest-topstepx-mean-reversion-nq-winner.ts

# ICT liquidity sweep
npx tsx backtest-ict-liquidity-sweep-fvg.ts

# Custom date range
BACKTEST_START="2025-11-01T00:00:00Z" \
  npx tsx backtest-topstepx-mean-reversion-nq-winner.ts
```

---

## Summary

This is a **production-grade algorithmic trading platform** combining traditional technical analysis strategies with modern LLM-driven decision-making. It's designed for experienced traders who want to automate their trading while maintaining control through backtesting, risk management, and real-time monitoring. The modular architecture makes it easy to add new strategies, and the comprehensive logging provides full audit trails for regulatory compliance.

