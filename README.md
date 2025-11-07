# AI Trading Agent

An intelligent trading analysis web application powered by OpenAI that integrates real-time data from Tradier, Coinbase, and Unusual Whales APIs to provide actionable trading recommendations.

## Features

- **Multi-Source Data Integration**
  - Tradier API for stock prices and options chains
  - Coinbase API for cryptocurrency pricing
  - Unusual Whales API for options flow, institutional trades, and market news

- **Advanced Analysis**
  - Bull/Bear signal detection based on options flow (put at bid/call at ask logic)
  - Institutional trading activity analysis
  - News sentiment analysis with importance weighting
  - Time-weighted importance curve for recent data
  - Market tide analysis (don't go against the trend)
  - Put/call ratio monitoring

- **AI-Powered Recommendations**
  - OpenAI GPT-4 agent for trade reasoning
  - Generates top 5 trade recommendations
  - Rating system: 1-10 scale (negative for bearish, positive for bullish)
  - Specific contract recommendations with strike and expiration
  - Confidence scoring based on factor agreement

- **Trading Strategies**
  - Scalp: Quick intraday trades
  - Intraday: Day trading opportunities
  - Swing: Multi-day positions

## Regime Backtesting

- API: `GET /api/regime/backtest`
  - Required query: `date=YYYY-MM-DD` (e.g. `date=2025-10-31`)
  - Optional: `mode` (`scalp` | `swing` | `leaps`), `interval` (minutes), `prices` (`true` | `false`), `whalePremium`, `whaleVolume`, `symbols` (comma separated), `flow=live`, `lookback` (minutes for live flow)
- Utilises cached flow snapshots under `data/{date}.json` and, when a `TRADIER_API_KEY` is configured, augments each minute with Tradier time & sales to track price interaction. Enable `flow=live` to append the latest Unusual Whales flow (subject to API quota/date availability).
- Response includes per-sector timelines, whale trade highlights, inferred regime bias transitions, simulated trade lifecycle (entries/exits, P&amp;L, drawdown), and aggregated statistics to audit the regime engine against historical flow.
- UI: navigate to `/regime/backtest` for an interactive replay with candlestick overlay, trade markers, equity stats, and a sortable trade ledger.

## Architecture

```
trading-agent/
├── app/
│   ├── api/analyze/    # API endpoint for trade analysis
│   ├── page.tsx        # Main UI dashboard
│   ├── layout.tsx      # App layout
│   └── globals.css     # Global styles
├── lib/
│   ├── tradier.ts      # Tradier API integration
│   ├── coinbase.ts     # Coinbase API integration
│   ├── unusualwhales.ts # Unusual Whales API integration
│   ├── analyzer.ts     # Core analysis algorithms
│   └── agent.ts        # OpenAI agent orchestration
├── types/
│   └── index.ts        # TypeScript type definitions
└── .env                # API keys (not in git)
```

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
# OpenAI API Key
OPENAI_API_KEY=sk-...

# Tradier API
TRADIER_API_KEY=your_key_here
TRADIER_BASE_URL=https://api.tradier.com/v1

# Coinbase API (optional for crypto)
COINBASE_API_KEY=your_key_here
COINBASE_API_SECRET=your_secret_here

# Unusual Whales API
UNUSUAL_WHALES_API_KEY=your_key_here
```

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## API Keys

### Required APIs

1. **OpenAI** - [Get API Key](https://platform.openai.com/api-keys)
   - Used for trade reasoning and analysis

2. **Tradier** - [Get API Key](https://developer.tradier.com/)
   - Free sandbox account available
   - Provides stock prices and options data

3. **Unusual Whales** - [Get API Key](https://unusualwhales.com/api)
   - Premium service for options flow and institutional data
   - Provides news, dark pool trades, and unusual activity

### Optional APIs

4. **Coinbase** - [Get API Key](https://www.coinbase.com/settings/api)
   - For cryptocurrency price data

## How It Works

### 1. Data Aggregation

The system fetches data from multiple sources:
- Options flow from Unusual Whales
- Institutional trades (dark pool)
- Market news with symbols
- Stock prices from Tradier
- Market indicators (VIX, SPY, put/call ratio)

### 2. Sentiment Analysis

For each symbol, the system analyzes:

**Options Flow Sentiment**
- Bull signals: Put at bid (selling puts), Call at ask (buying calls)
- Bear signals: Put at ask (buying puts), Call at bid (selling calls)
- Mid prices are ignored as instructed

**Institutional Activity**
- Large purchases = bullish
- Large sales = bearish
- Volume-weighted scoring

**News Analysis**
- Sentiment extraction
- Importance weighting
- Time-decay curve for recency

**Market Tide**
- Based on VIX, put/call ratio, SPY movement
- Rule: Don't go against the market tide

### 3. Importance Curve

Recent data is weighted more heavily using exponential decay:
- Half-life of 2 hours for intraday
- Most recent 3 items get recency bonus
- Older data exponentially loses importance

### 4. Trade Scoring

Composite score calculated from weighted factors:
- News Impact: 30%
- Institutional Activity: 25%
- Options Flow: 25%
- Market Tide: 15%
- Technical Indicators: 5%

Rating scale:
- +7 to +10: Strong bullish
- +4 to +6: Bullish
- -4 to -6: Bearish
- -7 to -10: Strong bearish
- -3 to +3: Filtered out (too weak)

### 5. Contract Selection

For each recommended trade:
- Bullish → Call options
- Bearish → Put options
- Strike: ATM or slightly OTM (2% from current price)
- Uses actual Tradier options chain data

### 6. AI Reasoning

OpenAI GPT-4 generates concise reasoning (2-3 sentences) based on all factors to explain why the trade is recommended.

## Usage

1. **Select Strategy**: Choose scalp, intraday, or swing
2. **Specify Symbols** (optional): Enter comma-separated symbols or leave blank for auto-discovery
3. **Run Analysis**: Click "Run Analysis" to fetch and analyze data
4. **Review Recommendations**: Get top 5 trades with detailed breakdowns

Each recommendation includes:
- Bull/Bear rating (1-10 scale)
- Confidence percentage
- Specific contract to buy
- Current price and strike
- AI-generated reasoning
- Factor breakdown showing what drove the signal

## API Endpoints

### `GET/POST /api/analyze`

Analyzes market data and returns trade recommendations.

**Query Parameters (GET):**
- `strategy`: scalp | intraday | swing (default: intraday)
- `limit`: number of trades to return (default: 5)
- `symbols`: comma-separated list of symbols (optional)

**Request Body (POST):**
```json
{
  "strategy": "intraday",
  "limit": 5,
  "symbols": ["AAPL", "TSLA"]
}
```

**Response:**
```json
{
  "trades": [
    {
      "symbol": "AAPL250131C00150000",
      "underlying": "AAPL",
      "contract": "AAPL 2025-01-31 150C",
      "strike": 150,
      "expiration": "2025-01-31",
      "type": "call",
      "action": "buy",
      "strategy": "intraday",
      "currentPrice": 3.50,
      "rating": 8,
      "confidence": 85,
      "reasoning": "Strong bullish momentum...",
      "factors": {
        "newsImpact": 0.65,
        "institutionalActivity": 0.42,
        "optionsFlow": 0.78,
        "marketTide": 0.8,
        "technicals": 0.3
      }
    }
  ],
  "marketOverview": {
    "putCallRatio": 0.85,
    "vix": 14.2,
    "spy": 485.50,
    "marketTide": "bullish"
  }
}
```

## Key Algorithms

### Bull/Bear Signal Detection

```typescript
// Bull signals
if (optionType === 'put' && side === 'bid') → Bullish (selling puts)
if (optionType === 'call' && side === 'ask') → Bullish (buying calls)

// Bear signals
if (optionType === 'put' && side === 'ask') → Bearish (buying puts)
if (optionType === 'call' && side === 'bid') → Bearish (selling calls)

// Ignored
if (side === 'mid') → Ignored as per requirements
```

### Importance Curve

```typescript
importance = exp(-ageHours / 2) + recencyBonus
where recencyBonus = 0.2 for top 3 most recent items
```

### Market Tide Rules

- If VIX < 15 AND put/call < 0.9 → Bullish tide
- If VIX > 25 OR put/call > 1.2 → Bearish tide
- Signal is boosted if aligned with market tide
- Signal is penalized if against market tide

## Tech Stack

- **Framework**: Next.js 14 with TypeScript
- **AI**: OpenAI GPT-4
- **Styling**: Tailwind CSS
- **APIs**: Tradier, Coinbase, Unusual Whales
- **HTTP Client**: Axios

## Development

```bash
# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Mean Reversion Live Trader

Run the mean reversion option strategy in real time (default: 15-minute bar logic with optional Twelve Data streaming).

```bash
npx tsx run-live-mean-reversion.ts
```

### Environment Variables

- `MR_SYMBOLS` (default `SPY,GLD,TSLA,NVDA`): Comma-separated tickers to monitor.
- `MR_OPTION_CONTRACTS` (default `2`): Contracts per entry in option mode.
- `MR_STOCK_SHARES` (optional): Override share quantity when `MR_TRADE_MODE=stock`.
- `MR_TRADE_MODE` (default `option`): `option` for OCC contracts, `stock` for shares.
- `MR_OPTION_MIN_DTE` (default `7`): Minimum days-to-expiration when picking contracts.
- `MR_POLL_MS` (default `15000`): Signal polling cadence.
- `MR_MINUTE_BACKFILL` (default `600`): Minute bars to retain when using Twelve Data.
- `MR_TRADE_LOG` (default `./logs/mean-reversion-trades.jsonl`): JSONL trade log.
- Twelve Data (`TWELVE_DATA_*`) and Alpaca (`ALPACA_*`) credentials are shared with the other runners.

## Mean Reversion 5-Minute Trader

Variant tuned for five-minute bars (same option workflow, faster cadence).

```bash
npx tsx run-live-mean-reversion-5min.ts
```

Key env overrides:

- `MR5_SYMBOLS` (default `SPY,GLD,TSLA,NVDA`)
- `MR5_TRADE_MODE` / `MR5_OPTION_CONTRACTS` / `MR5_OPTION_MIN_DTE`
- `MR5_STOCK_SHARES`, `MR5_POLL_MS`, `MR5_MINUTE_BACKFILL`
- `MR5_TRADE_LOG` (default `./logs/mean-reversion-5min-trades.jsonl`)

All other Twelve Data and Alpaca credentials mirror the primary runner.

## SMA Crossover Live Trader

Run a live simple moving average crossover strategy on Alpaca paper trading with automated logging of entries, exits, and realized PnL.

```bash
npx tsx run-live-sma-crossover.ts
```

### Environment Variables

- `SMA_SYMBOLS` (default `SPY`): Comma-separated list of tickers to monitor.
- `SMA_FAST` (default `9`): Fast SMA period.
- `SMA_SLOW` (default `21`): Slow SMA period; must be greater than `SMA_FAST`.
- `SMA_ORDER_QTY` (default `1`): Share quantity per trade.
- `SMA_POLL_MS` (default `60000`): Polling interval in milliseconds.
- `SMA_FILL_TIMEOUT_MS` (default `15000`): Time allowance for market orders to fill before cancellation.
- `SMA_TIMEFRAME` (default `1Min`): Bar timeframe requested from the Alpaca data API.
- `SMA_FLATTEN_BEFORE_CLOSE` (default `5`): Minutes before the U.S. equity close to automatically flatten positions (`0` disables).
- `SMA_TRADE_LOG` (default `./logs/sma-crossover-trades.jsonl`): Destination file for structured trade logs.
- `SMA_PRICE_CROSS` (default `false`): Set to `true` to trade price crossing a single SMA (e.g., price vs 9-SMA) instead of dual SMA crossover.
- `SMA_USE_TWELVE_DATA` (default auto): When `true`, stream real-time prices from Twelve Data WebSocket (requires `TWELVE_DATA_API_KEY`); set to `false` to force Alpaca historical bars.
- `SMA_TRADE_MODE` (default `option`): Set to `option` to trade OCC option contracts or `stock` to trade shares.
- `SMA_OPTION_CONTRACTS` (default `2`): Number of option contracts to trade per signal when `SMA_TRADE_MODE=option`.
- `SMA_MIN_DTE` (default `3`): Minimum days-to-expiration filter for option selection; the strategy chooses the nearest contract with DTE ≥ this value (or the closest available if none qualify).

Ensure `ALPACA_KEY_ID`, `ALPACA_SECRET_KEY`, and optional `ALPACA_BASE_URL` / `ALPACA_DATA_URL` are set for paper trading access prior to launching the strategy.
If `TWELVE_DATA_API_KEY` is present (and optionally `TWELVE_DATA_WS_URL`), the runner will automatically pull live prices from Twelve Data for faster crossover detection unless disabled with `SMA_USE_TWELVE_DATA=false`.

## Limitations & Disclaimers

- This is a trading analysis tool, NOT financial advice
- Past performance does not guarantee future results
- Always do your own due diligence before trading
- Options trading carries significant risk
- API rate limits may affect real-time performance
- Unusual Whales API requires paid subscription for full features

## Future Enhancements

- [ ] Real-time WebSocket data feeds
- [ ] Backtesting capabilities
- [ ] Technical analysis indicators (RSI, MACD, etc.)
- [ ] Portfolio tracking
- [ ] Trade execution integration
- [ ] More sophisticated ML models
- [ ] Historical performance tracking
- [ ] Alert system for new signals
- [ ] Mobile responsive improvements

## License

MIT

## Support

For issues or questions, please open an issue on GitHub.
