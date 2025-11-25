# DeepSeek TopstepX Trading Agent Bundle
**Date: 2025-11-25**

Latest stable version of the DeepSeek-powered TopstepX futures trading agent with all fixes applied.

## What's New in This Version

### Fixed Issues (2025-11-25)
- ✅ **DeepSeek Reasoner JSON Parsing**: Removed duplicate JSON extraction logic that was causing parsing failures
- ✅ **Dashboard Crash Fix**: Fixed null reference errors preventing AI decisions from displaying on dashboard
- ✅ **Markdown Cleanup**: Enhanced reasoning text cleanup to remove ```json, "Trade Decision:", and other markdown artifacts
- ✅ **AI Autonomy**: Removed orderFlowConfirmed gate - AI now analyzes all order flow data and makes autonomous decisions
- ✅ **API Rate Limiting**: Added exponential backoff retry logic (2s, 4s, 8s) for both TopStepX and ProjectX REST clients
- ✅ **Historical Data Optimization**: Changed from 1-minute to 5-minute bars to reduce API load and avoid rate limits

## What's Included

### Agent Files
- `live-fabio-agent-playbook.ts` - NQ futures trading agent (Nasdaq micro)
- `live-fabio-agent-playbook-mgc.ts` - Gold futures trading agent (Micro Gold)
- `ecosystem.config.js` - PM2 process manager configuration for running both agents

### Core Libraries (`lib/`)
- `openaiTradingAgent.ts` - DeepSeek Reasoner integration with fixed JSON parsing
- `topstepx.ts` - TopStepX API client with rate limit retry logic
- `fabioPlaybook.ts` - Volume profile and order flow analysis framework
- `fabioOpenAIIntegration.ts` - AI analysis prompt construction and market context
- `executionManager.ts` - Order execution, bracket management, position tracking
- `riskManagementAgent.ts` - Risk controls and position sizing
- `swingRiskManagementAgent.ts` - Swing trading risk management
- `enhancedFeatures.ts` - Advanced technical indicators
- `volumeProfile.ts` - Volume profile calculations (VAH, VAL, POC)
- `deepseekCache.ts` - Response caching for DeepSeek API calls

### Authentication (`auth/`)
- `fetch-jwt.ts` - JWT token fetching from TopStepX
- `jwt-manager.ts` - Token caching and refresh logic

### Configuration
- `package.json` - Dependencies (axios, openai, socket.io, @microsoft/signalr, etc.)
- `tsconfig.json` - TypeScript configuration
- `projectx-rest.ts` - ProjectX REST API client with rate limit handling

### Dashboards (`public/`)
- `fabio-agent-dashboard.html` - NQ agent real-time monitoring dashboard
- `fabio-agent-dashboard-mgc.html` - Gold agent real-time monitoring dashboard

## Quick Start

### 1. Install Dependencies
```bash
cd deepseek-topstepx-2025-11-25
npm install
```

### 2. Set Environment Variables
Create a `.env` file:
```bash
# DeepSeek API
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com

# TopStepX Credentials
TOPSTEPX_API_KEY=your_topstepx_api_key
TOPSTEPX_USERNAME=your_topstepx_username
TOPSTEPX_BASE_URL=https://gateway.topstepx.com

# Optional: Override REST base if different
TOPSTEPX_REST_BASE=https://api.topstepx.com
```

### 3. Run Individual Agents

**NQ Agent (Nasdaq Micro Futures):**
```bash
npx tsx live-fabio-agent-playbook.ts
```

**Gold Agent (Micro Gold Futures):**
```bash
npx tsx live-fabio-agent-playbook-mgc.ts
```

### 4. Run Both Agents with PM2 (Recommended)
```bash
npm install -g pm2
pm2 start ecosystem.config.js
pm2 logs
pm2 status
```

### 5. View Dashboards
Open in browser:
- NQ Dashboard: `public/fabio-agent-dashboard.html`
- Gold Dashboard: `public/fabio-agent-dashboard-mgc.html`

Default ports:
- NQ Agent: Socket.IO on port 3457
- Gold Agent: Socket.IO on port 3458

## How It Works

### Trading Strategy
The agent uses a hybrid approach combining:
1. **Volume Profile Analysis**: VAH, VAL, POC from trading day session (6pm ET start)
2. **Order Flow Analysis**: L2 data, CVD (Cumulative Volume Delta), absorption, exhaustion
3. **AI Decision Making**: DeepSeek Reasoner analyzes all market context and makes autonomous decisions
4. **Risk Management**: Dynamic stop-loss and take-profit brackets based on market conditions

### AI Analysis Flow
1. Market data streams in real-time via TopStepX SignalR WebSocket
2. Volume profile and order flow metrics are calculated
3. Every 60 seconds (configurable), the agent sends context to DeepSeek Reasoner
4. DeepSeek analyzes the market and returns: SELL, BUY, or HOLD with confidence level
5. If confidence ≥ 70%, the agent executes the trade with protective brackets
6. Position is managed dynamically based on order flow and AI guidance

### Rate Limit Handling
Both TopStepX API and ProjectX REST clients now include:
- Automatic retry with exponential backoff (2s, 4s, 8s)
- Up to 3 retry attempts on 429 errors
- Graceful degradation - agents continue running even during rate limits

## Key Configuration

### Agent Settings
Located in `live-fabio-agent-playbook.ts` and `live-fabio-agent-playbook-mgc.ts`:
- `MIN_CONFIDENCE_THRESHOLD`: 70% (only execute trades with ≥70% AI confidence)
- `ANALYSIS_INTERVAL_MS`: 60000 (analyze every 60 seconds)
- `MAX_TRADES_PER_DAY`: 10
- `USE_DEEPSEEK_REASONING`: true

### Volume Profile
- Session start: 6pm ET (23:00 UTC) for futures
- Uses 5-minute bars for historical data (more efficient than 1-minute)
- Calculates VAH (Value Area High), VAL (Value Area Low), POC (Point of Control)

### Risk Management
- Position sizing based on account balance
- Stop-loss: Dynamic based on market volatility
- Take-profit: Dynamic based on volume profile levels
- Max drawdown protection per trade

## Logs
Logs are written to `logs/` directory:
- `logs/fabio-nq-output.log` - NQ agent stdout
- `logs/fabio-nq-error.log` - NQ agent stderr
- `logs/fabio-gold-output.log` - Gold agent stdout
- `logs/fabio-gold-error.log` - Gold agent stderr

Trading decisions are also stored in `trading-db/decisions.jsonl`

## Troubleshooting

### Rate Limit Errors
If you see `429` errors, the retry logic will automatically handle them. However, if errors persist:
- Reduce `ANALYSIS_INTERVAL_MS` (analyze less frequently)
- Ensure you're using 5-minute bars (not 1-minute) for historical data
- Check your TopStepX account rate limits

### Dashboard Not Updating
- Verify the Socket.IO ports are correct (3457 for NQ, 3458 for Gold)
- Check firewall settings
- Look for connection errors in browser console

### AI Not Making Decisions
- Verify `OPENAI_API_KEY` and `OPENAI_BASE_URL` are set correctly
- Check that `USE_DEEPSEEK_REASONING` is true
- Ensure minimum confidence threshold is not too high (default 70%)

### Position Not Opening
- Check account balance is sufficient
- Verify `canTrade` is true on the selected TopStepX account
- Look for "Entry blocked" messages in logs
- Ensure confidence level meets threshold

## Testing
The bundle includes safety features:
- **Simulation Mode**: Agents run in demo/sim accounts by default (`live: false`)
- **Dry Run**: Set environment variable `DRY_RUN=true` to log trades without executing
- **Confidence Threshold**: Only executes trades with ≥70% confidence

## Architecture

```
DeepSeek Reasoner (AI)
         ↓
   Market Analysis
         ↓
   Decision Engine (70% threshold)
         ↓
   Execution Manager
         ↓
   TopStepX API → Futures Market
         ↓
   Position Management
```

## Dependencies
- `openai`: ^4.77.3 (DeepSeek API compatible)
- `axios`: ^1.7.9 (HTTP client with retry logic)
- `socket.io`: ^4.8.1 (Real-time dashboard)
- `@microsoft/signalr`: ^8.0.7 (TopStepX WebSocket)
- `typescript`: ^5.7.2
- `tsx`: ^4.19.2 (TypeScript execution)

## License
Proprietary - Internal use only

## Support
For issues or questions, check the logs first. Common issues:
1. Rate limiting - handled automatically with retries
2. Authentication errors - verify API keys and credentials
3. Position sync issues - check broker state in logs

---

**Last Updated**: 2025-11-25
**Version**: 2025-11-25
**Status**: Production Ready ✅
