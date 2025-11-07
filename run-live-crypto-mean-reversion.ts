import 'dotenv/config';
import { randomUUID } from 'crypto';
import {
  submitOrder,
  cancelOrder,
  getOrder,
  getPosition,
  getClock,
  sleep,
  AlpacaOrder,
  fetchLatestQuote,
} from './lib/alpaca';
import {
  generateMeanReversionSignalFromTechnicals,
  MeanReversionSignal,
} from './lib/meanReversionAgent';
import {
  CoinbaseWebSocketClient,
  CoinbaseTick,
  CoinbaseBar,
} from './lib/coinbaseWebSocket';

/**
 * Live Crypto Mean Reversion Strategy
 * - Market Data: Coinbase Public WebSocket (SOL-USD)
 * - Order Execution: Alpaca Paper Trading (crypto)
 * - Strategy: 2-unit position scaling with RSI + Bollinger Bands
 */

process.env.BYPASS_GEX = 'true'; // No GEX for crypto

type Direction = 'long' | 'short';

interface ActiveTrade {
  symbol: string; // Alpaca symbol (e.g., 'SOL')
  coinbaseSymbol: string; // Coinbase symbol (e.g., 'SOL-USD')
  direction: Direction;
  signal: MeanReversionSignal;
  entryPrice: number;
  entryTime: string;
  stopLoss: number | null;
  target: number | null;
  scaled: boolean;
  totalQty: number;
  remainingQty: number;
  executedValue: number;
  units: number;
  lastProcessedBar?: string | null;
  processing?: boolean;
}

interface OrderFillResult {
  filledQty: number;
  avgPrice: number;
  status: AlpacaOrder['status'];
}

// Configuration
const CONFIG = {
  // Crypto symbol mapping: Coinbase -> Alpaca
  symbols: {
    'SOL-USD': 'SOLUSD', // Coinbase format -> Alpaca format
  },
  shareQuantity: Number(process.env.CRYPTO_QTY || '1'), // e.g., 1 SOL (fractional supported)
  pollIntervalMs: Number(process.env.CRYPTO_POLL_MS || '15000'), // 15 seconds
  maxSpreadPct: Number(process.env.CRYPTO_MAX_SPREAD_PCT || '0.005'), // 0.5%
  walkRetries: Number(process.env.CRYPTO_WALK_RETRIES || '3'),
  walkIncrementPct: Number(process.env.CRYPTO_WALK_INCREMENT_PCT || '0.001'), // 0.1%
  walkTimeoutMs: Number(process.env.CRYPTO_WALK_TIMEOUT_MS || '5000'),
  marketFallback: process.env.CRYPTO_MARKET_FALLBACK !== 'false',
  // Mean reversion params
  rsiPeriod: 14,
  rsiOversold: 30,
  rsiOverbought: 70,
  bbPeriod: 20,
  bbStdDev: 2,
  bbThreshold: 0.005, // 0.5%
  stopLossPercent: 0.01, // 1%
};

const trades = new Map<string, ActiveTrade>();
const lastProcessedBar = new Map<string, string>();

// Initialize Coinbase WebSocket client
const coinbaseSymbols = Object.keys(CONFIG.symbols);
const coinbaseWS = new CoinbaseWebSocketClient({
  symbols: coinbaseSymbols,
});

// Shutdown handler
const shutdown = () => {
  console.log('\n🛑 Shutting down...');
  coinbaseWS.disconnect();
  process.exit(0);
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

function nowIso() {
  return new Date().toISOString();
}

function log(symbol: string, message: string) {
  console.log(`[${new Date().toISOString()}][${symbol}] ${message}`);
}

function sideForDirection(direction: Direction, intent: 'entry' | 'exit'): 'buy' | 'sell' {
  if (intent === 'entry') {
    return direction === 'long' ? 'buy' : 'sell';
  }
  return direction === 'long' ? 'sell' : 'buy';
}

/**
 * Wait for order to fill or timeout
 */
async function waitForOrderFill(
  orderId: string,
  timeoutMs: number,
  cancelOnTimeout = true,
): Promise<OrderFillResult> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const order = await getOrder(orderId);
    const filledQty = Number(order.filled_qty || '0');
    const avgPrice = Number(order.filled_avg_price || '0');

    if (order.status === 'filled') {
      return { filledQty, avgPrice, status: order.status };
    }

    if (
      order.status === 'canceled' ||
      order.status === 'rejected' ||
      order.status === 'stopped' ||
      order.status === 'suspended'
    ) {
      return { filledQty, avgPrice, status: order.status };
    }

    await sleep(1000);
  }

  if (cancelOnTimeout) {
    try {
      await cancelOrder(orderId);
    } catch (err) {
      log('system', `Cancel failed for ${orderId}: ${(err as Error).message}`);
    }
  }

  const order = await getOrder(orderId);
  return {
    filledQty: Number(order.filled_qty || '0'),
    avgPrice: Number(order.filled_avg_price || '0'),
    status: order.status,
  };
}

/**
 * Place adaptive order with price walking
 * Crypto typically has tighter spreads than stocks
 */
async function placeAdaptiveOrder(
  alpacaSymbol: string,
  side: 'buy' | 'sell',
  quantity: number,
  mode: 'entry' | 'scale' | 'exit-stop' | 'exit-eod',
  referencePrice: number,
): Promise<OrderFillResult | null> {
  // Check spread via Alpaca quote
  const quote = await fetchLatestQuote(alpacaSymbol);
  if (!quote || !Number.isFinite(quote.ap) || !Number.isFinite(quote.bp)) {
    log(alpacaSymbol, 'No quote available; skipping order');
    return null;
  }

  const spread = quote.ap - quote.bp;
  const spreadPct = spread / referencePrice;
  const throttle = mode !== 'exit-stop' && spreadPct > CONFIG.maxSpreadPct;

  if (throttle) {
    log(alpacaSymbol, `Spread ${(spreadPct * 100).toFixed(3)}% too wide; throttling ${mode}`);
    return null;
  }

  let remaining = quantity;
  let attempt = 0;
  let executedQty = 0;
  let executedValue = 0;

  while (remaining > 0 && attempt <= CONFIG.walkRetries) {
    const incrementPct = CONFIG.walkIncrementPct * attempt;
    const basePrice = side === 'buy' ? quote.ap : quote.bp;
    const limitPrice = side === 'buy'
      ? Number((basePrice * (1 + incrementPct)).toFixed(2))
      : Number((basePrice * (1 - incrementPct)).toFixed(2));

    const order = await submitOrder({
      symbol: alpacaSymbol,
      side,
      qty: String(remaining.toFixed(8)), // Crypto supports fractional shares
      type: 'limit',
      limit_price: limitPrice,
      time_in_force: 'gtc', // Crypto uses GTC not DAY
      client_order_id: randomUUID(),
    });

    const fill = await waitForOrderFill(order.id, CONFIG.walkTimeoutMs);
    executedQty += fill.filledQty;
    executedValue += fill.avgPrice * fill.filledQty;
    remaining -= fill.filledQty;

    if (remaining <= 0) {
      return {
        filledQty: executedQty,
        avgPrice: executedQty > 0 ? executedValue / executedQty : fill.avgPrice,
        status: fill.status,
      };
    }

    if (fill.status === 'rejected') {
      log(alpacaSymbol, `Order rejected at limit ${limitPrice}; retrying with more aggressive price`);
    } else {
      try {
        await cancelOrder(order.id);
      } catch (err) {
        log(alpacaSymbol, `Cancel failed for ${order.id}: ${(err as Error).message}`);
      }
    }

    attempt += 1;
  }

  // Fallback to market order
  if (remaining > 0 && CONFIG.marketFallback) {
    log(alpacaSymbol, `Falling back to market for ${remaining.toFixed(8)} (${mode})`);
    const order = await submitOrder({
      symbol: alpacaSymbol,
      side,
      qty: String(remaining.toFixed(8)),
      type: 'market',
      time_in_force: 'gtc',
      client_order_id: randomUUID(),
    });

    const fill = await waitForOrderFill(order.id, 15000, false);
    executedQty += fill.filledQty;
    executedValue += fill.avgPrice * fill.filledQty;
    remaining -= fill.filledQty;
  }

  if (executedQty === 0) {
    log(alpacaSymbol, `No fill achieved for ${mode} order`);
    return null;
  }

  return {
    filledQty: executedQty,
    avgPrice: executedValue / executedQty,
    status: remaining === 0 ? 'filled' : 'partially_filled',
  };
}

/**
 * Attempt to enter a new position
 */
async function attemptEntry(
  coinbaseSymbol: string,
  alpacaSymbol: string,
  signal: MeanReversionSignal,
  referencePrice: number,
) {
  const desiredQty = CONFIG.shareQuantity;
  if (desiredQty <= 0) return;

  const direction: Direction = signal.direction === 'long' ? 'long' : 'short';
  const side = sideForDirection(direction, 'entry');

  log(coinbaseSymbol, `Attempting ${direction.toUpperCase()} entry @ $${referencePrice.toFixed(2)} for ${desiredQty} units`);
  const fill = await placeAdaptiveOrder(alpacaSymbol, side, desiredQty, 'entry', referencePrice);

  if (!fill || fill.filledQty === 0) {
    log(coinbaseSymbol, 'Entry order not filled');
    return;
  }

  const actualQty = fill.filledQty;
  const trade: ActiveTrade = {
    symbol: alpacaSymbol,
    coinbaseSymbol,
    direction,
    signal,
    entryPrice: fill.avgPrice,
    entryTime: nowIso(),
    stopLoss: signal.stopLoss,
    target: signal.target,
    scaled: false,
    totalQty: actualQty,
    remainingQty: actualQty,
    executedValue: fill.avgPrice * actualQty,
    units: actualQty >= 0.5 ? 2 : 1, // 2 units if we have at least 0.5 crypto
  };

  trades.set(coinbaseSymbol, trade);
  log(coinbaseSymbol, `Entered ${direction.toUpperCase()} position @ $${fill.avgPrice.toFixed(2)} (${actualQty.toFixed(4)} units)`);
}

/**
 * Scale position at middle band
 */
async function scaleTrade(trade: ActiveTrade) {
  if (trade.scaled || trade.units < 2 || trade.remainingQty <= 0.1) {
    return;
  }

  let scaleQty = trade.totalQty / 2;
  if (scaleQty >= trade.remainingQty) {
    scaleQty = Math.max(0.1, trade.remainingQty - 0.1);
  }
  if (scaleQty <= 0) {
    return;
  }

  const side = sideForDirection(trade.direction, 'exit');
  log(trade.coinbaseSymbol, `Scaling out ${scaleQty.toFixed(4)} units at target $${trade.target?.toFixed(2) ?? 'n/a'}`);

  const referencePrice = trade.target ?? trade.entryPrice;
  const fill = await placeAdaptiveOrder(trade.symbol, side, scaleQty, 'scale', referencePrice);

  if (!fill || fill.filledQty === 0) {
    log(trade.coinbaseSymbol, 'Scale order not filled; retaining position');
    return;
  }

  trade.remainingQty = Math.max(0, trade.remainingQty - fill.filledQty);
  trade.scaled = true;
  trade.units = 1;

  // Adjust stops and targets after scaling
  if (trade.direction === 'long') {
    if (trade.target) {
      trade.stopLoss = trade.target * 0.99; // Lock in profit (~1%)
      trade.target = (trade.signal.bbUpper ?? trade.target) * 0.99;
    }
  } else {
    if (trade.target) {
      trade.stopLoss = trade.target * 1.01;
      trade.target = (trade.signal.bbLower ?? trade.target) * 1.01;
    }
  }

  log(trade.coinbaseSymbol, `Scaled position; remaining ${trade.remainingQty.toFixed(4)} units`);
}

/**
 * Exit trade completely
 */
async function exitTrade(trade: ActiveTrade, reason: 'stop' | 'target' | 'manual') {
  if (trade.remainingQty <= 0) {
    trades.delete(trade.coinbaseSymbol);
    return;
  }

  const side = sideForDirection(trade.direction, 'exit');
  const referencePrice = trade.stopLoss ?? trade.target ?? trade.entryPrice;
  log(trade.coinbaseSymbol, `Closing position (${reason}) for ${trade.remainingQty.toFixed(4)} units`);

  const fill = await placeAdaptiveOrder(
    trade.symbol,
    side,
    trade.remainingQty,
    reason === 'stop' ? 'exit-stop' : 'exit-eod',
    referencePrice,
  );

  if (!fill || fill.filledQty === 0) {
    log(trade.coinbaseSymbol, 'Exit order failed; leaving trade open');
    return;
  }

  log(
    trade.coinbaseSymbol,
    `Exit filled (${reason}) avg $${fill.avgPrice.toFixed(2)} for ${fill.filledQty.toFixed(4)} units`,
  );
  trades.delete(trade.coinbaseSymbol);
}

/**
 * Handle real-time tick from Coinbase WebSocket
 */
async function handleRealtimeTick(coinbaseSymbol: string, tick: CoinbaseTick) {
  const trade = trades.get(coinbaseSymbol);
  if (!trade || trade.processing) {
    return;
  }

  trade.processing = true;
  try {
    const price = tick.price;

    // Check scaling opportunity
    if (!trade.scaled && typeof trade.target === 'number') {
      const hitTarget =
        (trade.direction === 'long' && price >= trade.target) ||
        (trade.direction === 'short' && price <= trade.target);

      if (hitTarget) {
        await scaleTrade(trade);
        return;
      }
    }

    // Check stop loss
    if (trade.stopLoss !== null) {
      const hitStop =
        (trade.direction === 'long' && price <= trade.stopLoss) ||
        (trade.direction === 'short' && price >= trade.stopLoss);

      if (hitStop) {
        await exitTrade(trade, 'stop');
        return;
      }
    }

    // Check final target (after scaling)
    if (trade.units === 1 && typeof trade.target === 'number') {
      const finalTarget =
        (trade.direction === 'long' && price >= trade.target) ||
        (trade.direction === 'short' && price <= trade.target);

      if (finalTarget) {
        await exitTrade(trade, 'target');
      }
    }
  } catch (err) {
    log(coinbaseSymbol, `Realtime handler error: ${(err as Error).message}`);
  } finally {
    const stillActive = trades.get(coinbaseSymbol);
    if (stillActive) {
      stillActive.processing = false;
    }
  }
}

/**
 * Process active trade (fallback to 1-min bars if WebSocket lags)
 */
async function processActiveTrade(trade: ActiveTrade) {
  if (trade.processing) {
    return;
  }

  trade.processing = true;

  try {
    // Verify position still exists in Alpaca
    const position = await getPosition(trade.symbol);
    if (!position || Number(position.qty) === 0) {
      trades.delete(trade.coinbaseSymbol);
      log(trade.coinbaseSymbol, 'Position closed externally; removing from tracker');
      return;
    }

    // Normalize remaining quantity against broker position
    const brokerQty = Math.abs(Number(position.qty));
    trade.remainingQty = Math.min(trade.remainingQty, brokerQty);
  } finally {
    const stillActive = trades.get(trade.coinbaseSymbol);
    if (stillActive) {
      stillActive.processing = false;
    }
  }
}

/**
 * Process symbol for new entry signals
 */
async function processSymbol(coinbaseSymbol: string, alpacaSymbol: string) {
  // Fetch 15-minute bars from Coinbase
  const bars = await coinbaseWS.getRecentBars(coinbaseSymbol, '15Min', 40);

  if (bars.length < 20) {
    log(coinbaseSymbol, `Insufficient 15m bars (${bars.length}/20) for signal`);
    return;
  }

  const latestBar = bars[bars.length - 1];
  const barId = latestBar.t;
  const prevBar = lastProcessedBar.get(coinbaseSymbol);

  if (prevBar === barId) {
    return; // Already processed this bar
  }

  lastProcessedBar.set(coinbaseSymbol, barId);

  const closes = bars.map(b => b.c);
  const price = latestBar.c;

  // Generate mean reversion signal
  const signal = generateMeanReversionSignalFromTechnicals(
    coinbaseSymbol,
    price,
    closes,
    1, // netGex = 1 (positive, bypassed)
    {
      rsiPeriod: CONFIG.rsiPeriod,
      rsiOversold: CONFIG.rsiOversold,
      rsiOverbought: CONFIG.rsiOverbought,
      bbPeriod: CONFIG.bbPeriod,
      bbStdDev: CONFIG.bbStdDev,
      bbThreshold: CONFIG.bbThreshold,
      stopLossPercent: CONFIG.stopLossPercent,
    },
  );

  if (signal.direction === 'none') {
    log(coinbaseSymbol, `No signal on latest bar @ $${price.toFixed(2)}`);
    return;
  }

  if (trades.has(coinbaseSymbol)) {
    log(coinbaseSymbol, 'Signal generated but trade already active; ignoring');
    return;
  }

  log(coinbaseSymbol, `🚨 ${signal.direction.toUpperCase()} signal: ${signal.rationale.join(' | ')}`);
  await attemptEntry(coinbaseSymbol, alpacaSymbol, signal, price);
}

/**
 * Main loop
 */
async function main() {
  console.log('🚀 Starting Live Crypto Mean Reversion Strategy');
  console.log(`📊 Market Data: Coinbase Public WebSocket`);
  console.log(`💼 Order Execution: Alpaca Paper Trading`);
  console.log(`💰 Symbols: ${coinbaseSymbols.join(', ')}`);
  console.log(`📈 Strategy: 2-unit scaling with RSI + Bollinger Bands\n`);

  // Connect to Coinbase WebSocket
  console.log('Connecting to Coinbase WebSocket...');
  coinbaseWS.connect();

  // Wait for connection
  await sleep(3000);

  if (!coinbaseWS.connected()) {
    console.error('❌ Failed to connect to Coinbase WebSocket');
    return;
  }

  console.log('✅ Connected to Coinbase WebSocket\n');

  // Subscribe to real-time ticks
  coinbaseWS.on('tick', (symbol: string, tick: CoinbaseTick) => {
    handleRealtimeTick(symbol, tick).catch(err => {
      log(symbol, `Tick handler error: ${(err as Error).message}`);
    });
  });

  // Main polling loop
  while (true) {
    try {
      for (const [coinbaseSymbol, alpacaSymbol] of Object.entries(CONFIG.symbols)) {
        try {
          // Process for new signals
          await processSymbol(coinbaseSymbol, alpacaSymbol);

          // Monitor active trades
          const trade = trades.get(coinbaseSymbol);
          if (trade) {
            await processActiveTrade(trade);
          }
        } catch (err) {
          log(coinbaseSymbol, `Error in processing: ${(err as Error).message}`);
        }
      }
    } catch (err) {
      console.error(`[${nowIso()}] Fatal loop error`, err);
    }

    await sleep(CONFIG.pollIntervalMs);
  }
}

main().catch(err => {
  console.error('❌ Live strategy crashed:', err);
  coinbaseWS.disconnect();
  process.exit(1);
});
