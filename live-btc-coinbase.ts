import express from 'express';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import path from 'path';
import WebSocket from 'ws';
import fetch from 'node-fetch';

const ADVANCED_TRADE_WS_URL = 'wss://advanced-trade-ws.coinbase.com';
const ADVANCED_TRADE_JWT = process.env.COINBASE_ADVANCED_TRADE_JWT;

const app = express();
const httpServer = createServer(app);
const io = new SocketIOServer(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  }
});

const PORT = 3338;
const PRODUCT_ID = 'BTC-USD';

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Serve dashboard
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'btc-dashboard.html'));
});

console.log('================================================================================');
console.log('BITCOIN (BTC-USD) LIVE DASHBOARD - COINBASE PRO');
console.log('================================================================================');
console.log(`Product: ${PRODUCT_ID}`);
console.log(`Dashboard: http://localhost:${PORT}`);
console.log(`Data Source: Coinbase Pro WebSocket Feed`);
console.log('Features: Level 2 Orderbook | CVD | Large Order Detection');
console.log('================================================================================\n');

// State
let coinbaseWs: WebSocket | null = null;
let advancedTradeWs: WebSocket | null = null;
let orderbook: { bids: [string, string][], asks: [string, string][] } = { bids: [], asks: [] };
let fullOrderbook: { bids: [string, string][], asks: [string, string][] } = { bids: [], asks: [] };
let candles: any[] = [];
let currentCandle: any = null;
let lastPrice = 0;
let volume24h = 0;
let priceChange24h = 0;
const ORDERBOOK_RANGE_PERCENT = 0.10; // Keep orders within 10% of current price

// Logging
function log(message: string) {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
}

// Connect to Coinbase WebSocket
function connectToCoinbase() {
  log('Connecting to Coinbase Pro WebSocket...');

  coinbaseWs = new WebSocket('wss://ws-feed.exchange.coinbase.com');

  coinbaseWs.on('open', () => {
    log('✅ Connected to Coinbase WebSocket');

    // Subscribe to level2_batch (public orderbook), ticker, and matches channels
    const subscribeMessage = {
      type: 'subscribe',
      product_ids: [PRODUCT_ID],
      channels: [
        'level2_batch',  // Use level2_batch instead of level2 (doesn't require auth)
        'ticker',
        'matches'
      ]
    };

    coinbaseWs?.send(JSON.stringify(subscribeMessage));
    log(`📡 Subscribed to ${PRODUCT_ID} (level2_batch, ticker, matches)`);

    io.emit('log', { message: 'Connected to Coinbase Pro', type: 'success' });
  });

  coinbaseWs.on('message', (data: WebSocket.Data) => {
    try {
      const message = JSON.parse(data.toString());

      // Log ALL incoming messages to see what we're actually getting
      if (message.type !== 'heartbeat') {  // Skip heartbeat spam
        log(`📨 Received message type: ${message.type}`);
        if (message.type === 'snapshot' || message.type === 'l2update' || message.type === 'error' || message.type === 'subscriptions') {
          log(`📨 Message details: ${JSON.stringify(message)}`);
        }
      }

      // Handle snapshot (initial orderbook) - DISABLED, using deep REST API instead
      if (message.type === 'snapshot') {
        // Store in fullOrderbook for potential future use, but don't emit
        fullOrderbook.bids = message.bids || [];
        fullOrderbook.asks = message.asks || [];
        log(`📚 WebSocket orderbook snapshot received (${fullOrderbook.bids.length} bids, ${fullOrderbook.asks.length} asks) - using REST API instead`);
      }

      // Handle L2 updates - DISABLED, using deep REST API instead
      if (message.type === 'l2update') {
        // We're not using WebSocket orderbook updates anymore
        // The deep REST API orderbook is fetched every 5 seconds
        // This gives us much better depth (500+ levels vs 50 levels)
      }

      // Handle ticker updates (price, volume, etc.)
      if (message.type === 'ticker') {
        lastPrice = parseFloat(message.price);
        volume24h = parseFloat(message.volume_24h || 0);

        // Calculate 24h change
        const open24h = parseFloat(message.open_24h || lastPrice);
        priceChange24h = ((lastPrice - open24h) / open24h) * 100;

        // Log emits occasionally
        if (Math.random() < 0.01) {
          log(`💰 Emitting price: $${lastPrice}, connected clients: ${io.engine.clientsCount}`);
        }

        io.emit('price', { price: lastPrice });
        io.emit('stats', {
          volume: volume24h,
          change: priceChange24h
        });

        // Build candles from WebSocket ticker feed
        updateCandle(lastPrice, message.time);
      }

      // Handle match (trade) updates for CVD and aggressive whale detection
      if (message.type === 'match') {
        const price = parseFloat(message.price);
        const size = parseFloat(message.size);
        const makerSide = message.side as 'buy' | 'sell'; // Coinbase Pro: maker side
        const takerSide = makerSide === 'buy' ? 'sell' : 'buy'; // flip to aggressor
        const time = message.time;

        const trade = {
          price,
          size,
          side: takerSide, // use taker (aggressor) side for CVD
          time
        };

        // Emit regular trade for CVD
        io.emit('trade', trade);

        // Also emit as aggressive_trade for whale detection
        const timestampMs = new Date(time).getTime();
        const aggressiveTrade = {
          price,
          size,
          side: takerSide, // aggressor side
          makerSide,
          time,
          timestampMs,
          productId: PRODUCT_ID,
        };

        io.emit('aggressive_trade', aggressiveTrade);
      }

    } catch (error: any) {
      log(`❌ Error processing WebSocket message: ${error.message}`);
    }
  });

  coinbaseWs.on('error', (error) => {
    log(`❌ WebSocket error: ${error.message}`);
  });

  coinbaseWs.on('close', () => {
    log('⚠️ WebSocket connection closed. Reconnecting in 5 seconds...');
    io.emit('log', { message: 'Disconnected from Coinbase. Reconnecting...', type: 'warning' });

    setTimeout(() => {
      connectToCoinbase();
    }, 5000);
  });
}

// Connect to Coinbase Advanced Trade WebSocket for aggressive trade flow (market_trades)
function connectToAdvancedTradeMarketTrades() {
  if (!ADVANCED_TRADE_JWT) {
    log('⚠️ COINBASE_ADVANCED_TRADE_JWT not set - skipping Advanced Trade market_trades feed');
    io.emit('log', {
      message: 'Advanced Trade market_trades feed disabled (missing COINBASE_ADVANCED_TRADE_JWT)',
      type: 'warning',
    });
    return;
  }

  log('Connecting to Coinbase Advanced Trade WebSocket (market_trades)...');

  advancedTradeWs = new WebSocket(ADVANCED_TRADE_WS_URL);

  advancedTradeWs.on('open', () => {
    log('✅ Connected to Coinbase Advanced Trade WebSocket');

    const subscribeMessage = {
      type: 'subscribe',
      channel: 'market_trades',
      product_ids: [PRODUCT_ID],
      jwt: ADVANCED_TRADE_JWT,
    };

    advancedTradeWs?.send(JSON.stringify(subscribeMessage));
    log(`📡 Subscribed to Advanced Trade market_trades for ${PRODUCT_ID}`);
    io.emit('log', { message: 'Connected to Coinbase Advanced Trade (market_trades)', type: 'success' });
  });

  advancedTradeWs.on('message', (data: WebSocket.Data) => {
    try {
      const message = JSON.parse(data.toString());

      // Log ALL messages to debug (100% for troubleshooting)
      log(`📬 Advanced Trade message received: ${JSON.stringify(message).substring(0, 500)}`);

      if (message.channel !== 'market_trades' || !Array.isArray(message.events)) {
        // Log unexpected message types
        log(`📬 Advanced Trade non-market_trades message: ${JSON.stringify(message)}`);
        return;
      }

      let tradesEmitted = 0;
      for (const event of message.events) {
        if (!event || event.type !== 'update' || !Array.isArray(event.trades)) continue;

        for (const trade of event.trades) {
          const takerSideRaw = trade.side as string | undefined; // "BUY" or "SELL" (taker/aggressor side from Coinbase)
          const size = parseFloat(String(trade.size));
          const price = parseFloat(String(trade.price));
          const productId = trade.product_id as string | undefined;
          const timeStr = trade.time as string | undefined;

          if (!Number.isFinite(price) || !Number.isFinite(size) || !timeStr || !takerSideRaw) {
            log(`⚠️ Skipping invalid Advanced Trade message: ${JSON.stringify(trade)}`);
            continue;
          }

          // Coinbase sends the TAKER side (aggressor) directly - no need to flip!
          const aggressorSide = takerSideRaw.toLowerCase(); // 'buy' or 'sell'

          // Debug logging to verify side logic (log every 50th trade)
          if (Math.random() < 0.02) {
            log(`🔍 TRADE DEBUG: Price=$${price.toFixed(2)} Size=${size.toFixed(4)} TakerSide=${takerSideRaw} → AggressorSide=${aggressorSide}`);
          }

          const timestampMs = new Date(timeStr).getTime();
          if (!Number.isFinite(timestampMs)) {
            log(`⚠️ Invalid Advanced Trade timestamp: ${timeStr}`);
            continue;
          }

          const aggressiveTrade = {
            price,
            size,
            side: aggressorSide as 'buy' | 'sell',
            time: timeStr,
            timestampMs,
            productId: productId || PRODUCT_ID,
          };

          // Emit raw aggressive trade for client-side visualization (Bookmap-style bubbles, delta, etc.)
          io.emit('aggressive_trade', aggressiveTrade);
          tradesEmitted++;
        }
      }

      // Log successful emissions occasionally
      if (tradesEmitted > 0 && Math.random() < 0.05) {
        log(`🐋 Emitted ${tradesEmitted} aggressive trades to clients`);
      }
    } catch (error: any) {
      log(`❌ Error processing Advanced Trade WebSocket message: ${error.message}`);
    }
  });

  advancedTradeWs.on('error', (error) => {
    log(`❌ Advanced Trade WebSocket error: ${error.message}`);
  });

  advancedTradeWs.on('close', () => {
    log('⚠️ Advanced Trade WebSocket connection closed. Reconnecting in 10 seconds...');
    io.emit('log', { message: 'Disconnected from Coinbase Advanced Trade. Reconnecting...', type: 'warning' });

    setTimeout(() => {
      connectToAdvancedTradeMarketTrades();
    }, 10000);
  });
}

// Update current candle (1-minute bars)
function updateCandle(price: number, timestamp: string) {
  // Validate price
  if (!price || isNaN(price) || price <= 0) {
    log(`⚠️ Invalid price received: ${price}`);
    return;
  }

  // Log timestamp calculation occasionally for debugging
  if (Math.random() < 0.01) {
    const ms = new Date(timestamp).getTime();
    const candleTime = Math.floor(ms / 60000) * 60;
    log(`🕐 Timestamp debug: "${timestamp}" -> ${ms}ms -> ${candleTime}s (${new Date(candleTime * 1000).toISOString()})`);
  }

  const candleTime = Math.floor(new Date(timestamp).getTime() / 60000) * 60;

  // Validate timestamp
  if (!candleTime || isNaN(candleTime)) {
    log(`⚠️ Invalid timestamp: ${timestamp} -> ${candleTime}`);
    return;
  }

  if (!currentCandle || currentCandle.time !== candleTime) {
    // Save previous candle
    if (currentCandle) {
      candles.push(currentCandle);
      // Keep only last 500 candles
      if (candles.length > 500) {
        candles = candles.slice(-500);
      }
    }

    // Create new candle
    currentCandle = {
      time: candleTime,
      open: price,
      high: price,
      low: price,
      close: price
    };
  } else {
    // Update current candle
    currentCandle.high = Math.max(currentCandle.high, price);
    currentCandle.low = Math.min(currentCandle.low, price);
    currentCandle.close = price;
  }

  // Validate the candle before emitting
  if (currentCandle &&
      currentCandle.time != null &&
      currentCandle.open != null &&
      currentCandle.high != null &&
      currentCandle.low != null &&
      currentCandle.close != null &&
      !isNaN(currentCandle.time) &&
      !isNaN(currentCandle.open) &&
      !isNaN(currentCandle.high) &&
      !isNaN(currentCandle.low) &&
      !isNaN(currentCandle.close) &&
      currentCandle.high >= currentCandle.low &&
      currentCandle.high >= Math.min(currentCandle.open, currentCandle.close) &&
      currentCandle.low <= Math.max(currentCandle.open, currentCandle.close)) {

    // Emit a clean copy to avoid reference issues
    const candleData = {
      time: currentCandle.time,
      open: currentCandle.open,
      high: currentCandle.high,
      low: currentCandle.low,
      close: currentCandle.close
    };

    // Log every 10th candle update to avoid spam
    if (Math.random() < 0.1) {
      log(`📊 Emitting candle: ${JSON.stringify(candleData)}`);
    }

    io.emit('candle', candleData);
  } else {
    log(`⚠️ Invalid candle data, not emitting: ${JSON.stringify(currentCandle)}`);
  }
}

// Fetch historical candles from Coinbase REST API
async function fetchHistoricalCandles() {
  try {
    log('Fetching historical candles from Coinbase REST API...');

    const endTime = new Date();
    const startTime = new Date(endTime.getTime() - 5 * 60 * 60 * 1000); // 5 hours ago (300 minutes, Coinbase max)

    const url = `https://api.exchange.coinbase.com/products/${PRODUCT_ID}/candles?start=${startTime.toISOString()}&end=${endTime.toISOString()}&granularity=60`;
    log(`📍 Fetching from: ${url}`);

    const response = await fetch(url);
    log(`📍 Response status: ${response.status} ${response.statusText}`);

    const data: any = await response.json();
    log(`📍 Response data type: ${Array.isArray(data) ? 'array' : typeof data}, length: ${Array.isArray(data) ? data.length : 'N/A'}`);

    if (Array.isArray(data)) {
      // Coinbase returns: [time, low, high, open, close, volume]
      // Coinbase API returns max 300 candles per request
      candles = data
        .filter((candle: number[]) => {
          // Ensure all required fields are present and valid
          return candle &&
                 candle[0] != null &&
                 candle[1] != null &&
                 candle[2] != null &&
                 candle[3] != null &&
                 candle[4] != null;
        })
        .map((candle: number[]) => ({
          time: candle[0],
          open: parseFloat(String(candle[3])),
          high: parseFloat(String(candle[2])),
          low: parseFloat(String(candle[1])),
          close: parseFloat(String(candle[4])),
          volume: candle[5]
        }))
        .filter(c => !isNaN(c.open) && !isNaN(c.high) && !isNaN(c.low) && !isNaN(c.close))
        .reverse(); // Reverse to get oldest first

      log(`✅ Fetched ${candles.length} historical candles (${Math.floor(candles.length / 60)}h ${candles.length % 60}m)`);

      // Send to all connected clients
      io.emit('candles', candles);
    } else {
      log(`❌ Unexpected response format: ${JSON.stringify(data).substring(0, 200)}`);
    }
  } catch (error: any) {
    log(`❌ Error fetching historical candles: ${error.message}`);
    log(`❌ Stack: ${error.stack}`);
  }
}

// Fetch deep orderbook from Coinbase REST API (Level 2 - full book)
async function fetchDeepOrderbook() {
  try {
    // Coinbase Pro REST API - Level 2 (aggregated orderbook)
    // Level 3 is deprecated, but Level 2 gives us the full aggregated book
    const url = `https://api.exchange.coinbase.com/products/${PRODUCT_ID}/book?level=2`;

    const response = await fetch(url);
    if (!response.ok) {
      log(`❌ Failed to fetch orderbook: ${response.status} ${response.statusText}`);
      return;
    }

    const data: any = await response.json();

    if (data.bids && data.asks) {
      fullOrderbook.bids = data.bids;
      fullOrderbook.asks = data.asks;

      log(`📚 Fetched deep orderbook: ${data.bids.length} bids, ${data.asks.length} asks`);

      // Filter to 10% range and emit
      filterAndEmitOrderbook();
    }
  } catch (error: any) {
    log(`❌ Error fetching deep orderbook: ${error.message}`);
  }
}

// Filter orderbook to keep only orders within X% of current price
function filterAndEmitOrderbook() {
  // Use lastPrice if available, otherwise calculate mid-price from orderbook
  let referencePrice = lastPrice;

  if (!referencePrice || referencePrice === 0) {
    if (fullOrderbook.bids.length > 0 && fullOrderbook.asks.length > 0) {
      const bestBid = parseFloat(fullOrderbook.bids[0][0]);
      const bestAsk = parseFloat(fullOrderbook.asks[0][0]);
      referencePrice = (bestBid + bestAsk) / 2;
      log(`📍 Using orderbook mid-price as reference: $${referencePrice.toFixed(2)}`);
    } else {
      log(`⚠️ No price data or orderbook data yet, skipping filter`);
      return;
    }
  }

  const lowerBound = referencePrice * (1 - ORDERBOOK_RANGE_PERCENT);
  const upperBound = referencePrice * (1 + ORDERBOOK_RANGE_PERCENT);

  // Filter bids (buy orders) - keep those above lower bound
  const filteredBids = fullOrderbook.bids
    .filter(([price]) => parseFloat(price) >= lowerBound)
    .slice(0, 500); // Limit to 500 levels max

  // Filter asks (sell orders) - keep those below upper bound
  const filteredAsks = fullOrderbook.asks
    .filter(([price]) => parseFloat(price) <= upperBound)
    .slice(0, 500); // Limit to 500 levels max

  orderbook.bids = filteredBids;
  orderbook.asks = filteredAsks;

  // Log the range
  if (filteredBids.length > 0 && filteredAsks.length > 0) {
    const lowestBid = parseFloat(filteredBids[filteredBids.length - 1][0]);
    const highestBid = parseFloat(filteredBids[0][0]);
    const lowestAsk = parseFloat(filteredAsks[0][0]);
    const highestAsk = parseFloat(filteredAsks[filteredAsks.length - 1][0]);

    log(`📊 Orderbook filtered (${ORDERBOOK_RANGE_PERCENT * 100}% range around $${lastPrice.toFixed(2)})`);
    log(`   Bids: ${filteredBids.length} levels ($${lowestBid.toFixed(2)} - $${highestBid.toFixed(2)})`);
    log(`   Asks: ${filteredAsks.length} levels ($${lowestAsk.toFixed(2)} - $${highestAsk.toFixed(2)})`);
  }

  // Emit to all connected clients
  io.emit('orderbook', orderbook);
}

// Track which clients have received historical candles
const clientsWithCandles = new Set<string>();

// Socket.IO connection handling
io.on('connection', (socket) => {
  log(`[CONNECTION] Dashboard connected from ${socket.handshake.address}`);
  log(`[CONNECTION] Candles array length: ${candles.length}`);
  log(`[CONNECTION] Orderbook bids: ${orderbook.bids.length}, asks: ${orderbook.asks.length}`);

  if (!ADVANCED_TRADE_JWT) {
    socket.emit('log', {
      message: 'Advanced Trade aggressive flow (market_trades) disabled: COINBASE_ADVANCED_TRADE_JWT not set',
      type: 'warning',
    });
  }

  // Send current state
  socket.emit('orderbook', orderbook);
  socket.emit('price', { price: lastPrice });
  socket.emit('stats', { volume: volume24h, change: priceChange24h });

  // Log orderbook price range
  if (orderbook.bids.length > 0 && orderbook.asks.length > 0) {
    const lowestBid = parseFloat(orderbook.bids[orderbook.bids.length - 1][0]);
    const highestBid = parseFloat(orderbook.bids[0][0]);
    const lowestAsk = parseFloat(orderbook.asks[0][0]);
    const highestAsk = parseFloat(orderbook.asks[orderbook.asks.length - 1][0]);
    log(`[CONNECTION] Orderbook range: Bids $${lowestBid.toFixed(2)} - $${highestBid.toFixed(2)} | Asks $${lowestAsk.toFixed(2)} - $${highestAsk.toFixed(2)}`);
    log(`[CONNECTION] Spread: $${(lowestAsk - highestBid).toFixed(2)}`);
  }

  // Only send historical candles if this client hasn't received them yet
  const clientId = socket.id;
  if (!clientsWithCandles.has(clientId) && candles.length > 0) {
    log(`[CONNECTION] ✅ Sending ${candles.length} historical candles to NEW client ${clientId}`);
    socket.emit('candles', candles);
    clientsWithCandles.add(clientId);
  } else if (clientsWithCandles.has(clientId)) {
    log(`[CONNECTION] ⏭️ Skipping historical candles for RECONNECTED client ${clientId} (already has them)`);
  } else {
    log(`[CONNECTION] ⚠️ WARNING: No candles available to send!`);
  }

  socket.on('disconnect', () => {
    log(`[CONNECTION] Dashboard disconnected from ${socket.handshake.address}`);
    // Keep client in set so they don't get candles again on reconnect
    // clientsWithCandles.delete(socket.id); // DON'T delete - prevents reload on reconnect
  });
});

// Start server
httpServer.listen(PORT, async () => {
  log(`Dashboard server running on http://localhost:${PORT}`);

  // Fetch historical data
  await fetchHistoricalCandles();

  // Fetch initial deep orderbook
  await fetchDeepOrderbook();

  // Refresh deep orderbook every 5 seconds
  setInterval(async () => {
    await fetchDeepOrderbook();
  }, 5000);
  log(`🔄 Deep orderbook refresh enabled (every 5s, ${ORDERBOOK_RANGE_PERCENT * 100}% range)`);

  // Connect to Coinbase WebSocket
  connectToCoinbase();
  // Connect to Advanced Trade market_trades feed for aggressive flow
  connectToAdvancedTradeMarketTrades();
});

// Graceful shutdown
process.on('SIGINT', () => {
  log('\n[SHUTDOWN] Received SIGINT, closing connections...');

  if (coinbaseWs) {
    coinbaseWs.close();
  }
  if (advancedTradeWs) {
    advancedTradeWs.close();
  }

  httpServer.close(() => {
    log('[SHUTDOWN] Server closed');
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  log('\n[SHUTDOWN] Received SIGTERM, closing connections...');

  if (coinbaseWs) {
    coinbaseWs.close();
  }
  if (advancedTradeWs) {
    advancedTradeWs.close();
  }

  httpServer.close(() => {
    log('[SHUTDOWN] Server closed');
    process.exit(0);
  });
});
