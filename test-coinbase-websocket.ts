import 'dotenv/config';
import { CoinbaseWebSocketClient } from './lib/coinbaseWebSocket';

/**
 * Test Coinbase WebSocket connection
 * Tests real-time price feed and historical candle fetching
 */

async function testCoinbaseWebSocket() {
  console.log('🧪 Testing Coinbase WebSocket Client\n');

  const client = new CoinbaseWebSocketClient({
    symbols: ['SOL-USD', 'BTC-USD', 'ETH-USD'],
  });

  // Track ticks
  let tickCount = 0;
  const ticksBySymbol = new Map<string, number>();

  client.on('tick', (symbol: string, tick: any) => {
    tickCount++;
    ticksBySymbol.set(symbol, (ticksBySymbol.get(symbol) || 0) + 1);

    if (tickCount <= 10) {
      console.log(`[TICK] ${symbol}: $${tick.price.toFixed(2)} (${tick.side}) size: ${tick.size}`);
    }
  });

  client.on('trade', (symbol: string, trade: any) => {
    console.log(`[TRADE] ${symbol}: $${trade.price.toFixed(2)} (${trade.side}) size: ${trade.size}`);
  });

  // Connect
  console.log('Connecting to Coinbase WebSocket...');
  client.connect();

  // Wait for connection and ticks
  await new Promise(resolve => setTimeout(resolve, 5000));

  if (!client.connected()) {
    console.error('❌ Failed to connect to Coinbase WebSocket');
    process.exit(1);
  }

  console.log('\n✅ Connected to Coinbase WebSocket');
  console.log(`📊 Received ${tickCount} ticks in 5 seconds\n`);

  for (const [symbol, count] of ticksBySymbol.entries()) {
    const price = client.getLatestPrice(symbol);
    console.log(`${symbol}: ${count} ticks | Latest price: $${price?.toFixed(2) ?? 'N/A'}`);
  }

  // Test fetching historical candles
  console.log('\n📈 Fetching historical 15-minute candles...');

  for (const symbol of ['SOL-USD', 'BTC-USD']) {
    try {
      const bars = await client.getRecentBars(symbol, '15Min', 5);
      console.log(`\n${symbol} - Last 5 bars (15-minute):`);
      bars.forEach((bar, i) => {
        console.log(
          `  ${i + 1}. ${bar.t} | O: $${bar.o.toFixed(2)} H: $${bar.h.toFixed(2)} L: $${bar.l.toFixed(2)} C: $${bar.c.toFixed(2)} V: ${bar.v.toFixed(2)}`
        );
      });
    } catch (err) {
      console.error(`Failed to fetch candles for ${symbol}:`, err);
    }
  }

  // Wait a bit longer to see more ticks
  console.log('\n⏳ Collecting more ticks for 10 seconds...');
  await new Promise(resolve => setTimeout(resolve, 10000));

  console.log(`\n📊 Total ticks received: ${tickCount}`);
  console.log('✅ Test complete!');

  // Disconnect
  client.disconnect();
  process.exit(0);
}

testCoinbaseWebSocket().catch(err => {
  console.error('❌ Test failed:', err);
  process.exit(1);
});
