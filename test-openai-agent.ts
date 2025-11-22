import { analyzeFuturesMarket, FuturesMarketData } from './lib/openaiTradingAgent';
import { fabioPlaybook } from './lib/fabioPlaybook';

/**
 * Test the OpenAI Fabio trading agent with sample market data
 * Simulates a futures market out-of-balance uptrend setup
 */
async function testOpenAIAgent() {
  console.log('🤖 Testing OpenAI Fabio Trading Agent...\n');

  // Create sample market data simulating NQ (NASDAQ futures)
  // Scenario: Out-of-balance uptrend with price near an LVN and strong CVD
  const sampleMarketData: FuturesMarketData = {
    symbol: 'NQZ5',
    timestamp: new Date().toISOString(),
    currentPrice: 19850.50,

    // Last 5 candles (25 minutes)
    candles: [
      {
        timestamp: new Date(Date.now() - 20 * 60000).toISOString(),
        open: 19780,
        high: 19810,
        low: 19770,
        close: 19805,
        volume: 1250,
      },
      {
        timestamp: new Date(Date.now() - 15 * 60000).toISOString(),
        open: 19805,
        high: 19830,
        low: 19800,
        close: 19828,
        volume: 1380,
      },
      {
        timestamp: new Date(Date.now() - 10 * 60000).toISOString(),
        open: 19828,
        high: 19835,
        low: 19815,
        close: 19832,
        volume: 1290,
      },
      {
        timestamp: new Date(Date.now() - 5 * 60000).toISOString(),
        open: 19832,
        high: 19850,
        low: 19820,
        close: 19847,
        volume: 1420,
      },
      {
        timestamp: new Date().toISOString(),
        open: 19847,
        high: 19855,
        low: 19840,
        close: 19850,
        volume: 1350,
      },
    ],

    // CVD showing bullish momentum (uptrend)
    cvd: {
      value: 450,
      trend: 'up',
      ohlc: {
        timestamp: new Date().toISOString(),
        open: 200,
        high: 500,
        low: 180,
        close: 450,
      },
    },

    // Strong buyer absorption
    orderFlow: {
      buyAbsorption: 0.72,
      sellAbsorption: 0.28,
      buyExhaustion: 0.15,
      sellExhaustion: 0.85,
      bigPrints: [
        { price: 19845, size: 150, side: 'buy', timestamp: new Date(Date.now() - 2 * 60000).toISOString() },
        { price: 19847, size: 200, side: 'buy', timestamp: new Date(Date.now() - 1 * 60000).toISOString() },
        { price: 19850, size: 120, side: 'buy', timestamp: new Date().toISOString() },
      ],
    },

    // Volume profile with POC, VAH, VAL, and LVNs
    volumeProfile: {
      poc: 19815, // Fair value
      vah: 19835, // Value area high
      val: 19790, // Value area low
      lvns: [19820, 19840], // Low volume nodes (reaction zones)
      sessionHigh: 19855,
      sessionLow: 19770,
    },

    // Market state: out-of-balance uptrend
    marketState: {
      state: 'out_of_balance_uptrend',
      buyersControl: 0.75,
      sellersControl: 0.25,
    },

    // Order flow is confirmed
    orderFlowConfirmed: true,

    // Account info
    account: {
      balance: 50000,
      position: 0,
      unrealizedPnL: 0,
      realizedPnL: 1250,
    },
  };

  try {
    console.log('📊 Sample Market Data:');
    console.log(`Symbol: ${sampleMarketData.symbol}`);
    console.log(`Current Price: ${sampleMarketData.currentPrice}`);
    console.log(`Market State: ${sampleMarketData.marketState.state}`);
    console.log(`CVD Trend: ${sampleMarketData.cvd.trend}`);
    console.log(`Buyers Control: ${(sampleMarketData.marketState.buyersControl * 100).toFixed(1)}%\n`);

    console.log('🔄 Calling OpenAI agent...\n');
    const decision = await analyzeFuturesMarket(sampleMarketData);

    console.log('✅ Decision Result:');
    console.log(JSON.stringify(decision, null, 2));

    console.log('\n📋 Decision Summary:');
    console.log(`Decision: ${decision.decision}`);
    console.log(`Confidence: ${decision.confidence}%`);
    console.log(`Market State: ${decision.marketState}`);
    console.log(`Location: ${decision.location}`);
    console.log(`Setup Model: ${decision.setupModel || 'None'}`);
    console.log(`Risk Percent: ${(decision.riskPercent * 100).toFixed(2)}%`);
    if (decision.entryPrice) console.log(`Entry Price: ${decision.entryPrice}`);
    if (decision.stopLoss) console.log(`Stop Loss: ${decision.stopLoss}`);
    if (decision.target) console.log(`Target: ${decision.target}`);
    console.log(`Reasoning: ${decision.reasoning}\n`);

    console.log('✨ Test completed successfully!');
  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
}

// Run the test
testOpenAIAgent();
