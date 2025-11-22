import { analyzeFuturesMarket, FuturesMarketData } from './lib/openaiTradingAgent';

// Set API key directly
process.env.OPENAI_API_KEY = 'sk-49cc38b398554122be148e0f06209680';

// Simulated optimal market conditions for a LONG setup matching FuturesMarketData interface
const mockMarketData: FuturesMarketData = {
  symbol: 'NQZ5',
  timestamp: '2025-11-19T18:00:00.000Z',
  currentPrice: 20500.00,

  // Strong uptrend with pullback to value area
  candles: [
    { timestamp: '2025-11-19T17:50:00Z', open: 20480, high: 20495, low: 20475, close: 20490, volume: 1200 },
    { timestamp: '2025-11-19T17:55:00Z', open: 20490, high: 20510, low: 20488, close: 20505, volume: 1500 },
    { timestamp: '2025-11-19T18:00:00Z', open: 20505, high: 20515, low: 20500, close: 20500, volume: 1100 }, // pullback to POC
  ],

  // CVD showing strong uptrend
  cvd: {
    value: 2500,
    trend: 'up' as const,
    ohlc: {
      timestamp: '2025-11-19T18:00:00Z',
      open: 2200,
      high: 2600,
      low: 2150,
      close: 2500,
    },
  },

  // Strong buying pressure with absorption
  orderFlow: {
    buyAbsorption: 0.15,  // 15% selling being absorbed by buyers
    sellAbsorption: 0.05,
    buyExhaustion: 0.02,
    sellExhaustion: 0.20,  // Sellers exhausted
    bigPrints: [
      { price: 20500, size: 50, side: 'buy' as const, timestamp: '2025-11-19T18:00:00Z' },
    ],
  },

  // Volume profile showing acceptance at 20500
  volumeProfile: {
    poc: 20500,        // Point of Control - high volume node
    vah: 20520,        // Value Area High
    val: 20480,        // Value Area Low
    lvns: [20510, 20490],  // Low volume nodes
    sessionHigh: 20530,
    sessionLow: 20470,
  },

  // Market structure
  marketState: 'out_of_balance_uptrend' as const,
  orderFlowConfirmed: true,

  // Account info
  account: {
    balance: 38133.95,
    position: 0,
    unrealizedPnL: 0,
    realizedPnL: 0,
  },

  // POC cross stats (required for enhanced features)
  pocCrossStats: {
    totalCrosses: 5,
    successfulCrosses: 4,
    failedCrosses: 1,
    successRate: 0.80,
    avgPriceMovementOnSuccess: 15.5,
    avgTimeToReversal: 180,
  },

  // Market statistics
  marketStats: {
    avgBarRange: 8.5,
    avgVolume: 1200,
    priceStdDev: 12.3,
    cvdStdDev: 450,
  },

  // No performance data yet (first trade)
  performance: null,

  // No historical notes yet
  historicalNotes: [],
};

async function testReasonerWithSimulation() {
  console.log('================================================================================');
  console.log('🧪 TESTING DEEPSEEK REASONER WITH SIMULATED OPTIMAL DATA');
  console.log('================================================================================\n');

  console.log('📊 Simulated Market Conditions:');
  console.log(`Price: ${mockMarketData.currentPrice}`);
  console.log(`POC: ${mockMarketData.volumeProfile.poc} | VAH: ${mockMarketData.volumeProfile.vah} | VAL: ${mockMarketData.volumeProfile.val}`);
  console.log(`CVD: ${mockMarketData.cvd.value} (${mockMarketData.cvd.trend})`);
  console.log(`Market State: ${mockMarketData.marketState}`);
  console.log(`Order Flow Confirmed: ${mockMarketData.orderFlowConfirmed}`);
  console.log(`Buy Absorption: ${(mockMarketData.orderFlow.buyAbsorption * 100).toFixed(1)}%`);
  console.log(`Sell Exhaustion: ${(mockMarketData.orderFlow.sellExhaustion * 100).toFixed(1)}%`);
  console.log('\n' + '='.repeat(80) + '\n');

  try {
    console.log('🤖 Calling DeepSeek Reasoner...\n');

    const decision = await analyzeFuturesMarket(mockMarketData);

    console.log('\n' + '='.repeat(80));
    console.log('✅ REASONER OUTPUT RECEIVED');
    console.log('='.repeat(80) + '\n');

    console.log('📋 Trading Decision:');
    console.log(JSON.stringify(decision, null, 2));

    console.log('\n' + '='.repeat(80));
    console.log('📊 DECISION SUMMARY');
    console.log('='.repeat(80));
    console.log(`Decision: ${decision.decision}`);
    console.log(`Confidence: ${decision.confidence}%`);

    if (decision.decision !== 'HOLD') {
      console.log(`Entry Price: ${decision.entryPrice}`);
      console.log(`Stop Loss: ${decision.stopLoss}`);
      console.log(`Target: ${decision.target}`);
      console.log(`Risk: $${decision.entryPrice && decision.stopLoss ? Math.abs(decision.entryPrice - decision.stopLoss) * 20 : 'N/A'}`);
      console.log(`Reward: $${decision.entryPrice && decision.target ? Math.abs(decision.target - decision.entryPrice) * 20 : 'N/A'}`);
    }

    console.log(`\nReasoning: ${decision.reasoning}`);

    if (decision.plan) {
      console.log('\n📝 Trading Plan:');
      console.log(decision.plan);
    }

    if (decision.timingPlan) {
      console.log('\n⏰ Timing Plan:');
      console.log(decision.timingPlan);
    }

    if (decision.reEntryPlan) {
      console.log('\n🔄 Re-Entry Plan:');
      console.log(decision.reEntryPlan);
    }

    console.log('\n' + '='.repeat(80));
    console.log('✅ TEST COMPLETE - DeepSeek Reasoner is working!');
    console.log('='.repeat(80));

  } catch (error) {
    console.error('\n❌ ERROR during test:');
    console.error(error);
    process.exit(1);
  }
}

// Run the test
testReasonerWithSimulation()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
