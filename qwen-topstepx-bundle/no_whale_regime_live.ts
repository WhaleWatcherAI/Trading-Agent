#!/usr/bin/env npx tsx
/**
 * No Whale Regime Live Trading with TopstepX
 *
 * Uses the same XGBoost + LSTM ensemble from no_whale_regime_backtest.py
 * but executes real trades via TopstepX API
 */

import 'dotenv/config';
import { getProjectXRest } from './projectx-rest';
import { authenticate, fetchTopstepXFuturesBars, fetchTopstepXAccounts } from './lib/topstepx';

// Configuration from .env
const ACCOUNT_ID = parseInt(process.env.TOPSTEPX_ACCOUNT_ID || '0');
const CONTRACT_ID = process.env.TOPSTEPX_CONTRACT_ID || 'CON.F.US.NQ.Z25';
const POINT_VALUE = CONTRACT_ID.includes('MNQ') || CONTRACT_ID.includes('MES') ? 2 : 20;

// Trading parameters (from backtest)
const STOP_LOSS_POINTS = 7;
const TAKE_PROFIT_POINTS = 35;
const MIN_CONFIDENCE = 0.65;
const MAX_POSITION_SIZE = 1;

interface Position {
  side: 'long' | 'short';
  entryPrice: number;
  entryTime: Date;
  stopLoss: number;
  takeProfit: number;
  contracts: number;
}

let currentPosition: Position | null = null;
let restClient: ReturnType<typeof getProjectXRest> | null = null;

async function getRestClient() {
  if (!restClient) {
    restClient = getProjectXRest();
  }
  return restClient;
}

/**
 * Fetch recent 1-minute bars for analysis
 */
async function fetchRecentBars(lookbackMinutes: number = 60) {
  const endTime = new Date();
  const startTime = new Date(endTime.getTime() - lookbackMinutes * 60 * 1000);

  const bars = await fetchTopstepXFuturesBars({
    contractId: CONTRACT_ID,
    startTime: startTime.toISOString(),
    endTime: endTime.toISOString(),
    unit: 2, // Minutes
    unitNumber: 1,
    limit: lookbackMinutes,
    live: false,
    includePartialBar: true,
  });

  return bars;
}

/**
 * Calculate simple technical features
 * In production, this would call the Python ML model
 */
function calculateFeatures(bars: any[]) {
  if (bars.length < 20) return null;

  const closes = bars.map(b => b.close);
  const highs = bars.map(b => b.high);
  const lows = bars.map(b => b.low);
  const volumes = bars.map(b => b.volume || 0);

  // EMA calculations
  const ema9 = calculateEMA(closes, 9);
  const ema21 = calculateEMA(closes, 21);

  // RSI
  const rsi = calculateRSI(closes, 14);

  // VWAP approximation
  const vwap = calculateVWAP(closes, volumes);

  // ATR for volatility
  const atr = calculateATR(highs, lows, closes, 14);

  const currentPrice = closes[closes.length - 1];
  const priceVsEma9 = (currentPrice - ema9) / atr;
  const priceVsEma21 = (currentPrice - ema21) / atr;
  const emaSpread = (ema9 - ema21) / atr;

  return {
    currentPrice,
    ema9,
    ema21,
    rsi,
    vwap,
    atr,
    priceVsEma9,
    priceVsEma21,
    emaSpread,
  };
}

function calculateEMA(data: number[], period: number): number {
  const multiplier = 2 / (period + 1);
  let ema = data[0];
  for (let i = 1; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
  }
  return ema;
}

function calculateRSI(closes: number[], period: number): number {
  if (closes.length < period + 1) return 50;

  let gains = 0, losses = 0;
  for (let i = closes.length - period; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    if (change > 0) gains += change;
    else losses -= change;
  }

  const avgGain = gains / period;
  const avgLoss = losses / period;
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

function calculateVWAP(closes: number[], volumes: number[]): number {
  let sumPV = 0, sumV = 0;
  for (let i = 0; i < closes.length; i++) {
    sumPV += closes[i] * (volumes[i] || 1);
    sumV += volumes[i] || 1;
  }
  return sumV > 0 ? sumPV / sumV : closes[closes.length - 1];
}

function calculateATR(highs: number[], lows: number[], closes: number[], period: number): number {
  if (highs.length < period + 1) return 1;

  const trs: number[] = [];
  for (let i = 1; i < highs.length; i++) {
    const tr = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1])
    );
    trs.push(tr);
  }

  const recentTRs = trs.slice(-period);
  return recentTRs.reduce((a, b) => a + b, 0) / recentTRs.length;
}

/**
 * Generate trading signal based on features
 * Simplified version - in production would use trained XGBoost model
 */
function generateSignal(features: any): { direction: 'long' | 'short' | null; confidence: number } {
  // Trend-following logic based on EMA crossover and momentum
  const { priceVsEma9, priceVsEma21, emaSpread, rsi } = features;

  let longScore = 0;
  let shortScore = 0;

  // EMA trend
  if (emaSpread > 0.5) longScore += 0.3;
  else if (emaSpread < -0.5) shortScore += 0.3;

  // Price momentum
  if (priceVsEma9 > 0.3 && priceVsEma21 > 0) longScore += 0.25;
  else if (priceVsEma9 < -0.3 && priceVsEma21 < 0) shortScore += 0.25;

  // RSI confirmation
  if (rsi > 55 && rsi < 75) longScore += 0.2;
  else if (rsi < 45 && rsi > 25) shortScore += 0.2;

  // Mean reversion at extremes
  if (rsi > 75 && priceVsEma9 > 1.5) shortScore += 0.15;
  else if (rsi < 25 && priceVsEma9 < -1.5) longScore += 0.15;

  const confidence = Math.max(longScore, shortScore);

  if (longScore > shortScore && confidence >= MIN_CONFIDENCE) {
    return { direction: 'long', confidence };
  } else if (shortScore > longScore && confidence >= MIN_CONFIDENCE) {
    return { direction: 'short', confidence };
  }

  return { direction: null, confidence: 0 };
}

/**
 * Place a market order with stop loss and take profit
 */
async function placeOrder(side: 'buy' | 'sell', price: number): Promise<boolean> {
  const rest = await getRestClient();

  const stopLoss = side === 'buy'
    ? price - STOP_LOSS_POINTS
    : price + STOP_LOSS_POINTS;

  const takeProfit = side === 'buy'
    ? price + TAKE_PROFIT_POINTS
    : price - TAKE_PROFIT_POINTS;

  console.log(`\n🚀 PLACING ${side.toUpperCase()} ORDER`);
  console.log(`   Contract: ${CONTRACT_ID}`);
  console.log(`   Price: ${price}`);
  console.log(`   Stop Loss: ${stopLoss}`);
  console.log(`   Take Profit: ${takeProfit}`);

  try {
    // Place market order
    const orderResult = await rest.placeOrder({
      accountId: ACCOUNT_ID,
      contractId: CONTRACT_ID,
      side: side === 'buy' ? 0 : 1,
      size: MAX_POSITION_SIZE,
      type: 2, // Market
      timeInForce: 1, // GTC
    });

    if (!orderResult.success) {
      console.error(`❌ Order failed:`, orderResult);
      return false;
    }

    console.log(`✅ Order placed successfully: ${orderResult.orderId}`);

    // Place stop loss order
    const stopSide = side === 'buy' ? 1 : 0; // Opposite side
    const stopResult = await rest.placeOrder({
      accountId: ACCOUNT_ID,
      contractId: CONTRACT_ID,
      side: stopSide,
      size: MAX_POSITION_SIZE,
      type: 4, // Stop
      timeInForce: 1,
      stopPrice: stopLoss,
    });

    console.log(`   Stop Loss order: ${stopResult.success ? '✅' : '❌'}`);

    // Place take profit order
    const tpResult = await rest.placeOrder({
      accountId: ACCOUNT_ID,
      contractId: CONTRACT_ID,
      side: stopSide,
      size: MAX_POSITION_SIZE,
      type: 1, // Limit
      timeInForce: 1,
      limitPrice: takeProfit,
    });

    console.log(`   Take Profit order: ${tpResult.success ? '✅' : '❌'}`);

    // Update position tracking
    currentPosition = {
      side: side === 'buy' ? 'long' : 'short',
      entryPrice: price,
      entryTime: new Date(),
      stopLoss,
      takeProfit,
      contracts: MAX_POSITION_SIZE,
    };

    return true;
  } catch (error: any) {
    console.error(`❌ Order error:`, error.message);
    return false;
  }
}

/**
 * Check if we have an open position
 */
async function checkPosition(): Promise<boolean> {
  try {
    const rest = await getRestClient();
    const positions = await rest.getPositions(ACCOUNT_ID);

    if (positions && positions.length > 0) {
      const pos = positions.find((p: any) => p.contractId === CONTRACT_ID);
      if (pos && Math.abs(pos.size || pos.netSize || 0) > 0) {
        return true;
      }
    }

    currentPosition = null;
    return false;
  } catch (error) {
    console.error('Error checking position:', error);
    return currentPosition !== null;
  }
}

/**
 * Main trading loop
 */
async function tradingLoop() {
  console.log('\n📊 Analyzing market...');

  // Fetch recent bars
  const bars = await fetchRecentBars(60);
  if (!bars || bars.length < 20) {
    console.log('   Insufficient bar data');
    return;
  }

  // Calculate features
  const features = calculateFeatures(bars);
  if (!features) {
    console.log('   Could not calculate features');
    return;
  }

  console.log(`   Price: ${features.currentPrice.toFixed(2)}`);
  console.log(`   EMA9: ${features.ema9.toFixed(2)} | EMA21: ${features.ema21.toFixed(2)}`);
  console.log(`   RSI: ${features.rsi.toFixed(1)} | ATR: ${features.atr.toFixed(2)}`);

  // Check if we have a position
  const hasPosition = await checkPosition();

  if (hasPosition) {
    console.log(`   📍 Position open: ${currentPosition?.side} @ ${currentPosition?.entryPrice}`);
    return;
  }

  // Generate signal
  const signal = generateSignal(features);

  if (signal.direction) {
    console.log(`\n🎯 SIGNAL: ${signal.direction.toUpperCase()} (${(signal.confidence * 100).toFixed(0)}% confidence)`);

    if (signal.confidence >= MIN_CONFIDENCE) {
      await placeOrder(signal.direction === 'long' ? 'buy' : 'sell', features.currentPrice);
    }
  } else {
    console.log('   No trade signal');
  }
}

/**
 * Main entry point
 */
async function main() {
  console.log('\n═══════════════════════════════════════════════════════');
  console.log('       No Whale Regime Live Trading - TopstepX');
  console.log('═══════════════════════════════════════════════════════');

  // Validate configuration
  if (!ACCOUNT_ID) {
    console.error('❌ TOPSTEPX_ACCOUNT_ID not set in .env');
    process.exit(1);
  }

  console.log(`\n📋 Configuration:`);
  console.log(`   Account ID: ${ACCOUNT_ID}`);
  console.log(`   Contract: ${CONTRACT_ID}`);
  console.log(`   Point Value: $${POINT_VALUE}`);
  console.log(`   Stop Loss: ${STOP_LOSS_POINTS} pts ($${STOP_LOSS_POINTS * POINT_VALUE})`);
  console.log(`   Take Profit: ${TAKE_PROFIT_POINTS} pts ($${TAKE_PROFIT_POINTS * POINT_VALUE})`);
  console.log(`   Min Confidence: ${(MIN_CONFIDENCE * 100).toFixed(0)}%`);

  // Authenticate
  console.log('\n🔐 Authenticating with TopstepX...');
  try {
    await authenticate();
    console.log('   ✅ Authenticated successfully');
  } catch (error: any) {
    console.error(`   ❌ Authentication failed: ${error.message}`);
    process.exit(1);
  }

  // Verify account
  console.log('\n📊 Fetching account info...');
  try {
    const accounts = await fetchTopstepXAccounts(true);
    const account = accounts.find(a => a.id === ACCOUNT_ID);
    if (account) {
      console.log(`   ✅ Account: ${account.name}`);
      console.log(`   Balance: $${account.balance.toFixed(2)}`);
      console.log(`   Can Trade: ${account.canTrade}`);
    } else {
      console.error(`   ❌ Account ${ACCOUNT_ID} not found`);
      process.exit(1);
    }
  } catch (error: any) {
    console.error(`   ❌ Failed to fetch accounts: ${error.message}`);
    process.exit(1);
  }

  // Run trading loop
  console.log('\n🚀 Starting trading loop (30 second intervals)...');
  console.log('   Press Ctrl+C to stop\n');

  // Initial run
  await tradingLoop();

  // Schedule recurring runs
  const interval = setInterval(async () => {
    try {
      await tradingLoop();
    } catch (error: any) {
      console.error(`\n❌ Trading loop error: ${error.message}`);
    }
  }, 30000); // 30 seconds

  // Handle graceful shutdown
  process.on('SIGINT', () => {
    console.log('\n\n🛑 Shutting down...');
    clearInterval(interval);
    process.exit(0);
  });
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
