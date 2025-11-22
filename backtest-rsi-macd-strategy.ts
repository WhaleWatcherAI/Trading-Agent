#!/usr/bin/env tsx
/**
 * RSI Filter + MACD Cross Strategy Backtest - SHORT ONLY
 *
 * Strategy:
 * Step 1 (Filter): RSI + Bearish Candlestick Pattern creates a setup
 *   - SHORT setup: RSI > 65 (overbought) AND bearish candlestick pattern
 *   - Bearish patterns: Bearish Engulfing, Shooting Star, Evening Star, Strong Bearish Candle,
 *     Two Consecutive Red Candles, Close Below Previous Low, Lower High & Lower Low
 *
 * Step 2 (Entry): MACD cross triggers entry (no time limit after Step 1)
 *   - SHORT entry: MACD line crosses below signal line while short setup is active
 *
 * Exit:
 *   - Stop Loss: 16 ticks from entry
 *   - Take Profit: 32 ticks from entry
 */

import { RSI, MACD } from 'technicalindicators';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

interface BacktestConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;
  rsiPeriod: number;
  rsiOversold: number;
  rsiOverbought: number;
  macdFastPeriod: number;
  macdSlowPeriod: number;
  macdSignalPeriod: number;
  stopLossTicks: number;
  takeProfitTicks: number;
  numberOfContracts: number;
  commissionPerSide: number;
  slippageTicks: number;
}

interface TradeRecord {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  grossPnl: number;
  fees: number;
  exitReason: 'stop' | 'target' | 'end_of_data';
  entryRSI: number;
  entryMACD: number;
  entrySlippageTicks?: number;
  exitSlippageTicks?: number;
  slippageCost?: number;
}

interface PendingSetup {
  side: 'long' | 'short';
  setupTime: string;
  rsi: number;
  hasBearishPattern?: boolean;
}

// Candlestick Pattern Detection
function isBearishEngulfing(current: TopstepXFuturesBar, previous: TopstepXFuturesBar): boolean {
  // Current bar is bearish, completely engulfs previous bullish bar
  return (
    current.close < current.open && // Current is bearish
    previous.close > previous.open && // Previous is bullish
    current.open >= previous.close &&
    current.close <= previous.open
  );
}

function isShootingStar(current: TopstepXFuturesBar): boolean {
  const bodySize = Math.abs(current.close - current.open);
  const upperShadow = current.high - Math.max(current.open, current.close);
  const lowerShadow = Math.min(current.open, current.close) - current.low;

  // Shooting star: small body near bottom, long upper shadow
  return (
    current.close < current.open && // Bearish
    upperShadow > bodySize * 2 && // Upper shadow at least 2x body
    lowerShadow < bodySize * 0.3 // Small lower shadow
  );
}

function isEveningStar(current: TopstepXFuturesBar, previous: TopstepXFuturesBar, twoBarsAgo: TopstepXFuturesBar): boolean {
  // Three-candle pattern: bullish -> small body -> bearish
  const prevBodySize = Math.abs(previous.close - previous.open);
  const currentBodySize = Math.abs(current.close - current.open);
  const twoBarsBodySize = Math.abs(twoBarsAgo.close - twoBarsAgo.open);

  return (
    twoBarsAgo.close > twoBarsAgo.open && // First candle bullish
    prevBodySize < twoBarsBodySize * 0.3 && // Middle candle small
    current.close < current.open && // Last candle bearish
    current.close < twoBarsAgo.open // Bearish candle closes below first candle's open
  );
}

function isBearishPattern(bars: TopstepXFuturesBar[], currentIndex: number): boolean {
  if (currentIndex < 2) return false;

  const current = bars[currentIndex];
  const previous = bars[currentIndex - 1];
  const twoBarsAgo = bars[currentIndex - 2];

  // Original strict bearish pattern detection
  return (
    isBearishEngulfing(current, previous) ||
    isShootingStar(current) ||
    isEveningStar(current, previous, twoBarsAgo) ||
    (current.close < current.open && current.open - current.close > (current.high - current.low) * 0.6) // Strong bearish candle
  );
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DEFAULT_DAYS = 365;

const DEFAULT_SYMBOL = process.env.TOPSTEPX_MR_SYMBOL || 'NQZ5';
const DEFAULT_CONTRACT_ID = process.env.TOPSTEPX_MR_CONTRACT_ID;

// Load slippage configuration
interface SlippageConfig {
  tickSize: Record<string, number>;
  slipAvg: {
    entry: Record<string, number>;
    tp: Record<string, number>;
    stop: Record<string, number>;
  };
  avgSpreadTicks: Record<string, number>;
  feesPerSideUSD: Record<string, number>;
  p_tp_passive: Record<string, number>;
}

const loadSlippageConfig = (): SlippageConfig => {
  const configPath = path.join(__dirname, 'slip-config.json');
  const configData = fs.readFileSync(configPath, 'utf-8');
  return JSON.parse(configData);
};

const SLIP_CONFIG = loadSlippageConfig();

function fillEntry(sym: string, side: 'buy' | 'sell', mid: number): number {
  const t = SLIP_CONFIG.tickSize[sym];
  const S = 0.5 * t * SLIP_CONFIG.avgSpreadTicks[sym];
  const sigma = SLIP_CONFIG.slipAvg.entry[sym] * t;
  return side === 'buy' ? mid + S + sigma : mid - S - sigma;
}

function fillTP(sym: string, side: 'buy' | 'sell', mid: number): number {
  const t = SLIP_CONFIG.tickSize[sym];
  const S_ticks = SLIP_CONFIG.avgSpreadTicks[sym];
  const sigma_tp_ticks = SLIP_CONFIG.slipAvg.tp[sym];
  const p_passive = SLIP_CONFIG.p_tp_passive[sym];

  const E_tp_ticks = (1 - p_passive) * (S_ticks + sigma_tp_ticks);
  return side === 'sell' ? mid - E_tp_ticks * t : mid + E_tp_ticks * t;
}

function fillStop(sym: string, side: 'buy' | 'sell', triggerMid: number): number {
  const t = SLIP_CONFIG.tickSize[sym];
  const sigma = SLIP_CONFIG.slipAvg.stop[sym] * t;
  return side === 'buy' ? triggerMid + sigma : triggerMid - sigma;
}

function addFees(sym: string, contracts: number): number {
  return SLIP_CONFIG.feesPerSideUSD[sym] * contracts;
}

const getBaseSymbol = (fullSymbol: string): string => {
  return fullSymbol.replace(/[A-Z]\d+$/, '');
};

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_SYMBOL,
  contractId: DEFAULT_CONTRACT_ID,
  start: process.env.TOPSTEPX_MR_START || new Date(Date.now() - DEFAULT_DAYS * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.TOPSTEPX_MR_END || new Date().toISOString(),
  rsiPeriod: Number(process.env.RSI_PERIOD || '14'),
  rsiOversold: Number(process.env.RSI_OVERSOLD || '30'),
  rsiOverbought: Number(process.env.RSI_OVERBOUGHT || '65'),  // Lowered from 70 to 65 for more setups
  macdFastPeriod: Number(process.env.MACD_FAST || '12'),
  macdSlowPeriod: Number(process.env.MACD_SLOW || '26'),
  macdSignalPeriod: Number(process.env.MACD_SIGNAL || '9'),
  stopLossTicks: Number(process.env.STOP_LOSS_TICKS || '16'),
  takeProfitTicks: Number(process.env.TAKE_PROFIT_TICKS || '32'),
  numberOfContracts: Number(process.env.NUM_CONTRACTS || '1'),
  commissionPerSide: process.env.COMMISSION
    ? Number(process.env.COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_CONTRACT_ID, DEFAULT_SYMBOL], 0.35),
  slippageTicks: Number(process.env.SLIPPAGE_TICKS || '1'),
};

function toCentralTime(date: Date): Date {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isTradingAllowed(timestamp: string | Date): boolean {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const ctDate = toCentralTime(date);
  const day = ctDate.getUTCDay();
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  if (day === 6) return false;
  if (day === 0 && minutes < WEEKEND_REOPEN_MINUTES) return false;
  if (day === 5 && minutes >= CUT_OFF_MINUTES) return false;

  return minutes < CUT_OFF_MINUTES || minutes >= REOPEN_MINUTES;
}

function formatCurrency(value: number): string {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

async function runBacktest() {
  console.log('='.repeat(80));
  console.log('RSI FILTER + MACD CROSS STRATEGY BACKTEST');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start.split('T')[0]} to ${CONFIG.end.split('T')[0]}`);
  console.log(`\nStrategy Parameters:`);
  console.log(`  RSI Period: ${CONFIG.rsiPeriod}`);
  console.log(`  RSI Oversold: ${CONFIG.rsiOversold}`);
  console.log(`  RSI Overbought: ${CONFIG.rsiOverbought}`);
  console.log(`  MACD Fast: ${CONFIG.macdFastPeriod}, Slow: ${CONFIG.macdSlowPeriod}, Signal: ${CONFIG.macdSignalPeriod}`);
  console.log(`  Stop Loss: ${CONFIG.stopLossTicks} ticks`);
  console.log(`  Take Profit: ${CONFIG.takeProfitTicks} ticks`);
  console.log(`  Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`  Commission: $${CONFIG.commissionPerSide.toFixed(2)} per side`);
  console.log('='.repeat(80));

  // Fetch metadata
  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[rsi-macd] Unable to fetch metadata:', err.message);
    return null;
  });

  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  const contractId = metadata.id;
  const tickSize = metadata.tickSize || 0.25;
  const contractMultiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || 20;

  console.log(`Contract ID: ${contractId}`);
  console.log(`Tick Size: ${tickSize}`);
  console.log(`Contract Multiplier: ${contractMultiplier}`);

  // Fetch historical bars
  console.log(`\nFetching historical bars...`);
  const bars = await fetchTopstepXFuturesBars({
    contractId,
    startTime: CONFIG.start,
    endTime: CONFIG.end,
    unit: 2,           // Minutes
    unitNumber: 1,     // 1-minute bars
    limit: 20000,
    live: false,
  });

  if (!bars || bars.length === 0) {
    console.error('No bars retrieved. Exiting.');
    return;
  }

  console.log(`Fetched ${bars.length} bars`);

  const baseSymbol = getBaseSymbol(CONFIG.symbol);
  const trades: TradeRecord[] = [];
  const closePrices: number[] = [];

  let activePosition: {
    side: 'long' | 'short';
    entryPrice: number;
    entryTime: string;
    stopLoss: number;
    target: number;
    entryRSI: number;
    entryMACD: number;
  } | null = null;

  let pendingSetup: PendingSetup | null = null;
  let prevMacd: number | null = null;
  let prevSignal: number | null = null;

  // Main backtest loop
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    closePrices.push(bar.close);

    // Skip non-trading hours
    if (!isTradingAllowed(bar.timestamp)) {
      continue;
    }

    // Calculate RSI
    let currentRSI: number | undefined;
    if (closePrices.length >= CONFIG.rsiPeriod + 1) {
      const rsiResult = RSI.calculate({
        values: closePrices,
        period: CONFIG.rsiPeriod,
      });
      currentRSI = rsiResult[rsiResult.length - 1];
    }

    // Calculate MACD
    let currentMACD: number | null = null;
    let currentSignal: number | null = null;
    if (closePrices.length >= CONFIG.macdSlowPeriod + CONFIG.macdSignalPeriod) {
      const macdResult = MACD.calculate({
        values: closePrices,
        fastPeriod: CONFIG.macdFastPeriod,
        slowPeriod: CONFIG.macdSlowPeriod,
        signalPeriod: CONFIG.macdSignalPeriod,
        SimpleMAOscillator: false,
        SimpleMASignal: false,
      });

      if (macdResult.length > 0) {
        const latest = macdResult[macdResult.length - 1];
        currentMACD = latest.MACD ?? null;
        currentSignal = latest.signal ?? null;
      }
    }

    // Check if we have an active position
    if (activePosition) {
      // Check stop loss and take profit on every bar
      const hitStop = activePosition.side === 'long'
        ? bar.low <= activePosition.stopLoss
        : bar.high >= activePosition.stopLoss;

      const hitTarget = activePosition.side === 'long'
        ? bar.high >= activePosition.target
        : bar.low <= activePosition.target;

      if (hitStop) {
        // Stop loss hit
        const exitSide = activePosition.side === 'long' ? 'sell' : 'buy';
        const exitPrice = fillStop(baseSymbol, exitSide, activePosition.stopLoss);
        const pnlPerContract = activePosition.side === 'long'
          ? (exitPrice - activePosition.entryPrice) * contractMultiplier
          : (activePosition.entryPrice - exitPrice) * contractMultiplier;

        const grossPnl = pnlPerContract * CONFIG.numberOfContracts;
        const fees = addFees(baseSymbol, CONFIG.numberOfContracts) * 2; // Entry + Exit
        const netPnl = grossPnl - fees;

        trades.push({
          entryTime: activePosition.entryTime,
          exitTime: bar.timestamp,
          side: activePosition.side,
          entryPrice: activePosition.entryPrice,
          exitPrice,
          pnl: netPnl,
          grossPnl,
          fees,
          exitReason: 'stop',
          entryRSI: activePosition.entryRSI,
          entryMACD: activePosition.entryMACD,
        });

        activePosition = null;
        continue;
      }

      if (hitTarget) {
        // Take profit hit
        const exitSide = activePosition.side === 'long' ? 'sell' : 'buy';
        const exitPrice = fillTP(baseSymbol, exitSide, activePosition.target);
        const pnlPerContract = activePosition.side === 'long'
          ? (exitPrice - activePosition.entryPrice) * contractMultiplier
          : (activePosition.entryPrice - exitPrice) * contractMultiplier;

        const grossPnl = pnlPerContract * CONFIG.numberOfContracts;
        const fees = addFees(baseSymbol, CONFIG.numberOfContracts) * 2; // Entry + Exit
        const netPnl = grossPnl - fees;

        trades.push({
          entryTime: activePosition.entryTime,
          exitTime: bar.timestamp,
          side: activePosition.side,
          entryPrice: activePosition.entryPrice,
          exitPrice,
          pnl: netPnl,
          grossPnl,
          fees,
          exitReason: 'target',
          entryRSI: activePosition.entryRSI,
          entryMACD: activePosition.entryMACD,
        });

        activePosition = null;
        continue;
      }
    }

    // Check for bearish candlestick pattern
    const hasBearishPattern = isBearishPattern(bars, i);

    // If we have indicators, check for setups and entries
    if (currentRSI !== undefined && currentMACD !== null && currentSignal !== null) {

      // STEP 1: Check for SHORT setup (RSI > 70 + Bearish Pattern)
      if (!activePosition && !pendingSetup) {
        if (currentRSI > CONFIG.rsiOverbought && hasBearishPattern) {
          pendingSetup = {
            side: 'short',
            setupTime: bar.timestamp,
            rsi: currentRSI,
            hasBearishPattern: true,
          };
          console.log(`[${bar.timestamp}] SHORT SETUP: RSI=${currentRSI.toFixed(2)} (> ${CONFIG.rsiOverbought}) + Bearish Pattern`);
        }
      }

      // STEP 2: Check for MACD cross entry (SHORT only)
      if (!activePosition && pendingSetup && prevMacd !== null && prevSignal !== null) {

        // SHORT entry: MACD crosses below signal
        if (pendingSetup.side === 'short' && prevMacd >= prevSignal && currentMACD < currentSignal) {
          const entrySide = 'sell';
          const entryPrice = fillEntry(baseSymbol, entrySide, bar.close);
          const stopLoss = entryPrice + (CONFIG.stopLossTicks * tickSize);
          const target = entryPrice - (CONFIG.takeProfitTicks * tickSize);

          activePosition = {
            side: 'short',
            entryPrice,
            entryTime: bar.timestamp,
            stopLoss,
            target,
            entryRSI: currentRSI,
            entryMACD: currentMACD,
          };

          console.log(`[${bar.timestamp}] SHORT ENTRY: Price=${entryPrice.toFixed(2)}, SL=${stopLoss.toFixed(2)}, TP=${target.toFixed(2)}, RSI=${currentRSI.toFixed(2)}, MACD crossed below signal`);
          pendingSetup = null;
        }
      }

      // Store previous MACD values for cross detection
      prevMacd = currentMACD;
      prevSignal = currentSignal;
    }
  }

  // Close any remaining position at end of data
  if (activePosition) {
    const lastBar = bars[bars.length - 1];
    const exitSide = activePosition.side === 'long' ? 'sell' : 'buy';
    const exitPrice = fillEntry(baseSymbol, exitSide, lastBar.close);
    const pnlPerContract = activePosition.side === 'long'
      ? (exitPrice - activePosition.entryPrice) * contractMultiplier
      : (activePosition.entryPrice - exitPrice) * contractMultiplier;

    const grossPnl = pnlPerContract * CONFIG.numberOfContracts;
    const fees = addFees(baseSymbol, CONFIG.numberOfContracts) * 2;
    const netPnl = grossPnl - fees;

    trades.push({
      entryTime: activePosition.entryTime,
      exitTime: lastBar.timestamp,
      side: activePosition.side,
      entryPrice: activePosition.entryPrice,
      exitPrice,
      pnl: netPnl,
      grossPnl,
      fees,
      exitReason: 'end_of_data',
      entryRSI: activePosition.entryRSI,
      entryMACD: activePosition.entryMACD,
    });
  }

  // Calculate statistics
  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST RESULTS');
  console.log('='.repeat(80));

  if (trades.length === 0) {
    console.log('No trades executed.');
    return;
  }

  const totalTrades = trades.length;
  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl < 0);
  const winRate = (winningTrades.length / totalTrades) * 100;

  const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0);
  const totalGrossPnL = trades.reduce((sum, t) => sum + t.grossPnl, 0);
  const totalFees = trades.reduce((sum, t) => sum + t.fees, 0);

  const avgWin = winningTrades.length > 0
    ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length
    : 0;
  const avgLoss = losingTrades.length > 0
    ? losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length
    : 0;

  const profitFactor = losingTrades.length > 0
    ? Math.abs(winningTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.reduce((sum, t) => sum + t.pnl, 0))
    : Infinity;

  let maxDrawdown = 0;
  let peak = 0;
  let runningPnL = 0;
  for (const trade of trades) {
    runningPnL += trade.pnl;
    if (runningPnL > peak) peak = runningPnL;
    const drawdown = peak - runningPnL;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  }

  console.log(`\nTotal Trades: ${totalTrades}`);
  console.log(`Winning Trades: ${winningTrades.length}`);
  console.log(`Losing Trades: ${losingTrades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(2)}%`);
  console.log(`\nP&L Summary:`);
  console.log(`  Total Net P&L: ${formatCurrency(totalPnL)}`);
  console.log(`  Total Gross P&L: ${formatCurrency(totalGrossPnL)}`);
  console.log(`  Total Fees: ${formatCurrency(totalFees)}`);
  console.log(`  Average Win: ${formatCurrency(avgWin)}`);
  console.log(`  Average Loss: ${formatCurrency(avgLoss)}`);
  console.log(`  Profit Factor: ${profitFactor === Infinity ? 'N/A' : profitFactor.toFixed(2)}`);
  console.log(`  Max Drawdown: ${formatCurrency(maxDrawdown)}`);

  // Trade breakdown by exit reason
  const stopLosses = trades.filter(t => t.exitReason === 'stop').length;
  const takeProfits = trades.filter(t => t.exitReason === 'target').length;
  console.log(`\nExit Reasons:`);
  console.log(`  Stop Loss: ${stopLosses}`);
  console.log(`  Take Profit: ${takeProfits}`);
  console.log(`  End of Data: ${trades.filter(t => t.exitReason === 'end_of_data').length}`);

  // Long vs Short performance
  const longTrades = trades.filter(t => t.side === 'long');
  const shortTrades = trades.filter(t => t.side === 'short');
  const longPnL = longTrades.reduce((sum, t) => sum + t.pnl, 0);
  const shortPnL = shortTrades.reduce((sum, t) => sum + t.pnl, 0);
  const longWinRate = longTrades.length > 0 ? (longTrades.filter(t => t.pnl > 0).length / longTrades.length) * 100 : 0;
  const shortWinRate = shortTrades.length > 0 ? (shortTrades.filter(t => t.pnl > 0).length / shortTrades.length) * 100 : 0;

  console.log(`\nLong vs Short:`);
  console.log(`  Long Trades: ${longTrades.length} (Win Rate: ${longWinRate.toFixed(2)}%, P&L: ${formatCurrency(longPnL)})`);
  console.log(`  Short Trades: ${shortTrades.length} (Win Rate: ${shortWinRate.toFixed(2)}%, P&L: ${formatCurrency(shortPnL)})`);

  // Show first 10 and last 10 trades
  console.log(`\n${'='.repeat(80)}`);
  console.log('TRADE LOG (First 10)');
  console.log('='.repeat(80));
  trades.slice(0, 10).forEach((trade, idx) => {
    console.log(`${idx + 1}. [${trade.entryTime}] ${trade.side.toUpperCase()} @ ${trade.entryPrice.toFixed(2)} -> ${trade.exitPrice.toFixed(2)} | ${formatCurrency(trade.pnl)} | ${trade.exitReason} | RSI: ${trade.entryRSI.toFixed(2)}`);
  });

  if (trades.length > 10) {
    console.log(`\n... (${trades.length - 20} trades omitted) ...\n`);
    console.log('TRADE LOG (Last 10)');
    console.log('='.repeat(80));
    trades.slice(-10).forEach((trade, idx) => {
      console.log(`${trades.length - 10 + idx + 1}. [${trade.entryTime}] ${trade.side.toUpperCase()} @ ${trade.entryPrice.toFixed(2)} -> ${trade.exitPrice.toFixed(2)} | ${formatCurrency(trade.pnl)} | ${trade.exitReason} | RSI: ${trade.entryRSI.toFixed(2)}`);
    });
  }

  console.log('\n' + '='.repeat(80));
}

runBacktest().catch(console.error);
