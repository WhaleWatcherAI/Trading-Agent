#!/usr/bin/env tsx
/**
 * TopstepX NQ Trender Backtest on 1-Minute Bars
 *
 * TRENDING STRATEGY - Complementary to NQ Winner Mean Reversion
 *
 * Strategy:
 * - Bollinger Bands: 20-period SMA with 3.5 standard deviations (wider than mean reversion)
 * - RSI confirmation: RSI(24) with 50 threshold (trend direction)
 * - Entry triggers:
 *   LONG: Price >= upper band + RSI > 50 + TTM momentum cycle
 *   SHORT: Price <= lower band + RSI < 50 + TTM momentum cycle
 * - TTM Squeeze Momentum Cycle Filter:
 *   LONG: Wait for momentum to go red (if green), then green again
 *   SHORT: Wait for momentum to go green (if red), then red again
 * - Exit Strategy: RSI-based exits (not middle band like mean reversion)
 *   LONG exit: RSI <= 30
 *   SHORT exit: RSI >= 70
 * - Stop Loss: 0.01% from entry (tight stop)
 * - Take Profit: 0.1% (wider target for trend continuation)
 */

import { RSI, ADX, ATR } from 'technicalindicators';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';
import { calculateTtmSqueeze } from './lib/ttmSqueeze';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

interface BacktestConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;
  bbPeriod: number;
  bbStdDev: number;
  rsiPeriod: number;
  rsiEntryLong: number;   // RSI threshold for long entries (e.g., > 70)
  rsiEntryShort: number;  // RSI threshold for short entries (e.g., < 30)
  rsiExitLow: number;
  rsiExitHigh: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  contractMultiplier?: number;
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
  exitReason: 'stop' | 'target' | 'session' | 'end_of_data';
  entryRSI: number;
  exitRSI?: number;
  entrySlippageTicks?: number;
  exitSlippageTicks?: number;
  slippageCost?: number;
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DEFAULT_DAYS = 365;

const DEFAULT_SYMBOL = process.env.TOPSTEPX_TRENDER_SYMBOL || 'NQZ5';
const DEFAULT_CONTRACT_ID = process.env.TOPSTEPX_TRENDER_CONTRACT_ID;

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
  start: process.env.TOPSTEPX_TRENDER_START || new Date(Date.now() - DEFAULT_DAYS * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.TOPSTEPX_TRENDER_END || new Date().toISOString(),
  bbPeriod: Number(process.env.TOPSTEPX_TRENDER_BB_PERIOD || '20'),
  bbStdDev: Number(process.env.TOPSTEPX_TRENDER_BB_STDDEV || '3.5'), // Wider bands
  rsiPeriod: Number(process.env.TOPSTEPX_TRENDER_RSI_PERIOD || '24'),
  rsiEntryLong: Number(process.env.TOPSTEPX_TRENDER_RSI_ENTRY_LONG || '70'),  // Strong uptrend
  rsiEntryShort: Number(process.env.TOPSTEPX_TRENDER_RSI_ENTRY_SHORT || '30'), // Strong downtrend
  rsiExitLow: Number(process.env.TOPSTEPX_TRENDER_RSI_EXIT_LOW || '30'),
  rsiExitHigh: Number(process.env.TOPSTEPX_TRENDER_RSI_EXIT_HIGH || '70'),
  stopLossPercent: Number(process.env.TOPSTEPX_TRENDER_STOP_PERCENT || '0.0001'),
  takeProfitPercent: Number(process.env.TOPSTEPX_TRENDER_TP_PERCENT || '0.001'), // Wider target
  numberOfContracts: Number(process.env.TOPSTEPX_TRENDER_CONTRACTS || '3'),
  commissionPerSide: process.env.TOPSTEPX_TRENDER_COMMISSION
    ? Number(process.env.TOPSTEPX_TRENDER_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_CONTRACT_ID, DEFAULT_SYMBOL], 1.40),
  slippageTicks: Number(process.env.TOPSTEPX_TRENDER_SLIPPAGE_TICKS || '1'),
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

function calculateBollingerBands(
  values: number[],
  period: number,
  stdDev: number,
): { upper: number; middle: number; lower: number } | null {
  if (values.length < period) return null;

  const slice = values.slice(-period);
  const sum = slice.reduce((acc, val) => acc + val, 0);
  const mean = sum / period;

  const squaredDiffs = slice.map(val => Math.pow(val - mean, 2));
  const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / period;
  const standardDeviation = Math.sqrt(variance);

  return {
    upper: mean + standardDeviation * stdDev,
    middle: mean,
    lower: mean - standardDeviation * stdDev,
  };
}

function formatCurrency(value: number): string {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('TOPSTEPX NQ TRENDER BACKTEST (1-MINUTE BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end}`);
  console.log(`BB Period: ${CONFIG.bbPeriod} bars | Std Dev: ${CONFIG.bbStdDev} (WIDER - trending)`);
  console.log(`RSI Period: ${CONFIG.rsiPeriod} | Entry: Long > ${CONFIG.rsiEntryLong}, Short < ${CONFIG.rsiEntryShort}`);
  console.log(`Exit Logic: Target/Stop (no TTM exits)`);
  console.log(`Stop Loss: ${(CONFIG.stopLossPercent * 100).toFixed(3)}% | Take Profit: ${(CONFIG.takeProfitPercent * 100).toFixed(3)}%`);
  console.log(`Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`Commission/side: ${CONFIG.commissionPerSide.toFixed(2)} USD | Slippage: ${CONFIG.slippageTicks} tick(s)`);
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[trender] Unable to fetch metadata:', err.message);
    return null;
  });

  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  const contractId = metadata.id;
  const multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || CONFIG.contractMultiplier || 20;
  const tickSize = metadata.tickSize || 0.25;

  const baseSymbol = getBaseSymbol(CONFIG.symbol);

  console.log(`Resolved contract: ${metadata.name} (${contractId})`);
  console.log(`Point multiplier: ${multiplier}`);
  console.log(`Tick size: ${tickSize}`);
  console.log(`Spread Model (${baseSymbol}): Avg spread ${SLIP_CONFIG.avgSpreadTicks[baseSymbol]} tick, Fees $${SLIP_CONFIG.feesPerSideUSD[baseSymbol]} per side`);
  console.log(`Slippage (${baseSymbol}): Entry=${SLIP_CONFIG.slipAvg.entry[baseSymbol]} ticks, TP=${SLIP_CONFIG.slipAvg.tp[baseSymbol]} ticks, Stop=${SLIP_CONFIG.slipAvg.stop[baseSymbol]} ticks`);

  console.log('\nFetching 1-minute bars...');
  const bars = await fetchTopstepXFuturesBars({
    contractId,
    startTime: CONFIG.start,
    endTime: CONFIG.end,
    unit: 2,
    unitNumber: 1,
    limit: 50000,
  });

  if (!bars.length) {
    throw new Error('No 1-minute bars returned for configured window.');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length} one-minute bars`);

  const roundToTick = (price: number): number => {
    return Math.round(price / tickSize) * tickSize;
  };

  const closes: number[] = [];
  const trades: TradeRecord[] = [];

  interface MomentumSetup {
    side: 'long' | 'short';
    setupTime: string;
    setupPrice: number;
    rsi: number;
    bb: { upper: number; middle: number; lower: number };
    momentumState: 'waiting_for_red' | 'waiting_for_green' | 'ready';
    lastMomentumSign: number;
  }

  let momentumSetup: MomentumSetup | null = null;

  let position: {
    side: 'long' | 'short';
    entryPrice: number;
    entryTime: string;
    entryRSI: number;
    stopLoss: number | null;
    target: number | null;
    remainingQty: number;
    feesPaid: number;
  } | null = null;

  let realizedPnL = 0;

  const exitPosition = (
    exitPrice: number,
    exitTime: string,
    exitRSI: number,
    reason: TradeRecord['exitReason'],
    exitSlippagePoints: number = 0,
  ) => {
    if (!position) return;

    const direction = position.side === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - position.entryPrice) * direction * multiplier * position.remainingQty;
    const exitFees = addFees(baseSymbol, position.remainingQty);
    const totalFees = position.feesPaid + exitFees;
    const grossPnl = rawPnL;
    const netPnl = grossPnl - totalFees;

    const entrySlippagePoints = (SLIP_CONFIG.slipAvg.entry[baseSymbol] + 0.5 * SLIP_CONFIG.avgSpreadTicks[baseSymbol]) * tickSize;
    const totalSlippagePoints = entrySlippagePoints + exitSlippagePoints;
    const slippageCost = totalSlippagePoints * multiplier * position.remainingQty;
    const entrySlippageTicks = entrySlippagePoints / tickSize;
    const exitSlippageTicks = exitSlippagePoints / tickSize;

    const trade: TradeRecord = {
      entryTime: position.entryTime,
      exitTime,
      side: position.side,
      entryPrice: position.entryPrice,
      exitPrice,
      pnl: netPnl,
      grossPnl,
      fees: totalFees,
      exitReason: reason,
      entryRSI: position.entryRSI,
      exitRSI: exitRSI,
      entrySlippageTicks,
      exitSlippageTicks,
      slippageCost,
    };

    trades.push(trade);
    realizedPnL += netPnl;
    position = null;
  };

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    closes.push(bar.close);

    if (!isTradingAllowed(bar.timestamp)) {
      if (position) {
        const rsiValues = RSI.calculate({ values: closes, period: CONFIG.rsiPeriod });
        const currentRSI = rsiValues[rsiValues.length - 1];
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const sessionExitPrice = roundToTick(fillStop(baseSymbol, closeSide, bar.close));
        const stopSlippagePoints = (SLIP_CONFIG.slipAvg.stop[baseSymbol] * tickSize);
        exitPosition(sessionExitPrice, bar.timestamp, currentRSI, 'session', stopSlippagePoints);
      }
      continue;
    }

    if (closes.length < CONFIG.bbPeriod) {
      continue;
    }

    const bb = calculateBollingerBands(closes, CONFIG.bbPeriod, CONFIG.bbStdDev);
    if (!bb) continue;

    const rsiValues = RSI.calculate({ values: closes, period: CONFIG.rsiPeriod });
    const currentRSI = rsiValues[rsiValues.length - 1];
    if (currentRSI === undefined) continue;

    const ttmBars = bars.slice(Math.max(0, i - 20), i + 1);
    const ttmSqueeze = calculateTtmSqueeze(ttmBars, { lookback: 20, bbStdDev: 2, atrMultiplier: 1.5 });
    if (!ttmSqueeze) continue;

    if (position) {
      const direction = position.side === 'long' ? 1 : -1;

      // Take Profit
      if (position.target !== null) {
        const hitTarget = (direction === 1 && bar.high >= position.target) ||
                         (direction === -1 && bar.low <= position.target);

        if (hitTarget) {
          const basePrice = direction === 1
            ? Math.min(bar.high, position.target)
            : Math.max(bar.low, position.target);
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const exitPrice = roundToTick(fillTP(baseSymbol, closeSide, basePrice));

          const S_ticks = SLIP_CONFIG.avgSpreadTicks[baseSymbol];
          const sigma_tp_ticks = SLIP_CONFIG.slipAvg.tp[baseSymbol];
          const p_passive = SLIP_CONFIG.p_tp_passive[baseSymbol];
          const E_tp_ticks = (1 - p_passive) * (S_ticks + sigma_tp_ticks);
          const tpSlippagePoints = E_tp_ticks * tickSize;

          console.log(
            `[${bar.timestamp}] TARGET HIT ${CONFIG.symbol} ${position.side.toUpperCase()}: ` +
            `Exit @ ${exitPrice.toFixed(2)}`
          );
          exitPosition(exitPrice, bar.timestamp, currentRSI, 'target', tpSlippagePoints);
          continue;
        }
      }

      // Stop Loss
      if (position.stopLoss !== null) {
        const hitStop = (direction === 1 && bar.low <= position.stopLoss) ||
                       (direction === -1 && bar.high >= position.stopLoss);
        if (hitStop) {
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const stopExitPrice = roundToTick(fillStop(baseSymbol, closeSide, position.stopLoss));
          const stopSlippagePoints = SLIP_CONFIG.slipAvg.stop[baseSymbol] * tickSize;
          exitPosition(stopExitPrice, bar.timestamp, currentRSI, 'stop', stopSlippagePoints);
          continue;
        }
      }

      continue;
    }

    // Trending strategy entry logic
    const price = bar.close;
    const longSetupDetected = price >= bb.upper && currentRSI > CONFIG.rsiEntryLong;
    const shortSetupDetected = price <= bb.lower && currentRSI < CONFIG.rsiEntryShort;

    const currentMomentumSign = Math.sign(ttmSqueeze.momentum);

    // LONG entry logic
    if (longSetupDetected && !momentumSetup) {
      if (currentMomentumSign < 0) {
        // Currently red (bearish) - wait for green (bullish)
        momentumSetup = {
          side: 'long',
          setupTime: bar.timestamp,
          setupPrice: bar.close,
          rsi: currentRSI,
          bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
          momentumState: 'ready',
          lastMomentumSign: currentMomentumSign,
        };
        console.log(
          `[${bar.timestamp}] LONG setup ${CONFIG.symbol} @ ${bar.close.toFixed(2)} ` +
          `(RSI ${currentRSI.toFixed(1)}) | TTM: RED - waiting for GREEN`
        );
      } else {
        // Currently green (bullish) - need to see red first, then green again
        momentumSetup = {
          side: 'long',
          setupTime: bar.timestamp,
          setupPrice: bar.close,
          rsi: currentRSI,
          bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
          momentumState: 'waiting_for_red',
          lastMomentumSign: currentMomentumSign,
        };
        console.log(
          `[${bar.timestamp}] LONG setup ${CONFIG.symbol} @ ${bar.close.toFixed(2)} ` +
          `(RSI ${currentRSI.toFixed(1)}) | TTM: GREEN - waiting for RED -> GREEN cycle`
        );
      }
    }

    // SHORT entry logic
    if (shortSetupDetected && !momentumSetup) {
      if (currentMomentumSign > 0) {
        // Currently green (bullish) - wait for red (bearish)
        momentumSetup = {
          side: 'short',
          setupTime: bar.timestamp,
          setupPrice: bar.close,
          rsi: currentRSI,
          bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
          momentumState: 'ready',
          lastMomentumSign: currentMomentumSign,
        };
        console.log(
          `[${bar.timestamp}] SHORT setup ${CONFIG.symbol} @ ${bar.close.toFixed(2)} ` +
          `(RSI ${currentRSI.toFixed(1)}) | TTM: GREEN - waiting for RED`
        );
      } else {
        // Currently red (bearish) - need to see green first, then red again
        momentumSetup = {
          side: 'short',
          setupTime: bar.timestamp,
          setupPrice: bar.close,
          rsi: currentRSI,
          bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
          momentumState: 'waiting_for_green',
          lastMomentumSign: currentMomentumSign,
        };
        console.log(
          `[${bar.timestamp}] SHORT setup ${CONFIG.symbol} @ ${bar.close.toFixed(2)} ` +
          `(RSI ${currentRSI.toFixed(1)}) | TTM: RED - waiting for GREEN -> RED cycle`
        );
      }
    }

    // Process momentum state machine for LONG setups
    if (momentumSetup && momentumSetup.side === 'long') {
      const prevSign = momentumSetup.lastMomentumSign;

      if (momentumSetup.momentumState === 'waiting_for_red') {
        // Was green, waiting for red
        if (currentMomentumSign < 0 && prevSign >= 0) {
          console.log(`[${bar.timestamp}] TTM turned RED - now waiting for GREEN`);
          momentumSetup.momentumState = 'ready';
        }
      } else if (momentumSetup.momentumState === 'ready') {
        // Waiting for green signal
        if (currentMomentumSign > 0 && prevSign <= 0) {
          console.log(`[${bar.timestamp}] TTM turned GREEN - entering LONG`);

          const entrySide = 'buy';
          const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, bar.close));

          position = {
            side: 'long',
            entryPrice,
            entryTime: bar.timestamp,
            entryRSI: momentumSetup.rsi,
            stopLoss: roundToTick(entryPrice * (1 - CONFIG.stopLossPercent)),
            target: roundToTick(entryPrice * (1 + CONFIG.takeProfitPercent)),
            remainingQty: CONFIG.numberOfContracts,
            feesPaid: addFees(baseSymbol, CONFIG.numberOfContracts),
          };

          console.log(
            `[${bar.timestamp}] LONG entry ${CONFIG.symbol} @ ${entryPrice.toFixed(2)} ` +
            `(RSI ${momentumSetup.rsi.toFixed(1)}, Stop ${position.stopLoss.toFixed(2)}, Target ${position.target?.toFixed(2)})`
          );

          momentumSetup = null;
        }
      }

      if (momentumSetup) {
        momentumSetup.lastMomentumSign = currentMomentumSign;
      }
    }

    // Process momentum state machine for SHORT setups
    if (momentumSetup && momentumSetup.side === 'short') {
      const prevSign = momentumSetup.lastMomentumSign;

      if (momentumSetup.momentumState === 'waiting_for_green') {
        // Was red, waiting for green
        if (currentMomentumSign > 0 && prevSign <= 0) {
          console.log(`[${bar.timestamp}] TTM turned GREEN - now waiting for RED`);
          momentumSetup.momentumState = 'ready';
        }
      } else if (momentumSetup.momentumState === 'ready') {
        // Waiting for red signal
        if (currentMomentumSign < 0 && prevSign >= 0) {
          console.log(`[${bar.timestamp}] TTM turned RED - entering SHORT`);

          const entrySide = 'sell';
          const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, bar.close));

          position = {
            side: 'short',
            entryPrice,
            entryTime: bar.timestamp,
            entryRSI: momentumSetup.rsi,
            stopLoss: roundToTick(entryPrice * (1 + CONFIG.stopLossPercent)),
            target: roundToTick(entryPrice * (1 - CONFIG.takeProfitPercent)),
            remainingQty: CONFIG.numberOfContracts,
            feesPaid: addFees(baseSymbol, CONFIG.numberOfContracts),
          };

          console.log(
            `[${bar.timestamp}] SHORT entry ${CONFIG.symbol} @ ${entryPrice.toFixed(2)} ` +
            `(RSI ${momentumSetup.rsi.toFixed(1)}, Stop ${position.stopLoss.toFixed(2)}, Target ${position.target?.toFixed(2)})`
          );

          momentumSetup = null;
        }
      }

      if (momentumSetup) {
        momentumSetup.lastMomentumSign = currentMomentumSign;
      }
    }
  }

  if (position) {
    const lastBar = bars[bars.length - 1];
    const rsiValues = RSI.calculate({ values: closes, period: CONFIG.rsiPeriod });
    const lastRSI = rsiValues[rsiValues.length - 1];
    const closeSide = position.side === 'long' ? 'sell' : 'buy';
    const endExitPrice = roundToTick(fillStop(baseSymbol, closeSide, lastBar.close));
    const stopSlippagePoints = SLIP_CONFIG.slipAvg.stop[baseSymbol] * tickSize;
    exitPosition(endExitPrice, lastBar.timestamp, lastRSI, 'end_of_data', stopSlippagePoints);
  }

  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl <= 0);
  const winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;

  const avgWin = winningTrades.length
    ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length
    : 0;
  const avgLoss = losingTrades.length
    ? Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length)
    : 0;

  const grossProfit = trades
    .filter(t => t.grossPnl > 0)
    .reduce((sum, t) => sum + t.grossPnl, 0);
  const grossLoss = Math.abs(
    trades
      .filter(t => t.grossPnl <= 0)
      .reduce((sum, t) => sum + t.grossPnl, 0),
  );
  const totalFees = trades.reduce((sum, t) => sum + t.fees, 0);
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

  let runningPnL = 0;
  let peakPnL = 0;
  let maxDrawdown = 0;

  trades.forEach(trade => {
    runningPnL += trade.pnl;
    if (runningPnL > peakPnL) {
      peakPnL = runningPnL;
    }
    const drawdown = peakPnL - runningPnL;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  });

  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total Trades: ${trades.length} | Wins: ${winningTrades.length} | Losses: ${losingTrades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(1)}%`);
  console.log(`Net Realized PnL: ${formatCurrency(realizedPnL)} USD | Fees Paid: $${totalFees.toFixed(2)}`);
  console.log(`Gross Profit (pre-fees): ${formatCurrency(grossProfit)} | Gross Loss: ${formatCurrency(grossLoss)}`);
  console.log(`Avg Win: ${formatCurrency(avgWin)} | Avg Loss: ${formatCurrency(avgLoss)}`);
  console.log(`Profit Factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);
  console.log(`Max Drawdown: ${formatCurrency(maxDrawdown)} USD`);

  // Slippage Statistics
  if (trades.length > 0) {
    const totalSlippageCost = trades.reduce((sum, t) => sum + (t.slippageCost || 0), 0);
    const avgSlippageCost = totalSlippageCost / trades.length;
    const avgEntrySlippageTicks = trades.reduce((sum, t) => sum + (t.entrySlippageTicks || 0), 0) / trades.length;
    const avgExitSlippageTicks = trades.reduce((sum, t) => sum + (t.exitSlippageTicks || 0), 0) / trades.length;
    const avgTotalSlippageTicks = avgEntrySlippageTicks + avgExitSlippageTicks;

    console.log(`\nSlippage Impact (${baseSymbol}):`);
    console.log(`  Total Slippage Cost: ${formatCurrency(totalSlippageCost)}`);
    console.log(`  Avg Per Trade: ${formatCurrency(avgSlippageCost)} | ${avgTotalSlippageTicks.toFixed(2)} ticks`);
  }

  // Exit reason breakdown
  const exitReasons = trades.reduce((acc, t) => {
    acc[t.exitReason] = (acc[t.exitReason] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  console.log(`\nExit Reasons: ${Object.entries(exitReasons).map(([r, c]) => `${r}=${c}`).join(', ')}`);


  if (trades.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log('RECENT TRADES');
    console.log('='.repeat(80));
    trades.slice(-10).forEach(trade => {
      console.log(
        `${trade.side.toUpperCase().padEnd(5)} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ` +
        `${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | ${formatCurrency(trade.pnl).padStart(10)} (${trade.exitReason}, RSI: ${trade.entryRSI.toFixed(1)}->${trade.exitRSI?.toFixed(1) || 'N/A'})`
      );
    });
  }
}

runBacktest().catch(err => {
  console.error('TopstepX NQ trender backtest failed:', err);
  process.exit(1);
});
