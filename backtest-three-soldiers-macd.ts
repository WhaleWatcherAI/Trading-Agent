#!/usr/bin/env tsx
/**
 * Three Soldiers + MACD Strategy Backtest
 *
 * Strategy:
 * - Entry signals (both must occur within 4 bars of each other, in any order):
 *   1. Three Soldiers: 3 consecutive bullish candles (for long) or 3 consecutive bearish candles (for short)
 *   2. MACD Crossover: MACD line crosses above signal line (for long) or below signal line (for short)
 *
 * - Exit signals (either triggers exit):
 *   1. Inverse MACD crossover (MACD crosses opposite direction)
 *   2. Three opposite soldiers (3 consecutive candles in opposite direction)
 *
 * - Parameters:
 *   - MACD: 12, 26, 9 (standard)
 *   - Soldiers: 3 consecutive candles of same color
 *   - Signal window: Both signals must occur within 4 bars
 *   - Stop Loss: Optional safety stop (default 20 ticks)
 */

import { MACD } from 'technicalindicators';
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
  macdFastPeriod: number;
  macdSlowPeriod: number;
  macdSignalPeriod: number;
  soldiersCount: number;
  signalWindowBars: number;
  stopLossTicks: number;
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
  exitReason: 'macd_cross' | 'opposite_soldiers' | 'stop' | 'session' | 'end_of_data';
  entrySignal: string;
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DEFAULT_DAYS = 90;

const DEFAULT_SYMBOL = process.env.THREE_SOLDIERS_SYMBOL || 'NQZ5';
const DEFAULT_CONTRACT_ID = process.env.THREE_SOLDIERS_CONTRACT_ID;

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

// Helper functions for realistic fill simulation
function fillEntry(sym: string, side: 'buy' | 'sell', mid: number): number {
  const t = SLIP_CONFIG.tickSize[sym];
  const S = 0.5 * t * SLIP_CONFIG.avgSpreadTicks[sym];
  const sigma = SLIP_CONFIG.slipAvg.entry[sym] * t;
  return side === 'buy' ? mid + S + sigma : mid - S - sigma;
}

function fillStop(sym: string, side: 'buy' | 'sell', triggerMid: number): number {
  const t = SLIP_CONFIG.tickSize[sym];
  const sigma = SLIP_CONFIG.slipAvg.stop[sym] * t;
  return side === 'buy' ? triggerMid + sigma : triggerMid - sigma;
}

function addFees(sym: string, contracts: number): number {
  return SLIP_CONFIG.feesPerSideUSD[sym] * contracts;
}

// Extract base symbol (NQZ5 -> NQ, MESZ5 -> MES)
const getBaseSymbol = (fullSymbol: string): string => {
  return fullSymbol.replace(/[A-Z]\d+$/, '');
};

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_SYMBOL,
  contractId: DEFAULT_CONTRACT_ID,
  start: process.env.THREE_SOLDIERS_START || new Date(Date.now() - DEFAULT_DAYS * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.THREE_SOLDIERS_END || new Date().toISOString(),
  macdFastPeriod: Number(process.env.THREE_SOLDIERS_MACD_FAST || '12'),
  macdSlowPeriod: Number(process.env.THREE_SOLDIERS_MACD_SLOW || '26'),
  macdSignalPeriod: Number(process.env.THREE_SOLDIERS_MACD_SIGNAL || '9'),
  soldiersCount: Number(process.env.THREE_SOLDIERS_COUNT || '3'),
  signalWindowBars: Number(process.env.THREE_SOLDIERS_WINDOW || '4'),
  stopLossTicks: Number(process.env.THREE_SOLDIERS_STOP_TICKS || '20'),
  numberOfContracts: Number(process.env.THREE_SOLDIERS_CONTRACTS || '1'),
  commissionPerSide: process.env.THREE_SOLDIERS_COMMISSION
    ? Number(process.env.THREE_SOLDIERS_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_CONTRACT_ID, DEFAULT_SYMBOL], 0.35),
  slippageTicks: Number(process.env.THREE_SOLDIERS_SLIPPAGE_TICKS || '1'),
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

// Detect three consecutive soldiers (bullish or bearish)
function detectThreeSoldiers(bars: TopstepXFuturesBar[], index: number): {
  bullish: boolean;
  bearish: boolean;
} | null {
  if (index < 2) return null;

  const bar1 = bars[index - 2];
  const bar2 = bars[index - 1];
  const bar3 = bars[index];

  const bullish1 = bar1.close > bar1.open;
  const bullish2 = bar2.close > bar2.open;
  const bullish3 = bar3.close > bar3.open;

  const bearish1 = bar1.close < bar1.open;
  const bearish2 = bar2.close < bar2.open;
  const bearish3 = bar3.close < bar3.open;

  return {
    bullish: bullish1 && bullish2 && bullish3,
    bearish: bearish1 && bearish2 && bearish3,
  };
}

// Detect MACD crossover
function detectMacdCrossover(
  currentMacd: { MACD: number; signal: number; histogram: number },
  prevMacd: { MACD: number; signal: number; histogram: number } | null
): {
  bullishCross: boolean;
  bearishCross: boolean;
} {
  if (!prevMacd) return { bullishCross: false, bearishCross: false };

  const bullishCross = prevMacd.MACD <= prevMacd.signal && currentMacd.MACD > currentMacd.signal;
  const bearishCross = prevMacd.MACD >= prevMacd.signal && currentMacd.MACD < currentMacd.signal;

  return { bullishCross, bearishCross };
}

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('THREE SOLDIERS + MACD STRATEGY BACKTEST');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end}`);
  console.log(`MACD: ${CONFIG.macdFastPeriod}, ${CONFIG.macdSlowPeriod}, ${CONFIG.macdSignalPeriod}`);
  console.log(`Soldiers Count: ${CONFIG.soldiersCount} consecutive candles`);
  console.log(`Signal Window: ${CONFIG.signalWindowBars} bars`);
  console.log(`Stop Loss: ${CONFIG.stopLossTicks} ticks | Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`Commission/side: ${CONFIG.commissionPerSide.toFixed(2)} USD | Slippage: ${CONFIG.slippageTicks} tick(s)`);
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[three-soldiers] Unable to fetch metadata:', err.message);
    return null;
  });

  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  const contractId = metadata.id;
  const multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || 5;
  const tickSize = metadata.tickSize || 0.25;

  const baseSymbol = getBaseSymbol(CONFIG.symbol);

  console.log(`Resolved contract: ${metadata.name} (${contractId})`);
  console.log(`Point multiplier: ${multiplier}`);
  console.log(`Tick size: ${tickSize}`);
  console.log(`Spread Model (${baseSymbol}): Avg spread ${SLIP_CONFIG.avgSpreadTicks[baseSymbol]} tick, Fees $${SLIP_CONFIG.feesPerSideUSD[baseSymbol]} per side`);

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

  // Track pending signals within the window
  let pendingSignal: {
    type: 'soldiers' | 'macd';
    side: 'long' | 'short';
    barIndex: number;
    timestamp: string;
    price: number;
  } | null = null;

  let position: {
    side: 'long' | 'short';
    entryPrice: number;
    entryTime: string;
    stopLoss: number | null;
    entrySignal: string;
  } | null = null;

  let realizedPnL = 0;

  const exitPosition = (
    exitPrice: number,
    exitTime: string,
    reason: TradeRecord['exitReason'],
  ) => {
    if (!position) return;

    const direction = position.side === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - position.entryPrice) * direction * multiplier * CONFIG.numberOfContracts;
    const totalFees = addFees(baseSymbol, CONFIG.numberOfContracts) * 2; // entry + exit
    const grossPnl = rawPnL;
    const netPnl = grossPnl - totalFees;

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
      entrySignal: position.entrySignal,
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
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const sessionExitPrice = roundToTick(fillStop(baseSymbol, closeSide, bar.close));
        exitPosition(sessionExitPrice, bar.timestamp, 'session');
      }
      pendingSignal = null;
      continue;
    }

    if (closes.length < CONFIG.macdSlowPeriod + CONFIG.macdSignalPeriod) {
      continue;
    }

    // Calculate MACD
    const macdValues = MACD.calculate({
      values: closes,
      fastPeriod: CONFIG.macdFastPeriod,
      slowPeriod: CONFIG.macdSlowPeriod,
      signalPeriod: CONFIG.macdSignalPeriod,
      SimpleMAOscillator: false,
      SimpleMASignal: false,
    });

    const currentMACD = macdValues.length >= 2 ? macdValues[macdValues.length - 1] : null;
    const prevMACD = macdValues.length >= 2 ? macdValues[macdValues.length - 2] : null;

    if (!currentMACD || !prevMACD) continue;

    // Detect three soldiers
    const soldiers = detectThreeSoldiers(bars, i);

    // Detect MACD crossover
    const macdCross = detectMacdCrossover(currentMACD, prevMACD);

    // Check if we're in a position
    if (position) {
      // Exit logic: inverse MACD crossover OR three opposite soldiers

      // Check for inverse MACD crossover
      const inverseMacdCross = (position.side === 'long' && macdCross.bearishCross) ||
                               (position.side === 'short' && macdCross.bullishCross);

      // Check for opposite soldiers
      const oppositeSoldiers = (position.side === 'long' && soldiers?.bearish) ||
                               (position.side === 'short' && soldiers?.bullish);

      if (inverseMacdCross) {
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const exitPrice = roundToTick(fillEntry(baseSymbol, closeSide, bar.close));
        console.log(`[${bar.timestamp}] EXIT ${position.side.toUpperCase()} @ ${exitPrice.toFixed(2)} - Inverse MACD crossover`);
        exitPosition(exitPrice, bar.timestamp, 'macd_cross');
        continue;
      }

      if (oppositeSoldiers) {
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const exitPrice = roundToTick(fillEntry(baseSymbol, closeSide, bar.close));
        console.log(`[${bar.timestamp}] EXIT ${position.side.toUpperCase()} @ ${exitPrice.toFixed(2)} - Three opposite soldiers`);
        exitPosition(exitPrice, bar.timestamp, 'opposite_soldiers');
        continue;
      }

      // Stop loss disabled
      // if (position.stopLoss !== null) {
      //   const direction = position.side === 'long' ? 1 : -1;
      //   const hitStop = (direction === 1 && bar.low <= position.stopLoss) ||
      //                  (direction === -1 && bar.high >= position.stopLoss);
      //   if (hitStop) {
      //     const closeSide = position.side === 'long' ? 'sell' : 'buy';
      //     const stopExitPrice = roundToTick(fillStop(baseSymbol, closeSide, position.stopLoss));
      //     console.log(`[${bar.timestamp}] STOP hit ${position.side.toUpperCase()} @ ${stopExitPrice.toFixed(2)}`);
      //     exitPosition(stopExitPrice, bar.timestamp, 'stop');
      //     continue;
      //   }
      // }

      continue;
    }

    // Entry logic: Look for both signals within window

    // Check if pending signal expired (outside window)
    if (pendingSignal && (i - pendingSignal.barIndex) > CONFIG.signalWindowBars) {
      pendingSignal = null;
    }

    // Detect new soldiers signal
    if (soldiers) {
      if (soldiers.bullish) {
        if (pendingSignal?.type === 'macd' && pendingSignal.side === 'long') {
          // Both signals detected - enter long
          const entrySide = 'buy';
          const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, bar.close));
          const stopLoss = roundToTick(entryPrice - (CONFIG.stopLossTicks * tickSize));

          position = {
            side: 'long',
            entryPrice,
            entryTime: bar.timestamp,
            stopLoss,
            entrySignal: `Soldiers + MACD (${i - pendingSignal.barIndex} bars apart)`,
          };

          console.log(`[${bar.timestamp}] LONG entry @ ${entryPrice.toFixed(2)} - Three bullish soldiers + MACD cross (${i - pendingSignal.barIndex} bars apart)`);
          pendingSignal = null;
        } else if (!pendingSignal || pendingSignal.side !== 'long') {
          // New bullish soldiers signal
          pendingSignal = {
            type: 'soldiers',
            side: 'long',
            barIndex: i,
            timestamp: bar.timestamp,
            price: bar.close,
          };
          console.log(`[${bar.timestamp}] Three bullish soldiers detected @ ${bar.close.toFixed(2)} - awaiting MACD cross within ${CONFIG.signalWindowBars} bars`);
        }
      } else if (soldiers.bearish) {
        if (pendingSignal?.type === 'macd' && pendingSignal.side === 'short') {
          // Both signals detected - enter short
          const entrySide = 'sell';
          const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, bar.close));
          const stopLoss = roundToTick(entryPrice + (CONFIG.stopLossTicks * tickSize));

          position = {
            side: 'short',
            entryPrice,
            entryTime: bar.timestamp,
            stopLoss,
            entrySignal: `Soldiers + MACD (${i - pendingSignal.barIndex} bars apart)`,
          };

          console.log(`[${bar.timestamp}] SHORT entry @ ${entryPrice.toFixed(2)} - Three bearish soldiers + MACD cross (${i - pendingSignal.barIndex} bars apart)`);
          pendingSignal = null;
        } else if (!pendingSignal || pendingSignal.side !== 'short') {
          // New bearish soldiers signal
          pendingSignal = {
            type: 'soldiers',
            side: 'short',
            barIndex: i,
            timestamp: bar.timestamp,
            price: bar.close,
          };
          console.log(`[${bar.timestamp}] Three bearish soldiers detected @ ${bar.close.toFixed(2)} - awaiting MACD cross within ${CONFIG.signalWindowBars} bars`);
        }
      }
    }

    // Detect new MACD crossover signal
    if (macdCross.bullishCross) {
      if (pendingSignal?.type === 'soldiers' && pendingSignal.side === 'long') {
        // Both signals detected - enter long
        const entrySide = 'buy';
        const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, bar.close));
        const stopLoss = roundToTick(entryPrice - (CONFIG.stopLossTicks * tickSize));

        position = {
          side: 'long',
          entryPrice,
          entryTime: bar.timestamp,
          stopLoss,
          entrySignal: `MACD + Soldiers (${i - pendingSignal.barIndex} bars apart)`,
        };

        console.log(`[${bar.timestamp}] LONG entry @ ${entryPrice.toFixed(2)} - MACD bullish cross + Three soldiers (${i - pendingSignal.barIndex} bars apart)`);
        pendingSignal = null;
      } else if (!pendingSignal || pendingSignal.side !== 'long') {
        // New bullish MACD signal
        pendingSignal = {
          type: 'macd',
          side: 'long',
          barIndex: i,
          timestamp: bar.timestamp,
          price: bar.close,
        };
        console.log(`[${bar.timestamp}] MACD bullish crossover detected @ ${bar.close.toFixed(2)} - awaiting three soldiers within ${CONFIG.signalWindowBars} bars`);
      }
    } else if (macdCross.bearishCross) {
      if (pendingSignal?.type === 'soldiers' && pendingSignal.side === 'short') {
        // Both signals detected - enter short
        const entrySide = 'sell';
        const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, bar.close));
        const stopLoss = roundToTick(entryPrice + (CONFIG.stopLossTicks * tickSize));

        position = {
          side: 'short',
          entryPrice,
          entryTime: bar.timestamp,
          stopLoss,
          entrySignal: `MACD + Soldiers (${i - pendingSignal.barIndex} bars apart)`,
        };

        console.log(`[${bar.timestamp}] SHORT entry @ ${entryPrice.toFixed(2)} - MACD bearish cross + Three soldiers (${i - pendingSignal.barIndex} bars apart)`);
        pendingSignal = null;
      } else if (!pendingSignal || pendingSignal.side !== 'short') {
        // New bearish MACD signal
        pendingSignal = {
          type: 'macd',
          side: 'short',
          barIndex: i,
          timestamp: bar.timestamp,
          price: bar.close,
        };
        console.log(`[${bar.timestamp}] MACD bearish crossover detected @ ${bar.close.toFixed(2)} - awaiting three soldiers within ${CONFIG.signalWindowBars} bars`);
      }
    }
  }

  if (position) {
    const lastBar = bars[bars.length - 1];
    const closeSide = position.side === 'long' ? 'sell' : 'buy';
    const endExitPrice = roundToTick(fillStop(baseSymbol, closeSide, lastBar.close));
    exitPosition(endExitPrice, lastBar.timestamp, 'end_of_data');
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
    trades.slice(-20).forEach(trade => {
      console.log(
        `${trade.side.toUpperCase().padEnd(5)} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ` +
        `${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | ${formatCurrency(trade.pnl).padStart(10)} (${trade.exitReason}) [${trade.entrySignal}]`
      );
    });
  }
}

runBacktest().catch(err => {
  console.error('Three Soldiers + MACD backtest failed:', err);
  process.exit(1);
});
