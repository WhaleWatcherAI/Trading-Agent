#!/usr/bin/env tsx
/**
 * SVP-Only Mean Reversion Backtest
 *
 * Pure auction theory approach:
 * - Daily bias from POC migration
 * - Fade value edges (VAL/VAH) based on bias
 * - Enter at LVNs when TTM Squeeze is ON (confirming compression)
 * - Target POC then opposite edge
 * - Stop beyond LVN or edge + 0.05×VAW
 */

import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';
import { calculateTtmSqueeze } from './lib/ttmSqueeze';
import {
  calculateSessionProfile,
  calculateBias,
  evaluateTradeOpportunity,
  isInsidePriorValue,
  SessionProfile,
  Bias,
} from './lib/svpFramework';
import { RSI } from 'technicalindicators';

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

interface TradeRecord {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  entryFill: number;
  exitFill: number;
  pnl: number;
  grossPnl: number;
  grossBeforeFees: number;
  fees: number;
  exitReason: 'stop' | 'tp1_poc' | 'tp2_edge' | 'session';
  bias: Bias;
  entryReasoning: string;
}

const DEFAULT_SYMBOL = process.env.TOPSTEPX_SVP_SYMBOL || 'MESZ5';
const DEFAULT_CONTRACT_ID = process.env.TOPSTEPX_SVP_CONTRACT_ID;
const DEFAULT_DAYS = 7;

const CONFIG = {
  symbol: DEFAULT_SYMBOL,
  contractId: DEFAULT_CONTRACT_ID,
  start: process.env.TOPSTEPX_SVP_START || new Date(Date.now() - DEFAULT_DAYS * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.TOPSTEPX_SVP_END || new Date().toISOString(),
  numberOfContracts: Number(process.env.TOPSTEPX_SVP_CONTRACTS || '2'),
  commissionPerSide: process.env.TOPSTEPX_SVP_COMMISSION
    ? Number(process.env.TOPSTEPX_SVP_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_CONTRACT_ID, DEFAULT_SYMBOL], 0.35),
};

const RSI_CONFIG = {
  period: Number(process.env.TOPSTEPX_SVP_RSI_PERIOD || '14'),
  oversold: Number(process.env.TOPSTEPX_SVP_RSI_OVERSOLD || '27'),
  overbought: Number(process.env.TOPSTEPX_SVP_RSI_OVERBOUGHT || '73'),
};
const USE_RSI_FILTER = process.env.TOPSTEPX_SVP_USE_RSI === 'true';

const CT_OFFSET_MINUTES = 6 * 60;

function toCentralTime(date: Date): Date {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isRTH(timestamp: string | Date): boolean {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const ctDate = toCentralTime(date);
  const day = ctDate.getUTCDay();
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  // RTH: Mon-Fri 8:30 AM - 3:00 PM CT (510-900 minutes)
  if (day === 0 || day === 6) return false;
  return minutes >= 510 && minutes < 900;
}

function formatCurrency(value: number): string {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('SVP-ONLY MEAN REVERSION BACKTEST');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end}`);
  console.log(`Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`Commission/side: ${CONFIG.commissionPerSide.toFixed(2)} USD`);
  console.log('Strategy: Pure SVP auction theory (POC migration bias, fade value edges)');
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[svp-backtest] Unable to fetch metadata:', err.message);
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

  console.log('\nFetching 10-second bars...');
  const bars = await fetchTopstepXFuturesBars({
    contractId,
    startTime: CONFIG.start,
    endTime: CONFIG.end,
    unit: 1,
    unitNumber: 10, // 10-second bars for 1 week backtest
    limit: 50000,
  });

  if (!bars.length) {
    throw new Error('No bars returned');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length} ten-second bars`);

  // Separate bars by RTH session date
  const sessionBarsMap = new Map<string, TopstepXFuturesBar[]>();
  for (const bar of bars) {
    if (!isRTH(bar.timestamp)) continue;

    const sessionDate = new Date(bar.timestamp).toISOString().split('T')[0];
    if (!sessionBarsMap.has(sessionDate)) {
      sessionBarsMap.set(sessionDate, []);
    }
    sessionBarsMap.get(sessionDate)!.push(bar);
  }

  const sessionDates = Array.from(sessionBarsMap.keys()).sort();
  console.log(`Found ${sessionDates.length} RTH sessions`);

  const completedProfiles: SessionProfile[] = [];

  const roundToTick = (price: number): number => {
    return Math.round(price / tickSize) * tickSize;
  };

  const ttmBars: Array<{ high: number; low: number; close: number }> = [];
  const closes: number[] = [];
  const trades: TradeRecord[] = [];

  let pendingSetup: {
    side: 'long' | 'short';
    setupTime: string;
    setupTimestamp: number;
    setupPrice: number;
    stopLoss: number;
    tp1_POC: number;
    tp2_Edge: number;
    bias: Bias;
    reasoning: string;
    entryLimit: number;
    rsi: number | null;
  } | null = null;

  let position: {
    side: 'long' | 'short';
    entryPrice: number;
    entryTime: string;
    stopLoss: number;
    tp1_POC: number;
    tp2_Edge: number;
    tp1Hit: boolean;
    feesPaid: number;
    bias: Bias;
    reasoning: string;
  } | null = null;

  let lastEntryTime: number = 0;
  const MIN_ENTRY_INTERVAL_MS = 60 * 1000; // Min 60 seconds between entries

  let realizedPnL = 0;

  // Process bars session by session
  for (let sessionIdx = 0; sessionIdx < sessionDates.length; sessionIdx++) {
    const sessionDate = sessionDates[sessionIdx];
    const sessionBars = sessionBarsMap.get(sessionDate)!;
    const yesterdayProfile = completedProfiles[completedProfiles.length - 1] || null;
    const twoDaysAgoProfile = completedProfiles.length >= 2 ? completedProfiles[completedProfiles.length - 2] : null;
    let intradayProfile: SessionProfile | null = null;

    for (let i = 0; i < sessionBars.length; i++) {
      const bar = sessionBars[i];
      ttmBars.push({ high: bar.high, low: bar.low, close: bar.close });
      closes.push(bar.close);

      const partialBars = sessionBars.slice(0, i + 1);
      intradayProfile = calculateSessionProfile(partialBars, tickSize);
      const profileSnapshot = intradayProfile;
      const historicalProfiles = completedProfiles.concat([profileSnapshot]);
      const biasCalc = calculateBias(profileSnapshot, yesterdayProfile, twoDaysAgoProfile, historicalProfiles);
      if (i === 0) {
        console.log(
          `\n[${sessionDate}] Bias: ${biasCalc.bias.toUpperCase()} ` +
          `(POC drift: ${biasCalc.pocDrift.toFixed(2)}, gate: ${biasCalc.driftGate.toFixed(2)})`
        );
      }

      const rsiSeries = closes.length >= RSI_CONFIG.period
        ? RSI.calculate({ period: RSI_CONFIG.period, values: closes })
        : [];
      const currentRSI = rsiSeries.length ? rsiSeries[rsiSeries.length - 1] : null;

      // Calculate TTM Squeeze
      const ttmSqueeze = calculateTtmSqueeze(ttmBars);
      if (!ttmSqueeze) {
        continue; // Skip if not enough bars for TTM calculation
      }


      // Manage existing position
      if (position) {
        const direction = position.side === 'long' ? 1 : -1;

        // Check TP1 (POC)
        if (!position.tp1Hit) {
          const hitTP1 = (direction === 1 && bar.high >= position.tp1_POC) ||
                        (direction === -1 && bar.low <= position.tp1_POC);

          if (hitTP1) {
            // Take profit at POC
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const exitFill = roundToTick(fillTP(baseSymbol, closeSide, position.tp1_POC));

          const grossBeforeFees = (exitFill - position.entryPrice) * direction * multiplier * CONFIG.numberOfContracts;
          const exitFees = addFees(baseSymbol, CONFIG.numberOfContracts);
          const totalFees = position.feesPaid + exitFees;
          const netPnl = grossBeforeFees - totalFees;

          trades.push({
            entryTime: position.entryTime,
            exitTime: bar.timestamp,
            side: position.side,
            entryPrice: position.entryPrice,
            exitPrice: exitFill,
            entryFill: position.entryPrice,
            exitFill,
            pnl: netPnl,
            grossPnl: grossBeforeFees,
            grossBeforeFees,
            fees: totalFees,
            exitReason: 'tp1_poc',
            bias: position.bias,
            entryReasoning: position.reasoning,
          });

          realizedPnL += netPnl;
          position = null;
          continue;
        }
      }

        // Check stop
        const hitStop = (direction === 1 && bar.low <= position.stopLoss) ||
                       (direction === -1 && bar.high >= position.stopLoss);

        if (hitStop) {
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const stopFill = roundToTick(fillStop(baseSymbol, closeSide, position.stopLoss));

          const grossBeforeFees = (stopFill - position.entryPrice) * direction * multiplier * CONFIG.numberOfContracts;
          const exitFees = addFees(baseSymbol, CONFIG.numberOfContracts);
          const totalFees = position.feesPaid + exitFees;
          const netPnl = grossBeforeFees - totalFees;

          trades.push({
            entryTime: position.entryTime,
            exitTime: bar.timestamp,
            side: position.side,
            entryPrice: position.entryPrice,
            exitPrice: stopFill,
            entryFill: position.entryPrice,
            exitFill: stopFill,
            pnl: netPnl,
            grossPnl: grossBeforeFees,
            grossBeforeFees,
            fees: totalFees,
            exitReason: 'stop',
            bias: position.bias,
            entryReasoning: position.reasoning,
          });

          realizedPnL += netPnl;
          position = null;
          continue;
        }
        continue;
      }

      // Look for new entry opportunity (Stage 1)
      const opportunity = evaluateTradeOpportunity(bar.close, biasCalc.bias, profileSnapshot, tickSize);

      const currentTime = new Date(bar.timestamp).getTime();
      const timeSinceLastEntry = currentTime - lastEntryTime;

      if (opportunity.side && opportunity.entryLimit) {
        if (USE_RSI_FILTER) {
          if (currentRSI === null) {
            continue;
          }

          const rsiAligned =
            (opportunity.side === 'long' && currentRSI <= RSI_CONFIG.oversold) ||
            (opportunity.side === 'short' && currentRSI >= RSI_CONFIG.overbought);

          if (!rsiAligned) {
            console.log(
              `[${bar.timestamp}] ${opportunity.side.toUpperCase()} setup skipped (RSI ${currentRSI.toFixed(1)} outside ${RSI_CONFIG.oversold}/${RSI_CONFIG.overbought})`
            );
            continue;
          }
        }

        if (pendingSetup && pendingSetup.side !== opportunity.side) {
          console.log(
            `[${bar.timestamp}] Cancelled ${pendingSetup.side.toUpperCase()} setup (opposite edge detected)`
          );
          pendingSetup = null;
        }

        if (!pendingSetup) {
          pendingSetup = {
            side: opportunity.side,
            setupTime: bar.timestamp,
            setupTimestamp: currentTime,
            setupPrice: bar.close,
            stopLoss: roundToTick(opportunity.stopLoss),
            tp1_POC: roundToTick(opportunity.tp1_POC),
            tp2_Edge: roundToTick(opportunity.tp2_Edge),
            bias: biasCalc.bias,
            reasoning: opportunity.reasoning,
            entryLimit: roundToTick(opportunity.entryLimit),
            rsi: currentRSI ?? null,
          };

          const rsiNote = pendingSetup.rsi !== null ? `, RSI ${pendingSetup.rsi.toFixed(1)}` : '';
          console.log(
            `[${bar.timestamp}] ${opportunity.side.toUpperCase()} setup armed @ ${pendingSetup.entryLimit.toFixed(2)} ` +
            `(${opportunity.reasoning}${rsiNote}, awaiting TTM Squeeze)`
          );
        }
      }

      // Stage 2: wait for TTM Squeeze trigger + limit touched before entering
      if (pendingSetup) {
        // Cancel setup if too old (60 seconds timeout)
        const setupAge = currentTime - pendingSetup.setupTimestamp;
        if (setupAge > 60000) {
          pendingSetup = null;
        } else if (ttmSqueeze.squeezeOn && timeSinceLastEntry >= MIN_ENTRY_INTERVAL_MS) {
          // Check if price touched the limit order
          const limitTouched = pendingSetup.side === 'long'
            ? bar.low <= pendingSetup.entryLimit
            : bar.high >= pendingSetup.entryLimit;

          if (limitTouched) {
          // Fill at limit price with minimal slippage (market-like execution)
          const entrySide = pendingSetup.side === 'long' ? 'buy' : 'sell';
          // Add just 0.1 tick slippage for realism
          const minimalSlip = tickSize * 0.1;
          const entryPrice = roundToTick(
            pendingSetup.side === 'long'
              ? pendingSetup.entryLimit + minimalSlip
              : pendingSetup.entryLimit - minimalSlip
          );
          lastEntryTime = currentTime;

          // Recalculate stop based on actual entry price
          const stopOffset = 0.05 * profileSnapshot.vaw;
          const actualStop = pendingSetup.side === 'long'
            ? roundToTick(entryPrice - stopOffset)
            : roundToTick(entryPrice + stopOffset);

          position = {
            side: pendingSetup.side,
            entryPrice,
            entryTime: bar.timestamp,
            stopLoss: actualStop,
            tp1_POC: pendingSetup.tp1_POC,
            tp2_Edge: pendingSetup.tp2_Edge,
            tp1Hit: false,
            feesPaid: addFees(baseSymbol, CONFIG.numberOfContracts),
            bias: pendingSetup.bias,
            reasoning: pendingSetup.reasoning,
          };

          const rsiLog = pendingSetup.rsi !== null ? `, RSI ${pendingSetup.rsi.toFixed(1)}` : '';
          console.log(
            `[${bar.timestamp}] ${pendingSetup.side.toUpperCase()} entry @ ${entryPrice.toFixed(2)} ` +
            `(${pendingSetup.reasoning}${rsiLog}, ` +
            `TTM Squeeze trigger, ` +
            `TP1: ${position.tp1_POC.toFixed(2)}, Stop: ${position.stopLoss.toFixed(2)})`
          );

          pendingSetup = null;
          }
        }
      }

    }

    // Close position at session end
    if (position) {
      const lastBar = sessionBars[sessionBars.length - 1];
      const direction = position.side === 'long' ? 1 : -1;
      const closeSide = position.side === 'long' ? 'sell' : 'buy';
      const sessionExitFill = roundToTick(fillStop(baseSymbol, closeSide, lastBar.close));

      const grossBeforeFees = (sessionExitFill - position.entryPrice) * direction * multiplier * CONFIG.numberOfContracts;
      const exitFees = addFees(baseSymbol, CONFIG.numberOfContracts);
      const totalFees = position.feesPaid + exitFees;
      const netPnl = grossBeforeFees - totalFees;

      trades.push({
        entryTime: position.entryTime,
        exitTime: lastBar.timestamp,
        side: position.side,
        entryPrice: position.entryPrice,
        exitPrice: sessionExitFill,
        entryFill: position.entryPrice,
        exitFill: sessionExitFill,
        pnl: netPnl,
        grossPnl: grossBeforeFees,
        grossBeforeFees,
        fees: totalFees,
        exitReason: 'session',
        bias: position.bias,
        entryReasoning: position.reasoning,
      });

      realizedPnL += netPnl;
      position = null;
    }

    // Reset any leftover pending setup at session close
    if (pendingSetup) {
      console.log(
        `[${sessionDate}] Clearing pending ${pendingSetup.side.toUpperCase()} setup at session end`
      );
      pendingSetup = null;
    }

    const finalProfile = calculateSessionProfile(sessionBars, tickSize);
    completedProfiles.push(finalProfile);
    console.log(
      `[${sessionDate}] Final Profile POC: ${finalProfile.poc.toFixed(2)}, ` +
      `VAH: ${finalProfile.vah.toFixed(2)}, VAL: ${finalProfile.val.toFixed(2)}, VAW: ${finalProfile.vaw.toFixed(2)}`
    );
  }

  // Calculate statistics
  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl <= 0);
  const winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;

  const avgWin = winningTrades.length
    ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length
    : 0;
  const avgLoss = losingTrades.length
    ? Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length)
    : 0;

  const grossProfit = trades.filter(t => t.grossPnl > 0).reduce((sum, t) => sum + t.grossPnl, 0);
  const grossLoss = Math.abs(trades.filter(t => t.grossPnl <= 0).reduce((sum, t) => sum + t.grossPnl, 0));
  const totalFees = trades.reduce((sum, t) => sum + t.fees, 0);
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

  let runningPnL = 0;
  let peakPnL = 0;
  let maxDrawdown = 0;

  trades.forEach(trade => {
    runningPnL += trade.pnl;
    if (runningPnL > peakPnL) peakPnL = runningPnL;
    const drawdown = peakPnL - runningPnL;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  });

  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total Trades: ${trades.length} | Wins: ${winningTrades.length} | Losses: ${losingTrades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(1)}%`);
  console.log(`Net Realized PnL: ${formatCurrency(realizedPnL)} USD | Fees Paid: $${totalFees.toFixed(2)}`);
  console.log(`Gross Profit: ${formatCurrency(grossProfit)} | Gross Loss: -${grossLoss.toFixed(2)}`);
  console.log(`Avg Win: ${formatCurrency(avgWin)} | Avg Loss: -${avgLoss.toFixed(2)}`);
  console.log(`Profit Factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);
  console.log(`Max Drawdown: -${maxDrawdown.toFixed(2)} USD`);

  // Exit reason breakdown
  const exitReasons = trades.reduce((acc, t) => {
    acc[t.exitReason] = (acc[t.exitReason] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  console.log(`\nExit Reasons: ${Object.entries(exitReasons).map(([r, c]) => `${r}=${c}`).join(', ')}`);

  // Bias breakdown
  const biasCounts = trades.reduce((acc, t) => {
    acc[t.bias] = (acc[t.bias] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  console.log(`Trades by Bias: ${Object.entries(biasCounts).map(([b, c]) => `${b}=${c}`).join(', ')}`);

  if (trades.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log('RECENT TRADES');
    console.log('='.repeat(80));
    trades.slice(-10).forEach(trade => {
      const rawStr = formatCurrency(trade.grossBeforeFees);
      const feeStr = `$${trade.fees.toFixed(2)}`;
      console.log(
        `${trade.side.toUpperCase().padEnd(5)} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ` +
        `${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | ${formatCurrency(trade.pnl).padStart(10)} ` +
        `(${trade.exitReason}) [${trade.bias}] raw=${rawStr} fees=${feeStr}`
      );
    });
  }

  if (process.env.TOPSTEPX_SVP_SAVE_TRADES === 'true') {
    const filename = `backtest_trades_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    fs.writeFileSync(filename, JSON.stringify(trades, null, 2));
    console.log(`\nFull trade list saved to ${filename}`);
  }
}

runBacktest().catch(err => {
  console.error('SVP backtest failed:', err);
  process.exit(1);
});
