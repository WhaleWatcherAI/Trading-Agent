import path from 'path';
import { promises as fs } from 'fs';
import { analyzeVolatilityRegime } from './regimeAgent';
import { getHistoricalTimesales, TradierTimesaleBar } from './tradier';
import {
  RegimeTradeSignal,
  VolatilityRegimeAnalysis,
} from '@/types';

/**
 * Regime Backtester V2
 *
 * Uses the EXACT same strategy as regimeAgent.ts by:
 * 1. Running analyzeVolatilityRegime() to get trade signals
 * 2. Fetching historical prices for the backtest period
 * 3. Simulating trades based on regime signals
 */

export interface RegimeBacktestV2Config {
  date: string;
  symbols?: string[];
  mode?: 'scalp' | 'swing' | 'leaps';
  intervalMinutes?: number;
}

export interface BacktestV2Trade {
  id: string;
  symbol: string;
  direction: 'long' | 'short';
  entryTimestamp: string;
  exitTimestamp: string;
  entryPrice: number;
  exitPrice: number;
  entrySignal: RegimeTradeSignal;
  exitReason: 'stop' | 'target' | 'secondary_target' | 'time_stop';
  profit: number;
  profitPct: number;
  durationMinutes: number;
}

export interface BacktestV2Result {
  date: string;
  mode: 'scalp' | 'swing' | 'leaps';
  symbols: string[];
  trades: BacktestV2Trade[];
  regimeAnalyses: VolatilityRegimeAnalysis[];
  metrics: {
    totalTrades: number;
    winRate: number;
    totalProfit: number;
    averageDuration: number;
    maxDrawdown: number;
  };
  notes: string[];
}

/**
 * Run regime strategy backtest
 *
 * This fetches CURRENT regime analysis (GEX, whale flow, etc.)
 * and simulates trades using HISTORICAL prices.
 *
 * Note: This is a "hybrid" backtest - it uses current market structure
 * (options chains, whale flow) with historical prices. For true historical
 * backtesting, you'd need historical options data.
 */
export async function runRegimeBacktestV2(
  config: RegimeBacktestV2Config,
): Promise<BacktestV2Result> {
  const { date, mode = 'scalp', intervalMinutes = 1, symbols: configSymbols } = config;
  const notes: string[] = [];

  notes.push(`Running regime strategy backtest for ${date}`);
  notes.push(`Fetching whale flow to build universe for ${date}...`);

  // Step 1: Run regime analysis to get trade signals
  // Pass the backtest date to fetch historical whale flow and volatility data
  const regimeResponse = await analyzeVolatilityRegime({ mode, date, symbols: configSymbols });

  const symbols = regimeResponse.symbols || [];
  notes.push(`Stage 1 Universe: ${symbols.length} tickers from whale flow`);

  // TEMPORARY: Skip Stage 1 filtering, use all analyses
  const analyses: VolatilityRegimeAnalysis[] = regimeResponse.analyses;

  if (analyses.length === 0) {
    notes.push('No tickers with analysis - no trades generated');
    return {
      date,
      mode,
      symbols,
      trades: [],
      regimeAnalyses: [],
      metrics: {
        totalTrades: 0,
        winRate: 0,
        totalProfit: 0,
        averageDuration: 0,
        maxDrawdown: 0,
      },
      notes,
    };
  }

  notes.push(`TEMPORARY: Skipping Stage 1 filter, processing first ${analyses.length} tickers`);

  // Step 2: Fetch historical prices for the backtest period
  const pricesMap = new Map<string, TradierTimesaleBar[]>();
  for (const analysis of analyses) {
    try {
      const bars = await getHistoricalTimesales(analysis.symbol, date, intervalMinutes);
      pricesMap.set(analysis.symbol, bars);
      notes.push(`Fetched ${bars.length} price bars for ${analysis.symbol}`);
    } catch (error: any) {
      notes.push(`Failed to fetch prices for ${analysis.symbol}: ${error.message}`);
    }
  }

  // Step 3: Simulate trades from regime signals
  const trades: BacktestV2Trade[] = [];

  for (const analysis of analyses) {
    const signals = analysis.tradeSignals || [];
    const bars = pricesMap.get(analysis.symbol) || [];

    if (signals.length === 0 || bars.length === 0) {
      continue;
    }

    notes.push(`Processing ${signals.length} signals for ${analysis.symbol}`);

    // For each signal, simulate the trade using historical prices
    for (const signal of signals) {
      const trade = simulateTradeFromSignal(signal, bars, analysis.symbol, date);
      if (trade) {
        trades.push(trade);
      }
    }
  }

  // Step 4: Calculate metrics
  const totalTrades = trades.length;
  const wins = trades.filter(t => t.profit > 0).length;
  const winRate = totalTrades > 0 ? wins / totalTrades : 0;
  const totalProfit = trades.reduce((sum, t) => sum + t.profit, 0);
  const averageDuration = totalTrades > 0
    ? trades.reduce((sum, t) => sum + t.durationMinutes, 0) / totalTrades
    : 0;

  let maxDrawdown = 0;
  let peak = 0;
  let equity = 0;
  trades.forEach(trade => {
    equity += trade.profit;
    peak = Math.max(peak, equity);
    maxDrawdown = Math.max(maxDrawdown, peak - equity);
  });

  return {
    date,
    mode,
    symbols,
    trades,
    regimeAnalyses: analyses,
    metrics: {
      totalTrades,
      winRate,
      totalProfit,
      averageDuration,
      maxDrawdown,
    },
    notes,
  };
}

/**
 * Simulate a single trade from a regime signal using historical prices
 */
function simulateTradeFromSignal(
  signal: RegimeTradeSignal,
  bars: TradierTimesaleBar[],
  symbol: string,
  date: string,
): BacktestV2Trade | null {
  if (bars.length === 0) return null;

  // Entry: Wait for price to hit trigger level (logic varies by trigger type)
  let entryBar: TradierTimesaleBar | null = null;
  let entryPrice: number;
  const triggerLevel = signal.entry.triggerLevel;
  const triggerType = signal.entry.triggerType;

  // Scan through bars to find entry trigger
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];

    if (signal.direction === 'long') {
      if (triggerType === 'breakout') {
        // EXPANSION BREAKOUT: Enter when price breaks ABOVE trigger
        if (bar.high >= triggerLevel) {
          entryBar = bar;
          entryPrice = Math.max(triggerLevel, bar.open);
          break;
        }
      } else if (triggerType === 'range-reversion') {
        // PINNING REVERSION: Enter when price tests AT/BELOW trigger (buy the dip)
        if (bar.low <= triggerLevel * 1.01) { // Within 1% of trigger (widened from 0.2%)
          entryBar = bar;
          entryPrice = Math.min(triggerLevel, bar.close);
          break;
        }
      }
    } else {
      // SHORT trades
      if (triggerType === 'fade') {
        // PINNING FADE: Enter when price tests AT/ABOVE trigger (short the rip)
        if (bar.high >= triggerLevel * 0.99) { // Within 1% of trigger (widened from 0.2%)
          entryBar = bar;
          entryPrice = Math.max(triggerLevel, bar.close);
          break;
        }
      } else if (triggerType === 'breakdown') {
        // EXPANSION BREAKDOWN: Enter when price breaks BELOW trigger
        if (bar.low <= triggerLevel) {
          entryBar = bar;
          entryPrice = Math.min(triggerLevel, bar.open);
          break;
        }
      }
    }
  }

  // If trigger never hit, no trade
  if (!entryBar) return null;

  const entryTimestamp = entryBar.time;
  const entryBarIndex = bars.indexOf(entryBar);

  // Determine exit based on stop loss and targets
  let exitBar: TradierTimesaleBar | null = null;
  let exitReason: BacktestV2Trade['exitReason'] = 'time_stop';

  const stopLoss = signal.stopLoss;
  const firstTarget = signal.firstTarget;
  const secondaryTarget = signal.secondaryTarget;

  // Time-based exit: For scalps, exit after 2 hours (4x the timeframe)
  const maxDurationMinutes = signal.timeframeMinutes ? signal.timeframeMinutes * 4 : 240;
  const entryTime = new Date(entryTimestamp).getTime();

  // Scan through bars AFTER entry to find exit
  for (let i = entryBarIndex + 1; i < bars.length; i++) {
    const bar = bars[i];

    // Check if max duration exceeded
    const currentTime = new Date(bar.time).getTime();
    const durationMinutes = Math.round((currentTime - entryTime) / 60000);
    if (durationMinutes >= maxDurationMinutes) {
      exitBar = bar;
      exitReason = 'time_stop';
      break;
    }
    const high = bar.high;
    const low = bar.low;

    if (signal.direction === 'long') {
      // Check stop loss (hit low)
      if (low <= stopLoss) {
        exitBar = bar;
        exitReason = 'stop';
        break;
      }
      // Check secondary target first
      if (secondaryTarget && high >= secondaryTarget) {
        exitBar = bar;
        exitReason = 'secondary_target';
        break;
      }
      // Check first target
      if (high >= firstTarget) {
        exitBar = bar;
        exitReason = 'target';
        break;
      }
    } else {
      // Short trade
      // Check stop loss (hit high)
      if (high >= stopLoss) {
        exitBar = bar;
        exitReason = 'stop';
        break;
      }
      // Check secondary target first
      if (secondaryTarget && low <= secondaryTarget) {
        exitBar = bar;
        exitReason = 'secondary_target';
        break;
      }
      // Check first target
      if (low <= firstTarget) {
        exitBar = bar;
        exitReason = 'target';
        break;
      }
    }
  }

  // If no exit found, use last bar as time stop
  if (!exitBar) {
    exitBar = bars[bars.length - 1];
    exitReason = 'time_stop';
  }

  const exitPrice = signal.direction === 'long'
    ? (exitReason === 'stop' ? stopLoss : exitReason === 'target' ? firstTarget : exitReason === 'secondary_target' ? secondaryTarget! : exitBar.close)
    : (exitReason === 'stop' ? stopLoss : exitReason === 'target' ? firstTarget : exitReason === 'secondary_target' ? secondaryTarget! : exitBar.close);

  const exitTimestamp = exitBar.time;

  const profit = signal.direction === 'long'
    ? exitPrice - entryPrice
    : entryPrice - exitPrice;

  const profitPct = profit / entryPrice;

  const durationMinutes = Math.round(
    (new Date(exitTimestamp).getTime() - new Date(entryTimestamp).getTime()) / 60000,
  );

  return {
    id: `${symbol}-${entryTimestamp}`,
    symbol,
    direction: signal.direction,
    entryTimestamp,
    exitTimestamp,
    entryPrice,
    exitPrice,
    entrySignal: signal,
    exitReason,
    profit,
    profitPct,
    durationMinutes,
  };
}
