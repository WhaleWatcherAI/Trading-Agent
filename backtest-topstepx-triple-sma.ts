#!/usr/bin/env tsx
/**
 * 7/23 SMA Crossover Backtest with RSI & ADX Confirmation (Topstep data)
 *
 * - Uses 1-minute bars with SMA(7) and SMA(23) crossover
 * - Entry signals:
 *     Long setup  : SMA(7) crosses above SMA(23)
 *     Short setup : SMA(7) crosses below SMA(23)
 * - RSI confirmation: RSI(14) > 50 for longs, < 50 for shorts
 * - ADX filter: ADX(14) > 25 to confirm trend strength
 * - Opens position on bar close when crossover present and session is open
 * - Fixed % stop (0.1%) and target (1.1%) applied from entry price
 */

import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

interface BacktestConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;
  stopLossPercent: number;
  takeProfitPercent: number;
  commissionPerSide: number;
  contractMultiplier?: number;
}

interface TradeRecord {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  exitReason: 'target' | 'stop' | 'signal' | 'session' | 'end_of_data';
}

const SMA_FAST = 7;
const SMA_SLOW = 23;
const RSI_PERIOD = 14;
const ADX_PERIOD = 14;
const RSI_THRESHOLD = 50;
const ADX_MIN_THRESHOLD = 25;

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;

const DEFAULT_TRIPLE_SYMBOL = process.env.TOPSTEPX_TRIPLE_SMA_SYMBOL || 'ESZ5';
const DEFAULT_TRIPLE_CONTRACT_ID = process.env.TOPSTEPX_TRIPLE_SMA_CONTRACT_ID;

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_TRIPLE_SYMBOL,
  contractId: DEFAULT_TRIPLE_CONTRACT_ID,
  start: process.env.TOPSTEPX_TRIPLE_SMA_START || '2025-11-01T00:00:00Z',
  end: process.env.TOPSTEPX_TRIPLE_SMA_END || '2025-11-05T23:59:59Z',
  stopLossPercent: Number(process.env.TOPSTEPX_TRIPLE_STOP_LOSS_PERCENT || '0.001'),
  takeProfitPercent: Number(process.env.TOPSTEPX_TRIPLE_TAKE_PROFIT_PERCENT || '0.011'),
  commissionPerSide: process.env.TOPSTEPX_TRIPLE_COMMISSION
    ? Number(process.env.TOPSTEPX_TRIPLE_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_TRIPLE_CONTRACT_ID, DEFAULT_TRIPLE_SYMBOL]),
  contractMultiplier: process.env.TOPSTEPX_TRIPLE_CONTRACT_MULTIPLIER
    ? Number(process.env.TOPSTEPX_TRIPLE_CONTRACT_MULTIPLIER)
    : undefined,
};

function toCentralTime(date: Date) {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isTradingAllowed(timestamp: string | Date) {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const ctDate = toCentralTime(date);
  const day = ctDate.getUTCDay();
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  if (day === 6) return false;
  if (day === 0 && minutes < WEEKEND_REOPEN_MINUTES) return false;
  if (day === 5 && minutes >= CUT_OFF_MINUTES) return false;

  return minutes < CUT_OFF_MINUTES || minutes >= REOPEN_MINUTES;
}

interface SMAState {
  period: number;
  queue: number[];
  sum: number;
  current: number | null;
}

function createSMAState(period: number): SMAState {
  return { period, queue: [], sum: 0, current: null };
}

function updateSMA(state: SMAState, price: number): number | null {
  state.queue.push(price);
  state.sum += price;
  if (state.queue.length > state.period) {
    state.sum -= state.queue.shift() || 0;
  }
  if (state.queue.length === state.period) {
    state.current = state.sum / state.period;
  } else {
    state.current = null;
  }
  return state.current;
}

interface RSIState {
  period: number;
  prices: number[];
  avgGain: number;
  avgLoss: number;
  current: number | null;
}

function createRSIState(period: number): RSIState {
  return { period, prices: [], avgGain: 0, avgLoss: 0, current: null };
}

function updateRSI(state: RSIState, price: number): number | null {
  state.prices.push(price);

  if (state.prices.length < 2) {
    return null;
  }

  const change = price - state.prices[state.prices.length - 2];
  const gain = change > 0 ? change : 0;
  const loss = change < 0 ? -change : 0;

  if (state.prices.length <= state.period + 1) {
    // Initial period - calculate simple averages
    if (state.prices.length === state.period + 1) {
      let totalGain = 0;
      let totalLoss = 0;
      for (let i = 1; i < state.prices.length; i++) {
        const chg = state.prices[i] - state.prices[i - 1];
        if (chg > 0) totalGain += chg;
        else totalLoss += -chg;
      }
      state.avgGain = totalGain / state.period;
      state.avgLoss = totalLoss / state.period;
    } else {
      return null;
    }
  } else {
    // Smoothed averages
    state.avgGain = (state.avgGain * (state.period - 1) + gain) / state.period;
    state.avgLoss = (state.avgLoss * (state.period - 1) + loss) / state.period;
  }

  if (state.avgLoss === 0) {
    state.current = 100;
  } else {
    const rs = state.avgGain / state.avgLoss;
    state.current = 100 - (100 / (1 + rs));
  }

  return state.current;
}

interface ADXState {
  period: number;
  bars: { high: number; low: number; close: number }[];
  plusDM: number;
  minusDM: number;
  atr: number;
  adx: number | null;
  adxValues: number[];
}

function createADXState(period: number): ADXState {
  return {
    period,
    bars: [],
    plusDM: 0,
    minusDM: 0,
    atr: 0,
    adx: null,
    adxValues: [],
  };
}

function updateADX(state: ADXState, bar: { high: number; low: number; close: number }): number | null {
  state.bars.push(bar);

  if (state.bars.length < 2) {
    return null;
  }

  const prevBar = state.bars[state.bars.length - 2];
  const currBar = state.bars[state.bars.length - 1];

  const highDiff = currBar.high - prevBar.high;
  const lowDiff = prevBar.low - currBar.low;
  const plusDMCurr = highDiff > lowDiff && highDiff > 0 ? highDiff : 0;
  const minusDMCurr = lowDiff > highDiff && lowDiff > 0 ? lowDiff : 0;

  const tr = Math.max(
    currBar.high - currBar.low,
    Math.abs(currBar.high - prevBar.close),
    Math.abs(currBar.low - prevBar.close)
  );

  if (state.bars.length <= state.period + 1) {
    // Initial period - calculate simple averages
    if (state.bars.length === state.period + 1) {
      let totalPlusDM = 0;
      let totalMinusDM = 0;
      let totalTR = 0;

      for (let i = 1; i < state.bars.length; i++) {
        const prev = state.bars[i - 1];
        const curr = state.bars[i];
        const hDiff = curr.high - prev.high;
        const lDiff = prev.low - curr.low;
        totalPlusDM += hDiff > lDiff && hDiff > 0 ? hDiff : 0;
        totalMinusDM += lDiff > hDiff && lDiff > 0 ? lDiff : 0;
        totalTR += Math.max(
          curr.high - curr.low,
          Math.abs(curr.high - prev.close),
          Math.abs(curr.low - prev.close)
        );
      }

      state.plusDM = totalPlusDM;
      state.minusDM = totalMinusDM;
      state.atr = totalTR;
    } else {
      return null;
    }
  } else {
    // Smoothed values
    state.plusDM = state.plusDM - (state.plusDM / state.period) + plusDMCurr;
    state.minusDM = state.minusDM - (state.minusDM / state.period) + minusDMCurr;
    state.atr = state.atr - (state.atr / state.period) + tr;
  }

  if (state.atr === 0) {
    return null;
  }

  const plusDI = (state.plusDM / state.atr) * 100;
  const minusDI = (state.minusDM / state.atr) * 100;
  const diSum = plusDI + minusDI;

  if (diSum === 0) {
    return null;
  }

  const dx = (Math.abs(plusDI - minusDI) / diSum) * 100;
  state.adxValues.push(dx);

  if (state.adxValues.length < state.period) {
    return null;
  }

  if (state.adxValues.length === state.period) {
    state.adx = state.adxValues.reduce((sum, val) => sum + val, 0) / state.period;
  } else {
    state.adx = ((state.adx! * (state.period - 1)) + dx) / state.period;
  }

  return state.adx;
}

function formatCurrency(value: number) {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

function resolvePointValue(
  metadata: { tickValue?: number; tickSize?: number; multiplier?: number },
  fallback?: number,
) {
  if (metadata.tickValue && metadata.tickSize) {
    return metadata.tickValue / metadata.tickSize;
  }
  if (metadata.multiplier) {
    return metadata.multiplier;
  }
  if (fallback) {
    return fallback;
  }
  return 50; // default ES-style point value
}

async function fetchBars(contractId: string, start: string, end: string): Promise<TopstepXFuturesBar[]> {
  const bars = await fetchTopstepXFuturesBars({
    contractId,
    startTime: start,
    endTime: end,
    unit: 2,
    unitNumber: 1,
  });
  if (!bars.length) {
    throw new Error('No 1-minute bars returned for requested period.');
  }
  return bars.reverse(); // API returns newest-first
}

async function runTripleSMABacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('TOPSTEPX 7/23 SMA CROSSOVER + RSI + ADX BACKTEST (1-MIN BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Date range: ${CONFIG.start} -> ${CONFIG.end}`);
  console.log(`Strategy: SMA(${SMA_FAST})/SMA(${SMA_SLOW}) crossover`);
  console.log(`Stop: ${(CONFIG.stopLossPercent * 100).toFixed(2)}% | Target: ${(CONFIG.takeProfitPercent * 100).toFixed(2)}%`);
  console.log(`RSI(${RSI_PERIOD}) filter: Long>${RSI_THRESHOLD} Short<${RSI_THRESHOLD}`);
  console.log(`ADX(${ADX_PERIOD}) filter: Minimum ${ADX_MIN_THRESHOLD}`);
  console.log(`Commission/side: ${CONFIG.commissionPerSide.toFixed(2)} USD`);
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey);
  if (!metadata) {
    throw new Error(`Unable to resolve Topstep contract metadata for ${lookupKey}`);
  }

  const contractId = metadata.id;
  const pointValue = resolvePointValue(metadata, CONFIG.contractMultiplier);
  console.log(`Resolved contract: ${metadata.name} (${contractId})`);
  console.log(`Point value: ${pointValue}`);

  const bars = await fetchBars(contractId, CONFIG.start, CONFIG.end);
  console.log(`Loaded ${bars.length} one-minute bars`);

  const smaFast = createSMAState(SMA_FAST);
  const smaSlow = createSMAState(SMA_SLOW);
  const rsiState = createRSIState(RSI_PERIOD);
  const adxState = createADXState(ADX_PERIOD);
  const trades: TradeRecord[] = [];

  let prevSmaFast: number | null = null;
  let prevSmaSlow: number | null = null;
  let position: {
    side: 'long' | 'short';
    entryPrice: number;
    entryTime: string;
    stop: number;
    target: number;
  } | null = null;
  let realizedPnL = 0;

  const exitPosition = (
    exitPrice: number,
    exitTime: string,
    reason: TradeRecord['exitReason'],
  ) => {
    if (!position) return;
    const direction = position.side === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - position.entryPrice) * direction * pointValue;
    const commissionCost = CONFIG.commissionPerSide * 2;
    const pnl = rawPnL - commissionCost;
    trades.push({
      entryTime: position.entryTime,
      exitTime,
      side: position.side,
      entryPrice: position.entryPrice,
      exitPrice,
      pnl,
      exitReason: reason,
    });
    realizedPnL += pnl;
    position = null;
  };

  for (const bar of bars) {
    const currSmaFast = updateSMA(smaFast, bar.close);
    const currSmaSlow = updateSMA(smaSlow, bar.close);
    const rsi = updateRSI(rsiState, bar.close);
    const adx = updateADX(adxState, bar);

    const allReady = currSmaFast !== null && currSmaSlow !== null && rsi !== null && adx !== null;
    if (!allReady) {
      prevSmaFast = currSmaFast;
      prevSmaSlow = currSmaSlow;
      continue;
    }

    if (position && isTradingAllowed(bar.timestamp)) {
      if (position.side === 'long') {
        if (bar.high >= position.target) {
          exitPosition(position.target, bar.timestamp, 'target');
        } else if (bar.low <= position.stop) {
          exitPosition(position.stop, bar.timestamp, 'stop');
        }
      } else if (position.side === 'short') {
        if (bar.low <= position.target) {
          exitPosition(position.target, bar.timestamp, 'target');
        } else if (bar.high >= position.stop) {
          exitPosition(position.stop, bar.timestamp, 'stop');
        }
      }
    }

    if (position && !isTradingAllowed(bar.timestamp)) {
      exitPosition(bar.close, bar.timestamp, 'session');
    }

    if (position || !isTradingAllowed(bar.timestamp)) {
      prevSmaFast = currSmaFast;
      prevSmaSlow = currSmaSlow;
      continue;
    }

    // Detect crossovers
    let bullishCross = false;
    let bearishCross = false;

    if (prevSmaFast !== null && prevSmaSlow !== null) {
      // Bullish crossover: fast crosses above slow
      bullishCross = prevSmaFast <= prevSmaSlow && currSmaFast > currSmaSlow;
      // Bearish crossover: fast crosses below slow
      bearishCross = prevSmaFast >= prevSmaSlow && currSmaFast < currSmaSlow;
    }

    // RSI and ADX confirmation filters
    const rsiLongConfirmed = rsi > RSI_THRESHOLD;
    const rsiShortConfirmed = rsi < RSI_THRESHOLD;
    const adxConfirmed = adx > ADX_MIN_THRESHOLD;

    const longSetup = bullishCross && rsiLongConfirmed && adxConfirmed;
    const shortSetup = bearishCross && rsiShortConfirmed && adxConfirmed;

    if (longSetup || shortSetup) {
      const side: 'long' | 'short' = longSetup ? 'long' : 'short';
      const entryPrice = bar.close;
      const stop = side === 'long'
        ? entryPrice * (1 - CONFIG.stopLossPercent)
        : entryPrice * (1 + CONFIG.stopLossPercent);
      const target = side === 'long'
        ? entryPrice * (1 + CONFIG.takeProfitPercent)
        : entryPrice * (1 - CONFIG.takeProfitPercent);

      position = {
        side,
        entryPrice,
        entryTime: bar.timestamp,
        stop,
        target,
      };
    }

    prevSmaFast = currSmaFast;
    prevSmaSlow = currSmaSlow;
  }

  if (position) {
    exitPosition(bars[bars.length - 1].close, bars[bars.length - 1].timestamp, 'end_of_data');
  }

  const wins = trades.filter(trade => trade.pnl > 0);
  const losses = trades.filter(trade => trade.pnl < 0);
  const avgWin = wins.length ? wins.reduce((sum, t) => sum + t.pnl, 0) / wins.length : 0;
  const avgLoss = losses.length ? losses.reduce((sum, t) => sum + t.pnl, 0) / losses.length : 0;

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

  console.log('\n===== 7/23 SMA Crossover Backtest Summary =====');
  console.log(`Trades: ${trades.length} | Win rate: ${trades.length ? ((wins.length / trades.length) * 100).toFixed(1) : '0'}%`);
  console.log(`Realized PnL: ${formatCurrency(realizedPnL)} USD`);
  console.log(`Avg Win: ${formatCurrency(avgWin)} | Avg Loss: ${formatCurrency(avgLoss)}`);
  console.log(`Max Drawdown: ${formatCurrency(maxDrawdown)} USD`);

  if (trades.length) {
    console.log('\nRecent trades:');
    trades.slice(-5).forEach(trade => {
      console.log(
        ` - ${trade.side.toUpperCase()} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | ${formatCurrency(trade.pnl)} (${trade.exitReason})`,
      );
    });
  }
}

runTripleSMABacktest().catch(err => {
  console.error('Triple SMA backtest failed:', err);
  process.exit(1);
});
