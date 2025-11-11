import { RSI, ADX } from 'technicalindicators';
import * as fs from 'fs';
import * as path from 'path';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';
import { calculateTtmSqueeze } from './lib/ttmSqueeze';

interface BacktestConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;
  rsiPeriod: number;
  adxPeriod: number;
  adxThreshold: number;
  bypassAdx: boolean;
  stopLossPercent: number;
  takeProfitPercent: number;
  commissionPerSide: number;
  contractMultiplier: number;
  numberOfContracts: number;
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
  slippageCost: number;
  entrySlippageTicks: number;
  exitSlippageTicks: number;
  exitReason: 'target' | 'stop' | 'signal' | 'session' | 'end_of_data';
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_DAYS = 5;

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

function tpCostTicks(sym: string): number {
  const spreadTicks = SLIP_CONFIG.avgSpreadTicks[sym] ?? 0;
  const sigmaTicks = SLIP_CONFIG.slipAvg.tp[sym] ?? 0;
  const pPassive = SLIP_CONFIG.p_tp_passive[sym] ?? 1;
  return (1 - pPassive) * (spreadTicks + sigmaTicks);
}

function fillEntry(sym: string, side: 'buy' | 'sell', mid: number): number {
  const t = SLIP_CONFIG.tickSize[sym] ?? 0;
  const spreadTicks = SLIP_CONFIG.avgSpreadTicks[sym] ?? 0;
  const sigmaTicks = SLIP_CONFIG.slipAvg.entry[sym] ?? 0;
  const halfSpread = 0.5 * spreadTicks * t;
  const sigma = sigmaTicks * t;
  return side === 'buy' ? mid + halfSpread + sigma : mid - halfSpread - sigma;
}

function fillTP(sym: string, side: 'buy' | 'sell', mid: number): number {
  const t = SLIP_CONFIG.tickSize[sym] ?? 0;
  const tpTicks = tpCostTicks(sym);
  return side === 'sell' ? mid - tpTicks * t : mid + tpTicks * t;
}

function fillStop(sym: string, side: 'buy' | 'sell', triggerMid: number): number {
  const t = SLIP_CONFIG.tickSize[sym] ?? 0;
  const sigmaTicks = SLIP_CONFIG.slipAvg.stop[sym] ?? 0;
  const sigma = sigmaTicks * t;
  return side === 'buy' ? triggerMid + sigma : triggerMid - sigma;
}

const AVAILABLE_SLIP_SYMBOLS = Object.keys(SLIP_CONFIG.tickSize);

function getBaseSymbol(fullSymbol: string): string {
  return fullSymbol.replace(/[A-Z]\d+$/, '');
}

function resolveSlipSymbol(symbol: string): string {
  if (SLIP_CONFIG.tickSize[symbol]) {
    return symbol;
  }
  const exactMatch = AVAILABLE_SLIP_SYMBOLS.find(entry => symbol.startsWith(entry));
  return exactMatch ?? 'ES';
}

const DEFAULT_SR_SYMBOL = process.env.TOPSTEPX_SR_SYMBOL || 'ESZ5';
const DEFAULT_SR_CONTRACT_ID = process.env.TOPSTEPX_SR_CONTRACT_ID;

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_SR_SYMBOL,
  contractId: DEFAULT_SR_CONTRACT_ID,
  start: process.env.TOPSTEPX_SR_START
    || new Date(Date.now() - DEFAULT_DAYS * DAY_MS).toISOString(),
  end: process.env.TOPSTEPX_SR_END || new Date().toISOString(),
  rsiPeriod: Number(process.env.TOPSTEPX_SR_RSI_PERIOD || '24'),
  adxPeriod: Number(process.env.TOPSTEPX_SR_ADX_PERIOD || '24'),
  adxThreshold: Number(process.env.TOPSTEPX_SR_ADX_THRESHOLD || '25'),
  bypassAdx: process.env.TOPSTEPX_SR_BYPASS_ADX === 'true',
  stopLossPercent: Number(process.env.TOPSTEPX_SR_STOP_LOSS_PERCENT || '0.0009'),
  takeProfitPercent: Number(process.env.TOPSTEPX_SR_TAKE_PROFIT_PERCENT || '0'),
  commissionPerSide: process.env.TOPSTEPX_SR_COMMISSION
    ? Number(process.env.TOPSTEPX_SR_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_SR_CONTRACT_ID, DEFAULT_SR_SYMBOL]),
  contractMultiplier: Number(process.env.TOPSTEPX_SR_CONTRACT_MULTIPLIER || '50'),
  numberOfContracts: Number(process.env.TOPSTEPX_SR_CONTRACTS || '5'),
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

function formatCurrency(value: number) {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

async function fetchSecondBarsInChunks(
  contractId: string,
  start: string,
  end: string,
): Promise<TopstepXFuturesBar[]> {
  const startDate = new Date(start);
  const endDate = new Date(end);
  const bars: TopstepXFuturesBar[] = [];
  let cursor = startDate;
  const SECOND_CHUNK_MS = 4 * 60 * 60 * 1000; // 4 hours

  while (cursor < endDate) {
    const chunkEnd = new Date(Math.min(cursor.getTime() + SECOND_CHUNK_MS, endDate.getTime()));
    console.log(`Fetching 1s bars from ${cursor.toISOString()} to ${chunkEnd.toISOString()}`);
    const chunkBars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: cursor.toISOString(),
      endTime: chunkEnd.toISOString(),
      unit: 1,
      unitNumber: 1,
      limit: 20000, // Use the API's max limit per call
    });
    bars.push(...chunkBars);
    cursor = new Date(chunkEnd.getTime() + 1_000); // Move cursor past the last fetched bar
    await new Promise(resolve => setTimeout(resolve, 1000)); // Add a delay to avoid hitting API rate limits
  }

  return bars.reverse();
}

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('TOPSTEPX SECOND-BAR S/R BACKTEST');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end} (~${DEFAULT_DAYS} days)`);
  console.log(`RSI Period: ${CONFIG.rsiPeriod}`);
  console.log(`ADX Period: ${CONFIG.adxPeriod} (threshold ${CONFIG.adxThreshold}; bypass ${CONFIG.bypassAdx ? 'enabled' : 'disabled'})`);
  console.log(`Commission/side: ${CONFIG.commissionPerSide.toFixed(2)} USD | Contracts: ${CONFIG.numberOfContracts}`);
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[topstepx-sr] Unable to fetch metadata:', err.message);
    return null;
  });
  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  const contractId = metadata.id;
  const baseSymbol = getBaseSymbol(CONFIG.symbol);
  const slipSymbol = resolveSlipSymbol(baseSymbol);
  const tickSize = metadata.tickSize ?? SLIP_CONFIG.tickSize[slipSymbol];
  if (!tickSize || !Number.isFinite(tickSize) || tickSize <= 0) {
    throw new Error(`Invalid tick size for ${lookupKey}`);
  }
  const multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || CONFIG.contractMultiplier;
  const roundToTick = (value: number) => Math.round(value / tickSize) * tickSize;
  const entrySlipTicksForSymbol =
    0.5 * (SLIP_CONFIG.avgSpreadTicks[slipSymbol] ?? 0) + (SLIP_CONFIG.slipAvg.entry[slipSymbol] ?? 0);
  const stopSlippageTicks = SLIP_CONFIG.slipAvg.stop[slipSymbol] ?? 0;
  const contracts = CONFIG.numberOfContracts;

  console.log(`Resolved contract: ${metadata.name} (${contractId})`);
  console.log(`Point multiplier: ${multiplier} | Tick size: ${tickSize} | Slip symbol: ${slipSymbol}`);

  console.log('\nFetching 1-second bars in chunks...');
  const bars = await fetchSecondBarsInChunks(contractId, CONFIG.start, CONFIG.end);

  if (!bars.length) {
    throw new Error('No second bars returned for configured window.');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length} one-second bars`);
  const closes: number[] = [];
  const highs: number[] = [];
  const lows: number[] = [];
  let position: 'long' | 'short' | null = null;
  let entryPrice = 0;
  let entryTime = '';
  let entrySlippageTicks = 0;
  const trades: TradeRecord[] = [];
  let realizedPnL = 0;

  const rsiIndicator = new RSI({ period: CONFIG.rsiPeriod, values: [] });
  const adxIndicator = new ADX({
    period: CONFIG.adxPeriod,
    high: [],
    low: [],
    close: [],
  });
  let prevRsiValue: number | null = null;

  // --- 5-Min Bar Aggregation and Pivot Point Calculation ---
  const FIVE_MINUTES_MS = 5 * 60 * 1000;
  let fiveMinBars: TopstepXFuturesBar[] = [];
  let currentFiveMinBar: TopstepXFuturesBar | null = null;
  let lastFiveMinBar: TopstepXFuturesBar | null = null; // To store the last completed 5-min bar for pivot calculation

  interface PivotPoints {
    P: number;
    R1: number;
    S1: number;
    R2: number;
    S2: number;
  }

  let currentPivotPoints: PivotPoints | null = null;

  function calculatePivotPoints(prevBar: TopstepXFuturesBar): PivotPoints {
    const P = (prevBar.high + prevBar.low + prevBar.close) / 3;
    const R1 = (2 * P) - prevBar.low;
    const S1 = (2 * P) - prevBar.high;
    const R2 = P + (prevBar.high - prevBar.low);
    const S2 = P - (prevBar.high - prevBar.low);
    return { P, R1, S1, R2, S2 };
  }

  const exitPosition = (
    exitPrice: number,
    exitTime: string,
    reason: TradeRecord['exitReason'],
    exitSlippageTicks: number,
  ) => {
    if (!position) return;
    const direction = position === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - entryPrice) * direction * multiplier * contracts;
    const fees = CONFIG.commissionPerSide * 2 * contracts;
    const entrySlippagePoints = entrySlippageTicks * tickSize;
    const exitSlippagePoints = exitSlippageTicks * tickSize;
    const slippageCost = (entrySlippagePoints + exitSlippagePoints) * multiplier * contracts;
    const netPnL = rawPnL - fees - slippageCost;
    trades.push({
      entryTime,
      exitTime,
      side: position,
      entryPrice,
      exitPrice,
      pnl: netPnL,
      grossPnl: rawPnL,
      fees,
      slippageCost,
      entrySlippageTicks,
      exitSlippageTicks,
      exitReason: reason,
    });
    realizedPnL += netPnL;
    position = null;
    entryPrice = 0;
    entryTime = '';
    entrySlippageTicks = 0;
  };

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    closes.push(bar.close);
    highs.push(bar.high);
    lows.push(bar.low);

    // --- 5-Min Bar Aggregation ---
    const barTime = new Date(bar.timestamp).getTime();
    const fiveMinWindowStart = Math.floor(barTime / FIVE_MINUTES_MS) * FIVE_MINUTES_MS;

    if (!currentFiveMinBar || new Date(currentFiveMinBar.timestamp).getTime() < fiveMinWindowStart) {
      // A new 5-minute bar window has started
      if (currentFiveMinBar) {
        // Complete the previous 5-minute bar
        fiveMinBars.push(currentFiveMinBar);
        lastFiveMinBar = currentFiveMinBar;
        // Calculate new pivot points for the next 5-min period
        currentPivotPoints = calculatePivotPoints(lastFiveMinBar);
      }
      // Start a new 5-minute bar
      currentFiveMinBar = {
        timestamp: new Date(fiveMinWindowStart).toISOString(),
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        volume: bar.volume,
      };
    } else {
      // Update current 5-minute bar
      currentFiveMinBar.high = Math.max(currentFiveMinBar.high, bar.high);
      currentFiveMinBar.low = Math.min(currentFiveMinBar.low, bar.low);
      currentFiveMinBar.close = bar.close;
      currentFiveMinBar.volume = (currentFiveMinBar.volume ?? 0) + (bar.volume ?? 0);
    }

    if (!isTradingAllowed(bar.timestamp)) {
      if (position) {
        const exitSide = position === 'long' ? 'sell' : 'buy';
        const exitPrice = roundToTick(fillStop(slipSymbol, exitSide, bar.close));
        exitPosition(exitPrice, bar.timestamp, 'session', stopSlippageTicks);
      }
      continue;
    }

    if (position) {
      const direction = position === 'long' ? 1 : -1;
      const target = direction === 1
        ? entryPrice * (1 + CONFIG.takeProfitPercent)
        : entryPrice * (1 - CONFIG.takeProfitPercent);
      const stop = direction === 1
        ? entryPrice * (1 - CONFIG.stopLossPercent)
        : entryPrice * (1 + CONFIG.stopLossPercent);
      const hasTarget = CONFIG.takeProfitPercent > 0;
      const hasStop = CONFIG.stopLossPercent > 0;

      if (direction === 1 && hasTarget && bar.high >= target) {
        const basePrice = Math.min(bar.high, target);
        const exitPrice = roundToTick(fillTP(slipSymbol, 'sell', basePrice));
        exitPosition(exitPrice, bar.timestamp, 'target', tpCostTicks(slipSymbol));
        continue;
      } else if (direction === 1 && hasStop && bar.low <= stop) {
        const exitPrice = roundToTick(fillStop(slipSymbol, 'sell', stop));
        exitPosition(exitPrice, bar.timestamp, 'stop', stopSlippageTicks);
        continue;
      } else if (direction === -1 && hasTarget && bar.low <= target) {
        const basePrice = Math.max(bar.low, target);
        const exitPrice = roundToTick(fillTP(slipSymbol, 'buy', basePrice));
        exitPosition(exitPrice, bar.timestamp, 'target', tpCostTicks(slipSymbol));
        continue;
      } else if (direction === -1 && hasStop && bar.high >= stop) {
        const exitPrice = roundToTick(fillStop(slipSymbol, 'buy', stop));
        exitPosition(exitPrice, bar.timestamp, 'stop', stopSlippageTicks);
        continue;
      }
    }

    // --- S/R Logic using Pivot Points ---
    if (!currentPivotPoints) {
      continue; // Not enough data for pivot points yet
    }

    const resistance = currentPivotPoints.R1;
    const support = currentPivotPoints.S1;

    const rsiValue = rsiIndicator.nextValue(bar.close);
    const adxValue = adxIndicator.nextValue({
      high: bar.high,
      low: bar.low,
      close: bar.close,
    });

    // TTM Squeeze calculation
    const ttmBars = bars.slice(Math.max(0, i - 20), i + 1); // Lookback for TTM Squeeze
    const ttmSqueeze = calculateTtmSqueeze(ttmBars, { lookback: 20, bbStdDev: 2, atrMultiplier: 1.5 });
    if (!ttmSqueeze) continue; // Not enough data for TTM Squeeze

    // Entry conditions based on S/R Rejections, TTM Squeeze, and RSI
    if (!position && ttmSqueeze.squeezeOn) { // Only enter if TTM Squeeze is ON
      // Rejection Short (from Resistance): Price hits resistance but closes below it, with RSI >= 70
      if (bar.high >= resistance && bar.close < resistance && rsiValue >= 70) {
        position = 'short';
        entryPrice = roundToTick(fillEntry(slipSymbol, 'sell', bar.close));
        entryTime = bar.timestamp;
        entrySlippageTicks = entrySlipTicksForSymbol;
      }
      // Rejection Long (from Support): Price hits support but closes above it, with RSI <= 30
      else if (bar.low <= support && bar.close > support && rsiValue <= 30) {
        position = 'long';
        entryPrice = roundToTick(fillEntry(slipSymbol, 'buy', bar.close));
        entryTime = bar.timestamp;
        entrySlippageTicks = entrySlipTicksForSymbol;
      }
    }

    if (rsiValue !== undefined) {
      prevRsiValue = rsiValue;
    }

    if (rsiValue !== undefined) {
      prevRsiValue = rsiValue;
    }
  }

  if (position) {
    const lastBar = bars[bars.length - 1];
    const exitSide = position === 'long' ? 'sell' : 'buy';
    const exitPrice = roundToTick(fillStop(slipSymbol, exitSide, lastBar.close));
    exitPosition(exitPrice, lastBar.timestamp, 'end_of_data', stopSlippageTicks);
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

  console.log('\n===== Second-Bar S/R Backtest Summary =====');
  console.log(`Trades: ${trades.length} | Win rate: ${trades.length ? ((wins.length / trades.length) * 100).toFixed(1) : '0'}%`);
  console.log(`Realized PnL: ${formatCurrency(realizedPnL)} USD`);
  console.log(`Avg Win: ${formatCurrency(avgWin)} | Avg Loss: ${formatCurrency(avgLoss)}`);
  console.log(`Max Drawdown: ${formatCurrency(maxDrawdown)} USD`);

  if (trades.length) {
    console.log('\nRecent trades:');
    trades.slice(-5).forEach(trade => {
      console.log(
        ` - ${trade.side.toUpperCase()} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | PnL: ${formatCurrency(trade.pnl)} | Gross PnL: ${formatCurrency(trade.grossPnl)} | Fees: ${formatCurrency(trade.fees)} | Slippage Cost: ${formatCurrency(trade.slippageCost)} | Entry Slip Ticks: ${trade.entrySlippageTicks.toFixed(2)} | Exit Slip Ticks: ${trade.exitSlippageTicks.toFixed(2)} (${trade.exitReason})`,
      );
    });
  }
}

runBacktest().catch(err => {
  console.error('TopstepX second-bar S/R backtest failed:', err);
  process.exit(1);
});
