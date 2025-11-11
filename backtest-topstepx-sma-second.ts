import { RSI, ADX, ATR } from 'technicalindicators';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';
import { calculateTtmSqueeze, type TtmSqueezeResult } from './lib/ttmSqueeze';

interface BacktestConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;
  smaPeriod: number;
  sma100Period: number;
  sma200Period: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  trailingStopPercent: number;
  commissionPerSide: number;
  contractMultiplier: number;
  rsiPeriod: number;
  adxPeriod: number;
  adxThreshold: number;
  chopPeriod: number;
  chopThreshold: number;
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
  exitReason: 'target' | 'stop' | 'signal' | 'session' | 'end_of_data';
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_DAYS = 5;

function fillEntry(side: 'buy' | 'sell', mid: number): number {
  return mid;
}

function fillTP(side: 'buy' | 'sell', mid: number): number {
  return mid;
}

function fillStop(side: 'buy' | 'sell', triggerMid: number): number {
  return triggerMid;
}

const DEFAULT_SECOND_SYMBOL = process.env.TOPSTEPX_SECOND_SMA_SYMBOL || 'ESZ5';
const DEFAULT_SECOND_CONTRACT_ID = process.env.TOPSTEPX_SECOND_SMA_CONTRACT_ID;

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_SECOND_SYMBOL,
  contractId: DEFAULT_SECOND_CONTRACT_ID,
  start: process.env.TOPSTEPX_SECOND_SMA_START
    || new Date(Date.now() - DEFAULT_DAYS * DAY_MS).toISOString(),
  end: process.env.TOPSTEPX_SECOND_SMA_END || new Date().toISOString(),
  smaPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_PERIOD || '10'),
  sma100Period: Number(process.env.TOPSTEPX_SECOND_SMA_100_PERIOD || '20'),
  sma200Period: Number(process.env.TOPSTEPX_SECOND_SMA_200_PERIOD || '40'),
  stopLossPercent: Number(process.env.TOPSTEPX_SECOND_STOP_LOSS_PERCENT || '0'),
  takeProfitPercent: Number(process.env.TOPSTEPX_SECOND_TAKE_PROFIT_PERCENT || '0.008'),
  trailingStopPercent: Number(process.env.TOPSTEPX_SECOND_TRAILING_STOP_PERCENT || '0.001'),
  commissionPerSide: process.env.TOPSTEPX_SECOND_COMMISSION
    ? Number(process.env.TOPSTEPX_SECOND_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_SECOND_CONTRACT_ID, DEFAULT_SECOND_SYMBOL]),
  contractMultiplier: Number(process.env.TOPSTEPX_SECOND_CONTRACT_MULTIPLIER || '50'),
  rsiPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_RSI_PERIOD || '24'),
  adxPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_ADX_PERIOD || '14'),
  adxThreshold: Number(process.env.TOPSTEPX_SECOND_SMA_ADX_THRESHOLD || '20'),
  chopPeriod: Number(process.env.TOPSTEPX_SECOND_CHOP_PERIOD || '14'),
  chopThreshold: Number(process.env.TOPSTEPX_SECOND_CHOP_THRESHOLD || '50'),
  numberOfContracts: Number(process.env.TOPSTEPX_SECOND_CONTRACTS || '5'),
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

function calculateSMA(values: number[], period: number) {
  if (values.length < period) return null;
  const sum = values.slice(-period).reduce((acc, value) => acc + value, 0);
  return sum / period;
}

function calculateTrueRange(prev: number, high: number, low: number): number {
  if (prev === 0) return high - low;
  return Math.max(high - low, Math.abs(high - prev), Math.abs(low - prev));
}

function calculateChop(bars: any[], period: number): number | null {
  if (bars.length < period) return null;

  const recentBars = bars.slice(-period);
  let trSum = 0;

  // Calculate sum of true ranges
  for (let i = 0; i < recentBars.length; i++) {
    const prevClose = i === 0 ? bars[bars.length - period - 1]?.close || recentBars[i].close : recentBars[i - 1].close;
    const tr = calculateTrueRange(prevClose, recentBars[i].high, recentBars[i].low);
    trSum += tr;
  }

  // Find highest high and lowest low over period
  const highs = recentBars.map(b => b.high);
  const lows = recentBars.map(b => b.low);
  const maxHigh = Math.max(...highs);
  const minLow = Math.min(...lows);

  const range = maxHigh - minLow;
  if (range === 0) return 50; // Avoid division by zero

  const chop = 100 * (Math.log10(trSum / range) / Math.log10(period));
  return Math.max(0, Math.min(100, chop)); // Clamp between 0-100
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
    console.log(`Fetching 5s bars from ${cursor.toISOString()} to ${chunkEnd.toISOString()}`);
    const chunkBars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: cursor.toISOString(),
      endTime: chunkEnd.toISOString(),
      unit: 1,
      unitNumber: 5,
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
  console.log('TOPSTEPX SECOND-BAR MULTI-SMA BACKTEST');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end} (~${DEFAULT_DAYS} days)`);
  console.log(`SMA Periods (seconds): 50 | 100 | 200`);
  console.log(`Entry Logic: 50 SMA crosses 100 SMA in trending direction (100 vs 200) + RSI confirmation`);
  console.log(`Exit Logic: 50 SMA crosses back through 100 SMA OR Trailing Stop Hit`);
  console.log(`Trailing Stop: ${(CONFIG.trailingStopPercent * 100).toFixed(2)}%`);
  console.log(`Commission/side: ${CONFIG.commissionPerSide.toFixed(2)} USD | Contracts: ${CONFIG.numberOfContracts}`);
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[topstepx-second] Unable to fetch metadata:', err.message);
    return null;
  });
  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  const contractId = metadata.id;
  const tickSize = metadata.tickSize;
  if (!tickSize || !Number.isFinite(tickSize) || tickSize <= 0) {
    throw new Error(`Invalid tick size for ${lookupKey}`);
  }
  const multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || CONFIG.contractMultiplier;
  const roundToTick = (value: number) => Math.round(value / tickSize) * tickSize;
  const contracts = CONFIG.numberOfContracts;

  console.log(`Resolved contract: ${metadata.name} (${contractId})`);
  console.log(`Point multiplier: ${multiplier} | Tick size: ${tickSize}`);

  console.log('\nFetching 1-second bars in chunks...');
  const bars = await fetchSecondBarsInChunks(contractId, CONFIG.start, CONFIG.end);

  if (!bars.length) {
    throw new Error('No second bars returned for configured window.');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length} five-second bars`);
  const closes: number[] = [];
  const barBuffer: { high: number; low: number; close: number }[] = [];
  let position: 'long' | 'short' | null = null;
  let entryPrice = 0;
  let entryTime = '';
  let highestPrice = 0; // For trailing stop on longs
  let lowestPrice = Infinity; // For trailing stop on shorts
  const trades: TradeRecord[] = [];
  let realizedPnL = 0;

  const adxIndicator = new ADX({
    period: CONFIG.adxPeriod,
    high: [],
    low: [],
    close: [],
  });
  let prevSMA50: number | null = null;
  let prevSMA100: number | null = null;
  let prevSMA200: number | null = null;
  let prevTtmSqueeze: TtmSqueezeResult | null = null;

  const exitPosition = (
    exitPrice: number,
    exitTime: string,
    reason: TradeRecord['exitReason'],
  ) => {
    if (!position) return;
    const direction = position === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - entryPrice) * direction * multiplier * contracts;
    const fees = CONFIG.commissionPerSide * 2 * contracts;
    const netPnL = rawPnL - fees;
    trades.push({
      entryTime,
      exitTime,
      side: position,
      entryPrice,
      exitPrice,
      pnl: netPnL,
      grossPnl: rawPnL,
      fees,
      exitReason: reason,
    });
    realizedPnL += netPnL;
    position = null;
    entryPrice = 0;
    entryTime = '';
    highestPrice = 0;
    lowestPrice = Infinity;
  };

  for (let barIndex = 0; barIndex < bars.length; barIndex++) {
    const bar = bars[barIndex];
    // Calculate SMAs BEFORE adding new bar (this is the SMA value we "knew" at the previous moment)
    prevSMA50 = closes.length >= CONFIG.smaPeriod
      ? calculateSMA(closes, CONFIG.smaPeriod)
      : null;
    prevSMA100 = closes.length >= CONFIG.sma100Period
      ? calculateSMA(closes, CONFIG.sma100Period)
      : null;
    prevSMA200 = closes.length >= CONFIG.sma200Period
      ? calculateSMA(closes, CONFIG.sma200Period)
      : null;

    const prevPrice = closes.length >= 1 ? closes[closes.length - 1] : bar.close;

    // Now add the new bar
    closes.push(bar.close);
    barBuffer.push({ high: bar.high, low: bar.low, close: bar.close });
    // Keep only last 30 bars for TTM calculation (lookback 20 + buffer)
    if (barBuffer.length > 30) {
      barBuffer.shift();
    }

    // Calculate NEW SMAs with the latest bar included
    const currSMA50 = calculateSMA(closes, CONFIG.smaPeriod);
    const currSMA100 = calculateSMA(closes, CONFIG.sma100Period);
    const currSMA200 = calculateSMA(closes, CONFIG.sma200Period);

    if (!isTradingAllowed(bar.timestamp)) {
      if (position) {
        const exitSide = position === 'long' ? 'sell' : 'buy';
        const exitPrice = roundToTick(fillStop(exitSide, bar.close));
        exitPosition(exitPrice, bar.timestamp, 'session');
      }
      continue;
    }

    if (position) {
      const direction = position === 'long' ? 1 : -1;

      // Update highest/lowest price for trailing stop
      if (position === 'long') {
        highestPrice = Math.max(highestPrice, bar.high);
      } else {
        lowestPrice = Math.min(lowestPrice, bar.low);
      }

      const target = direction === 1
        ? entryPrice * (1 + CONFIG.takeProfitPercent)
        : entryPrice * (1 - CONFIG.takeProfitPercent);
      const stop = direction === 1
        ? entryPrice * (1 - CONFIG.stopLossPercent)
        : entryPrice * (1 + CONFIG.stopLossPercent);

      // Trailing stop calculation
      const trailingStop = direction === 1
        ? highestPrice * (1 - CONFIG.trailingStopPercent)
        : lowestPrice * (1 + CONFIG.trailingStopPercent);

      const hasTarget = CONFIG.takeProfitPercent > 0;
      const hasStop = CONFIG.stopLossPercent > 0;
      const hasTrailingStop = CONFIG.trailingStopPercent > 0;

      if (direction === 1 && hasTarget && bar.high >= target) {
        const basePrice = Math.min(bar.high, target);
        const exitPrice = roundToTick(fillTP('sell', basePrice));
        exitPosition(exitPrice, bar.timestamp, 'target');
        continue;
      } else if (direction === 1 && hasStop && bar.low <= stop) {
        const exitPrice = roundToTick(fillStop('sell', stop));
        exitPosition(exitPrice, bar.timestamp, 'stop');
        continue;
      } else if (direction === 1 && hasTrailingStop && bar.low <= trailingStop) {
        const exitPrice = roundToTick(fillStop('sell', trailingStop));
        exitPosition(exitPrice, bar.timestamp, 'stop');
        continue;
      } else if (direction === -1 && hasTarget && bar.low <= target) {
        const basePrice = Math.max(bar.low, target);
        const exitPrice = roundToTick(fillTP('buy', basePrice));
        exitPosition(exitPrice, bar.timestamp, 'target');
        continue;
      } else if (direction === -1 && hasStop && bar.high >= stop) {
        const exitPrice = roundToTick(fillStop('buy', stop));
        exitPosition(exitPrice, bar.timestamp, 'stop');
        continue;
      } else if (direction === -1 && hasTrailingStop && bar.high >= trailingStop) {
        const exitPrice = roundToTick(fillStop('buy', trailingStop));
        exitPosition(exitPrice, bar.timestamp, 'stop');
        continue;
      }
    }

    // Need all SMAs to proceed
    if (!prevSMA50 || !currSMA50 || !prevSMA100 || !currSMA100 || !prevSMA200 || !currSMA200 || closes.length < 2) {
      continue;
    }

    const adxValue = (adxIndicator.nextValue as any)({
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }) as { adx?: number } | number | undefined;
    const currPrice = bar.close;

    // Calculate TTM Squeeze (needs 20 bars lookback minimum)
    const ttmSqueeze = barBuffer.length >= 20
      ? calculateTtmSqueeze(barBuffer, { lookback: 20, bbStdDev: 2, atrMultiplier: 1.5, momentumThreshold: 1e-5 })
      : null;

    // Calculate Choppiness Index
    const chop = calculateChop(bars.slice(0, barIndex + 1), CONFIG.chopPeriod);

    if (ttmSqueeze === null || adxValue === undefined || chop === null) {
      continue;
    }

    // 50 SMA crossing 100 SMA (not price crossing 50 SMA)
    const crossedUp50Over100 = prevSMA50 <= prevSMA100 && currSMA50 > currSMA100;
    const crossedDown50Below100 = prevSMA50 >= prevSMA100 && currSMA50 < currSMA100;

    // Trend filter: 100 SMA relative to 200 SMA
    const isUptrend = currSMA100 > currSMA200;
    const isDowntrend = currSMA100 < currSMA200;

    // Choppiness filter: only trade when market is NOT too choppy
    const isTrendingEnough = chop < CONFIG.chopThreshold;

    // ADX trend strength filter (ADX > 20 indicates strong trend)
    const adxNumValue =
      typeof adxValue === 'number'
        ? adxValue
        : adxValue && typeof adxValue === 'object'
          ? adxValue.adx ?? null
          : null;
    const adxStrong = adxNumValue !== null && adxNumValue >= CONFIG.adxThreshold;

    // TTM Squeeze signal logic:
    // Enter when TTM turns from OFF/bearish to ON/bullish
    const ttmTurnedBullish =
      prevTtmSqueeze &&
      (prevTtmSqueeze.sentiment !== 'bullish' || !prevTtmSqueeze.squeezeOn) &&
      (ttmSqueeze.sentiment === 'bullish' || ttmSqueeze.squeezeOn);

    const ttmTurnedBearish =
      prevTtmSqueeze &&
      (prevTtmSqueeze.sentiment !== 'bearish' || !prevTtmSqueeze.squeezeOn) &&
      (ttmSqueeze.sentiment === 'bearish' || ttmSqueeze.squeezeOn);

    if (crossedUp50Over100 && isUptrend && ttmTurnedBullish) {
      // Long signal: 50 SMA crossed above 100 SMA while in uptrend (100>200) and TTM turns bullish
      if (position === 'short') {
        const exitPrice = roundToTick(fillStop('buy', currPrice));
        exitPosition(exitPrice, bar.timestamp, 'signal');
      }
      if (!position) {
        position = 'long';
        entryPrice = roundToTick(fillEntry('buy', currPrice));
        entryTime = bar.timestamp;
        highestPrice = entryPrice; // Initialize highest price at entry
      }
    } else if (crossedDown50Below100 && isDowntrend && ttmTurnedBearish) {
      // Short signal: 50 SMA crossed below 100 SMA while in downtrend (100<200) and TTM turns bearish
      if (position === 'long') {
        const exitPrice = roundToTick(fillStop('sell', currPrice));
        exitPosition(exitPrice, bar.timestamp, 'signal');
      }
      if (!position) {
        position = 'short';
        entryPrice = roundToTick(fillEntry('sell', currPrice));
        entryTime = bar.timestamp;
        lowestPrice = entryPrice; // Initialize lowest price at entry
      }
    } else if (position === 'long' && crossedDown50Below100) {
      // Exit long on 50 SMA crossing below 100 SMA
      const exitPrice = roundToTick(fillStop('sell', currPrice));
      exitPosition(exitPrice, bar.timestamp, 'signal');
    } else if (position === 'short' && crossedUp50Over100) {
      // Exit short on 50 SMA crossing above 100 SMA
      const exitPrice = roundToTick(fillStop('buy', currPrice));
      exitPosition(exitPrice, bar.timestamp, 'signal');
    }

    prevTtmSqueeze = ttmSqueeze;
  }

  if (position) {
    const lastBar = bars[bars.length - 1];
    const exitSide = position === 'long' ? 'sell' : 'buy';
    const exitPrice = roundToTick(fillStop(exitSide, lastBar.close));
    exitPosition(exitPrice, lastBar.timestamp, 'end_of_data');
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

  console.log('\n===== Second-Bar SMA Backtest Summary =====');
  console.log(`Trades: ${trades.length} | Win rate: ${trades.length ? ((wins.length / trades.length) * 100).toFixed(1) : '0'}%`);
  console.log(`Realized PnL: ${formatCurrency(realizedPnL)} USD`);
  console.log(`Avg Win: ${formatCurrency(avgWin)} | Avg Loss: ${formatCurrency(avgLoss)}`);
  console.log(`Max Drawdown: ${formatCurrency(maxDrawdown)} USD`);

  if (trades.length) {
    console.log('\nRecent trades:');
    trades.slice(-5).forEach(trade => {
    console.log(
        ` - ${trade.side.toUpperCase()} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | PnL: ${formatCurrency(trade.pnl)} | Gross PnL: ${formatCurrency(trade.grossPnl)} | Fees: ${formatCurrency(trade.fees)} (${trade.exitReason})`,
    );
    });
  }
}

runBacktest().catch(err => {
  console.error('TopstepX second-bar SMA backtest failed:', err);
  process.exit(1);
});
