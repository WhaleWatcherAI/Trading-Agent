import { ADX, RSI, ATR, VWAP, BollingerBands, MACD } from 'technicalindicators';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
} from './lib/topstepx';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

interface BacktestConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;
  smaPeriod: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  commissionPerSide: number;
  contractMultiplier: number;
  rsiPeriod: number;
  adxPeriod: number;
  adxThreshold: number;
  numberOfContracts: number;

  // Enhanced filters
  minVolumeMultiplier: number; // Min volume as multiplier of average
  maxTradesPerDay: number;
  minTimeBetweenTrades: number; // seconds
  volatilityFilterEnabled: boolean;
  trendStrengthThreshold: number;
  momentumConfirmationPeriod: number;
  microstructureEnabled: boolean;
  sessionFilterEnabled: boolean;
  multiTimeframeEnabled: boolean;
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
  signalStrength: number;
}

interface MarketMicrostructure {
  bidAskImbalance: number;
  volumeImbalance: number;
  priceVelocity: number;
  microTrend: number;
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_DAYS = 5;

// Prime trading sessions (CT)
const PRIME_SESSIONS = [
  { start: 8 * 60 + 30, end: 11 * 60 }, // Morning session
  { start: 14 * 60, end: 15 * 60 + 10 }, // Close
];

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
  smaPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_PERIOD || '200'),
  stopLossPercent: Number(process.env.TOPSTEPX_SECOND_STOP_LOSS_PERCENT || '0.0015'),
  takeProfitPercent: Number(process.env.TOPSTEPX_SECOND_TAKE_PROFIT_PERCENT || '0.0025'),
  commissionPerSide: process.env.TOPSTEPX_SECOND_COMMISSION
    ? Number(process.env.TOPSTEPX_SECOND_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_SECOND_CONTRACT_ID, DEFAULT_SECOND_SYMBOL]),
  contractMultiplier: Number(process.env.TOPSTEPX_SECOND_CONTRACT_MULTIPLIER || '50'),
  rsiPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_RSI_PERIOD || '24'),
  adxPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_ADX_PERIOD || '24'),
  adxThreshold: Number(process.env.TOPSTEPX_SECOND_SMA_ADX_THRESHOLD || '30'),
  numberOfContracts: Number(process.env.TOPSTEPX_SECOND_CONTRACTS || '5'),

  // Enhanced filter defaults
  minVolumeMultiplier: 1.5,
  maxTradesPerDay: 10,
  minTimeBetweenTrades: 300, // 5 minutes
  volatilityFilterEnabled: true,
  trendStrengthThreshold: 0.7,
  momentumConfirmationPeriod: 10,
  microstructureEnabled: true,
  sessionFilterEnabled: true,
  multiTimeframeEnabled: true,
};

function toCentralTime(date: Date) {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isPrimeSession(timestamp: string | Date): boolean {
  if (!CONFIG.sessionFilterEnabled) return true;

  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const ctDate = toCentralTime(date);
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  return PRIME_SESSIONS.some(session =>
    minutes >= session.start && minutes <= session.end
  );
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

function calculateEMA(values: number[], period: number): number | null {
  if (values.length < period) return null;

  const k = 2 / (period + 1);
  let ema = values.slice(0, period).reduce((a, b) => a + b) / period;

  for (let i = period; i < values.length; i++) {
    ema = values[i] * k + ema * (1 - k);
  }

  return ema;
}

function calculateVWAP(bars: TopstepXFuturesBar[], lookback: number): number | null {
  if (bars.length < lookback) return null;

  const recentBars = bars.slice(-lookback);
  let sumPV = 0;
  let sumV = 0;

  for (const bar of recentBars) {
    const typicalPrice = (bar.high + bar.low + bar.close) / 3;
    sumPV += typicalPrice * bar.volume;
    sumV += bar.volume;
  }

  return sumV > 0 ? sumPV / sumV : null;
}

function analyzeMicrostructure(
  bars: TopstepXFuturesBar[],
  lookback: number = 20
): MarketMicrostructure {
  if (bars.length < lookback) {
    return { bidAskImbalance: 0, volumeImbalance: 0, priceVelocity: 0, microTrend: 0 };
  }

  const recent = bars.slice(-lookback);

  // Analyze volume imbalance (buying vs selling pressure)
  let buyVolume = 0;
  let sellVolume = 0;

  for (let i = 1; i < recent.length; i++) {
    const priceChange = recent[i].close - recent[i - 1].close;
    if (priceChange > 0) {
      buyVolume += recent[i].volume;
    } else if (priceChange < 0) {
      sellVolume += recent[i].volume;
    }
  }

  const totalVolume = buyVolume + sellVolume;
  const volumeImbalance = totalVolume > 0 ? (buyVolume - sellVolume) / totalVolume : 0;

  // Calculate price velocity (rate of change)
  const priceChanges = [];
  for (let i = 1; i < recent.length; i++) {
    priceChanges.push((recent[i].close - recent[i - 1].close) / recent[i - 1].close);
  }
  const avgPriceChange = priceChanges.reduce((a, b) => a + b, 0) / priceChanges.length;
  const priceVelocity = avgPriceChange * 10000; // Scale for readability

  // Micro trend strength
  let trendUp = 0;
  let trendDown = 0;
  for (let i = 1; i < recent.length; i++) {
    if (recent[i].close > recent[i - 1].close) trendUp++;
    else if (recent[i].close < recent[i - 1].close) trendDown++;
  }
  const microTrend = (trendUp - trendDown) / (recent.length - 1);

  // Simplified bid-ask imbalance (using high-low spread as proxy)
  const spreads = recent.map(b => (b.high - b.low) / b.close);
  const avgSpread = spreads.reduce((a, b) => a + b, 0) / spreads.length;
  const currentSpread = (recent[recent.length - 1].high - recent[recent.length - 1].low) / recent[recent.length - 1].close;
  const bidAskImbalance = (avgSpread - currentSpread) / avgSpread;

  return { bidAskImbalance, volumeImbalance, priceVelocity, microTrend };
}

function calculateSignalStrength(
  crossoverStrength: number,
  rsiStrength: number,
  adxStrength: number,
  volumeStrength: number,
  microstructure: MarketMicrostructure,
  volatilityOK: boolean,
  trendAlignment: boolean
): number {
  let strength = 0;
  let weights = 0;

  // Core signals
  strength += crossoverStrength * 20;
  weights += 20;

  strength += rsiStrength * 15;
  weights += 15;

  strength += adxStrength * 15;
  weights += 15;

  // Volume confirmation
  strength += volumeStrength * 20;
  weights += 20;

  // Microstructure
  if (CONFIG.microstructureEnabled) {
    strength += Math.abs(microstructure.volumeImbalance) * 10;
    strength += Math.abs(microstructure.microTrend) * 10;
    weights += 20;
  }

  // Volatility filter
  if (volatilityOK) {
    strength += 5;
  }
  weights += 5;

  // Trend alignment
  if (trendAlignment) {
    strength += 5;
  }
  weights += 5;

  return strength / weights;
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
      limit: 20000,
    });
    bars.push(...chunkBars);
    cursor = new Date(chunkEnd.getTime() + 1_000);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  return bars.reverse();
}

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('ENHANCED TOPSTEPX SECOND-BAR SMA BACKTEST');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end}`);
  console.log(`SMA Period: ${CONFIG.smaPeriod} seconds`);
  console.log(`RSI Period: ${CONFIG.rsiPeriod} | ADX: ${CONFIG.adxPeriod} (threshold ${CONFIG.adxThreshold})`);
  console.log(`Contracts: ${CONFIG.numberOfContracts} | Commission: $${CONFIG.commissionPerSide}/side`);
  console.log('\nEnhanced Filters:');
  console.log(`- Volume Multiplier: ${CONFIG.minVolumeMultiplier}x`);
  console.log(`- Max Trades/Day: ${CONFIG.maxTradesPerDay}`);
  console.log(`- Min Time Between: ${CONFIG.minTimeBetweenTrades}s`);
  console.log(`- Volatility Filter: ${CONFIG.volatilityFilterEnabled}`);
  console.log(`- Microstructure: ${CONFIG.microstructureEnabled}`);
  console.log(`- Session Filter: ${CONFIG.sessionFilterEnabled}`);
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[topstepx-enhanced] Unable to fetch metadata:', err.message);
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

  console.log(`Resolved: ${metadata.name} (${contractId})`);
  console.log(`Multiplier: ${multiplier} | Tick: ${tickSize}`);

  console.log('\nFetching 1-second bars...');
  const bars = await fetchSecondBarsInChunks(contractId, CONFIG.start, CONFIG.end);

  if (!bars.length) {
    throw new Error('No bars returned');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length} bars\n`);

  const closes: number[] = [];
  const volumes: number[] = [];
  let position: 'long' | 'short' | null = null;
  let entryPrice = 0;
  let entryTime = '';
  const trades: TradeRecord[] = [];
  let realizedPnL = 0;
  let lastTradeTime = 0;
  const dailyTrades = new Map<string, number>();

  // Indicators
  const rsiIndicator = new RSI({ period: CONFIG.rsiPeriod, values: [] });
  const adxIndicator = new ADX({
    period: CONFIG.adxPeriod,
    high: [],
    low: [],
    close: [],
  });
  const atrIndicator = new ATR({
    period: 14,
    high: [],
    low: [],
    close: [],
  });
  const macdIndicator = new MACD({
    fastPeriod: 12,
    slowPeriod: 26,
    signalPeriod: 9,
    SimpleMAOscillator: false,
    SimpleMASignal: false,
    values: [],
  });

  let prevRsiValue: number | null = null;
  let prevSMA: number | null = null;
  let signalsBlocked = 0;
  let signalsPassed = 0;

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
      signalStrength: 0,
    });
    realizedPnL += netPnL;
    position = null;
    entryPrice = 0;
    entryTime = '';
  };

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];

    // Track daily trades
    const dateKey = new Date(bar.timestamp).toISOString().split('T')[0];
    if (!dailyTrades.has(dateKey)) {
      dailyTrades.set(dateKey, 0);
    }

    // Calculate indicators BEFORE adding new bar
    prevSMA = closes.length >= CONFIG.smaPeriod
      ? calculateSMA(closes, CONFIG.smaPeriod)
      : null;

    const prevPrice = closes.length >= 1 ? closes[closes.length - 1] : bar.close;

    // Add new data
    closes.push(bar.close);
    volumes.push(bar.volume);

    // Calculate current indicators
    const currSMA = calculateSMA(closes, CONFIG.smaPeriod);
    const avgVolume = volumes.length > 50
      ? volumes.slice(-50).reduce((a, b) => a + b) / 50
      : bar.volume;

    if (!isTradingAllowed(bar.timestamp)) {
      if (position) {
        const exitSide = position === 'long' ? 'sell' : 'buy';
        const exitPrice = roundToTick(fillStop(exitSide, bar.close));
        exitPosition(exitPrice, bar.timestamp, 'session');
      }
      continue;
    }

    // Check stop/target for existing position
    if (position) {
      const direction = position === 'long' ? 1 : -1;
      const target = direction === 1
        ? entryPrice * (1 + CONFIG.takeProfitPercent)
        : entryPrice * (1 - CONFIG.takeProfitPercent);
      const stop = direction === 1
        ? entryPrice * (1 - CONFIG.stopLossPercent)
        : entryPrice * (1 + CONFIG.stopLossPercent);

      if (direction === 1 && bar.high >= target) {
        const exitPrice = roundToTick(fillTP('sell', target));
        exitPosition(exitPrice, bar.timestamp, 'target');
        continue;
      } else if (direction === 1 && bar.low <= stop) {
        const exitPrice = roundToTick(fillStop('sell', stop));
        exitPosition(exitPrice, bar.timestamp, 'stop');
        continue;
      } else if (direction === -1 && bar.low <= target) {
        const exitPrice = roundToTick(fillTP('buy', target));
        exitPosition(exitPrice, bar.timestamp, 'target');
        continue;
      } else if (direction === -1 && bar.high >= stop) {
        const exitPrice = roundToTick(fillStop('buy', stop));
        exitPosition(exitPrice, bar.timestamp, 'stop');
        continue;
      }
    }

    // Skip if not enough data
    if (prevSMA === null || currSMA === null || closes.length < CONFIG.smaPeriod + 10) {
      continue;
    }

    // Calculate all indicators
    const rsiValue = rsiIndicator.nextValue(bar.close);
    const adxValue = adxIndicator.nextValue({
      high: bar.high,
      low: bar.low,
      close: bar.close,
    });
    const atrValue = atrIndicator.nextValue({
      high: bar.high,
      low: bar.low,
      close: bar.close,
    });
    const macdValue = macdIndicator.nextValue(bar.close);
    const vwap = calculateVWAP(bars.slice(Math.max(0, i - 200), i + 1), 200);

    if (!rsiValue || !adxValue?.adx) {
      if (rsiValue !== undefined) prevRsiValue = rsiValue;
      continue;
    }

    // Detect crossovers
    const currPrice = bar.close;
    const crossedUp = prevPrice <= prevSMA && currPrice > currSMA;
    const crossedDown = prevPrice >= prevSMA && currPrice < currSMA;

    if (!crossedUp && !crossedDown) {
      if (rsiValue !== undefined) prevRsiValue = rsiValue;
      continue;
    }

    // ENHANCED SIGNAL FILTERING

    // 1. Volume filter
    const volumeOK = bar.volume >= avgVolume * CONFIG.minVolumeMultiplier;

    // 2. Time filter
    const timeSinceLastTrade = Date.parse(bar.timestamp) - lastTradeTime;
    const timeOK = timeSinceLastTrade >= CONFIG.minTimeBetweenTrades * 1000;

    // 3. Daily trade limit
    const dailyTradeCount = dailyTrades.get(dateKey) || 0;
    const dailyLimitOK = dailyTradeCount < CONFIG.maxTradesPerDay;

    // 4. Session filter
    const primeSessionOK = isPrimeSession(bar.timestamp);

    // 5. Volatility filter
    const volatilityOK = !CONFIG.volatilityFilterEnabled ||
      (atrValue && atrValue > tickSize * 2 && atrValue < tickSize * 10);

    // 6. Microstructure analysis
    const microstructure = CONFIG.microstructureEnabled
      ? analyzeMicrostructure(bars.slice(Math.max(0, i - 20), i + 1))
      : { bidAskImbalance: 0, volumeImbalance: 0, priceVelocity: 0, microTrend: 0 };

    // 7. Trend strength
    const smaSlope = (currSMA - prevSMA) / prevSMA;
    const trendStrong = Math.abs(smaSlope) > 0.00001;

    // 8. Multi-timeframe confirmation
    const ema50 = calculateEMA(closes, 50);
    const ema200 = calculateEMA(closes, 200);
    const mtfAligned = !CONFIG.multiTimeframeEnabled ||
      (crossedUp ? (ema50 && ema200 && ema50 > ema200) :
       crossedDown ? (ema50 && ema200 && ema50 < ema200) : false);

    // 9. MACD confirmation
    const macdOK = !macdValue ||
      (crossedUp ? macdValue.MACD > macdValue.signal :
       crossedDown ? macdValue.MACD < macdValue.signal : false);

    // 10. VWAP alignment
    const vwapOK = !vwap ||
      (crossedUp ? currPrice > vwap :
       crossedDown ? currPrice < vwap : false);

    // Calculate signal components
    const rsiLongOK = rsiValue > 50 && prevRsiValue && rsiValue > prevRsiValue;
    const rsiShortOK = rsiValue < 50 && prevRsiValue && rsiValue < prevRsiValue;
    const adxTrending = adxValue.adx >= CONFIG.adxThreshold;
    const adxLongOK = adxTrending && adxValue.pdi > adxValue.mdi;
    const adxShortOK = adxTrending && adxValue.mdi > adxValue.pdi;

    // Microstructure confirmation
    const microLongOK = !CONFIG.microstructureEnabled ||
      (microstructure.volumeImbalance > 0.2 && microstructure.microTrend > 0.3);
    const microShortOK = !CONFIG.microstructureEnabled ||
      (microstructure.volumeImbalance < -0.2 && microstructure.microTrend < -0.3);

    // FINAL SIGNAL DECISION
    const longSignal = crossedUp &&
      rsiLongOK &&
      adxLongOK &&
      volumeOK &&
      timeOK &&
      dailyLimitOK &&
      primeSessionOK &&
      volatilityOK &&
      trendStrong &&
      mtfAligned &&
      macdOK &&
      vwapOK &&
      microLongOK;

    const shortSignal = crossedDown &&
      rsiShortOK &&
      adxShortOK &&
      volumeOK &&
      timeOK &&
      dailyLimitOK &&
      primeSessionOK &&
      volatilityOK &&
      trendStrong &&
      mtfAligned &&
      macdOK &&
      vwapOK &&
      microShortOK;

    // Track blocked signals for analysis
    if ((crossedUp || crossedDown) && !longSignal && !shortSignal) {
      signalsBlocked++;

      // Log why signals were blocked (for debugging)
      if (i === bars.length - 1 || signalsBlocked <= 5) {
        const reasons = [];
        if (!volumeOK) reasons.push('volume');
        if (!timeOK) reasons.push('time');
        if (!dailyLimitOK) reasons.push('daily_limit');
        if (!primeSessionOK) reasons.push('session');
        if (!volatilityOK) reasons.push('volatility');
        if (!trendStrong) reasons.push('weak_trend');
        if (!mtfAligned) reasons.push('mtf');
        if (!macdOK) reasons.push('macd');
        if (!vwapOK) reasons.push('vwap');
        if (crossedUp && !microLongOK) reasons.push('micro');
        if (crossedDown && !microShortOK) reasons.push('micro');
        if (crossedUp && !rsiLongOK) reasons.push('rsi');
        if (crossedDown && !rsiShortOK) reasons.push('rsi');
        if (crossedUp && !adxLongOK) reasons.push('adx');
        if (crossedDown && !adxShortOK) reasons.push('adx');

        console.log(`Signal blocked at ${bar.timestamp}: ${reasons.join(', ')}`);
      }
    }

    // Execute trades
    if (longSignal) {
      signalsPassed++;
      if (position === 'short') {
        const exitPrice = roundToTick(fillStop('buy', currPrice));
        exitPosition(exitPrice, bar.timestamp, 'signal');
      }
      if (!position) {
        position = 'long';
        entryPrice = roundToTick(fillEntry('buy', currPrice));
        entryTime = bar.timestamp;
        lastTradeTime = Date.parse(bar.timestamp);
        dailyTrades.set(dateKey, dailyTradeCount + 1);

        // Calculate and store signal strength
        const strength = calculateSignalStrength(
          1.0, // crossover confirmed
          (rsiValue - 50) / 50, // RSI strength
          adxValue.adx / 100, // ADX strength
          bar.volume / (avgVolume * 2), // Volume strength
          microstructure,
          volatilityOK,
          mtfAligned
        );

        if (trades.length < 10) {
          console.log(`LONG entry @ ${entryPrice.toFixed(2)} | Signal strength: ${(strength * 100).toFixed(1)}%`);
        }
      }
    } else if (shortSignal) {
      signalsPassed++;
      if (position === 'long') {
        const exitPrice = roundToTick(fillStop('sell', currPrice));
        exitPosition(exitPrice, bar.timestamp, 'signal');
      }
      if (!position) {
        position = 'short';
        entryPrice = roundToTick(fillEntry('sell', currPrice));
        entryTime = bar.timestamp;
        lastTradeTime = Date.parse(bar.timestamp);
        dailyTrades.set(dateKey, dailyTradeCount + 1);

        const strength = calculateSignalStrength(
          1.0,
          (50 - rsiValue) / 50,
          adxValue.adx / 100,
          bar.volume / (avgVolume * 2),
          microstructure,
          volatilityOK,
          mtfAligned
        );

        if (trades.length < 10) {
          console.log(`SHORT entry @ ${entryPrice.toFixed(2)} | Signal strength: ${(strength * 100).toFixed(1)}%`);
        }
      }
    }

    if (rsiValue !== undefined) prevRsiValue = rsiValue;
  }

  // Close any open position
  if (position) {
    const lastBar = bars[bars.length - 1];
    const exitSide = position === 'long' ? 'sell' : 'buy';
    const exitPrice = roundToTick(fillStop(exitSide, lastBar.close));
    exitPosition(exitPrice, lastBar.timestamp, 'end_of_data');
  }

  // Calculate statistics
  const wins = trades.filter(trade => trade.pnl > 0);
  const losses = trades.filter(trade => trade.pnl < 0);
  const avgWin = wins.length ? wins.reduce((sum, t) => sum + t.pnl, 0) / wins.length : 0;
  const avgLoss = losses.length ? losses.reduce((sum, t) => sum + t.pnl, 0) / losses.length : 0;
  const profitFactor = losses.length && avgLoss !== 0
    ? (wins.length * avgWin) / Math.abs(losses.length * avgLoss)
    : 0;

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
  console.log('ENHANCED BACKTEST RESULTS');
  console.log('='.repeat(80));
  console.log(`Total Signals: ${signalsBlocked + signalsPassed} | Blocked: ${signalsBlocked} | Passed: ${signalsPassed}`);
  console.log(`Filter Efficiency: ${((signalsBlocked / (signalsBlocked + signalsPassed)) * 100).toFixed(1)}% blocked`);
  console.log(`\nTrades: ${trades.length} | Win Rate: ${trades.length ? ((wins.length / trades.length) * 100).toFixed(1) : '0'}%`);
  console.log(`Realized PnL: ${formatCurrency(realizedPnL)} USD`);
  console.log(`Avg Win: ${formatCurrency(avgWin)} | Avg Loss: ${formatCurrency(avgLoss)}`);
  console.log(`Profit Factor: ${profitFactor.toFixed(2)}`);
  console.log(`Max Drawdown: ${formatCurrency(maxDrawdown)} USD`);
  console.log(`Avg Trades/Day: ${(trades.length / dailyTrades.size).toFixed(1)}`);

  if (trades.length) {
    console.log('\nRecent Trades:');
    trades.slice(-10).forEach(trade => {
      const duration = (Date.parse(trade.exitTime) - Date.parse(trade.entryTime)) / 1000;
      console.log(
        ` ${trade.side.toUpperCase().padEnd(5)} | Entry: ${trade.entryPrice.toFixed(2)} @ ${new Date(trade.entryTime).toLocaleTimeString()} | ` +
        `Exit: ${trade.exitPrice.toFixed(2)} @ ${new Date(trade.exitTime).toLocaleTimeString()} | ` +
        `PnL: ${formatCurrency(trade.pnl)} | Duration: ${duration}s | Reason: ${trade.exitReason}`
      );
    });
  }

  // Performance metrics
  const totalTradingDays = dailyTrades.size;
  const avgDailyPnL = realizedPnL / totalTradingDays;
  const winningDays = Array.from(dailyTrades.keys()).filter(date => {
    const dayTrades = trades.filter(t => t.entryTime.startsWith(date));
    const dayPnL = dayTrades.reduce((sum, t) => sum + t.pnl, 0);
    return dayPnL > 0;
  }).length;

  console.log('\n' + '='.repeat(80));
  console.log('PERFORMANCE METRICS');
  console.log('='.repeat(80));
  console.log(`Trading Days: ${totalTradingDays}`);
  console.log(`Winning Days: ${winningDays} (${((winningDays / totalTradingDays) * 100).toFixed(1)}%)`);
  console.log(`Avg Daily PnL: ${formatCurrency(avgDailyPnL)}`);
  console.log(`Sharpe Ratio Estimate: ${avgDailyPnL > 0 && maxDrawdown > 0 ? (avgDailyPnL / (maxDrawdown / totalTradingDays)).toFixed(2) : 'N/A'}`);
  console.log('='.repeat(80));
}

runBacktest().catch(err => {
  console.error('Enhanced backtest failed:', err);
  process.exit(1);
});