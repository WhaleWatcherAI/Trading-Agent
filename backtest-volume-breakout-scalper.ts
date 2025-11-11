import { RSI, EMA, ATR, MACD, StochasticRSI } from 'technicalindicators';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
} from './lib/topstepx';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

interface VolumeBreakoutConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;

  // Support/Resistance Detection
  lookbackPeriod: number;        // Bars to look back for S/R levels
  minTouchesForLevel: number;     // Min times price must test a level
  levelTolerance: number;         // Tolerance in ticks for S/R levels

  // Volume Breakout Settings
  volumeMultiplier: number;       // Min volume vs average for breakout
  volumeLookback: number;         // Period for average volume

  // Momentum Confirmation
  emaFast: number;
  emaSlow: number;
  rsiPeriod: number;
  rsiNeutralMin: number;
  rsiNeutralMax: number;

  // Risk Management
  stopLossTicks: number;
  takeProfitTicks: number;
  trailingActivationTicks: number;
  trailingStopTicks: number;

  // Trade Management
  maxTradesPerDay: number;
  minSecondsBetweenTrades: number;
  maxOpenPositions: number;

  // Position Sizing
  numberOfContracts: number;
  commissionPerSide: number;
}

interface BreakoutTrade {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  grossPnl: number;
  fees: number;
  exitReason: 'target' | 'stop' | 'trailing' | 'reversal' | 'eod';
  holdTime: number;
  entryVolume: number;
  breakoutLevel: number;
}

interface SupportResistanceLevel {
  price: number;
  touches: number;
  lastTested: number;  // Bar index
  type: 'support' | 'resistance';
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;

const DEFAULT_SYMBOL = process.env.BREAKOUT_SYMBOL || 'ESZ5';
const DEFAULT_CONTRACT_ID = process.env.BREAKOUT_CONTRACT_ID;

const CONFIG: VolumeBreakoutConfig = {
  symbol: DEFAULT_SYMBOL,
  contractId: DEFAULT_CONTRACT_ID,
  start: process.env.BREAKOUT_START || new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.BREAKOUT_END || new Date().toISOString(),

  // S/R Detection
  lookbackPeriod: 100,
  minTouchesForLevel: 2,
  levelTolerance: 2,  // 2 ticks tolerance

  // Volume Breakout
  volumeMultiplier: 1.5,
  volumeLookback: 30,

  // Momentum
  emaFast: 9,
  emaSlow: 21,
  rsiPeriod: 14,
  rsiNeutralMin: 35,
  rsiNeutralMax: 65,

  // Risk Management (Tight for scalping)
  stopLossTicks: 4,
  takeProfitTicks: 6,
  trailingActivationTicks: 4,
  trailingStopTicks: 2,

  // Trade Management
  maxTradesPerDay: 15,
  minSecondsBetweenTrades: 30,
  maxOpenPositions: 1,

  // Position
  numberOfContracts: Number(process.env.BREAKOUT_CONTRACTS || '2'),
  commissionPerSide: 0,
};

function toCentralTime(date: Date) {
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

function identifySupportResistance(
  bars: TopstepXFuturesBar[],
  lookback: number,
  tolerance: number,
  minTouches: number
): SupportResistanceLevel[] {
  if (bars.length < lookback) return [];

  const levels: SupportResistanceLevel[] = [];
  const recentBars = bars.slice(-lookback);

  // Find local highs and lows
  const pivots: { price: number; index: number; type: 'high' | 'low' }[] = [];

  for (let i = 2; i < recentBars.length - 2; i++) {
    const bar = recentBars[i];

    // Local high
    if (bar.high > recentBars[i - 1].high &&
        bar.high > recentBars[i - 2].high &&
        bar.high > recentBars[i + 1].high &&
        bar.high > recentBars[i + 2].high) {
      pivots.push({ price: bar.high, index: i, type: 'high' });
    }

    // Local low
    if (bar.low < recentBars[i - 1].low &&
        bar.low < recentBars[i - 2].low &&
        bar.low < recentBars[i + 1].low &&
        bar.low < recentBars[i + 2].low) {
      pivots.push({ price: bar.low, index: i, type: 'low' });
    }
  }

  // Group similar price levels
  const grouped = new Map<number, { touches: number; lastIndex: number; type: 'support' | 'resistance' }>();

  pivots.forEach(pivot => {
    let found = false;

    grouped.forEach((value, key) => {
      if (Math.abs(pivot.price - key) <= tolerance) {
        value.touches++;
        value.lastIndex = Math.max(value.lastIndex, pivot.index);
        found = true;
      }
    });

    if (!found) {
      grouped.set(pivot.price, {
        touches: 1,
        lastIndex: pivot.index,
        type: pivot.type === 'high' ? 'resistance' : 'support',
      });
    }
  });

  // Filter by minimum touches
  grouped.forEach((value, key) => {
    if (value.touches >= minTouches) {
      levels.push({
        price: key,
        touches: value.touches,
        lastTested: value.lastIndex,
        type: value.type,
      });
    }
  });

  return levels;
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

async function fetchSecondBarsInChunks(
  contractId: string,
  start: string,
  end: string,
): Promise<TopstepXFuturesBar[]> {
  const startDate = new Date(start);
  const endDate = new Date(end);
  const bars: TopstepXFuturesBar[] = [];
  let cursor = startDate;
  const CHUNK_MS = 4 * 60 * 60 * 1000;

  while (cursor < endDate) {
    const chunkEnd = new Date(Math.min(cursor.getTime() + CHUNK_MS, endDate.getTime()));
    console.log(`  Fetching: ${cursor.toISOString().slice(11, 19)} -> ${chunkEnd.toISOString().slice(11, 19)}`);

    const chunkBars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: cursor.toISOString(),
      endTime: chunkEnd.toISOString(),
      unit: 1,
      unitNumber: 1,
      limit: 20000,
    });

    bars.push(...chunkBars);
    cursor = new Date(chunkEnd.getTime() + 1000);
    await new Promise(r => setTimeout(r, 500));
  }

  return bars.reverse();
}

async function runVolumeBreakoutBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('VOLUME BREAKOUT SCALPER - SUPPORT/RESISTANCE BREAKS');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${new Date(CONFIG.start).toLocaleDateString()} -> ${new Date(CONFIG.end).toLocaleDateString()}`);
  console.log('\nStrategy: Trade volume breakouts at key S/R levels');
  console.log(`- S/R Lookback: ${CONFIG.lookbackPeriod} bars`);
  console.log(`- Volume Threshold: ${CONFIG.volumeMultiplier}x average`);
  console.log(`- EMA: ${CONFIG.emaFast}/${CONFIG.emaSlow}`);
  console.log(`- Risk: ${CONFIG.stopLossTicks} tick stop, ${CONFIG.takeProfitTicks} tick target`);
  console.log(`- Trailing: Activates at ${CONFIG.trailingActivationTicks} ticks`);
  console.log('='.repeat(80));

  // Fetch metadata
  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey);

  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  const contractId = metadata.id;
  const tickSize = metadata.tickSize;
  const multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || 50;
  CONFIG.commissionPerSide = inferFuturesCommissionPerSide([contractId, CONFIG.symbol]);

  const roundToTick = (value: number) => Math.round(value / tickSize) * tickSize;

  console.log(`\nContract: ${metadata.name}`);
  console.log(`Tick Size: ${tickSize} | Multiplier: $${multiplier}/tick`);
  console.log(`Commission: $${CONFIG.commissionPerSide}/side\n`);

  // Fetch data
  console.log('Fetching 1-second bars...');
  const bars = await fetchSecondBarsInChunks(contractId, CONFIG.start, CONFIG.end);

  if (bars.length === 0) {
    throw new Error('No bars returned');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length.toLocaleString()} bars\n`);

  // Initialize
  const closes: number[] = [];
  const volumes: number[] = [];

  let position: 'long' | 'short' | null = null;
  let entryPrice = 0;
  let entryTime = '';
  let entryIndex = 0;
  let stopPrice = 0;
  let targetPrice = 0;
  let trailingStop = 0;
  let highWaterMark = 0;

  const trades: BreakoutTrade[] = [];
  let lastTradeTime = 0;
  const dailyTrades = new Map<string, number>();

  // Indicators
  const rsiIndicator = new RSI({ period: CONFIG.rsiPeriod, values: [] });
  const macdIndicator = new MACD({
    fastPeriod: 12,
    slowPeriod: 26,
    signalPeriod: 9,
    SimpleMAOscillator: false,
    SimpleMASignal: false,
    values: [],
  });

  // Signal tracking
  let totalBreakouts = 0;
  let breakoutsTaken = 0;

  const closePosition = (exitPrice: number, bar: TopstepXFuturesBar, reason: BreakoutTrade['exitReason']) => {
    if (!position) return;

    const direction = position === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - entryPrice) * direction * multiplier * CONFIG.numberOfContracts;
    const fees = CONFIG.commissionPerSide * 2 * CONFIG.numberOfContracts;
    const netPnL = rawPnL - fees;
    const holdTime = Math.round((Date.parse(bar.timestamp) - Date.parse(entryTime)) / 1000);

    trades.push({
      entryTime,
      exitTime: bar.timestamp,
      side: position,
      entryPrice,
      exitPrice,
      pnl: netPnL,
      grossPnl: rawPnL,
      fees,
      exitReason: reason,
      holdTime,
      entryVolume: 0,
      breakoutLevel: 0,
    });

    position = null;
    highWaterMark = 0;
    trailingStop = 0;
  };

  // Main loop
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const dateKey = new Date(bar.timestamp).toISOString().split('T')[0];

    if (!dailyTrades.has(dateKey)) {
      dailyTrades.set(dateKey, 0);
    }

    // Add data
    closes.push(bar.close);
    volumes.push(bar.volume);

    if (closes.length > 200) {
      closes.shift();
      volumes.shift();
    }

    // Update indicators
    const rsiValue = rsiIndicator.nextValue(bar.close);
    const macdValue = macdIndicator.nextValue(bar.close);

    // Check trading allowed
    if (!isTradingAllowed(bar.timestamp)) {
      if (position) {
        closePosition(bar.close, bar, 'eod');
      }
      continue;
    }

    // Manage existing position
    if (position) {
      const direction = position === 'long' ? 1 : -1;

      // Update high water mark
      if (direction === 1) {
        highWaterMark = Math.max(highWaterMark, bar.high);
      } else {
        highWaterMark = Math.min(highWaterMark, bar.low);
      }

      // Check stop loss
      if ((direction === 1 && bar.low <= stopPrice) ||
          (direction === -1 && bar.high >= stopPrice)) {
        closePosition(stopPrice, bar, 'stop');
        continue;
      }

      // Check take profit
      if ((direction === 1 && bar.high >= targetPrice) ||
          (direction === -1 && bar.low <= targetPrice)) {
        closePosition(targetPrice, bar, 'target');
        continue;
      }

      // Trailing stop management
      const profitTicks = direction === 1
        ? (highWaterMark - entryPrice) / tickSize
        : (entryPrice - highWaterMark) / tickSize;

      if (profitTicks >= CONFIG.trailingActivationTicks) {
        const newTrailing = direction === 1
          ? highWaterMark - CONFIG.trailingStopTicks * tickSize
          : highWaterMark + CONFIG.trailingStopTicks * tickSize;

        if ((direction === 1 && newTrailing > trailingStop) ||
            (direction === -1 && newTrailing < trailingStop)) {
          trailingStop = newTrailing;
          stopPrice = trailingStop;  // Update stop to trailing
        }
      }

      continue;
    }

    // Need enough data
    if (closes.length < CONFIG.lookbackPeriod || !rsiValue || !macdValue) {
      continue;
    }

    // Identify S/R levels
    const levels = identifySupportResistance(
      bars.slice(Math.max(0, i - CONFIG.lookbackPeriod), i + 1),
      CONFIG.lookbackPeriod,
      CONFIG.levelTolerance * tickSize,
      CONFIG.minTouchesForLevel
    );

    if (levels.length === 0) continue;

    // Calculate EMAs
    const emaFast = calculateEMA(closes, CONFIG.emaFast);
    const emaSlow = calculateEMA(closes, CONFIG.emaSlow);

    if (!emaFast || !emaSlow) continue;

    // Calculate volume metrics
    const avgVolume = volumes.slice(-CONFIG.volumeLookback).reduce((a, b) => a + b, 0) / CONFIG.volumeLookback;
    const volumeBreakout = bar.volume >= avgVolume * CONFIG.volumeMultiplier;

    if (!volumeBreakout) continue;

    // BREAKOUT DETECTION

    let signal: 'long' | 'short' | null = null;
    let breakoutLevel = 0;

    // Check for resistance breakout (long signal)
    const resistanceLevels = levels.filter(l => l.type === 'resistance' && l.price < bar.close);
    if (resistanceLevels.length > 0) {
      const nearestResistance = resistanceLevels.reduce((prev, curr) =>
        Math.abs(curr.price - bar.close) < Math.abs(prev.price - bar.close) ? curr : prev
      );

      if (bar.close > nearestResistance.price &&
          bar.close - nearestResistance.price <= 2 * tickSize &&
          emaFast > emaSlow &&
          rsiValue >= CONFIG.rsiNeutralMin &&
          rsiValue <= CONFIG.rsiNeutralMax) {
        signal = 'long';
        breakoutLevel = nearestResistance.price;
      }
    }

    // Check for support breakdown (short signal)
    const supportLevels = levels.filter(l => l.type === 'support' && l.price > bar.close);
    if (!signal && supportLevels.length > 0) {
      const nearestSupport = supportLevels.reduce((prev, curr) =>
        Math.abs(curr.price - bar.close) < Math.abs(prev.price - bar.close) ? curr : prev
      );

      if (bar.close < nearestSupport.price &&
          nearestSupport.price - bar.close <= 2 * tickSize &&
          emaFast < emaSlow &&
          rsiValue >= CONFIG.rsiNeutralMin &&
          rsiValue <= CONFIG.rsiNeutralMax) {
        signal = 'short';
        breakoutLevel = nearestSupport.price;
      }
    }

    if (!signal) continue;

    totalBreakouts++;

    // Apply filters
    if ((dailyTrades.get(dateKey) || 0) >= CONFIG.maxTradesPerDay) continue;

    const timeSinceLast = (Date.parse(bar.timestamp) - lastTradeTime) / 1000;
    if (timeSinceLast < CONFIG.minSecondsBetweenTrades) continue;

    // MACD confirmation
    if (macdValue) {
      if (signal === 'long' && macdValue.MACD < macdValue.signal) continue;
      if (signal === 'short' && macdValue.MACD > macdValue.signal) continue;
    }

    // Execute trade
    breakoutsTaken++;

    position = signal;
    entryPrice = roundToTick(bar.close);
    entryTime = bar.timestamp;
    entryIndex = i;
    highWaterMark = bar.close;

    const direction = position === 'long' ? 1 : -1;
    stopPrice = roundToTick(entryPrice - direction * CONFIG.stopLossTicks * tickSize);
    targetPrice = roundToTick(entryPrice + direction * CONFIG.takeProfitTicks * tickSize);
    trailingStop = stopPrice;

    lastTradeTime = Date.parse(bar.timestamp);
    dailyTrades.set(dateKey, (dailyTrades.get(dateKey) || 0) + 1);

    if (trades.length < 10) {
      const typeStr = signal === 'long' ? 'Resistance Break' : 'Support Break';
      console.log(
        `${signal.toUpperCase().padEnd(5)} @ ${entryPrice.toFixed(2)} | ` +
        `${new Date(bar.timestamp).toLocaleTimeString()} | ` +
        `${typeStr} at ${breakoutLevel.toFixed(2)} | Vol: ${(bar.volume/avgVolume).toFixed(1)}x`
      );
    }
  }

  // Close final position
  if (position) {
    closePosition(bars[bars.length - 1].close, bars[bars.length - 1], 'eod');
  }

  // Calculate statistics
  const wins = trades.filter(t => t.pnl > 0);
  const losses = trades.filter(t => t.pnl < 0);
  const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0);
  const avgWin = wins.length ? wins.reduce((sum, t) => sum + t.pnl, 0) / wins.length : 0;
  const avgLoss = losses.length ? losses.reduce((sum, t) => sum + t.pnl, 0) / losses.length : 0;
  const winRate = trades.length ? (wins.length / trades.length) * 100 : 0;
  const profitFactor = Math.abs(avgLoss) > 0
    ? (wins.length * avgWin) / Math.abs(losses.length * avgLoss)
    : 0;

  // Calculate drawdown
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

  // Hold time stats
  const avgHoldTime = trades.length
    ? trades.reduce((sum, t) => sum + t.holdTime, 0) / trades.length
    : 0;

  // Exit analysis
  const exitReasons = new Map<string, number>();
  trades.forEach(t => {
    exitReasons.set(t.exitReason, (exitReasons.get(t.exitReason) || 0) + 1);
  });

  // Display results
  console.log('\n' + '='.repeat(80));
  console.log('VOLUME BREAKOUT SCALPER RESULTS');
  console.log('='.repeat(80));

  console.log('\n📊 PERFORMANCE METRICS:');
  console.log(`Total Trades: ${trades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(1)}% (${wins.length} wins / ${losses.length} losses)`);
  console.log(`Profit Factor: ${profitFactor.toFixed(2)}`);
  console.log(`Total P&L: ${totalPnL >= 0 ? '+' : ''}$${totalPnL.toFixed(2)}`);
  console.log(`Avg Win: +$${avgWin.toFixed(2)}`);
  console.log(`Avg Loss: -$${Math.abs(avgLoss).toFixed(2)}`);
  console.log(`Max Drawdown: $${maxDrawdown.toFixed(2)}`);

  const drawdownRatio = totalPnL > 0 ? (maxDrawdown / totalPnL) * 100 : 0;
  console.log(`Drawdown/Profit: ${drawdownRatio.toFixed(1)}%`);

  console.log('\n⏱️ TIMING ANALYSIS:');
  console.log(`Avg Hold Time: ${avgHoldTime.toFixed(0)} seconds`);
  console.log(`Breakouts Detected: ${totalBreakouts}`);
  console.log(`Breakouts Taken: ${breakoutsTaken} (${((breakoutsTaken/totalBreakouts)*100).toFixed(1)}%)`);
  console.log(`Trades Per Day: ${(trades.length / dailyTrades.size).toFixed(1)}`);

  console.log('\n🎯 EXIT BREAKDOWN:');
  exitReasons.forEach((count, reason) => {
    const pct = (count / trades.length) * 100;
    console.log(`- ${reason}: ${count} (${pct.toFixed(1)}%)`);
  });

  // Show recent trades
  if (trades.length > 0) {
    console.log('\n📝 RECENT TRADES:');
    trades.slice(-10).forEach(trade => {
      const time = new Date(trade.entryTime).toLocaleTimeString();
      const pnlStr = trade.pnl >= 0 ? `+$${trade.pnl.toFixed(2)}` : `-$${Math.abs(trade.pnl).toFixed(2)}`;

      console.log(
        `${trade.side.toUpperCase().padEnd(5)} ${time} | ` +
        `${trade.entryPrice.toFixed(2)} → ${trade.exitPrice.toFixed(2)} | ` +
        `P&L: ${pnlStr} | ${trade.holdTime}s | ${trade.exitReason}`
      );
    });
  }

  console.log('\n' + '='.repeat(80));

  // Success check
  const meetsWinRate = winRate >= 60;
  const meetsProfitFactor = profitFactor >= 2.0;
  const meetsDrawdown = drawdownRatio < 50;
  const isMultipleTrades = trades.length / Math.max(dailyTrades.size, 1) >= 3;

  console.log('✅ SUCCESS CRITERIA:');
  console.log(`- Win Rate > 60%: ${meetsWinRate ? '✅' : '❌'} (${winRate.toFixed(1)}%)`);
  console.log(`- Profit Factor > 2: ${meetsProfitFactor ? '✅' : '❌'} (${profitFactor.toFixed(2)})`);
  console.log(`- Drawdown < 50%: ${meetsDrawdown ? '✅' : '❌'} (${drawdownRatio.toFixed(1)}%)`);
  console.log(`- Multiple Trades/Day: ${isMultipleTrades ? '✅' : '❌'} (${(trades.length/Math.max(dailyTrades.size, 1)).toFixed(1)})`);

  const allCriteriaMet = meetsWinRate && meetsProfitFactor && meetsDrawdown && isMultipleTrades;

  console.log('\n' + '='.repeat(80));
  console.log(`FINAL VERDICT: ${allCriteriaMet ? '🎯 READY FOR LIVE TRADING' : '❌ NEEDS OPTIMIZATION'}`);
  console.log('='.repeat(80) + '\n');
}

runVolumeBreakoutBacktest().catch(err => {
  console.error('Volume breakout backtest failed:', err);
  process.exit(1);
});