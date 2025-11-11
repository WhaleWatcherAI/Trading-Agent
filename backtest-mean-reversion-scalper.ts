import { RSI, BollingerBands, EMA, VWAP, ATR, StochasticRSI } from 'technicalindicators';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
} from './lib/topstepx';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

interface ScalperConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;

  // Bollinger Band Settings
  bbPeriod: number;
  bbStdDev: number;
  bbSqueezeThreshold: number;  // Min distance between bands for entry

  // Mean Reversion Settings
  rsiPeriod: number;
  rsiOversoldExtreme: number;  // Below this = extreme oversold
  rsiOverboughtExtreme: number; // Above this = extreme overbought

  // Volume Settings
  volumeLookback: number;
  minVolumeSurge: number;  // Multiplier of average volume

  // Risk Management (ULTRA TIGHT for scalping)
  stopLossTicks: number;
  takeProfitTicks: number;

  // Trade Management
  maxPositionsPerDay: number;
  maxConsecutiveLosses: number;
  minSecondsBetweenTrades: number;

  // Session Filter
  tradeOnlyPrimeSessions: boolean;

  // Position Sizing
  numberOfContracts: number;
  commissionPerSide: number;
}

interface ScalpTrade {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  grossPnl: number;
  fees: number;
  exitReason: 'target' | 'stop' | 'eod';
  holdTime: number;  // seconds
  entryReason: string;
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;

// Prime scalping hours (highest liquidity)
const SCALPING_SESSIONS = [
  { start: 8 * 60 + 30, end: 11 * 60 },     // Morning session
  { start: 13 * 60 + 30, end: 15 * 60 },    // Afternoon session
];

const DEFAULT_SYMBOL = process.env.SCALPER_SYMBOL || 'ESZ5';
const DEFAULT_CONTRACT_ID = process.env.SCALPER_CONTRACT_ID;

const CONFIG: ScalperConfig = {
  symbol: DEFAULT_SYMBOL,
  contractId: DEFAULT_CONTRACT_ID,
  start: process.env.SCALPER_START || new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.SCALPER_END || new Date().toISOString(),

  // Bollinger Bands
  bbPeriod: 20,
  bbStdDev: 2.0,
  bbSqueezeThreshold: 0.0015,  // 0.15% minimum band width

  // RSI Extremes
  rsiPeriod: 9,  // Faster RSI for 1-sec bars
  rsiOversoldExtreme: 25,
  rsiOverboughtExtreme: 75,

  // Volume
  volumeLookback: 50,
  minVolumeSurge: 2.0,  // 2x average volume

  // Ultra-tight scalping stops
  stopLossTicks: 3,      // 0.75 points on ES
  takeProfitTicks: 5,    // 1.25 points on ES (1.67:1 RR)

  // Trade Management
  maxPositionsPerDay: 20,
  maxConsecutiveLosses: 3,
  minSecondsBetweenTrades: 60,  // 1 minute minimum

  // Session
  tradeOnlyPrimeSessions: true,

  // Position
  numberOfContracts: Number(process.env.SCALPER_CONTRACTS || '2'),
  commissionPerSide: 0,  // Will be set per symbol
};

function toCentralTime(date: Date) {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isScalpingTime(timestamp: string | Date): boolean {
  if (!CONFIG.tradeOnlyPrimeSessions) return true;

  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const ctDate = toCentralTime(date);
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  return SCALPING_SESSIONS.some(session =>
    minutes >= session.start && minutes <= session.end
  );
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

function detectMicroStructure(bars: TopstepXFuturesBar[], lookback: number = 10): {
  buyPressure: number;
  sellPressure: number;
  momentum: number;
  spread: number;
} {
  if (bars.length < lookback) {
    return { buyPressure: 0, sellPressure: 0, momentum: 0, spread: 0 };
  }

  const recent = bars.slice(-lookback);
  let buyVolume = 0;
  let sellVolume = 0;
  let upMoves = 0;
  let downMoves = 0;

  for (let i = 1; i < recent.length; i++) {
    const priceChange = recent[i].close - recent[i - 1].close;

    if (priceChange > 0) {
      buyVolume += recent[i].volume;
      upMoves++;
    } else if (priceChange < 0) {
      sellVolume += recent[i].volume;
      downMoves++;
    }
  }

  const totalVolume = buyVolume + sellVolume;
  const buyPressure = totalVolume > 0 ? buyVolume / totalVolume : 0.5;
  const sellPressure = totalVolume > 0 ? sellVolume / totalVolume : 0.5;

  const momentum = (upMoves - downMoves) / lookback;

  // Average spread
  const spreads = recent.map(b => (b.high - b.low) / b.close);
  const spread = spreads.reduce((a, b) => a + b, 0) / spreads.length;

  return { buyPressure, sellPressure, momentum, spread };
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

async function runMeanReversionBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('HIGH-FREQUENCY MEAN REVERSION SCALPER - 1 SECOND BARS');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${new Date(CONFIG.start).toLocaleDateString()} -> ${new Date(CONFIG.end).toLocaleDateString()}`);
  console.log('\nStrategy: Fade extreme moves with volume confirmation');
  console.log(`- Bollinger Bands: ${CONFIG.bbPeriod} period, ${CONFIG.bbStdDev} StdDev`);
  console.log(`- RSI Extremes: <${CONFIG.rsiOversoldExtreme} or >${CONFIG.rsiOverboughtExtreme}`);
  console.log(`- Volume Surge: ${CONFIG.minVolumeSurge}x average`);
  console.log(`- Risk: ${CONFIG.stopLossTicks} tick stop, ${CONFIG.takeProfitTicks} tick target`);
  console.log(`- Max Daily Trades: ${CONFIG.maxPositionsPerDay}`);
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
  const highs: number[] = [];
  const lows: number[] = [];
  const volumes: number[] = [];

  let position: 'long' | 'short' | null = null;
  let entryPrice = 0;
  let entryTime = '';
  let entryIndex = 0;

  const trades: ScalpTrade[] = [];
  let consecutiveLosses = 0;
  let lastTradeTime = 0;
  const dailyTrades = new Map<string, number>();

  // Indicators
  const rsiIndicator = new RSI({ period: CONFIG.rsiPeriod, values: [] });
  const bbIndicator = new BollingerBands({
    period: CONFIG.bbPeriod,
    stdDev: CONFIG.bbStdDev,
    values: [],
  });
  const stochRSI = new StochasticRSI({
    values: [],
    rsiPeriod: 14,
    stochasticPeriod: 14,
    kPeriod: 3,
    dPeriod: 3,
  });

  // Signal tracking
  let signalsAnalyzed = 0;
  let signalsTaken = 0;
  const rejectionReasons = new Map<string, number>();

  const closePosition = (exitPrice: number, bar: TopstepXFuturesBar, reason: 'target' | 'stop' | 'eod') => {
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
      entryReason: '',
    });

    if (netPnL < 0) {
      consecutiveLosses++;
    } else {
      consecutiveLosses = 0;
    }

    position = null;
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
    highs.push(bar.high);
    lows.push(bar.low);
    volumes.push(bar.volume);

    // Keep bounded
    if (closes.length > 500) {
      closes.shift();
      highs.shift();
      lows.shift();
      volumes.shift();
    }

    // Update indicators
    const rsiValue = rsiIndicator.nextValue(bar.close);
    const bbValue = bbIndicator.nextValue(bar.close);
    const stochValue = stochRSI.nextValue(bar.close);

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
      const stopPrice = roundToTick(entryPrice - direction * CONFIG.stopLossTicks * tickSize);
      const targetPrice = roundToTick(entryPrice + direction * CONFIG.takeProfitTicks * tickSize);

      // Check stop
      if ((direction === 1 && bar.low <= stopPrice) ||
          (direction === -1 && bar.high >= stopPrice)) {
        closePosition(stopPrice, bar, 'stop');
        continue;
      }

      // Check target
      if ((direction === 1 && bar.high >= targetPrice) ||
          (direction === -1 && bar.low <= targetPrice)) {
        closePosition(targetPrice, bar, 'target');
        continue;
      }

      continue;  // Skip signal detection while in position
    }

    // Need enough data
    if (!rsiValue || !bbValue || !stochValue || volumes.length < CONFIG.volumeLookback) {
      continue;
    }

    // Calculate volume metrics
    const avgVolume = volumes.slice(-CONFIG.volumeLookback).reduce((a, b) => a + b, 0) / CONFIG.volumeLookback;
    const volumeSurge = bar.volume / avgVolume;

    // Calculate band squeeze
    const bandWidth = (bbValue.upper - bbValue.lower) / bbValue.middle;

    // Microstructure
    const micro = detectMicroStructure(bars.slice(Math.max(0, i - 10), i + 1), 10);

    // MEAN REVERSION SIGNALS

    // Long Signal: Extreme oversold bounce
    const longSignal =
      bar.close < bbValue.lower &&                          // Below lower band
      rsiValue < CONFIG.rsiOversoldExtreme &&              // Extreme oversold RSI
      volumeSurge >= CONFIG.minVolumeSurge &&              // Volume surge
      micro.sellPressure > 0.7 &&                          // Heavy selling = exhaustion
      stochValue.k < 20 &&                                 // Stochastic oversold
      bandWidth > CONFIG.bbSqueezeThreshold;               // Not in squeeze

    // Short Signal: Extreme overbought fade
    const shortSignal =
      bar.close > bbValue.upper &&                         // Above upper band
      rsiValue > CONFIG.rsiOverboughtExtreme &&            // Extreme overbought RSI
      volumeSurge >= CONFIG.minVolumeSurge &&              // Volume surge
      micro.buyPressure > 0.7 &&                           // Heavy buying = exhaustion
      stochValue.k > 80 &&                                 // Stochastic overbought
      bandWidth > CONFIG.bbSqueezeThreshold;               // Not in squeeze

    if (!longSignal && !shortSignal) continue;

    signalsAnalyzed++;

    // FILTERS
    let canTrade = true;
    let rejectReason = '';

    // Check consecutive losses
    if (consecutiveLosses >= CONFIG.maxConsecutiveLosses) {
      canTrade = false;
      rejectReason = 'consecutive_losses';
    }

    // Check daily limit
    if (canTrade && (dailyTrades.get(dateKey) || 0) >= CONFIG.maxPositionsPerDay) {
      canTrade = false;
      rejectReason = 'daily_limit';
    }

    // Check time since last trade
    if (canTrade) {
      const timeSinceLast = (Date.parse(bar.timestamp) - lastTradeTime) / 1000;
      if (timeSinceLast < CONFIG.minSecondsBetweenTrades) {
        canTrade = false;
        rejectReason = 'too_soon';
      }
    }

    // Check session filter
    if (canTrade && CONFIG.tradeOnlyPrimeSessions && !isScalpingTime(bar.timestamp)) {
      canTrade = false;
      rejectReason = 'outside_session';
    }

    // Track rejections
    if (!canTrade) {
      rejectionReasons.set(rejectReason, (rejectionReasons.get(rejectReason) || 0) + 1);
      continue;
    }

    // EXECUTE TRADE
    signalsTaken++;

    if (longSignal) {
      position = 'long';
      entryPrice = roundToTick(bar.close);
      entryTime = bar.timestamp;
      entryIndex = i;
      lastTradeTime = Date.parse(bar.timestamp);
      dailyTrades.set(dateKey, (dailyTrades.get(dateKey) || 0) + 1);

      if (trades.length < 10) {
        console.log(
          `LONG  @ ${entryPrice.toFixed(2)} | ${new Date(bar.timestamp).toLocaleTimeString()} | ` +
          `RSI: ${rsiValue.toFixed(0)} | Vol: ${volumeSurge.toFixed(1)}x | Below BB`
        );
      }
    } else if (shortSignal) {
      position = 'short';
      entryPrice = roundToTick(bar.close);
      entryTime = bar.timestamp;
      entryIndex = i;
      lastTradeTime = Date.parse(bar.timestamp);
      dailyTrades.set(dateKey, (dailyTrades.get(dateKey) || 0) + 1);

      if (trades.length < 10) {
        console.log(
          `SHORT @ ${entryPrice.toFixed(2)} | ${new Date(bar.timestamp).toLocaleTimeString()} | ` +
          `RSI: ${rsiValue.toFixed(0)} | Vol: ${volumeSurge.toFixed(1)}x | Above BB`
        );
      }
    }
  }

  // Close any open position
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

  // Drawdown calculation
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

  // Hold time analysis
  const avgHoldTime = trades.length
    ? trades.reduce((sum, t) => sum + t.holdTime, 0) / trades.length
    : 0;

  // Exit reason breakdown
  const exitReasons = new Map<string, number>();
  trades.forEach(t => {
    exitReasons.set(t.exitReason, (exitReasons.get(t.exitReason) || 0) + 1);
  });

  // Display results
  console.log('\n' + '='.repeat(80));
  console.log('MEAN REVERSION SCALPER RESULTS');
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
  console.log(`Signals Analyzed: ${signalsAnalyzed}`);
  console.log(`Signals Taken: ${signalsTaken} (${((signalsTaken/signalsAnalyzed)*100).toFixed(1)}%)`);

  console.log('\n🎯 EXIT BREAKDOWN:');
  exitReasons.forEach((count, reason) => {
    const pct = (count / trades.length) * 100;
    console.log(`- ${reason}: ${count} (${pct.toFixed(1)}%)`);
  });

  console.log('\n🚫 REJECTION REASONS:');
  rejectionReasons.forEach((count, reason) => {
    console.log(`- ${reason}: ${count}`);
  });

  // Show sample trades
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

  // Success criteria check
  const meetsWinRate = winRate >= 60;
  const meetsProfitFactor = profitFactor >= 2.0;
  const meetsDrawdown = drawdownRatio < 50;
  const isMultipleTrades = trades.length / dailyTrades.size >= 3;

  console.log('✅ SUCCESS CRITERIA:');
  console.log(`- Win Rate > 60%: ${meetsWinRate ? '✅' : '❌'} (${winRate.toFixed(1)}%)`);
  console.log(`- Profit Factor > 2: ${meetsProfitFactor ? '✅' : '❌'} (${profitFactor.toFixed(2)})`);
  console.log(`- Drawdown < 50%: ${meetsDrawdown ? '✅' : '❌'} (${drawdownRatio.toFixed(1)}%)`);
  console.log(`- Multiple Trades/Day: ${isMultipleTrades ? '✅' : '❌'} (${(trades.length/dailyTrades.size).toFixed(1)})`);

  const allCriteriaMet = meetsWinRate && meetsProfitFactor && meetsDrawdown && isMultipleTrades;

  console.log('\n' + '='.repeat(80));
  console.log(`FINAL VERDICT: ${allCriteriaMet ? '🎯 READY FOR LIVE TRADING' : '❌ NEEDS OPTIMIZATION'}`);
  console.log('='.repeat(80) + '\n');
}

runMeanReversionBacktest().catch(err => {
  console.error('Mean reversion backtest failed:', err);
  process.exit(1);
});