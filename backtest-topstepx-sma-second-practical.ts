import { RSI, ATR, EMA } from 'technicalindicators';
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
  fastSMA: number;
  slowSMA: number;
  rsiPeriod: number;
  rsiOversold: number;
  rsiOverbought: number;
  minVolume: number;
  maxDailyTrades: number;
  minSecondsBetweenTrades: number;
  stopLossTicks: number;
  takeProfitTicks: number;
  trailingStopTicks: number;
  commissionPerSide: number;
  numberOfContracts: number;
  useVolumeFilter: boolean;
  useMomentumFilter: boolean;
  useTimeFilter: boolean;
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
  exitReason: 'target' | 'stop' | 'trailing' | 'signal' | 'session' | 'eod';
  bars: number;
  maxProfit: number;
  maxLoss: number;
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_DAYS = 5;

// High probability trading windows (CT)
const TRADING_WINDOWS = [
  { start: 8 * 60 + 30, end: 11 * 60 + 30 },  // Morning 8:30-11:30 CT
  { start: 13 * 60, end: 15 * 60 },           // Afternoon 1:00-3:00 CT
];

const DEFAULT_SYMBOL = process.env.TOPSTEPX_SECOND_SMA_SYMBOL || 'ESZ5';
const DEFAULT_CONTRACT_ID = process.env.TOPSTEPX_SECOND_SMA_CONTRACT_ID;

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_SYMBOL,
  contractId: DEFAULT_CONTRACT_ID,
  start: process.env.TOPSTEPX_START || new Date(Date.now() - DEFAULT_DAYS * DAY_MS).toISOString(),
  end: process.env.TOPSTEPX_END || new Date().toISOString(),

  // SMA periods (in seconds)
  fastSMA: Number(process.env.TOPSTEPX_FAST_SMA || '500'),    // ~8 minutes
  slowSMA: Number(process.env.TOPSTEPX_SLOW_SMA || '1500'),   // ~25 minutes

  // RSI settings
  rsiPeriod: Number(process.env.TOPSTEPX_RSI_PERIOD || '50'),
  rsiOversold: Number(process.env.TOPSTEPX_RSI_OVERSOLD || '45'),
  rsiOverbought: Number(process.env.TOPSTEPX_RSI_OVERBOUGHT || '55'),

  // Trade management
  stopLossTicks: Number(process.env.TOPSTEPX_STOP_TICKS || '8'),      // 2 points for ES
  takeProfitTicks: Number(process.env.TOPSTEPX_TARGET_TICKS || '16'), // 4 points for ES
  trailingStopTicks: Number(process.env.TOPSTEPX_TRAILING_TICKS || '6'),

  // Risk controls
  minVolume: Number(process.env.TOPSTEPX_MIN_VOLUME || '50'),
  maxDailyTrades: Number(process.env.TOPSTEPX_MAX_DAILY_TRADES || '5'),
  minSecondsBetweenTrades: Number(process.env.TOPSTEPX_MIN_SECONDS || '600'), // 10 minutes

  // Position sizing
  numberOfContracts: Number(process.env.TOPSTEPX_CONTRACTS || '2'),
  commissionPerSide: process.env.TOPSTEPX_COMMISSION
    ? Number(process.env.TOPSTEPX_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_CONTRACT_ID, DEFAULT_SYMBOL]),

  // Filters
  useVolumeFilter: process.env.TOPSTEPX_USE_VOLUME !== 'false',
  useMomentumFilter: process.env.TOPSTEPX_USE_MOMENTUM !== 'false',
  useTimeFilter: process.env.TOPSTEPX_USE_TIME !== 'false',
};

function toCentralTime(date: Date) {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isInTradingWindow(timestamp: string | Date): boolean {
  if (!CONFIG.useTimeFilter) return true;

  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const ctDate = toCentralTime(date);
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  return TRADING_WINDOWS.some(window =>
    minutes >= window.start && minutes <= window.end
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

function calculateSMA(values: number[], period: number): number | null {
  if (values.length < period) return null;
  const sum = values.slice(-period).reduce((acc, val) => acc + val, 0);
  return sum / period;
}

function calculateMomentum(values: number[], period: number = 20): number {
  if (values.length < period) return 0;
  const oldPrice = values[values.length - period];
  const newPrice = values[values.length - 1];
  return ((newPrice - oldPrice) / oldPrice) * 100;
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
  const CHUNK_MS = 4 * 60 * 60 * 1000; // 4 hours

  while (cursor < endDate) {
    const chunkEnd = new Date(Math.min(cursor.getTime() + CHUNK_MS, endDate.getTime()));
    console.log(`Fetching: ${cursor.toISOString().slice(11, 19)} -> ${chunkEnd.toISOString().slice(11, 19)}`);

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

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('PRACTICAL SMA CROSSOVER STRATEGY - 1 SECOND BARS');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${new Date(CONFIG.start).toLocaleDateString()} -> ${new Date(CONFIG.end).toLocaleDateString()}`);
  console.log(`\nStrategy Parameters:`);
  console.log(`- Fast SMA: ${CONFIG.fastSMA}s (${(CONFIG.fastSMA/60).toFixed(1)} min)`);
  console.log(`- Slow SMA: ${CONFIG.slowSMA}s (${(CONFIG.slowSMA/60).toFixed(1)} min)`);
  console.log(`- RSI: ${CONFIG.rsiPeriod} period | OS: ${CONFIG.rsiOversold} | OB: ${CONFIG.rsiOverbought}`);
  console.log(`\nRisk Management:`);
  console.log(`- Stop: ${CONFIG.stopLossTicks} ticks | Target: ${CONFIG.takeProfitTicks} ticks | Trailing: ${CONFIG.trailingStopTicks} ticks`);
  console.log(`- Contracts: ${CONFIG.numberOfContracts} | Commission: $${CONFIG.commissionPerSide}/side`);
  console.log(`- Max Daily Trades: ${CONFIG.maxDailyTrades} | Min Gap: ${CONFIG.minSecondsBetweenTrades}s`);
  console.log(`\nFilters: Volume[${CONFIG.useVolumeFilter}] Momentum[${CONFIG.useMomentumFilter}] Time[${CONFIG.useTimeFilter}]`);
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

  const roundToTick = (value: number) => Math.round(value / tickSize) * tickSize;

  console.log(`\nContract: ${metadata.name} (${contractId})`);
  console.log(`Tick Size: ${tickSize} | Value/Tick: $${multiplier}`);

  // Fetch bars
  console.log('\nFetching 1-second bars...');
  const bars = await fetchSecondBarsInChunks(contractId, CONFIG.start, CONFIG.end);

  if (bars.length === 0) {
    throw new Error('No bars returned');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length.toLocaleString()} bars\n`);

  // Initialize tracking
  const closes: number[] = [];
  const volumes: number[] = [];
  let position: 'long' | 'short' | null = null;
  let entryPrice = 0;
  let entryTime = '';
  let entryBar = 0;
  let positionHighWaterMark = 0;
  let positionLowWaterMark = 0;
  const trades: TradeRecord[] = [];
  let lastTradeTime = 0;
  const dailyTrades = new Map<string, number>();

  // Indicators
  const rsiIndicator = new RSI({ period: CONFIG.rsiPeriod, values: [] });
  const atrIndicator = new ATR({ period: 14, high: [], low: [], close: [] });

  // Signal tracking
  let totalSignals = 0;
  let filteredSignals = 0;
  let volumeFiltered = 0;
  let momentumFiltered = 0;
  let timeFiltered = 0;
  let rsiFiltered = 0;
  let limitFiltered = 0;
  let gapFiltered = 0;

  const closePosition = (
    exitPrice: number,
    bar: TopstepXFuturesBar,
    reason: TradeRecord['exitReason']
  ) => {
    if (!position) return;

    const direction = position === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - entryPrice) * direction * multiplier * CONFIG.numberOfContracts;
    const fees = CONFIG.commissionPerSide * 2 * CONFIG.numberOfContracts;
    const netPnL = rawPnL - fees;

    const maxProfit = direction === 1
      ? (positionHighWaterMark - entryPrice) * multiplier * CONFIG.numberOfContracts
      : (entryPrice - positionLowWaterMark) * multiplier * CONFIG.numberOfContracts;

    const maxLoss = direction === 1
      ? (entryPrice - positionLowWaterMark) * multiplier * CONFIG.numberOfContracts
      : (positionHighWaterMark - entryPrice) * multiplier * CONFIG.numberOfContracts;

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
      bars: entryBar,
      maxProfit,
      maxLoss,
    });

    position = null;
    entryPrice = 0;
    entryTime = '';
    entryBar = 0;
    positionHighWaterMark = 0;
    positionLowWaterMark = 0;
  };

  // Main loop
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];

    // Update daily trade count
    const dateKey = new Date(bar.timestamp).toISOString().split('T')[0];
    if (!dailyTrades.has(dateKey)) {
      dailyTrades.set(dateKey, 0);
    }

    // Add data
    closes.push(bar.close);
    volumes.push(bar.volume);

    // Keep arrays bounded
    if (closes.length > CONFIG.slowSMA + 100) {
      closes.shift();
      volumes.shift();
    }

    // Update indicators
    const rsiValue = rsiIndicator.nextValue(bar.close);
    const atrValue = atrIndicator.nextValue({
      high: bar.high,
      low: bar.low,
      close: bar.close,
    });

    // Check if trading allowed
    if (!isTradingAllowed(bar.timestamp)) {
      if (position) {
        closePosition(bar.close, bar, 'session');
      }
      continue;
    }

    // Manage existing position
    if (position) {
      entryBar++;

      // Update watermarks
      positionHighWaterMark = Math.max(positionHighWaterMark, bar.high);
      positionLowWaterMark = Math.min(positionLowWaterMark, bar.low);

      const direction = position === 'long' ? 1 : -1;
      const stopPrice = roundToTick(entryPrice - direction * CONFIG.stopLossTicks * tickSize);
      const targetPrice = roundToTick(entryPrice + direction * CONFIG.takeProfitTicks * tickSize);

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

      // Trailing stop (after minimum profit)
      if (CONFIG.trailingStopTicks > 0) {
        const profitTicks = direction === 1
          ? (bar.close - entryPrice) / tickSize
          : (entryPrice - bar.close) / tickSize;

        if (profitTicks >= CONFIG.trailingStopTicks) {
          const trailingStop = direction === 1
            ? positionHighWaterMark - CONFIG.trailingStopTicks * tickSize
            : positionLowWaterMark + CONFIG.trailingStopTicks * tickSize;

          if ((direction === 1 && bar.close <= trailingStop) ||
              (direction === -1 && bar.close >= trailingStop)) {
            closePosition(bar.close, bar, 'trailing');
            continue;
          }
        }
      }
    }

    // Calculate SMAs
    const fastSMA = calculateSMA(closes, CONFIG.fastSMA);
    const slowSMA = calculateSMA(closes, CONFIG.slowSMA);

    if (!fastSMA || !slowSMA || !rsiValue) continue;

    // Detect crossovers
    const prevFastSMA = closes.length > CONFIG.fastSMA + 1
      ? calculateSMA(closes.slice(0, -1), CONFIG.fastSMA)
      : null;
    const prevSlowSMA = closes.length > CONFIG.slowSMA + 1
      ? calculateSMA(closes.slice(0, -1), CONFIG.slowSMA)
      : null;

    if (!prevFastSMA || !prevSlowSMA) continue;

    const crossUp = prevFastSMA <= prevSlowSMA && fastSMA > slowSMA;
    const crossDown = prevFastSMA >= prevSlowSMA && fastSMA < slowSMA;

    if (!crossUp && !crossDown) continue;

    totalSignals++;

    // Apply filters
    let signalValid = true;
    let filterReason = '';

    // 1. Volume filter
    if (CONFIG.useVolumeFilter) {
      const avgVolume = volumes.slice(-100).reduce((a, b) => a + b, 0) / 100;
      if (bar.volume < CONFIG.minVolume || bar.volume < avgVolume * 0.8) {
        signalValid = false;
        volumeFiltered++;
        filterReason = 'volume';
      }
    }

    // 2. RSI filter
    if (signalValid) {
      if (crossUp && rsiValue <= CONFIG.rsiOversold) {
        signalValid = false;
        rsiFiltered++;
        filterReason = 'rsi_oversold';
      } else if (crossDown && rsiValue >= CONFIG.rsiOverbought) {
        signalValid = false;
        rsiFiltered++;
        filterReason = 'rsi_overbought';
      }
    }

    // 3. Momentum filter
    if (signalValid && CONFIG.useMomentumFilter) {
      const momentum = calculateMomentum(closes, 50);
      if (Math.abs(momentum) < 0.05) { // Need at least 0.05% momentum
        signalValid = false;
        momentumFiltered++;
        filterReason = 'momentum';
      }
    }

    // 4. Time window filter
    if (signalValid && CONFIG.useTimeFilter) {
      if (!isInTradingWindow(bar.timestamp)) {
        signalValid = false;
        timeFiltered++;
        filterReason = 'time_window';
      }
    }

    // 5. Daily trade limit
    if (signalValid) {
      const todayTrades = dailyTrades.get(dateKey) || 0;
      if (todayTrades >= CONFIG.maxDailyTrades) {
        signalValid = false;
        limitFiltered++;
        filterReason = 'daily_limit';
      }
    }

    // 6. Minimum time between trades
    if (signalValid) {
      const timeSinceLast = (Date.parse(bar.timestamp) - lastTradeTime) / 1000;
      if (timeSinceLast < CONFIG.minSecondsBetweenTrades) {
        signalValid = false;
        gapFiltered++;
        filterReason = 'time_gap';
      }
    }

    if (!signalValid) {
      filteredSignals++;
      continue;
    }

    // Execute trade
    if (crossUp && !position) {
      position = 'long';
      entryPrice = roundToTick(bar.close);
      entryTime = bar.timestamp;
      entryBar = 0;
      positionHighWaterMark = bar.high;
      positionLowWaterMark = bar.low;
      lastTradeTime = Date.parse(bar.timestamp);
      dailyTrades.set(dateKey, (dailyTrades.get(dateKey) || 0) + 1);

      if (trades.length < 10) {
        console.log(`LONG  @ ${entryPrice.toFixed(2)} | ${new Date(bar.timestamp).toLocaleTimeString()} | RSI: ${rsiValue.toFixed(0)}`);
      }
    } else if (crossDown && !position) {
      position = 'short';
      entryPrice = roundToTick(bar.close);
      entryTime = bar.timestamp;
      entryBar = 0;
      positionHighWaterMark = bar.high;
      positionLowWaterMark = bar.low;
      lastTradeTime = Date.parse(bar.timestamp);
      dailyTrades.set(dateKey, (dailyTrades.get(dateKey) || 0) + 1);

      if (trades.length < 10) {
        console.log(`SHORT @ ${entryPrice.toFixed(2)} | ${new Date(bar.timestamp).toLocaleTimeString()} | RSI: ${rsiValue.toFixed(0)}`);
      }
    } else if ((crossUp && position === 'short') || (crossDown && position === 'long')) {
      // Exit on opposite signal
      closePosition(bar.close, bar, 'signal');
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

  // Calculate drawdown
  let runningPnL = 0;
  let peakPnL = 0;
  let maxDrawdown = 0;
  let maxDrawdownPercent = 0;
  const initialCapital = 50000; // Assumed starting capital

  trades.forEach(trade => {
    runningPnL += trade.pnl;
    if (runningPnL > peakPnL) {
      peakPnL = runningPnL;
    }
    const drawdown = peakPnL - runningPnL;
    const drawdownPercent = peakPnL > 0 ? (drawdown / (initialCapital + peakPnL)) * 100 : 0;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
      maxDrawdownPercent = drawdownPercent;
    }
  });

  // Daily analysis
  const dailyPnL = new Map<string, number>();
  trades.forEach(trade => {
    const date = new Date(trade.entryTime).toISOString().split('T')[0];
    dailyPnL.set(date, (dailyPnL.get(date) || 0) + trade.pnl);
  });
  const winningDays = Array.from(dailyPnL.values()).filter(pnl => pnl > 0).length;
  const totalDays = dailyPnL.size;

  // Print results
  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST RESULTS');
  console.log('='.repeat(80));

  console.log('\nSignal Analysis:');
  console.log(`- Total Crossovers: ${totalSignals}`);
  console.log(`- Filtered Out: ${filteredSignals} (${((filteredSignals/totalSignals)*100).toFixed(1)}%)`);
  console.log(`  • Volume: ${volumeFiltered}`);
  console.log(`  • RSI: ${rsiFiltered}`);
  console.log(`  • Momentum: ${momentumFiltered}`);
  console.log(`  • Time Window: ${timeFiltered}`);
  console.log(`  • Daily Limit: ${limitFiltered}`);
  console.log(`  • Time Gap: ${gapFiltered}`);
  console.log(`- Trades Taken: ${trades.length}`);

  console.log('\nPerformance:');
  console.log(`- Total P&L: ${totalPnL >= 0 ? '+' : ''}$${totalPnL.toFixed(2)}`);
  console.log(`- Win Rate: ${winRate.toFixed(1)}% (${wins.length}W / ${losses.length}L)`);
  console.log(`- Avg Win: +$${avgWin.toFixed(2)}`);
  console.log(`- Avg Loss: $${avgLoss.toFixed(2)}`);
  console.log(`- Profit Factor: ${profitFactor.toFixed(2)}`);
  console.log(`- Max Drawdown: $${maxDrawdown.toFixed(2)} (${maxDrawdownPercent.toFixed(1)}%)`);

  console.log('\nDaily Statistics:');
  console.log(`- Trading Days: ${totalDays}`);
  console.log(`- Winning Days: ${winningDays} (${((winningDays/totalDays)*100).toFixed(1)}%)`);
  console.log(`- Avg Daily P&L: $${(totalPnL/totalDays).toFixed(2)}`);
  console.log(`- Avg Trades/Day: ${(trades.length/totalDays).toFixed(1)}`);

  // Exit reason analysis
  const exitReasons = new Map<string, number>();
  trades.forEach(t => {
    exitReasons.set(t.exitReason, (exitReasons.get(t.exitReason) || 0) + 1);
  });

  console.log('\nExit Analysis:');
  exitReasons.forEach((count, reason) => {
    const pct = (count / trades.length) * 100;
    console.log(`- ${reason}: ${count} (${pct.toFixed(1)}%)`);
  });

  // Show sample trades
  if (trades.length > 0) {
    console.log('\nRecent Trades:');
    trades.slice(-10).forEach(trade => {
      const time = new Date(trade.entryTime).toLocaleTimeString();
      const duration = trade.bars;
      console.log(
        `${trade.side.toUpperCase().padEnd(5)} ${time} | ` +
        `${trade.entryPrice.toFixed(2)} → ${trade.exitPrice.toFixed(2)} | ` +
        `P&L: ${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)} | ` +
        `${duration}s | ${trade.exitReason}`
      );
    });
  }

  console.log('\n' + '='.repeat(80));
  console.log(`Strategy: ${totalPnL >= 0 ? '✓ PROFITABLE' : '✗ UNPROFITABLE'}`);
  console.log('='.repeat(80));
}

runBacktest().catch(err => {
  console.error('Backtest failed:', err);
  process.exit(1);
});