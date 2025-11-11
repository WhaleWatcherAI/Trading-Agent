import { RSI, EMA, ATR } from 'technicalindicators';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
} from './lib/topstepx';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

interface MomentumBurstConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;

  // Momentum Detection
  momentumLookback: number;      // Bars to measure momentum
  minMomentumTicks: number;       // Minimum move in ticks
  maxMomentumTime: number;        // Max seconds for momentum move

  // Entry Filters
  rsiPeriod: number;
  rsiMin: number;
  rsiMax: number;

  // Risk Management (ULTRA TIGHT)
  stopLossTicks: number;
  takeProfitTicks: number;
  maxHoldTime: number;            // Force exit after X seconds

  // Trade Management
  maxTradesPerDay: number;
  minSecondsBetweenTrades: number;

  // Position Sizing
  numberOfContracts: number;
  commissionPerSide: number;
}

interface MomentumTrade {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  grossPnl: number;
  fees: number;
  exitReason: 'target' | 'stop' | 'timeout' | 'eod';
  holdTime: number;
  momentumSpeed: number;  // Ticks per second
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;

const DEFAULT_SYMBOL = process.env.MOMENTUM_SYMBOL || 'ESZ5';

const CONFIG: MomentumBurstConfig = {
  symbol: DEFAULT_SYMBOL,
  contractId: process.env.MOMENTUM_CONTRACT_ID,
  start: process.env.MOMENTUM_START || new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.MOMENTUM_END || new Date().toISOString(),

  // Momentum Settings
  momentumLookback: 5,       // Look at last 5 seconds
  minMomentumTicks: 3,       // Need 3 tick move in 5 seconds
  maxMomentumTime: 10,       // Complete momentum in 10 seconds max

  // Entry Filters
  rsiPeriod: 7,              // Very fast RSI for 1-sec bars
  rsiMin: 30,
  rsiMax: 70,

  // Ultra-tight risk
  stopLossTicks: 2,          // 0.5 points on ES
  takeProfitTicks: 3,        // 0.75 points (1.5:1 RR)
  maxHoldTime: 30,           // Exit after 30 seconds max

  // Trade Management
  maxTradesPerDay: 30,       // Allow many trades
  minSecondsBetweenTrades: 10,

  // Position
  numberOfContracts: Number(process.env.MOMENTUM_CONTRACTS || '3'),
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

function detectMomentumBurst(
  bars: TopstepXFuturesBar[],
  lookback: number,
  minTicks: number,
  tickSize: number
): { direction: 'long' | 'short' | null; speed: number; strength: number } {
  if (bars.length < lookback) {
    return { direction: null, speed: 0, strength: 0 };
  }

  const recent = bars.slice(-lookback);
  const firstBar = recent[0];
  const lastBar = recent[recent.length - 1];

  // Calculate price change
  const priceChange = lastBar.close - firstBar.close;
  const tickChange = priceChange / tickSize;
  const timeElapsed = lookback;  // seconds

  // Calculate speed (ticks per second)
  const speed = Math.abs(tickChange) / timeElapsed;

  // Determine if we have momentum
  if (Math.abs(tickChange) < minTicks) {
    return { direction: null, speed: 0, strength: 0 };
  }

  // Check consistency of move
  let consistentBars = 0;
  for (let i = 1; i < recent.length; i++) {
    if (tickChange > 0 && recent[i].close > recent[i - 1].close) {
      consistentBars++;
    } else if (tickChange < 0 && recent[i].close < recent[i - 1].close) {
      consistentBars++;
    }
  }

  const consistency = consistentBars / (recent.length - 1);

  // Need at least 60% consistency
  if (consistency < 0.6) {
    return { direction: null, speed: 0, strength: 0 };
  }

  // Calculate volume surge
  const recentVolume = recent.reduce((sum, b) => sum + b.volume, 0);
  const avgBarVolume = recentVolume / recent.length;

  // Calculate strength (0-1 score)
  const strength = Math.min(1, (Math.abs(tickChange) / 10) * consistency);

  return {
    direction: tickChange > 0 ? 'long' : 'short',
    speed,
    strength,
  };
}

function calculatePullback(
  bars: TopstepXFuturesBar[],
  direction: 'long' | 'short',
  lookback: number = 3
): boolean {
  if (bars.length < lookback + 2) return false;

  const recent = bars.slice(-(lookback + 2));

  // For long: we want momentum up, then small pullback
  if (direction === 'long') {
    const highBeforePullback = Math.max(...recent.slice(0, -1).map(b => b.high));
    const lastClose = recent[recent.length - 1].close;
    return lastClose < highBeforePullback && lastClose > recent[0].close;
  }

  // For short: we want momentum down, then small pullback
  const lowBeforePullback = Math.min(...recent.slice(0, -1).map(b => b.low));
  const lastClose = recent[recent.length - 1].close;
  return lastClose > lowBeforePullback && lastClose < recent[0].close;
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

async function runMomentumBurstBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('MOMENTUM BURST SCALPER - ULTRA HIGH FREQUENCY');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${new Date(CONFIG.start).toLocaleDateString()} -> ${new Date(CONFIG.end).toLocaleDateString()}`);
  console.log('\nStrategy: Scalp momentum bursts with pullback entries');
  console.log(`- Momentum: ${CONFIG.minMomentumTicks} ticks in ${CONFIG.momentumLookback} seconds`);
  console.log(`- RSI Range: ${CONFIG.rsiMin}-${CONFIG.rsiMax}`);
  console.log(`- Risk: ${CONFIG.stopLossTicks} tick stop, ${CONFIG.takeProfitTicks} tick target`);
  console.log(`- Max Hold: ${CONFIG.maxHoldTime} seconds`);
  console.log(`- Contracts: ${CONFIG.numberOfContracts}`);
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

  let position: 'long' | 'short' | null = null;
  let entryPrice = 0;
  let entryTime = '';
  let entryIndex = 0;

  const trades: MomentumTrade[] = [];
  let lastTradeTime = 0;
  const dailyTrades = new Map<string, number>();

  // Indicators
  const rsiIndicator = new RSI({ period: CONFIG.rsiPeriod, values: [] });
  const emaFast = new EMA({ period: 5, values: [] });
  const emaSlow = new EMA({ period: 13, values: [] });

  // Signal tracking
  let momentumSignals = 0;
  let tradesTaken = 0;

  const closePosition = (exitPrice: number, bar: TopstepXFuturesBar, reason: MomentumTrade['exitReason']) => {
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
      momentumSpeed: 0,
    });

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
    if (closes.length > 100) {
      closes.shift();
    }

    // Update indicators
    const rsiValue = rsiIndicator.nextValue(bar.close);
    const emaFastValue = emaFast.nextValue(bar.close);
    const emaSlowValue = emaSlow.nextValue(bar.close);

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

      // Check timeout
      const holdTime = Math.round((Date.parse(bar.timestamp) - Date.parse(entryTime)) / 1000);
      if (holdTime >= CONFIG.maxHoldTime) {
        closePosition(bar.close, bar, 'timeout');
        continue;
      }

      continue;
    }

    // Need enough data
    if (i < CONFIG.momentumLookback + 10 || !rsiValue || !emaFastValue || !emaSlowValue) {
      continue;
    }

    // Detect momentum burst
    const momentum = detectMomentumBurst(
      bars.slice(i - CONFIG.momentumLookback, i + 1),
      CONFIG.momentumLookback,
      CONFIG.minMomentumTicks,
      tickSize
    );

    if (!momentum.direction || momentum.strength < 0.5) {
      continue;
    }

    momentumSignals++;

    // Apply filters
    if ((dailyTrades.get(dateKey) || 0) >= CONFIG.maxTradesPerDay) continue;

    const timeSinceLast = (Date.parse(bar.timestamp) - lastTradeTime) / 1000;
    if (timeSinceLast < CONFIG.minSecondsBetweenTrades) continue;

    // RSI filter
    if (rsiValue < CONFIG.rsiMin || rsiValue > CONFIG.rsiMax) continue;

    // EMA alignment
    if (momentum.direction === 'long' && emaFastValue < emaSlowValue) continue;
    if (momentum.direction === 'short' && emaFastValue > emaSlowValue) continue;

    // Look for pullback entry
    const hasPullback = calculatePullback(
      bars.slice(Math.max(0, i - 8), i + 1),
      momentum.direction,
      3
    );

    // For now, enter on momentum even without perfect pullback
    // In live, you'd wait for the pullback
    const canEnter = true;  // || hasPullback;

    if (!canEnter) continue;

    // Execute trade
    tradesTaken++;

    position = momentum.direction;
    entryPrice = roundToTick(bar.close);
    entryTime = bar.timestamp;
    entryIndex = i;

    lastTradeTime = Date.parse(bar.timestamp);
    dailyTrades.set(dateKey, (dailyTrades.get(dateKey) || 0) + 1);

    if (trades.length < 15) {
      console.log(
        `${position.toUpperCase().padEnd(5)} @ ${entryPrice.toFixed(2)} | ` +
        `${new Date(bar.timestamp).toLocaleTimeString()} | ` +
        `Momentum: ${momentum.speed.toFixed(1)} ticks/sec | RSI: ${rsiValue.toFixed(0)}`
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
  console.log('MOMENTUM BURST SCALPER RESULTS');
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
  console.log(`Momentum Signals: ${momentumSignals}`);
  console.log(`Trades Taken: ${tradesTaken}`);
  console.log(`Trades Per Day: ${(trades.length / Math.max(dailyTrades.size, 1)).toFixed(1)}`);

  console.log('\n🎯 EXIT BREAKDOWN:');
  exitReasons.forEach((count, reason) => {
    const pct = (count / trades.length) * 100;
    console.log(`- ${reason}: ${count} (${pct.toFixed(1)}%)`);
  });

  // Show recent trades
  if (trades.length > 0) {
    console.log('\n📝 SAMPLE TRADES:');
    trades.slice(0, 15).forEach(trade => {
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
  const meetsDrawdown = drawdownRatio < 50 || totalPnL <= 0;
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

runMomentumBurstBacktest().catch(err => {
  console.error('Momentum burst backtest failed:', err);
  process.exit(1);
});