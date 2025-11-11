import { RSI, ATR, MACD, BollingerBands, Stochastic } from 'technicalindicators';
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

  // SMA Settings
  fastSMA: number;
  slowSMA: number;
  confirmationSMA: number;

  // Risk Management (TIGHT)
  stopLossTicks: number;
  takeProfitTicks: number;
  breakEvenTicks: number;
  trailingStartTicks: number;
  trailingStepTicks: number;

  // Position Sizing
  baseContracts: number;
  maxContracts: number;

  // Signal Quality
  minSignalStrength: number;
  rsiPeriod: number;
  rsiNeutralZone: number[];

  // Trade Management
  maxDailyLoss: number;
  maxDailyTrades: number;
  minSecondsBetweenTrades: number;
  maxConsecutiveLosses: number;

  // Volatility Management
  atrPeriod: number;
  maxAtrMultiplier: number;
  minAtrMultiplier: number;

  commissionPerSide: number;
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
  exitReason: string;
  signalStrength: number;
  contracts: number;
  maxDrawdownDuringTrade: number;
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_DAYS = 5;

// Only trade during most liquid times
const PRIME_HOURS = [
  { start: 8 * 60 + 30, end: 10 * 60 },      // 8:30-10:00 CT (Opening)
  { start: 14 * 60 + 30, end: 15 * 60 },     // 2:30-3:00 CT (Pre-close)
];

const DEFAULT_SYMBOL = process.env.TOPSTEPX_SYMBOL || 'ESZ5';
const DEFAULT_CONTRACT_ID = process.env.TOPSTEPX_CONTRACT_ID;

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_SYMBOL,
  contractId: DEFAULT_CONTRACT_ID,
  start: process.env.TOPSTEPX_START || new Date(Date.now() - DEFAULT_DAYS * DAY_MS).toISOString(),
  end: process.env.TOPSTEPX_END || new Date().toISOString(),

  // Longer SMAs for stability
  fastSMA: 600,        // 10 minutes
  slowSMA: 1800,       // 30 minutes
  confirmationSMA: 300, // 5 minutes for momentum

  // TIGHT risk management
  stopLossTicks: 4,        // 1 point only
  takeProfitTicks: 12,     // 3 points (3:1 RR)
  breakEvenTicks: 4,       // Move stop to breakeven at 1 point profit
  trailingStartTicks: 8,   // Start trailing at 2 points
  trailingStepTicks: 2,    // Trail by 0.5 points

  // Conservative position sizing
  baseContracts: 1,
  maxContracts: 2,

  // High quality signals only
  minSignalStrength: 0.7,  // 70% minimum strength
  rsiPeriod: 50,
  rsiNeutralZone: [40, 60], // Avoid extremes

  // Strict trade management
  maxDailyLoss: 500,        // Stop trading after $500 loss
  maxDailyTrades: 3,        // Maximum 3 trades per day
  minSecondsBetweenTrades: 1800, // 30 minutes between trades
  maxConsecutiveLosses: 2,  // Stop after 2 consecutive losses

  // Volatility filters
  atrPeriod: 20,
  maxAtrMultiplier: 2.0,    // Skip if volatility too high
  minAtrMultiplier: 0.5,    // Skip if no volatility

  commissionPerSide: inferFuturesCommissionPerSide([DEFAULT_CONTRACT_ID, DEFAULT_SYMBOL]),
};

function toCentralTime(date: Date) {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isPrimeHours(timestamp: string | Date): boolean {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const ctDate = toCentralTime(date);
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  return PRIME_HOURS.some(window =>
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

function calculateSignalStrength(
  smaSpread: number,
  momentum: number,
  rsi: number,
  volume: number,
  atr: number,
  isPrime: boolean,
  direction: 'long' | 'short'
): number {
  let strength = 0;

  // 1. SMA spread strength (0-30 points)
  const spreadStrength = Math.min(Math.abs(smaSpread) * 1000, 30);
  strength += spreadStrength;

  // 2. Momentum alignment (0-20 points)
  const momentumStrength = direction === 'long'
    ? Math.max(0, Math.min(momentum * 100, 20))
    : Math.max(0, Math.min(-momentum * 100, 20));
  strength += momentumStrength;

  // 3. RSI position (0-20 points)
  if (direction === 'long') {
    if (rsi > 50 && rsi < 70) strength += 20;
    else if (rsi > 45 && rsi <= 50) strength += 10;
  } else {
    if (rsi < 50 && rsi > 30) strength += 20;
    else if (rsi >= 50 && rsi < 55) strength += 10;
  }

  // 4. Volume confirmation (0-15 points)
  const volumeStrength = Math.min(volume / 100, 15);
  strength += volumeStrength;

  // 5. Prime hours bonus (0-15 points)
  if (isPrime) strength += 15;

  return strength / 100; // Normalize to 0-1
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
  console.log('LOW DRAWDOWN SMA STRATEGY - ULTRA TIGHT RISK MANAGEMENT');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${new Date(CONFIG.start).toLocaleDateString()} -> ${new Date(CONFIG.end).toLocaleDateString()}`);
  console.log(`\nRISK PARAMETERS:`);
  console.log(`- Stop: ${CONFIG.stopLossTicks} ticks (1 point)`);
  console.log(`- Target: ${CONFIG.takeProfitTicks} ticks (3 points) = 3:1 RR`);
  console.log(`- Breakeven at: ${CONFIG.breakEvenTicks} ticks`);
  console.log(`- Trailing starts: ${CONFIG.trailingStartTicks} ticks`);
  console.log(`- Max Daily Loss: $${CONFIG.maxDailyLoss}`);
  console.log(`- Max Daily Trades: ${CONFIG.maxDailyTrades}`);
  console.log(`- Min Signal Strength: ${(CONFIG.minSignalStrength * 100).toFixed(0)}%`);
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

  console.log(`\nContract: ${metadata.name}`);
  console.log(`Tick: ${tickSize} | Value/Tick: $${multiplier}`);

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
  let stopPrice = 0;
  let targetPrice = 0;
  let positionContracts = 0;
  let positionHighWaterMark = 0;
  let entryBar = 0;
  let trailingStop = 0;
  let breakEvenSet = false;

  const trades: TradeRecord[] = [];
  let consecutiveLosses = 0;
  let lastTradeTime = 0;
  let dailyPnL = new Map<string, number>();
  let dailyTrades = new Map<string, number>();

  // Indicators
  const rsiIndicator = new RSI({ period: CONFIG.rsiPeriod, values: [] });
  const atrIndicator = new ATR({ period: CONFIG.atrPeriod, high: [], low: [], close: [] });
  const macdIndicator = new MACD({
    fastPeriod: 12,
    slowPeriod: 26,
    signalPeriod: 9,
    SimpleMAOscillator: false,
    SimpleMASignal: false,
    values: [],
  });
  const bbIndicator = new BollingerBands({
    period: 20,
    stdDev: 2,
    values: [],
  });

  // Signal tracking
  let signalsAnalyzed = 0;
  let signalsRejected = new Map<string, number>();

  const closePosition = (
    exitPrice: number,
    bar: TopstepXFuturesBar,
    reason: string
  ) => {
    if (!position) return;

    const direction = position === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - entryPrice) * direction * multiplier * positionContracts;
    const fees = CONFIG.commissionPerSide * 2 * positionContracts;
    const netPnL = rawPnL - fees;

    // Track max drawdown during trade
    const worstPrice = position === 'long'
      ? Math.min(...bars.slice(entryBar, entryBar + 100).map(b => b.low))
      : Math.max(...bars.slice(entryBar, entryBar + 100).map(b => b.high));
    const maxDrawdown = Math.abs((worstPrice - entryPrice) * direction * multiplier * positionContracts);

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
      signalStrength: 0,
      contracts: positionContracts,
      maxDrawdownDuringTrade: maxDrawdown,
    });

    // Update daily P&L
    const dateKey = new Date(bar.timestamp).toISOString().split('T')[0];
    dailyPnL.set(dateKey, (dailyPnL.get(dateKey) || 0) + netPnL);

    // Track consecutive losses
    if (netPnL < 0) {
      consecutiveLosses++;
    } else {
      consecutiveLosses = 0;
    }

    position = null;
    entryPrice = 0;
    entryTime = '';
    stopPrice = 0;
    targetPrice = 0;
    positionContracts = 0;
    positionHighWaterMark = 0;
    entryBar = 0;
    trailingStop = 0;
    breakEvenSet = false;
  };

  // Main loop
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const dateKey = new Date(bar.timestamp).toISOString().split('T')[0];

    // Initialize daily tracking
    if (!dailyTrades.has(dateKey)) {
      dailyTrades.set(dateKey, 0);
      dailyPnL.set(dateKey, 0);
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
    const macdValue = macdIndicator.nextValue(bar.close);
    const bbValue = bbIndicator.nextValue(bar.close);

    // Check if trading allowed
    if (!isTradingAllowed(bar.timestamp)) {
      if (position) {
        closePosition(bar.close, bar, 'session_close');
      }
      continue;
    }

    // Check daily loss limit
    if ((dailyPnL.get(dateKey) || 0) <= -CONFIG.maxDailyLoss) {
      if (position) {
        closePosition(bar.close, bar, 'daily_loss_limit');
      }
      continue;
    }

    // Check consecutive losses
    if (consecutiveLosses >= CONFIG.maxConsecutiveLosses) {
      if (position) {
        closePosition(bar.close, bar, 'consecutive_losses');
      }
      continue;
    }

    // Manage existing position
    if (position) {
      entryBar++;
      const direction = position === 'long' ? 1 : -1;

      // Update high water mark
      if (position === 'long') {
        positionHighWaterMark = Math.max(positionHighWaterMark, bar.high);
      } else {
        positionHighWaterMark = Math.min(positionHighWaterMark, bar.low);
      }

      // Check stop loss
      if ((direction === 1 && bar.low <= stopPrice) ||
          (direction === -1 && bar.high >= stopPrice)) {
        closePosition(stopPrice, bar, breakEvenSet ? 'breakeven' : 'stop_loss');
        continue;
      }

      // Check take profit
      if ((direction === 1 && bar.high >= targetPrice) ||
          (direction === -1 && bar.low <= targetPrice)) {
        closePosition(targetPrice, bar, 'take_profit');
        continue;
      }

      // Breakeven management
      if (!breakEvenSet) {
        const profitTicks = direction === 1
          ? (bar.close - entryPrice) / tickSize
          : (entryPrice - bar.close) / tickSize;

        if (profitTicks >= CONFIG.breakEvenTicks) {
          stopPrice = entryPrice;
          breakEvenSet = true;
        }
      }

      // Trailing stop management
      if (breakEvenSet) {
        const profitTicks = direction === 1
          ? (positionHighWaterMark - entryPrice) / tickSize
          : (entryPrice - positionHighWaterMark) / tickSize;

        if (profitTicks >= CONFIG.trailingStartTicks) {
          const newTrailing = direction === 1
            ? positionHighWaterMark - CONFIG.trailingStepTicks * tickSize
            : positionHighWaterMark + CONFIG.trailingStepTicks * tickSize;

          if (direction === 1 && newTrailing > stopPrice) {
            stopPrice = newTrailing;
          } else if (direction === -1 && newTrailing < stopPrice) {
            stopPrice = newTrailing;
          }
        }
      }

      continue; // Skip signal detection while in position
    }

    // Calculate SMAs
    const fastSMA = calculateSMA(closes, CONFIG.fastSMA);
    const slowSMA = calculateSMA(closes, CONFIG.slowSMA);
    const confirmSMA = calculateSMA(closes, CONFIG.confirmationSMA);

    if (!fastSMA || !slowSMA || !confirmSMA || !rsiValue || !atrValue) continue;

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

    signalsAnalyzed++;

    // SIGNAL QUALITY FILTERS
    let rejectReason = '';

    // 1. Check daily trade limit
    if ((dailyTrades.get(dateKey) || 0) >= CONFIG.maxDailyTrades) {
      rejectReason = 'daily_limit';
    }

    // 2. Check time since last trade
    if (!rejectReason) {
      const timeSinceLast = (Date.parse(bar.timestamp) - lastTradeTime) / 1000;
      if (timeSinceLast < CONFIG.minSecondsBetweenTrades) {
        rejectReason = 'time_gap';
      }
    }

    // 3. Check volatility
    if (!rejectReason) {
      const avgATR = atrValue;
      const normalATR = tickSize * 4; // Normal is 1 point
      if (avgATR > normalATR * CONFIG.maxAtrMultiplier) {
        rejectReason = 'high_volatility';
      } else if (avgATR < normalATR * CONFIG.minAtrMultiplier) {
        rejectReason = 'low_volatility';
      }
    }

    // 4. Check RSI neutral zone
    if (!rejectReason) {
      if (rsiValue < CONFIG.rsiNeutralZone[0] || rsiValue > CONFIG.rsiNeutralZone[1]) {
        rejectReason = 'rsi_extreme';
      }
    }

    // 5. Check confirmation SMA alignment
    if (!rejectReason) {
      if (crossUp && bar.close < confirmSMA) {
        rejectReason = 'confirmation_misaligned';
      } else if (crossDown && bar.close > confirmSMA) {
        rejectReason = 'confirmation_misaligned';
      }
    }

    // 6. Calculate signal strength
    const smaSpread = (fastSMA - slowSMA) / slowSMA;
    const momentum = (bar.close - closes[closes.length - 100]) / closes[closes.length - 100];
    const avgVolume = volumes.slice(-100).reduce((a, b) => a + b, 0) / 100;
    const isPrime = isPrimeHours(bar.timestamp);

    const signalStrength = calculateSignalStrength(
      smaSpread,
      momentum,
      rsiValue,
      bar.volume / avgVolume,
      atrValue,
      isPrime,
      crossUp ? 'long' : 'short'
    );

    if (!rejectReason && signalStrength < CONFIG.minSignalStrength) {
      rejectReason = 'weak_signal';
    }

    // 7. Check Bollinger Bands
    if (!rejectReason && bbValue) {
      if (crossUp && bar.close > bbValue.upper) {
        rejectReason = 'bb_overbought';
      } else if (crossDown && bar.close < bbValue.lower) {
        rejectReason = 'bb_oversold';
      }
    }

    // 8. Check MACD confirmation
    if (!rejectReason && macdValue) {
      if (crossUp && macdValue.MACD < macdValue.signal) {
        rejectReason = 'macd_divergence';
      } else if (crossDown && macdValue.MACD > macdValue.signal) {
        rejectReason = 'macd_divergence';
      }
    }

    // Track rejection reasons
    if (rejectReason) {
      signalsRejected.set(rejectReason, (signalsRejected.get(rejectReason) || 0) + 1);
      continue;
    }

    // EXECUTE TRADE
    const direction = crossUp ? 'long' : 'short';

    // Dynamic position sizing based on signal strength
    positionContracts = signalStrength >= 0.85 ? CONFIG.maxContracts : CONFIG.baseContracts;

    position = direction;
    entryPrice = roundToTick(bar.close);
    entryTime = bar.timestamp;
    entryBar = i;
    positionHighWaterMark = bar.close;

    // Set initial stop and target
    if (direction === 'long') {
      stopPrice = roundToTick(entryPrice - CONFIG.stopLossTicks * tickSize);
      targetPrice = roundToTick(entryPrice + CONFIG.takeProfitTicks * tickSize);
    } else {
      stopPrice = roundToTick(entryPrice + CONFIG.stopLossTicks * tickSize);
      targetPrice = roundToTick(entryPrice - CONFIG.takeProfitTicks * tickSize);
    }

    breakEvenSet = false;
    trailingStop = 0;

    lastTradeTime = Date.parse(bar.timestamp);
    dailyTrades.set(dateKey, (dailyTrades.get(dateKey) || 0) + 1);

    if (trades.length < 10) {
      const rr = CONFIG.takeProfitTicks / CONFIG.stopLossTicks;
      console.log(
        `${direction.toUpperCase().padEnd(5)} @ ${entryPrice.toFixed(2)} | ` +
        `${new Date(bar.timestamp).toLocaleTimeString()} | ` +
        `Strength: ${(signalStrength * 100).toFixed(0)}% | ` +
        `Contracts: ${positionContracts} | RR: 1:${rr}`
      );
    }
  }

  // Close any remaining position
  if (position) {
    closePosition(bars[bars.length - 1].close, bars[bars.length - 1], 'end_of_data');
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

  // Calculate max drawdown
  let runningPnL = 0;
  let peakPnL = 0;
  let maxDrawdown = 0;
  let currentDrawdown = 0;
  const drawdownPeriods: number[] = [];

  trades.forEach(trade => {
    runningPnL += trade.pnl;
    if (runningPnL > peakPnL) {
      if (currentDrawdown > 0) {
        drawdownPeriods.push(currentDrawdown);
      }
      peakPnL = runningPnL;
      currentDrawdown = 0;
    } else {
      currentDrawdown = peakPnL - runningPnL;
      if (currentDrawdown > maxDrawdown) {
        maxDrawdown = currentDrawdown;
      }
    }
  });

  // Daily analysis
  const winningDays = Array.from(dailyPnL.values()).filter(pnl => pnl > 0).length;
  const tradingDays = dailyPnL.size;

  // Print results
  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST RESULTS - LOW DRAWDOWN STRATEGY');
  console.log('='.repeat(80));

  console.log('\n📊 PERFORMANCE METRICS:');
  console.log(`- Total P&L: ${totalPnL >= 0 ? '+' : ''}$${totalPnL.toFixed(2)}`);
  console.log(`- Max Drawdown: $${maxDrawdown.toFixed(2)}`);
  console.log(`- Drawdown/Profit Ratio: ${totalPnL > 0 ? (maxDrawdown/totalPnL).toFixed(2) : 'N/A'}`);
  console.log(`- Win Rate: ${winRate.toFixed(1)}% (${wins.length}W / ${losses.length}L)`);
  console.log(`- Profit Factor: ${profitFactor.toFixed(2)}`);
  console.log(`- Avg Win: +$${avgWin.toFixed(2)}`);
  console.log(`- Avg Loss: $${avgLoss.toFixed(2)}`);

  const targetMet = maxDrawdown < totalPnL / 2;
  console.log(`\n🎯 TARGET: Drawdown < 50% of Profit: ${targetMet ? '✅ ACHIEVED' : '❌ FAILED'}`);
  if (totalPnL > 0) {
    console.log(`   Required: <$${(totalPnL/2).toFixed(2)} | Actual: $${maxDrawdown.toFixed(2)}`);
  }

  console.log('\n📈 SIGNAL ANALYSIS:');
  console.log(`- Signals Analyzed: ${signalsAnalyzed}`);
  console.log(`- Signals Taken: ${trades.length} (${((trades.length/signalsAnalyzed)*100).toFixed(1)}%)`);
  console.log(`- Signals Rejected: ${signalsAnalyzed - trades.length}`);

  if (signalsRejected.size > 0) {
    console.log('\n  Rejection Reasons:');
    Array.from(signalsRejected.entries())
      .sort((a, b) => b[1] - a[1])
      .forEach(([reason, count]) => {
        console.log(`  - ${reason}: ${count}`);
      });
  }

  console.log('\n📅 DAILY STATISTICS:');
  console.log(`- Trading Days: ${tradingDays}`);
  console.log(`- Winning Days: ${winningDays} (${((winningDays/tradingDays)*100).toFixed(1)}%)`);
  console.log(`- Avg Daily P&L: $${(totalPnL/tradingDays).toFixed(2)}`);
  console.log(`- Avg Trades/Day: ${(trades.length/tradingDays).toFixed(1)}`);

  // Exit analysis
  const exitReasons = new Map<string, number>();
  trades.forEach(t => {
    exitReasons.set(t.exitReason, (exitReasons.get(t.exitReason) || 0) + 1);
  });

  console.log('\n🚪 EXIT ANALYSIS:');
  exitReasons.forEach((count, reason) => {
    const pct = (count / trades.length) * 100;
    console.log(`- ${reason}: ${count} (${pct.toFixed(1)}%)`);
  });

  // Show sample trades
  if (trades.length > 0) {
    console.log('\n📝 RECENT TRADES:');
    trades.slice(-5).forEach(trade => {
      const time = new Date(trade.entryTime).toLocaleTimeString();
      console.log(
        `${trade.side.toUpperCase().padEnd(5)} ${time} | ` +
        `${trade.entryPrice.toFixed(2)} → ${trade.exitPrice.toFixed(2)} | ` +
        `P&L: ${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)} | ` +
        `Contracts: ${trade.contracts} | ${trade.exitReason}`
      );
    });
  }

  console.log('\n' + '='.repeat(80));
  console.log(`FINAL RESULT: ${targetMet ? '✅ SUCCESS' : '❌ NEEDS ADJUSTMENT'} - Drawdown Control ${targetMet ? 'Achieved' : 'Failed'}`);
  console.log('='.repeat(80));
}

runBacktest().catch(err => {
  console.error('Backtest failed:', err);
  process.exit(1);
});