import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
} from './lib/topstepx';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';
import { RSI } from 'technicalindicators';

interface SymbolConfig {
  symbol: string;
  contractId?: string;
}

interface TradeResult {
  symbol: string;
  totalTrades: number;
  winRate: number;
  totalPnL: number;
  avgWin: number;
  avgLoss: number;
  maxDrawdown: number;
  commission: number;
  multiplier: number;
  tickSize: number;
  avgDailyPnL: number;
  tradingDays: number;
}

const SYMBOLS: SymbolConfig[] = [
  { symbol: 'ESZ5' },   // ES
  { symbol: 'MESZ5' },  // MES
  { symbol: 'NQZ5' },   // NQ
  { symbol: 'MNQZ5' },  // MNQ
  { symbol: 'GCZ5' },   // Gold
  { symbol: '6EZ5' },   // Euro
];

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_DAYS = 5;

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

function calculateSMA(values: number[], period: number): number | null {
  if (values.length < period) return null;
  const sum = values.slice(-period).reduce((acc, val) => acc + val, 0);
  return sum / period;
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
    await new Promise(r => setTimeout(r, 800));
  }

  return bars.reverse();
}

async function backTestSymbol(symbolConfig: SymbolConfig): Promise<TradeResult | null> {
  try {
    const lookupKey = symbolConfig.contractId || symbolConfig.symbol;
    const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(() => null);

    if (!metadata) {
      console.log(`  ⚠️  ${symbolConfig.symbol}: Metadata fetch failed`);
      return null;
    }

    const contractId = metadata.id;
    const tickSize = metadata.tickSize;
    const multiplier = metadata.tickValue && metadata.tickSize
      ? metadata.tickValue / metadata.tickSize
      : metadata.multiplier || 50;
    const commission = inferFuturesCommissionPerSide([contractId, symbolConfig.symbol]);

    const roundToTick = (value: number) => Math.round(value / tickSize) * tickSize;

    console.log(`  📊 ${symbolConfig.symbol}: Fetching bars...`);

    const start = new Date(Date.now() - DEFAULT_DAYS * DAY_MS).toISOString();
    const end = new Date().toISOString();

    const bars = await fetchSecondBarsInChunks(contractId, start, end);

    if (bars.length === 0) {
      console.log(`  ⚠️  ${symbolConfig.symbol}: No bars returned`);
      return null;
    }

    bars.reverse();

    // Run backtest
    const closes: number[] = [];
    let position: 'long' | 'short' | null = null;
    let entryPrice = 0;
    let entryTime = '';
    let positionHighWaterMark = 0;
    let positionLowWaterMark = 0;
    const trades: Array<{ pnl: number; entryTime: string }> = [];
    let lastTradeTime = 0;
    const dailyTrades = new Map<string, number>();
    const dailyPnL = new Map<string, number>();

    const rsiIndicator = new RSI({ period: 50, values: [] });
    let totalSignals = 0;
    let signalsTaken = 0;

    const CONFIG = {
      fastSMA: 500,
      slowSMA: 1500,
      rsiOversold: 45,
      rsiOverbought: 55,
      stopLossTicks: 6,
      takeProfitTicks: 15,
      trailingStopTicks: 8,
      minSecondsBetweenTrades: 900,
      maxDailyTrades: 6,
      numberOfContracts: 1,
    };

    const closePosition = (exitPrice: number, bar: TopstepXFuturesBar) => {
      if (!position) return;
      const direction = position === 'long' ? 1 : -1;
      const rawPnL = (exitPrice - entryPrice) * direction * multiplier * CONFIG.numberOfContracts;
      const fees = commission * 2 * CONFIG.numberOfContracts;
      const netPnL = rawPnL - fees;

      trades.push({ pnl: netPnL, entryTime });

      const dateKey = new Date(bar.timestamp).toISOString().split('T')[0];
      dailyPnL.set(dateKey, (dailyPnL.get(dateKey) || 0) + netPnL);

      position = null;
    };

    for (let i = 0; i < bars.length; i++) {
      const bar = bars[i];
      const dateKey = new Date(bar.timestamp).toISOString().split('T')[0];

      if (!dailyTrades.has(dateKey)) {
        dailyTrades.set(dateKey, 0);
        dailyPnL.set(dateKey, 0);
      }

      closes.push(bar.close);
      if (closes.length > CONFIG.slowSMA + 100) {
        closes.shift();
      }

      const rsiValue = rsiIndicator.nextValue(bar.close);

      if (!isTradingAllowed(bar.timestamp)) {
        if (position) {
          closePosition(bar.close, bar);
        }
        continue;
      }

      if (position) {
        const direction = position === 'long' ? 1 : -1;
        const stopPrice = roundToTick(entryPrice - direction * CONFIG.stopLossTicks * tickSize);
        const targetPrice = roundToTick(entryPrice + direction * CONFIG.takeProfitTicks * tickSize);

        if ((direction === 1 && bar.low <= stopPrice) ||
            (direction === -1 && bar.high >= stopPrice)) {
          closePosition(stopPrice, bar);
          continue;
        }

        if ((direction === 1 && bar.high >= targetPrice) ||
            (direction === -1 && bar.low <= targetPrice)) {
          closePosition(targetPrice, bar);
          continue;
        }

        const profitTicks = direction === 1
          ? (positionHighWaterMark - entryPrice) / tickSize
          : (entryPrice - positionLowWaterMark) / tickSize;

        if (profitTicks >= CONFIG.trailingStopTicks) {
          const trailingStop = direction === 1
            ? positionHighWaterMark - CONFIG.trailingStopTicks * tickSize
            : positionLowWaterMark + CONFIG.trailingStopTicks * tickSize;

          if ((direction === 1 && bar.close <= trailingStop) ||
              (direction === -1 && bar.close >= trailingStop)) {
            closePosition(bar.close, bar);
            continue;
          }
        }
      }

      const fastSMA = calculateSMA(closes, CONFIG.fastSMA);
      const slowSMA = calculateSMA(closes, CONFIG.slowSMA);

      if (!fastSMA || !slowSMA || !rsiValue) continue;

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
      if (crossUp && rsiValue <= CONFIG.rsiOversold) signalValid = false;
      if (crossDown && rsiValue >= CONFIG.rsiOverbought) signalValid = false;

      const todayTrades = dailyTrades.get(dateKey) || 0;
      if (todayTrades >= CONFIG.maxDailyTrades) signalValid = false;

      const timeSinceLast = (Date.parse(bar.timestamp) - lastTradeTime) / 1000;
      if (timeSinceLast < CONFIG.minSecondsBetweenTrades) signalValid = false;

      if (!signalValid) continue;

      signalsTaken++;

      if ((crossUp && !position) || (crossDown && !position)) {
        position = crossUp ? 'long' : 'short';
        entryPrice = roundToTick(bar.close);
        entryTime = bar.timestamp;
        positionHighWaterMark = bar.high;
        positionLowWaterMark = bar.low;
        lastTradeTime = Date.parse(bar.timestamp);
        dailyTrades.set(dateKey, todayTrades + 1);
      } else if ((crossUp && position === 'short') || (crossDown && position === 'long')) {
        closePosition(bar.close, bar);
      }
    }

    if (position) {
      closePosition(bars[bars.length - 1].close, bars[bars.length - 1]);
    }

    // Calculate stats
    const wins = trades.filter(t => t.pnl > 0);
    const losses = trades.filter(t => t.pnl < 0);
    const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0);
    const avgWin = wins.length ? wins.reduce((sum, t) => sum + t.pnl, 0) / wins.length : 0;
    const avgLoss = losses.length ? losses.reduce((sum, t) => sum + t.pnl, 0) / losses.length : 0;
    const winRate = trades.length ? (wins.length / trades.length) * 100 : 0;

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

    return {
      symbol: symbolConfig.symbol,
      totalTrades: trades.length,
      winRate,
      totalPnL,
      avgWin,
      avgLoss,
      maxDrawdown,
      commission,
      multiplier,
      tickSize,
      avgDailyPnL: dailyPnL.size > 0 ? totalPnL / dailyPnL.size : 0,
      tradingDays: dailyPnL.size,
    };
  } catch (error) {
    console.log(`  ❌ ${symbolConfig.symbol}: ${(error as Error).message}`);
    return null;
  }
}

async function runMultiSymbolBacktest() {
  console.log('\n' + '='.repeat(100));
  console.log('MULTI-SYMBOL 1-SECOND SMA STRATEGY BACKTEST');
  console.log('='.repeat(100));
  console.log(`Period: ${new Date(Date.now() - DEFAULT_DAYS * DAY_MS).toLocaleDateString()} -> ${new Date().toLocaleDateString()}`);
  console.log(`\nTesting: ES, MES, NQ, MNQ, GC, 6E`);
  console.log('\nFetching data...\n');

  const results: TradeResult[] = [];

  for (const symbolConfig of SYMBOLS) {
    const result = await backTestSymbol(symbolConfig);
    if (result) {
      results.push(result);
    }
    // Delay between symbols to avoid rate limiting
    await new Promise(r => setTimeout(r, 2000));
  }

  // Display results
  console.log('\n' + '='.repeat(100));
  console.log('BACKTEST RESULTS - MULTI-SYMBOL COMPARISON');
  console.log('='.repeat(100));
  console.log(
    '\nSymbol | Trades | Win% | Total P&L | Avg Win | Avg Loss | Max DD | Comm/Side | Mult | Avg Daily PnL'
  );
  console.log(
    '-------|--------|------|-----------|---------|----------|--------|-----------|------|---------------'
  );

  let totalPnL = 0;
  let totalTrades = 0;
  let totalDD = 0;

  for (const result of results) {
    const pnlStr = result.totalPnL >= 0 ? `+$${result.totalPnL.toFixed(0)}` : `-$${Math.abs(result.totalPnL).toFixed(0)}`;
    const avgWinStr = `+$${result.avgWin.toFixed(0)}`;
    const avgLossStr = `-$${Math.abs(result.avgLoss).toFixed(0)}`;
    const ddStr = `-$${result.maxDrawdown.toFixed(0)}`;
    const dailyStr = result.avgDailyPnL >= 0 ? `+$${result.avgDailyPnL.toFixed(0)}` : `-$${Math.abs(result.avgDailyPnL).toFixed(0)}`;

    console.log(
      `${result.symbol.padEnd(6)} | ${String(result.totalTrades).padStart(6)} | ${result.winRate.toFixed(1).padStart(4)}% | ` +
      `${pnlStr.padStart(9)} | ${avgWinStr.padStart(7)} | ${avgLossStr.padStart(8)} | ${ddStr.padStart(6)} | ` +
      `$${result.commission.toFixed(2).padStart(8)} | ${String(result.multiplier).padStart(4)} | ${dailyStr.padStart(12)}`
    );

    totalPnL += result.totalPnL;
    totalTrades += result.totalTrades;
    totalDD += result.maxDrawdown;
  }

  console.log(
    '-------|--------|------|-----------|---------|----------|--------|-----------|------|---------------'
  );
  const totalPnLStr = totalPnL >= 0 ? `+$${totalPnL.toFixed(0)}` : `-$${Math.abs(totalPnL).toFixed(0)}`;
  const totalDDStr = `-$${totalDD.toFixed(0)}`;
  console.log(
    `TOTAL  | ${String(totalTrades).padStart(6)} |       | ${totalPnLStr.padStart(9)} |         |          | ${totalDDStr.padStart(6)} |           |      |`
  );

  console.log('\n' + '='.repeat(100));
  console.log('COMMISSION RATES (TopstepX):');
  console.log('- ES/NQ: $1.40/side');
  console.log('- MES/MNQ: $0.37/side');
  console.log('- GC/6E: $1.62/side');
  console.log('='.repeat(100) + '\n');
}

runMultiSymbolBacktest().catch(err => {
  console.error('Multi-symbol backtest failed:', err);
  process.exit(1);
});