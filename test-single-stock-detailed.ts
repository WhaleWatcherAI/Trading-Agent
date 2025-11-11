import { SMA, EMA, ATR, BollingerBands } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';

interface TradeDetail {
  tradeNumber: number;
  direction: 'CALL' | 'PUT';
  entry: {
    time: string;
    stockPrice: number;
    shares: number;
    totalCost: number;
    sma: number;
    squeezeOn: boolean;
  };
  exit: {
    time: string;
    stockPrice: number;
    totalValue: number;
    reason: string;
  };
  stockMove: number; // exit - entry stock price
  stockMovePercent: number;
  optionPnL: number; // Net P&L after commission
  optionReturnPercent: number;
  holdMinutes: number;
}

interface BacktestStats {
  symbol: string;
  totalTrades: number;
  winners: number;
  losers: number;
  winRate: number;
  totalPnL: number;
  avgPnL: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  avgHoldMinutes: number;
  profitFactor: number; // Total wins / Total losses
}

interface MinuteBar {
  time: number;
  close: number;
}

const SMA_PERIOD = 9;
const BOLLINGER_PERIOD = 20;
const BOLLINGER_STD_DEV = 2;
const KELTNER_PERIOD = 20;
const KELTNER_MULTIPLIER = 1.5;
const STOP_LOSS_PERCENT = 1; // 1% adverse move on the underlying
const COMMISSION_PER_CONTRACT = 0;
const SLIPPAGE_PERCENT = 0.03;
const TIMEFRAME = '1min';
const LOOKBACK_DAYS = 2;
const DAYS_TO_TEST = 5;
const TRADE_BUDGET = 200; // dollars deployed per trade
const SYMBOLS = ['TSLA', 'SPY', 'QQQ', 'NVDA', 'AMD'];
const LEVERAGE_MULTIPLIER = 100;

function toMillis(timestamp: string | number): number {
  if (typeof timestamp === 'number') {
    return timestamp > 1e12 ? timestamp : timestamp * 1000;
  }
  const numeric = Number(timestamp);
  return numeric > 1e12 ? numeric : numeric * 1000;
}

function formatTime(timestamp: string | number): string {
  const date = new Date(Number(timestamp) * 1000);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
    timeZone: 'America/New_York',
  });
}

function makeSeriesAccessor<T>(series: T[], offset: number) {
  return (index: number): T | null => {
    const relative = index - offset;
    if (relative < 0 || relative >= series.length) return null;
    return series[relative];
  };
}

function calculateIndicators(history: { high: number; low: number; close: number }[]) {
  const closes = history.map(bar => bar.close);
  const highs = history.map(bar => bar.high);
  const lows = history.map(bar => bar.low);

  const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });
  const smaAccessor = makeSeriesAccessor(smaValues, closes.length - smaValues.length);

  const bbValues = BollingerBands.calculate({
    period: BOLLINGER_PERIOD,
    stdDev: BOLLINGER_STD_DEV,
    values: closes,
  });
  const bbAccessor = makeSeriesAccessor(bbValues, closes.length - bbValues.length);

  const emaValues = EMA.calculate({ values: closes, period: KELTNER_PERIOD });
  const emaAccessor = makeSeriesAccessor(emaValues, closes.length - emaValues.length);

  const atrValues = ATR.calculate({
    period: KELTNER_PERIOD,
    high: highs,
    low: lows,
    close: closes,
  });
  const atrAccessor = makeSeriesAccessor(atrValues, closes.length - atrValues.length);

  // Returns: true = squeeze ON, false = squeeze OFF, null = not enough data
  const squeezeAccessor = (index: number): boolean | null => {
    const bb = bbAccessor(index);
    const ema = emaAccessor(index);
    const atr = atrAccessor(index);
    if (!bb || ema == null || atr == null) return null;
    const range = atr * KELTNER_MULTIPLIER;
    const kcUpper = ema + range;
    const kcLower = ema - range;
    return bb.upper <= kcUpper && bb.lower >= kcLower;
  };

  // Returns: 'bullish' | 'bearish' | null when squeeze is OFF
  // Determines direction by checking if price is in upper or lower half of bands
  const squeezeDirectionAccessor = (index: number): 'bullish' | 'bearish' | null => {
    const squeezeOn = squeezeAccessor(index);
    if (squeezeOn !== false) return null; // Only when squeeze is OFF

    const bb = bbAccessor(index);
    const price = closes[index];
    if (!bb || price == null) return null;

    const bbMid = (bb.upper + bb.lower) / 2;
    return price >= bbMid ? 'bullish' : 'bearish';
  };

  return {
    getSma: smaAccessor,
    getSqueeze: squeezeAccessor,
    getSqueezeDirection: squeezeDirectionAccessor,
  };
}

function getRecentTradingDays(count: number, referenceDate?: Date): string[] {
  const dates: string[] = [];
  const cursor = referenceDate ? new Date(referenceDate) : new Date();
  cursor.setHours(0, 0, 0, 0);
  cursor.setDate(cursor.getDate() - 1);

  while (dates.length < count) {
    const day = cursor.getDay();
    if (day !== 0 && day !== 6) {
      dates.push(cursor.toISOString().slice(0, 10));
    }
    cursor.setDate(cursor.getDate() - 1);
  }

  return dates;
}

async function backtestSingleStock(symbol: string, testDate: string): Promise<{ trades: TradeDetail[]; stats: BacktestStats }> {
  console.log('═══════════════════════════════════════════════════');
  console.log(`  DETAILED BACKTEST: ${symbol}`);
  console.log('═══════════════════════════════════════════════════');
  console.log(`Strategy: SMA(${SMA_PERIOD}) price crossover w/ TTM Squeeze filter`);
  console.log(`Timeframe: ${TIMEFRAME}`);
  console.log(`Date: ${testDate}`);
  console.log(`Stop Loss: ${STOP_LOSS_PERCENT}% adverse move on price`);
  console.log(`Trade Budget: $${TRADE_BUDGET.toFixed(2)} per position`);
  console.log(`Commission: $${COMMISSION_PER_CONTRACT * 2} round-trip`);
  console.log('═══════════════════════════════════════════════════\n');

  const history = await getHistoricalData(symbol, TIMEFRAME, LOOKBACK_DAYS, testDate);
  const minuteHistory = TIMEFRAME === '1min'
    ? history
    : await getHistoricalData(symbol, '1min', 1, testDate);

  console.log(`Loaded ${history.length} bars of ${TIMEFRAME} data`);
  console.log(`Loaded ${minuteHistory.length} bars of 1min data\n`);

  const closes = history.map(bar => bar.close);
  const { getSma, getSqueeze, getSqueezeDirection } = calculateIndicators(history);

  const minuteSeries: MinuteBar[] = minuteHistory
    .map(bar => ({
      time: toMillis(bar.date),
      close: bar.close,
    }))
    .sort((a, b) => a.time - b.time);

interface ActiveTrade {
  tradeNumber: number;
  direction: 'CALL' | 'PUT';
  entryIndex: number;
  entryTime: string;
  entryStockPrice: number;
  shares: number;
  capitalUsed: number;
  notionalExposure: number;
  entrySma: number;
  squeezeOnEntry: boolean;
  entryTimeMs: number;
  minuteIdx: number;
}

  const trades: TradeDetail[] = [];
  let currentTrade: ActiveTrade | null = null;
  let tradeNumber = 0;

  const checkIntrabarStop = (
    trade: ActiveTrade,
    currentBarTimeMs: number
  ): MinuteBar | null => {
    if (minuteSeries.length === 0) return null;
    let idx = trade.minuteIdx;
    while (idx < minuteSeries.length && minuteSeries[idx].time <= currentBarTimeMs) {
      const minuteBar = minuteSeries[idx];
      const priceMovePercent = ((minuteBar.close - trade.entryStockPrice) / trade.entryStockPrice) * 100;
      if (trade.direction === 'CALL') {
        if (priceMovePercent <= -STOP_LOSS_PERCENT) {
          trade.minuteIdx = idx + 1;
          return minuteBar;
        }
      } else if (priceMovePercent >= STOP_LOSS_PERCENT) {
        trade.minuteIdx = idx + 1;
        return minuteBar;
      }
      idx++;
    }
    trade.minuteIdx = idx;
    return null;
  };

  const startIndex = Math.max(1, SMA_PERIOD);

  for (let i = startIndex; i < history.length; i++) {
    const currentBar = history[i];
    const previousBar = history[i - 1];
    if (!previousBar) continue;

    const currentPrice = currentBar.close;
    const previousPrice = previousBar.close;
    const currentBarTimeMs = toMillis(currentBar.date);

    const currentSma = getSma(i);
    const previousSma = getSma(i - 1);
    const squeezeStatus = getSqueeze(i);
    const squeezeDirection = getSqueezeDirection(i);

    const hasSma = currentSma != null && previousSma != null;
    const bullishCross = hasSma ? (previousPrice <= previousSma! && currentPrice > currentSma!) : false;
    const bearishCross = hasSma ? (previousPrice >= previousSma! && currentPrice < currentSma!) : false;

    if (currentTrade) {
      const stopMinute = checkIntrabarStop(currentTrade, currentBarTimeMs);
      const exitDueToStop = stopMinute != null;
      const priceMovePercent = ((currentPrice - currentTrade.entryStockPrice) / currentTrade.entryStockPrice) * 100;
      const stopHit = currentTrade.direction === 'CALL'
        ? priceMovePercent <= -STOP_LOSS_PERCENT
        : priceMovePercent >= STOP_LOSS_PERCENT;
      const oppositeCross = hasSma
        ? (currentTrade.direction === 'CALL' ? bearishCross : bullishCross)
        : false;
      const endOfDay = i === history.length - 1;

      let exitReason: string | null = null;
      let exitStockPrice = currentPrice;
      let exitTimeMs = currentBarTimeMs;

      if (exitDueToStop) {
        exitReason = `${STOP_LOSS_PERCENT}% Stop`;
        exitStockPrice = stopMinute!.close;
        exitTimeMs = stopMinute!.time;
      } else if (oppositeCross) {
        exitReason = 'Opposite SMA Cross';
        if (currentSma != null) {
          exitStockPrice = currentSma;
        }
      } else if (stopHit) {
        exitReason = `${STOP_LOSS_PERCENT}% Stop`;
      } else if (endOfDay) {
        exitReason = 'End of Day';
      }

      if (exitReason) {
        const shares = currentTrade.shares;
        const positionMultiplier = currentTrade.direction === 'CALL' ? 1 : -1;
        const stockMove = exitStockPrice - currentTrade.entryStockPrice;
        const stockMovePercent = (stockMove / currentTrade.entryStockPrice) * 100;
        const holdMinutes = Math.max(1, Math.round((exitTimeMs - currentTrade.entryTimeMs) / 60000));
        const netPnl = stockMove * shares * positionMultiplier;

        const trade: TradeDetail = {
          tradeNumber: currentTrade.tradeNumber,
          direction: currentTrade.direction,
          entry: {
            time: formatTime(currentTrade.entryTime),
            stockPrice: currentTrade.entryStockPrice,
            shares,
            totalCost: currentTrade.capitalUsed,
            sma: currentTrade.entrySma,
            squeezeOn: currentTrade.squeezeOnEntry,
          },
          exit: {
            time: formatTime(Math.floor(exitTimeMs / 1000)),
            stockPrice: exitStockPrice,
            totalValue: exitStockPrice * shares,
            reason: exitReason,
          },
          stockMove,
          stockMovePercent: stockMovePercent * positionMultiplier,
          optionPnL: netPnl,
          optionReturnPercent: currentTrade.capitalUsed > 0 ? (netPnl / currentTrade.capitalUsed) * 100 : 0,
          holdMinutes,
        };

        trades.push(trade);
        currentTrade = null;
        continue;
      }
    }

    if (!currentTrade && (bullishCross || bearishCross)) {
      if (!hasSma) continue;

      // Only trade when squeeze is OFF and in the direction of expansion
      if (squeezeStatus !== false) continue; // Squeeze must be OFF
      if (!squeezeDirection) continue; // Must have direction

      // Only trade crossovers that match squeeze direction
      if (bullishCross && squeezeDirection !== 'bullish') continue;
      if (bearishCross && squeezeDirection !== 'bearish') continue;

      const direction: 'CALL' | 'PUT' = bullishCross ? 'CALL' : 'PUT';
      const underlyingEntryPrice = currentSma ?? currentPrice;
      const capitalUsed = TRADE_BUDGET;
      const exposureDollars = capitalUsed * LEVERAGE_MULTIPLIER;
      const sharesRaw = exposureDollars / underlyingEntryPrice;
      if (!isFinite(sharesRaw) || sharesRaw <= 0) continue;
      const shares = Math.max(1, Math.floor(sharesRaw));
      const notionalExposure = shares * underlyingEntryPrice;
      const entryTimeMs = currentBarTimeMs;
      const initialMinuteIdx = minuteSeries.findIndex(bar => bar.time > entryTimeMs);

      tradeNumber++;
      currentTrade = {
        tradeNumber,
        direction,
        entryIndex: i,
        entryTime: currentBar.date,
        entryStockPrice: underlyingEntryPrice,
        shares,
        capitalUsed,
        notionalExposure,
        entrySma: currentSma!,
        squeezeOnEntry: squeezeStatus === true, // Will be false since we only trade when squeeze is OFF
        entryTimeMs,
        minuteIdx: initialMinuteIdx >= 0 ? initialMinuteIdx : minuteSeries.length,
      };
    }
  }

  const winners = trades.filter(t => t.optionPnL > 0);
  const losers = trades.filter(t => t.optionPnL <= 0);
  const totalPnL = trades.reduce((sum, t) => sum + t.optionPnL, 0);
  const avgPnL = trades.length > 0 ? totalPnL / trades.length : 0;
  const totalWins = winners.reduce((sum, t) => sum + t.optionPnL, 0);
  const totalLosses = Math.abs(losers.reduce((sum, t) => sum + t.optionPnL, 0));
  const avgWin = winners.length > 0 ? totalWins / winners.length : 0;
  const avgLoss = losers.length > 0 ? losers.reduce((sum, t) => sum + t.optionPnL, 0) / losers.length : 0;
  const largestWin = winners.length > 0 ? Math.max(...winners.map(t => t.optionPnL)) : 0;
  const largestLoss = losers.length > 0 ? Math.min(...losers.map(t => t.optionPnL)) : 0;
  const avgHoldMinutes = trades.length > 0 ? trades.reduce((sum, t) => sum + t.holdMinutes, 0) / trades.length : 0;
  const profitFactor = totalLosses > 0 ? totalWins / totalLosses : 0;

  const stats: BacktestStats = {
    symbol,
    totalTrades: trades.length,
    winners: winners.length,
    losers: losers.length,
    winRate: trades.length > 0 ? (winners.length / trades.length) * 100 : 0,
    totalPnL,
    avgPnL,
    avgWin,
    avgLoss,
    largestWin,
    largestLoss,
    avgHoldMinutes,
    profitFactor,
  };

  return { trades, stats };
}

function aggregateStats(symbol: string, statsList: BacktestStats[]): BacktestStats {
  const totals = statsList.reduce(
    (acc, stats) => {
      acc.totalTrades += stats.totalTrades;
      acc.winners += stats.winners;
      acc.losers += stats.losers;
      acc.totalPnL += stats.totalPnL;
      acc.holdMinutes += stats.avgHoldMinutes * stats.totalTrades;
      acc.totalWins += stats.avgWin * stats.winners;
      acc.totalLosses += Math.abs(stats.avgLoss * stats.losers);
      acc.largestWin = Math.max(acc.largestWin, stats.largestWin);
      acc.largestLoss = Math.min(acc.largestLoss, stats.largestLoss);
      return acc;
    },
    {
      totalTrades: 0,
      winners: 0,
      losers: 0,
      totalPnL: 0,
      holdMinutes: 0,
      totalWins: 0,
      totalLosses: 0,
      largestWin: -Infinity,
      largestLoss: Infinity,
    }
  );

  const avgPnL = totals.totalTrades > 0 ? totals.totalPnL / totals.totalTrades : 0;
  const avgHoldMinutes = totals.totalTrades > 0 ? totals.holdMinutes / totals.totalTrades : 0;
  const avgWin = totals.winners > 0 ? totals.totalWins / totals.winners : 0;
  const avgLoss = totals.losers > 0 ? -(totals.totalLosses / totals.losers) : 0;
  const profitFactor = totals.totalLosses > 0 ? totals.totalWins / totals.totalLosses : 0;

  return {
    symbol,
    totalTrades: totals.totalTrades,
    winners: totals.winners,
    losers: totals.losers,
    winRate: totals.totalTrades > 0 ? (totals.winners / totals.totalTrades) * 100 : 0,
    totalPnL: totals.totalPnL,
    avgPnL,
    avgWin,
    avgLoss,
    largestWin: totals.largestWin === -Infinity ? 0 : totals.largestWin,
    largestLoss: totals.largestLoss === Infinity ? 0 : totals.largestLoss,
    avgHoldMinutes,
    profitFactor,
  };
}

async function run() {
  const dates = getRecentTradingDays(DAYS_TO_TEST);
  const overallDailyStats: BacktestStats[] = [];

  for (const symbol of SYMBOLS) {
    const statsPerDate: BacktestStats[] = [];
    console.log('\n===================================================');
    console.log(`Symbol: ${symbol}`);
    console.log('===================================================');

    for (const date of dates) {
      const { stats } = await backtestSingleStock(symbol, date);
      statsPerDate.push(stats);
      overallDailyStats.push(stats);
    }

    const summary = aggregateStats(symbol, statsPerDate);
    console.log(
      `Trades: ${summary.totalTrades} | Win Rate: ${summary.winRate.toFixed(1)}% | ` +
      `P&L: $${summary.totalPnL.toFixed(2)} | Profit Factor: ${summary.profitFactor.toFixed(2)}`
    );
  }

  const overallSummary = aggregateStats('ALL', overallDailyStats);
  console.log('\n═══════════════════════════════════════════════════');
  console.log('  OVERALL SUMMARY');
  console.log('═══════════════════════════════════════════════════');
  console.log(`Symbols: ${SYMBOLS.join(', ')}`);
  console.log(`Dates:   ${dates.join(', ')}`);
  console.log(`Total Trades: ${overallSummary.totalTrades}`);
  console.log(`Winners: ${overallSummary.winners} | Losers: ${overallSummary.losers}`);
  console.log(`Win Rate: ${overallSummary.winRate.toFixed(1)}%`);
  console.log(`Total P&L: $${overallSummary.totalPnL.toFixed(2)}`);
  console.log(`Average P&L: $${overallSummary.avgPnL.toFixed(2)}`);
  console.log(`Average Win: $${overallSummary.avgWin.toFixed(2)}`);
  console.log(`Average Loss: $${overallSummary.avgLoss.toFixed(2)}`);
  console.log(`Largest Win: $${overallSummary.largestWin.toFixed(2)}`);
  console.log(`Largest Loss: $${overallSummary.largestLoss.toFixed(2)}`);
  console.log(`Average Hold Time: ${overallSummary.avgHoldMinutes.toFixed(1)} minutes`);
  console.log(`Profit Factor: ${overallSummary.profitFactor.toFixed(2)}`);
  console.log('═══════════════════════════════════════════════════\n');
}

run()
  .then(() => process.exit(0))
  .catch(error => {
    console.error('Backtest failed:', error);
    process.exit(1);
  });
