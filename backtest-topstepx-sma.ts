import { RSI } from 'technicalindicators';
import {
  TopstepXFuturesBar,
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
  rsiPeriod: number;
  contractMultiplier: number;
  commissionPerSide: number;
  stopLossPercent: number;
  takeProfitPercent: number;
}

interface TradeRecord {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  exitReason: 'target' | 'stop' | 'signal' | 'session' | 'end_of_data';
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const FIVE_MINUTES_MS = 5 * 60 * 1000;
const SECOND_CHUNK_MS = 4 * 60 * 60 * 1000;

const DEFAULT_SMA_SYMBOL = process.env.TOPSTEPX_SMA_SYMBOL || 'ESZ5';
const DEFAULT_SMA_CONTRACT_ID = process.env.TOPSTEPX_CONTRACT_ID;

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_SMA_SYMBOL,
  contractId: DEFAULT_SMA_CONTRACT_ID,
  start: process.env.TOPSTEPX_SMA_START || '2025-10-30T00:00:00Z',
  end: process.env.TOPSTEPX_SMA_END || '2025-11-07T23:59:59Z',
  smaPeriod: Number(process.env.TOPSTEPX_SMA_PERIOD || '9'),
  rsiPeriod: Number(process.env.TOPSTEPX_SMA_RSI_PERIOD || '14'),
  contractMultiplier: Number(process.env.TOPSTEPX_CONTRACT_MULTIPLIER || '50'),
  commissionPerSide: process.env.TOPSTEPX_SMA_COMMISSION
    ? Number(process.env.TOPSTEPX_SMA_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_SMA_CONTRACT_ID, DEFAULT_SMA_SYMBOL], 0.35),
  stopLossPercent: Number(process.env.TOPSTEPX_STOP_LOSS_PERCENT || '0.001'),
  takeProfitPercent: Number(process.env.TOPSTEPX_TAKE_PROFIT_PERCENT || '0.005'),
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

function calculateSMA(values: number[], period: number): Array<number | null> {
  const result: Array<number | null> = new Array(values.length).fill(null);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
    if (i >= period) {
      sum -= values[i - period];
    }
    if (i >= period - 1) {
      result[i] = sum / period;
    }
  }
  return result;
}

function formatCurrency(value: number) {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

async function fetchSecondBars(
  contractId: string,
  start: string,
  end: string,
): Promise<TopstepXFuturesBar[]> {
  const startDate = new Date(start);
  const endDate = new Date(end);
  const bars: TopstepXFuturesBar[] = [];
  let cursor = startDate;

  while (cursor < endDate) {
    const chunkEnd = new Date(Math.min(cursor.getTime() + SECOND_CHUNK_MS, endDate.getTime()));
    console.log(`Fetching 1s bars from ${cursor.toISOString()} to ${chunkEnd.toISOString()}`);
    const chunkBars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: cursor.toISOString(),
      endTime: chunkEnd.toISOString(),
      unit: 1,
      unitNumber: 1,
    });
    bars.push(...chunkBars);
    cursor = new Date(chunkEnd.getTime() + 1_000);
    await new Promise(resolve => setTimeout(resolve, 1000)); // Add delay
  }

  return bars.reverse();
}

function aggregateSecondsToFiveMinutes(
  seconds: TopstepXFuturesBar[],
): { bars: TopstepXFuturesBar[]; segments: TopstepXFuturesBar[][] } {
  const bars: TopstepXFuturesBar[] = [];
  const segments: TopstepXFuturesBar[][] = [];
  let currentWindowStart: number | null = null;
  let currentBar: TopstepXFuturesBar | null = null;
  let currentSegment: TopstepXFuturesBar[] = [];

  for (const secondBar of seconds) {
    const time = new Date(secondBar.timestamp).getTime();
    if (!Number.isFinite(time)) continue;
    const windowStart = Math.floor(time / FIVE_MINUTES_MS) * FIVE_MINUTES_MS;

    if (currentBar && windowStart !== currentWindowStart) {
      bars.push(currentBar);
      segments.push(currentSegment);
      currentSegment = [];
      currentBar = null;
    }

    if (!currentBar) {
      currentWindowStart = windowStart;
      currentBar = {
        timestamp: secondBar.timestamp,
        open: secondBar.open,
        high: secondBar.high,
        low: secondBar.low,
        close: secondBar.close,
        volume: secondBar.volume,
      };
    } else {
      currentBar.high = Math.max(currentBar.high, secondBar.high);
      currentBar.low = Math.min(currentBar.low, secondBar.low);
      currentBar.close = secondBar.close;
      currentBar.timestamp = secondBar.timestamp;
      if (secondBar.volume != null) {
        currentBar.volume = (currentBar.volume ?? 0) + secondBar.volume;
      }
    }

    currentSegment.push(secondBar);
  }

  if (currentBar) {
    bars.push(currentBar);
    segments.push(currentSegment);
  }

  return { bars, segments };
}

function findCrossWithinBar(
  startPrice: number,
  intrabars: TopstepXFuturesBar[],
  sma: number,
  direction: 'above' | 'below',
) {
  let lastPrice = startPrice;

  for (const mini of intrabars) {
    const price = mini.close;
    const allowed = isTradingAllowed(mini.timestamp);
    if (direction === 'above') {
      if (lastPrice <= sma && price > sma && allowed) {
        return { price, timestamp: mini.timestamp };
      }
    } else if (direction === 'below') {
      if (lastPrice >= sma && price < sma && allowed) {
        return { price, timestamp: mini.timestamp };
      }
    }
    lastPrice = price;
  }

  return null;
}

async function runBacktest() {
  console.log('Starting TopstepX 5-min SMA crossover backtest (built from 1s bars) with config:', CONFIG);

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[topstepx-backtest] Unable to fetch metadata:', err.message);
    return null;
  });
  if (!metadata) {
    throw new Error(`Unable to resolve Topstep symbol/contractid for ${lookupKey}`);
  }

  const contractId = metadata.id;
  const multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || CONFIG.contractMultiplier;

  console.log(`Resolved contract: ${metadata.name} (${contractId})`);
  console.log(`Using point multiplier ${multiplier}`);

  const secondBars = await fetchSecondBars(contractId, CONFIG.start, CONFIG.end);
  if (!secondBars.length) {
    throw new Error('Topstep returned zero second bars for the requested range.');
  }

  const { bars: fiveMinBars, segments: intrabarSegments } = aggregateSecondsToFiveMinutes(secondBars);
  if (!fiveMinBars.length) {
    throw new Error('Failed to build any five-minute bars from second data.');
  }

  const closes = fiveMinBars.map(bar => bar.close);
  const smaSeries = calculateSMA(closes, CONFIG.smaPeriod);
  const rsiSeries = RSI.calculate({ values: closes, period: CONFIG.rsiPeriod });

  let position: 'long' | 'short' | null = null;
  let entryPrice = 0;
  let entryTime = '';
  let realizedPnL = 0;
  const trades: TradeRecord[] = [];

  const exitPosition = (exitPrice: number, exitTime: string, reason: TradeRecord['exitReason']) => {
    if (!position) return;
    const direction = position === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - entryPrice) * direction * multiplier;
    const commissionCost = CONFIG.commissionPerSide * 2;
    const pnl = rawPnL - commissionCost;
    trades.push({
      entryTime,
      exitTime,
      side: position,
      entryPrice,
      exitPrice,
      pnl,
      exitReason: reason,
    });
    realizedPnL += pnl;
    position = null;
    entryPrice = 0;
    entryTime = '';
  };

  for (let i = 1; i < fiveMinBars.length; i += 1) {
    const bar = fiveMinBars[i];
    const prevBar = fiveMinBars[i - 1];
    const prevPrice = prevBar.close;
    const currPrice = bar.close;
    const prevSma = smaSeries[i - 1];
    const currSma = smaSeries[i];
    const currRsi = rsiSeries[i - 1];
    const prevRsi = i - 2 >= 0 ? rsiSeries[i - 2] : null;
    if (
      prevSma == null ||
      currSma == null ||
      currRsi == null ||
      prevRsi == null
    ) {
      continue;
    }

    const intrabars = intrabarSegments[i] || [bar];

    for (const mini of intrabars) {
      if (position && !isTradingAllowed(mini.timestamp)) {
        exitPosition(mini.close, mini.timestamp, 'session');
      }

      if (position) {
        const direction = position === 'long' ? 1 : -1;
        const target = direction === 1
          ? entryPrice * (1 + CONFIG.takeProfitPercent)
          : entryPrice * (1 - CONFIG.takeProfitPercent);
        const stop = direction === 1
          ? entryPrice * (1 - CONFIG.stopLossPercent)
          : entryPrice * (1 + CONFIG.stopLossPercent);

        if (direction === 1 && mini.high >= target) {
          exitPosition(target, mini.timestamp, 'target');
        } else if (direction === 1 && mini.low <= stop) {
          exitPosition(stop, mini.timestamp, 'stop');
        } else if (direction === -1 && mini.low <= target) {
          exitPosition(target, mini.timestamp, 'target');
        } else if (direction === -1 && mini.high >= stop) {
          exitPosition(stop, mini.timestamp, 'stop');
        }
      }

      if (!position) {
        break;
      }
    }

    if (position && !isTradingAllowed(bar.timestamp)) {
      exitPosition(bar.close, bar.timestamp, 'session');
    }

    if (position) {
      continue;
    }

    const crossedUp = prevPrice <= prevSma && currPrice > currSma;
    const crossedDown = prevPrice >= prevSma && currPrice < currSma;

    const crossUpInfo = crossedUp
      ? findCrossWithinBar(prevPrice, intrabars, currSma, 'above')
      : null;
    const crossDownInfo = crossedDown
      ? findCrossWithinBar(prevPrice, intrabars, currSma, 'below')
      : null;

    const rsiBullish = currRsi > 50 && currRsi > prevRsi;
    const rsiBearish = currRsi < 50 && currRsi < prevRsi;

    if (crossedUp && rsiBullish) {
      if (crossUpInfo && isTradingAllowed(crossUpInfo.timestamp)) {
        position = 'long';
        entryPrice = crossUpInfo.price;
        entryTime = crossUpInfo.timestamp;
      }
    } else if (crossedDown && rsiBearish) {
      if (crossDownInfo && isTradingAllowed(crossDownInfo.timestamp)) {
        position = 'short';
        entryPrice = crossDownInfo.price;
        entryTime = crossDownInfo.timestamp;
      }
    }
  }

  if (position) {
    const lastBar = fiveMinBars[fiveMinBars.length - 1];
    exitPosition(lastBar.close, lastBar.timestamp, 'end_of_data');
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

  console.log('\n===== 5-Min SMA Backtest Summary =====');
  console.log(`Trades: ${trades.length} | Win rate: ${trades.length ? ((wins.length / trades.length) * 100).toFixed(1) : '0'}%`);
  console.log(`Realized PnL: ${formatCurrency(realizedPnL)} USD`);
  console.log(`Avg Win: ${formatCurrency(avgWin)} | Avg Loss: ${formatCurrency(avgLoss)}`);
  console.log(`Max Drawdown: ${formatCurrency(maxDrawdown)} USD`);

  if (trades.length) {
    console.log('\nRecent trades:');
    trades.slice(-5).forEach(trade => {
      console.log(
        ` - ${trade.side.toUpperCase()} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | ${formatCurrency(trade.pnl)} (${trade.exitReason})`,
      );
    });
  }
}

runBacktest().catch(err => {
  console.error('TopstepX SMA backtest failed:', err);
  process.exit(1);
});
