import 'dotenv/config';
import {
  generateMeanReversionSignalFromTechnicals,
  MeanReversionSignal,
} from './lib/meanReversionAgent';

type Direction = 'long' | 'short';

interface Candle {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

interface TradeResult {
  direction: Direction;
  entryPrice: number;
  exitPrice: number;
  entryTime: string;
  exitTime: string;
  reason: 'target' | 'stop' | 'final';
  pnl: number;
  pnlPct: number;
  signal: MeanReversionSignal;
  scaled: boolean;
  scalePrice: number | null;
  notional: number;
  fees: number;
}

interface ActivePosition {
  direction: Direction;
  entryPrice: number;
  entryTime: string;
  stop: number | null;
  target: number | null;
  scaled: boolean;
  remainingUnits: number;
  scalePrice: number | null;
  signal: MeanReversionSignal;
  entryIndex: number;
  qtyPerUnit: number;
  realizedPnl: number;
  feesPaid: number;
}

interface BacktestConfig {
  symbol: string;
  start: Date;
  end: Date;
  granularity: number;
}

async function fetchCoinbaseCandles(symbol: string, granularity: number, start: Date, end: Date): Promise<Candle[]> {
  const supportedGranularities = new Set([60, 300, 900, 3600, 21600, 86400]);
  let fetchGranularity = granularity;
  let aggregateFactor = 1;

  if (!supportedGranularities.has(granularity)) {
    if (granularity === 1800) {
      fetchGranularity = 900; // use 15m and combine pairs
      aggregateFactor = 2;
    } else {
      throw new Error(`Unsupported Coinbase granularity: ${granularity}`);
    }
  }

  const maxChunkSeconds = fetchGranularity * 300; // Coinbase returns up to 300 candles per request
  const bars: Candle[] = [];

  let chunkStart = new Date(start);

  while (chunkStart < end) {
    const chunkEnd = new Date(Math.min(end.getTime(), chunkStart.getTime() + maxChunkSeconds * 1000));
    const url = new URL(`https://api.exchange.coinbase.com/products/${symbol}/candles`);
    url.searchParams.set('granularity', String(fetchGranularity));
    url.searchParams.set('start', chunkStart.toISOString());
    url.searchParams.set('end', chunkEnd.toISOString());

    const response = await fetch(url.toString());
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Coinbase candle fetch failed: ${response.status} ${response.statusText} - ${body}`);
    }

    const data = (await response.json()) as number[][];
    for (const candle of data) {
      if (!Array.isArray(candle) || candle.length < 6) continue;
      const [timestamp, low, high, open, close, volume] = candle;
      bars.push({
        t: new Date(timestamp * 1000).toISOString(),
        o: open,
        h: high,
        l: low,
        c: close,
        v: volume,
      });
    }

    chunkStart = new Date(chunkEnd.getTime() + fetchGranularity * 1000);
    await new Promise(resolve => setTimeout(resolve, 200)); // Respect public rate limits
  }

  // Coinbase returns newest first when using start/end – sort ascending for chronological processing
  bars.sort((a, b) => new Date(a.t).getTime() - new Date(b.t).getTime());

  if (aggregateFactor > 1) {
    const aggregated: Candle[] = [];
    for (let i = 0; i + aggregateFactor - 1 < bars.length; i += aggregateFactor) {
      const slice = bars.slice(i, i + aggregateFactor);
      if (slice.length < aggregateFactor) {
        continue;
      }

      aggregated.push({
        t: slice[0].t,
        o: slice[0].o,
        h: Math.max(...slice.map(bar => bar.h)),
        l: Math.min(...slice.map(bar => bar.l)),
        c: slice[slice.length - 1].c,
        v: slice.reduce((sum, bar) => sum + bar.v, 0),
      });
    }

    return aggregated;
  }

  return bars;
}

function formatUsd(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1) {
    return `$${value.toFixed(2)}`;
  }
  if (abs >= 0.01) {
    return `$${value.toFixed(4)}`;
  }
  return `$${value.toFixed(6)}`;
}

const PER_UNIT_NOTIONAL = Number(process.env.CRYPTO_UNIT_NOTIONAL || '50'); // $50 per unit by default
const BASE_UNITS = 2;
const BASE_NOTIONAL = PER_UNIT_NOTIONAL * BASE_UNITS;
const FEE_RATE = 0.006; // 0.6% per side

async function runBacktest(config: BacktestConfig) {
  const { symbol, start, end, granularity } = config;
  console.log(`\n📊 Backtesting ${symbol} from ${start.toISOString()} to ${end.toISOString()} (${granularity / 60} min bars)\n`);

  const candles = await fetchCoinbaseCandles(symbol, granularity, start, end);
  if (candles.length === 0) {
    console.log('No candles fetched; aborting.');
    return;
  }

  console.log(`Fetched ${candles.length} candles`);

  const priceHistory: number[] = [];
  const trades: TradeResult[] = [];

  let position: ActivePosition | null = null;

  for (let index = 0; index < candles.length; index += 1) {
    const bar = candles[index];
    priceHistory.push(bar.c);

    if (position && index > position.entryIndex) {
      const scaledThisBar = maybeScalePosition(position, bar);
      if (scaledThisBar) {
        continue;
      }

      const exitDecision = evaluateExit(position, bar);
      if (exitDecision) {
        trades.push(
          finalizeTrade(position, exitDecision.exitPrice, bar.t, exitDecision.reason),
        );
        position = null;
      }
    }

    if (position) {
      continue;
    }

    const signal = generateMeanReversionSignalFromTechnicals(
      symbol,
      bar.c,
      priceHistory.slice(),
      1,
      {
        rsiPeriod: 14,
        rsiOversold: 30,
        rsiOverbought: 70,
        bbPeriod: 20,
        bbStdDev: 2,
        bbThreshold: 0.005,
        stopLossPercent: 0.01,
      },
    );

    if (signal.direction === 'none') {
      continue;
    }

    const direction: Direction = signal.direction === 'long' ? 'long' : 'short';
    const defaultStop =
      direction === 'long'
        ? bar.c * (1 - 0.01)
        : bar.c * (1 + 0.01);
    const defaultTarget =
      signal.target ?? signal.bbMiddle ?? bar.c;
    const qtyPerUnit = PER_UNIT_NOTIONAL / bar.c;
    const entryNotional = bar.c * qtyPerUnit * BASE_UNITS;

    position = {
      direction,
      entryPrice: bar.c,
      entryTime: bar.t,
      stop: signal.stopLoss ?? defaultStop,
      target: defaultTarget,
      scaled: false,
      remainingUnits: BASE_UNITS,
      scalePrice: null,
      signal: { ...signal, stopLoss: signal.stopLoss ?? defaultStop, target: defaultTarget },
      entryIndex: index,
      qtyPerUnit,
      realizedPnl: 0,
      feesPaid: entryNotional * FEE_RATE,
    };
  }

  if (position) {
    const lastBar = candles[candles.length - 1];
    trades.push(
      finalizeTrade(position, lastBar.c, lastBar.t, 'final'),
    );
  }

  if (trades.length === 0) {
    console.log('No trades generated.');
    return;
  }

  const totalPnl = trades.reduce((sum, trade) => sum + trade.pnl, 0);
  const totalFees = trades.reduce((sum, trade) => sum + trade.fees, 0);
  const wins = trades.filter(trade => trade.pnl > 0).length;
  const losses = trades.filter(trade => trade.pnl <= 0).length;
  const avgWin = wins ? trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0) / wins : 0;
  const avgLoss = losses ? trades.filter(t => t.pnl <= 0).reduce((sum, t) => sum + t.pnl, 0) / losses : 0;

  console.log(`Trades: ${trades.length}`);
  console.log(`Wins: ${wins} | Losses: ${losses} | Win rate: ${(wins / trades.length * 100).toFixed(1)}%`);
  console.log(`Total PnL (2 units ≈ ${formatUsd(BASE_NOTIONAL)}): ${formatUsd(totalPnl)} | Avg win: ${formatUsd(avgWin)} | Avg loss: ${formatUsd(avgLoss)} | Fees: ${formatUsd(totalFees)}`);

  console.log('\nDetailed trades:');
  for (const trade of trades) {
    const directionEmoji = trade.direction === 'long' ? '🟢' : '🔴';
    const scaledNote = trade.scaled ? ` | Scaled @ ${formatUsd(trade.scalePrice ?? 0)}` : '';
    console.log(
      `${directionEmoji} ${trade.direction.toUpperCase()} | ${trade.entryTime} -> ${trade.exitTime} | ` +
        `Entry ${formatUsd(trade.entryPrice)} (~${formatUsd(trade.notional)}) | Exit ${formatUsd(trade.exitPrice)} | ` +
        `PnL ${formatUsd(trade.pnl)} (${(trade.pnlPct * 100).toFixed(2)}%) | Fees ${formatUsd(trade.fees)} | Reason: ${trade.reason}${scaledNote}`,
    );
  }
}

function maybeScalePosition(position: ActivePosition, bar: Candle): boolean {
  if (position.scaled || position.target === null) {
    return false;
  }

  const { direction, target } = position;
  const barHigh = bar.h;
  const barLow = bar.l;

  const targetHit =
    (direction === 'long' && Number.isFinite(barHigh) && barHigh >= target) ||
    (direction === 'short' && Number.isFinite(barLow) && barLow <= target);

  if (!targetHit) {
    return false;
  }

  position.scaled = true;
  position.remainingUnits = Math.max(1, position.remainingUnits - 1);
  position.scalePrice = target;

  const unitProfit = position.direction === 'long'
    ? (target - position.entryPrice) * position.qtyPerUnit
    : (position.entryPrice - target) * position.qtyPerUnit;
  position.realizedPnl += unitProfit;
  position.feesPaid += (target * position.qtyPerUnit) * FEE_RATE;

  if (direction === 'long') {
    const middleBand = target;
    const outerBand = position.signal.bbUpper ?? target;
    const newStop = middleBand * 0.999;
    position.stop = newStop;
    position.target = outerBand;
    position.signal.stopLoss = newStop;
    position.signal.target = outerBand;
  } else {
    const middleBand = target;
    const outerBand = position.signal.bbLower ?? target;
    const newStop = middleBand * 1.001;
    position.stop = newStop;
    position.target = outerBand;
    position.signal.stopLoss = newStop;
    position.signal.target = outerBand;
  }

  return true;
}

function evaluateExit(position: ActivePosition, bar: Candle): { exitPrice: number; reason: 'stop' | 'target' } | null {
  const { direction, stop, target } = position;
  const barHigh = bar.h;
  const barLow = bar.l;
  const barOpen = bar.o;

  const referencePrice = Number.isFinite(barOpen) && barOpen > 0 ? barOpen : position.entryPrice;

  let exitTriggered = false;
  let exitPrice = position.entryPrice;
  let exitReason: 'stop' | 'target' = 'stop';

  const considerExit = (reason: 'stop' | 'target', price: number) => {
    if (!Number.isFinite(price)) {
      return;
    }

    if (!exitTriggered) {
      exitTriggered = true;
      exitReason = reason;
      exitPrice = price;
      return;
    }

    const existingDistance = Math.abs(referencePrice - exitPrice);
    const candidateDistance = Math.abs(referencePrice - price);
    if (candidateDistance < existingDistance) {
      exitReason = reason;
      exitPrice = price;
    }
  };

  if (direction === 'long') {
    if (typeof stop === 'number' && Number.isFinite(barLow) && barLow <= stop) {
      considerExit('stop', stop);
    }
    if (typeof target === 'number' && Number.isFinite(barHigh) && barHigh >= target) {
      considerExit('target', target);
    }
  } else {
    if (typeof stop === 'number' && Number.isFinite(barHigh) && barHigh >= stop) {
      considerExit('stop', stop);
    }
    if (typeof target === 'number' && Number.isFinite(barLow) && barLow <= target) {
      considerExit('target', target);
    }
  }

  return exitTriggered ? { exitPrice, reason: exitReason } : null;
}

function finalizeTrade(
  position: ActivePosition,
  exitPrice: number,
  exitTime: string,
  reason: 'stop' | 'target' | 'final',
): TradeResult {
  const {
    direction,
    entryPrice,
    entryTime,
    signal,
    scaled,
    scalePrice,
    remainingUnits,
    qtyPerUnit,
    realizedPnl,
    feesPaid,
  } = position;

  const unitsLeft = Math.max(0, remainingUnits);
  const perUnitMove = direction === 'long'
    ? exitPrice - entryPrice
    : entryPrice - exitPrice;
  const finalProfit = perUnitMove * qtyPerUnit * unitsLeft;
  const exitNotional = exitPrice * qtyPerUnit * unitsLeft;
  const grossPnl = realizedPnl + finalProfit;
  const totalFees = feesPaid + exitNotional * FEE_RATE;
  const netPnl = grossPnl - totalFees;
  const pnlPct = netPnl / BASE_NOTIONAL;
  const notional = entryPrice * qtyPerUnit * BASE_UNITS;

  return {
    direction,
    entryPrice,
    exitPrice,
    entryTime,
    exitTime,
    reason,
    pnl: netPnl,
    pnlPct,
    signal,
    scaled,
    scalePrice: scalePrice ?? null,
    notional,
    fees: totalFees,
  };
}

function parseArgs(): BacktestConfig {
  const symbol = process.argv[2] || 'BTC-USD';
  const days = Number(process.argv[3] || '5');
  const granularityMinutes = Number(process.argv[4] || '15');

  if (Number.isNaN(days) || days <= 0) {
    throw new Error('Days argument must be a positive number');
  }

  if (![1, 5, 15, 30, 60].includes(granularityMinutes)) {
    throw new Error('Granularity minutes must be one of 1, 5, 15, 30, or 60');
  }

  const end = new Date();
  const start = new Date(end.getTime() - days * 24 * 60 * 60 * 1000);

  return {
    symbol,
    start,
    end,
    granularity: granularityMinutes * 60,
  };
}

async function main() {
  try {
    const config = parseArgs();
    await runBacktest(config);
  } catch (error) {
    console.error('Backtest failed:', (error as Error).message);
    process.exit(1);
  }
}

main();
