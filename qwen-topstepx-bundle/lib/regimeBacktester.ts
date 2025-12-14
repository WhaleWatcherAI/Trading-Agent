import path from 'path';
import { promises as fs } from 'fs';
import {
  RegimeWhaleTrade,
  BacktestPricePoint,
  BacktestStrikeHighlight,
  SectorTimelinePoint,
  SectorBacktestSummary,
  SectorBacktestResult,
  AggregatedBacktestSummary,
  RegimeBacktestResult,
  BacktestTrade,
  BacktestTradeMarker,
  EquityCurvePoint,
  RegimeBias,
  RegimeBiasStrength,
} from '@/types';
import { getHistoricalTimesales, TradierTimesaleBar } from './tradier';
import { getWhaleFlowAlerts, WhaleFlowAlert } from './unusualwhales';

type AgentMode = 'scalp' | 'swing' | 'leaps';

interface HistoricalOptionTrade {
  symbol: string;
  underlying?: string;
  strike: number;
  expiration: string;
  type: 'call' | 'put';
  side: 'bid' | 'ask' | 'mid';
  premium: number;
  volume: number;
  openInterest: number;
  timestamp: string;
  unusual?: boolean;
}

interface HistoricalInstitutionalTrade {
  symbol: string;
  shares: number;
  price: number;
  value: number;
  side: 'buy' | 'sell';
  institution?: string;
  timestamp: string;
}

interface HistoricalDaySnapshot {
  date: string;
  optionsTrades: HistoricalOptionTrade[];
  institutionalTrades: HistoricalInstitutionalTrade[];
}

export interface RegimeBacktestConfig {
  date: string;
  mode?: AgentMode;
  intervalMinutes?: number;
  useTradierPrices?: boolean;
  whalePremiumThreshold?: number;
  whaleVolumeThreshold?: number;
  sectorTickerMap?: Record<string, string>;
  fetchLiveFlow?: boolean;
  liveLookbackMinutes?: number;
  symbols?: string[];
}

const DEFAULT_SECTOR_TICKERS: Record<string, string> = {
  Technology: 'XLK',
  'Technology-SECTOR': 'XLK',
  Healthcare: 'XLV',
  'Healthcare-SECTOR': 'XLV',
  'Financial Services': 'XLF',
  'Financial Services-SECTOR': 'XLF',
  Energy: 'XLE',
  'Energy-SECTOR': 'XLE',
  'Consumer Cyclical': 'XLY',
  'Consumer Cyclical-SECTOR': 'XLY',
};

const DEFAULT_WHALE_PREMIUM = 500_000;
const DEFAULT_WHALE_VOLUME = 20_000;
const DEFAULT_INTERVAL = 1;

function resolveSectorKey(trade: HistoricalOptionTrade): string {
  return (trade.underlying || trade.symbol || 'UNKNOWN').trim() || 'UNKNOWN';
}

function getEasternOffset(date: string): string {
  const month = parseInt(date.slice(5, 7), 10);
  const day = parseInt(date.slice(8, 10), 10);

  if (Number.isNaN(month) || Number.isNaN(day)) {
    return '-04:00';
  }

  if (month > 3 && month < 11) return '-04:00';
  if (month === 3) {
    return day >= 8 ? '-04:00' : '-05:00';
  }
  if (month === 11) {
    return day <= 7 ? '-04:00' : '-05:00';
  }
  return '-05:00';
}

function toMinuteIso(date: Date): string {
  const iso = date.toISOString();
  return `${iso.slice(0, 16)}:00Z`;
}

function parseTimesaleTimestamp(input: string, date: string): Date {
  const trimmed = (input || '').trim();
  if (!trimmed) return new Date(NaN);

  let withDate = trimmed;
  if (/^\d{2}:\d{2}:\d{2}$/.test(trimmed)) {
    withDate = `${date}T${trimmed}`;
  } else if (trimmed.includes(' ')) {
    withDate = trimmed.replace(' ', 'T');
  } else if (!trimmed.includes('T')) {
    withDate = `${date}T${trimmed}`;
  }

  if (!/[+-]\d{2}:\d{2}$/.test(withDate) && !withDate.endsWith('Z')) {
    withDate = `${withDate}${getEasternOffset(date)}`;
  }

  return new Date(withDate);
}

function determineStrength(ratio: number, totalPremium: number): RegimeBiasStrength {
  if (!Number.isFinite(ratio) || totalPremium <= 0) {
    return 'low';
  }
  const magnitude = Math.abs(ratio);
  if (magnitude >= 0.35) return 'high';
  if (magnitude >= 0.2) return 'medium';
  return 'low';
}

function determineBias(callPremium: number, putPremium: number): { bias: RegimeBias; ratio: number } {
  const total = callPremium + putPremium;
  if (total <= 0) {
    return { bias: 'balanced', ratio: 0 };
  }
  const ratio = (callPremium - putPremium) / total;
  if (ratio > 0.15) {
    return { bias: 'bullish', ratio };
  }
  if (ratio < -0.15) {
    return { bias: 'bearish', ratio };
  }
  return { bias: 'balanced', ratio };
}

async function readHistoricalSnapshot(date: string): Promise<HistoricalDaySnapshot | null> {
  const filePath = path.join(process.cwd(), 'data', `${date}.json`);
  try {
    const raw = await fs.readFile(filePath, 'utf-8');
    const parsed = JSON.parse(raw);
    return {
      date: parsed.date || date,
      optionsTrades: Array.isArray(parsed.optionsTrades) ? parsed.optionsTrades : [],
      institutionalTrades: Array.isArray(parsed.institutionalTrades) ? parsed.institutionalTrades : [],
    };
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

async function fetchLiveWhaleFlow(config: RegimeBacktestConfig): Promise<HistoricalOptionTrade[]> {
  if (!config.fetchLiveFlow) {
    return [];
  }

  try {
    const alerts = await getWhaleFlowAlerts({
      symbols: config.symbols,
      lookbackMinutes: config.liveLookbackMinutes ?? 120,
      limit: 500,
    });

    return alerts.map((alert: WhaleFlowAlert, index: number) => ({
      symbol: alert.ticker,
      underlying: alert.underlying,
      strike: Number.isFinite(alert.strike) ? alert.strike : 0,
      expiration: alert.expiration,
      type: alert.optionType,
      side: alert.direction === 'bullish' ? 'ask' : alert.direction === 'bearish' ? 'bid' : 'mid',
      premium: alert.premium,
      volume: alert.contracts,
      openInterest: 0,
      timestamp: alert.timestamp || new Date().toISOString(),
      unusual: true,
    })) as HistoricalOptionTrade[];
  } catch (error) {
    console.error('Failed to fetch live whale flow for backtest:', error);
    return [];
  }
}

interface TimelineBucket {
  callPremium: number;
  putPremium: number;
  callVolume: number;
  putVolume: number;
  totalTrades: number;
  whaleTrades: RegimeWhaleTrade[];
  callHighlights: Map<string, BacktestStrikeHighlight>;
  putHighlights: Map<string, BacktestStrikeHighlight>;
}

interface SectorComputation {
  timeline: SectorTimelinePoint[];
  summary: SectorBacktestSummary;
}

function buildStrikeHighlight(trade: HistoricalOptionTrade): BacktestStrikeHighlight {
  return {
    strike: trade.strike,
    expiration: trade.expiration,
    premium: trade.premium,
    volume: trade.volume,
    openInterest: trade.openInterest,
  };
}

function mergeHighlight(
  acc: Map<string, BacktestStrikeHighlight>,
  trade: HistoricalOptionTrade,
): void {
  const key = `${trade.strike}_${trade.expiration}`;
  const existing = acc.get(key);
  if (!existing) {
    acc.set(key, buildStrikeHighlight(trade));
    return;
  }
  existing.premium += trade.premium;
  existing.volume += trade.volume;
  existing.openInterest = Math.max(existing.openInterest, trade.openInterest);
}

function computeSectorTimeline(
  trades: HistoricalOptionTrade[],
  config: RegimeBacktestConfig,
): SectorComputation {
  const whalePremiumThreshold = config.whalePremiumThreshold ?? DEFAULT_WHALE_PREMIUM;
  const whaleVolumeThreshold = config.whaleVolumeThreshold ?? DEFAULT_WHALE_VOLUME;
  const bucketMap = new Map<string, TimelineBucket>();
  const expirationTotals = new Map<string, { callPremium: number; putPremium: number; trades: number }>();
  const whaleCandidates: RegimeWhaleTrade[] = [];

  trades.forEach(trade => {
    const date = new Date(trade.timestamp);
    if (Number.isNaN(date.getTime())) {
      return;
    }

    const bucketKey = toMinuteIso(date);
    const bucket = bucketMap.get(bucketKey) || {
      callPremium: 0,
      putPremium: 0,
      callVolume: 0,
      putVolume: 0,
      totalTrades: 0,
      whaleTrades: [] as RegimeWhaleTrade[],
      callHighlights: new Map<string, BacktestStrikeHighlight>(),
      putHighlights: new Map<string, BacktestStrikeHighlight>(),
    };

    if (trade.type === 'call') {
      bucket.callPremium += trade.premium;
      bucket.callVolume += trade.volume;
      mergeHighlight(bucket.callHighlights, trade);
    } else if (trade.type === 'put') {
      bucket.putPremium += trade.premium;
      bucket.putVolume += trade.volume;
      mergeHighlight(bucket.putHighlights, trade);
    }

    bucket.totalTrades += 1;

    const premiumPerContract =
      trade.volume > 0 ? trade.premium / (trade.volume * 100) : trade.premium;

    if (
      trade.premium >= whalePremiumThreshold ||
      trade.volume >= whaleVolumeThreshold
    ) {
      const whale: RegimeWhaleTrade = {
        optionType: trade.type,
        direction: trade.type === 'call' ? 'bullish' : 'bearish',
        contracts: trade.volume,
        premium: trade.premium,
        strike: trade.strike,
        expiration: trade.expiration,
        midpointPrice: premiumPerContract,
        timestamp: trade.timestamp,
      };
      bucket.whaleTrades.push(whale);
      whaleCandidates.push(whale);
    }

    bucketMap.set(bucketKey, bucket);

    const expKey = trade.expiration || 'unknown';
    const expTotals = expirationTotals.get(expKey) || { callPremium: 0, putPremium: 0, trades: 0 };
    if (trade.type === 'call') {
      expTotals.callPremium += trade.premium;
    } else {
      expTotals.putPremium += trade.premium;
    }
    expTotals.trades += 1;
    expirationTotals.set(expKey, expTotals);
  });

  const sortedBuckets = Array.from(bucketMap.entries()).sort(
    (a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime(),
  );

  let cumulativeCall = 0;
  let cumulativePut = 0;

  const timeline: SectorTimelinePoint[] = sortedBuckets.map(([timestamp, bucket]) => {
    const { bias, ratio } = determineBias(bucket.callPremium, bucket.putPremium);
    const strength = determineStrength(ratio, bucket.callPremium + bucket.putPremium);
    cumulativeCall += bucket.callPremium;
    cumulativePut += bucket.putPremium;

    const topCall = Array.from(bucket.callHighlights.values()).sort(
      (a, b) => b.premium - a.premium,
    )[0];
    const topPut = Array.from(bucket.putHighlights.values()).sort(
      (a, b) => b.premium - a.premium,
    )[0];

    return {
      timestamp,
      callPremium: bucket.callPremium,
      putPremium: bucket.putPremium,
      netPremium: bucket.callPremium - bucket.putPremium,
      callVolume: bucket.callVolume,
      putVolume: bucket.putVolume,
      totalTrades: bucket.totalTrades,
      whaleTrades: bucket.whaleTrades.sort((a, b) => b.premium - a.premium).slice(0, 5),
      bias,
      strength,
      cumulativeCallPremium: cumulativeCall,
      cumulativePutPremium: cumulativePut,
      topCallHighlight: topCall,
      topPutHighlight: topPut,
      markers: [],
    };
  });

  const biasCounts: Record<RegimeBias, number> = {
    bullish: 0,
    bearish: 0,
    balanced: 0,
  };
  let transitions = 0;
  let lastBias: RegimeBias | null = null;

  timeline.forEach(point => {
    biasCounts[point.bias] += 1;
    if (lastBias && lastBias !== point.bias) {
      transitions += 1;
    }
    if (point.bias !== 'balanced') {
      lastBias = point.bias;
    }
  });

  const dominantBias =
    (Object.entries(biasCounts).sort((a, b) => b[1] - a[1])[0]?.[0] as RegimeBias) ||
    'balanced';

  const totalCallPremium = timeline.reduce((sum, point) => sum + point.callPremium, 0);
  const totalPutPremium = timeline.reduce((sum, point) => sum + point.putPremium, 0);

  const dominantExpirations = Array.from(expirationTotals.entries())
    .map(([expiration, data]) => ({
      expiration,
      callPremium: data.callPremium,
      putPremium: data.putPremium,
      netPremium: data.callPremium - data.putPremium,
      trades: data.trades,
    }))
    .sort((a, b) => Math.abs(b.netPremium) - Math.abs(a.netPremium))
    .slice(0, 5);

  const topWhales = whaleCandidates.sort((a, b) => b.premium - a.premium).slice(0, 10);

  const summary: SectorBacktestSummary = {
    minutes: timeline.length,
    totalCallPremium,
    totalPutPremium,
    netPremium: totalCallPremium - totalPutPremium,
    whaleTrades: whaleCandidates.length,
    dominantBias,
    biasCounts,
    regimeTransitions: transitions,
    dominantExpirations,
    topWhales,
    tradeCount: 0,
    totalProfit: 0,
    grossProfit: 0,
    grossLoss: 0,
  };

  return { timeline, summary };
}

function attachPricesToTimeline(
  timeline: SectorTimelinePoint[],
  bars: TradierTimesaleBar[],
  date: string,
): { pricePoints: BacktestPricePoint[]; priceChangePct?: number } {
  if (!timeline.length || !bars.length) {
    return { pricePoints: [] };
  }

  const barMap = new Map<string, TradierTimesaleBar>();
  bars.forEach(bar => {
    // Use bar.time (ISO string) instead of bar.timestamp (Unix number)
    const parsed = parseTimesaleTimestamp(bar.time || bar.timestamp, date);
    const key = toMinuteIso(parsed);
    barMap.set(key, bar);
  });

  const pricePoints: BacktestPricePoint[] = [];

  timeline.forEach(point => {
    const bar = barMap.get(point.timestamp);
    if (!bar) return;
    const close = Number(bar.close);
    if (!Number.isFinite(close) || close <= 0) return;

    const open = Number.isFinite(bar.open) ? bar.open : close;
    const high = Number.isFinite(bar.high) ? bar.high : Math.max(open, close);
    const low = Number.isFinite(bar.low) ? bar.low : Math.min(open, close);
    const volume = Number.isFinite(bar.volume) ? bar.volume : 0;

    const pricePoint: BacktestPricePoint = {
      timestamp: point.timestamp,
      open,
      high,
      low,
      close,
      volume,
    };

    point.price = pricePoint;
    pricePoints.push(pricePoint);
  });

  let priceChangePct: number | undefined;
  if (pricePoints.length >= 2) {
    const first = pricePoints[0];
    const last = pricePoints[pricePoints.length - 1];
    if (first && last && first.close > 0) {
      priceChangePct = ((last.close - first.close) / first.close) * 100;
    }
  }

  return { pricePoints, priceChangePct };
}

interface ActivePosition {
  id: string;
  direction: 'long' | 'short';
  entryTimestamp: string;
  entryPrice: number;
  entryReason: string;
}

interface TradeSimulationResult {
  trades: BacktestTrade[];
  equityCurve: EquityCurvePoint[];
  metrics: {
    tradeCount: number;
    winRate?: number;
    totalProfit: number;
    grossProfit: number;
    grossLoss: number;
    maxDrawdown?: number;
    averageDurationMinutes?: number;
  };
}

const STOP_LOSS_PCT = 0.01;
const TAKE_PROFIT_PCT = 0.015;
const MIN_HOLD_MINUTES = 5;

function minutesBetween(start: string, end: string): number {
  const delta = new Date(end).getTime() - new Date(start).getTime();
  if (!Number.isFinite(delta)) return 0;
  return Math.max(1, Math.round(delta / 60000));
}

function ensureMarkers(point: SectorTimelinePoint): BacktestTradeMarker[] {
  if (!point.markers) {
    point.markers = [];
  }
  return point.markers;
}

function simulateTradesFromTimeline(timeline: SectorTimelinePoint[]): TradeSimulationResult {
  const trades: BacktestTrade[] = [];
  const equityCurve: EquityCurvePoint[] = [];
  let position: ActivePosition | null = null;
  let equity = 0;
  let peakEquity = 0;
  let maxDrawdown = 0;
  let wins = 0;
  let totalDuration = 0;
  let lastPointWithPrice: (SectorTimelinePoint & { price: BacktestPricePoint }) | null = null;

  const finalizeTrade = (
    exitPoint: SectorTimelinePoint,
    price: number,
    timestamp: string,
    reason: BacktestTrade['exitReason'],
  ) => {
    if (!position) return;
    const durationMinutes = minutesBetween(position.entryTimestamp, timestamp);
    const rawProfit = position.direction === 'long'
      ? price - position.entryPrice
      : position.entryPrice - price;
    const profitPct = rawProfit / Math.max(position.entryPrice, 1e-6);

    const trade: BacktestTrade = {
      id: position.id,
      direction: position.direction,
      entryTimestamp: position.entryTimestamp,
      exitTimestamp: timestamp,
      entryPrice: position.entryPrice,
      exitPrice: price,
      durationMinutes,
      profit: rawProfit,
      profitPct,
      tradeCost: position.entryPrice,
      exitReason: reason,
      entryReason: position.entryReason,
    };

    trades.push(trade);
    totalDuration += durationMinutes;
    if (rawProfit > 0) {
      wins += 1;
    }

    equity += rawProfit;
    peakEquity = Math.max(peakEquity, equity);
    maxDrawdown = Math.max(maxDrawdown, peakEquity - equity);
    equityCurve.push({ timestamp, equity });

    const markers = ensureMarkers(exitPoint);
    markers.push({
      type: 'exit',
      tradeId: trade.id,
      direction: trade.direction,
      timestamp,
      price,
      reason,
    });

    position = null;
  };

  timeline.forEach((point, index) => {
    if (!point.price) {
      return;
    }

    lastPointWithPrice = point as SectorTimelinePoint & { price: BacktestPricePoint };
    ensureMarkers(point);

    const price = point.price.close;
    const timestamp = point.timestamp;

    if (position) {
      const durationMinutes = minutesBetween(position.entryTimestamp, timestamp);
      const rawProfit = position.direction === 'long'
        ? price - position.entryPrice
        : position.entryPrice - price;
      const profitPct = rawProfit / Math.max(position.entryPrice, 1e-6);

      let exitReason: BacktestTrade['exitReason'] | null = null;

      if (profitPct <= -STOP_LOSS_PCT) {
        exitReason = 'stop';
      } else if (profitPct >= TAKE_PROFIT_PCT) {
        exitReason = 'target';
      } else if (
        (position.direction === 'long' && point.bias === 'bearish' && point.strength !== 'low') ||
        (position.direction === 'short' && point.bias === 'bullish' && point.strength !== 'low')
      ) {
        exitReason = 'flip';
      } else if (point.bias === 'balanced' && durationMinutes >= MIN_HOLD_MINUTES) {
        exitReason = 'flat';
      }

      if (exitReason) {
        finalizeTrade(point, price, timestamp, exitReason);
      }
    }

    if (!position) {
      const entryReason = point.bias === 'bullish'
        ? `Bias bullish (${point.strength})`
        : point.bias === 'bearish'
          ? `Bias bearish (${point.strength})`
          : null;

      if (entryReason && point.strength !== 'low') {
        const direction = point.bias === 'bullish' ? 'long' : 'short';
        const id = `${new Date(timestamp).getTime()}-${direction}-${index}`;

        position = {
          id,
          direction,
          entryTimestamp: timestamp,
          entryPrice: price,
          entryReason,
        };

        const markers = ensureMarkers(point);
        markers.push({
          type: 'entry',
          tradeId: id,
          direction,
          timestamp,
          price,
          reason: entryReason,
        });
      }
    }
  });

  if (position && lastPointWithPrice && lastPointWithPrice.price) {
    const exitPoint = lastPointWithPrice;
    const timestamp = exitPoint.timestamp;
    const price = exitPoint.price.close;
    finalizeTrade(exitPoint, price, timestamp, 'end_of_day');
  }

  const tradeCount = trades.length;
  const totalProfit = trades.reduce((sum, trade) => sum + trade.profit, 0);
  const grossProfit = trades.filter(trade => trade.profit > 0).reduce((sum, trade) => sum + trade.profit, 0);
  const grossLoss = trades.filter(trade => trade.profit <= 0).reduce((sum, trade) => sum + trade.profit, 0);
  const winRate = tradeCount > 0 ? wins / tradeCount : undefined;
  const averageDurationMinutes = tradeCount > 0 ? totalDuration / tradeCount : undefined;

  return {
    trades,
    equityCurve,
    metrics: {
      tradeCount,
      winRate,
      totalProfit,
      grossProfit,
      grossLoss,
      maxDrawdown: tradeCount > 0 ? maxDrawdown : undefined,
      averageDurationMinutes,
    },
  };
}

export async function runRegimeBacktest(
  config: RegimeBacktestConfig,
): Promise<RegimeBacktestResult> {
  const date = config.date;
  if (!date) {
    throw new Error('Regime backtest requires a target date (YYYY-MM-DD)');
  }

  const mode: AgentMode = config.mode || 'scalp';
  const intervalMinutes = config.intervalMinutes ?? DEFAULT_INTERVAL;
  const sectorTickerMap = {
    ...DEFAULT_SECTOR_TICKERS,
    ...(config.sectorTickerMap || {}),
  };

  const notes: string[] = [];

  const snapshot = await readHistoricalSnapshot(date);
  if (!snapshot) {
    notes.push(`No cached flow data found at data/${date}.json – proceeding with empty flow set.`);
  }

  let optionsTrades = snapshot?.optionsTrades ? [...snapshot.optionsTrades] : [];

  if (config.symbols && config.symbols.length > 0) {
    const symbolsUpper = config.symbols.map(symbol => symbol.toUpperCase());
    optionsTrades = optionsTrades.filter(trade => {
      const key = (trade.underlying || trade.symbol || '').toUpperCase();
      return symbolsUpper.includes(key);
    });
  }

  const liveFlow = await fetchLiveWhaleFlow(config);
  if (liveFlow.length > 0) {
    optionsTrades = optionsTrades.concat(liveFlow);
    notes.push(`Appended ${liveFlow.length} live whale flow records from Unusual Whales API.`);
  } else if (config.fetchLiveFlow) {
    notes.push('Live flow requested but no Unusual Whales records were returned for the selected window.');
  }

  if (optionsTrades.length === 0) {
    notes.push('Option flow dataset empty – results will only reflect price data.');
  }

  const groupedTrades = new Map<string, HistoricalOptionTrade[]>();
  optionsTrades.forEach(trade => {
    const key = resolveSectorKey(trade);
    if (!groupedTrades.has(key)) {
      groupedTrades.set(key, []);
    }
    groupedTrades.get(key)!.push(trade);
  });

  const sectors: SectorBacktestResult[] = [];
  const priceSourceAvailable =
    config.useTradierPrices !== false && Boolean(process.env.TRADIER_API_KEY);

  for (const [sectorKey, trades] of groupedTrades.entries()) {
    const computation = computeSectorTimeline(trades, config);
    let mappedSymbol = sectorTickerMap[sectorKey] || sectorTickerMap[sectorKey.replace(/-SECTOR$/, '')];
    if (mappedSymbol === undefined && sectorKey.includes('-')) {
      const [prefix] = sectorKey.split('-');
      mappedSymbol = sectorTickerMap[prefix];
    }

    let priceSource: 'tradier' | 'unavailable' = 'unavailable';
    let simulation: TradeSimulationResult | null = null;

    if (priceSourceAvailable && mappedSymbol) {
      try {
        const bars = await getHistoricalTimesales(mappedSymbol, date, intervalMinutes);
        const { pricePoints, priceChangePct } = attachPricesToTimeline(computation.timeline, bars, date);
        computation.summary.priceChangePct = priceChangePct;
        if (pricePoints.length > 0) {
          priceSource = 'tradier';
        } else {
          notes.push(`No Tradier timesales matched the flow timeline for ${mappedSymbol} on ${date}.`);
        }
      } catch (error: any) {
        notes.push(`Failed to fetch Tradier prices for ${mappedSymbol}: ${error?.message || 'Unknown error'}.`);
      }
    } else if (mappedSymbol && !priceSourceAvailable) {
      notes.push('Tradier API key missing or disabled – skipping price backfill.');
    }

    simulation = simulateTradesFromTimeline(computation.timeline);
    const { metrics } = simulation;
    computation.summary.tradeCount = metrics.tradeCount;
    computation.summary.totalProfit = metrics.totalProfit;
    computation.summary.grossProfit = metrics.grossProfit;
    computation.summary.grossLoss = metrics.grossLoss;
    computation.summary.winRate = metrics.winRate;
    computation.summary.maxDrawdown = metrics.maxDrawdown;
    computation.summary.averageDurationMinutes = metrics.averageDurationMinutes;

    sectors.push({
      sector: sectorKey,
      mappedSymbol,
      timeline: computation.timeline,
      priceSource,
      summary: computation.summary,
      trades: simulation.trades,
      equityCurve: simulation.equityCurve,
    });
  }

  if (sectors.length === 0 && !optionsTrades.length) {
    notes.push('Nothing to backtest – returning empty result.');
  }

  const aggregatedBiasCounts: Record<RegimeBias, number> = {
    bullish: 0,
    bearish: 0,
    balanced: 0,
  };
  let aggregatedCall = 0;
  let aggregatedPut = 0;
  let aggregatedWhales = 0;
  let aggregatedTransitions = 0;
  let aggregatedProfit = 0;
  let aggregatedTradeCount = 0;
  let aggregatedWins = 0;

  sectors.forEach(sector => {
    aggregatedCall += sector.summary.totalCallPremium;
    aggregatedPut += sector.summary.totalPutPremium;
    aggregatedWhales += sector.summary.whaleTrades;
    aggregatedTransitions += sector.summary.regimeTransitions;
    aggregatedBiasCounts.bullish += sector.summary.biasCounts.bullish;
    aggregatedBiasCounts.bearish += sector.summary.biasCounts.bearish;
    aggregatedBiasCounts.balanced += sector.summary.biasCounts.balanced;
    aggregatedProfit += sector.summary.totalProfit;
    aggregatedTradeCount += sector.summary.tradeCount;
    aggregatedWins += sector.trades.filter(trade => trade.profit > 0).length;
  });

  const aggregatedDominantBias =
    (Object.entries(aggregatedBiasCounts).sort((a, b) => b[1] - a[1])[0]?.[0] as RegimeBias) ||
    'balanced';

  const aggregated: AggregatedBacktestSummary = {
    sectors: sectors.length,
    totalCallPremium: aggregatedCall,
    totalPutPremium: aggregatedPut,
    netPremium: aggregatedCall - aggregatedPut,
    whaleTrades: aggregatedWhales,
    dominantBias: aggregatedDominantBias,
    biasCounts: aggregatedBiasCounts,
    regimeTransitions: aggregatedTransitions,
    totalProfit: aggregatedProfit,
    tradeCount: aggregatedTradeCount,
    winRate: aggregatedTradeCount > 0 ? aggregatedWins / aggregatedTradeCount : undefined,
  };

  return {
    date,
    mode,
    resolutionMinutes: intervalMinutes,
    dataSources: {
      flow: snapshot ? `data/${date}.json` : 'unavailable',
      prices: priceSourceAvailable ? 'tradier' : 'unavailable',
    },
    sectors,
    aggregated,
    notes,
  };
}

/**
 * Lists all available cached flow dates from the data directory
 * @returns Array of date strings in YYYY-MM-DD format, sorted descending (newest first)
 */
export async function listCachedFlowDates(): Promise<string[]> {
  const dataDir = path.join(process.cwd(), 'data');

  try {
    const files = await fs.readdir(dataDir);

    // Filter for .json files and extract dates matching YYYY-MM-DD pattern
    const datePattern = /^(\d{4}-\d{2}-\d{2})\.json$/;
    const dates = files
      .filter(file => datePattern.test(file))
      .map(file => {
        const match = file.match(datePattern);
        return match ? match[1] : null;
      })
      .filter((date): date is string => date !== null)
      .sort()
      .reverse(); // Most recent first

    return dates;
  } catch (error) {
    console.error('Error reading data directory:', error);
    // If directory doesn't exist or can't be read, return empty array
    return [];
  }
}
