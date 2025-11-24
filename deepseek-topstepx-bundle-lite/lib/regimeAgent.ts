import axios from 'axios';
import {
  calculateGexForSymbol,
  GexCalculationResult,
  GexMode,
  GexOptionContract,
  GexByStrike,
  GexExpirationSummary,
} from './gexCalculator';
import {
  VolatilityRegimeResponse,
  VolatilityRegimeAnalysis,
  RegimeStage1Result,
  RegimeStage1Metrics,
  RegimeStage1Thresholds,
  RegimeWhaleTrade,
  RegimeStage2Summary,
  RegimeStage3Profile,
  RegimeTradeSignal,
  RegimeGexLevel,
  RegimeWallDetail,
  LiquidityTier,
  GammaRegime,
  VolatilityStats,
  RegimeExpirationContribution,
  RegimeTradeLifecycle,
} from '@/types';
import {
  getVolatilityStats,
  getWhaleFlowAlerts,
  WhaleFlowAlert,
} from './unusualwhales';
import { getCached, setCache } from './dataCache';
import { manageLifecycle, loadAllActiveTrades } from './regimeLifecycle';

const TRADIER_API_KEY = process.env.TRADIER_API_KEY || '';
const TRADIER_BASE_URL = process.env.TRADIER_BASE_URL || 'https://sandbox.tradier.com/v1';

const tradierClient = axios.create({
  baseURL: TRADIER_BASE_URL,
  headers: {
    Authorization: `Bearer ${TRADIER_API_KEY}`,
    Accept: 'application/json',
  },
});

const DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA', 'MSFT'];

export interface RegimeAgentOptions {
  symbols?: string[];
  mode?: 'scalp' | 'swing' | 'leaps';
  date?: string; // For historical backtesting - fetch data from this date
}

interface QuoteDetails {
  symbol: string;
  last: number;
  marketCap?: number | null;
  averageVolume?: number | null;
  optionsVolume?: number | null;
  sharesOutstanding?: number | null;
}

interface IvHistoryEntry {
  timestamp: number;
  iv: number;
}

interface GexHistoryEntry {
  timestamp: number;
  netGex: number;
}

interface TransitionCacheEntry {
  regime: GammaRegime;
  timestamp: number;
}

const MODE_TO_GEX: Record<'scalp' | 'swing' | 'leaps', GexMode> = {
  scalp: 'intraday',
  swing: 'swing',
  leaps: 'leaps',
};

async function fetchQuote(symbol: string, date?: string): Promise<QuoteDetails> {
  // For historical data, use /markets/history
  if (date) {
    const response = await tradierClient.get('/markets/history', {
      params: {
        symbol,
        start: date,
        end: date,
      },
    });

    const history = response.data?.history?.day;
    if (!history) {
      throw new Error(`No historical data for ${symbol} on ${date}`);
    }

    const bar = Array.isArray(history) ? history[0] : history;

    return {
      symbol,
      last: bar.close,
      marketCap: null,
      averageVolume: null,
      optionsVolume: null,
      timestamp: new Date(date),
    };
  }

  // Current quote
  const response = await tradierClient.get('/markets/quotes', {
    params: { symbols: symbol },
  });

  const quote = response.data?.quotes?.quote;
  if (!quote) {
    throw new Error(`No quote data for ${symbol}`);
  }

  const parseNumeric = (value: any): number | null => {
    const parsed = parseFloat(value || 0);
    return Number.isFinite(parsed) && parsed !== 0 ? parsed : null;
  };

  return {
    symbol,
    last: parseNumeric(quote.last || quote.close || quote.prevclose) || 0,
    marketCap: parseNumeric(quote.market_cap || quote.marketcap || quote.marketCap),
    averageVolume: parseNumeric(quote.average_volume || quote.avg_volume || quote.avg_daily_volume),
    optionsVolume: parseNumeric(quote.options_volume || quote.option_volume),
    sharesOutstanding: parseNumeric(quote.shares || quote.shares_outstanding || quote.float),
  };
}

function determineLiquidityTier(
  quote: QuoteDetails,
  totalOptionsVolume: number,
): LiquidityTier {
  const isLargeCap =
    (quote.marketCap && quote.marketCap > 10_000_000_000) ||
    (quote.averageVolume && quote.averageVolume > 10_000_000);
  const hasDeepOptions = totalOptionsVolume > 50_000;

  return isLargeCap && hasDeepOptions ? 'large' : 'mid_small';
}

function computeStage1Thresholds(tier: LiquidityTier): RegimeStage1Thresholds {
  if (tier === 'large') {
    return {
      ivRank: 35, // IV Rank as percentage (0-100), prefer low IVR
      volumeToOi: 1.5,
      whaleContracts: 200,
      whalePremium: 500_000,
    };
  }

  return {
    ivRank: 25, // IV Rank as percentage (0-100), prefer low IVR
    volumeToOi: 2.0,
    whaleContracts: 100,
    whalePremium: 250_000,
  };
}

function updateIvHistory(symbol: string, iv: number | null | undefined): IvHistoryEntry[] {
  if (iv === null || iv === undefined || !Number.isFinite(iv)) {
    const existing = getCached<IvHistoryEntry[]>(`regime_iv_history_${symbol}`)?.data;
    return existing ? [...existing] : [];
  }

  const cacheKey = `regime_iv_history_${symbol}`;
  const cached = getCached<IvHistoryEntry[]>(cacheKey);
  const history = cached?.data ? [...cached.data] : [];

  history.push({
    iv,
    timestamp: Date.now(),
  });

  const cutoff = Date.now() - 2 * 60 * 60 * 1000; // retain last 2 hours
  const pruned = history.filter(entry => entry.timestamp >= cutoff);

  setCache(cacheKey, pruned, 'regime-agent');

  return pruned;
}

function getIvDelta(history: IvHistoryEntry[], minutes: number): number | null {
  if (!history || history.length < 2) {
    return null;
  }

  const now = Date.now();
  const cutoff = now - minutes * 60 * 1000;
  const recent = history[history.length - 1];

  if (recent.timestamp < cutoff) {
    return null;
  }

  const past =
    [...history]
      .reverse()
      .find(entry => entry.timestamp <= cutoff) ||
    history[0];

  if (!past || past === recent) {
    return null;
  }

  return recent.iv - past.iv;
}

function adaptAlertToRegimeTrade(alert: WhaleFlowAlert): RegimeWhaleTrade {
  return {
    optionType: alert.optionType,
    direction: alert.direction === 'bearish' ? 'bearish' : 'bullish',
    contracts: alert.contracts,
    premium: alert.premium,
    strike: alert.strike,
    expiration: alert.expiration,
    midpointPrice: alert.price,
    timestamp: alert.timestamp,
  };
}

function detectWhaleTrades(
  symbol: string,
  alerts: WhaleFlowAlert[],
  thresholds: RegimeStage1Thresholds,
  fallbackContracts: GexOptionContract[],
): RegimeWhaleTrade[] {
  const symbolAlerts = alerts.filter(alert => alert.ticker === symbol.toUpperCase());
  const qualifyingAlerts = symbolAlerts.filter(
    alert =>
      alert.contracts >= thresholds.whaleContracts ||
      alert.premium >= thresholds.whalePremium,
  );

  const mappedAlerts = qualifyingAlerts
    .map(alert => adaptAlertToRegimeTrade(alert))
    .sort((a, b) => b.premium - a.premium);

  if (mappedAlerts.length > 0) {
    return mappedAlerts.slice(0, 5);
  }

  const fallback: RegimeWhaleTrade[] = [];

  fallbackContracts.forEach(contract => {
    const midpoint =
      contract.bid > 0 && contract.ask > 0 ? (contract.bid + contract.ask) / 2 : contract.last;

    const premium = midpoint * 100 * contract.volume;

    if (
      contract.volume >= thresholds.whaleContracts ||
      premium >= thresholds.whalePremium
    ) {
      fallback.push({
        optionType: contract.type,
        direction: contract.type === 'call' ? 'bullish' : 'bearish',
        contracts: contract.volume,
        premium,
        strike: contract.strike,
        expiration: contract.expiration,
        midpointPrice: midpoint,
      });
    }
  });

  fallback.sort((a, b) => b.premium - a.premium);
  return fallback.slice(0, 5);
}

function buildStage1Result(
  symbol: string,
  tier: LiquidityTier,
  metrics: RegimeStage1Metrics,
  thresholds: RegimeStage1Thresholds,
  whaleTrades: RegimeWhaleTrade[],
): RegimeStage1Result {
  const failedCriteria: string[] = [];

  if (
    metrics.ivRank === undefined ||
    metrics.ivRank === null ||
    metrics.ivRank > thresholds.ivRank
  ) {
    failedCriteria.push('IV Rank');
  }

  if (
    metrics.volumeToOi === undefined ||
    metrics.volumeToOi === null ||
    metrics.volumeToOi < thresholds.volumeToOi
  ) {
    failedCriteria.push('Volume/OI');
  }

  // Note: IV rising check (ivDelta) is in Stage 2, not Stage 1
  // Stage 1 is just universe filtering from whale flow

  if (whaleTrades.length === 0) {
    failedCriteria.push('Whale Flow');
  }

  const notes: string[] = [];
  if (metrics.ivRank !== null && metrics.ivRank !== undefined) {
    notes.push(`IVR ${metrics.ivRank.toFixed(2)} vs target ${thresholds.ivRank}`);
  }
  if (metrics.volumeToOi !== null && metrics.volumeToOi !== undefined) {
    notes.push(`Volume/OI ${metrics.volumeToOi.toFixed(2)} vs target ${thresholds.volumeToOi}`);
  }
  if (metrics.ivDelta15m !== null && metrics.ivDelta15m !== undefined) {
    notes.push(`IVΔ15 ≈ ${metrics.ivDelta15m.toFixed(4)}`);
  }
  if (metrics.ivDelta30m !== null && metrics.ivDelta30m !== undefined) {
    notes.push(`IVΔ30 ≈ ${metrics.ivDelta30m.toFixed(4)}`);
  }
  if (whaleTrades.length > 0) {
    const primary = whaleTrades[0];
    notes.push(
      `Largest whale: ${primary.direction.toUpperCase()} ${primary.contracts}c @ ~$${(
        primary.premium / 1000
      ).toFixed(1)}k premium`,
    );
  }

  return {
    symbol,
    tier,
    passes: failedCriteria.length === 0,
    metrics,
    thresholds,
    whaleTrades,
    failedCriteria,
    notes,
  };
}

function computeGammaFlipLevel(gexData: GexByStrike[]): number | undefined {
  for (let i = 1; i < gexData.length; i += 1) {
    const prev = gexData[i - 1];
    const curr = gexData[i];
    if (prev.netGex === 0) return prev.strike;
    if (curr.netGex === 0) return curr.strike;

    if ((prev.netGex < 0 && curr.netGex > 0) || (prev.netGex > 0 && curr.netGex < 0)) {
      return (prev.strike + curr.strike) / 2;
    }
  }
  return undefined;
}

function updateGexHistory(symbol: string, mode: 'scalp' | 'swing' | 'leaps', netGex: number): GexHistoryEntry[] {
  const cacheKey = `regime_gex_history_${symbol}_${mode}`;
  const cached = getCached<GexHistoryEntry[]>(cacheKey);
  const history = cached?.data ? [...cached.data] : [];

  history.push({
    timestamp: Date.now(),
    netGex,
  });

  const cutoff = Date.now() - 90 * 60 * 1000; // retain last 90 minutes
  const pruned = history.filter(entry => entry.timestamp >= cutoff);

  setCache(cacheKey, pruned, 'regime-agent');

  return pruned;
}

function computeGexSlopeFromHistory(history: GexHistoryEntry[]): 'rising' | 'falling' | 'flat' {
  if (!history || history.length < 2) {
    return 'flat';
  }

  const latest = history[history.length - 1];
  const lookbacks = [15, 30]; // minutes

  let classified: 'rising' | 'falling' | 'flat' = 'flat';

  for (const minutes of lookbacks) {
    const cutoff = latest.timestamp - minutes * 60 * 1000;
    const comparison = [...history].reverse().find(entry => entry.timestamp <= cutoff);

    if (!comparison) continue;

    const diff = latest.netGex - comparison.netGex;
    const base = comparison.netGex === 0 ? Math.abs(latest.netGex) : Math.abs(comparison.netGex);
    const pct = base === 0 ? 0 : diff / base;

    if (pct > 0.1 || diff > 5_000_000) {
      classified = 'rising';
      break;
    }

    if (pct < -0.1 || diff < -5_000_000) {
      classified = 'falling';
      break;
    }
  }

  return classified;
}

function classifyGexLevels(gexData: GexByStrike[]): RegimeGexLevel[] {
  return gexData.map(level => {
    let classification: 'call_wall' | 'put_zone' | 'neutral' = 'neutral';
    if (level.netGex > 0) {
      classification = 'call_wall';
    } else if (level.netGex < 0) {
      classification = 'put_zone';
    }

    return {
      strike: level.strike,
      netGex: level.netGex,
      callGex: level.callGex,
      putGex: level.putGex,
      netGexPerDollar: level.netGexPerDollar,
      callGexPerDollar: level.callGexPerDollar,
      putGexPerDollar: level.putGexPerDollar,
      oi: level.oi,
      volume: level.volume,
      classification,
    };
  });
}

function selectKeyLevels(
  levels: RegimeGexLevel[],
  price: number,
  count: number,
  type: 'call_wall' | 'put_zone',
): RegimeWallDetail[] {
  const filtered = levels.filter(level => level.classification === type);
  if (filtered.length === 0) {
    return [];
  }

  const strengths = filtered.map(level => Math.abs(level.netGex));
  const mean = strengths.reduce((sum, value) => sum + value, 0) / strengths.length;
  const variance =
    strengths.reduce((sum, value) => sum + (value - mean) * (value - mean), 0) / strengths.length;
  const stdDev = Math.sqrt(variance);

  return filtered
    .map(level => {
      const strength = Math.abs(level.netGex);
      const distancePct = price > 0 ? Math.abs(price - level.strike) / price : 0;

      return {
        ...level,
        strength,
        distancePct,
        isNearPrice: distancePct <= 0.01,
        zScore: stdDev > 0 ? (strength - mean) / stdDev : 0,
      };
    })
    .sort((a, b) => {
      const zDiff = b.zScore - a.zScore;
      if (zDiff !== 0) return zDiff;
      const strengthDiff = b.strength - a.strength;
      if (strengthDiff !== 0) return strengthDiff;
      return a.distancePct - b.distancePct;
    })
    .slice(0, count);
}

function buildDominantExpirations(summaries: GexExpirationSummary[]): RegimeExpirationContribution[] {
  return summaries
    .map(summary => ({
      expiration: summary.expiration,
      dte: summary.dte,
      netGex: summary.netGex,
      totalCallGex: summary.totalCallGex,
      totalPutGex: summary.totalPutGex,
    }))
    .sort((a, b) => Math.abs(b.netGex) - Math.abs(a.netGex))
    .slice(0, 3);
}

function determineRangeOutlook(
  regime: GammaRegime,
  callWalls: RegimeWallDetail[],
  putZones: RegimeWallDetail[],
  price: number,
  gammaFlipLevel?: number,
): RegimeStage3Profile['rangeOutlook'] {
  const nearestCall = [...callWalls].sort((a, b) => a.distancePct - b.distancePct)[0];
  const nearestPut = [...putZones].sort((a, b) => a.distancePct - b.distancePct)[0];

  if (regime === 'pinning') {
    if (nearestCall && nearestPut && nearestCall.distancePct < 0.015 && nearestPut.distancePct < 0.015) {
      return 'range_bound';
    }
    if (nearestCall && (!nearestPut || nearestCall.distancePct < nearestPut.distancePct)) {
      return 'upside_risk';
    }
    if (nearestPut) {
      return 'downside_risk';
    }
    return 'range_bound';
  }

  if (nearestCall && nearestCall.isNearPrice) {
    return 'breakout_watch';
  }

  if (nearestPut && nearestPut.isNearPrice) {
    return 'downside_risk';
  }

  if (gammaFlipLevel) {
    const distancePct = Math.abs(price - gammaFlipLevel) / price;
    if (distancePct <= 0.012) {
      return 'breakout_watch';
    }
  }

  return 'range_bound';
}

function deriveSlopeInfo(history: GexHistoryEntry[]): {
  slope: 'rising' | 'falling' | 'flat';
  strength: 'weak' | 'moderate' | 'strong';
  delta: number;
} {
  const slope = computeGexSlopeFromHistory(history);

  if (!history || history.length < 2) {
    return { slope: 'flat', strength: 'weak', delta: 0 };
  }

  const latest = history[history.length - 1];
  const window = history.slice(-Math.min(history.length, 10));
  const avg = window.reduce((sum, entry) => sum + entry.netGex, 0) / (window.length || 1);
  const delta = latest.netGex - avg;

  const cutoff = latest.timestamp - 30 * 60 * 1000;
  const reference = [...history].reverse().find(entry => entry.timestamp <= cutoff) || history[0];
  const diff = Math.abs(latest.netGex - reference.netGex);

  let strength: 'weak' | 'moderate' | 'strong' = 'weak';
  if (diff >= 15_000_000) {
    strength = 'strong';
  } else if (diff >= 5_000_000) {
    strength = 'moderate';
  }

  return { slope, strength, delta };
}

function buildTrendNarrative(
  regime: GammaRegime,
  slope: 'rising' | 'falling' | 'flat',
  strength: 'weak' | 'moderate' | 'strong',
  delta: number,
  dominantExpirations: RegimeExpirationContribution[],
): string {
  const direction =
    slope === 'rising'
      ? 'Net gamma is climbing'
      : slope === 'falling'
        ? 'Net gamma is deteriorating'
        : 'Net gamma is stable';

  const strengthText =
    strength === 'strong'
      ? 'with strong momentum'
      : strength === 'moderate'
        ? 'with moderate momentum'
        : 'with minimal momentum';

  const deltaText =
    Math.abs(delta) >= 5_000_000
      ? ` (≈$${(delta / 1_000_000).toFixed(1)}M shift vs recent average)`
      : '';

  const regimeText =
    regime === 'expansion'
      ? 'Dealers remain short gamma, keeping breakout risk elevated.'
      : 'Dealers remain long gamma, favouring mean-reversion.';

  const expiryFocus = dominantExpirations
    .map(exp => `${exp.expiration} (${exp.dte} DTE)`)
    .join(', ');

  const expiryText =
    dominantExpirations.length > 0 ? ` Key expiry focus: ${expiryFocus}.` : '';

  return `${direction} ${strengthText}${deltaText}. ${regimeText}${expiryText}`;
}

function determineRegimeTransition(
  symbol: string,
  mode: 'scalp' | 'swing' | 'leaps',
  currentRegime: GammaRegime,
): 'stable' | 'flip_to_expansion' | 'flip_to_pinning' {
  const cacheKey = `regime_transition_${symbol}_${mode}`;
  const cached = getCached<TransitionCacheEntry>(cacheKey);

  let transition: 'stable' | 'flip_to_expansion' | 'flip_to_pinning' = 'stable';

  if (cached?.data) {
    if (cached.data.regime !== currentRegime) {
      transition = currentRegime === 'expansion' ? 'flip_to_expansion' : 'flip_to_pinning';
    }
  }

  setCache(cacheKey, { regime: currentRegime, timestamp: Date.now() }, 'regime-agent');

  return transition;
}

function buildStage2Summary(
  symbol: string,
  price: number,
  mode: 'scalp' | 'swing' | 'leaps',
  gex: GexCalculationResult,
): RegimeStage2Summary {
  const netGex = gex.summary.netGex;
  const regime: GammaRegime = netGex <= 0 ? 'expansion' : 'pinning';
  const gammaFlipLevel = computeGammaFlipLevel(gex.gexData);
  const gexHistory = updateGexHistory(symbol, mode, netGex);
  const slopeInfo = deriveSlopeInfo(gexHistory);
  const transition = determineRegimeTransition(symbol, mode, regime);
  const dominantExpirations = buildDominantExpirations(gex.expirationSummaries || []);
  const trendNarrative = buildTrendNarrative(regime, slopeInfo.slope, slopeInfo.strength, slopeInfo.delta, dominantExpirations);

  return {
    symbol,
    price,
    netGex,
    netGexPerDollar: gex.summary.netGexPerDollar,
    totalCallGex: gex.summary.totalCallGex,
    totalPutGex: gex.summary.totalPutGex,
    regime,
    gammaWall: gex.gammaWall,
    gammaFlipLevel,
    slope: slopeInfo.slope,
    slopeStrength: slopeInfo.strength,
    gammaFlipDistance: gammaFlipLevel !== undefined ? price - gammaFlipLevel : undefined,
    dominantExpirations,
    trendNarrative,
    regimeTransition: transition,
    recentSlopeDelta: slopeInfo.delta,
    expirations: gex.expirationDetails.map(({ date, dte }) => ({ date, dte })),
    mode,
  };
}

function evaluatePriceInteraction(price: number, callWalls: RegimeWallDetail[], putZones: RegimeWallDetail[]): RegimeStage3Profile['priceInteraction'] {
  const nearestCall = [...callWalls].sort((a, b) => a.distancePct - b.distancePct)[0];
  const nearestPut = [...putZones].sort((a, b) => a.distancePct - b.distancePct)[0];

  if (nearestCall && nearestCall.distancePct < 0.01) {
    return price >= nearestCall.strike ? 'above_call_wall' : 'inside_range';
  }

  if (nearestPut && nearestPut.distancePct < 0.01) {
    return price <= nearestPut.strike ? 'below_put_wall' : 'inside_range';
  }

  return 'inside_range';
}

function buildStage3Profile(
  symbol: string,
  price: number,
  gex: GexCalculationResult,
  stage2: RegimeStage2Summary,
): RegimeStage3Profile {
  const levels = classifyGexLevels(gex.gexData);
  const callWalls = selectKeyLevels(levels, price, 3, 'call_wall');
  const putZones = selectKeyLevels(levels, price, 3, 'put_zone');
  const gammaFlipLevel = computeGammaFlipLevel(gex.gexData);
  const rangeOutlook = determineRangeOutlook(stage2.regime, callWalls, putZones, price, gammaFlipLevel);

  return {
    symbol,
    gammaWall: gex.gammaWall,
    callWalls,
    putZones,
    profile: levels,
    gammaFlipLevel,
    priceInteraction: evaluatePriceInteraction(price, callWalls, putZones),
    rangeOutlook,
  };
}

function determinePositionSize(ivRank?: number | null): 'full' | 'half' {
  if (ivRank === null || ivRank === undefined) {
    return 'half';
  }
  if (ivRank < 0.35) {
    return 'full';
  }
  if (ivRank > 0.6) {
    return 'half';
  }
  return 'full';
}

function buildTradeSignals(
  symbol: string,
  price: number,
  stage1: RegimeStage1Result,
  stage2: RegimeStage2Summary,
  stage3: RegimeStage3Profile,
): RegimeTradeSignal[] {
  // TEMPORARY: Skip Stage 1 filtering - generate signals purely based on GEX levels
  // if (!stage1.passes) {
  //   return [];
  // }

  const signals: RegimeTradeSignal[] = [];
  const ivDelta = stage1.metrics.ivDelta15m ?? 0;
  // TEMPORARY: Skip IV delta check for backtesting
  // if (ivDelta <= 0) {
  //   return [];
  // }

  const size = determinePositionSize(stage1.metrics.ivRank);
  const timeframeByMode: Record<'scalp' | 'swing' | 'leaps', number> = {
    scalp: 30,
    swing: 90,
    leaps: 240,
  };
  const timeframe = timeframeByMode[stage2.mode];
  const whaleCalls = stage1.whaleTrades.filter(trade => trade.optionType === 'call');
  const whalePuts = stage1.whaleTrades.filter(trade => trade.optionType === 'put');

  // Calculate net whale flow: positive = bullish, negative = bearish
  const totalCallPremium = whaleCalls.reduce((sum, t) => sum + t.premium, 0);
  const totalPutPremium = whalePuts.reduce((sum, t) => sum + t.premium, 0);
  const netWhaleFlow = totalCallPremium - totalPutPremium;
  const whaleFlowBias = netWhaleFlow > 0 ? 'bullish' : netWhaleFlow < 0 ? 'bearish' : 'neutral';

  // Stop loss percentages by mode
  const stopPctByMode: Record<'scalp' | 'swing' | 'leaps', number> = {
    scalp: 0.01,   // 1% for scalp
    swing: 0.015,  // 1.5% for swing
    leaps: 0.02,   // 2% for leaps
  };
  const stopPct = stopPctByMode[stage2.mode];

  // Target percentages by mode
  const targetPctByMode: Record<'scalp' | 'swing' | 'leaps', number> = {
    scalp: 0.003,   // 0.3% for scalp first target (tighter)
    swing: 0.015,   // 1.5% for swing first target
    leaps: 0.03,    // 3% for leaps first target
  };
  const targetPct = targetPctByMode[stage2.mode];

  // Minimum whale flow strength filter
  const minWhaleFlowStrength = 500_000; // $500K minimum net flow (relaxed from $1M)
  if (Math.abs(netWhaleFlow) < minWhaleFlowStrength) {
    return signals; // Whale flow too weak, skip
  }

  // Maximum distance from price for trigger levels (by mode)
  const maxTriggerDistanceByMode: Record<'scalp' | 'swing' | 'leaps', number> = {
    scalp: 0.005,  // 0.5% max for scalp (very tight - must be at the wall NOW)
    swing: 0.03,   // 3% max for swing
    leaps: 0.08,   // 8% max for leaps
  };
  const maxTriggerDistance = maxTriggerDistanceByMode[stage2.mode];

  if (stage2.regime === 'expansion') {
    const breakoutWall = stage3.callWalls[0] || stage3.profile.find(level => level.classification === 'call_wall');
    if (breakoutWall) {
      // For expansion breakout LONG: wall must be ABOVE current price (we're waiting to break higher)
      if (breakoutWall.strike <= price) {
        return signals; // Price already above breakout level, skip
      }

      // Check if breakout wall is close enough to current price
      const distanceToWall = Math.abs(breakoutWall.strike - price) / price;
      if (distanceToWall > maxTriggerDistance) {
        return signals; // Wall too far from current price, skip
      }

      // Only generate LONG signal if whale flow is bullish or neutral
      if (whaleFlowBias === 'bearish') {
        return signals; // Skip - whale flow contradicts regime
      }

      // Stop: Below trigger level by stop %
      const stopBase = breakoutWall.strike * (1 - stopPct);
      // Target: Use percentage-based targets from trigger level
      const firstTarget = breakoutWall.strike * (1 + targetPct);
      const secondaryTarget = breakoutWall.strike * (1 + targetPct * 2);

      const rationale = [
        `Negative net GEX ($${(stage2.netGex / 1_000_000).toFixed(1)}M), dealers short gamma`,
        `Price at $${price.toFixed(2)}, breakout trigger at call wall $${breakoutWall.strike.toFixed(2)}`,
        `Whale flow: ${whaleFlowBias.toUpperCase()} (net $${(netWhaleFlow / 1_000_000).toFixed(1)}M)`,
      ];
      if (whaleCalls.length > 0) {
        rationale.push(`Top call sweep: ${whaleCalls[0].contracts} contracts @ $${whaleCalls[0].strike} (~$${(whaleCalls[0].premium / 1000).toFixed(0)}k)`);
      }
      const risk = breakoutWall.strike - stopBase;

      signals.push({
        id: `${symbol}-expansion-long`,
        symbol,
        action: 'buy',
        direction: 'long',
        strategy: stage2.mode,
        regime: stage2.regime,
        positionSize: size,
        entry: {
          price,
          triggerLevel: breakoutWall.strike,
          triggerType: 'breakout',
        },
        stopLoss: stopBase,
        firstTarget,
        secondaryTarget,
        rationale,
        whaleConfirmation: whaleCalls[0] || null,
        riskPerShare: risk,
        timeframeMinutes: timeframe,
      });
    }
  } else if (stage2.regime === 'pinning') {
    const upperWall = stage3.callWalls[0];
    const lowerZone = stage3.putZones[0];
    if (upperWall && lowerZone) {
      const distanceToUpper = Math.abs(price - upperWall.strike) / price;
      const distanceToLower = Math.abs(price - lowerZone.strike) / price;

      // Only trade if price is near a wall (use mode-based distance filter)
      if (Math.min(distanceToUpper, distanceToLower) > maxTriggerDistance) {
        return signals; // Price not near any wall, skip
      }

      if (distanceToUpper < distanceToLower) {
        // Price near upper wall - consider SHORT fade if whale flow supports
        // For SHORT fade: wall must be ABOVE current price (we're waiting for price to test resistance)
        if (upperWall.strike <= price) {
          return signals; // Price already above upper wall, skip
        }

        if (whaleFlowBias === 'bullish') {
          return signals; // Skip - whale flow contradicts fade
        }

        const stop = upperWall.strike * (1 + stopPct);
        const target = upperWall.strike * (1 - targetPct); // Target from trigger towards lower zone
        const secondaryTarget = lowerZone.strike * (1 + targetPct * 2);

        const rationale = [
          `Positive net GEX ($${(stage2.netGex / 1_000_000).toFixed(1)}M) indicates range`,
          `Price at $${price.toFixed(2)} near call wall $${upperWall.strike.toFixed(2)} (dealer fade)`,
          `Whale flow: ${whaleFlowBias.toUpperCase()} (net $${(netWhaleFlow / 1_000_000).toFixed(1)}M)`,
        ];
        if (whalePuts.length > 0) {
          rationale.push(`Put hedge: ${whalePuts[0].contracts} contracts @ $${whalePuts[0].strike}`);
        }
        const risk = Math.abs(stop - upperWall.strike);

        signals.push({
          id: `${symbol}-pin-short`,
          symbol,
          action: 'sell',
          direction: 'short',
          strategy: stage2.mode,
          regime: stage2.regime,
          positionSize: size,
          entry: {
            price,
            triggerLevel: upperWall.strike,
            triggerType: 'fade',
          },
          stopLoss: stop,
          firstTarget: target,
          secondaryTarget,
          rationale,
          whaleConfirmation: whalePuts[0] || whaleCalls[0] || null,
          riskPerShare: risk,
          timeframeMinutes: timeframe,
        });
      } else {
        // Price near lower zone - consider LONG reversion if whale flow supports
        // For LONG reversion: wall must be BELOW current price (we're waiting for price to dip to support)
        if (lowerZone.strike >= price) {
          return signals; // Price already below support level, skip
        }

        if (whaleFlowBias === 'bearish') {
          return signals; // Skip - whale flow contradicts reversion
        }

        const stop = lowerZone.strike * (1 - stopPct);
        const target = lowerZone.strike * (1 + targetPct); // Target from trigger upwards
        const secondaryTarget = lowerZone.strike * (1 + targetPct * 2);

        const rationale = [
          `Positive net GEX ($${(stage2.netGex / 1_000_000).toFixed(1)}M) provides support`,
          `Price at $${price.toFixed(2)} near put wall $${lowerZone.strike.toFixed(2)} (reversion)`,
          `Whale flow: ${whaleFlowBias.toUpperCase()} (net $${(netWhaleFlow / 1_000_000).toFixed(1)}M)`,
        ];
        if (whaleCalls.length > 0) {
          rationale.push(`Call support: ${whaleCalls[0].contracts} contracts @ $${whaleCalls[0].strike}`);
        }
        const risk = Math.abs(lowerZone.strike - stop);

        signals.push({
          id: `${symbol}-pin-long`,
          symbol,
          action: 'buy',
          direction: 'long',
          strategy: stage2.mode,
          regime: stage2.regime,
          positionSize: size,
          entry: {
            price,
            triggerLevel: lowerZone.strike,
            triggerType: 'range-reversion',
          },
          stopLoss: stop,
          firstTarget: target,
          secondaryTarget,
          rationale,
          whaleConfirmation: whaleCalls[0] || null,
          riskPerShare: risk,
          timeframeMinutes: timeframe,
        });
      }
    }
  }

  return signals;
}

function aggregateStage1Metrics(
  quote: QuoteDetails,
  stats: VolatilityStats | undefined,
  gex: GexCalculationResult,
  ivHistory: IvHistoryEntry[],
): RegimeStage1Metrics {
  const totalVolume = gex.gexData.reduce((sum, level) => sum + level.volume, 0);
  const totalOi = gex.gexData.reduce((sum, level) => sum + level.oi, 0);
  const volumeToOi = totalOi > 0 ? totalVolume / totalOi : null;

  const ivRank = stats?.iv_rank ?? null;
  const ivDelta15 = getIvDelta(ivHistory, 15);
  const ivDelta30 = getIvDelta(ivHistory, 30);

  return {
    marketCap: quote.marketCap,
    averageVolume: quote.averageVolume,
    optionsVolume: totalVolume,
    openInterest: totalOi,
    ivRank,
    ivDelta15m: ivDelta15,
    ivDelta30m: ivDelta30,
    volumeToOi,
  };
}

function buildAnalysisRecord(
  symbol: string,
  price: number,
  mode: 'scalp' | 'swing' | 'leaps',
  stage1: RegimeStage1Result,
  stage2: RegimeStage2Summary,
  stage3: RegimeStage3Profile,
  tradeSignals: RegimeTradeSignal[],
  activeTrades: RegimeTradeLifecycle[],
): VolatilityRegimeAnalysis {
  return {
    symbol,
    price,
    timestamp: new Date().toISOString(),
    mode,
    stage1,
    stage2,
    stage3,
    tradeSignals,
    activeTrades,
  };
}

export async function analyzeVolatilityRegime(
  options: RegimeAgentOptions = {},
): Promise<VolatilityRegimeResponse> {
  const mode: 'scalp' | 'swing' | 'leaps' = options.mode || 'scalp';
  const gexMode = MODE_TO_GEX[mode];
  const date = options.date; // Optional date for historical backtesting
  let symbols = options.symbols || [];
  const whaleMap = new Map<string, WhaleFlowAlert[]>();

  // If symbols are not provided, fetch them from whale flow
  if (symbols.length === 0) {
    // Step 1: Fetch ALL whale flow alerts (no symbol filter)
    // This gives us the universe of tickers that whales traded today (or historical date)
    const whaleAlerts = await getWhaleFlowAlerts({
      symbols: [], // Empty = fetch all
      lookbackMinutes: 390, // Full trading day
      limit: 500,
      date, // Pass date for historical data
    })
      .then(alerts => {
        console.log('✅ Whale alerts received:', alerts.length);
        return alerts;
      })
      .catch((err) => {
        console.error('❌ getWhaleFlowAlerts error:', err.message || err);
        return [] as WhaleFlowAlert[];
      });

    // Step 2: Extract unique tickers from whale alerts - this is our Stage 1 universe
    whaleAlerts.forEach(alert => {
      const key = alert.ticker.toUpperCase();
      const existing = whaleMap.get(key) || [];
      existing.push(alert);
      whaleMap.set(key, existing);
    });

    symbols = Array.from(whaleMap.keys());
    console.log(`📋 Stage 1 Universe: ${symbols.length} tickers from whale flow`);
  } else {
    console.log(`📋 Using provided universe of ${symbols.length} tickers.`);
  }

  if (symbols.length === 0) {
    console.log('⚠️ No whale flow alerts found - empty universe');
    return {
      symbols: [],
      mode,
      universe: [],
      analyses: [],
    };
  }

  // Step 3: Fetch volatility stats for whale tickers
  console.log(`Fetching volatility stats for ${date ? date : 'today'}...`);
  const volatilityStats = await getVolatilityStats(date, symbols)
    .then(stats => {
      console.log('✅ Volatility stats received:', stats.length);
      return stats;
    })
    .catch((err) => {
      console.error('❌ getVolatilityStats error:', err.message || err);
      console.error('Full error details:', err.response?.data || err);
      return [] as VolatilityStats[];
    });

  const statsMap = new Map(
    volatilityStats.map(item => [item.ticker.toUpperCase(), item]),
  );

  const analyses: VolatilityRegimeAnalysis[] = [];
  const universeResults: RegimeStage1Result[] = [];

  // Step 4: Process each whale ticker through Stage 1 filtering
  for (const symbol of symbols) {
    try {
      const [quote, gex] = await Promise.all([
        fetchQuote(symbol, date),
        calculateGexForSymbol(symbol, gexMode, date),
      ]);

      const stats = statsMap.get(symbol.toUpperCase());
      const ivHistory = updateIvHistory(symbol, stats?.iv ?? null);
      const metrics = aggregateStage1Metrics(quote, stats, gex, ivHistory);
      const tier = determineLiquidityTier(
        quote,
        Math.max(metrics.optionsVolume || 0, quote.optionsVolume || 0),
      );
      const thresholds = computeStage1Thresholds(tier);
      const whaleTrades = detectWhaleTrades(
        symbol,
        whaleMap.get(symbol.toUpperCase()) || [],
        thresholds,
        gex.contracts,
      );
      const stage1 = buildStage1Result(symbol, tier, metrics, thresholds, whaleTrades);

      universeResults.push(stage1);

      const stage2 = buildStage2Summary(symbol, quote.last, mode, gex);
      const stage3 = buildStage3Profile(symbol, quote.last, gex, stage2);
      const tradeSignals = buildTradeSignals(symbol, quote.last, stage1, stage2, stage3);
      const lifecycleTrades = manageLifecycle({
        mode,
        symbol,
        price: quote.last,
        stage2,
        stage3,
        signals: tradeSignals,
      });

      analyses.push(
        buildAnalysisRecord(
          symbol,
          quote.last,
          mode,
          stage1,
          stage2,
          stage3,
          tradeSignals,
          lifecycleTrades,
        ),
      );
    } catch (error: any) {
      console.error(`Failed to process ${symbol}:`, error.message || error);
    }
  }

  return {
    symbols,
    mode,
    universe: universeResults,
    analyses,
    activeTrades: loadAllActiveTrades(mode),
    generatedAt: new Date().toISOString(),
  };
}
