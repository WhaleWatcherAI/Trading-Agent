/**
 * Fabio Agent + OpenAI Integration
 * Bridges existing Fabio calculations with OpenAI decision making
 * and execution system with self-learning database
 */

import {
  AbsorptionSignal,
  FlowSignals,
  analyzeFuturesMarket,
  FuturesMarketData,
  OpenAITradingDecision,
  TradeLegProfile,
  PullbackProfile,
  WatchZoneProfile,
} from './openaiTradingAgent';
import { buildMlFeatureSnapshot } from './mlFeatureExtractor';
import { appendFileSync, existsSync, mkdirSync } from 'fs';
import path from 'path';
import { ExecutionManager } from './executionManager';
import { tradingDB } from './tradingDatabase';
import { MarketState } from './fabioPlaybook';
import {
  POCCrossTracker,
  MarketStatsCalculator,
  PerformanceTracker,
  HistoricalNotesManager,
} from './enhancedFeatures';

// Global enhanced trackers for self-learning system
const pocCrossTracker = new POCCrossTracker();
const marketStatsCalc = new MarketStatsCalculator();
const performanceTracker = new PerformanceTracker();
const notesManager = new HistoricalNotesManager();

const mlSnapshotThrottleMs = 55_000;
const lastMlSnapshotWrite: Record<string, number> = {};

function isSelfLearningEnabled(): boolean {
  const flag = process.env.SELF_LEARNING_DISABLED?.toLowerCase();
  return flag !== 'true' && flag !== '1' && flag !== 'yes';
}

function computeATR(bars: TopstepXFuturesBar[], period: number): number | null {
  if (!bars || bars.length < period + 1) return null;
  const recent = bars.slice(-1 * Math.max(period + 1, 2));
  if (recent.length < 2) return null;
  const trs: number[] = [];
  for (let i = 1; i < recent.length; i += 1) {
    const cur = recent[i];
    const prevClose = recent[i - 1].close;
    const tr = Math.max(
      cur.high - cur.low,
      Math.abs(cur.high - prevClose),
      Math.abs(cur.low - prevClose),
    );
    trs.push(tr);
  }
  if (trs.length === 0) return null;
  const sum = trs.reduce((acc, v) => acc + v, 0);
  return sum / trs.length;
}

// Type imports from live-fabio-agent-playbook
export interface TopstepXFuturesBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface VolumeProfile {
  nodes: any[];
  poc: number;
  vah: number;
  val: number;
  lvns: number[];
}

export interface OrderFlowData {
  bigPrints: Array<{ price: number; size: number; side: 'buy' | 'sell'; timestamp: number }>;
  cvd: number;
  footprintImbalance: { [price: number]: number };
  absorption: { buy: number; sell: number };
  exhaustion: { buy: number; sell: number };
  cvdHistory: Array<{ timestamp: number; cvd: number; delta: number }>;
  volumeAtPrice: { [price: number]: { buy: number; sell: number; timestamp: number } };
}

export interface MarketStructure {
  state: MarketState;
  impulseLegs: any[];
  balanceAreas: any[];
  failedBreakouts: any[];
}

export interface CurrentCvdBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface HigherTimeframeSnapshot {
  timeframe: string;
  candles: Array<{
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
  }>;
}

export interface SessionVolumeProfileSummary {
  sessionStart: string;
  sessionEnd: string;
  poc: number;
  vah: number;
  val: number;
  lvns: number[];
  sessionHigh: number;
  sessionLow: number;
}

/**
 * Build FuturesMarketData from Fabio agent's existing calculations
 * This is the bridge between Fabio's analysis and OpenAI's decision making
 */
export function buildFuturesMarketData(
  symbol: string,
  bars: TopstepXFuturesBar[],
  volumeProfile: VolumeProfile | null,
  orderFlowData: OrderFlowData,
  marketStructure: MarketStructure,
  currentCvdBar: CurrentCvdBar | null,
  accountBalance: number,
  currentPosition: any | null,
  realizedPnL: number = 0,
  higherTimeframes: HigherTimeframeSnapshot[] = [],
  recentVolumeProfiles: SessionVolumeProfileSummary[] = [],
  cvdCandles: CurrentCvdBar[] = []
): FuturesMarketData {
  const selfLearning = isSelfLearningEnabled();
  // Get current price from latest bar
  const currentPrice = bars.length > 0 ? bars[bars.length - 1].close : 0;
  const tickSize = estimateTickSize(orderFlowData, bars);

  // Get last 5 candles (25 minutes for 5-min bars)
  const recentCandles = bars.slice(-20);

  // Calculate CVD trend from history
  let cvdTrend: 'up' | 'down' | 'neutral' = 'neutral';
  if (orderFlowData.cvd > 100) {
    cvdTrend = 'up';
  } else if (orderFlowData.cvd < -100) {
    cvdTrend = 'down';
  }

  // Get buy/sell control from CVD
  const totalCVD = Math.abs(orderFlowData.cvd);
  const buyersControl = totalCVD > 0 ? (orderFlowData.cvd + totalCVD) / (2 * totalCVD) : 0.5;
  const sellersControl = 1 - buyersControl;

  // Determine if order flow is confirmed (alignment of multiple signals)
  const orderFlowConfirmed =
    orderFlowData.bigPrints.length > 0 &&
    Math.abs(orderFlowData.cvd) > 50;

  // Get session high/low for tracking
  const sessionHigh = bars.reduce((max, bar) => Math.max(max, bar.high), 0);
  const sessionLow = bars.reduce((min, bar) => (min === Infinity ? bar.low : Math.min(min, bar.low)), Infinity);

  // Update enhanced trackers
  const poc = volumeProfile?.poc || currentPrice;
  const vah = volumeProfile?.vah || currentPrice;
  const val = volumeProfile?.val || currentPrice;

  // Update POC cross tracking
  const pocCrossStats = pocCrossTracker.update(currentPrice, poc);

  // Update market statistics
  marketStatsCalc.updateSession(sessionHigh, sessionLow);
  marketStatsCalc.updateTimeInValue(currentPrice, vah, val);
  marketStatsCalc.updateCVD(orderFlowData.cvd);

  const marketStats = marketStatsCalc.calculate(currentPrice, poc, vah, val, 0.25);
  const atr5m = computeATR(bars, 14);
  const atrFast = computeATR(bars, 5);
  const currentRange = bars.length > 0 ? bars[bars.length - 1].high - bars[bars.length - 1].low : null;
  const refAtr = atrFast || atr5m;
  const volatilityRegime = refAtr && currentRange !== null
    ? (currentRange > 1.5 * refAtr ? 'high' : currentRange < 0.7 * refAtr ? 'low' : 'normal')
    : undefined;
  marketStats.volatilityRegime = volatilityRegime;
  marketStats.atr5m = atr5m ?? atrFast ?? undefined;
  marketStats.currentRangeTicks = currentRange !== null && tickSize ? currentRange / tickSize : undefined;

  // Get performance metrics
  const performance = selfLearning ? performanceTracker.getMetrics() : null;

  // Get historical notes (last 10)
  const historicalNotes = selfLearning ? notesManager.getRecentNotes(10) : [];

  // Build microstructure snapshot from order flow
  const microstructure = buildMicrostructureFromOrderFlow(orderFlowData);
  // Derive a directional hint for walls when flat: prefer trend direction if known
  const s = marketStructure.state.toLowerCase();
  const preferredWallSide =
    currentPosition?.side === 'long' ? 'long' :
    currentPosition?.side === 'short' ? 'short' :
    (s.includes('uptrend') || s.includes('trend_up') || s.endsWith('_up'))
      ? 'long'
      : (s.includes('downtrend') || s.includes('trend_down') || s.endsWith('_down'))
        ? 'short'
        : undefined;

  const wallInfo = computeWallInfo(microstructure, currentPrice, tickSize, preferredWallSide);
  if (microstructure) {
    microstructure.nearestRestingWallInDirection = wallInfo.nearestWall;
    microstructure.liquidityPullDetected = wallInfo.pullDetected;
    microstructure.weakWallDetected = wallInfo.pullDetected; // compatibility: weak heuristic only
  }

  // Stable promote gate support: cache last promotion flag across scans if needed (not implemented here, see router)

  // Build macrostructure (session/multi-hour context) from bars + profile
  const macrostructure = buildMacrostructureFromBars(
    bars,
    volumeProfile,
    sessionHigh,
    sessionLow,
    higherTimeframes,
    recentVolumeProfiles
  );

  // Event/zone profiles for LLM
  const tradeLegProfile = buildTradeLegProfile(bars, currentPosition, volumeProfile, tickSize ?? 0.25);
  const pullbackProfile = buildPullbackProfile(bars, currentPosition, tickSize ?? 0.25);
  const watchZoneProfiles = buildWatchZoneProfiles(bars, volumeProfile, tickSize ?? 0.25);

  // Flow derivatives and microstructure proxies
  const flowSignals = computeFlowSignals(orderFlowData);
  const absorptionSignals = computeAbsorptionSignals(
    currentPrice,
    { poc, vah, val, sessionHigh, sessionLow },
    orderFlowData,
    flowSignals,
    tickSize
  );
  const exhaustionSignals = computeExhaustionSignals(
    currentPrice,
    bars,
    orderFlowData,
    flowSignals,
    tickSize
  );
  const reversalScores = computeReversalScores(
    currentPrice,
    volumeProfile,
    flowSignals,
    marketStructure.state,
    absorptionSignals,
    exhaustionSignals
  );

  // Build the market data object
  const marketData: FuturesMarketData = {
    symbol,
    timestamp: new Date().toISOString(),
    currentPrice,

    // Price candles (last 5 for 25-minute window)
    candles: recentCandles,

    // CVD data with OHLC candlestick
    cvd: {
      value: orderFlowData.cvd,
      trend: cvdTrend,
      ohlc: currentCvdBar || {
        timestamp: new Date().toISOString(),
        open: 0,
        high: 0,
        low: 0,
        close: 0,
      },
    },

    // Order flow metrics (absorption/exhaustion)
    orderFlow: {
      bigTrades: orderFlowData.bigPrints.slice(-25).map(print => ({
        price: print.price,
        size: print.size,
        side: print.side,
        timestamp: new Date(print.timestamp).toISOString(),
      })),
    },

    // Volume profile structure
    volumeProfile: volumeProfile || {
      poc: 0,
      vah: 0,
      val: 0,
      lvns: [],
      sessionHigh: currentPrice,
      sessionLow: currentPrice,
    },

    // Market state detection
    marketState: {
      state: marketStructure.state,
      buyersControl,
      sellersControl,
    },

    // Order flow confirmation (all 3 layers aligned)
    orderFlowConfirmed,

  // Account information
  account: {
    balance: accountBalance,
    position: currentPosition
      ? (currentPosition.side === 'long' ? currentPosition.contracts : -currentPosition.contracts)
      : 0,
    unrealizedPnL: currentPosition?.unrealizedPnL || 0,
    realizedPnL,
  },
  openPosition: currentPosition
    ? {
        decisionId: currentPosition.decisionId,
        side: currentPosition.side,
        contracts: currentPosition.contracts,
        entryPrice: currentPosition.entryPrice,
        entryTime: currentPosition.entryTime,
        stopLoss: currentPosition.stopLoss,
        target: currentPosition.target,
        unrealizedPnL: currentPosition.unrealizedPnL,
        stopOrderId: currentPosition.stopOrderId,
        targetOrderId: currentPosition.targetOrderId,
        distanceToStopPoints: Number((currentPosition.side === 'long'
          ? currentPrice - currentPosition.stopLoss
          : currentPosition.stopLoss - currentPrice).toFixed(2)),
        distanceToTargetPoints: Number((currentPosition.side === 'long'
          ? currentPosition.target - currentPrice
          : currentPrice - currentPosition.target).toFixed(2)),
        positionAgeSeconds: Math.max(0, Math.round((Date.now() - new Date(currentPosition.entryTime).getTime()) / 1000)),
        positionVersion: executionManager.getPositionVersion(symbol),
      }
    : null,

    // === ENHANCED FEATURES ===
    pocCrossStats,
    marketStats,
    cvdCandles,
    performance,
    historicalNotes,
    microstructure,
    macrostructure,
    reversalScores,
    tradeLegProfile,
    pullbackProfile,
    watchZoneProfiles,
    flowSignals,
    absorption: absorptionSignals,
    exhaustion: exhaustionSignals,
  };

  // Persist lightweight snapshot for ML dataset building (JSONL, throttled per symbol)
  try {
    const mlSnapshot = buildMlFeatureSnapshot(marketData);
    maybeWriteMlSnapshot(mlSnapshot);
  } catch (error) {
    console.warn('[ML] Failed to write snapshot for meta-label dataset:', (error as Error)?.message || error);
  }

  return marketData;
}

function maybeWriteMlSnapshot(snapshot: ReturnType<typeof buildMlFeatureSnapshot>): void {
  const ts = new Date(snapshot.timestamp || Date.now()).getTime();
  if (!Number.isFinite(ts)) return;

  const last = lastMlSnapshotWrite[snapshot.symbol] || 0;
  if (ts - last < mlSnapshotThrottleMs) {
    return;
  }
  lastMlSnapshotWrite[snapshot.symbol] = ts;

  const targetDir = path.resolve(__dirname, '..', 'ml', 'data');
  const targetFile = path.join(targetDir, 'snapshots.jsonl');
  if (!existsSync(targetDir)) {
    mkdirSync(targetDir, { recursive: true });
  }

  const payload = JSON.stringify(snapshot);
  try {
    appendFileSync(targetFile, `${payload}\n`, { encoding: 'utf-8' });
  } catch (err) {
    console.warn('[ML] Failed to append snapshot:', (err as Error)?.message || err);
  }
}

function buildMicrostructureFromOrderFlow(
  orderFlowData: OrderFlowData,
): FuturesMarketData['microstructure'] {
  const largeWhaleTrades = orderFlowData.bigPrints
    .slice(-25)
    .map(t => ({
      price: t.price,
      size: t.size,
      side: t.side,
      timestamp: new Date(t.timestamp).toISOString(),
    }));

  const restingLimitOrders = Object.entries(orderFlowData.volumeAtPrice || {})
    .map(([priceStr, vol]) => ({
      price: Number(priceStr),
      restingBid: Number(vol.buy ?? 0),
      restingAsk: Number(vol.sell ?? 0),
      total: Number(vol.buy ?? 0) + Number(vol.sell ?? 0),
      lastSeen: new Date(vol.timestamp).toISOString(),
    }))
    .filter(entry => !Number.isNaN(entry.price))
    .sort((a, b) => b.total - a.total)
    .slice(0, 10);

  return {
    largeWhaleTrades,
    restingLimitOrders,
  };
}

function buildMacrostructureFromBars(
  bars: TopstepXFuturesBar[],
  volumeProfile: VolumeProfile | null,
  sessionHigh: number,
  sessionLow: number,
  additionalTimeframes: HigherTimeframeSnapshot[] = [],
  recentProfiles: SessionVolumeProfileSummary[] = [],
): FuturesMarketData['macrostructure'] {
  if (!bars || bars.length === 0) {
    return undefined;
  }

  // Multi-session / session profile (approximate 24h trading day)
  const multiDayProfile = volumeProfile
    ? {
        lookbackHours: 24,
        poc: volumeProfile.poc,
        vah: volumeProfile.vah,
        val: volumeProfile.val,
        high: sessionHigh,
        low: sessionLow,
      }
    : undefined;

  // Aggregate 5-minute bars into 60-minute candles
  const oneHourMs = 60 * 60 * 1000;
  const bucketMap = new Map<number, {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>();

  for (const bar of bars) {
    const ts = new Date(bar.timestamp).getTime();
    if (Number.isNaN(ts)) {
      continue;
    }
    const bucketKey = Math.floor(ts / oneHourMs) * oneHourMs;
    const existing = bucketMap.get(bucketKey);
    if (!existing) {
      bucketMap.set(bucketKey, {
        timestamp: new Date(bucketKey).toISOString(),
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        volume: bar.volume ?? 0,
      });
    } else {
      existing.high = Math.max(existing.high, bar.high);
      existing.low = Math.min(existing.low, bar.low);
      existing.close = bar.close;
      existing.volume += bar.volume ?? 0;
    }
  }

  const hourlyCandles = Array.from(bucketMap.values())
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

  const recentHourly = hourlyCandles.slice(-12); // last ~12 hours

  const higherTimeframes: HigherTimeframeSnapshot[] = [];
  if (recentHourly.length > 0) {
    higherTimeframes.push({
      timeframe: '60m',
      candles: recentHourly,
    });
  }

  additionalTimeframes
    .filter(tf => tf && tf.candles && tf.candles.length > 0)
    .forEach(tf => {
      higherTimeframes.push({
        timeframe: tf.timeframe,
        candles: tf.candles,
      });
    });

  return {
    multiDayProfile,
    higherTimeframes: higherTimeframes.length > 0 ? higherTimeframes : undefined,
    recentVolumeProfiles: recentProfiles.length > 0 ? recentProfiles : undefined,
  };
}

// ------------------- Profile helpers -------------------
function buildTradeLegProfile(
  bars: TopstepXFuturesBar[],
  position: any | null,
  volumeProfile: VolumeProfile | null,
  tickSize: number,
): TradeLegProfile | undefined {
  if (!position || !position.entryTime || bars.length === 0) return undefined;

  const entryTs = new Date(position.entryTime).getTime();
  if (Number.isNaN(entryTs)) return undefined;

  const nowTs = Date.now();
  const maxWindow = 45 * 60 * 1000; // 45 minutes
  const startTs = Math.max(entryTs, nowTs - maxWindow);

  const legBars = bars.filter(b => {
    const ts = new Date(b.timestamp).getTime();
    return !Number.isNaN(ts) && ts >= startTs;
  });
  if (legBars.length === 0) return undefined;

  const profile = buildProfileFromBars(legBars, tickSize);
  if (!profile) return undefined;

  const valueMigration =
    profile.poc > position.entryPrice + 0.5 ? 'up' :
    profile.poc < position.entryPrice - 0.5 ? 'down' : 'flat';

  const startPrice = legBars[0].close;
  const endPrice = legBars[legBars.length - 1].close;
  const move = endPrice - startPrice;
  const range = Math.max(...legBars.map(b => b.high)) - Math.min(...legBars.map(b => b.low));
  const acceptanceStrengthDir = range > 0 ? Math.min(1, Math.abs(move) / range) : 0;

  const hvns = profile.hvns || [];
  const lvns = profile.lvns || [];
  const hvnDir = position.side === 'long'
    ? hvns.find(h => h >= endPrice)
    : hvns.slice().reverse().find(h => h <= endPrice);
  const lvnDir = position.side === 'long'
    ? lvns.find(l => l >= endPrice)
    : lvns.slice().reverse().find(l => l <= endPrice);

  const airPocketAhead = Boolean(lvnDir && ((position.side === 'long' && lvnDir > endPrice) || (position.side === 'short' && lvnDir < endPrice)));

  return {
    ...profile,
    valueMigration,
    acceptanceStrengthDir,
    hvnDir,
    lvnDir,
    airPocketAhead,
  };
}

function buildPullbackProfile(
  bars: TopstepXFuturesBar[],
  position: any | null,
  tickSize: number,
): PullbackProfile | undefined {
  if (!position || bars.length === 0) return undefined;

  const entryPrice = position.entryPrice;
  const closes = bars.map(b => b.close);
  const highs = bars.map(b => b.high);
  const lows = bars.map(b => b.low);

  let mfe = 0;
  let mfeIndex = -1;
  if (position.side === 'long') {
    const maxHigh = Math.max(...highs);
    mfe = maxHigh - entryPrice;
    mfeIndex = highs.indexOf(maxHigh);
  } else {
    const minLow = Math.min(...lows);
    mfe = entryPrice - minLow;
    mfeIndex = lows.indexOf(minLow);
  }

  if (mfe <= 0) return { poc: 0, vah: 0, val: 0, hvns: [], lvns: [], active: false };

  const currentPrice = closes[closes.length - 1];
  const retrace = position.side === 'long'
    ? (mfe - (currentPrice - entryPrice)) / mfe
    : (mfe - (entryPrice - currentPrice)) / mfe;

  if (retrace < 0.3 || mfeIndex === -1) {
    return { poc: 0, vah: 0, val: 0, hvns: [], lvns: [], active: false };
  }

  const pullbackBarsAll = bars.slice(Math.max(0, mfeIndex));
  const nowTs = Date.now();
  const pullbackBars = pullbackBarsAll.filter(b => {
    const ts = new Date(b.timestamp).getTime();
    return !Number.isNaN(ts) && nowTs - ts <= 5 * 60 * 1000;
  });
  if (pullbackBars.length === 0) return { poc: 0, vah: 0, val: 0, hvns: [], lvns: [], active: false };

  const profile = buildProfileFromBars(pullbackBars, tickSize);
  if (!profile) return { poc: 0, vah: 0, val: 0, hvns: [], lvns: [], active: false };

  const lastClose = pullbackBars[pullbackBars.length - 1].close;
  const acceptanceState =
    lastClose >= profile.val && lastClose <= profile.vah ? 'accepting'
    : 'rejecting';

  return {
    ...profile,
    acceptanceState,
    active: true,
  };
}

function buildWatchZoneProfiles(
  bars: TopstepXFuturesBar[],
  volumeProfile: VolumeProfile | null,
  tickSize: number,
): WatchZoneProfile[] | undefined {
  if (!volumeProfile || bars.length === 0) return undefined;
  const tick = tickSize || 0.25;
  const zones: Array<{ name: string; low: number; high: number }> = [];

  zones.push({ name: 'VAH_retest', low: volumeProfile.vah - 4 * tick, high: volumeProfile.vah + 4 * tick });
  zones.push({ name: 'VAL_retest', low: volumeProfile.val - 4 * tick, high: volumeProfile.val + 4 * tick });

  if (volumeProfile.lvns && volumeProfile.lvns.length > 0) {
    const currentClose = bars[bars.length - 1].close;
    const biasLong = currentClose >= volumeProfile.poc;
    const lvnsSorted = volumeProfile.lvns.slice().sort((a, b) => a - b);
    const candidate = biasLong
      ? lvnsSorted.find(l => l > currentClose)
      : lvnsSorted.slice().reverse().find(l => l < currentClose);
    const nearestLvn = candidate !== undefined ? candidate : lvnsSorted[0];
    if (nearestLvn !== undefined) {
      zones.push({ name: 'LVN_lane', low: nearestLvn - 2 * tick, high: nearestLvn + 2 * tick });
    }
  }

  const nowTs = Date.now();
  const windowMs = 20 * 60 * 1000;

  const profiles: WatchZoneProfile[] = zones.slice(0, 3).map(zone => {
    const zoneBars = bars.filter(b => {
      const ts = new Date(b.timestamp).getTime();
      if (Number.isNaN(ts) || nowTs - ts > windowMs) return false;
      return b.high >= zone.low && b.low <= zone.high;
    });
    const profile = buildProfileFromBars(zoneBars, tickSize) || null;
    const close = zoneBars.length ? zoneBars[zoneBars.length - 1].close : undefined;
    const acceptanceState = close !== undefined && profile
      ? (close >= profile.val && close <= profile.vah ? 'accepting' : 'rejecting')
      : undefined;
    const acceptanceStrength = profile && close !== undefined && profile.vah !== profile.val
      ? Math.min(1, Math.abs(close - profile.poc) / Math.max(1, Math.abs(profile.vah - profile.val)))
      : undefined;

    return {
      name: zone.name,
      low: zone.low,
      high: zone.high,
      poc: profile?.poc,
      acceptanceState,
      acceptanceStrength,
      breakthroughScore: acceptanceState === 'accepting' ? 0.6 : 0.4,
      absorptionScore: undefined,
    };
  });

  return profiles.length > 0 ? profiles : undefined;
}

function buildProfileFromBars(bars: TopstepXFuturesBar[], tickSize: number = 0.25): { poc: number; vah: number; val: number; hvns: number[]; lvns: number[] } | null {
  if (!bars || bars.length === 0) return null;

  const priceVolume = new Map<number, number>();

  const tick = tickSize || 0.25; // quarter tick granularity default
  bars.forEach(bar => {
    // Use typical price to reduce fake HVNs/LVNs from OHLC splits
    const typicalPrice = (bar.high + bar.low + 2 * bar.close) / 4;
    const rounded = Math.round(typicalPrice / tick) * tick;
    const prev = priceVolume.get(rounded) || 0;
    priceVolume.set(rounded, prev + (bar.volume || 0));
  });

  const nodes = Array.from(priceVolume.entries()).map(([price, volume]) => ({ price, volume })).sort((a, b) => a.price - b.price);
  if (nodes.length === 0) return null;

  const pocNode = nodes.reduce((max, n) => n.volume > max.volume ? n : max, nodes[0]);
  const poc = pocNode.price;

  const totalVolume = nodes.reduce((sum, n) => sum + n.volume, 0);
  const targetVolume = totalVolume * 0.7;

  let vah = poc;
  let val = poc;
  let currentVolume = pocNode.volume;
  let upperIndex = nodes.findIndex(n => n.price === poc) + 1;
  let lowerIndex = nodes.findIndex(n => n.price === poc) - 1;

  while (currentVolume < targetVolume && (upperIndex < nodes.length || lowerIndex >= 0)) {
    const upperVol = upperIndex < nodes.length ? nodes[upperIndex].volume : 0;
    const lowerVol = lowerIndex >= 0 ? nodes[lowerIndex].volume : 0;

    if (upperVol >= lowerVol && upperIndex < nodes.length) {
      currentVolume += upperVol;
      vah = nodes[upperIndex].price;
      upperIndex += 1;
    } else if (lowerIndex >= 0) {
      currentVolume += lowerVol;
      val = nodes[lowerIndex].price;
      lowerIndex -= 1;
    } else {
      break;
    }
  }

  const avgVolume = totalVolume / nodes.length;
  const hvns = nodes.slice().sort((a, b) => b.volume - a.volume).slice(0, 3).map(n => n.price);
  const lvns = nodes.filter(n => n.volume < avgVolume * 0.5).map(n => n.price).slice(0, 5);

  return { poc, vah, val, hvns, lvns };
}

function computeWallInfo(
  microstructure: FuturesMarketData['microstructure'] | undefined,
  currentPrice: number,
  tickSize: number,
  preferredSide?: 'long' | 'short'
): { nearestWall: { side: 'bid' | 'ask'; price: number; size: number; distance: number } | undefined; pullDetected: boolean } {
  if (!microstructure || !microstructure.restingLimitOrders || microstructure.restingLimitOrders.length === 0) {
    return { nearestWall: undefined, pullDetected: false };
  }

  const bids = microstructure.restingLimitOrders.filter(l => l.restingBid > 0 && l.price <= currentPrice);
  const asks = microstructure.restingLimitOrders.filter(l => l.restingAsk > 0 && l.price >= currentPrice);

  const nearestBid = bids.sort((a, b) => Math.abs(currentPrice - a.price) - Math.abs(currentPrice - b.price))[0];
  const nearestAsk = asks.sort((a, b) => Math.abs(currentPrice - a.price) - Math.abs(currentPrice - b.price))[0];

  let nearestWall: { side: 'bid' | 'ask'; price: number; size: number; distance: number } | undefined;
  if (nearestBid) {
    nearestWall = {
      side: 'bid',
      price: nearestBid.price,
      size: nearestBid.restingBid,
      distance: Number(((currentPrice - nearestBid.price) / (tickSize || 1)).toFixed(2)),
    };
  }
  if (nearestAsk) {
    const askInfo = {
      side: 'ask' as const,
      price: nearestAsk.price,
      size: nearestAsk.restingAsk,
      distance: Number(((nearestAsk.price - currentPrice) / (tickSize || 1)).toFixed(2)),
    };
    if (!nearestWall || askInfo.distance < nearestWall.distance) {
      nearestWall = askInfo;
    }
  }

  if (preferredSide === 'long' && nearestAsk) {
    nearestWall = {
      side: 'ask',
      price: nearestAsk.price,
      size: nearestAsk.restingAsk,
      distance: Number(((nearestAsk.price - currentPrice) / (tickSize || 1)).toFixed(2)),
    };
  } else if (preferredSide === 'short' && nearestBid) {
    nearestWall = {
      side: 'bid',
      price: nearestBid.price,
      size: nearestBid.restingBid,
      distance: Number(((currentPrice - nearestBid.price) / (tickSize || 1)).toFixed(2)),
    };
  }

  const sizes = microstructure.restingLimitOrders.map(l => l.total);
  const avgSize = sizes.length ? sizes.reduce((a, b) => a + b, 0) / sizes.length : 0;
  const pullDetected = nearestWall ? nearestWall.size < avgSize * 0.5 : false; // weak wall heuristic

  return { nearestWall, pullDetected };
}

function computeReversalScores(
  currentPrice: number,
  volumeProfile: VolumeProfile | null,
  flowSignals: FlowSignals | undefined,
  marketState: MarketState,
  absorptionSignals: AbsorptionSignal[] | undefined,
  exhaustionSignals: AbsorptionSignal[] | undefined
): { long: number; short: number } {
  let revLong = 0.45;
  let revShort = 0.45;

  const marketStateStr = typeof marketState === 'string'
    ? marketState
    : (marketState as any)?.state ?? '';

  if (volumeProfile) {
    if (currentPrice < volumeProfile.val) revLong += 0.1;
    if (currentPrice > volumeProfile.vah) revShort += 0.1;
  }

  const absorptionBid = absorptionSignals?.find(a => (a.side === 'bid' || a.side === 'buy') && a.strength >= 0.5);
  const absorptionAsk = absorptionSignals?.find(a => (a.side === 'ask' || a.side === 'sell') && a.strength >= 0.5);
  if (absorptionBid) revLong += 0.1;
  if (absorptionAsk) revShort += 0.1;

  const exhaustBuy = exhaustionSignals?.find(e => (e.side === 'buy' || e.side === 'ask') && e.strength >= 0.5);
  const exhaustSell = exhaustionSignals?.find(e => (e.side === 'sell' || e.side === 'bid') && e.strength >= 0.5);
  if (exhaustBuy) revShort += 0.1;
  if (exhaustSell) revLong += 0.1;

  if (flowSignals?.cvdDivergence === 'strong') {
    if (marketStateStr.includes('uptrend')) {
      revShort += 0.08;
    } else if (marketStateStr.includes('downtrend')) {
      revLong += 0.08;
    } else {
      revLong += 0.05;
      revShort += 0.05;
    }
  }

  if (marketStateStr.includes('uptrend')) {
    revShort -= 0.1;
    revLong += 0.05;
  } else if (marketStateStr.includes('downtrend')) {
    revLong -= 0.1;
    revShort += 0.05;
  }

  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  return {
    long: clamp(revLong),
    short: clamp(revShort),
  };
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function estimateTickSize(orderFlowData: OrderFlowData, bars: TopstepXFuturesBar[]): number {
  const prices: number[] = [];
  orderFlowData.bigPrints?.forEach(t => prices.push(t.price));
  Object.keys(orderFlowData.volumeAtPrice || {}).forEach(p => prices.push(Number(p)));
  bars?.forEach(b => prices.push(b.close, b.high, b.low, b.open));

  const sorted = prices
    .filter(p => Number.isFinite(p))
    .sort((a, b) => a - b);

  let minDiff = Number.POSITIVE_INFINITY;
  for (let i = 1; i < sorted.length; i += 1) {
    const diff = sorted[i] - sorted[i - 1];
    if (diff > 0 && diff < minDiff) {
      minDiff = diff;
    }
  }

  if (!Number.isFinite(minDiff) || minDiff <= 0) {
    return 0.25; // NQ/MNQ default tick
  }
  return Number(minDiff.toFixed(6));
}

function computeFlowSignals(orderFlowData: OrderFlowData): FlowSignals {
  const history = (orderFlowData.cvdHistory || [])
    .map(h => ({
      timestamp: Number(h.timestamp),
      cvd: Number(h.cvd),
      delta: Number(h.delta),
    }))
    .filter(h => Number.isFinite(h.timestamp) && Number.isFinite(h.cvd))
    .sort((a, b) => a.timestamp - b.timestamp);

  const now = Date.now();
  const deltaLast1m = history
    .filter(h => now - h.timestamp <= 60_000)
    .reduce((sum, h) => sum + (Number.isFinite(h.delta) ? h.delta : 0), 0);
  const deltaLast5m = history
    .filter(h => now - h.timestamp <= 300_000)
    .reduce((sum, h) => sum + (Number.isFinite(h.delta) ? h.delta : 0), 0);

  const cvdSlopeShort = computeCvdSlope(history, 60_000);
  const cvdSlopeLong = computeCvdSlope(history, 300_000);

  let cvdDivergence: FlowSignals['cvdDivergence'] = 'none';
  if (cvdSlopeShort !== 0 && Math.sign(cvdSlopeShort) !== Math.sign(cvdSlopeLong || 0)) {
    cvdDivergence = Math.abs(cvdSlopeShort) > Math.abs(cvdSlopeLong || 0) * 1.2 ? 'strong' : 'weak';
  }

  return {
    deltaLast1m: Number.isFinite(deltaLast1m) ? Number(deltaLast1m.toFixed(2)) : undefined,
    deltaLast5m: Number.isFinite(deltaLast5m) ? Number(deltaLast5m.toFixed(2)) : undefined,
    cvdSlopeShort: Number.isFinite(cvdSlopeShort) ? Number(cvdSlopeShort.toFixed(2)) : undefined,
    cvdSlopeLong: Number.isFinite(cvdSlopeLong) ? Number(cvdSlopeLong.toFixed(2)) : undefined,
    cvdDivergence,
  };
}

function computeCvdSlope(
  history: Array<{ timestamp: number; cvd: number }>,
  windowMs: number
): number {
  const now = Date.now();
  const slice = history.filter(h => now - h.timestamp <= windowMs);
  if (slice.length < 2) return 0;
  const first = slice[0];
  const last = slice[slice.length - 1];
  const seconds = Math.max((last.timestamp - first.timestamp) / 1000, 1);
  return (last.cvd - first.cvd) / seconds;
}

function computeAbsorptionSignals(
  currentPrice: number,
  levels: { poc: number; vah: number; val: number; sessionHigh: number; sessionLow: number },
  orderFlowData: OrderFlowData,
  flowSignals: FlowSignals | undefined,
  tickSize: number
): AbsorptionSignal[] {
  const levelCandidates = [
    { levelName: 'POC', price: levels.poc },
    { levelName: 'VAH', price: levels.vah },
    { levelName: 'VAL', price: levels.val },
    { levelName: 'SessionHigh', price: levels.sessionHigh },
    { levelName: 'SessionLow', price: levels.sessionLow },
  ]
    .filter(l => Number.isFinite(l.price))
    .reduce((acc: typeof levelCandidates, curr) => {
      if (!acc.some(entry => entry.price === curr.price && entry.levelName === curr.levelName)) {
        acc.push(curr);
      }
      return acc;
    }, []);

  const bandTicks = 12;
  const band = Math.max(tickSize || 0.25, 0.0001) * bandTicks;
  const now = Date.now();
  const windowMs = 90_000;
  const tradesWindow = (orderFlowData.bigPrints || [])
    .map(t => ({
      ...t,
      timestamp: Number(t.timestamp),
    }))
    .filter(t => Number.isFinite(t.timestamp) && now - t.timestamp <= windowMs);

  const windowStartPrice = tradesWindow[0]?.price ?? currentPrice;
  const windowDurationSec = tradesWindow.length > 0
    ? Math.max(1, (now - tradesWindow[0].timestamp) / 1000)
    : Math.round(windowMs / 1000);

  const volAtPrice = Object.entries(orderFlowData.volumeAtPrice || {})
    .map(([priceStr, vol]) => ({
      price: Number(priceStr),
      bid: Number((vol as any)?.buy ?? 0),
      ask: Number((vol as any)?.sell ?? 0),
    }))
    .filter(v => Number.isFinite(v.price));

  const medianBid = median(volAtPrice.map(v => v.bid).filter(v => v > 0));
  const medianAsk = median(volAtPrice.map(v => v.ask).filter(v => v > 0));

  const totalAggression = tradesWindow.reduce((sum, t) => sum + (t.size || 0), 0);
  const aggressionRef = Math.max(totalAggression * 0.5, 10);

  const signals: AbsorptionSignal[] = [];

  const buildSignal = (side: 'ask' | 'bid', levelName: string, price: number): AbsorptionSignal | null => {
    const aggression = tradesWindow
      .filter(t => t.side === (side === 'ask' ? 'buy' : 'sell') && Math.abs(t.price - price) <= band)
      .reduce((sum, t) => sum + (t.size || 0), 0);

    if (!Number.isFinite(aggression) || aggression <= 0) {
      return null;
    }

    const progress = side === 'ask'
      ? Math.max((currentPrice - windowStartPrice) / Math.max(tickSize, 0.0001), 1)
      : Math.max((windowStartPrice - currentPrice) / Math.max(tickSize, 0.0001), 1);

    const stall = aggression / progress;

    const wallsInBand = volAtPrice.filter(v => Math.abs(v.price - price) <= band);
    const wallSize = side === 'ask'
      ? Math.max(...wallsInBand.map(w => w.ask), 0)
      : Math.max(...wallsInBand.map(w => w.bid), 0);
    const wallBoost = clamp01(wallSize / Math.max(side === 'ask' ? medianAsk : medianBid || 1, 1));

    const stallScore = clamp01(stall / 25);
    const aggressionScore = clamp01(aggression / aggressionRef);
    const strength = clamp01(0.5 * stallScore + 0.3 * aggressionScore + 0.2 * wallBoost);
    const cvdSlope = flowSignals?.cvdSlopeShort ?? 0;
    const confirmedByCvd = side === 'ask' ? cvdSlope <= 0 : cvdSlope >= 0;

    return {
      levelName,
      price: Number(price),
      side,
      strength: Number(strength.toFixed(2)),
      durationSec: Math.round(windowDurationSec),
      confirmedByCvd,
    };
  };

  levelCandidates.forEach(level => {
    const askSignal = buildSignal('ask', level.levelName, level.price);
    const bidSignal = buildSignal('bid', level.levelName, level.price);
    [askSignal, bidSignal].forEach(signal => {
      if (signal && signal.strength > 0.2) {
        signals.push(signal);
      }
    });
  });

  return signals
    .sort((a, b) => b.strength - a.strength)
    .slice(0, 3);
}

function computeExhaustionSignals(
  currentPrice: number,
  bars: TopstepXFuturesBar[],
  orderFlowData: OrderFlowData,
  flowSignals: FlowSignals | undefined,
  tickSize: number
): AbsorptionSignal[] {
  const now = Date.now();
  const windowShortMs = 60_000;
  const windowLongMs = 180_000;

  const trades = (orderFlowData.bigPrints || [])
    .map(t => ({ ...t, timestamp: Number(t.timestamp) }))
    .filter(t => Number.isFinite(t.timestamp));

  const windowShort = trades.filter(t => now - t.timestamp <= windowShortMs);
  const windowLong = trades.filter(t => now - t.timestamp > windowShortMs && now - t.timestamp <= windowLongMs);

  const sumAgg = (list: typeof trades, side: 'buy' | 'sell') =>
    list.filter(t => t.side === side).reduce((sum, t) => sum + (t.size || 0), 0);

  const buyDecay = clamp01((sumAgg(windowLong, 'buy') - sumAgg(windowShort, 'buy')) / Math.max(sumAgg(windowLong, 'buy'), 1));
  const sellDecay = clamp01((sumAgg(windowLong, 'sell') - sumAgg(windowShort, 'sell')) / Math.max(sumAgg(windowLong, 'sell'), 1));

  const recentProgress = computeProgress(bars.slice(-3), tickSize);
  const priorProgress = computeProgress(bars.slice(-7, -3), tickSize);
  const failMove = (progress: number) => priorProgress > 0 && progress < priorProgress * 0.4;

  const cvdFlip = !!flowSignals && (flowSignals.cvdSlopeShort || 0) !== 0 &&
    Math.sign(flowSignals.cvdSlopeShort || 0) !== Math.sign(flowSignals.cvdSlopeLong || 0);

  const buildSignal = (side: 'ask' | 'bid', decay: number, failed: boolean): AbsorptionSignal | null => {
    const strength = clamp01(0.5 * decay + 0.3 * (failed ? 1 : 0) + 0.2 * (cvdFlip ? 1 : 0));
    if (strength <= 0.2) {
      return null;
    }
    return {
      levelName: 'RecentMove',
      price: currentPrice,
      side,
      strength: Number(strength.toFixed(2)),
      durationSec: Math.round(windowShortMs / 1000),
      confirmedByCvd: cvdFlip,
    };
  };

  const signals: AbsorptionSignal[] = [];
  const failRecent = failMove(recentProgress);

  const buySignal = buildSignal('ask', buyDecay, failRecent);
  const sellSignal = buildSignal('bid', sellDecay, failRecent);
  if (buySignal) signals.push(buySignal);
  if (sellSignal) signals.push(sellSignal);

  return signals.sort((a, b) => b.strength - a.strength).slice(0, 2);
}

function computeProgress(bars: TopstepXFuturesBar[], tickSize: number): number {
  if (!bars || bars.length === 0) return 0;
  const highs = bars.map(b => b.high);
  const lows = bars.map(b => b.low);
  const range = Math.max(...highs) - Math.min(...lows);
  return Math.max(range / Math.max(tickSize || 0.25, 0.0001), 0);
}

function median(values: number[]): number {
  const filtered = values.filter(v => Number.isFinite(v)).sort((a, b) => a - b);
  if (filtered.length === 0) return 0;
  const mid = Math.floor(filtered.length / 2);
  if (filtered.length % 2 === 0) {
    return (filtered[mid - 1] + filtered[mid]) / 2;
  }
  return filtered[mid];
}

/**
 * Process OpenAI decision and execute if criteria met
 */
export async function processOpenAIDecision(
  openaiDecision: OpenAITradingDecision | null,
  executionManager: ExecutionManager,
  currentPrice: number,
  symbol: string,
  orderFlowData: OrderFlowData,
  volumeProfile: VolumeProfile | null,
  marketStructure: MarketStructure
): Promise<{ executed: boolean; decisionId?: string }> {
  if (!openaiDecision) {
    return { executed: false };
  }

  // Only execute if decision is BUY/SELL and confidence is high
  if (openaiDecision.decision === 'HOLD') {
    await maybeAdjustExistingPosition(executionManager, openaiDecision);
    console.log('[OpenAI] HOLD - Not executing');
    return { executed: false };
  }

  // Minimum confidence threshold
  if (openaiDecision.confidence < 70) {
    console.log(`[OpenAI] Confidence ${openaiDecision.confidence}% below 70% threshold - Not executing`);
    return { executed: false };
  }

  // Check if already in position (one position at a time)
  const activePosition = executionManager.getActivePosition();
  if (activePosition) {
    // Check if signal is opposite to current position
    const isOppositeSignal =
      (activePosition.side === 'long' && openaiDecision.decision === 'SELL') ||
      (activePosition.side === 'short' && openaiDecision.decision === 'BUY');

    if (isOppositeSignal) {
      console.log(`[OpenAI] ⚠️  OPPOSITE SIGNAL DETECTED: Current position is ${activePosition.side.toUpperCase()} but got ${openaiDecision.decision} signal`);
      console.log(`[OpenAI] This would auto-reverse the position. Skipping to prevent unintended reversal.`);
      console.log(`[OpenAI] To close position, use explicit exit logic or let stop-loss/target handle it.`);
      return { executed: false };
    }

    await maybeAdjustExistingPosition(executionManager, openaiDecision);

    console.log('[OpenAI] Already in position - Not executing new entry');
    return { executed: false };
  }

  // Execute the decision
  const order = await executionManager.executeDecision(
    openaiDecision,
    currentPrice,
    {
      entryPrice: openaiDecision.entryPrice ?? null,
      stopLoss: openaiDecision.stopLoss ?? null,
      takeProfit: openaiDecision.target ?? null,
    }
  );

  if (!order) {
    // Even if no new order, allow position management adjustments
    await maybeAdjustExistingPosition(executionManager, openaiDecision);
    return { executed: false };
  }

  // Get the active position to update with additional context
  const position = executionManager.getActivePosition();
  if (position) {
    // Record in database with full Fabio context
    const decision = tradingDB.recordDecision({
      symbol,
      marketState: openaiDecision.marketState,
      location: openaiDecision.location,
      setupModel: openaiDecision.setupModel,
      decision: openaiDecision.decision as 'BUY' | 'SELL' | 'HOLD',
      confidence: openaiDecision.confidence,
      entryPrice: currentPrice,
      stopLoss: openaiDecision.stopLoss || position.stopLoss || currentPrice - 30,
      target: openaiDecision.target || position.target || currentPrice + 30,
      riskPercent: openaiDecision.riskPercent,
      source: 'openai',
      reasoning: openaiDecision.reasoning,
      cvd: orderFlowData.cvd,
      cvdTrend: openaiDecision.decision === 'BUY' ? 'up' : 'down',
      currentPrice,
      buyAbsorption: orderFlowData.absorption.buy,
      sellAbsorption: orderFlowData.absorption.sell,
    });

    console.log(
      `[OpenAI] ✅ Executed ${openaiDecision.decision} @ ${currentPrice} | Entry: ${openaiDecision.entryPrice} | SL: ${openaiDecision.stopLoss} | TP: ${openaiDecision.target} | Confidence: ${openaiDecision.confidence}%`
    );

    await maybeAdjustExistingPosition(executionManager, openaiDecision);

    return { executed: true, decisionId: decision.id };
  }

  return { executed: false };
}

async function maybeAdjustExistingPosition(
  executionManager: ExecutionManager,
  openaiDecision: OpenAITradingDecision
) {
  const activePosition = executionManager.getActivePosition();
  if (!activePosition) {
    return;
  }

  const desiredStop = typeof openaiDecision.stopLoss === 'number' ? openaiDecision.stopLoss : undefined;
  const desiredTarget = typeof openaiDecision.target === 'number' ? openaiDecision.target : undefined;

  if (desiredStop == null && desiredTarget == null) {
    console.log('[OpenAI] Active position detected but no stop/target guidance provided in JSON.');
    return;
  }

  const matchingSide =
    (activePosition.side === 'long' && openaiDecision.decision === 'BUY') ||
    (activePosition.side === 'short' && openaiDecision.decision === 'SELL') ||
    openaiDecision.decision === 'HOLD';

  if (!matchingSide) {
    return;
  }

  console.log(
    `[OpenAI] Monitoring active ${activePosition.side.toUpperCase()} — Proposed stop: ${desiredStop ?? 'unchanged'}, target: ${desiredTarget ?? 'unchanged'}`
  );

  const adjusted = await executionManager.adjustActiveProtection(desiredStop, desiredTarget, openaiDecision.positionVersion);
  if (adjusted) {
    console.log('[OpenAI] 🔧 Updated protective orders based on latest plan.');
  } else {
    console.log('[OpenAI] Protective orders unchanged (levels already aligned or missing order IDs).');
  }
}

/**
 * Update active position and check for exits
 */
export async function updatePositionAndCheckExits(
  executionManager: ExecutionManager,
  currentPrice: number,
  bars: TopstepXFuturesBar[],
  openaiDecision?: OpenAITradingDecision,
  marketStructure?: MarketStructure
): Promise<{ exited: boolean; closedDecisionId?: string; reason?: string }> {
  const selfLearning = isSelfLearningEnabled();
  const activePosition = executionManager.getActivePosition();

  if (!activePosition) {
    return { exited: false };
  }

  // Update position with current price
  executionManager.updatePositionPrice(activePosition.decisionId, currentPrice);

  // Check for exit conditions
  const closedDecisionId = await executionManager.checkExits(currentPrice);

  if (closedDecisionId) {
    const outcome = tradingDB.getOutcome(closedDecisionId);

    if (outcome) {
      if (selfLearning) {
        // Record performance for self-learning
        performanceTracker.recordTrade(outcome.profitLoss);

        // Add note if significant loss (> $200)
        if (outcome.profitLoss < -200) {
          const decision = tradingDB.getDecision(closedDecisionId);
          if (decision) {
            notesManager.addNote(
              `Large loss (${outcome.profitLoss.toFixed(2)}) on ${decision.setupModel || 'unknown'} setup - review entry conditions`,
              marketStructure?.state || 'unknown'
            );
          }
        }

        // Add note from OpenAI if provided
        if (openaiDecision?.noteForFuture) {
          notesManager.addNote(
            openaiDecision.noteForFuture,
            marketStructure?.state || 'unknown'
          );
        }
      }

      console.log(
        `[Position] ✅ Closed: ${outcome.reason} | P&L: ${outcome.profitLoss > 0 ? '+' : ''}$${outcome.profitLoss.toFixed(2)} (${outcome.profitLossPercent.toFixed(2)}%)`
      );
    }

    return { exited: true, closedDecisionId, reason: outcome?.reason };
  }

  return { exited: false };
}

/**
 * Get trading statistics for learning
 */
export function getTradeStats(symbol: string) {
  return tradingDB.calculateStats(symbol);
}

/**
 * Log statistics to console for monitoring
 */
export function logTradeStats(symbol: string) {
  const stats = tradingDB.calculateStats(symbol);

  if (stats.totalOutcomes === 0) {
    console.log(`[Stats] No completed trades yet for ${symbol}`);
    return;
  }

  console.log(`
    📊 Trading Statistics for ${symbol}:
    ├─ Total Decisions: ${stats.totalDecisions}
    ├─ Filled Orders: ${stats.totalFilled}
    ├─ Completed Trades: ${stats.totalOutcomes}
    ├─ Win Rate: ${stats.winRate.toFixed(1)}%
    ├─ Avg Win: $${stats.avgWin.toFixed(2)}
    ├─ Avg Loss: $${stats.avgLoss.toFixed(2)}
    ├─ Profit Factor: ${stats.profitFactor.toFixed(2)}
    │
    ├─ By Source:
    │  ├─ OpenAI: ${stats.bySource['openai']?.count || 0} trades (${((stats.bySource['openai']?.wins || 0) / (stats.bySource['openai']?.count || 1) * 100).toFixed(1)}% win rate)
    │  └─ Rule-based: ${stats.bySource['rule_based']?.count || 0} trades (${((stats.bySource['rule_based']?.wins || 0) / (stats.bySource['rule_based']?.count || 1) * 100).toFixed(1)}% win rate)
    │
    └─ By Setup Model:
       ├─ Trend Continuation: ${stats.bySetupModel['trend_continuation']?.count || 0} trades (${((stats.bySetupModel['trend_continuation']?.wins || 0) / (stats.bySetupModel['trend_continuation']?.count || 1) * 100).toFixed(1)}% win rate)
       └─ Mean Reversion: ${stats.bySetupModel['mean_reversion']?.count || 0} trades (${((stats.bySetupModel['mean_reversion']?.wins || 0) / (stats.bySetupModel['mean_reversion']?.count || 1) * 100).toFixed(1)}% win rate)
  `);
}

/**
 * Export all trading data for external analysis
 */
export function exportTradingData(symbol: string) {
  return {
    symbol,
    timestamp: new Date().toISOString(),
    data: tradingDB.exportData(),
  };
}

/**
 * Get recent high-confidence decisions for analysis
 */
export function getHighConfidenceDecisions(symbol: string, minConfidence: number = 75) {
  const decisions = tradingDB.getDecisionsBySymbol(symbol);
  return decisions
    .filter((d) => d.confidence >= minConfidence)
    .map((d) => ({
      id: d.id,
      timestamp: d.timestamp,
      decision: d.decision,
      confidence: d.confidence,
      outcome: tradingDB.getOutcome(d.id),
    }));
}

/**
 * Analyze which confidence levels actually work
 */
export function analyzeConfidenceCalibration(symbol: string) {
  const decisions = tradingDB.getDecisionsBySymbol(symbol);
  const groups = tradingDB.getDecisionsByConfidence(symbol);

  const analysis: { [key: string]: { count: number; wins: number; winRate: number } } = {};

  Object.entries(groups).forEach(([group, decisions]) => {
    const outcomes = decisions.map((d) => tradingDB.getOutcome(d.id)).filter((o) => !!o);
    const wins = outcomes.filter((o) => o?.profitLoss! > 0).length;

    analysis[group] = {
      count: decisions.length,
      wins,
      winRate: decisions.length > 0 ? (wins / decisions.length) * 100 : 0,
    };
  });

  console.log(`
    📈 Confidence Calibration Analysis for ${symbol}:
    ├─ Very High (80%+): ${analysis['very_high']?.count || 0} trades, ${analysis['very_high']?.winRate.toFixed(1) || 0}% win rate
    ├─ High (60-79%): ${analysis['high']?.count || 0} trades, ${analysis['high']?.winRate.toFixed(1) || 0}% win rate
    ├─ Medium (40-59%): ${analysis['medium']?.count || 0} trades, ${analysis['medium']?.winRate.toFixed(1) || 0}% win rate
    └─ Low (<40%): ${analysis['low']?.count || 0} trades, ${analysis['low']?.winRate.toFixed(1) || 0}% win rate
  `);

  return analysis;
}

export { ExecutionManager };
