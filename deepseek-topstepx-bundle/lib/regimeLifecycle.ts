import { getCached, setCache } from './dataCache';
import {
  RegimeTradeLifecycle,
  RegimeTradeSignal,
  RegimeStage2Summary,
  RegimeStage3Profile,
  RegimeTradeStatus,
} from '@/types';
import { subscribeSymbolForMode, unsubscribeSymbolForMode } from './tradierStream';
import { emitLifecycleUpdate } from './lifecycleBus';

interface LifecycleCache {
  [symbol: string]: RegimeTradeLifecycle[];
}

interface ManageLifecycleParams {
  mode: 'scalp' | 'swing' | 'leaps';
  symbol: string;
  price: number;
  stage2: RegimeStage2Summary;
  stage3: RegimeStage3Profile;
  signals: RegimeTradeSignal[];
}

const cacheKeyForMode = (mode: 'scalp' | 'swing' | 'leaps') => `regime_trades_${mode}`;
type LifecycleMode = 'scalp' | 'swing' | 'leaps';
const lifecycleContext = new Map<string, { stage2: RegimeStage2Summary; stage3: RegimeStage3Profile }>();

const ACTIVE_STATUSES: RegimeTradeStatus[] = ['watching', 'entered', 'scaled'];

const contextKey = (mode: LifecycleMode, symbol: string) => `${mode}:${symbol.toUpperCase()}`;

const hasActiveTrades = (trades: RegimeTradeLifecycle[]) =>
  trades.some(trade => ACTIVE_STATUSES.includes(trade.status));

function broadcastLifecycle(
  mode: LifecycleMode,
  symbol: string,
  stage2: RegimeStage2Summary,
  stage3: RegimeStage3Profile,
  trades: RegimeTradeLifecycle[],
) {
  emitLifecycleUpdate({
    mode,
    symbol,
    trades,
    stage2,
    stage3,
    timestamp: new Date().toISOString(),
  });
}

function loadCache(mode: 'scalp' | 'swing' | 'leaps'): LifecycleCache {
  const cached = getCached<LifecycleCache>(cacheKeyForMode(mode));
  return cached?.data ? { ...cached.data } : {};
}

function persistCache(mode: 'scalp' | 'swing' | 'leaps', data: LifecycleCache): void {
  setCache(cacheKeyForMode(mode), data, 'regime-lifecycle');
}

function rMultiple(
  trade: RegimeTradeLifecycle,
  currentPrice: number,
): number {
  if (trade.riskPerShare === 0) {
    return 0;
  }

  if (trade.direction === 'long') {
    return (currentPrice - trade.entryPrice) / trade.riskPerShare;
  }

  return (trade.entryPrice - currentPrice) / trade.riskPerShare;
}

function recordHistory(
  trade: RegimeTradeLifecycle,
  status: RegimeTradeStatus,
  note: string,
  timestamp: string,
) {
  trade.history.push({ timestamp, status, note });
  trade.status = status;
  trade.lastUpdated = timestamp;
}

function createLifecycleFromSignal(signal: RegimeTradeSignal, nowISO: string): RegimeTradeLifecycle {
  const derivedRisk = Math.abs(signal.entry.triggerLevel - signal.stopLoss);
  const baseRisk = signal.riskPerShare ?? derivedRisk;
  const riskPerShare = Math.max(0.01, baseRisk);

  const timeframeMinutes = signal.timeframeMinutes ?? 30;
  const timerExpiry = new Date(Date.now() + timeframeMinutes * 60 * 1000).toISOString();
  const initialNextAction =
    signal.strategy === 'leaps'
      ? `Prepare OTM hedge once entry confirms above ${signal.entry.triggerLevel.toFixed(2)}.`
      : `Watch for ${signal.entry.triggerType} confirmation at ${signal.entry.triggerLevel.toFixed(2)}.`;

  return {
    id: signal.id,
    symbol: signal.symbol,
    direction: signal.direction,
    strategy: signal.strategy,
    status: 'watching',
    positionSize: signal.positionSize,
    triggerLevel: signal.entry.triggerLevel,
    entryPrice: signal.entry.price,
    stopLoss: signal.stopLoss,
    firstTarget: signal.firstTarget,
    secondaryTarget: signal.secondaryTarget,
    riskPerShare,
    rMultipleAchieved: 0,
    timeframeMinutes,
    timerExpiry,
    lastUpdated: nowISO,
    addOnDone: false,
    exits: [],
    history: [
      {
        timestamp: nowISO,
        status: 'watching',
        note:
          signal.strategy === 'leaps'
            ? `LEAPS signal generated – plan protective hedge once price clears ${signal.entry.triggerLevel.toFixed(2)}.`
            : `Signal generated (${signal.entry.triggerType}) – monitoring for trigger through ${signal.entry.triggerLevel.toFixed(
                2,
              )}.`,
      },
    ],
    nextAction: initialNextAction,
    whaleConfirmation: signal.whaleConfirmation ?? null,
    hedgeActive: false,
    hedgeNote: signal.strategy === 'leaps' ? 'Awaiting entry before engaging hedge.' : undefined,
  };
}

function updateWatchingTrade(
  trade: RegimeTradeLifecycle,
  price: number,
  stage2: RegimeStage2Summary,
  nowISO: string,
) {
  const triggerMet =
    trade.direction === 'long'
      ? price >= trade.triggerLevel
      : price <= trade.triggerLevel;

  if (triggerMet) {
    const conflictingFlip =
      (trade.direction === 'long' && stage2.regimeTransition === 'flip_to_pinning') ||
      (trade.direction === 'short' && stage2.regimeTransition === 'flip_to_expansion');

    if (!conflictingFlip) {
      trade.enteredAt = nowISO;
      trade.rMultipleAchieved = 0;
      trade.nextAction =
        trade.strategy === 'leaps'
          ? 'Initiate OTM hedge on first pullback; manage core position to +1R.'
          : 'Manage risk – move to +1R for scale-out.';
      recordHistory(trade, 'entered', `Trigger hit at ${price.toFixed(2)}, position opened.`, nowISO);
      return;
    }

    recordHistory(
      trade,
      'cancelled',
      `Trigger hit at ${price.toFixed(2)} but conflicting regime flip (${stage2.regimeTransition}).`,
      nowISO,
    );
    trade.nextAction = 'Await next qualifying signal.';
    return;
  }

  if (trade.timerExpiry && Date.now() > new Date(trade.timerExpiry).getTime()) {
    recordHistory(
      trade,
      'expired',
      'Setup timed out before trigger confirmation.',
      nowISO,
    );
    trade.nextAction = 'Expired – remove from watchlist.';
    return;
  }
}

function updateActiveTrade(
  trade: RegimeTradeLifecycle,
  price: number,
  stage2: RegimeStage2Summary,
  stage3: RegimeStage3Profile,
  nowISO: string,
) {
  const r = rMultiple(trade, price);
  trade.rMultipleAchieved = r;

  const stopHit =
    trade.direction === 'long' ? price <= trade.stopLoss : price >= trade.stopLoss;

  if (stopHit) {
    trade.exits.push({ type: 'stop', price, timestamp: nowISO });
    recordHistory(
      trade,
      'stopped',
      `Stop-loss triggered at ${price.toFixed(2)}.`,
      nowISO,
    );
    trade.nextAction = 'Stopped out – reassess structure for re-entry.';
    return;
  }

  if (!trade.addOnDone && stage3.priceInteraction === 'inside_range' && Math.abs(r) > 0.3 && Math.abs(r) < 0.7) {
    trade.addOnDone = true;
    trade.nextAction = 'Add-on executed on shallow pullback.';
    trade.history.push({
      timestamp: nowISO,
      status: trade.status,
      note: 'Add-on completed after shallow pullback (<0.5R).',
    });
  }

  if (r >= 1 && (!trade.scaledAt || trade.status === 'entered')) {
    trade.scaledAt = nowISO;
    trade.exits.push({ type: 'target', price, timestamp: nowISO });
    recordHistory(
      trade,
      'scaled',
      `Scaled at +1R (price ${price.toFixed(2)}). Stops to breakeven.`,
      nowISO,
    );
    trade.nextAction =
      trade.strategy === 'leaps'
        ? 'Hold runner and maintain hedge; target secondary level or roll hedge on strength.'
        : 'Hold runner – target secondary level or +1.5R.';
    trade.stopLoss = trade.entryPrice;
    return;
  }

  const targetReached =
    trade.secondaryTarget !== undefined
      ? trade.direction === 'long'
        ? price >= trade.secondaryTarget
        : price <= trade.secondaryTarget
      : r >= 1.5;

  if (targetReached) {
    trade.exits.push({ type: 'target', price, timestamp: nowISO });
    recordHistory(
      trade,
      'target_hit',
      `Final target achieved at ${price.toFixed(2)}.`,
      nowISO,
    );
    trade.nextAction = 'Position closed – monitor for fresh setup.';
    return;
  }

  if (trade.enteredAt) {
    const elapsedMinutes = (Date.now() - new Date(trade.enteredAt).getTime()) / 60000;
    if (elapsedMinutes > trade.timeframeMinutes && r < 1) {
      trade.exits.push({ type: 'time', price, timestamp: nowISO });
      recordHistory(
        trade,
        'expired',
        `Time-based exit after ${Math.round(elapsedMinutes)} minutes without progress.`,
        nowISO,
      );
      trade.nextAction =
        trade.strategy === 'leaps'
          ? 'Time stop hit – consider resetting hedge and reassessing thesis.'
          : 'Time stop hit – reassess.';
    }
  }
}

export function manageLifecycle({
  mode,
  symbol,
  price,
  stage2,
  stage3,
  signals,
}: ManageLifecycleParams): RegimeTradeLifecycle[] {
  const cache = loadCache(mode);
  const nowISO = new Date().toISOString();
  const trades = cache[symbol] ? [...cache[symbol]] : [];

  signals.forEach(signal => {
    if (!trades.find(trade => trade.id === signal.id)) {
      trades.push(createLifecycleFromSignal(signal, nowISO));
    }
  });

  trades.forEach(trade => {
    if (trade.status === 'watching') {
      updateWatchingTrade(trade, price, stage2, nowISO);
    } else if (trade.status === 'entered' || trade.status === 'scaled') {
      updateActiveTrade(trade, price, stage2, stage3, nowISO);
    }
  });

  cache[symbol] = trades
    .sort((a, b) => new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime())
    .slice(0, 10);

  persistCache(mode, cache);
  const key = contextKey(mode, symbol);
  if (hasActiveTrades(cache[symbol])) {
    lifecycleContext.set(key, { stage2, stage3 });
    subscribeSymbolForMode(symbol.toUpperCase(), mode);
  } else {
    lifecycleContext.delete(key);
    unsubscribeSymbolForMode(symbol.toUpperCase(), mode);
  }

  broadcastLifecycle(mode, symbol, stage2, stage3, cache[symbol]);

  return cache[symbol];
}

export function loadAllActiveTrades(mode: 'scalp' | 'swing' | 'leaps'): RegimeTradeLifecycle[] {
  const cache = loadCache(mode);
  return Object.values(cache)
    .flat()
    .filter(trade => !['stopped', 'target_hit', 'expired', 'cancelled'].includes(trade.status))
    .sort((a, b) => new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime());
}

export function handlePriceTick(
  mode: LifecycleMode,
  symbol: string,
  price: number,
): RegimeTradeLifecycle[] | null {
  const cache = loadCache(mode);
  const trades = cache[symbol];
  if (!trades || trades.length === 0) {
    return null;
  }

  const context = lifecycleContext.get(contextKey(mode, symbol));
  if (!context) {
    return null;
  }

  const nowISO = new Date().toISOString();

  trades.forEach(trade => {
    if (trade.status === 'watching') {
      updateWatchingTrade(trade, price, context.stage2, nowISO);
    } else if (trade.status === 'entered' || trade.status === 'scaled') {
      updateActiveTrade(trade, price, context.stage2, context.stage3, nowISO);
    }
  });

  cache[symbol] = trades
    .sort((a, b) => new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime())
    .slice(0, 10);

  persistCache(mode, cache);

  if (hasActiveTrades(cache[symbol])) {
    subscribeSymbolForMode(symbol.toUpperCase(), mode);
  } else {
    lifecycleContext.delete(contextKey(mode, symbol));
    unsubscribeSymbolForMode(symbol.toUpperCase(), mode);
  }

  if (context) {
    broadcastLifecycle(mode, symbol, context.stage2, context.stage3, cache[symbol]);
  }

  return cache[symbol];
}

export function getLifecycleSnapshot(
  mode: LifecycleMode,
  symbol: string,
): { trades: RegimeTradeLifecycle[]; stage2?: RegimeStage2Summary; stage3?: RegimeStage3Profile } {
  const cache = loadCache(mode);
  const trades = cache[symbol] || [];
  const context = lifecycleContext.get(contextKey(mode, symbol));
  return {
    trades,
    stage2: context?.stage2,
    stage3: context?.stage3,
  };
}
