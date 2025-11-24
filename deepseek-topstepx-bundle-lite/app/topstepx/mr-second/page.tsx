'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { IChartApi, ISeriesApi, UTCTimestamp } from 'lightweight-charts';
import { createChart, ColorType, CandlestickSeries, LineSeries } from 'lightweight-charts';

interface StrategyLogLine {
  timestamp: string;
  type: 'stdout' | 'stderr';
  line: string;
}

interface StrategyProcessStatus {
  running: boolean;
  pid?: number;
  startedAt?: string;
  lastExitAt?: string;
  exitCode?: number | null;
  exitSignal?: string | null;
  logs: StrategyLogLine[];
  accountId?: number | null;
}

interface MrTrade {
  tradeId: string;
  side: 'long' | 'short';
  entryPrice: number;
  entryTime: string;
  qty: number;
  stopLoss?: number | null;
  target?: number | null;
  scalePrice?: number | null;
  scalePnL?: number | null;
  exitPrice?: number | null;
  exitTime?: string;
  exitReason?: string;
  pnl?: number | null;
  status: 'open' | 'closed';
}

interface StatusResponse {
  status: StrategyProcessStatus;
  trades: { openTrade: MrTrade | null; closedTrades: MrTrade[] };
  symbol: string;
  contractId: string | null;
  multiplier: number;
}

interface TopstepAccount {
  id: number;
  name: string;
  balance: number;
  canTrade: boolean;
  isVisible: boolean;
}

interface ChartCandle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  sma?: number | null;
  rsi?: number | null;
  adx?: number | null;
  bbUpper?: number | null;
  bbLower?: number | null;
  bbBasis?: number | null;
  kcUpper?: number | null;
  kcLower?: number | null;
  ttmSqueezeOn?: boolean | null;
  ttmSqueezeOff?: boolean | null;
  ttmMomentum?: number | null;
  ttmSentiment?: string | null;
}

interface TopstepLiveSnapshot {
  accountId: number;
  account?: Record<string, any> | null;
  positions: TopstepPosition[];
  lastUpdate?: string;
}

interface TopstepPosition {
  key: string;
  symbol: string;
  netQty: number;
  avgPrice: number;
  contractId?: string;
  lastUpdate: string;
}

const MAX_CANDLES = 3600;

const formatCurrency = (value: number | null | undefined) => {
  if (value == null || Number.isNaN(value)) return '$0.00';
  const sign = value >= 0 ? '' : '-';
  return `${sign}$${Math.abs(value).toFixed(2)}`;
};

const formatDate = (value?: string) => {
  if (!value) return '—';
  return new Date(value).toLocaleString();
};

const pickNumericField = (source: Record<string, any> | null | undefined, keys: string[]) => {
  if (!source) return null;
  for (const key of keys) {
    const value = source[key];
    if (value == null) continue;
    const num = typeof value === 'number' ? value : Number(value);
    if (Number.isFinite(num)) {
      return num;
    }
  }
  return null;
};

export default function TopstepxMrSecondDashboard() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [price, setPrice] = useState<number | null>(null);
  const [priceTimestamp, setPriceTimestamp] = useState<string | null>(null);
  const [chartCandles, setChartCandles] = useState<ChartCandle[]>([]);
  const [loadingAction, setLoadingAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [accounts, setAccounts] = useState<TopstepAccount[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState<number | null>(null);
  const [topstepLiveSnapshot, setTopstepLiveSnapshot] = useState<TopstepLiveSnapshot | null>(null);
  const [indicatorValues, setIndicatorValues] = useState<{
    sma?: number | null;
    rsi?: number | null;
    adx?: number | null;
    bbUpper?: number | null;
    bbLower?: number | null;
    bbBasis?: number | null;
    ttmMomentum?: number | null;
    ttmSqueezeOn?: boolean | null;
    ttmSqueezeOff?: boolean | null;
    ttmSentiment?: string | null;
  } | null>(null);

  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const smaSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const bbUpperSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const bbLowerSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  const symbol = status?.symbol ?? 'MES';
  const multiplier = status?.multiplier ?? 50;
  const openTrade = status?.trades.openTrade ?? null;
  const closedTrades = status?.trades.closedTrades ?? [];
  const runningAccountId = status?.status.accountId ?? null;
  const selectedAccount = useMemo(
    () => accounts.find(acc => acc.id === selectedAccountId) ?? null,
    [accounts, selectedAccountId],
  );
  useEffect(() => {
    if (runningAccountId && !selectedAccountId) {
      setSelectedAccountId(runningAccountId);
    }
  }, [runningAccountId, selectedAccountId]);

  const mergedAccount = useMemo(() => {
    if (topstepLiveSnapshot?.account) {
      return topstepLiveSnapshot.account;
    }
    if (selectedAccount) {
      return {
        balance: selectedAccount.balance,
        buyingPower: selectedAccount.balance,
        name: selectedAccount.name,
      } as Record<string, any>;
    }
    return null;
  }, [topstepLiveSnapshot?.account, selectedAccount]);

  const unrealizedPnl = useMemo(() => {
    if (!openTrade || price == null) return null;
    const direction = openTrade.side === 'long' ? 1 : -1;
    return (price - openTrade.entryPrice) * direction * multiplier * Math.max(openTrade.qty, 1);
  }, [openTrade, price, multiplier]);

  const loadStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/topstepx/mr-second/status');
      if (!response.ok) {
        throw new Error('Failed to load strategy status');
      }
      const payload: StatusResponse = await response.json();
      setStatus(payload);
      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load status');
    }
  }, []);

  const loadPrice = useCallback(async () => {
    try {
      const response = await fetch('/api/topstepx/mr-second/price');
      if (!response.ok) return;
      const payload = await response.json();
      setPrice(payload.price);
      setPriceTimestamp(payload.timestamp);
    } catch (err) {
      console.error('Failed to fetch latest price', err);
    }
  }, []);

  const loadTopstepAccounts = useCallback(async () => {
    try {
      const response = await fetch('/api/topstepx/accounts');
      if (!response.ok) {
        throw new Error('Failed to load TopstepX accounts');
      }
      const payload = await response.json();
      const fetched: TopstepAccount[] = payload.accounts || [];
      setAccounts(fetched);
      if (!selectedAccountId && fetched.length) {
        setSelectedAccountId(fetched[0].id);
      }
    } catch (err) {
      console.error('Failed to fetch TopstepX accounts', err);
    }
  }, [selectedAccountId]);

  const runAction = useCallback(async (action: 'start' | 'stop' | 'flatten') => {
    setLoadingAction(action);
    try {
      if (action === 'start' && !selectedAccountId) {
        throw new Error('Select a TopstepX account before starting the strategy.');
      }
      const response = await fetch('/api/topstepx/mr-second/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, accountId: selectedAccountId ?? undefined }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload?.message || payload?.error || 'Control action failed');
      }
      await loadStatus();
    } catch (err: any) {
      setError(err?.message || 'Control action failed');
    } finally {
      setLoadingAction(null);
    }
  }, [loadStatus, selectedAccountId]);

  const loadTopstepAccountLive = useCallback(async () => {
    if (!selectedAccountId) {
      setTopstepLiveSnapshot(null);
      return;
    }
    try {
      const response = await fetch(`/api/topstepx/account/live?id=${selectedAccountId}`);
      const payload: TopstepLiveSnapshot | { error?: string } = await response
        .json()
        .catch(() => ({} as any));
      if (!response.ok) {
        console.warn('TopstepX live account unavailable:', (payload as any)?.error);
        setTopstepLiveSnapshot(null);
        return;
      }
      setTopstepLiveSnapshot(payload as TopstepLiveSnapshot);
    } catch (err) {
      console.error('Failed to fetch TopstepX live snapshot', err);
      setTopstepLiveSnapshot(null);
    }
  }, [selectedAccountId]);

  useEffect(() => {
    loadStatus();
    loadPrice();
    loadTopstepAccounts();
    const statusInterval = setInterval(loadStatus, 5000);
    const priceInterval = setInterval(loadPrice, 2000);
    const topstepInterval = setInterval(loadTopstepAccounts, 60000);
    return () => {
      clearInterval(statusInterval);
      clearInterval(priceInterval);
      clearInterval(topstepInterval);
    };
  }, [loadStatus, loadPrice, loadTopstepAccounts]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch('/api/topstepx/mr-second/bars?seconds=3600');
        if (!response.ok) return;
        const payload = await response.json();
        const candles: ChartCandle[] = payload?.candles || [];
        if (!candles.length) return;
        const initial = candles
          .slice(-MAX_CANDLES)
          .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
        if (!cancelled) {
          setChartCandles(initial);
        }
      } catch (err) {
        console.error('Failed to bootstrap bars', err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedAccountId) {
      setTopstepLiveSnapshot(null);
      return;
    }

    let disposed = false;
    let source: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const applySnapshot = (payload: any) => {
      if (!payload) return;
      setTopstepLiveSnapshot(prev => {
        const nextPositions = Array.isArray(payload.positions)
          ? payload.positions
          : prev?.positions ?? [];
        return {
          accountId: payload.accountId ?? prev?.accountId ?? selectedAccountId,
          account: payload.account ?? prev?.account ?? null,
          positions: nextPositions,
          lastUpdate: payload.lastUpdate ?? payload.timestamp ?? new Date().toISOString(),
        };
      });
    };

    const connect = () => {
      if (disposed) {
        return;
      }
      if (source) {
        source.close();
        source = null;
      }
      source = new EventSource(`/api/topstepx/account/stream?id=${selectedAccountId}`);
      source.onmessage = event => {
        try {
          const payload = JSON.parse(event.data);
          applySnapshot(payload);
        } catch (err) {
          console.error('TopstepX account stream parse error', err);
        }
      };
      source.onerror = err => {
        console.error('TopstepX account stream error', err);
        if (source) {
          source.close();
          source = null;
        }
        if (!disposed) {
          reconnectTimer = setTimeout(connect, 2000);
        }
      };
    };

    loadTopstepAccountLive();
    connect();

    return () => {
      disposed = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      if (source) {
        source.close();
        source = null;
      }
    };
  }, [selectedAccountId, loadTopstepAccountLive]);

  useEffect(() => {
    if (!chartContainerRef.current || chartRef.current) {
      return;
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0f172a' },
        textColor: '#e2e8f0',
      },
      width: chartContainerRef.current.clientWidth,
      height: 420,
      grid: {
        vertLines: { color: 'rgba(14,23,38,0.4)' },
        horzLines: { color: 'rgba(14,23,38,0.4)' },
      },
      timeScale: { borderColor: '#1e293b' },
      rightPriceScale: { borderColor: '#1e293b' },
      crosshair: { mode: 1 },
    });

    chartRef.current = chart;
    candleSeriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#dc2626',
      borderUpColor: '#16a34a',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    });
    smaSeriesRef.current = chart.addSeries(LineSeries, {
      color: '#fbbf24',
      lineWidth: 2,
      priceLineVisible: false,
    });
    bbUpperSeriesRef.current = chart.addSeries(LineSeries, {
      color: '#38bdf8',
      lineWidth: 1,
      priceLineVisible: false,
    });
    bbLowerSeriesRef.current = chart.addSeries(LineSeries, {
      color: '#f472b6',
      lineWidth: 1,
      priceLineVisible: false,
    });

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      smaSeriesRef.current = null;
      bbUpperSeriesRef.current = null;
      bbLowerSeriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!chartCandles.length || !candleSeriesRef.current) return;
    const data = chartCandles.map(candle => ({
      time: (new Date(candle.time).getTime() / 1000) as UTCTimestamp,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }));
    candleSeriesRef.current.setData(data);
    if (smaSeriesRef.current) {
      const smaData = chartCandles
        .filter(candle => typeof candle.sma === 'number')
        .map(candle => ({
          time: (new Date(candle.time).getTime() / 1000) as UTCTimestamp,
          value: candle.sma as number,
        }));
      if (smaData.length) {
        smaSeriesRef.current.setData(smaData);
      }
    }
    if (bbUpperSeriesRef.current) {
      const bbUpperData = chartCandles
        .filter(candle => typeof candle.bbUpper === 'number')
        .map(candle => ({
          time: (new Date(candle.time).getTime() / 1000) as UTCTimestamp,
          value: candle.bbUpper as number,
        }));
      if (bbUpperData.length) {
        bbUpperSeriesRef.current.setData(bbUpperData);
      }
    }
    if (bbLowerSeriesRef.current) {
      const bbLowerData = chartCandles
        .filter(candle => typeof candle.bbLower === 'number')
        .map(candle => ({
          time: (new Date(candle.time).getTime() / 1000) as UTCTimestamp,
          value: candle.bbLower as number,
        }));
      if (bbLowerData.length) {
        bbLowerSeriesRef.current.setData(bbLowerData);
      }
    }
  }, [chartCandles]);

  useEffect(() => {
    const source = new EventSource('/api/topstepx/mr-second/stream');
    source.onmessage = event => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'snapshot') {
          if (Array.isArray(payload.candles)) {
            const recent = payload.candles
              .slice(-MAX_CANDLES)
              .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
            setChartCandles(recent);
            const last = recent[recent.length - 1];
            if (last) {
              setIndicatorValues({
                sma: last.sma ?? null,
                rsi: last.rsi ?? null,
                adx: last.adx ?? null,
                bbUpper: last.bbUpper ?? null,
                bbLower: last.bbLower ?? null,
                bbBasis: last.bbBasis ?? null,
                ttmMomentum: last.ttmMomentum ?? null,
                ttmSqueezeOn: last.ttmSqueezeOn ?? null,
                ttmSqueezeOff: last.ttmSqueezeOff ?? null,
                ttmSentiment: last.ttmSentiment ?? null,
              });
            }
          }
          if (typeof payload.price === 'number') {
            setPrice(payload.price);
          }
          if (payload.timestamp) {
            setPriceTimestamp(payload.timestamp);
          }
        } else if (payload.type === 'candle' && payload.candle) {
          const candle: ChartCandle = payload.candle;
          setChartCandles(prev => {
            const filtered = prev.filter(item => item.time !== candle.time);
            const updated = [...filtered, candle].sort(
              (a, b) => new Date(a.time).getTime() - new Date(b.time).getTime(),
            );
            return updated.slice(-MAX_CANDLES);
          });
          setIndicatorValues({
            sma: candle.sma ?? null,
            rsi: candle.rsi ?? null,
            adx: candle.adx ?? null,
            bbUpper: candle.bbUpper ?? null,
            bbLower: candle.bbLower ?? null,
            bbBasis: candle.bbBasis ?? null,
            ttmMomentum: candle.ttmMomentum ?? null,
            ttmSqueezeOn: candle.ttmSqueezeOn ?? null,
            ttmSqueezeOff: candle.ttmSqueezeOff ?? null,
            ttmSentiment: candle.ttmSentiment ?? null,
          });
          setPrice(candle.close);
          setPriceTimestamp(payload.timestamp);
        } else if (payload.type === 'tick') {
          if (typeof payload.price === 'number') {
            setPrice(payload.price);
          }
          if (payload.timestamp) {
            setPriceTimestamp(payload.timestamp);
          }
        }
      } catch (err) {
        console.error('Failed to parse feed payload', err);
      }
    };
    source.onerror = () => {
      console.warn('TopstepX stream disconnected, attempting to reconnect...');
    };
    return () => {
      source.close();
    };
  }, []);

  const liveAccount = topstepLiveSnapshot?.account ?? null;
  const accountBalanceValue =
    pickNumericField(mergedAccount, [
      'netLiquidationValue',
      'netValue',
      'balance',
      'totalCash',
      'cashValue',
    ]) ?? (selectedAccount ? Number(selectedAccount.balance) : null);
  const buyingPowerValue =
    pickNumericField(mergedAccount, ['buyingPower', 'buyPower', 'availableFunds', 'availFunds']) ??
    (selectedAccount ? Number(selectedAccount.balance) : null);
  const openPnlValue = pickNumericField(mergedAccount, [
    'unrealizedPnL',
    'openProfit',
    'unrealizedProfit',
    'openPnL',
  ]);
  const displayAccountName =
    (typeof (mergedAccount as any)?.name === 'string' && (mergedAccount as any).name) ||
    selectedAccount?.name ||
    null;
  const topstepPositions = topstepLiveSnapshot?.positions ?? [];
  const fallbackPosition = useMemo(() => {
    if (openTrade) {
      return null;
    }
    return topstepPositions.find(position => position.netQty !== 0) ?? null;
  }, [openTrade, topstepPositions]);

  const renderPositionSummary = () => {
    if (openTrade) {
      return (
        <>
          <div className="flex justify-between">
            <span className="text-slate-400">Direction</span>
            <span className="font-semibold uppercase">{openTrade.side}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Contracts</span>
            <span className="font-semibold">{openTrade.qty}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Entry Price</span>
            <span className="font-semibold">${openTrade.entryPrice.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Entry Time</span>
            <span className="font-semibold">{new Date(openTrade.entryTime).toLocaleTimeString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Target</span>
            <span className="font-semibold">
              {openTrade.target != null ? `$${openTrade.target.toFixed(2)}` : '—'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Stop</span>
            <span className="font-semibold">
              {openTrade.stopLoss != null ? `$${openTrade.stopLoss.toFixed(2)}` : '—'}
            </span>
          </div>
        </>
      );
    }

    if (fallbackPosition) {
      return (
        <>
          <div className="flex justify-between">
            <span className="text-slate-400">Symbol</span>
            <span className="font-semibold uppercase">{fallbackPosition.symbol}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Direction</span>
            <span className="font-semibold uppercase">
              {fallbackPosition.netQty > 0 ? 'LONG' : 'SHORT'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Contracts</span>
            <span className="font-semibold">{Math.abs(fallbackPosition.netQty)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Avg Price</span>
            <span className="font-semibold">${fallbackPosition.avgPrice.toFixed(2)}</span>
          </div>
        </>
      );
    }

    return <p className="text-slate-400 text-sm">No open position.</p>;
  };

  const statusColor = status?.status.running ? 'text-emerald-400' : 'text-rose-400';
  const statusLabel = status?.status.running ? 'Running' : 'Stopped';

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-6 space-y-6">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold">TopstepX Mean Reversion (1s) Dashboard</h1>
        <p className="text-slate-400">
          Live controls and telemetry for the Bollinger + RSI + TTM Squeeze scalper ({symbol})
        </p>
        <div className="flex flex-wrap gap-3 items-center text-sm">
          <span className={`font-semibold ${statusColor}`}>{statusLabel}</span>
          {status?.status.pid && <span>PID {status.status.pid}</span>}
          {status?.status.startedAt && (
            <span>Started {new Date(status.status.startedAt).toLocaleTimeString()}</span>
          )}
          {runningAccountId && <span>Running account #{runningAccountId}</span>}
          {price != null && (
            <span>
              Last trade {price.toFixed(2)} ({priceTimestamp ? new Date(priceTimestamp).toLocaleTimeString() : '—'})
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <label className="text-slate-400">Account:</label>
          <select
            value={selectedAccountId ?? ''}
            onChange={event => {
              const next = Number(event.target.value);
              setSelectedAccountId(Number.isFinite(next) ? next : null);
            }}
            className="bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"
            disabled={!accounts.length}
          >
            {accounts.length === 0 && <option value="">Loading...</option>}
            {accounts.map(account => (
              <option key={account.id} value={account.id}>
                #{account.id} • {account.name}
              </option>
            ))}
          </select>
        </div>
        {selectedAccountId && (
          <p className="text-xs text-slate-500">
            Start/flatten actions will trade TopstepX account #{selectedAccountId}.
          </p>
        )}
        {error && <p className="text-rose-400 text-sm">{error}</p>}
      </header>

      <section className="bg-slate-800 rounded-xl p-4 shadow-xl relative">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-lg font-semibold">TradingView Chart</h2>
            <p className="text-slate-400 text-sm">Lightweight Charts rendering last hour of 1-second bars</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => runAction('start')}
              disabled={
                loadingAction === 'start' ||
                status?.status.running ||
                !selectedAccountId
              }
              className="px-4 py-2 rounded bg-emerald-500 hover:bg-emerald-600 disabled:bg-emerald-900 text-sm font-semibold"
            >
              {loadingAction === 'start' ? 'Starting...' : 'Start Strategy'}
            </button>
            <button
              onClick={() => runAction('stop')}
              disabled={loadingAction === 'stop' || !status?.status.running}
              className="px-4 py-2 rounded bg-rose-500 hover:bg-rose-600 disabled:bg-rose-900 text-sm font-semibold"
            >
              {loadingAction === 'stop' ? 'Stopping...' : 'Stop Strategy'}
            </button>
            <button
              onClick={() => runAction('flatten')}
              disabled={loadingAction === 'flatten' || !status?.status.running}
              className="px-4 py-2 rounded bg-amber-500 hover:bg-amber-600 disabled:bg-amber-900 text-sm font-semibold"
            >
              {loadingAction === 'flatten' ? 'Flattening...' : 'Flatten Position'}
            </button>
          </div>
        </div>
        <div ref={chartContainerRef} className="w-full h-[420px]" />
      </section>

            <section className="grid grid-cols-1 xl:grid-cols-4 gap-5">
        <div className="bg-slate-800 rounded-xl p-4 shadow space-y-3">
          <h3 className="text-lg font-semibold">Topstep Account</h3>
          {selectedAccount || mergedAccount ? (
            <>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Account ID</span>
                <span className="font-semibold">#{selectedAccount?.id ?? mergedAccount?.accountId ?? '—'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Name</span>
                <span className="font-semibold">{displayAccountName ?? '—'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Live Balance</span>
                <span className="font-semibold">
                  {accountBalanceValue != null ? formatCurrency(accountBalanceValue) : '—'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Buying Power</span>
                <span className="font-semibold">
                  {buyingPowerValue != null ? formatCurrency(buyingPowerValue) : '—'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Open P&amp;L</span>
                <span className={`font-semibold ${openPnlValue && openPnlValue >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {openPnlValue != null ? formatCurrency(openPnlValue) : '—'}
                </span>
              </div>
              <div className="flex justify-between text-xs text-slate-500">
                <span>Tradable</span>
                <span className={selectedAccount?.canTrade ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
                  {selectedAccount?.canTrade ? 'Yes' : 'No'}
                </span>
              </div>
              <div className="text-xs text-slate-500">
                Last update: {topstepLiveSnapshot?.lastUpdate ? formatDate(topstepLiveSnapshot.lastUpdate) : '—'}
              </div>
            </>
          ) : (
            <p className="text-slate-400 text-sm">Select an account to view balances.</p>
          )}
        </div>

        <div className="bg-slate-800 rounded-xl p-4 shadow space-y-2">
          <h3 className="text-lg font-semibold mb-2">Open Position</h3>
          <div className="space-y-2">{renderPositionSummary()}</div>
          <div className="flex justify-between pt-2">
            <span className="text-slate-400">Current Price</span>
            <span className="font-semibold">{price ? `$${price.toFixed(2)}` : '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Unrealized P&amp;L</span>
            <span className={`font-semibold ${unrealizedPnl && unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {unrealizedPnl != null ? formatCurrency(unrealizedPnl) : '—'}
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-3 mt-3 border-t border-slate-700 text-center text-xs uppercase tracking-wide text-slate-400">
            <div>
              <p className="text-slate-500">SMA</p>
              <p className="text-base text-slate-100">
                {indicatorValues?.sma != null ? indicatorValues.sma.toFixed(2) : '—'}
              </p>
            </div>
            <div>
              <p className="text-slate-500">BB Upper</p>
              <p className="text-base text-slate-100">
                {indicatorValues?.bbUpper != null ? indicatorValues.bbUpper.toFixed(2) : '—'}
              </p>
            </div>
            <div>
              <p className="text-slate-500">BB Lower</p>
              <p className="text-base text-slate-100">
                {indicatorValues?.bbLower != null ? indicatorValues.bbLower.toFixed(2) : '—'}
              </p>
            </div>
            <div>
              <p className="text-slate-500">RSI</p>
              <p className="text-base text-slate-100">
                {indicatorValues?.rsi != null ? indicatorValues.rsi.toFixed(1) : '—'}
              </p>
            </div>
            <div>
              <p className="text-slate-500">ADX</p>
              <p className="text-base text-slate-100">
                {indicatorValues?.adx != null ? indicatorValues.adx.toFixed(1) : '—'}
              </p>
            </div>
            <div>
              <p className="text-slate-500">TTM Squeeze</p>
              <p className={`text-base ${indicatorValues?.ttmSqueezeOn ? 'text-amber-300' : 'text-slate-100'}`}>
                {indicatorValues?.ttmSqueezeOn
                  ? 'ON'
                  : indicatorValues?.ttmSqueezeOff
                    ? 'OFF'
                    : '—'}
              </p>
            </div>
            <div>
              <p className="text-slate-500">Momentum</p>
              <p className={`text-base ${Number(indicatorValues?.ttmMomentum ?? 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                {indicatorValues?.ttmMomentum != null ? indicatorValues.ttmMomentum.toFixed(4) : '—'}
              </p>
            </div>
            <div>
              <p className="text-slate-500">Sentiment</p>
              <p className="text-base text-slate-100">
                {indicatorValues?.ttmSentiment
                  ? indicatorValues.ttmSentiment.toUpperCase()
                  : '—'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-4 shadow overflow-hidden">
          <h3 className="text-lg font-semibold mb-3">Live Positions</h3>
          <div className="max-h-52 overflow-auto text-sm">
            {topstepPositions.length ? (
              <table className="w-full text-left text-xs">
                <thead className="text-slate-400 border-b border-slate-700">
                  <tr>
                    <th className="py-1">Symbol</th>
                    <th className="py-1">Qty</th>
                    <th className="py-1">Avg Price</th>
                  </tr>
                </thead>
                <tbody>
                  {topstepPositions.slice(0, 6).map(position => (
                    <tr key={position.key} className="border-b border-slate-800 last:border-none">
                      <td className="py-1">{position.symbol}</td>
                      <td className="py-1">{position.netQty}</td>
                      <td className="py-1">${position.avgPrice.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-slate-400 text-sm">No active positions.</p>
            )}
          </div>
        </div>

        <div className="bg-slate-800 rounded-xl p-4 shadow">
          <h3 className="text-lg font-semibold mb-3">Recent Logs</h3>
          <div className="space-y-2 max-h-48 overflow-auto text-xs font-mono">
            {status?.status.logs.slice(-8).map((logLine, idx) => (
              <div key={`${logLine.timestamp}-${idx}`} className="text-slate-400">
                <span className="text-slate-500 mr-2">{new Date(logLine.timestamp).toLocaleTimeString()}</span>
                <span className={logLine.type === 'stderr' ? 'text-rose-300' : 'text-slate-200'}>
                  {logLine.line}
                </span>
              </div>
            ))}
            {!status?.status.logs?.length && <p className="text-slate-500">No logs yet.</p>}
          </div>
        </div>
      </section>

      <section className="bg-slate-800 rounded-xl p-4 shadow">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h3 className="text-lg font-semibold">Closed Trades</h3>
            <p className="text-slate-400 text-sm">Latest exits with realized P&amp;L</p>
          </div>
        </div>
        <div className="overflow-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-slate-400 border-b border-slate-700">
              <tr>
                <th className="py-2">Trade ID</th>
                <th>Side</th>
                <th>Qty</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>Reason</th>
                <th>P&amp;L</th>
              </tr>
            </thead>
            <tbody>
              {closedTrades.slice(0, 20).map(trade => (
                <tr key={trade.tradeId} className="border-b border-slate-800 last:border-none">
                  <td className="py-2">{trade.tradeId}</td>
                  <td className="uppercase">{trade.side}</td>
                  <td>{trade.qty}</td>
                  <td>
                    ${trade.entryPrice.toFixed(2)}
                    <div className="text-xs text-slate-500">{formatDate(trade.entryTime)}</div>
                  </td>
                  <td>
                    {trade.exitPrice ? `$${trade.exitPrice.toFixed(2)}` : '—'}
                    <div className="text-xs text-slate-500">{formatDate(trade.exitTime)}</div>
                  </td>
                  <td className="capitalize">{trade.exitReason ?? '—'}</td>
                  <td className={trade.pnl != null && trade.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                    {trade.pnl != null ? formatCurrency(trade.pnl) : '—'}
                  </td>
                </tr>
              ))}
              {!closedTrades.length && (
                <tr>
                  <td colSpan={7} className="text-center py-4 text-slate-500">
                    No completed trades yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
