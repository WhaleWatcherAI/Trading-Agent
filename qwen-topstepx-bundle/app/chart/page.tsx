'use client';

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type TouchEvent as ReactTouchEvent,
} from 'react';
import {
  createChart,
  CrosshairMode,
  LineStyle,
  CandlestickSeries,
  LineSeries,
  HistogramSeries
} from 'lightweight-charts';
import {
  VolatilityRegimeResponse,
  VolatilityRegimeAnalysis,
  RegimeStage1Result,
  RegimeGexLevel,
  RegimeTradeSignal,
  RegimeTradeStatus,
  RegimeTradeLifecycle,
  AccountSnapshot,
} from '@/types';

const sanitizeTicker = (value: string) =>
  value
    .toUpperCase()
    .replace(/[^A-Z0-9.\-]/g, '')
    .slice(0, 21);

export default function ChartPage() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const rsiSeriesRef = useRef<any>(null);
  const macdSeriesRef = useRef<any>(null);
  const macdSignalSeriesRef = useRef<any>(null);
  const macdHistogramSeriesRef = useRef<any>(null);

  const [symbol, setSymbol] = useState('TSLA');
  const [debouncedSymbol, setDebouncedSymbol] = useState('TSLA'); // Debounced version for fetching
  const [timeInterval, setTimeInterval] = useState('1min'); // Renamed to avoid conflict with window.setInterval
  const [days, setDays] = useState(2); // Set to 2 days to match backtest (48 hours)
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showVolume, setShowVolume] = useState(false);
  const [showRSI, setShowRSI] = useState(true);
  const [showMACD, setShowMACD] = useState(false);
  const [showSMA9, setShowSMA9] = useState(false);
  const [showSMA20, setShowSMA20] = useState(true); // Enable SMA20 by default for backtest
  const [showSMA50, setShowSMA50] = useState(false);
  const [showSMA200, setShowSMA200] = useState(false);
  const [showBB, setShowBB] = useState(false);
  const [showVolumeProfile, setShowVolumeProfile] = useState(true);
  const [showEMA, setShowEMA] = useState(false);
  const [showGEX, setShowGEX] = useState(false);
  const [gexMode, setGexMode] = useState<'intraday' | 'swing' | 'leaps'>('intraday');
  const [regimeMode, setRegimeMode] = useState<'scalp' | 'swing' | 'leaps'>('scalp');
const [regimeAnalysis, setRegimeAnalysis] = useState<VolatilityRegimeAnalysis | null>(null);
const [regimeLoading, setRegimeLoading] = useState(false);
const [regimeError, setRegimeError] = useState<string | null>(null);
const [activeTrades, setActiveTrades] = useState<RegimeTradeLifecycle[]>([]);
const [globalActiveTrades, setGlobalActiveTrades] = useState<RegimeTradeLifecycle[]>([]);
const [selectedTradeId, setSelectedTradeId] = useState<string | null>(null);
const [sessionProfiles, setSessionProfiles] = useState<any[] | null>(null);
  const [volumeConfluence, setVolumeConfluence] = useState<string[]>([]);
  const enableRegimeStream = useMemo(
    () => process.env.NEXT_PUBLIC_ENABLE_REGIME_STREAM === 'true',
    [],
  );
  const [accountSnapshot, setAccountSnapshot] = useState<AccountSnapshot | null>(null);
  const [accountError, setAccountError] = useState<string | null>(null);
  const accountPollRef = useRef<NodeJS.Timeout | null>(null);
  const [gexMeta, setGexMeta] = useState<{ expirations: { date: string; dte: number }[] } | null>(null);
  const [gexSummary, setGexSummary] = useState<{
    netGex: number;
    totalCallGex: number;
    totalPutGex: number;
    netGexPerDollar?: number;
    totalCallGexPerDollar?: number;
    totalPutGexPerDollar?: number;
  } | null>(null);
  const [showIndicatorPanel, setShowIndicatorPanel] = useState(false);
  const [chartHeight, setChartHeight] = useState(600);
  const [rsiHeight, setRsiHeight] = useState(200);
  const [macdHeight, setMacdHeight] = useState(200);

  const sma9Ref = useRef<any>(null);
  const sma20Ref = useRef<any>(null);
  const sma50Ref = useRef<any>(null);
  const sma200Ref = useRef<any>(null);
  const sma9BuySignalsRef = useRef<any>(null); // For buy markers
  const sma9SellSignalsRef = useRef<any>(null); // For sell markers
  const sma20BuySignalsRef = useRef<any>(null); // For SMA20 buy markers
  const sma20SellSignalsRef = useRef<any>(null); // For SMA20 sell markers

  // Backtest state
  const [backtestData, setBacktestData] = useState<any>(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [backtestError, setBacktestError] = useState<string | null>(null);
  const bbUpperRef = useRef<any>(null);
  const bbMiddleRef = useRef<any>(null);
  const bbLowerRef = useRef<any>(null);
  const ema12Ref = useRef<any>(null);
  const ema26Ref = useRef<any>(null);
  const volumeProfilePOCRef = useRef<any>(null);
  const volumeProfileVAHRef = useRef<any>(null);
  const volumeProfileVALRef = useRef<any>(null);

  const rsiChartRef = useRef<any>(null);
  const macdChartRef = useRef<any>(null);
  const rsiContainerRef = useRef<HTMLDivElement>(null);
  const macdContainerRef = useRef<HTMLDivElement>(null);
  const volumeProfileCanvasRef = useRef<HTMLCanvasElement>(null);
  const volumeProfileDataRef = useRef<any>(null);
  const candlestickDataRef = useRef<any[]>([]);
  const fetchRequestIdRef = useRef<number>(0);
  const gexCanvasRef = useRef<HTMLCanvasElement>(null);
  const gexDataRef = useRef<any>(null);
  const overlayAnimationFrameRef = useRef<number | null>(null);
  const lastVisibleRangeRef = useRef<any>(null);
  const redrawThrottleRef = useRef<number>(0);
  const indicatorCacheRef = useRef<Map<string, any>>(new Map());
  const tradePriceLinesRef = useRef<any[]>([]);
  const structuralLinesRef = useRef<any[]>([]);

  const selectedTrade = useMemo(
    () => activeTrades.find(trade => trade.id === selectedTradeId) || null,
    [activeTrades, selectedTradeId],
  );
  const handleTradeSelect = useCallback((id: string) => {
    setSelectedTradeId(id);
  }, []);
  const regimeModes: Array<'scalp' | 'swing' | 'leaps'> = ['scalp', 'swing', 'leaps'];
  const stage1 = regimeAnalysis?.stage1;
  const stage2 = regimeAnalysis?.stage2;
  const stage3 = regimeAnalysis?.stage3;
  const transitionLabel = stage2?.regimeTransition === 'flip_to_expansion'
    ? 'Flip to Expansion'
    : stage2?.regimeTransition === 'flip_to_pinning'
      ? 'Flip to Pinning'
      : 'Stable';
  const positionSummary = useMemo(() => {
    if (!accountSnapshot?.positions || accountSnapshot.positions.length === 0) {
      return { marketValue: 0, costBasis: 0, unrealized: 0, unrealizedPercent: 0 };
    }
    const marketValue = accountSnapshot.positions.reduce((sum, pos) => sum + pos.marketValue, 0);
    const costBasis = accountSnapshot.positions.reduce((sum, pos) => sum + pos.costBasis, 0);
    const unrealized = marketValue - costBasis;
    const unrealizedPercent = costBasis !== 0 ? unrealized / costBasis : 0;
    return { marketValue, costBasis, unrealized, unrealizedPercent };
  }, [accountSnapshot]);

  const callPutMix = useMemo(() => {
    if (!backtestData || !Array.isArray(backtestData.trades)) {
      return { callCount: 0, putCount: 0 };
    }
    const callCount = backtestData.trades.filter((trade: any) => trade.direction === 'CALL').length;
    const putCount = backtestData.trades.length - callCount;
    return { callCount, putCount };
  }, [backtestData]);

useEffect(() => {
  if (!stage3 || !sessionProfiles || sessionProfiles.length === 0) {
    setVolumeConfluence([]);
    return;
  }

  const latestSession = sessionProfiles[sessionProfiles.length - 1];
  if (!latestSession) {
    setVolumeConfluence([]);
    return;
  }

  const messages: string[] = [];
  const tolerance = latestSession.priceStep ? latestSession.priceStep * 2 : 1;

  stage3.callWalls.slice(0, 3).forEach(wall => {
    if (Math.abs(wall.strike - latestSession.poc) <= tolerance) {
      messages.push(`Call wall ${wall.strike.toFixed(2)} aligns with session POC ${latestSession.poc.toFixed(2)}.`);
    } else if (Math.abs(wall.strike - latestSession.vah) <= tolerance) {
      messages.push(`Call wall ${wall.strike.toFixed(2)} overlaps volume VAH ${latestSession.vah.toFixed(2)} (potential ceiling).`);
    }
  });

  stage3.putZones.slice(0, 3).forEach(zone => {
    if (Math.abs(zone.strike - latestSession.poc) <= tolerance) {
      messages.push(`Put zone ${zone.strike.toFixed(2)} aligns with session POC ${latestSession.poc.toFixed(2)}.`);
    } else if (Math.abs(zone.strike - latestSession.val) <= tolerance) {
      messages.push(`Put zone ${zone.strike.toFixed(2)} overlaps volume VAL ${latestSession.val.toFixed(2)} (support shelf).`);
    }
  });

  setVolumeConfluence(messages);
}, [stage3, sessionProfiles]);

  const baseToggleClasses =
    'relative flex items-center justify-between p-3.5 rounded-xl bg-slate-900 border-2 border-slate-700 transition-all duration-200 cursor-pointer group hover:bg-slate-800';
  const indicatorToggleClasses = `${baseToggleClasses} hover:border-sky-500`;
  const gexToggleClasses = `${baseToggleClasses} hover:border-purple-500`;
  const indicatorToggleTextClass =
    'text-base font-semibold text-slate-100 group-hover:text-sky-300 transition-colors';
  const gexToggleTextClass =
    'text-base font-semibold text-slate-100 group-hover:text-purple-300 transition-colors';
  const baseCheckboxClasses =
    "w-6 h-6 rounded border-2 border-slate-700 bg-slate-900 cursor-pointer appearance-none transition-all duration-150 checked:after:content-['✓'] checked:after:text-slate-950 checked:after:text-lg checked:after:font-bold checked:after:flex checked:after:items-center checked:after:justify-center";
  const checkboxClasses = `${baseCheckboxClasses} checked:bg-sky-400 checked:border-sky-400`;
  const gexCheckboxClasses = `${baseCheckboxClasses} checked:bg-purple-400 checked:border-purple-400`;

  const resizeStateRef = useRef<{ panel: 'chart' | 'rsi' | 'macd' | null; startY: number; startHeight: number }>({
    panel: null,
    startY: 0,
    startHeight: 0,
  });

  const handlePanelResizeStart = useCallback(
    (panel: 'chart' | 'rsi' | 'macd') =>
      (event: ReactMouseEvent<HTMLDivElement> | ReactTouchEvent<HTMLDivElement>) => {
        const clientY =
          'touches' in event ? event.touches[0]?.clientY ?? 0 : event.clientY;

        resizeStateRef.current = {
          panel,
          startY: clientY,
          startHeight: panel === 'chart' ? chartHeight : panel === 'rsi' ? rsiHeight : macdHeight,
        };

        document.body.style.cursor = 'row-resize';
        event.preventDefault();
      },
    [chartHeight, macdHeight, rsiHeight]
  );

  // Auto-adjust days when timeInterval changes to prevent API errors
  useEffect(() => {
    const maxDays: Record<string, number> = {
      '1min': 5,
      '5min': 15,
      '15min': 30,
      '30min': 30,
      '1hour': 60,
      '4hour': 90,
    };

    if (maxDays[timeInterval] && days > maxDays[timeInterval]) {
      setDays(maxDays[timeInterval]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeInterval]);

  useEffect(() => {
    if (regimeMode === 'scalp' && gexMode !== 'intraday') {
      setGexMode('intraday');
    } else if (regimeMode === 'swing' && gexMode !== 'swing') {
      setGexMode('swing');
    } else if (regimeMode === 'leaps' && gexMode !== 'leaps') {
      setGexMode('leaps');
    }
  }, [regimeMode, gexMode]);

  // Calculate SMA with caching (optimized for live updates)
  const calculateSMA = useCallback((data: any[], period: number) => {
    // Round data length to nearest 5 to reduce cache misses on live updates while maintaining accuracy
    const roundedLength = Math.floor(data.length / 5) * 5;
    const cacheKey = `sma-${period}-${roundedLength}-${data[0]?.time}`;
    if (indicatorCacheRef.current.has(cacheKey)) {
      return indicatorCacheRef.current.get(cacheKey);
    }

    const sma = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((acc, val) => acc + val.close, 0);
      sma.push({ time: data[i].time, value: sum / period });
    }

    indicatorCacheRef.current.set(cacheKey, sma);
    return sma;
  }, []);

  // Calculate RSI with caching (optimized for live updates)
  const calculateRSI = useCallback((data: Array<{ time: number; close: number }>, period: number = 14) => {
    if (!Array.isArray(data) || data.length <= period) {
      return [];
    }

    // Round data length to nearest 5 to reduce cache misses on live updates while maintaining accuracy
    const roundedLength = Math.floor(data.length / 5) * 5;
    const cacheKey = `rsi-${period}-${roundedLength}-${data[0]?.time}`;
    if (indicatorCacheRef.current.has(cacheKey)) {
      return indicatorCacheRef.current.get(cacheKey);
    }

    const gains: number[] = [];
    const losses: number[] = [];

    for (let i = 1; i < data.length; i++) {
      const change = data[i].close - data[i - 1].close;
      gains.push(Math.max(change, 0));
      losses.push(Math.max(-change, 0));
    }

    let avgGain = gains.slice(0, period).reduce((acc, val) => acc + val, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((acc, val) => acc + val, 0) / period;

    const rsi = [];

    const initialRS = avgLoss === 0 ? Number.POSITIVE_INFINITY : avgGain / avgLoss;
    const firstValue = avgLoss === 0 ? 100 : 100 - 100 / (1 + initialRS);
    rsi.push({
      time: data[period].time,
      value: Number.isFinite(firstValue) ? firstValue : 0,
    });

    for (let i = period; i < gains.length; i++) {
      avgGain = (avgGain * (period - 1) + gains[i]) / period;
      avgLoss = (avgLoss * (period - 1) + losses[i]) / period;

      const rs = avgLoss === 0 ? Number.POSITIVE_INFINITY : avgGain / avgLoss;
      const value = avgLoss === 0 ? 100 : 100 - 100 / (1 + rs);
      rsi.push({
        time: data[i + 1].time,
        value: Number.isFinite(value) ? value : 0,
      });
    }

    indicatorCacheRef.current.set(cacheKey, rsi);
    return rsi;
  }, []);

  // Calculate EMA with caching (optimized for live updates)
  const calculateEMA = useCallback((data: Array<{ time: number; close: number }>, period: number) => {
    // Round data length to nearest 5 to reduce cache misses on live updates while maintaining accuracy
    const roundedLength = Math.floor(data.length / 5) * 5;
    const cacheKey = `ema-${period}-${roundedLength}-${data[0]?.time}`;
    if (indicatorCacheRef.current.has(cacheKey)) {
      return indicatorCacheRef.current.get(cacheKey);
    }

    const k = 2 / (period + 1);
    const ema: Array<{ time: number; value: number }> = [];
    let previousEMA = data[0].close;

    for (let i = 0; i < data.length; i++) {
      const currentEMA = data[i].close * k + previousEMA * (1 - k);
      ema.push({ time: data[i].time, value: currentEMA });
      previousEMA = currentEMA;
    }

    indicatorCacheRef.current.set(cacheKey, ema);
    return ema;
  }, []);

  // Calculate MACD with caching (optimized for live updates)
  const calculateMACD = useCallback((data: Array<{ time: number; close: number }>) => {
    // Round data length to nearest 5 to reduce cache misses on live updates while maintaining accuracy
    const roundedLength = Math.floor(data.length / 5) * 5;
    const cacheKey = `macd-${roundedLength}-${data[0]?.time}`;
    if (indicatorCacheRef.current.has(cacheKey)) {
      return indicatorCacheRef.current.get(cacheKey);
    }

    const ema12 = calculateEMA(data, 12);
    const ema26 = calculateEMA(data, 26);
    const macdLine = ema12.map((val: { time: number; value: number }, i: number) => ({
      time: val.time,
      value: val.value - (ema26[i]?.value || 0),
    }));
    const signalLine = calculateEMA(macdLine.map((d: { time: number; value: number }) => ({ ...d, close: d.value })), 9);
    const histogram = macdLine.map((val: { time: number; value: number }, i: number) => ({
      time: val.time,
      value: val.value - (signalLine[i]?.value || 0),
      color: val.value >= (signalLine[i]?.value || 0) ? 'rgba(0, 150, 136, 0.5)' : 'rgba(255, 82, 82, 0.5)',
    }));

    const result = { macdLine, signalLine, histogram };
    indicatorCacheRef.current.set(cacheKey, result);
    return result;
  }, [calculateEMA]);

  // Calculate Bollinger Bands with caching (optimized for live updates)
  const calculateBB = useCallback((data: Array<{ time: number; close: number }>, period: number = 20, stdDev: number = 2) => {
    // Round data length to nearest 5 to reduce cache misses on live updates while maintaining accuracy
    const roundedLength = Math.floor(data.length / 5) * 5;
    const cacheKey = `bb-${period}-${stdDev}-${roundedLength}-${data[0]?.time}`;
    if (indicatorCacheRef.current.has(cacheKey)) {
      return indicatorCacheRef.current.get(cacheKey);
    }

    const bb = {
      upper: [] as Array<{ time: number; value: number }>,
      middle: [] as Array<{ time: number; value: number }>,
      lower: [] as Array<{ time: number; value: number }>,
    };

    for (let i = period - 1; i < data.length; i++) {
      const slice = data.slice(i - period + 1, i + 1);
      const sma = slice.reduce((acc, val) => acc + val.close, 0) / period;
      const variance = slice.reduce((acc, val) => acc + Math.pow(val.close - sma, 2), 0) / period;
      const std = Math.sqrt(variance);

      bb.middle.push({ time: data[i].time, value: sma });
      bb.upper.push({ time: data[i].time, value: sma + stdDev * std });
      bb.lower.push({ time: data[i].time, value: sma - stdDev * std });
    }

    indicatorCacheRef.current.set(cacheKey, bb);
    return bb;
  }, []);

  const formatGammaNotional = (value?: number | null) => {
    if (value === undefined || value === null || Number.isNaN(value)) {
      return 'N/A';
    }

    const absValue = Math.abs(value);
    const sign = value >= 0 ? '+' : '-';

    const formatWithSuffix = (val: number, suffix: string) => `${sign}${val.toFixed(2)}${suffix}`;

    if (absValue >= 1e12) {
      return formatWithSuffix(absValue / 1e12, 'T');
    }
    if (absValue >= 1e9) {
      return formatWithSuffix(absValue / 1e9, 'B');
    }
    if (absValue >= 1e6) {
      return formatWithSuffix(absValue / 1e6, 'M');
    }
    if (absValue >= 1e3) {
      return formatWithSuffix(absValue / 1e3, 'K');
    }

    return `${sign}${absValue.toFixed(2)}`;
  };

  const formatNumber = (value: number | null | undefined, digits = 2) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '—';
    }

    const abs = Math.abs(value);
    if (abs >= 1_000_000_000) {
      return `${(value / 1_000_000_000).toFixed(digits)}B`;
    }
    if (abs >= 1_000_000) {
      return `${(value / 1_000_000).toFixed(digits)}M`;
    }
    if (abs >= 1_000) {
      return `${(value / 1_000).toFixed(digits)}K`;
    }
    return value.toFixed(digits);
  };

  const formatPercent = (value: number | null | undefined, digits = 2) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '—';
    }
    return `${(value * 100).toFixed(digits)}%`;
  };

  const formatCurrency = (value: number | null | undefined, digits = 2) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '$—';
    }
    return value.toLocaleString(undefined, {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  };

  const lifecycleStatusStyles: Record<RegimeTradeStatus, string> = {
    watching: 'bg-sky-500/15 text-sky-200 border border-sky-400/40',
    entered: 'bg-emerald-500/15 text-emerald-200 border border-emerald-400/40',
    scaled: 'bg-blue-500/15 text-blue-200 border border-blue-400/40',
    target_hit: 'bg-purple-500/15 text-purple-200 border border-purple-400/40',
    stopped: 'bg-rose-500/15 text-rose-200 border border-rose-400/40',
    expired: 'bg-amber-500/15 text-amber-200 border border-amber-400/40',
    cancelled: 'bg-slate-500/15 text-slate-200 border border-slate-400/40',
  };

  const formatLifecycleStatus = (status: RegimeTradeStatus) =>
    status.replace(/_/g, ' ');

  const TradeLifecycleCard = ({
    trade,
    selected,
    onSelect,
  }: {
    trade: RegimeTradeLifecycle;
    selected: boolean;
    onSelect: (id: string) => void;
  }) => {
    const statusClass =
      lifecycleStatusStyles[trade.status] ||
      'bg-slate-600/25 text-slate-200 border border-slate-500/40';
    const recentHistory = [...trade.history].slice(-3).reverse();

    return (
      <button
        type="button"
        onClick={() => onSelect(trade.id)}
        className={`w-full text-left rounded-2xl border p-5 transition-all duration-200 ${
          selected
            ? 'border-sky-400/70 shadow-[0_0_30px_rgba(56,189,248,0.25)] bg-slate-900/90'
            : 'border-slate-800/80 bg-slate-900/70 hover:border-sky-400/60 hover:bg-slate-900/80'
        }`}
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">
              {trade.strategy} · {trade.positionSize} size
            </p>
            <p className="mt-1 text-lg font-semibold text-slate-100">
              {trade.symbol} · {trade.direction.toUpperCase()}
            </p>
          </div>
          <span
            className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${statusClass}`}
          >
            {formatLifecycleStatus(trade.status)}
          </span>
        </div>

        <div className="mt-4 grid gap-3 text-sm text-slate-300 sm:grid-cols-2">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Entry & Trigger</p>
            <p className="font-semibold">
              ${trade.entryPrice.toFixed(2)} · Trigger {trade.direction === 'long' ? '≥' : '≤'}{' '}
              {trade.triggerLevel.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Stops & Targets</p>
            <p className="font-semibold">
              Stop {trade.stopLoss.toFixed(2)} · TP1 {trade.firstTarget.toFixed(2)}
              {trade.secondaryTarget ? ` · TP2 ${trade.secondaryTarget.toFixed(2)}` : ''}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Risk Metrics</p>
            <p className="font-semibold">
              Risk {formatNumber(trade.riskPerShare, 2)} · R {trade.rMultipleAchieved.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Timeline</p>
            <p className="font-semibold">
              {trade.enteredAt
                ? `Entered ${new Date(trade.enteredAt).toLocaleTimeString()}`
                : 'Awaiting trigger'}
              {trade.timerExpiry
                ? ` · Time stop ${new Date(trade.timerExpiry).toLocaleTimeString()}`
                : ''}
            </p>
          </div>
        </div>

        <div className="mt-4 rounded-xl border border-slate-800/60 bg-slate-950/60 p-3 text-sm text-slate-300">
          <p className="font-semibold text-slate-100">Next Action</p>
          <p className="mt-1 text-slate-400">{trade.nextAction}</p>
        </div>

        {trade.whaleConfirmation && (
          <div className="mt-3 rounded-xl border border-purple-500/30 bg-purple-500/10 p-3 text-xs text-purple-200">
            Whale flow: {trade.whaleConfirmation.direction.toUpperCase()} {trade.whaleConfirmation.contracts}{' '}
            {trade.whaleConfirmation.optionType.toUpperCase()}s @ {trade.whaleConfirmation.midpointPrice.toFixed(2)} exp{' '}
            {trade.whaleConfirmation.expiration}
          </div>
        )}

        <div className="mt-4 text-xs text-slate-500">
          <p className="font-semibold text-slate-400 mb-1">Recent Updates</p>
          <ul className="space-y-1">
            {recentHistory.map(entry => (
              <li key={`${entry.timestamp}-${entry.status}`}>
                <span className="text-slate-400">
                  [{new Date(entry.timestamp).toLocaleTimeString()}]
                </span>{' '}
                {formatLifecycleStatus(entry.status)} – {entry.note}
              </li>
            ))}
          </ul>
        </div>
      </button>
    );
  };

  // Calculate Volume Profile for a single session/day
  const calculateSessionVolumeProfile = (
    data: Array<{ high: number; low: number; volume: number; time: number }>,
    numBins: number = 50,
  ) => {
    if (data.length === 0) return null;

    // Find price range for this session
    const prices = data.flatMap(bar => [bar.high, bar.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceStep = (maxPrice - minPrice) / numBins;

    if (priceStep === 0) return null;

    // Create price bins
    const bins: Record<number, number> = {};
    for (let i = 0; i < numBins; i++) {
      const price = minPrice + i * priceStep;
      bins[price] = 0;
    }

    // Distribute volume across bins
    data.forEach((bar: any) => {
      const volume = bar.volume || 0;
      const midPrice = (bar.high + bar.low) / 2;
      const binPrice = Math.floor((midPrice - minPrice) / priceStep) * priceStep + minPrice;
      bins[binPrice] = (bins[binPrice] || 0) + volume;
    });

    // Find Point of Control (POC) - highest volume price
    let poc = minPrice;
    let maxVolume = 0;
    Object.entries(bins).forEach(([price, volume]) => {
      if (volume > maxVolume) {
        maxVolume = volume;
        poc = parseFloat(price);
      }
    });

    // Calculate Value Area (70% of volume)
    const totalVolume = Object.values(bins).reduce((sum, vol) => sum + vol, 0);
    const targetVolume = totalVolume * 0.70;

    const priceList = Object.keys(bins).map(p => parseFloat(p)).sort((a, b) => a - b);
    const pocIndex = priceList.findIndex(p => Math.abs(p - poc) < priceStep / 2);

    let valueAreaVolume = bins[poc] || 0;
    let vah = poc;
    let val = poc;
    let upperIndex = pocIndex;
    let lowerIndex = pocIndex;

    while (valueAreaVolume < targetVolume && (upperIndex < priceList.length - 1 || lowerIndex > 0)) {
      const upperPrice = upperIndex < priceList.length - 1 ? priceList[upperIndex + 1] : null;
      const lowerPrice = lowerIndex > 0 ? priceList[lowerIndex - 1] : null;
      const upperVolume = upperPrice !== null ? (bins[upperPrice] || 0) : 0;
      const lowerVolume = lowerPrice !== null ? (bins[lowerPrice] || 0) : 0;

      if (upperVolume >= lowerVolume && upperPrice !== null) {
        upperIndex++;
        vah = upperPrice;
        valueAreaVolume += upperVolume;
      } else if (lowerPrice !== null) {
        lowerIndex--;
        val = lowerPrice;
        valueAreaVolume += lowerVolume;
      } else {
        break;
      }
    }

    return {
      poc,
      vah,
      val,
      bins,
      maxVolume,
      minPrice,
      maxPrice,
      priceStep,
      startTime: data[0].time,
      endTime: data[data.length - 1].time
    };
  };

  // Calculate Session Volume Profiles - one per trading day (starting at 18:00 previous day)
  const calculateSessionVolumeProfiles = (
    data: Array<{ time: number; high: number; low: number; volume: number }>,
  ) => {
    if (data.length === 0) return [];

    const sessions: any[] = [];
    let currentSession: any[] = [];
    let currentSessionId = '';

    data.forEach((bar: any, index: number) => {
      // Get timestamp
      const barDate = new Date(typeof bar.time === 'number' ? bar.time * 1000 : bar.time);

      // Trading session starts at 18:00 (6 PM) and goes to next day 18:00
      // If before 18:00, belongs to previous day's session
      // If 18:00 or after, belongs to current day's session
      const hour = barDate.getHours();

      // Determine which session this bar belongs to
      let sessionDate = new Date(barDate);
      if (hour < 18) {
        // Before 6 PM - belongs to previous day's session (which started at 6 PM previous day)
        sessionDate.setDate(sessionDate.getDate() - 1);
      }

      // Session ID is the date when the session started (at 18:00)
      const sessionId = sessionDate.toISOString().split('T')[0]; // YYYY-MM-DD

      if (sessionId !== currentSessionId) {
        // New session started
        if (currentSession.length > 0) {
          const profile = calculateSessionVolumeProfile(currentSession);
          if (profile) {
            // Override startTime to be 18:00 ET (Eastern Time)
            // 18:00 ET = 22:00 UTC during EDT (most of trading year)
            const sessionDate = new Date(currentSessionId + 'T00:00:00Z'); // Midnight UTC
            sessionDate.setUTCHours(22, 0, 0, 0); // 22:00 UTC = 18:00 EDT
            const sessionStartTimestamp = Math.floor(sessionDate.getTime() / 1000);
            profile.startTime = sessionStartTimestamp;
            sessions.push(profile);
          }
        }
        currentSession = [bar];
        currentSessionId = sessionId;
      } else {
        currentSession.push(bar);
      }
    });

    // Don't forget the last session
    if (currentSession.length > 0) {
      const profile = calculateSessionVolumeProfile(currentSession);
      if (profile) {
        // Override startTime to be 18:00 ET (Eastern Time)
        // 18:00 ET = 22:00 UTC during EDT (most of trading year)
        const sessionDate = new Date(currentSessionId + 'T00:00:00Z'); // Midnight UTC
        sessionDate.setUTCHours(22, 0, 0, 0); // 22:00 UTC = 18:00 EDT
        const sessionStartTimestamp = Math.floor(sessionDate.getTime() / 1000);
        profile.startTime = sessionStartTimestamp;
        sessions.push(profile);
      }
    }

    return sessions;
  };

  // Helper function to draw session shading on a given canvas context
  const drawSessionShadingOnCanvas = (ctx: CanvasRenderingContext2D, chart: any, data: any[], containerWidth: number, containerHeight: number) => {
    if (!data || data.length === 0) return;

    // Helper to get ET hour from timestamp
    const getETHour = (timestamp: number) => {
      const date = new Date(timestamp * 1000);
      // Convert to ET using proper locale string
      const etString = date.toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false });
      const timeMatch = etString.match(/(\d+):(\d+):(\d+)/);
      if (timeMatch) {
        const hour = parseInt(timeMatch[1]);
        const minutes = parseInt(timeMatch[2]);
        return hour + minutes / 60;
      }
      return 0;
    };

    // Get time scale for coordinate conversion
    const timeScale = chart.timeScale();

    // Group consecutive bars by session and draw
    let currentSession = '';
    let sessionStart = 0;

    data.forEach((bar: any, index: number) => {
      if (typeof bar.time !== 'number') return;

      const hour = getETHour(bar.time);
      let session = '';

      if (hour >= 4 && hour < 9.5) {
        session = 'premarket';
      } else if (hour >= 9.5 && hour < 16) {
        session = 'regular';
      } else if (hour >= 16 && hour < 20) {
        session = 'afterhours';
      }

      // Draw shading when session changes or at the end
      if (session !== currentSession || index === data.length - 1) {
        if (currentSession && sessionStart >= 0) {
          const startTime = data[sessionStart].time as number;
          const endTime = (index === data.length - 1 ? bar.time : data[index - 1].time) as number;

          // Use chart's timeScale to convert timestamps to X coordinates
          const x1 = timeScale.timeToCoordinate(startTime);
          const x2 = timeScale.timeToCoordinate(endTime);

          // Only draw if coordinates are valid and visible
          if (x1 !== null && x2 !== null) {
            let color = '';
            if (currentSession === 'premarket') {
              color = 'rgba(100, 100, 255, 0.08)'; // Blue for pre-market
            } else if (currentSession === 'afterhours') {
              color = 'rgba(255, 100, 100, 0.08)'; // Red for after-hours
            }
            // Regular hours: no shading (transparent)

            if (color) {
              ctx.fillStyle = color;
              ctx.fillRect(x1, 0, x2 - x1, containerHeight);
            }
          }
        }

        currentSession = session;
        sessionStart = index;
      }
    });
  };

  // Draw session volume profile bars on canvas - one profile per trading day
  const drawVolumeProfileBars = () => {
    const canvas = volumeProfileCanvasRef.current;
    const chart = chartRef.current;
    const sessions = volumeProfileDataRef.current; // Now this is an array of sessions

    if (!canvas || !chart) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const chartContainer = chartContainerRef.current;
    if (!chartContainer) return;

    const containerWidth = chartContainer.clientWidth;
    const containerHeight = chartContainer.clientHeight;
    const chartWidth = chart.timeScale().width();

    // Set canvas size to match chart area only (exclude price axis)
    canvas.width = chartWidth;
    canvas.height = containerHeight;
    canvas.style.width = chartWidth + 'px';
    canvas.style.height = containerHeight + 'px';

    // Clear canvas
    ctx.clearRect(0, 0, chartWidth, containerHeight);

    // FIRST: Draw session shading (as background)
    if (candlestickDataRef.current && candlestickDataRef.current.length > 0) {
      drawSessionShadingOnCanvas(ctx, chart, candlestickDataRef.current, chartWidth, containerHeight);
    }

    // SECOND: Draw volume profile bars on top of shading
    if (!sessions || !showVolumeProfile || !Array.isArray(sessions) || sessions.length === 0) return;

    const candlestickSeries = candlestickSeriesRef.current;
    if (!candlestickSeries) return;

    const vpMaxWidth = 80; // Maximum width for volume profile bars

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, Math.max(0, chartWidth - 1), containerHeight);
    ctx.clip();

    // Draw each session's volume profile
    sessions.forEach((session: any) => {
      const { bins, maxVolume, poc, vah, val, priceStep, startTime } = session;

      // Get X coordinate for start of session
      const sessionX = chart.timeScale().timeToCoordinate(startTime);
      if (sessionX === null || sessionX === undefined) return;

      // Sort bins by price
      const sortedBins = Object.entries(bins).sort((a, b) => parseFloat(b[0]) - parseFloat(a[0]));

      // Calculate bar height based on price step
      if (sortedBins.length === 0) return;
      const samplePrice = parseFloat(sortedBins[0][0]);
      const nextPrice = samplePrice + priceStep;
      const y1 = candlestickSeries.priceToCoordinate(samplePrice);
      const y2 = candlestickSeries.priceToCoordinate(nextPrice);
      const calculatedBarHeight = y1 !== null && y2 !== null ? Math.abs(y1 - y2) : 5;
      const barHeight = Math.max(calculatedBarHeight * 0.9, 2);

      // Draw volume bars for this session
      sortedBins.forEach(([priceStr, volume]: [string, any]) => {
        const price = parseFloat(priceStr);
        const volumeRatio = volume / maxVolume;
        const barWidth = vpMaxWidth * volumeRatio;

        // Get Y coordinate for this price
        const y = candlestickSeries.priceToCoordinate(price);
        if (y === null || y === undefined) return;

        // Color based on value area
        let color = 'rgba(120, 120, 120, 0.4)';
        if (Math.abs(price - poc) < priceStep / 2) {
          color = 'rgba(255, 87, 34, 0.8)'; // POC - orange
        } else if (price >= val && price <= vah) {
          color = 'rgba(76, 175, 80, 0.6)'; // Value area - green
        }

        // Draw bar extending to the right from session start
        ctx.fillStyle = color;
        ctx.fillRect(sessionX, y - barHeight / 2, barWidth, barHeight);

        // Draw border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(sessionX, y - barHeight / 2, barWidth, barHeight);
      });

      // Draw POC, VAH, VAL lines for this session (extending right from session start)
      const sessionEndX = Math.min(chartWidth - 1, sessionX + vpMaxWidth + 20);

      // POC line
      const pocY = candlestickSeries.priceToCoordinate(poc);
      if (pocY !== null) {
        ctx.strokeStyle = 'rgba(255, 87, 34, 0.9)';
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(sessionX, pocY);
        ctx.lineTo(sessionEndX, pocY);
        ctx.stroke();
      }

      // VAH line
      const vahY = candlestickSeries.priceToCoordinate(vah);
      if (vahY !== null) {
        ctx.strokeStyle = 'rgba(76, 175, 80, 0.8)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        ctx.moveTo(sessionX, vahY);
        ctx.lineTo(sessionEndX, vahY);
        ctx.stroke();
      }

      // VAL line
      const valY = candlestickSeries.priceToCoordinate(val);
      if (valY !== null) {
        ctx.strokeStyle = 'rgba(76, 175, 80, 0.8)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        ctx.moveTo(sessionX, valY);
        ctx.lineTo(sessionEndX, valY);
        ctx.stroke();
      }

      ctx.setLineDash([]); // Reset dash
    });

    ctx.restore();
  };

  // Draw GEX (Gamma Exposure) bars
  // NEGATIVE GEX (puts/support) = RED bars extending LEFT
  // POSITIVE GEX (calls/resistance) = GREEN bars extending RIGHT
  const drawGEXBars = () => {
    if (!gexCanvasRef.current || !candlestickSeriesRef.current || !chartRef.current || !gexDataRef.current || !chartContainerRef.current) {
      return;
    }

    const canvas = gexCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const container = chartContainerRef.current;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const chartWidth = chartRef.current.timeScale().width();

    // Set canvas size to match chart area only (exclude price axis)
    canvas.width = chartWidth;
    canvas.height = containerHeight;
    canvas.style.width = chartWidth + 'px';
    canvas.style.height = containerHeight + 'px';

    // Clear and draw session shading first (as background)
    if (candlestickDataRef.current && candlestickDataRef.current.length > 0 && chartRef.current) {
      drawSessionShadingOnCanvas(ctx, chartRef.current, candlestickDataRef.current, chartWidth, containerHeight);
    }

    const gexData = gexDataRef.current;
    if (!gexData || !gexData.gexData || gexData.gexData.length === 0) {
      return;
    }

    // Calculate max GEX magnitude for scaling
    const maxGex = Math.max(...gexData.gexData.map((item: any) => Math.abs(item.netGex)));
    if (maxGex === 0) return;

    const barMaxWidth = 120; // Maximum width of GEX bars in pixels (increased for more visibility)
    const barHeight = 4; // Height of each bar (slightly thicker)

    // Anchor bars to the visible chart area (timeScale width) so they sit on top of price data.
    const priceAxisX = Math.max(0, chartWidth - 1);

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, Math.max(0, priceAxisX), containerHeight);
    ctx.clip();

    gexData.gexData.forEach((item: any) => {
      const price = item.strike;
      const netGex = item.netGex;

      // Convert strike price to Y coordinate
      const yCoord = candlestickSeriesRef.current.priceToCoordinate(price);
      if (yCoord === null) return; // Skip if price is not visible

      // Calculate bar width based on GEX magnitude
      const barWidth = (Math.abs(netGex) / maxGex) * barMaxWidth;
      const isPositive = netGex > 0;

      // Color and direction:
      // Positive (calls) = GREEN extending RIGHT from price axis
      // Negative (puts) = RED extending LEFT from price axis
      const alpha = Math.min(0.7, 0.4 + (Math.abs(netGex) / maxGex) * 0.3);
      const color = isPositive
        ? `rgba(34, 197, 94, ${alpha})`  // Green for calls (positive GEX)
        : `rgba(239, 68, 68, ${alpha})`; // Red for puts (negative GEX)

      // Always extend bars to the LEFT of the price axis so volume profile stays visible
      const barStartX = priceAxisX - barWidth;
      ctx.fillStyle = color;
      ctx.fillRect(barStartX, Math.floor(yCoord - barHeight / 2), barWidth, barHeight);

      // Outline with theme-specific colors
      ctx.strokeStyle = isPositive ? 'rgba(34, 197, 94, 0.9)' : 'rgba(239, 68, 68, 0.9)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(barStartX, Math.floor(yCoord - barHeight / 2), barWidth, barHeight);

      // Highlight gamma wall (strike with max absolute GEX)
      if (item.strike === gexData.gammaWall) {
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.9)'; // Purple
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.beginPath();
        ctx.moveTo(0, yCoord);
        ctx.lineTo(priceAxisX - 5, yCoord);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label for gamma wall
        ctx.font = 'bold 11px monospace';
        const label = `Gamma Wall $${item.strike.toFixed(2)}`;
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;

        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(8, yCoord - 16, textWidth + 8, 14);

        // Text
        ctx.fillStyle = 'rgba(168, 85, 247, 1)';
        ctx.fillText(label, 12, yCoord - 6);
      }
    });
    ctx.restore();

    if (containerWidth > chartWidth) {
      ctx.save();
      ctx.fillStyle = '#0B0E11';
      ctx.fillRect(chartWidth, 0, containerWidth - chartWidth, containerHeight);
      ctx.restore();
    }
  };

  const scheduleOverlayRedraw = useCallback((force = false) => {
    if (typeof window === 'undefined') {
      return;
    }

    // Throttle redraws to max 30fps (33ms) unless forced
    const now = Date.now();
    if (!force && now - redrawThrottleRef.current < 33) {
      return;
    }

    if (overlayAnimationFrameRef.current !== null) {
      cancelAnimationFrame(overlayAnimationFrameRef.current);
    }
    overlayAnimationFrameRef.current = window.requestAnimationFrame(() => {
      overlayAnimationFrameRef.current = null;
      redrawThrottleRef.current = Date.now();
      drawVolumeProfileBars();
      drawGEXBars();
    });
  }, []);

  // Fetch GEX data separately (for auto-refresh)
  const fetchGEXData = async () => {
    const normalizedSymbol = sanitizeTicker(debouncedSymbol);
    if (!showGEX || !normalizedSymbol) {
      gexDataRef.current = null;
      setGexMeta(null);
      setGexSummary(null);
      return;
    }

    try {
      // console.log(`📊 Fetching GEX data for ${normalizedSymbol} (${gexMode})...`);
      const gexResponse = await fetch(`/api/options-gex?symbol=${encodeURIComponent(normalizedSymbol)}&mode=${encodeURIComponent(gexMode)}`);
      const gexData = await gexResponse.json();

      if (gexData.error) {
        console.warn('GEX data not available:', gexData.error);
        gexDataRef.current = null;
        setGexMeta(null);
        setGexSummary(null);
      } else {
        // const expirationsForLog = (gexData.expirationDetails || [])
        //   .map((detail: any) => `${detail.date} (${detail.dte} DTE)`);
        // console.log(`✅ GEX data loaded from ${gexData.source || 'API'}: ${gexData.gexData?.length || 0} strikes, Gamma Wall at $${gexData.gammaWall}, expirations: ${expirationsForLog.join(', ')}`);
        gexDataRef.current = gexData;
        setGexMeta(
          gexData.expirationDetails
            ? { expirations: gexData.expirationDetails }
            : null
        );
        setGexSummary(gexData.summary || null);

        scheduleOverlayRedraw(true); // Force redraw when data changes
      }
    } catch (gexError) {
      console.error('Error fetching GEX data:', gexError);
      gexDataRef.current = null;
      setGexMeta(null);
      setGexSummary(null);
    }
  };

  const fetchBacktestData = useCallback(
    async (overrideSymbol?: string) => {
      const targetSymbol = sanitizeTicker(overrideSymbol ?? debouncedSymbol);
      if (!targetSymbol) {
        setBacktestError('Invalid symbol');
        return;
      }

      setBacktestLoading(true);
      setBacktestError(null);

      try {
        const params = new URLSearchParams({
          symbol: targetSymbol,
          days: '2',
          interval: '15',
          intrabar: '1',
        });
        const response = await fetch(`/api/mean-reversion/backtest?${params.toString()}`);
        const data = await response.json();

        if (!response.ok || data.error) {
          setBacktestError(data.error || 'Failed to fetch mean reversion backtest data');
          setBacktestData(null);
        } else {
          setBacktestData(data);
          console.log(
            `✅ Mean reversion backtest loaded for ${targetSymbol}: ${data.stats.totalTrades} trades (${data.stats.startDate} → ${data.stats.endDate})`,
          );
        }
      } catch (error: any) {
        console.error('Error fetching mean reversion backtest data:', error);
        setBacktestError(error.message || 'Failed to fetch mean reversion backtest');
        setBacktestData(null);
      } finally {
        setBacktestLoading(false);
      }
    },
    [debouncedSymbol],
  );

  const fetchChartData = useCallback(async () => {
    // Guard against undefined values (not empty strings, just undefined/null)
    if (timeInterval === undefined || timeInterval === null) {
      console.warn('⚠️  Skipping fetch - timeInterval is undefined/null:', { symbol: debouncedSymbol, timeInterval, days });
      return;
    }

    const normalizedSymbol = sanitizeTicker(debouncedSymbol);
    if (!normalizedSymbol) {
      setError('Please enter a valid ticker symbol.');
      return;
    }

    // Clear indicator cache when fetching new data
    indicatorCacheRef.current.clear();

    // Increment request ID to track this specific request
    const currentRequestId = ++fetchRequestIdRef.current;
    console.log(`🔍 Request #${currentRequestId}: Fetching chart data: symbol=${normalizedSymbol}, timeInterval=${timeInterval}, days=${days}`);

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/chart-data?symbol=${encodeURIComponent(normalizedSymbol)}&interval=${encodeURIComponent(timeInterval)}&days=${days}`
      );
      const data = await response.json();

      if (!response.ok) {
        const errorMessage = data?.details || data?.error || `Request for ${normalizedSymbol} failed (${response.status})`;
        console.error('❌ Chart data request failed:', errorMessage);
        setError(errorMessage);
        return;
      }

      // Check if this request is stale (a newer request was made)
      if (currentRequestId !== fetchRequestIdRef.current) {
        console.log(`⚠️  Request #${currentRequestId} is stale, ignoring (current request is #${fetchRequestIdRef.current})`);
        return;
      }

      console.log(`📊 Request #${currentRequestId}: Received data: ${data.candlesticks?.length || 0} bars, interval=${data.interval}`);

      if (data.error) {
        setError(data.details || data.error);
        return;
      }

      if (!data.candlesticks || data.candlesticks.length === 0) {
        setError('No chart data available for the selected parameters.');
        return;
      }

      // Initialize main chart if needed
      if (!chartRef.current && chartContainerRef.current) {
        console.log(`🎨 Creating new chart for ${normalizedSymbol} - container width: ${chartContainerRef.current.clientWidth}px, height: ${chartHeight}px`);

        const chart = createChart(chartContainerRef.current, {
          width: chartContainerRef.current.clientWidth,
          height: chartHeight,
          layout: {
            background: { color: '#131722' },
            textColor: '#D1D4DC',
          },
          grid: {
            vertLines: { color: 'rgba(42, 46, 57, 0.6)' },
            horzLines: { color: 'rgba(42, 46, 57, 0.6)' },
          },
          crosshair: {
            mode: CrosshairMode.Normal,
            vertLine: {
              labelBackgroundColor: '#2962FF',
              color: 'rgba(41, 98, 255, 0.3)',
              labelVisible: true,
            },
            horzLine: {
              labelBackgroundColor: '#2962FF',
              color: 'rgba(41, 98, 255, 0.3)',
              labelVisible: true,
            },
          },
          rightPriceScale: {
            borderColor: 'rgba(42, 46, 57, 0.6)',
            textColor: '#D1D4DC',
          },
          timeScale: {
            borderColor: 'rgba(42, 46, 57, 0.6)',
            timeVisible: true,
            secondsVisible: false,
          },
          localization: {
            timeFormatter: (timestamp: number) => {
              // Convert UTC timestamp to Eastern Time for display
              const date = new Date(timestamp * 1000);
              return date.toLocaleString('en-US', {
                timeZone: 'America/New_York',
                hour: '2-digit',
                minute: '2-digit',
                hour12: true,
              });
            },
          },
        });

        chartRef.current = chart;
        console.log('✅ Chart created successfully');

        // Candlestick series with modern colors (v5 API)
        const candlestickSeries = chart.addSeries(CandlestickSeries, {
          upColor: '#22C55E',
          downColor: '#EF4444',
          borderVisible: false,
          wickUpColor: '#22C55E',
          wickDownColor: '#EF4444',
        });
        candlestickSeriesRef.current = candlestickSeries;
        console.log('✅ Candlestick series added');

        const rerenderOverlays = () => {
          scheduleOverlayRedraw();
        };

        chart.timeScale().subscribeVisibleLogicalRangeChange((timeRange) => {
          if (timeRange) {
            if (rsiChartRef.current) {
              rsiChartRef.current.timeScale().setVisibleLogicalRange(timeRange);
            }
            if (macdChartRef.current) {
              macdChartRef.current.timeScale().setVisibleLogicalRange(timeRange);
            }
          }
          rerenderOverlays();
        });

        chart.timeScale().subscribeVisibleTimeRangeChange(() => {
          rerenderOverlays();
        });

        let redrawScheduled = false;
        chart.subscribeCrosshairMove(() => {
          if (!redrawScheduled) {
            redrawScheduled = true;
            requestAnimationFrame(() => {
              rerenderOverlays();
              redrawScheduled = false;
            });
          }
        });

        chart.timeScale().subscribeSizeChange(() => {
          rerenderOverlays();
        });

      } else {
        console.log(`📈 Chart already exists, reusing for ${debouncedSymbol}`);
      }

      // Set candlestick data
      if (candlestickSeriesRef.current) {
        console.log(`Setting ${data.candlesticks.length} candlesticks for ${normalizedSymbol}`);
        candlestickSeriesRef.current.setData(data.candlesticks);
        scheduleOverlayRedraw(true); // Force redraw when data changes
      } else {
        console.error('❌ Candlestick series not initialized!');
      }

      // Handle Volume series - add or remove based on toggle
      if (showVolume && chartRef.current) {
        if (!volumeSeriesRef.current) {
          const volumeSeries = chartRef.current.addSeries(HistogramSeries, {
            color: '#26a69a',
            priceFormat: {
              type: 'volume',
            },
            priceScaleId: 'volume',
            scaleMargins: {
              top: 0.8,
              bottom: 0,
            },
          });
          volumeSeriesRef.current = volumeSeries;
          console.log('✅ Volume series added');
        }
        // Set volume data
        if (volumeSeriesRef.current && data.volume) {
          volumeSeriesRef.current.setData(data.volume);
        }
      } else if (!showVolume && volumeSeriesRef.current && chartRef.current) {
        // Remove volume series when toggled off
        chartRef.current.removeSeries(volumeSeriesRef.current);
        volumeSeriesRef.current = null;
        console.log('✅ Volume series removed');
      }

      // Fit all candles to screen - ensures all data is visible
      // Use requestAnimationFrame + setTimeout to ensure chart has fully rendered
      // Lower timeframes (1min, 5min) need more time to process timestamps
      const fitChartContent = (attempt: number) => {
        requestAnimationFrame(() => {
          if (chartRef.current) {
            const timeScale = chartRef.current.timeScale();
            const visibleRange = timeScale.getVisibleLogicalRange();
            console.log(`✅ fitContent attempt ${attempt} for ${debouncedSymbol} - visible range:`, visibleRange);
            timeScale.fitContent();

            // Force a redraw
            if (attempt === 1 && candlestickSeriesRef.current) {
              const seriesData = candlestickSeriesRef.current.data ? candlestickSeriesRef.current.data() : [];
              console.log(`📊 Candlestick series has ${seriesData?.length || 0} data points`);
            }
          }
        });
      };

      // Call immediately via requestAnimationFrame
      fitChartContent(1);

      // Backup calls for slower-rendering cases (especially intraday data)
      setTimeout(() => fitChartContent(2), 300);
      setTimeout(() => fitChartContent(3), 600);
      setTimeout(() => fitChartContent(4), 1000);

      // Add SMA20 (signals are handled separately in useEffect after backtest loads)
      if (showSMA20 && chartRef.current) {
        const sma20 = calculateSMA(data.candlesticks, 20);
        if (!sma20Ref.current) {
          sma20Ref.current = chartRef.current.addSeries(LineSeries, {
            color: '#2962FF',
            lineWidth: 2,
            title: 'SMA 20',
          });
        }
        sma20Ref.current.setData(sma20);
      } else if (!showSMA20 && sma20Ref.current && chartRef.current) {
        chartRef.current.removeSeries(sma20Ref.current);
        sma20Ref.current = null;
        if (sma20BuySignalsRef.current) {
          chartRef.current.removeSeries(sma20BuySignalsRef.current);
          sma20BuySignalsRef.current = null;
        }
        if (sma20SellSignalsRef.current) {
          chartRef.current.removeSeries(sma20SellSignalsRef.current);
          sma20SellSignalsRef.current = null;
        }
      }

      if (showSMA50 && chartRef.current) {
        const sma50 = calculateSMA(data.candlesticks, 50);
        if (!sma50Ref.current) {
          sma50Ref.current = chartRef.current.addSeries(LineSeries, {
            color: '#FF6D00',
            lineWidth: 2,
            title: 'SMA 50',
          });
        }
        sma50Ref.current.setData(sma50);
      } else if (!showSMA50 && sma50Ref.current && chartRef.current) {
        chartRef.current.removeSeries(sma50Ref.current);
        sma50Ref.current = null;
      }

      if (showSMA200 && chartRef.current) {
        const sma200 = calculateSMA(data.candlesticks, 200);
        if (!sma200Ref.current) {
          sma200Ref.current = chartRef.current.addSeries(LineSeries, {
            color: '#E91E63',
            lineWidth: 2,
            title: 'SMA 200',
          });
        }
        sma200Ref.current.setData(sma200);
      } else if (!showSMA200 && sma200Ref.current && chartRef.current) {
        chartRef.current.removeSeries(sma200Ref.current);
        sma200Ref.current = null;
      }

      // Add SMA9 with Buy/Sell Signals
      if (showSMA9 && chartRef.current) {
        const sma9 = calculateSMA(data.candlesticks, 9);
        if (!sma9Ref.current) {
          sma9Ref.current = chartRef.current.addSeries(LineSeries, {
            color: '#00E676',
            lineWidth: 2,
            title: 'SMA 9',
          });
        }
        sma9Ref.current.setData(sma9);

        // Calculate buy/sell signals using LineSeries for markers
        const buySignals: any[] = [];
        const sellSignals: any[] = [];
        const candlesticks = data.candlesticks;

        for (let i = 9; i < candlesticks.length; i++) {
          const currentBar = candlesticks[i];
          const previousClose = candlesticks[i - 1].close;
          const currentSma = sma9[i - 9]?.value;
          const previousSma = i > 9 ? sma9[i - 10]?.value : null;

          if (currentSma && previousSma) {
            // Bullish crossover: price crosses ABOVE SMA
            const crossesUp = previousClose <= previousSma && currentBar.high > currentSma;
            // Bearish crossover: price crosses BELOW SMA
            const crossesDown = previousClose >= previousSma && currentBar.low < currentSma;

            if (crossesUp) {
              // Place marker below the bar
              buySignals.push({
                time: currentBar.time,
                value: currentBar.low * 0.998, // Slightly below the low
              });
            } else if (crossesDown) {
              // Place marker above the bar
              sellSignals.push({
                time: currentBar.time,
                value: currentBar.high * 1.002, // Slightly above the high
              });
            }
          }
        }

        console.log(`📍 Generated ${buySignals.length} buy signals and ${sellSignals.length} sell signals for SMA9`);

        // Create buy signal markers (green triangles pointing up)
        if (!sma9BuySignalsRef.current) {
          sma9BuySignalsRef.current = chartRef.current.addSeries(LineSeries, {
            color: '#00E676',
            lineWidth: 0, // No line connecting the points
            pointMarkersVisible: true,
            pointMarkersRadius: 5,
            title: 'BUY',
          });
        }
        sma9BuySignalsRef.current.setData(buySignals);

        // Create sell signal markers (red triangles pointing down)
        if (!sma9SellSignalsRef.current) {
          sma9SellSignalsRef.current = chartRef.current.addSeries(LineSeries, {
            color: '#FF1744',
            lineWidth: 0, // No line connecting the points
            pointMarkersVisible: true,
            pointMarkersRadius: 5,
            title: 'SELL',
          });
        }
        sma9SellSignalsRef.current.setData(sellSignals);

        console.log('✅ Buy/Sell marker series created successfully');
      } else if (!showSMA9 && sma9Ref.current && chartRef.current) {
        // Remove SMA9 line
        chartRef.current.removeSeries(sma9Ref.current);
        sma9Ref.current = null;

        // Remove buy/sell marker series
        if (sma9BuySignalsRef.current) {
          chartRef.current.removeSeries(sma9BuySignalsRef.current);
          sma9BuySignalsRef.current = null;
        }
        if (sma9SellSignalsRef.current) {
          chartRef.current.removeSeries(sma9SellSignalsRef.current);
          sma9SellSignalsRef.current = null;
        }
      }

      // Add Bollinger Bands
      if (showBB && chartRef.current) {
        const bb = calculateBB(data.candlesticks);
        if (!bbUpperRef.current) {
          bbUpperRef.current = chartRef.current.addSeries(LineSeries, {
            color: '#9C27B0',
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            title: 'BB Upper',
          });
        }
        if (!bbMiddleRef.current) {
          bbMiddleRef.current = chartRef.current.addSeries(LineSeries, {
            color: '#9C27B0',
            lineWidth: 1,
            title: 'BB Middle',
          });
        }
        if (!bbLowerRef.current) {
          bbLowerRef.current = chartRef.current.addSeries(LineSeries, {
            color: '#9C27B0',
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            title: 'BB Lower',
          });
        }
        bbUpperRef.current.setData(bb.upper);
        bbMiddleRef.current.setData(bb.middle);
        bbLowerRef.current.setData(bb.lower);
      } else if (!showBB && chartRef.current) {
        // Remove Bollinger Bands when toggled off
        if (bbUpperRef.current) {
          chartRef.current.removeSeries(bbUpperRef.current);
          bbUpperRef.current = null;
        }
        if (bbMiddleRef.current) {
          chartRef.current.removeSeries(bbMiddleRef.current);
          bbMiddleRef.current = null;
        }
        if (bbLowerRef.current) {
          chartRef.current.removeSeries(bbLowerRef.current);
          bbLowerRef.current = null;
        }
      }

      // Add EMAs
      if (showEMA && chartRef.current) {
        const ema12 = calculateEMA(data.candlesticks, 12);
        const ema26 = calculateEMA(data.candlesticks, 26);
        if (!ema12Ref.current) {
          ema12Ref.current = chartRef.current.addSeries(LineSeries, {
            color: '#00BCD4',
            lineWidth: 2,
            title: 'EMA 12',
          });
        }
        if (!ema26Ref.current) {
          ema26Ref.current = chartRef.current.addSeries(LineSeries, {
            color: '#FFC107',
            lineWidth: 2,
            title: 'EMA 26',
          });
        }
        ema12Ref.current.setData(ema12);
        ema26Ref.current.setData(ema26);
      } else if (!showEMA && chartRef.current) {
        // Remove EMAs when toggled off
        if (ema12Ref.current) {
          chartRef.current.removeSeries(ema12Ref.current);
          ema12Ref.current = null;
        }
        if (ema26Ref.current) {
          chartRef.current.removeSeries(ema26Ref.current);
          ema26Ref.current = null;
        }
      }

      // Add Session Volume Profile - one profile per trading day
      if (showVolumeProfile && chartRef.current && data.candlesticks.length > 0 && data.volume) {
        // Merge volume data with candlesticks for volume profile calculation
        const candlesticksWithVolume = data.candlesticks.map((candle: any, index: number) => ({
          ...candle,
          volume: data.volume[index]?.value || 0,
        }));

        // Calculate volume profiles for each trading session/day
        const sessionProfiles = calculateSessionVolumeProfiles(candlesticksWithVolume);

        // Store session profiles for canvas drawing
        volumeProfileDataRef.current = sessionProfiles;
        setSessionProfiles(sessionProfiles);

        // Store candlestick data for resize handling
        candlestickDataRef.current = data.candlesticks;

        // Draw volume profile bars (session shading is included)
          scheduleOverlayRedraw(true); // Force redraw when data changes
      } else {
        // Clear volume profile data when hidden
        volumeProfileDataRef.current = null;
        setSessionProfiles(null);
        candlestickDataRef.current = data.candlesticks;
        // Still draw session shading via volume profile bars
        scheduleOverlayRedraw(true); // Force redraw when data changes
      }

      // RSI Chart
      if (showRSI && rsiContainerRef.current) {
        if (!rsiChartRef.current) {
          const rsiChart = createChart(rsiContainerRef.current, {
            width: rsiContainerRef.current.clientWidth,
            height: rsiHeight,
            layout: {
              background: { color: '#131722' },
              textColor: '#D1D4DC',
            },
            grid: {
              vertLines: { color: 'rgba(42, 46, 57, 0.6)' },
              horzLines: { color: 'rgba(42, 46, 57, 0.6)' },
            },
            crosshair: {
              mode: CrosshairMode.Normal,
              vertLine: {
                labelBackgroundColor: '#2962FF',
                color: 'rgba(41, 98, 255, 0.3)',
              },
              horzLine: {
                labelBackgroundColor: '#2962FF',
                color: 'rgba(41, 98, 255, 0.3)',
              },
            },
            rightPriceScale: {
              borderColor: 'rgba(42, 46, 57, 0.6)',
              scaleMargins: {
                top: 0.1,
                bottom: 0.1,
              },
            },
            timeScale: {
              borderColor: 'rgba(42, 46, 57, 0.6)',
              visible: false,
            },
          });
          rsiChartRef.current = rsiChart;

          const rsiSeries = rsiChart.addSeries(LineSeries, {
            color: '#3B82F6',
            lineWidth: 2,
            title: 'RSI',
            priceScaleId: 'right',
          });
          rsiSeriesRef.current = rsiSeries;

          // Set RSI scale margins
          rsiChart.priceScale('right').applyOptions({
            scaleMargins: {
              top: 0.1,
              bottom: 0.1,
            },
          });

          // Add RSI levels
          rsiChart.addSeries(LineSeries, {
            color: 'rgba(239, 68, 68, 0.4)',
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
          }).setData([{ time: data.candlesticks[0].time, value: 70 }, { time: data.candlesticks[data.candlesticks.length - 1].time, value: 70 }]);

          rsiChart.addSeries(LineSeries, {
            color: 'rgba(34, 197, 94, 0.4)',
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
          }).setData([{ time: data.candlesticks[0].time, value: 30 }, { time: data.candlesticks[data.candlesticks.length - 1].time, value: 30 }]);
        }

        const rsi = calculateRSI(data.candlesticks);
        if (rsiSeriesRef.current) {
          rsiSeriesRef.current.setData(rsi);
        }
      } else if (!showRSI && rsiChartRef.current) {
        // Cleanup RSI chart when toggled off
        rsiChartRef.current.remove();
        rsiChartRef.current = null;
        rsiSeriesRef.current = null;
        console.log('✅ RSI chart removed');
      }

      // MACD Chart
      if (showMACD && macdContainerRef.current) {
        if (!macdChartRef.current) {
          const macdChart = createChart(macdContainerRef.current, {
            width: macdContainerRef.current.clientWidth,
            height: macdHeight,
            layout: {
              background: { color: '#131722' },
              textColor: '#D1D4DC',
            },
            grid: {
              vertLines: { color: 'rgba(42, 46, 57, 0.6)' },
              horzLines: { color: 'rgba(42, 46, 57, 0.6)' },
            },
            crosshair: {
              mode: CrosshairMode.Normal,
              vertLine: {
                labelBackgroundColor: '#2962FF',
                color: 'rgba(41, 98, 255, 0.3)',
              },
              horzLine: {
                labelBackgroundColor: '#2962FF',
                color: 'rgba(41, 98, 255, 0.3)',
              },
            },
            rightPriceScale: {
              borderColor: 'rgba(42, 46, 57, 0.6)',
            },
            timeScale: {
              borderColor: 'rgba(42, 46, 57, 0.6)',
              visible: false,
            },
          });
          macdChartRef.current = macdChart;

          macdSeriesRef.current = macdChart.addSeries(LineSeries, {
            color: '#3B82F6',
            lineWidth: 2,
            title: 'MACD',
          });

          macdSignalSeriesRef.current = macdChart.addSeries(LineSeries, {
            color: '#F59E0B',
            lineWidth: 2,
            title: 'Signal',
          });

          macdHistogramSeriesRef.current = macdChart.addSeries(HistogramSeries, {
            color: '#22C55E',
            title: 'Histogram',
          });
        }

        const macd = calculateMACD(data.candlesticks);
        if (macdSeriesRef.current) macdSeriesRef.current.setData(macd.macdLine);
        if (macdSignalSeriesRef.current) macdSignalSeriesRef.current.setData(macd.signalLine);
        if (macdHistogramSeriesRef.current) macdHistogramSeriesRef.current.setData(macd.histogram);
      } else if (!showMACD && macdChartRef.current) {
        // Cleanup MACD chart when toggled off
        macdChartRef.current.remove();
        macdChartRef.current = null;
        macdSeriesRef.current = null;
        macdSignalSeriesRef.current = null;
        macdHistogramSeriesRef.current = null;
        console.log('✅ MACD chart removed');
      }

      // Fetch GEX data using separate function (this also handles auto-refresh)
      fetchGEXData();

    } catch (error: any) {
      console.error('Error fetching chart data:', error);
      setError(`Failed to fetch chart data: ${error.message || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  }, [
    debouncedSymbol,
    timeInterval,
    days,
    chartHeight,
    showVolume,
    showRSI,
    showMACD,
    showSMA20,
    showSMA50,
    showSMA200,
    showBB,
    showEMA,
    showVolumeProfile,
    calculateSMA,
    calculateEMA,
    calculateMACD,
    calculateBB,
    scheduleOverlayRedraw,
    setError,
  ]);

  // Debounce symbol changes - wait 300ms after user stops typing for faster auto-update
  useEffect(() => {
    console.log('⌨️  Symbol changed to:', symbol);
    const timer = setTimeout(() => {
      console.log('✅ Debounce complete, setting debouncedSymbol to:', symbol);
      setDebouncedSymbol(symbol);
    }, 300);

    return () => clearTimeout(timer);
  }, [symbol]);

  // Auto-fetch chart data when debounced symbol, interval, or days change
  useEffect(() => {
    if (debouncedSymbol && timeInterval) {
      console.log(`🔄 Auto-fetching chart for ${debouncedSymbol} (${timeInterval}, ${days} days)`);
      fetchChartData();
    }
  }, [debouncedSymbol, timeInterval, days, fetchChartData]);

  // Re-render indicators when they are toggled on/off
  // Note: backtestData is intentionally NOT in dependencies to avoid unnecessary re-renders
  useEffect(() => {
    if (chartRef.current && candlestickDataRef.current && candlestickDataRef.current.length > 0) {
      console.log('🔄 Indicator toggled, re-rendering chart data');
      fetchChartData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showVolume, showRSI, showMACD, showSMA9, showSMA20, showSMA50, showSMA200, showBB, showEMA, showVolumeProfile]);

  // Auto-fetch GEX when symbol changes or mode changes
  useEffect(() => {
    if (showGEX && debouncedSymbol) {
      console.log(`🔄 Auto-fetching GEX for ${debouncedSymbol} (${gexMode} mode)`);
      fetchGEXData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSymbol, gexMode, showGEX]);

  // Auto-fetch backtest data when symbol changes
  useEffect(() => {
    if (debouncedSymbol) {
      console.log(`🔄 Auto-fetching mean reversion backtest for ${debouncedSymbol}`);
      fetchBacktestData();
    }
  }, [debouncedSymbol, fetchBacktestData]);

  const resolveIntervalSeconds = useCallback((interval: string) => {
    switch (interval) {
      case '1min':
        return 60;
      case '5min':
        return 5 * 60;
      case '15min':
        return 15 * 60;
      case '30min':
        return 30 * 60;
      case '1hour':
        return 60 * 60;
      case '4hour':
        return 4 * 60 * 60;
      case 'daily':
        return 24 * 60 * 60;
      case 'weekly':
        return 7 * 24 * 60 * 60;
      case 'monthly':
        return 30 * 24 * 60 * 60;
      default:
        return 60;
    }
  }, []);

  // Update signal markers when backtest data loads (without refetching chart data)
  useEffect(() => {
    if (!backtestData || !backtestData.trades || !candlestickDataRef.current || !chartRef.current) {
      return;
    }

    const candlesticks = candlestickDataRef.current;
    if (!Array.isArray(candlesticks) || candlesticks.length === 0) {
      return;
    }

    console.log(`📍 Updating signal markers with ${backtestData.trades.length} backtest trades`);
    console.log(`📊 Chart has ${candlesticks.length} candlesticks at interval ${timeInterval}`);

    const intervalSeconds = resolveIntervalSeconds(timeInterval);
    const buySignals: any[] = [];
    const sellSignals: any[] = [];

    backtestData.trades.forEach((trade: any, idx: number) => {
      const entryTimeIso = trade.entryTimestamp || trade.entry?.time;
      if (!entryTimeIso) {
        return;
      }
      const entryEpoch = Math.floor(new Date(entryTimeIso).getTime() / 1000);

      let targetCandle: any | null = null;
      for (let i = 0; i < candlesticks.length; i++) {
        const start =
          typeof candlesticks[i].time === 'number'
            ? candlesticks[i].time
            : Math.floor(new Date(candlesticks[i].time).getTime() / 1000);
        const end =
          i < candlesticks.length - 1
            ? typeof candlesticks[i + 1].time === 'number'
              ? candlesticks[i + 1].time
              : Math.floor(new Date(candlesticks[i + 1].time).getTime() / 1000)
            : start + intervalSeconds;
        if (entryEpoch >= start && entryEpoch < end) {
          targetCandle = candlesticks[i];
          if (idx === 0) {
            console.log(
              `✅ Matched trade at ${new Date(entryEpoch * 1000).toLocaleString()} to candle starting ${new Date(
                start * 1000,
              ).toLocaleString()}`,
            );
          }
          break;
        }
      }

      if (!targetCandle) {
        if (idx < 3) {
          console.log(
            `❌ No candle match for trade ${idx + 1} (${new Date(entryEpoch * 1000).toLocaleString()})`,
          );
        }
        return;
      }

      if (trade.direction === 'CALL') {
        buySignals.push({
          time: targetCandle.time,
          value: typeof targetCandle.low === 'number' ? targetCandle.low * 0.998 : targetCandle.close * 0.998,
        });
      } else {
        sellSignals.push({
          time: targetCandle.time,
          value: typeof targetCandle.high === 'number' ? targetCandle.high * 1.002 : targetCandle.close * 1.002,
        });
      }
    });

    if (!sma20BuySignalsRef.current && chartRef.current) {
      sma20BuySignalsRef.current = chartRef.current.addSeries(LineSeries, {
        color: '#00E676',
        lineWidth: 0,
        pointMarkersVisible: true,
        pointMarkersRadius: 5,
      });
    }
    if (!sma20SellSignalsRef.current && chartRef.current) {
      sma20SellSignalsRef.current = chartRef.current.addSeries(LineSeries, {
        color: '#FF1744',
        lineWidth: 0,
        pointMarkersVisible: true,
        pointMarkersRadius: 5,
      });
    }

    sma20BuySignalsRef.current?.setData(buySignals);
    sma20SellSignalsRef.current?.setData(sellSignals);

    console.log(
      `✅ Plotted ${buySignals.length} CALL markers and ${sellSignals.length} PUT markers (out of ${backtestData.trades.length} trades)`,
    );
  }, [backtestData, timeInterval, resolveIntervalSeconds]);

const fetchRegime = useCallback(
  async (options?: { silent?: boolean }) => {
    const normalizedSymbol = sanitizeTicker(debouncedSymbol);
    if (!normalizedSymbol) {
      setRegimeAnalysis(null);
      setActiveTrades([]);
      setGlobalActiveTrades([]);
      return;
    }

    const silent = options?.silent ?? false;
    if (!silent) {
      setRegimeLoading(true);
      setRegimeError(null);
    }

    try {
      const params = new URLSearchParams({
        mode: regimeMode,
        symbols: normalizedSymbol,
      });
      const response = await fetch(`/api/regime?${params.toString()}`);
      if (!response.ok) {
        throw new Error('Failed to fetch strategy data');
      }
      const payload: VolatilityRegimeResponse = await response.json();
      if (sanitizeTicker(debouncedSymbol) !== normalizedSymbol) {
        return;
      }

      const analysis =
        payload.analyses.find(item => item.symbol.toUpperCase() === normalizedSymbol) ||
        payload.analyses[0] ||
        null;

      setRegimeAnalysis(analysis || null);
      setActiveTrades(analysis?.activeTrades || []);
      setGlobalActiveTrades(payload.activeTrades || []);
      scheduleOverlayRedraw(true);
      setRegimeError(null);
    } catch (err: any) {
      console.error('Error fetching regime data:', err);
      setRegimeAnalysis(null);
      setActiveTrades([]);
      setGlobalActiveTrades([]);
      scheduleOverlayRedraw(true);
      if (!silent) {
        setRegimeError(err?.message || 'Failed to fetch strategy data');
      }
    } finally {
      if (!silent) {
        setRegimeLoading(false);
      }
    }
  },
  [debouncedSymbol, regimeMode, scheduleOverlayRedraw],
);

useEffect(() => {
  fetchRegime();
}, [fetchRegime]);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchRegime({ silent: true });
    }, 60_000);

    return () => clearInterval(interval);
  }, [fetchRegime]);

  const fetchAccountSnapshot = useCallback(async () => {
    try {
      const response = await fetch('/api/account');
      if (!response.ok) {
        throw new Error('Failed to fetch account snapshot');
      }
      const payload: AccountSnapshot = await response.json();
      setAccountSnapshot(payload);
      setAccountError(null);
    } catch (error: any) {
      setAccountError(error?.message || 'Failed to fetch account snapshot');
    }
  }, []);

  useEffect(() => {
    fetchAccountSnapshot();
    if (accountPollRef.current) {
      clearInterval(accountPollRef.current);
    }
    accountPollRef.current = setInterval(() => {
      fetchAccountSnapshot();
    }, 5_000);

    return () => {
      if (accountPollRef.current) {
        clearInterval(accountPollRef.current);
        accountPollRef.current = null;
      }
    };
  }, [fetchAccountSnapshot]);

  // Disabled auto-refresh to prevent chart zoom reset during backtesting
  // useEffect(() => {
  //   const interval = setInterval(() => {
  //     fetchChartData();
  //   }, 60_000);

  //   return () => clearInterval(interval);
  // }, [fetchChartData]);

  useEffect(() => {
    if (typeof window === 'undefined' || !enableRegimeStream) {
      return undefined;
    }

    const symbol = sanitizeTicker(debouncedSymbol);
    if (!symbol) return undefined;

    const params = new URLSearchParams({ symbol, mode: regimeMode });
    const source = new EventSource(`/api/stream/regime?${params.toString()}`);

    source.onmessage = event => {
      try {
        const payload = JSON.parse(event.data);
        if (!payload || payload.symbol !== symbol) return;

        setActiveTrades(payload.trades || []);
        setGlobalActiveTrades(prev => {
          const map = new Map<string, RegimeTradeLifecycle>();
          prev
            .filter(item => item.symbol !== symbol)
            .forEach(item => map.set(item.id, item));
          (payload.trades || []).forEach((trade: RegimeTradeLifecycle) => {
            map.set(trade.id, trade);
          });
          return Array.from(map.values());
        });
        setRegimeAnalysis(prev =>
          prev
            ? {
                ...prev,
                stage2: payload.stage2 || prev.stage2,
                stage3: payload.stage3 || prev.stage3,
                activeTrades: payload.trades || prev.activeTrades,
              }
            : prev,
        );
      } catch (error) {
        console.error('Failed to parse lifecycle stream payload', error);
      }
    };

    source.addEventListener('error', () => {
      source.close();
    });

    return () => {
      source.close();
    };
  }, [enableRegimeStream, debouncedSymbol, regimeMode]);

  // Resize main chart when height changes
  useEffect(() => {
    if (chartRef.current && chartContainerRef.current) {
      chartRef.current.applyOptions({
        width: chartContainerRef.current.clientWidth,
        height: chartHeight,
      });
      scheduleOverlayRedraw(true); // Force redraw on resize
    }
  }, [chartHeight]);

  useEffect(() => {
    if (activeTrades.length === 0) {
      setSelectedTradeId(null);
      return;
    }

    setSelectedTradeId(prev => {
      if (prev && activeTrades.some(trade => trade.id === prev)) {
        return prev;
      }
      const priority =
        activeTrades.find(trade => trade.status === 'entered' || trade.status === 'scaled') ||
        activeTrades[0];
      return priority?.id ?? null;
    });
  }, [activeTrades]);

  // Resize RSI chart when height changes
  useEffect(() => {
    if (rsiChartRef.current && rsiContainerRef.current) {
      rsiChartRef.current.applyOptions({
        width: rsiContainerRef.current.clientWidth,
        height: rsiHeight,
      });
    }
  }, [rsiHeight]);

  // Resize MACD chart when height changes
  useEffect(() => {
    if (macdChartRef.current && macdContainerRef.current) {
      macdChartRef.current.applyOptions({
        width: macdContainerRef.current.clientWidth,
        height: macdHeight,
      });
    }
  }, [macdHeight]);

  useEffect(() => {
    const series = candlestickSeriesRef.current;
    if (!series) return undefined;

    tradePriceLinesRef.current.forEach(line => {
      try {
        series.removePriceLine(line);
      } catch (error) {
        // ignore removal errors
      }
    });
    tradePriceLinesRef.current = [];

    if (!selectedTrade) {
      return undefined;
    }

    const addLine = (
      price: number,
      color: string,
      title: string,
      lineStyle: LineStyle = LineStyle.Solid,
    ) => {
      const line = series.createPriceLine({
        price,
        color,
        lineStyle,
        lineWidth: 2,
        axisLabelVisible: true,
        title,
      });
      tradePriceLinesRef.current.push(line);
    };

    addLine(
      selectedTrade.entryPrice,
      '#38bdf8',
      `Entry ${selectedTrade.entryPrice.toFixed(2)}`,
    );
    addLine(
      selectedTrade.stopLoss,
      '#f87171',
      `Stop ${selectedTrade.stopLoss.toFixed(2)}`,
    );
    addLine(
      selectedTrade.firstTarget,
      '#4ade80',
      `Target 1 ${selectedTrade.firstTarget.toFixed(2)}`,
    );
    if (selectedTrade.secondaryTarget) {
      addLine(
        selectedTrade.secondaryTarget,
        '#22d3ee',
        `Target 2 ${selectedTrade.secondaryTarget.toFixed(2)}`,
        LineStyle.Dashed,
      );
    }

    return () => {
      tradePriceLinesRef.current.forEach(line => {
        try {
          series.removePriceLine(line);
        } catch (error) {
          // ignore removal errors
        }
      });
      tradePriceLinesRef.current = [];
    };
  }, [selectedTrade]);

  useEffect(() => {
    const series = candlestickSeriesRef.current;
    if (!series) return undefined;

    structuralLinesRef.current.forEach(line => {
      try {
        series.removePriceLine(line);
      } catch (error) {
        // ignore removal errors
      }
    });
    structuralLinesRef.current = [];

    if (!regimeAnalysis) {
      return undefined;
    }

    const addLine = (
      price: number,
      color: string,
      title: string,
      lineStyle: LineStyle = LineStyle.Dotted,
      lineWidth = 1,
    ) => {
      const line = series.createPriceLine({
        price,
        color,
        lineStyle,
        lineWidth,
        axisLabelVisible: true,
        title,
      });
      structuralLinesRef.current.push(line);
    };

    const stage3 = regimeAnalysis.stage3;
    stage3.callWalls.slice(0, 2).forEach(level =>
      addLine(level.strike, '#fb7185', `Call Wall ${level.strike.toFixed(2)}`),
    );
    stage3.putZones.slice(0, 2).forEach(level =>
      addLine(level.strike, '#22d3ee', `Put Zone ${level.strike.toFixed(2)}`),
    );
    if (regimeAnalysis.stage2.gammaFlipLevel) {
      addLine(
        regimeAnalysis.stage2.gammaFlipLevel,
        '#fbbf24',
        `Gamma Flip ${regimeAnalysis.stage2.gammaFlipLevel.toFixed(2)}`,
        LineStyle.Solid,
        2,
      );
    }

    return () => {
      structuralLinesRef.current.forEach(line => {
        try {
          series.removePriceLine(line);
        } catch (error) {
          // ignore removal errors
        }
      });
      structuralLinesRef.current = [];
    };
  }, [regimeAnalysis]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setShowIndicatorPanel(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // useEffect(() => {
  //   console.log('📋 Indicator panel visible:', showIndicatorPanel);
  // }, [showIndicatorPanel]);

  useEffect(() => {
    const container = chartContainerRef.current;
    if (!container) {
      return;
    }

    // Use throttled redraws for better performance
    const handlePassive: EventListener = () => scheduleOverlayRedraw(false);
    const handleActive: EventListener = () => scheduleOverlayRedraw(false);

    container.addEventListener('wheel', handlePassive, { passive: true });
    container.addEventListener('touchmove', handlePassive, { passive: true });
    container.addEventListener('touchstart', handleActive, { passive: true });
    container.addEventListener('touchend', handleActive, { passive: true });
    // Removed mousemove/mouseleave to reduce redraw frequency
    container.addEventListener('mousedown', handleActive);
    container.addEventListener('mouseup', handleActive);

    return () => {
      container.removeEventListener('wheel', handlePassive);
      container.removeEventListener('touchmove', handlePassive);
      container.removeEventListener('touchstart', handleActive);
      container.removeEventListener('touchend', handleActive);
      container.removeEventListener('mousedown', handleActive);
      container.removeEventListener('mouseup', handleActive);
      if (overlayAnimationFrameRef.current !== null) {
        cancelAnimationFrame(overlayAnimationFrameRef.current);
        overlayAnimationFrameRef.current = null;
      }
    };
  }, [scheduleOverlayRedraw]);

  useEffect(() => {
    const handlePointerMove = (event: MouseEvent | TouchEvent) => {
      const state = resizeStateRef.current;
      if (!state.panel) {
        return;
      }

      let clientY: number | null = null;
      if ('touches' in event) {
        if (event.touches.length === 0) {
          return;
        }
        clientY = event.touches[0]?.clientY ?? null;
        if (event.cancelable) {
          event.preventDefault();
        }
      } else {
        clientY = event.clientY;
      }

      if (clientY === null) {
        return;
      }

      const delta = clientY - state.startY;

      if (state.panel === 'chart') {
        const nextHeight = Math.max(300, Math.min(1200, state.startHeight + delta));
        setChartHeight(nextHeight);
      } else {
        const nextHeight = Math.max(120, Math.min(600, state.startHeight + delta));
        if (state.panel === 'rsi') {
          setRsiHeight(nextHeight);
        } else {
          setMacdHeight(nextHeight);
        }
      }
    };

    const handlePointerEnd = () => {
      if (resizeStateRef.current.panel) {
        resizeStateRef.current = { panel: null, startY: 0, startHeight: 0 };
        document.body.style.cursor = '';
      }
    };

    const pointerMoveListener = handlePointerMove as unknown as EventListener;
    const pointerEndListener = handlePointerEnd as unknown as EventListener;

    window.addEventListener('mousemove', pointerMoveListener);
    window.addEventListener('touchmove', pointerMoveListener, { passive: false });
    window.addEventListener('mouseup', pointerEndListener);
    window.addEventListener('mouseleave', pointerEndListener);
    window.addEventListener('touchend', pointerEndListener);
    window.addEventListener('touchcancel', pointerEndListener);

    return () => {
      window.removeEventListener('mousemove', pointerMoveListener);
      window.removeEventListener('touchmove', pointerMoveListener);
      window.removeEventListener('mouseup', pointerEndListener);
      window.removeEventListener('mouseleave', pointerEndListener);
      window.removeEventListener('touchend', pointerEndListener);
      window.removeEventListener('touchcancel', pointerEndListener);
      document.body.style.cursor = '';
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100 font-[family:'Manrope',sans-serif] antialiased flex flex-col" id="chart-page">
      {/* Professional Top Bar */}
      <header className="bg-gradient-to-r from-slate-950/95 via-slate-900/80 to-slate-950/95 border-b border-slate-800/70 px-6 py-4 flex items-center justify-between shadow-2xl backdrop-blur-sm">
        <div className="flex items-center gap-6">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent drop-shadow-lg">
            Pro Charts 2.0
          </h1>

          {/* Symbol Input - Professional Style */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-sky-500/20 to-cyan-400/20 rounded-xl opacity-0 group-hover:opacity-80 transition-opacity blur-sm"></div>
            <div className="relative flex items-center gap-2 bg-slate-900 rounded-xl px-4 py-2.5 border-2 border-slate-700 hover:border-sky-400 focus-within:border-sky-400 transition-all duration-200">
              <svg className="w-5 h-5 text-sky-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(sanitizeTicker(e.target.value))}
                className="bg-transparent border-none outline-none text-white font-bold text-lg w-32 placeholder-slate-400 uppercase tracking-wide"
                placeholder="SYMBOL"
              />
            </div>
          </div>

          {/* Timeframe Selector - High Contrast */}
          <div className="flex items-center gap-2 bg-slate-950 rounded-xl p-2 border-2 border-slate-700">
            {['1min', '5min', '15min', 'daily', 'weekly', 'monthly'].map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeInterval(tf)}
                className={`px-4 py-2.5 rounded-lg text-sm font-semibold uppercase tracking-wide transition-all ${
                  timeInterval === tf
                    ? 'bg-sky-500 text-white border-2 border-sky-400'
                    : 'text-slate-200 bg-slate-800 hover:text-white hover:bg-slate-700 border-2 border-slate-700'
                }`}
              >
                {tf === 'daily' ? 'D' : tf === 'weekly' ? 'W' : tf === 'monthly' ? 'M' : tf.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Period Selector - High Contrast */}
          <div className="relative">
            <select
              value={days}
              onChange={(e) => setDays(parseInt(e.target.value))}
              className="appearance-none bg-slate-900 px-4 py-2.5 pr-12 rounded-lg border-2 border-slate-700 text-sm text-white font-semibold hover:border-sky-400 focus:border-sky-400 focus:outline-none transition-all cursor-pointer"
            >
              <option value="1" className="bg-slate-900 text-white font-semibold">1 Day</option>
              <option value="5" className="bg-slate-900 text-white font-semibold">5 Days</option>
              <option value="7" className="bg-slate-900 text-white font-semibold">1 Week</option>
              <option value="14" className="bg-slate-900 text-white font-semibold">2 Weeks</option>
              <option value="30" className="bg-slate-900 text-white font-semibold">1 Month</option>
              <option value="90" className="bg-slate-900 text-white font-semibold">3 Months</option>
              <option value="180" className="bg-slate-900 text-white font-semibold">6 Months</option>
              <option value="365" className="bg-slate-900 text-white font-semibold">1 Year</option>
            </select>
            <svg className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-sky-400 pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M19 9l-7 7-7-7" />
            </svg>
          </div>

          <button
            onClick={() => fetchBacktestData(symbol)}
            disabled={backtestLoading}
            className="px-5 py-2.5 rounded-lg border-2 border-purple-500/40 bg-purple-500/15 hover:bg-purple-500/30 text-sm font-semibold uppercase tracking-wide text-purple-100 transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed shadow-[0_8px_24px_rgba(168,85,247,0.25)]"
          >
            {backtestLoading ? 'Running Backtest…' : 'Run Mean Reversion Backtest'}
          </button>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={fetchChartData}
            disabled={loading}
            className="relative overflow-hidden px-5 py-2.5 rounded-lg bg-gradient-to-r from-sky-500 via-cyan-400 to-sky-500 hover:from-sky-400 hover:via-cyan-300 hover:to-sky-400 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:from-sky-500 disabled:hover:via-cyan-400 disabled:hover:to-sky-500 text-slate-950 font-semibold text-sm uppercase tracking-wide transition-all duration-200 shadow-lg shadow-sky-500/40 hover:shadow-xl hover:scale-[1.03] active:scale-95 border border-transparent"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Loading...
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh
              </span>
            )}
          </button>
        </div>
      </header>

      {error && (
        <div className="mx-4 mt-4 bg-red-900/20 border border-red-500/50 text-red-200 px-4 py-3 rounded-lg flex items-center gap-3">
          <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
          {error}
        </div>
      )}

      <main className="relative flex-1 overflow-hidden">
        <div className="h-full w-full flex flex-col overflow-auto p-4 space-y-4">
          <div className="bg-slate-950 rounded-2xl shadow-[0_8px_32px_rgba(0,0,0,0.6)] border-2 border-slate-700">
            <div className="bg-gradient-to-r from-slate-900 to-slate-800 px-5 py-4 border-b-2 border-slate-700 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-slate-100 tracking-wide">{symbol}</h2>
                <p className="text-xs uppercase tracking-[0.28em] text-slate-400/70">Live Market View</p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setChartHeight(Math.max(300, chartHeight - 50))}
                  className="p-1.5 rounded-md bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                  title="Decrease height"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                <button
                  onClick={() => setChartHeight(Math.min(1200, chartHeight + 50))}
                  className="p-1.5 rounded-md bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                  title="Increase height"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                  </svg>
                </button>
              </div>
            </div>
            <div className="p-5">
              <div style={{ position: 'relative' }}>
          <div
            ref={chartContainerRef}
            style={{ height: `${chartHeight}px` }}
            className="border-2 border-slate-700 rounded-xl bg-slate-950"
          />
          <canvas
            ref={volumeProfileCanvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              pointerEvents: 'none',
              zIndex: 1
            }}
          />
          <canvas
            ref={gexCanvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              pointerEvents: 'none',
              zIndex: 2
            }}
          />
          </div>
          <div
            className="mt-3 h-3 rounded-full bg-slate-800 hover:bg-slate-700 cursor-row-resize flex items-center justify-center text-[10px] font-semibold tracking-[0.2em] uppercase text-slate-400 select-none"
            onMouseDown={handlePanelResizeStart('chart')}
            onTouchStart={handlePanelResizeStart('chart')}
            role="separator"
            aria-label="Resize chart"
          >
            drag
          </div>
            </div>
          </div>

          {/* RSI Chart */}
          {showRSI && (
            <div className="bg-slate-950 rounded-xl shadow-[0_6px_24px_rgba(14,165,233,0.15)] border-2 border-sky-500/40 overflow-hidden">
              <div className="bg-gradient-to-r from-slate-900 to-slate-800 px-4 py-3 border-b-2 border-sky-500/30 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-200">RSI (14)</h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setRsiHeight(Math.max(120, rsiHeight - 25))}
                    className="p-1.5 rounded-md bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                    title="Decrease height"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  <button
                    onClick={() => setRsiHeight(Math.min(600, rsiHeight + 25))}
                    className="p-1.5 rounded-md bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                    title="Increase height"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                    </svg>
                  </button>
                </div>
              </div>
              <div className="p-4 pt-5 bg-slate-950/40">
                <div
                  ref={rsiContainerRef}
                  style={{ height: `${rsiHeight}px` }}
                  className="border-2 border-sky-500/60 rounded-lg bg-slate-950 shadow-[0_4px_16px_rgba(14,165,233,0.2),inset_0_2px_4px_rgba(0,0,0,0.3)]"
                />
                <div
                  className="mt-3 h-3 rounded-full bg-slate-800 hover:bg-slate-700 cursor-row-resize flex items-center justify-center text-[10px] font-semibold tracking-[0.2em] uppercase text-slate-400 select-none"
                  onMouseDown={handlePanelResizeStart('rsi')}
                  onTouchStart={handlePanelResizeStart('rsi')}
                  role="separator"
                  aria-label="Resize RSI panel"
                >
                  drag
                </div>
              </div>
            </div>
          )}

          {/* MACD Chart */}
          {showMACD && (
            <div className="bg-slate-950 rounded-xl shadow-[0_6px_24px_rgba(16,185,129,0.15)] border-2 border-emerald-500/40 overflow-hidden">
              <div className="bg-gradient-to-r from-slate-900 to-slate-800 px-4 py-3 border-b-2 border-emerald-500/30 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-200">MACD (12, 26, 9)</h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setMacdHeight(Math.max(120, macdHeight - 25))}
                    className="p-1.5 rounded-md bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                    title="Decrease height"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  <button
                    onClick={() => setMacdHeight(Math.min(600, macdHeight + 25))}
                    className="p-1.5 rounded-md bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                    title="Increase height"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                    </svg>
                  </button>
                </div>
              </div>
              <div className="p-4 pt-5 bg-slate-950/40">
                <div
                  ref={macdContainerRef}
                  style={{ height: `${macdHeight}px` }}
                  className="border-2 border-emerald-500/60 rounded-lg bg-slate-950 shadow-[0_4px_16px_rgba(16,185,129,0.2),inset_0_2px_4px_rgba(0,0,0,0.3)]"
                />
                <div
                  className="mt-3 h-3 rounded-full bg-slate-800 hover:bg-slate-700 cursor-row-resize flex items-center justify-center text-[10px] font-semibold tracking-[0.2em] uppercase text-slate-400 select-none"
                  onMouseDown={handlePanelResizeStart('macd')}
                  onTouchStart={handlePanelResizeStart('macd')}
                  role="separator"
                  aria-label="Resize MACD panel"
                >
                  drag
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Backtest Statistics and Trade Log */}
        {backtestData && backtestData.stats && (
          <div className="px-6 pb-8">
            <div className="bg-slate-950 rounded-xl shadow-[0_6px_24px_rgba(147,51,234,0.15)] border-2 border-purple-500/40 overflow-hidden">
              <div className="bg-gradient-to-r from-slate-900 to-slate-800 px-6 py-4 border-b-2 border-purple-500/30">
                <h3 className="text-lg font-bold text-slate-100">Mean Reversion Options Backtest (15m ➜ 1m)</h3>
                <p className="text-sm text-slate-400 mt-1">
                  {backtestData.stats.symbol} • {backtestData.stats.startDate} → {backtestData.stats.endDate} • ATM weekly contracts with 2-leg scaling
                </p>
              </div>

              {/* Call/Put Ratio Banner */}
              <div className="px-6 pt-4">
                <div className={`rounded-lg p-4 border-2 ${
                  callPutMix.callCount > callPutMix.putCount
                    ? 'bg-emerald-900/20 border-emerald-500/40'
                    : callPutMix.putCount > callPutMix.callCount
                    ? 'bg-rose-900/20 border-rose-500/40'
                    : 'bg-slate-900/40 border-slate-600/40'
                }`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Trade Mix</p>
                      <p className="text-3xl font-bold text-slate-100">
                        {callPutMix.callCount} CALL / {callPutMix.putCount} PUT
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Call/Put Ratio</p>
                      <p className={`text-2xl font-bold ${
                        backtestData.stats.bias === 'BULLISH'
                          ? 'text-emerald-400'
                          : backtestData.stats.bias === 'BEARISH'
                          ? 'text-rose-400'
                          : 'text-slate-400'
                      }`}>
                        {backtestData.stats.bias}
                      </p>
                      <p className="text-xs text-slate-500 mt-1">
                        Ratio {backtestData.stats.callPutRatio.toFixed(2)} across sampled trades
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* GEX Filter Banner */}
              <div className="px-6 pt-4">
                <div className={`rounded-lg p-4 border-2 ${
                  backtestData.stats.hasNegativeGex
                    ? 'bg-emerald-900/20 border-emerald-500/40'
                    : 'bg-rose-900/20 border-rose-500/40'
                }`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Average Net GEX (Informational)</p>
                      <p className="text-3xl font-bold text-slate-100">{formatNumber(backtestData.stats.netGex, 2)}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">GEX Status</p>
                      <p className={`text-2xl font-bold ${
                        backtestData.stats.hasNegativeGex
                          ? 'text-emerald-400'
                          : 'text-rose-400'
                      }`}>
                        {backtestData.stats.hasNegativeGex ? 'AVERAGE NEGATIVE' : 'AVERAGE POSITIVE'}
                      </p>
                      <p className="text-xs text-slate-500 mt-1">
                        Calculated from 15m signal bars for reference; live strategy bypasses GEX gating.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Statistics Grid */}
              <div className="p-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-900/60 rounded-lg p-4 border border-slate-700">
                  <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Total Trades</p>
                  <p className="text-2xl font-bold text-slate-100">{backtestData.stats.totalTrades}</p>
                </div>

                <div className="bg-slate-900/60 rounded-lg p-4 border border-slate-700">
                  <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Win Rate</p>
                  <p className="text-2xl font-bold text-emerald-400">{backtestData.stats.winRate.toFixed(1)}%</p>
                  <p className="text-xs text-slate-500 mt-1">{backtestData.stats.winners}W / {backtestData.stats.losers}L</p>
                </div>

                <div className="bg-slate-900/60 rounded-lg p-4 border border-slate-700">
                  <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Total P&L</p>
                  <p className={`text-2xl font-bold ${backtestData.stats.totalPnL >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    ${backtestData.stats.totalPnL.toFixed(2)}
                  </p>
                </div>

                <div className="bg-slate-900/60 rounded-lg p-4 border border-slate-700">
                  <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Avg P&L</p>
                  <p className={`text-2xl font-bold ${backtestData.stats.avgPnL >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    ${backtestData.stats.avgPnL.toFixed(2)}
                  </p>
                </div>

                <div className="bg-slate-900/60 rounded-lg p-4 border border-slate-700">
                  <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Avg Win</p>
                  <p className="text-2xl font-bold text-emerald-400">${backtestData.stats.avgWin.toFixed(2)}</p>
                </div>

                <div className="bg-slate-900/60 rounded-lg p-4 border border-slate-700">
                  <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Avg Loss</p>
                  <p className="text-2xl font-bold text-rose-400">${backtestData.stats.avgLoss.toFixed(2)}</p>
                </div>

                <div className="bg-slate-900/60 rounded-lg p-4 border border-slate-700">
                  <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Profit Factor</p>
                  <p className="text-2xl font-bold text-slate-100">{backtestData.stats.profitFactor.toFixed(2)}</p>
                </div>

                <div className="bg-slate-900/60 rounded-lg p-4 border border-slate-700">
                  <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">Avg Hold</p>
                  <p className="text-2xl font-bold text-slate-100">{backtestData.stats.avgHoldMinutes.toFixed(1)} min</p>
                </div>
              </div>

              {/* Trade Log Table */}
              {backtestData.trades && backtestData.trades.length > 0 && (
                <div className="px-6 pb-6">
                  <h4 className="text-sm font-bold text-slate-200 mb-3 uppercase tracking-wide">Trade Log</h4>
                  <div className="bg-slate-950/80 rounded-lg border border-slate-700 overflow-hidden">
                    <div className="overflow-x-auto max-h-96 overflow-y-auto">
                      <table className="w-full text-sm">
                        <thead className="bg-slate-900 sticky top-0 z-10">
                          <tr className="border-b border-slate-700">
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">#</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Direction</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Entry Time</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Entry Stock</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Option Strike</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Entry Premium</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Exit Time</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Exit Stock</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Stock Move</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Exit Premium</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">P&L</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Hold</th>
                            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-300 uppercase tracking-wider">Exit Reason</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                          {backtestData.trades.map((trade: any) => (
                            <tr key={trade.tradeNumber} className={`hover:bg-slate-900/40 ${trade.optionPnL >= 0 ? 'bg-emerald-950/20' : 'bg-rose-950/20'}`}>
                              <td className="px-3 py-3 text-slate-300 font-medium">{trade.tradeNumber}</td>
                              <td className="px-3 py-3">
                                <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                  trade.direction === 'CALL' ? 'bg-emerald-900/40 text-emerald-300 border border-emerald-500/30' : 'bg-rose-900/40 text-rose-300 border border-rose-500/30'
                                }`}>
                                  {trade.direction}
                                </span>
                              </td>
                              <td className="px-3 py-3 text-slate-400 font-mono text-xs">{trade.entry.time}</td>
                              <td className="px-3 py-3 text-slate-300 font-mono">${trade.entry.stockPrice.toFixed(2)}</td>
                              <td className="px-3 py-3 text-slate-300 font-mono">
                                {trade.entry.optionStrike !== null ? `$${trade.entry.optionStrike.toFixed(2)}` : '—'}
                              </td>
                              <td className="px-3 py-3 text-slate-300 font-mono">
                                {trade.entry.optionPremium !== null ? `$${trade.entry.optionPremium.toFixed(2)}` : '—'}
                              </td>
                              <td className="px-3 py-3 text-slate-400 font-mono text-xs">{trade.exit.time}</td>
                              <td className="px-3 py-3 text-slate-300 font-mono">${trade.exit.stockPrice.toFixed(2)}</td>
                              <td className="px-3 py-3 font-mono">
                                <span className={trade.stockMove >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                                  {trade.stockMove >= 0 ? '+' : ''}{trade.stockMove.toFixed(2)}
                                </span>
                                <span className="text-slate-500 text-xs ml-1">
                                  ({trade.stockMovePercent >= 0 ? '+' : ''}{trade.stockMovePercent.toFixed(2)}%)
                                </span>
                              </td>
                              <td className="px-3 py-3 text-slate-300 font-mono">
                                {trade.exit.optionPremium !== null ? `$${trade.exit.optionPremium.toFixed(2)}` : '—'}
                              </td>
                              <td className="px-3 py-3 font-mono font-semibold">
                                <span className={trade.optionPnL >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                                  {trade.optionPnL >= 0 ? '+' : ''}${trade.optionPnL.toFixed(2)}
                                </span>
                                {trade.optionReturnPercent !== null && (
                                  <span className="text-slate-500 text-xs ml-1">
                                    ({trade.optionReturnPercent >= 0 ? '+' : ''}{trade.optionReturnPercent.toFixed(1)}%)
                                  </span>
                                )}
                              </td>
                              <td className="px-3 py-3 text-slate-400 text-xs">{trade.holdMinutes} min</td>
                              <td className="px-3 py-3 text-slate-400 text-xs">{trade.exit.reason}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {backtestLoading && (
          <div className="px-6 pb-8">
            <div className="bg-slate-950 rounded-xl shadow-lg border-2 border-purple-500/40 p-8 text-center">
              <p className="text-slate-400">Running mean reversion backtest...</p>
            </div>
          </div>
        )}

        {backtestError && (
          <div className="px-6 pb-8">
            <div className="bg-slate-950 rounded-xl shadow-lg border-2 border-rose-500/40 p-8 text-center">
              <p className="text-rose-400">Error running mean reversion backtest: {backtestError}</p>
            </div>
          </div>
        )}

        <section className="px-6 pb-12 space-y-8">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-2xl font-semibold text-slate-100">Volatility Regime Breakdown</h2>
              <p className="text-sm text-slate-400">
                Strategy snapshot for {sanitizeTicker(debouncedSymbol)} · {regimeMode.toUpperCase()} mode.
              </p>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
              <div className="flex items-center gap-2 bg-slate-950 rounded-xl p-1.5 border border-slate-700">
                {regimeModes.map((modeOption) => (
                  <button
                    key={modeOption}
                    type="button"
                    onClick={() => setRegimeMode(modeOption)}
                    className={`px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-wide transition-all ${
                      regimeMode === modeOption
                        ? 'bg-sky-500 text-slate-950 border border-sky-400'
                        : 'bg-slate-900 text-slate-300 border border-transparent hover:border-slate-600'
                    }`}
                  >
                    {modeOption.toUpperCase()}
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => fetchRegime({ silent: false })}
                className="px-4 py-2 rounded-lg text-xs font-semibold uppercase tracking-wide bg-slate-900 border border-slate-700 text-slate-200 hover:border-sky-400 hover:text-white transition-all"
              >
                Refresh Snapshot
              </button>
            </div>
          </div>

          {accountError && (
            <div className="rounded-2xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
              {accountError}
            </div>
          )}

          {accountSnapshot && (
            <div className="space-y-6">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/80 p-5 shadow-lg shadow-slate-900/40">
                  <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Account Balances</p>
                  <p className="text-sm text-slate-300">Cash</p>
                  <p className="text-lg font-semibold text-slate-100">{formatCurrency(accountSnapshot.balances?.totalCash)}</p>
                  <p className="mt-3 text-sm text-slate-300">Net Liquidation</p>
                  <p className="text-lg font-semibold text-slate-100">{formatCurrency(accountSnapshot.balances?.netValue)}</p>
                  <p className="mt-3 text-sm text-slate-300">Buying Power</p>
                  <p className="text-lg font-semibold text-emerald-300">{formatCurrency(accountSnapshot.balances?.buyingPower)}</p>
                  <p className="mt-3 text-xs text-slate-500">Updated {accountSnapshot.balances?.timestamp ? new Date(accountSnapshot.balances.timestamp).toLocaleTimeString() : '—'}</p>
                </div>
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/80 p-5 shadow-lg shadow-slate-900/40">
                  <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Position Summary</p>
                  <p className="text-sm text-slate-300">Market Value</p>
                  <p className="text-lg font-semibold text-slate-100">{formatCurrency(positionSummary.marketValue)}</p>
                  <p className="mt-3 text-sm text-slate-300">Cost Basis</p>
                  <p className="text-lg font-semibold text-slate-100">{formatCurrency(positionSummary.costBasis)}</p>
                  <p className="mt-3 text-sm text-slate-300">Unrealized P/L</p>
                  <p className={`text-lg font-semibold ${positionSummary.unrealized >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                    {formatCurrency(positionSummary.unrealized)} ({formatPercent(positionSummary.unrealizedPercent)})
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/80 p-5 shadow-lg shadow-slate-900/40">
                  <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Orders & Activity</p>
                  <p className="text-sm text-slate-300">Open Orders</p>
                  <p className="text-lg font-semibold text-slate-100">{accountSnapshot.orders.length}</p>
                  <p className="mt-3 text-sm text-slate-300">Pending Orders</p>
                  <p className="text-lg font-semibold text-slate-100">{accountSnapshot.balances?.pendingOrdersCount ?? 0}</p>
                  <p className="mt-3 text-sm text-slate-300">Day Trading BP</p>
                  <p className="text-lg font-semibold text-slate-100">{formatCurrency(accountSnapshot.balances?.dayTradingBuyingPower)}</p>
                </div>
              </div>

              <div className="grid gap-4 lg:grid-cols-2">
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-slate-100">Open Positions</h3>
                    <span className="text-xs uppercase tracking-wide text-slate-500">
                      {accountSnapshot.positions.length} total
                    </span>
                  </div>
                  {accountSnapshot.positions.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead className="text-xs uppercase tracking-wide text-slate-500 border-b border-slate-800">
                          <tr>
                            <th className="py-2 text-left">Symbol</th>
                            <th className="py-2 text-right">Qty</th>
                            <th className="py-2 text-right">Last</th>
                            <th className="py-2 text-right">Market</th>
                            <th className="py-2 text-right">P/L</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                          {accountSnapshot.positions.slice(0, 6).map(position => (
                            <tr key={position.symbol}>
                              <td className="py-2 text-slate-200 font-semibold">{position.symbol}</td>
                              <td className="py-2 text-right text-slate-300">{position.quantity.toLocaleString()}</td>
                              <td className="py-2 text-right text-slate-300">{position.lastPrice.toFixed(2)}</td>
                              <td className="py-2 text-right text-slate-300">{formatCurrency(position.marketValue)}</td>
                              <td className={`py-2 text-right font-semibold ${position.unrealizedPL >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>
                                {formatCurrency(position.unrealizedPL)} ({formatPercent(position.unrealizedPLPercent / 100)})
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-sm text-slate-400">No open positions in the Tradier sandbox account.</p>
                  )}
                </div>
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-slate-100">Open Orders</h3>
                    <span className="text-xs uppercase tracking-wide text-slate-500">
                      {accountSnapshot.orders.length} orders
                    </span>
                  </div>
                  {accountSnapshot.orders.length > 0 ? (
                    <ul className="space-y-3 text-sm text-slate-300">
                      {accountSnapshot.orders.slice(0, 6).map(order => (
                        <li key={order.id} className="rounded-xl border border-slate-800/60 bg-slate-950/60 p-3">
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-slate-100">{order.symbol}</span>
                            <span className={`text-xs uppercase tracking-wide ${order.side === 'buy' ? 'text-emerald-300' : 'text-rose-300'}`}>
                              {order.side} {order.type}
                            </span>
                          </div>
                          <div className="mt-1 flex flex-wrap gap-2 text-xs text-slate-400">
                            <span>Qty {order.quantity}</span>
                            <span>Filled {order.filledQuantity}</span>
                            {order.limitPrice !== undefined && <span>Limit {order.limitPrice.toFixed(2)}</span>}
                            {order.stopPrice !== undefined && <span>Stop {order.stopPrice.toFixed(2)}</span>}
                          </div>
                          <div className="mt-1 text-xs text-slate-500">
                            Status: {order.status}
                          </div>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-slate-400">No open orders at this time.</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {regimeLoading ? (
            <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-6 text-sm text-slate-400">
              Loading strategy data...
            </div>
          ) : regimeError ? (
            <div className="rounded-2xl border border-rose-500/40 bg-rose-500/10 p-6 text-sm text-rose-200">
              {regimeError}
            </div>
          ) : regimeAnalysis && stage1 && stage2 && stage3 ? (
            <>
              <div className="grid gap-5 lg:grid-cols-3">
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-6 shadow-lg shadow-slate-900/40 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-slate-100">Stage 1 · Universe Filters</h3>
                    <span className={`text-xs uppercase tracking-wide ${stage1.passes ? 'text-emerald-300' : 'text-rose-300'}`}>
                      {stage1.passes ? 'Qualified' : 'Filtered'}
                    </span>
                  </div>
                  <ul className="space-y-2 text-sm text-slate-300">
                    <li><span className="text-slate-500">Liquidity Tier:</span> {stage1.tier === 'large' ? 'Large Cap Liquidity' : 'Mid/Small Cap Liquidity'}</li>
                    <li><span className="text-slate-500">IV Rank:</span> {formatNumber(stage1.metrics.ivRank, 2)} (≤ {stage1.thresholds.ivRank.toFixed(2)})</li>
                    <li><span className="text-slate-500">IV Δ 15m / 30m:</span> {formatNumber(stage1.metrics.ivDelta15m, 4)} / {formatNumber(stage1.metrics.ivDelta30m, 4)}</li>
                    <li><span className="text-slate-500">Volume / OI:</span> {formatNumber(stage1.metrics.volumeToOi, 2)} (≥ {stage1.thresholds.volumeToOi.toFixed(2)})</li>
                    <li><span className="text-slate-500">Whale Flow:</span> {stage1.whaleTrades.length > 0 ? `${stage1.whaleTrades.length} trades · ${stage1.whaleTrades[0].direction.toUpperCase()} ${stage1.whaleTrades[0].contracts}@${formatNumber(stage1.whaleTrades[0].midpointPrice, 2)}` : 'No outsized prints in last 30m'}</li>
                  </ul>
                </div>
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-6 shadow-lg shadow-slate-900/40 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-slate-100">Stage 2 · Regime Detection</h3>
                    <span className="text-xs uppercase tracking-wide text-slate-400">
                      {stage2.regime.charAt(0).toUpperCase() + stage2.regime.slice(1)}
                    </span>
                  </div>
                  <ul className="space-y-2 text-sm text-slate-300">
                    <li><span className="text-slate-500">Net GEX:</span> {formatNumber(stage2.netGex, 2)} ({stage2.regime === 'expansion' ? 'Short gamma' : 'Long gamma'})</li>
                    <li><span className="text-slate-500">Gamma Wall / Flip:</span> {stage2.gammaWall.toFixed(2)} {stage2.gammaFlipLevel ? ` · Flip ${stage2.gammaFlipLevel.toFixed(2)}` : ''}</li>
                    <li><span className="text-slate-500">Slope:</span> {stage2.slope} ({stage2.slopeStrength}) · Δ {formatNumber(stage2.recentSlopeDelta, 2)}</li>
                    <li><span className="text-slate-500">Regime Transition:</span> {transitionLabel}</li>
                    <li><span className="text-slate-500">Dominant Expiries:</span> {stage2.dominantExpirations.length > 0 ? stage2.dominantExpirations.map(exp => `${exp.expiration} (${exp.dte} DTE, ${formatNumber(exp.netGex, 2)})`).join(' · ') : 'No dominant expiration clusters'}</li>
                  </ul>
                  <p className="text-sm text-slate-400 leading-relaxed">{stage2.trendNarrative}</p>
                </div>
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-6 shadow-lg shadow-slate-900/40 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-slate-100">Stage 3 · Gamma Structure</h3>
                    <span className="text-xs uppercase tracking-wide text-slate-400">
                      Outlook: {stage3.rangeOutlook.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <p className="text-sm text-slate-400">
                    Price interaction: {stage3.priceInteraction?.replace(/_/g, ' ') || 'Neutral'}
                  </p>
                  <div className="grid gap-3 text-sm text-slate-300 sm:grid-cols-2">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">Call Walls</p>
                      <ul className="mt-1 space-y-1.5">
                        {stage3.callWalls.length > 0 ? stage3.callWalls.slice(0, 3).map(level => (
                          <li key={`call-${level.strike}`} className="rounded-lg border border-slate-800/70 bg-slate-950/60 px-3 py-2">
                            <p className="font-semibold text-slate-100">${level.strike.toFixed(2)}</p>
                            <p className="text-xs text-slate-400">
                              Strength {formatNumber(level.strength, 2)} · Z {level.zScore.toFixed(2)} · Dist {formatPercent(level.distancePct, 2)}
                            </p>
                          </li>
                        )) : <li className="text-xs text-slate-500">No major call walls detected.</li>}
                      </ul>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">Put Zones</p>
                      <ul className="mt-1 space-y-1.5">
                        {stage3.putZones.length > 0 ? stage3.putZones.slice(0, 3).map(level => (
                          <li key={`put-${level.strike}`} className="rounded-lg border border-slate-800/70 bg-slate-950/60 px-3 py-2">
                            <p className="font-semibold text-slate-100">${level.strike.toFixed(2)}</p>
                            <p className="text-xs text-slate-400">
                              Strength {formatNumber(level.strength, 2)} · Z {level.zScore.toFixed(2)} · Dist {formatPercent(level.distancePct, 2)}
                            </p>
                          </li>
                        )) : <li className="text-xs text-slate-500">No major put zones detected.</li>}
                      </ul>
                    </div>
                  </div>
                  <div className="mt-4 rounded-xl border border-slate-800/70 bg-slate-950/60 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Volume Confluence</p>
                    {volumeConfluence.length > 0 ? (
                      <ul className="space-y-1.5 text-xs text-slate-400">
                        {volumeConfluence.map((item, idx) => (
                          <li key={`confluence-${idx}`}>• {item}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-xs text-slate-500">No major HVN/POC alignment detected in the latest session.</p>
                    )}
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-6 shadow-lg shadow-slate-900/30">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <h3 className="text-lg font-semibold text-slate-100">Current Trade Signals</h3>
                  <span className="text-xs uppercase tracking-wide text-slate-500">
                    {regimeAnalysis.tradeSignals.length} potential setups
                  </span>
                </div>
                {regimeAnalysis.tradeSignals.length > 0 ? (
                  <div className="mt-3 space-y-3">
                    {regimeAnalysis.tradeSignals.map(signal => (
                      <div
                        key={signal.id}
                        className="rounded-xl border border-slate-800/70 bg-slate-950/70 p-4 text-sm text-slate-300 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between"
                      >
                        <div>
                          <p className="font-semibold text-slate-100">
                            {signal.symbol} · {signal.direction.toUpperCase()} {signal.strategy.toUpperCase()}
                          </p>
                          <p className="text-xs text-slate-400">
                            {signal.entry.triggerType} at {signal.entry.triggerLevel.toFixed(2)} · Stop {signal.stopLoss.toFixed(2)} · Target {signal.firstTarget.toFixed(2)}
                            {signal.secondaryTarget ? ` · Secondary ${signal.secondaryTarget.toFixed(2)}` : ''}
                          </p>
                        </div>
                        <div className="text-xs text-slate-400">
                          Risk {formatNumber(signal.riskPerShare ?? Math.abs(signal.entry.triggerLevel - signal.stopLoss), 2)} · Window {signal.timeframeMinutes ?? 30}m
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mt-3 text-sm text-slate-400">No qualified trade signals at this time.</p>
                )}
              </div>

              <section className="space-y-4">
                <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                  <h3 className="text-lg font-semibold text-slate-100">Lifecycle Management</h3>
                  <div className="text-xs uppercase tracking-wide text-slate-500">
                    {activeTrades.length} active for {sanitizeTicker(debouncedSymbol)} · {globalActiveTrades.length} across universe
                  </div>
                </div>
                {activeTrades.length > 0 ? (
                  <div className="grid gap-4">
                    {activeTrades.map(trade => (
                      <TradeLifecycleCard
                        key={trade.id}
                        trade={trade}
                        selected={selectedTradeId === trade.id}
                        onSelect={handleTradeSelect}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-6 text-sm text-slate-400">
                    No active lifecycle items for this symbol. Signals will appear here once they progress into entries.
                  </div>
                )}
              </section>
            </>
          ) : (
            <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-6 text-sm text-slate-400">
              Strategy snapshot unavailable for {sanitizeTicker(debouncedSymbol)}.
            </div>
          )}
        </section>
      </main>

      {/* Indicator Toggle Button - Fixed top-left */}
      <button
        onClick={() => setShowIndicatorPanel((prev) => !prev)}
        className="fixed top-20 left-6 z-20 flex items-center gap-2 px-4 py-3 rounded-xl bg-gradient-to-br from-slate-800 to-slate-900 hover:from-slate-750 hover:to-slate-850 text-white transition-all duration-200 text-sm font-semibold uppercase tracking-wide border-2 border-slate-700 hover:border-sky-400 shadow-[0_6px_20px_rgba(0,0,0,0.5),inset_0_1px_0_rgba(255,255,255,0.1)] hover:shadow-[0_8px_24px_rgba(0,0,0,0.6),inset_0_1px_0_rgba(255,255,255,0.15)] hover:scale-105"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        {showIndicatorPanel ? 'Hide' : 'Indicators'}
      </button>

      {/* Indicator Panel - Positioned outside main to avoid overflow-hidden clipping */}
      {showIndicatorPanel && (
        <div className="fixed top-36 left-6 z-10 pointer-events-none">
          <div className="pointer-events-auto bg-slate-950 border-2 border-slate-700 rounded-2xl shadow-[0_8px_32px_rgba(0,0,0,0.6)] p-5 max-h-[calc(100vh-10rem)] overflow-y-auto w-80" data-indicator-panel>
            <button
              onClick={() => setShowIndicatorPanel(false)}
              className="absolute right-3 top-3 rounded-full bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white transition-colors p-1"
              aria-label="Close indicator panel"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            <div className="flex items-center justify-between mb-5 pr-6">
              <h3 className="text-sm font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400 uppercase tracking-wider">
                Technical Indicators
              </h3>
              <div className="w-12 h-0.5 bg-gradient-to-r from-blue-500 to-transparent rounded-full"></div>
            </div>

            <div className="space-y-2.5 mb-6">
              <label className={indicatorToggleClasses}>
                <span className={indicatorToggleTextClass}>Volume</span>
                <input
                  type="checkbox"
                  checked={showVolume}
                  onChange={(e) => setShowVolume(e.target.checked)}
                  className={checkboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>

              <label className={indicatorToggleClasses}>
                <span className={indicatorToggleTextClass}>RSI (14)</span>
                <input
                  type="checkbox"
                  checked={showRSI}
                  onChange={(e) => setShowRSI(e.target.checked)}
                  className={checkboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>

              <label className={indicatorToggleClasses}>
                <span className={indicatorToggleTextClass}>MACD (12,26,9)</span>
                <input
                  type="checkbox"
                  checked={showMACD}
                  onChange={(e) => setShowMACD(e.target.checked)}
                  className={checkboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>

              <label className={indicatorToggleClasses}>
                <span className={indicatorToggleTextClass}>SMA 9 + Signals</span>
                <input
                  type="checkbox"
                  checked={showSMA9}
                  onChange={(e) => setShowSMA9(e.target.checked)}
                  className={checkboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>

              <label className={indicatorToggleClasses}>
                <span className={indicatorToggleTextClass}>SMA 20</span>
                <input
                  type="checkbox"
                  checked={showSMA20}
                  onChange={(e) => setShowSMA20(e.target.checked)}
                  className={checkboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>

              <label className={indicatorToggleClasses}>
                <span className={indicatorToggleTextClass}>SMA 50</span>
                <input
                  type="checkbox"
                  checked={showSMA50}
                  onChange={(e) => setShowSMA50(e.target.checked)}
                  className={checkboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>

              <label className={indicatorToggleClasses}>
                <span className={indicatorToggleTextClass}>SMA 200</span>
                <input
                  type="checkbox"
                  checked={showSMA200}
                  onChange={(e) => setShowSMA200(e.target.checked)}
                  className={checkboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>

              <label className={indicatorToggleClasses}>
                <span className={indicatorToggleTextClass}>Bollinger Bands</span>
                <input
                  type="checkbox"
                  checked={showBB}
                  onChange={(e) => setShowBB(e.target.checked)}
                  className={checkboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>

              <label className={indicatorToggleClasses}>
                <span className={indicatorToggleTextClass}>EMA (12,26)</span>
                <input
                  type="checkbox"
                  checked={showEMA}
                  onChange={(e) => setShowEMA(e.target.checked)}
                  className={checkboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>

              <label className={indicatorToggleClasses}>
                <span className={indicatorToggleTextClass}>Volume Profile</span>
                <input
                  type="checkbox"
                  checked={showVolumeProfile}
                  onChange={(e) => setShowVolumeProfile(e.target.checked)}
                  className={checkboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>
            </div>

            <div className="border-t-2 border-slate-700 pt-5 mt-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 uppercase tracking-wider">
                  Gamma Exposure
                </h3>
                <div className="w-12 h-0.5 bg-gradient-to-r from-purple-500 to-transparent rounded-full"></div>
              </div>

              <label className={`${gexToggleClasses} mb-4`}>
                <span className={gexToggleTextClass}>Show GEX</span>
                <input
                  type="checkbox"
                  checked={showGEX}
                  onChange={(e) => {
                    setShowGEX(e.target.checked);
                    if (e.target.checked) {
                      fetchGEXData();
                    }
                  }}
                  className={gexCheckboxClasses}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                />
              </label>

              {showGEX && (
                <div className="space-y-3.5">
                  <div>
                    <label className="text-xs font-bold text-slate-300 mb-2 block uppercase tracking-wide">GEX Mode</label>
                    <div className="relative">
                      <select
                        value={gexMode}
                        onChange={(e) => {
                          setGexMode(e.target.value as 'intraday' | 'swing');
                          fetchGEXData();
                        }}
                        className="appearance-none w-full bg-slate-900 px-4 py-2.5 pr-10 rounded-lg border-2 border-slate-700 text-sm text-white font-medium hover:border-purple-400 focus:border-purple-400 focus:outline-none transition-all cursor-pointer"
                      >
                        <option value="intraday" className="bg-slate-900 text-white">Intraday (3-7 DTE)</option>
                        <option value="swing" className="bg-slate-900 text-white">Swing (10-20 DTE)</option>
                      </select>
                      <svg className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </div>

                  {gexMeta && gexMeta.expirations && (
                    <div className="bg-slate-900 rounded-xl p-4 border-2 border-purple-500 shadow-lg shadow-purple-900/30">
                      <div className="text-xs font-bold text-purple-200 mb-3 uppercase tracking-wide">Expirations</div>
                      <div className="space-y-2">
                        {gexMeta.expirations.map((exp, i) => (
                          <div key={i} className="flex items-center justify-between text-xs bg-slate-950 rounded-lg px-3 py-2 border-2 border-slate-700">
                            <span className="text-slate-100 font-medium">{exp.date}</span>
                            <span className="text-slate-300 bg-slate-800 px-2 py-1 rounded border border-slate-700">({exp.dte} DTE)</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {gexSummary && (
                    <div className="bg-slate-900 rounded-xl p-4 space-y-3 border-2 border-purple-500 shadow-lg shadow-purple-900/30">
                      <div className="text-xs font-bold text-purple-200 mb-3 uppercase tracking-wide">GEX Summary</div>
                      <div className="flex items-center justify-between text-xs bg-green-900 rounded-lg px-3 py-2.5 border-2 border-green-500">
                        <span className="text-green-300 font-semibold">Call GEX</span>
                        <span className="font-mono text-green-200 font-bold">{formatGammaNotional(gexSummary.totalCallGex)}</span>
                      </div>
                      <div className="flex items-center justify-between text-xs bg-red-900 rounded-lg px-3 py-2.5 border-2 border-red-500">
                        <span className="text-red-300 font-semibold">Put GEX</span>
                        <span className="font-mono text-red-200 font-bold">{formatGammaNotional(gexSummary.totalPutGex)}</span>
                      </div>
                      <div className="flex items-center justify-between text-xs bg-blue-900 rounded-lg px-3 py-2.5 border-2 border-blue-500">
                        <span className="text-blue-400 font-bold">Net GEX</span>
                        <span className={`font-mono font-bold text-base ${gexSummary.netGex >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {formatGammaNotional(gexSummary.netGex)}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
