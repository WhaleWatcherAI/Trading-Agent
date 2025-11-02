'use client';

import { useCallback, useEffect, useMemo, useRef, useState, FormEvent } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import {
  RegimeBacktestResult,
  SectorBacktestResult,
  BacktestTrade,
  AggregatedBacktestSummary,
} from '@/types';

type AgentMode = 'scalp' | 'swing' | 'leaps';

const currency = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 2,
});

const numberFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 2,
});

const percentFormatter = new Intl.NumberFormat('en-US', {
  style: 'percent',
  maximumFractionDigits: 2,
});

const modes: AgentMode[] = ['scalp', 'swing', 'leaps'];

function isoToSeconds(iso: string): number {
  return new Date(iso).getTime() / 1000;
}

function buildMarkers(trades: BacktestTrade[]) {
  const markers = trades.flatMap(trade => {
    const entryColor = trade.direction === 'long' ? '#22c55e' : '#ef4444';
    const exitColor = trade.profit >= 0 ? '#22c55e' : '#ef4444';
    const profitText = `${trade.profit >= 0 ? '+' : ''}${numberFormatter.format(trade.profit)}`;

    return [
      {
        time: isoToSeconds(trade.entryTimestamp),
        position: 'belowBar' as const,
        color: entryColor,
        shape: trade.direction === 'long' ? 'arrowUp' as const : 'arrowDown' as const,
        text: `${trade.direction === 'long' ? 'L' : 'S'} ${numberFormatter.format(trade.entryPrice)}`,
      },
      {
        time: isoToSeconds(trade.exitTimestamp),
        position: 'aboveBar' as const,
        color: exitColor,
        shape: trade.direction === 'long' ? 'arrowDown' as const : 'arrowUp' as const,
        text: `${profitText} (${trade.exitReason})`,
      },
    ];
  });

  return markers.sort((a, b) => Number(a.time) - Number(b.time));
}

interface UseChartParams {
  sector: SectorBacktestResult | null;
}

function useBacktestChart({ sector }: UseChartParams) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<any>(null);
  const candleSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;

    if (!chartRef.current) {
      const chart = createChart(container, {
        layout: {
          background: { type: ColorType.Solid, color: '#0f172a' },
          textColor: '#cbd5f5',
        },
        width: container.clientWidth,
        height: 420,
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
        },
        rightPriceScale: {
          borderColor: '#1e293b',
        },
        grid: {
          horzLines: { color: '#1e293b' },
          vertLines: { color: '#1e293b' },
        },
      });

      const candleSeries = (chart as any).addCandlestickSeries({
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444',
      });

      const volumeSeries = (chart as any).addHistogramSeries({
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
        color: '#60a5fa',
        base: 0,
      });

      chart.priceScale('volume').applyOptions({
        scaleMargins: {
          top: 0.85,
          bottom: 0,
        },
      });

      chartRef.current = chart;
      candleSeriesRef.current = candleSeries;
      volumeSeriesRef.current = volumeSeries;
    }

    const handleResize = () => {
      if (!chartRef.current || !containerRef.current) return;
      chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
    };

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current as any;
    const candleSeries = candleSeriesRef.current as any;
    const volumeSeries = volumeSeriesRef.current as any;
    if (!chart || !candleSeries || !volumeSeries) return;

    if (!sector) {
      candleSeries.setData([]);
      volumeSeries.setData([]);
      candleSeries.setMarkers([]);
      return;
    }

    const candles = sector.timeline
      .filter(point => point.price)
      .map(point => ({
        time: isoToSeconds(point.timestamp),
        open: point.price!.open,
        high: point.price!.high,
        low: point.price!.low,
        close: point.price!.close,
      }));

    candleSeries.setData(candles);

    const volumes = sector.timeline
      .filter(point => point.price)
      .map(point => ({
        time: isoToSeconds(point.timestamp),
        value: point.price!.volume,
        color: point.netPremium >= 0 ? '#38bdf8' : '#f97316',
      }));

    volumeSeries.setData(volumes);

    candleSeries.setMarkers(buildMarkers(sector.trades));
    chart.timeScale().fitContent();
  }, [sector]);

  return containerRef;
}

export default function RegimeBacktestPage() {
  const [date, setDate] = useState('2025-10-31');
  const [mode, setMode] = useState<AgentMode>('scalp');
  const [interval, setInterval] = useState(1);
  const [usePrices, setUsePrices] = useState(true);
  const [useLiveFlow, setUseLiveFlow] = useState(false);
  const [symbolsInput, setSymbolsInput] = useState('');
  const [lookbackMinutes, setLookbackMinutes] = useState(240);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RegimeBacktestResult | null>(null);
  const [selectedSectorIdx, setSelectedSectorIdx] = useState(0);

  const selectedSector = useMemo<SectorBacktestResult | null>(() => {
    if (!result || result.sectors.length === 0) return null;
    return result.sectors[selectedSectorIdx] || result.sectors[0];
  }, [result, selectedSectorIdx]);

  const chartContainerRef = useBacktestChart({ sector: selectedSector });

  const aggregated: AggregatedBacktestSummary | null = result ? result.aggregated : null;

  const fetchBacktest = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        date,
        mode,
        interval: String(interval),
        prices: usePrices ? 'true' : 'false',
      });

      if (useLiveFlow) {
        params.set('flow', 'live');
        params.set('lookback', String(lookbackMinutes));
      }

      if (symbolsInput.trim().length > 0) {
        params.set(
          'symbols',
          symbolsInput
            .split(',')
            .map(symbol => symbol.trim().toUpperCase())
            .filter(Boolean)
            .join(','),
        );
      }

      const response = await fetch(`/api/regime/backtest?${params.toString()}`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload?.details || payload?.error || response.statusText);
      }

      const payload: RegimeBacktestResult = await response.json();
      setResult(payload);
      setSelectedSectorIdx(0);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch backtest results');
    } finally {
      setLoading(false);
    }
  }, [date, mode, interval, usePrices, useLiveFlow, lookbackMinutes, symbolsInput]);

  useEffect(() => {
    fetchBacktest().catch(() => {
      /* handled above */
    });
  }, []); // initial load

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    fetchBacktest().catch(() => {
      /* handled */
    });
  };

  return (
    <div className="px-6 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-semibold text-slate-100">Regime Backtest</h1>
        <p className="mt-2 text-sm text-slate-400">
          Replay intraday flow, overlay simulated trade lifecycle, and review performance for a chosen session.
        </p>
      </header>

      <form
        onSubmit={handleSubmit}
        className="grid gap-4 rounded-2xl border border-slate-800 bg-slate-900/70 p-4 md:grid-cols-2 xl:grid-cols-4"
      >
        <label className="flex flex-col gap-1 text-sm text-slate-300">
          <span className="font-semibold text-slate-100">Session Date</span>
          <input
            type="date"
            value={date}
            onChange={event => setDate(event.target.value)}
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-sky-500 focus:outline-none"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm text-slate-300">
          <span className="font-semibold text-slate-100">Mode</span>
          <select
            value={mode}
            onChange={event => setMode(event.target.value as AgentMode)}
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-sky-500 focus:outline-none"
          >
            {modes.map(item => (
              <option key={item} value={item}>
                {item.toUpperCase()}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-sm text-slate-300">
          <span className="font-semibold text-slate-100">Interval (minutes)</span>
          <input
            type="number"
            min={1}
            max={15}
            value={interval}
            onChange={event => setInterval(Number(event.target.value))}
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-sky-500 focus:outline-none"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm text-slate-300">
          <span className="font-semibold text-slate-100">Filter Symbols (comma separated)</span>
          <input
            type="text"
            value={symbolsInput}
            onChange={event => setSymbolsInput(event.target.value)}
            placeholder="SPY,QQQ"
            className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-sky-500 focus:outline-none"
          />
        </label>

        <label className="flex items-center gap-2 text-sm text-slate-300 md:col-span-2">
          <input
            type="checkbox"
            checked={usePrices}
            onChange={event => setUsePrices(event.target.checked)}
            className="h-5 w-5 rounded border border-slate-600 bg-slate-950 text-sky-500 focus:ring-sky-500"
          />
          <span className="font-semibold text-slate-100">Include Tradier timesales</span>
        </label>

        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={useLiveFlow}
            onChange={event => setUseLiveFlow(event.target.checked)}
            className="h-5 w-5 rounded border border-slate-600 bg-slate-950 text-sky-500 focus:ring-sky-500"
          />
          <span className="font-semibold text-slate-100">Append live Unusual Whales flow</span>
        </label>

        {useLiveFlow && (
          <label className="flex flex-col gap-1 text-sm text-slate-300">
            <span className="font-semibold text-slate-100">Live lookback (minutes)</span>
            <input
              type="number"
              min={30}
              max={1440}
              value={lookbackMinutes}
              onChange={event => setLookbackMinutes(Number(event.target.value))}
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 focus:border-sky-500 focus:outline-none"
            />
          </label>
        )}

        <button
          type="submit"
          disabled={loading}
          className="md:col-span-2 xl:col-span-4 rounded-xl bg-sky-500 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-700"
        >
          {loading ? 'Running backtest…' : 'Run backtest'}
        </button>
      </form>

      {error && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
          {error}
        </div>
      )}

      {result && (
        <section className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
              <p className="text-xs uppercase tracking-wide text-slate-500">Total Profit</p>
              <p className="mt-1 text-2xl font-semibold text-slate-100">
                {currency.format(aggregated?.totalProfit ?? 0)}
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
              <p className="text-xs uppercase tracking-wide text-slate-500">Trade Count</p>
              <p className="mt-1 text-2xl font-semibold text-slate-100">
                {aggregated?.tradeCount ?? 0}
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
              <p className="text-xs uppercase tracking-wide text-slate-500">Win Rate</p>
              <p className="mt-1 text-2xl font-semibold text-slate-100">
                {aggregated?.winRate !== undefined ? percentFormatter.format(aggregated.winRate) : '—'}
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
              <p className="text-xs uppercase tracking-wide text-slate-500">Net Premium</p>
              <p className="mt-1 text-2xl font-semibold text-slate-100">
                {currency.format((aggregated?.netPremium ?? 0) / 1_000_000)}M
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {result.sectors.map((sector, index) => {
              const isActive = index === selectedSectorIdx;
              return (
                <button
                  key={sector.sector}
                  type="button"
                  onClick={() => setSelectedSectorIdx(index)}
                  className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                    isActive
                      ? 'bg-sky-500 text-slate-950 shadow'
                      : 'bg-slate-800 text-slate-200 hover:bg-slate-700'
                  }`}
                >
                  {sector.sector}
                  {sector.mappedSymbol ? ` · ${sector.mappedSymbol}` : ''}
                </button>
              );
            })}
          </div>

          <div className="grid gap-6 xl:grid-cols-[2fr_1fr]">
            <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
              <h2 className="text-lg font-semibold text-slate-100">Price & Trade Overlay</h2>
              <div ref={chartContainerRef} className="mt-4 h-[420px] w-full" />
            </div>

            {selectedSector && (
              <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
                <h2 className="text-lg font-semibold text-slate-100">Session Summary</h2>
                <dl className="mt-4 space-y-2 text-sm text-slate-300">
                  <div className="flex items-center justify-between">
                    <dt>Trades</dt>
                    <dd>{selectedSector.summary.tradeCount}</dd>
                  </div>
                  <div className="flex items-center justify-between">
                    <dt>Win rate</dt>
                    <dd>
                      {selectedSector.summary.winRate !== undefined
                        ? percentFormatter.format(selectedSector.summary.winRate)
                        : '—'}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between">
                    <dt>Total P&amp;L</dt>
                    <dd>{currency.format(selectedSector.summary.totalProfit)}</dd>
                  </div>
                  <div className="flex items-center justify-between">
                    <dt>Max drawdown</dt>
                    <dd>
                      {selectedSector.summary.maxDrawdown !== undefined
                        ? currency.format(selectedSector.summary.maxDrawdown)
                        : '—'}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between">
                    <dt>Avg. duration</dt>
                    <dd>
                      {selectedSector.summary.averageDurationMinutes !== undefined
                        ? `${numberFormatter.format(selectedSector.summary.averageDurationMinutes)} min`
                        : '—'}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between">
                    <dt>Bias transitions</dt>
                    <dd>{selectedSector.summary.regimeTransitions}</dd>
                  </div>
                  <div className="flex items-center justify-between">
                    <dt>Dominant bias</dt>
                    <dd className="capitalize">{selectedSector.summary.dominantBias}</dd>
                  </div>
                  <div className="flex items-center justify-between">
                    <dt>Price change</dt>
                    <dd>
                      {selectedSector.summary.priceChangePct !== undefined
                        ? `${numberFormatter.format(selectedSector.summary.priceChangePct)}%`
                        : '—'}
                    </dd>
                  </div>
                </dl>
              </div>
            )}
          </div>

          {selectedSector && (
            <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-slate-900/70">
              <table className="min-w-full text-sm text-slate-300">
                <thead className="bg-slate-800 text-xs uppercase tracking-wide text-slate-400">
                  <tr>
                    <th className="px-4 py-3 text-left">Direction</th>
                    <th className="px-4 py-3 text-left">Entry</th>
                    <th className="px-4 py-3 text-left">Exit</th>
                    <th className="px-4 py-3 text-right">Duration (min)</th>
                    <th className="px-4 py-3 text-right">Entry Price</th>
                    <th className="px-4 py-3 text-right">Exit Price</th>
                    <th className="px-4 py-3 text-right">Profit</th>
                    <th className="px-4 py-3 text-right">Profit %</th>
                    <th className="px-4 py-3 text-right">Trade Cost</th>
                    <th className="px-4 py-3 text-left">Exit Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedSector.trades.length === 0 && (
                    <tr>
                      <td colSpan={10} className="px-4 py-6 text-center text-slate-500">
                        No trades generated for this sector/timeframe.
                      </td>
                    </tr>
                  )}
                  {selectedSector.trades.map(trade => (
                    <tr key={trade.id} className="border-t border-slate-800">
                      <td className="px-4 py-3 font-semibold capitalize text-slate-100">
                        {trade.direction}
                      </td>
                      <td className="px-4 py-3">{new Date(trade.entryTimestamp).toLocaleTimeString()}</td>
                      <td className="px-4 py-3">{new Date(trade.exitTimestamp).toLocaleTimeString()}</td>
                      <td className="px-4 py-3 text-right">{trade.durationMinutes}</td>
                      <td className="px-4 py-3 text-right">{currency.format(trade.entryPrice)}</td>
                      <td className="px-4 py-3 text-right">{currency.format(trade.exitPrice)}</td>
                      <td
                        className={`px-4 py-3 text-right font-semibold ${
                          trade.profit >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {currency.format(trade.profit)}
                      </td>
                      <td
                        className={`px-4 py-3 text-right ${
                          trade.profitPct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {percentFormatter.format(trade.profitPct)}
                      </td>
                      <td className="px-4 py-3 text-right">{currency.format(trade.tradeCost)}</td>
                      <td className="px-4 py-3 text-left capitalize">{trade.exitReason.replace(/_/g, ' ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {result.notes.length > 0 && (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
              <h2 className="text-lg font-semibold text-slate-100">Diagnostics</h2>
              <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-400">
                {result.notes.map((note, index) => (
                  <li key={`${note}-${index}`}>{note}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
