'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  VolatilityRegimeResponse,
  VolatilityRegimeAnalysis,
  RegimeStage1Result,
  RegimeGexLevel,
  RegimeTradeSignal,
  RegimeTradeLifecycle,
} from '@/types';

type AgentMode = 'scalp' | 'swing' | 'leaps';

const DEFAULT_SYMBOLS = 'SPY,QQQ,AAPL,NVDA,TSLA,MSFT';

const MODE_LABELS: Record<AgentMode, string> = {
  scalp: 'Scalp (3-7 DTE)',
  swing: 'Swing (10-20 DTE)',
  leaps: 'LEAPS (45+ DTE)',
};

interface UniverseCardProps {
  result: RegimeStage1Result;
  isActive: boolean;
  onSelect: (symbol: string) => void;
}

interface GexProfileProps {
  levels: RegimeGexLevel[];
  price: number;
}

interface TradeSignalCardProps {
  signal: RegimeTradeSignal;
}

const formatNumber = (value: number | null | undefined, digits = 2): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  if (Math.abs(value) >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(digits)}M`;
  }
  if (Math.abs(value) >= 1_000) {
    return `${(value / 1_000).toFixed(digits)}K`;
  }
  return value.toFixed(digits);
};

const formatPercent = (value: number | null | undefined, digits = 2): string => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return `${(value * 100).toFixed(digits)}%`;
};

const UniverseCard = ({ result, isActive, onSelect }: UniverseCardProps) => {
  const ivDelta15 = result.metrics.ivDelta15m ?? null;
  const ivDelta30 = result.metrics.ivDelta30m ?? null;
  const ivMomentum = ivDelta15 ?? ivDelta30 ?? null;

  const statusClasses = result.passes
    ? 'border-emerald-400/50 bg-emerald-500/10 shadow-emerald-500/30'
    : 'border-rose-400/40 bg-rose-500/10 shadow-rose-500/30';

  const statusLabel = result.passes ? 'Qualified' : 'Filtered';

  return (
    <button
      type="button"
      onClick={() => onSelect(result.symbol)}
      className={`w-full text-left rounded-2xl border p-4 transition-all duration-200 hover:scale-[1.01] focus:outline-none focus:ring-2 focus:ring-sky-400 ${
        isActive ? statusClasses : 'border-slate-800/80 bg-slate-900/70 shadow-lg'
      }`}
    >
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold text-slate-100">{result.symbol}</h3>
          <p className="text-sm text-slate-400 capitalize">
            {result.tier === 'large' ? 'Large Cap Liquidity' : 'Mid/Small Cap Focus'}
          </p>
        </div>
        <span
          className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${
            result.passes ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-200'
          }`}
        >
          {statusLabel}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 text-sm text-slate-300 sm:grid-cols-4">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">IV Rank</p>
          <p className="font-semibold">{formatNumber(result.metrics.ivRank, 2)}</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">IV Δ (approx)</p>
          <p
            className={`font-semibold ${
              ivMomentum && ivMomentum > 0 ? 'text-emerald-300' : 'text-rose-200'
            }`}
          >
            {ivDelta15 !== null
              ? `${formatNumber(ivDelta15, 4)} (15m)`
              : ivDelta30 !== null
                ? `${formatNumber(ivDelta30, 4)} (30m)`
                : '—'}
          </p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">Vol / OI</p>
          <p className="font-semibold">{formatNumber(result.metrics.volumeToOi, 2)}</p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">Whale Trades</p>
          <p className="font-semibold">{result.whaleTrades.length}</p>
        </div>
      </div>

      <div className="mt-3 text-xs text-slate-400">
        {result.passes ? (
          <p>{result.notes[0]}</p>
        ) : (
          <p>
            Missed:{' '}
            {result.failedCriteria.length > 0 ? result.failedCriteria.join(', ') : 'Not enough data (retry soon)'}
          </p>
        )}
      </div>
    </button>
  );
};

const GexProfile = ({ levels, price }: GexProfileProps) => {
  const sortedLevels = useMemo(() => {
    return [...levels].sort((a, b) => a.strike - b.strike).slice(0, 25);
  }, [levels]);

  const maxAbs = sortedLevels.reduce((max, level) => Math.max(max, Math.abs(level.netGex)), 1);

  return (
    <div className="mt-4 rounded-xl border border-slate-800/80 bg-slate-900/70 p-4 shadow-lg">
      <div className="mb-3 flex items-baseline justify-between">
        <h4 className="text-lg font-semibold text-slate-100">Per-Strike GEX Profile</h4>
        <div className="text-xs text-slate-500">
          Current price reference&nbsp;
          <span className="rounded-full bg-sky-500/10 px-2 py-0.5 text-sky-300">${price.toFixed(2)}</span>
        </div>
      </div>
      <div className="flex w-full flex-col gap-2">
        {sortedLevels.map(level => {
          const barWidth = Math.min(100, (Math.abs(level.netGex) / maxAbs) * 100);
          const isPositive = level.netGex >= 0;
          const color = isPositive ? 'bg-emerald-400/80' : 'bg-rose-400/80';
          const alignment = isPositive ? 'ml-auto' : 'mr-auto';
          return (
            <div key={`${level.strike}-${level.classification}`} className="grid grid-cols-[80px_1fr_60px] items-center gap-2 text-xs">
              <span className="font-medium text-slate-300">${level.strike.toFixed(2)}</span>
              <div className="relative h-3 overflow-hidden rounded-full bg-slate-800/60">
                <div
                  className={`${color} ${alignment} h-full rounded-full shadow`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>
              <span className={`text-right ${isPositive ? 'text-emerald-300' : 'text-rose-300'}`}>
                {level.classification === 'call_wall' ? 'Call Wall' : level.classification === 'put_zone' ? 'Put Zone' : ''}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

const TradeSignalCard = ({ signal }: TradeSignalCardProps) => {
  const directionBadge =
    signal.direction === 'long'
      ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
      : 'bg-rose-500/10 text-rose-200 border border-rose-500/30';

  return (
    <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-5 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">{signal.regime} regime</p>
          <h4 className="text-xl font-semibold text-slate-100">{signal.action.toUpperCase()} {signal.strategy}</h4>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${directionBadge}`}>
          {signal.direction} · {signal.positionSize} size
        </span>
      </div>

      <div className="mt-4 grid gap-4 text-sm text-slate-300 md:grid-cols-2">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">Entry Trigger</p>
          <p className="font-semibold">
            {signal.entry.triggerType} at ${signal.entry.triggerLevel.toFixed(2)} (spot ${signal.entry.price.toFixed(2)})
          </p>
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">Risk Controls</p>
          <p className="font-semibold">
            Stop ${signal.stopLoss.toFixed(2)} · TP1 ${signal.firstTarget.toFixed(2)}
            {signal.secondaryTarget ? ` · TP2 ${signal.secondaryTarget.toFixed(2)}` : ''}
          </p>
        </div>
      </div>

      <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-slate-300">
        {signal.rationale.map(item => (
          <li key={item}>{item}</li>
        ))}
      </ul>

      {signal.whaleConfirmation && (
        <div className="mt-4 rounded-xl border border-sky-500/30 bg-sky-500/10 p-3 text-xs text-sky-200">
          Whale flow confirmation: {signal.whaleConfirmation.direction.toUpperCase()} {signal.whaleConfirmation.contracts}{' '}
          contracts on {signal.whaleConfirmation.optionType.toUpperCase()} strike ${signal.whaleConfirmation.strike.toFixed(2)} exp {signal.whaleConfirmation.expiration}
        </div>
      )}
    </div>
  );
};

export default function VolatilityRegimePage() {
  const [mode, setMode] = useState<AgentMode>('scalp');
  const [symbolInput, setSymbolInput] = useState<string>(DEFAULT_SYMBOLS);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<VolatilityRegimeResponse | null>(null);
  const [activeSymbol, setActiveSymbol] = useState<string | null>(null);
  const [showJson, setShowJson] = useState<boolean>(false);

  const runAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      params.append('mode', mode);
      if (symbolInput.trim()) {
        params.append('symbols', symbolInput.trim());
      }
      const response = await fetch(`/api/regime?${params.toString()}`);
      if (!response.ok) {
        throw new Error('Failed to fetch regime analysis');
      }
      const payload: VolatilityRegimeResponse = await response.json();
      setData(payload);
      if (!activeSymbol && payload.analyses.length > 0) {
        setActiveSymbol(payload.analyses[0].symbol);
      } else if (activeSymbol) {
        const stillExists = payload.analyses.find(analysis => analysis.symbol === activeSymbol);
        if (!stillExists && payload.analyses.length > 0) {
          setActiveSymbol(payload.analyses[0].symbol);
        }
      }
    } catch (err: any) {
      setError(err.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedAnalysis: VolatilityRegimeAnalysis | undefined = useMemo(() => {
    if (!data || data.analyses.length === 0) return undefined;
    return data.analyses.find(item => item.symbol === activeSymbol) || data.analyses[0];
  }, [data, activeSymbol]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-900 text-white pb-16">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-slate-100">Volatility Regime-Based Options Strategy Agent</h1>
          <p className="mt-2 text-slate-400">
            Multi-stage CLI-inspired workflow that mirrors the AI Trading Agent, but tuned for gamma regime detection,
            whale flow validation, and option-centric risk management.
          </p>
        </header>

        <div className="mb-8 rounded-3xl border border-slate-800/80 bg-slate-900/70 p-6 shadow-2xl">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div className="grid flex-1 grid-cols-1 gap-4 md:grid-cols-3">
              <div>
                <label className="mb-2 block text-sm font-semibold text-slate-300">Mode</label>
                <div className="flex gap-2">
                  {(Object.keys(MODE_LABELS) as AgentMode[]).map(value => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setMode(value)}
                      className={`flex-1 rounded-xl border px-3 py-2 text-sm font-semibold transition ${
                        mode === value
                          ? 'border-sky-500/60 bg-sky-500/20 text-sky-200'
                          : 'border-slate-700 bg-slate-900 text-slate-400 hover:border-slate-600'
                      }`}
                    >
                      {MODE_LABELS[value]}
                    </button>
                  ))}
                </div>
              </div>
              <div className="md:col-span-2">
                <label className="mb-2 block text-sm font-semibold text-slate-300">Universe Symbols</label>
                <input
                  type="text"
                  value={symbolInput}
                  onChange={(event) => setSymbolInput(event.target.value)}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950/80 px-4 py-2 text-sm text-slate-100 outline-none focus:border-sky-500 focus:ring-2 focus:ring-sky-500/40"
                  placeholder="Comma separated tickers e.g. SPY,QQQ,AAPL"
                />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={runAnalysis}
                disabled={loading}
                className="rounded-xl bg-sky-500 px-5 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-sky-500/40 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                {loading ? 'Scanning…' : 'Run Regime Scan'}
              </button>
              <button
                type="button"
                onClick={() => setShowJson(prev => !prev)}
                className="rounded-xl border border-slate-700 px-4 py-3 text-sm font-semibold text-slate-300 hover:border-sky-500/50 hover:text-sky-200"
              >
                {showJson ? 'Hide JSON' : 'Show JSON'}
              </button>
            </div>
          </div>
          {error && (
            <div className="mt-4 rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-200">
              {error}
            </div>
          )}
        </div>

        <section className="mb-10">
          <h2 className="mb-4 text-2xl font-semibold text-slate-200">Stage 1 · Universe Filtering</h2>
          <p className="mb-6 text-sm text-slate-400">
            Applying liquidity-tier aware filters: IVR targets, Volume/OI, intraday IV momentum, and whale flow confirmations.
            Click a ticker to drill into its gamma regime profile.
          </p>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {data?.universe.map(result => (
              <UniverseCard
                key={result.symbol}
                result={result}
                isActive={result.symbol === selectedAnalysis?.symbol}
                onSelect={setActiveSymbol}
              />
            ))}
            {!data && !loading && (
              <div className="rounded-2xl border border-dashed border-slate-700/60 bg-slate-900/40 p-6 text-center text-sm text-slate-500">
                Run the scan to populate the universe candidates.
              </div>
            )}
          </div>
        </section>

        {selectedAnalysis && (
          <section className="space-y-8">
            <header className="flex flex-col gap-2 rounded-3xl border border-slate-800/80 bg-gradient-to-r from-slate-900/80 to-slate-900/40 p-6 shadow-2xl md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Stage 2 & 3 Focus</p>
                <h2 className="mt-2 text-3xl font-bold text-slate-100">
                  {selectedAnalysis.symbol} · {MODE_LABELS[selectedAnalysis.mode as AgentMode]}
                </h2>
              </div>
              <div className="flex gap-3 text-sm text-slate-300">
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-500">Net GEX</p>
                  <p className={selectedAnalysis.stage2.netGex <= 0 ? 'font-semibold text-rose-200' : 'font-semibold text-emerald-300'}>
                    ${selectedAnalysis.stage2.netGex.toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-500">Regime</p>
                  <p className="font-semibold capitalize text-slate-200">{selectedAnalysis.stage2.regime}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-500">Gamma Wall</p>
                  <p className="font-semibold text-slate-200">${selectedAnalysis.stage2.gammaWall.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-500">Slope</p>
                  <p className="font-semibold capitalize text-slate-200">
                    {selectedAnalysis.stage2.slope}
                    <span className="ml-1 text-xs uppercase tracking-wide text-slate-500">
                      ({selectedAnalysis.stage2.slopeStrength})
                    </span>
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-500">Gamma Flip Δ</p>
                  <p className="font-semibold text-slate-200">
                    {selectedAnalysis.stage2.gammaFlipDistance !== undefined
                      ? `${selectedAnalysis.stage2.gammaFlipDistance > 0 ? '+' : ''}${selectedAnalysis.stage2.gammaFlipDistance.toFixed(2)}`
                      : 'n/a'}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-500">Regime Shift</p>
                  <p className="font-semibold capitalize text-slate-200">
                    {selectedAnalysis.stage2.regimeTransition.replace(/_/g, ' ')}
                  </p>
                </div>
              </div>
            </header>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <div className="space-y-6 lg:col-span-2">
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-6 shadow-xl">
                  <h3 className="text-xl font-semibold text-slate-100">Regime Detection (Stage 2)</h3>
                  <div className="mt-4 grid gap-4 text-sm text-slate-300 sm:grid-cols-2">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">Expirations in scope</p>
                      <p className="font-semibold">
                        {selectedAnalysis.stage2.expirations
                          .map(item => `${item.date} (${item.dte} DTE)`)
                          .join(' · ')}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">Gamma Flip Level</p>
                      <p className="font-semibold">
                        {selectedAnalysis.stage2.gammaFlipLevel
                          ? `$${selectedAnalysis.stage2.gammaFlipLevel.toFixed(2)}`
                          : 'Not detected'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">Call GEX Notional</p>
                      <p className="font-semibold">
                        ${formatNumber(selectedAnalysis.stage2.totalCallGex, 2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">Put GEX Notional</p>
                      <p className="font-semibold">
                        ${formatNumber(selectedAnalysis.stage2.totalPutGex, 2)}
                      </p>
                    </div>
                    <div className="sm:col-span-2">
                      <p className="text-xs uppercase tracking-wide text-slate-500">Dominant Expirations</p>
                      <p className="font-semibold text-slate-200">
                        {selectedAnalysis.stage2.dominantExpirations.length > 0
                          ? selectedAnalysis.stage2.dominantExpirations
                              .map(exp => `${exp.expiration} (${exp.dte} DTE · ${formatNumber(exp.netGex, 2)})`)
                              .join(' · ')
                          : 'Insufficient data'}
                      </p>
                    </div>
                  </div>

                  <div className="mt-5 rounded-xl border border-slate-800/60 bg-slate-950/50 p-4 text-sm text-slate-300">
                    <p className="font-semibold text-slate-100">Gamma Narrative</p>
                    <p className="mt-2 text-slate-400 leading-relaxed">
                      {selectedAnalysis.stage2.trendNarrative}
                    </p>
                    {selectedAnalysis.stage2.recentSlopeDelta !== undefined && (
                      <p className="mt-2 text-xs uppercase tracking-wide text-slate-500">
                        Recent slope delta: {formatNumber(selectedAnalysis.stage2.recentSlopeDelta, 2)}
                      </p>
                    )}
                  </div>

                  <div className="mt-6 rounded-xl border border-slate-800/60 bg-slate-950/50 p-4 text-sm text-slate-300">
                    <p className="font-semibold text-slate-100">Interpretation</p>
                    {selectedAnalysis.stage2.regime === 'expansion' ? (
                      <p className="mt-2 text-slate-400">
                        Dealers are short gamma. Expect amplified moves when price challenges structural GEX levels.
                        Watch for breakouts with confirmation from whale flow and rising IV.
                      </p>
                    ) : (
                      <p className="mt-2 text-slate-400">
                        Dealers are long gamma. Expect range-bound action with strong mean-reversion at major call/put walls.
                        Premium-selling strategies preferred until GEX slope shifts to falling.
                      </p>
                    )}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-6 shadow-xl">
                  <h3 className="text-xl font-semibold text-slate-100">Stage 3 · Gamma Wall Drill Down</h3>
                  <div className="mt-3 grid gap-3 text-sm text-slate-300 sm:grid-cols-4">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">Primary Call Wall</p>
                      <p className="font-semibold">
                        {selectedAnalysis.stage3.callWalls[0]
                          ? `$${selectedAnalysis.stage3.callWalls[0].strike.toFixed(2)}`
                          : 'n/a'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">Primary Put Zone</p>
                      <p className="font-semibold">
                        {selectedAnalysis.stage3.putZones[0]
                          ? `$${selectedAnalysis.stage3.putZones[0].strike.toFixed(2)}`
                          : 'n/a'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">Price Interaction</p>
                      <p className="font-semibold capitalize">
                        {selectedAnalysis.stage3.priceInteraction?.replace('-', ' ') || 'Neutral'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500">Range Outlook</p>
                      <p className="font-semibold capitalize">
                        {selectedAnalysis.stage3.rangeOutlook.replace('_', ' ')}
                      </p>
                    </div>
                  </div>
                  <GexProfile levels={selectedAnalysis.stage3.profile} price={selectedAnalysis.price} />
                  <div className="mt-5 grid gap-4 text-sm text-slate-300 md:grid-cols-2">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Call Wall Focus</p>
                      <ul className="space-y-2">
                        {selectedAnalysis.stage3.callWalls.length > 0 ? (
                          selectedAnalysis.stage3.callWalls.map(level => (
                            <li key={`call-${level.strike}`} className="rounded-xl border border-slate-800/60 bg-slate-950/60 p-3">
                              <p className="font-semibold text-slate-100">${level.strike.toFixed(2)}</p>
                              <p className="text-xs text-slate-400">
                                Strength {formatNumber(level.strength, 2)} · Z {level.zScore.toFixed(2)} · Distance {formatPercent(level.distancePct, 2)}
                                {level.isNearPrice ? ' · Near price' : ''}
                              </p>
                            </li>
                          ))
                        ) : (
                          <li className="rounded-xl border border-slate-800/60 bg-slate-950/60 p-3 text-xs text-slate-400">
                            No significant call walls detected in scope.
                          </li>
                        )}
                      </ul>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-500 mb-2">Put Zone Focus</p>
                      <ul className="space-y-2">
                        {selectedAnalysis.stage3.putZones.length > 0 ? (
                          selectedAnalysis.stage3.putZones.map(level => (
                            <li key={`put-${level.strike}`} className="rounded-xl border border-slate-800/60 bg-slate-950/60 p-3">
                              <p className="font-semibold text-slate-100">${level.strike.toFixed(2)}</p>
                              <p className="text-xs text-slate-400">
                                Strength {formatNumber(level.strength, 2)} · Z {level.zScore.toFixed(2)} · Distance {formatPercent(level.distancePct, 2)}
                                {level.isNearPrice ? ' · Near price' : ''}
                              </p>
                            </li>
                          ))
                        ) : (
                          <li className="rounded-xl border border-slate-800/60 bg-slate-950/60 p-3 text-xs text-slate-400">
                            No significant put zones detected in scope.
                          </li>
                        )}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              <aside className="space-y-6">
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-5 shadow-xl">
                  <h3 className="text-lg font-semibold text-slate-100">Stage 1 Snapshot</h3>
                  <ul className="mt-4 space-y-3 text-sm text-slate-300">
                    <li className="flex justify-between">
                      <span>Liquidity Tier</span>
                      <span className="font-semibold capitalize">{selectedAnalysis.stage1.tier}</span>
                    </li>
                    <li className="flex justify-between">
                      <span>IV Rank</span>
                      <span className="font-semibold">{formatNumber(selectedAnalysis.stage1.metrics.ivRank, 3)}</span>
                    </li>
                    <li className="flex justify-between">
                      <span>IV Δ (approx)</span>
                      <span
                        className={`font-semibold ${
                          ((selectedAnalysis.stage1.metrics.ivDelta15m ?? selectedAnalysis.stage1.metrics.ivDelta30m) ?? 0) > 0
                            ? 'text-emerald-300'
                            : 'text-rose-200'
                        }`}
                      >
                        {selectedAnalysis.stage1.metrics.ivDelta15m !== null &&
                        selectedAnalysis.stage1.metrics.ivDelta15m !== undefined
                          ? `${formatNumber(selectedAnalysis.stage1.metrics.ivDelta15m, 4)} (15m)`
                          : selectedAnalysis.stage1.metrics.ivDelta30m !== null &&
                              selectedAnalysis.stage1.metrics.ivDelta30m !== undefined
                            ? `${formatNumber(selectedAnalysis.stage1.metrics.ivDelta30m, 4)} (30m)`
                            : '—'}
                      </span>
                    </li>
                    <li className="flex justify-between">
                      <span>Volume / OI</span>
                      <span className="font-semibold">
                        {formatNumber(selectedAnalysis.stage1.metrics.volumeToOi, 2)}
                      </span>
                    </li>
                    <li className="flex justify-between">
                      <span>Whale Trades</span>
                      <span className="font-semibold">{selectedAnalysis.stage1.whaleTrades.length}</span>
                    </li>
                  </ul>
                  <div className="mt-4 rounded-xl border border-slate-800/60 bg-slate-950/50 p-3 text-xs text-slate-400">
                    {selectedAnalysis.stage1.notes.map(note => (
                      <p key={note} className="leading-relaxed">
                        {note}
                      </p>
                    ))}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-5 shadow-xl">
                  <h3 className="text-lg font-semibold text-slate-100">Whale Flow Evidence</h3>
                  {selectedAnalysis.stage1.whaleTrades.length > 0 ? (
                    <ul className="mt-4 space-y-3 text-sm text-slate-300">
                      {selectedAnalysis.stage1.whaleTrades.map(trade => (
                        <li key={`${trade.strike}-${trade.expiration}-${trade.contracts}`} className="rounded-xl border border-slate-800/60 bg-slate-950/60 p-3">
                          <p className="font-semibold text-slate-100">
                            {trade.direction.toUpperCase()} {trade.optionType.toUpperCase()} &middot; {trade.contracts} contracts
                          </p>
                          <p className="text-xs text-slate-400">
                            Strike ${trade.strike.toFixed(2)} &middot; Exp {trade.expiration} &middot; Premium ≈ ${formatNumber(trade.premium, 1)}
                          </p>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-3 text-sm text-slate-400">
                      No qualifying whale prints detected within thresholds. Re-run soon as new flow appears.
                    </p>
                  )}
                </div>
              </aside>
            </div>

            <section>
              <h3 className="mb-4 text-2xl font-semibold text-slate-200">Actionable Trade Plans</h3>
              {selectedAnalysis.tradeSignals.length > 0 ? (
                <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
                  {selectedAnalysis.tradeSignals.map(signal => (
                    <TradeSignalCard key={signal.id} signal={signal} />
                  ))}
                </div>
              ) : (
                <div className="rounded-2xl border border-slate-800/60 bg-slate-900/60 p-6 text-sm text-slate-400">
                  No qualified trade entries under current criteria. Monitor gamma slope and whale flow for updates or loosen
                  filters via settings if desired.
                </div>
              )}
            </section>

            <section>
              <div className="mt-10 flex items-center justify-between">
                <h3 className="text-2xl font-semibold text-slate-200">Lifecycle Management</h3>
                <span className="text-xs uppercase tracking-wide text-slate-500">
                  Active trades across universe: {data?.activeTrades?.length ?? 0}
                </span>
              </div>
              {selectedAnalysis.activeTrades.length > 0 ? (
                <div className="mt-4 grid grid-cols-1 gap-5">
                  {selectedAnalysis.activeTrades.map(trade => (
                    <TradeLifecycleCard key={trade.id} trade={trade} />
                  ))}
                </div>
              ) : (
                <div className="mt-4 rounded-2xl border border-slate-800/60 bg-slate-900/60 p-6 text-sm text-slate-400">
                  No active lifecycle items for {selectedAnalysis.symbol}. When a signal progresses into an entry, it will appear here with ongoing management guidance.
                </div>
              )}
            </section>
          </section>
        )}

        {showJson && data && (
          <section className="mt-10">
            <h3 className="mb-2 text-xl font-semibold text-slate-100">Raw JSON Output</h3>
            <pre className="max-h-[480px] overflow-x-auto overflow-y-auto rounded-2xl border border-slate-800/80 bg-slate-950/80 p-4 text-xs text-slate-200">
{JSON.stringify(data, null, 2)}
            </pre>
          </section>
        )}
      </div>
    </div>
  );
}
const statusColors: Record<string, string> = {
  watching: 'bg-sky-500/20 text-sky-200 border border-sky-500/40',
  entered: 'bg-emerald-500/20 text-emerald-200 border border-emerald-500/40',
  scaled: 'bg-blue-500/20 text-blue-200 border border-blue-500/40',
  target_hit: 'bg-purple-500/20 text-purple-200 border border-purple-500/40',
  stopped: 'bg-rose-500/20 text-rose-200 border border-rose-500/40',
  expired: 'bg-amber-500/20 text-amber-200 border border-amber-500/40',
  cancelled: 'bg-slate-600/20 text-slate-200 border border-slate-600/40',
};

const formatStatus = (status: string) => status.replace(/_/g, ' ');

const TradeLifecycleCard = ({ trade }: { trade: RegimeTradeLifecycle }) => {
  return (
    <div className="rounded-2xl border border-slate-800/80 bg-slate-900/70 p-5 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">{trade.strategy} · {trade.positionSize} size</p>
          <h4 className="text-xl font-semibold text-slate-100">{trade.symbol} · {trade.direction.toUpperCase()}</h4>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${statusColors[trade.status] || 'bg-slate-700/50 text-slate-200 border border-slate-700/80'}`}>
          {formatStatus(trade.status)}
        </span>
      </div>

      <div className="mt-4 grid gap-3 text-sm text-slate-300 sm:grid-cols-2">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">Entry / Trigger</p>
          <p className="font-semibold">
            ${trade.entryPrice.toFixed(2)} · Trigger {trade.direction === 'long' ? '≥' : '≤'} {trade.triggerLevel.toFixed(2)}
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
            {trade.enteredAt ? `Entered ${new Date(trade.enteredAt).toLocaleTimeString()}` : 'Awaiting trigger'}
            {trade.timerExpiry ? ` · Time stop ${new Date(trade.timerExpiry).toLocaleTimeString()}` : ''}
          </p>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-slate-800/60 bg-slate-950/50 p-3 text-sm text-slate-300">
        <p className="font-semibold text-slate-100">Next Action</p>
        <p className="mt-1 text-slate-400">{trade.nextAction}</p>
      </div>

      <div className="mt-4 text-xs text-slate-500">
        <p className="font-semibold text-slate-400 mb-1">Lifecycle Notes</p>
        <ul className="space-y-1">
          {trade.history.slice(-3).map(entry => (
            <li key={`${entry.timestamp}-${entry.status}`}>
              <span className="text-slate-400">[{new Date(entry.timestamp).toLocaleTimeString()}]</span> {formatStatus(entry.status)} – {entry.note}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
