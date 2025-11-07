'use client';

import { useState, useRef, useEffect } from 'react';
import { createChart, ColorType, LineStyle } from 'lightweight-charts';

interface BacktestResult {
  symbol: string;
  date: string;
  smaPeriod: number;
  initialCapital: number;
  finalCapital: number;
  totalReturn: number;
  totalReturnPercent: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  profitFactor: number;
  trades: any[];
  priceData: Array<{ time: string; price: number; sma: number }>;
  tradeMarkers: Array<{
    time: string;
    price: number;
    type: 'buy' | 'sell';
    optionType: 'call' | 'put';
    strike: number;
    premium: number;
    sentiment: string;
    reasoning: string[];
  }>;
  sentimentLog: Array<{ time: string; sentiment: string; reasoning: string[] }>;
}

export default function SmaSentimentPage() {
  const [symbol, setSymbol] = useState('SPY');
  const [date, setDate] = useState('2025-11-04');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);

  const loadResults = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/sma-sentiment/load-results?symbol=${symbol}&date=${date}`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to load results');
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runBacktest = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/sma-sentiment/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, date }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Backtest failed');
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!result || !chartContainerRef.current) return;

    // Clear existing chart
    if (chartRef.current) {
      try {
        chartRef.current.remove();
      } catch (e) {
        console.error('Error removing chart:', e);
      }
      chartRef.current = null;
    }

    try {
      // Create new chart with proper typing
      const chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: 500,
        layout: {
          background: { type: ColorType.Solid, color: '#0f172a' },
          textColor: '#cbd5e1',
        },
        grid: {
          vertLines: { color: '#1e293b' },
          horzLines: { color: '#1e293b' },
        },
        crosshair: {
          mode: 1,
        },
        timeScale: {
          borderColor: '#334155',
          timeVisible: true,
          secondsVisible: false,
        },
        rightPriceScale: {
          borderColor: '#334155',
        },
      });

      chartRef.current = chart;

      // Add price line series
      const priceSeries = chart.addLineSeries({
        color: '#3b82f6',
        lineWidth: 2,
      });

      // Add SMA line series
      const smaSeries = chart.addLineSeries({
        color: '#f59e0b',
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
      });

      // Convert data for chart (convert ISO timestamps to Unix timestamps)
      const priceData = result.priceData.map(d => ({
        time: Math.floor(new Date(d.time).getTime() / 1000) as any,
        value: d.price,
      }));

      const smaData = result.priceData.map(d => ({
        time: Math.floor(new Date(d.time).getTime() / 1000) as any,
        value: d.sma,
      }));

      priceSeries.setData(priceData);
      smaSeries.setData(smaData);

      // Add trade markers showing stock price (convert to Unix timestamps)
      const buyMarkers = result.tradeMarkers
        .filter(m => m.type === 'buy')
        .map(m => {
          const trade = result.trades.find(t => t.entryTime === m.time);
          return {
            time: Math.floor(new Date(m.time).getTime() / 1000) as any,
            position: 'belowBar' as const,
            color: m.optionType === 'call' ? '#10b981' : '#ef4444',
            shape: 'arrowUp' as const,
            text: `ENTRY ${m.optionType.toUpperCase()} @ $${m.price.toFixed(2)}`,
          };
        });

      const sellMarkers = result.tradeMarkers
        .filter(m => m.type === 'sell')
        .map(m => {
          const trade = result.trades.find(t => t.exitTime === m.time);
          const pnl = trade ? trade.pnl : 0;
          const pnlSign = pnl >= 0 ? '+' : '';
          return {
            time: Math.floor(new Date(m.time).getTime() / 1000) as any,
            position: 'aboveBar' as const,
            color: pnl >= 0 ? '#10b981' : '#ef4444',
            shape: 'arrowDown' as const,
            text: `EXIT @ $${m.price.toFixed(2)} (${pnlSign}$${pnl.toFixed(0)})`,
          };
        });

      priceSeries.setMarkers([...buyMarkers, ...sellMarkers]);

      chart.timeScale().fitContent();

      // Cleanup
      return () => {
        try {
          chart.remove();
        } catch (e) {
          console.error('Error removing chart on cleanup:', e);
        }
      };
    } catch (error) {
      console.error('Error creating chart:', error);
      setError('Failed to create chart. Please try again.');
    }
  }, [result]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-sky-400 to-purple-500 bg-clip-text text-transparent">
          SMA Sentiment Strategy Backtester
        </h1>
        <p className="text-slate-400 mb-2">
          5-minute candles with 20 SMA, filtered by call/put ratio and options flow sentiment
        </p>
        <div className="mb-8 space-y-1">
          <p className="text-xs text-amber-400">
            ⚠️ Data available from Sept 9, 2025 onwards. Options flow data limited to last 3 trading days.
          </p>
          <p className="text-xs text-slate-500">
            💡 Tip: Use "Load Results" to view pre-generated backtests, or "Run Backtest" to run a new one.
          </p>
        </div>

        {/* Input Form */}
        <div className="bg-slate-900 rounded-xl p-6 mb-8 border border-slate-800">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Symbol
              </label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-500"
                placeholder="SPY"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Date
              </label>
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-500"
              />
            </div>

            <div className="flex items-end gap-2">
              <button
                onClick={loadResults}
                disabled={loading}
                className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-slate-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors"
              >
                {loading ? 'Loading...' : 'Load Results'}
              </button>
              <button
                onClick={runBacktest}
                disabled={loading}
                className="flex-1 bg-sky-600 hover:bg-sky-700 disabled:bg-slate-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors"
              >
                {loading ? 'Running...' : 'Run Backtest'}
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-900/20 border border-red-800 rounded-lg text-red-300">
              {error}
            </div>
          )}
        </div>

        {/* Results */}
        {result && (
          <>
            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
                <div className="text-sm text-slate-400 mb-1">Total Return</div>
                <div className={`text-2xl font-bold ${result.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${result.totalReturn.toFixed(2)}
                </div>
                <div className="text-sm text-slate-500">
                  {result.totalReturnPercent.toFixed(2)}%
                </div>
              </div>

              <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
                <div className="text-sm text-slate-400 mb-1">Win Rate</div>
                <div className="text-2xl font-bold text-sky-400">
                  {result.winRate.toFixed(1)}%
                </div>
                <div className="text-sm text-slate-500">
                  {result.winningTrades}W / {result.losingTrades}L
                </div>
              </div>

              <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
                <div className="text-sm text-slate-400 mb-1">Profit Factor</div>
                <div className="text-2xl font-bold text-purple-400">
                  {result.profitFactor === null || result.profitFactor === Infinity ? '∞' : result.profitFactor.toFixed(2)}
                </div>
                <div className="text-sm text-slate-500">
                  Total: {result.totalTrades}
                </div>
              </div>

              <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
                <div className="text-sm text-slate-400 mb-1">Avg Win/Loss</div>
                <div className="text-2xl font-bold text-yellow-400">
                  ${result.avgWin.toFixed(2)}
                </div>
                <div className="text-sm text-slate-500">
                  / ${result.avgLoss.toFixed(2)}
                </div>
              </div>
            </div>

            {/* Chart */}
            <div className="bg-slate-900 rounded-xl p-6 mb-8 border border-slate-800">
              <h2 className="text-xl font-bold mb-4">Price Chart with Trades</h2>

              {/* Trade Summary */}
              {result.trades.length > 0 && (
                <div className="mb-4 p-4 bg-slate-800 rounded-lg border border-slate-700">
                  {result.trades.map((trade, idx) => (
                    <div key={`trade-${idx}`} className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm mb-4 last:mb-0">
                      <div>
                        <div className="text-slate-400 mb-1">Entry</div>
                        <div className="font-semibold text-green-400">{trade.entryTime.split('T')[1]}</div>
                        <div className="text-xs text-slate-500">Option: ${trade.entryPrice.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-slate-400 mb-1">Exit</div>
                        <div className="font-semibold text-orange-400">{trade.exitTime.split('T')[1]}</div>
                        <div className="text-xs text-slate-500">Option: ${trade.exitPrice.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-slate-400 mb-1">Contracts</div>
                        <div className="font-semibold">{trade.contracts}x {trade.type.toUpperCase()}</div>
                        <div className="text-xs text-slate-500">Strike: {trade.strike}</div>
                      </div>
                      <div>
                        <div className="text-slate-400 mb-1">P&L</div>
                        <div className={`font-semibold ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ${trade.pnl.toFixed(2)}
                        </div>
                        <div className="text-xs text-slate-500">{trade.pnlPercent.toFixed(1)}%</div>
                      </div>
                      <div>
                        <div className="text-slate-400 mb-1">Sentiment</div>
                        <div className="font-semibold capitalize">{trade.sentiment}</div>
                        <div className="text-xs text-slate-500">{(trade.confidence * 100).toFixed(0)}% conf</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div ref={chartContainerRef} className="w-full"></div>
              <div className="mt-4 space-y-2">
                <div className="flex flex-wrap gap-4 text-sm text-slate-300">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-1 bg-blue-500"></div>
                    <span>Stock Price</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-1 bg-orange-500 border-dashed border"></div>
                    <span>SMA({result.smaPeriod})</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-0 h-0 border-l-4 border-l-transparent border-r-4 border-r-transparent border-b-8 border-b-green-500"></div>
                    <span>Entry (stock price)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-0 h-0 border-l-4 border-l-transparent border-r-4 border-r-transparent border-t-8 border-t-green-500"></div>
                    <span>Winning Exit (stock price + P&L)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-0 h-0 border-l-4 border-l-transparent border-r-4 border-r-transparent border-t-8 border-t-red-500"></div>
                    <span>Losing Exit (stock price + P&L)</span>
                  </div>
                </div>
                <div className="text-xs text-slate-500">
                  Note: Chart markers show stock prices. Option prices shown in trade summary above.
                </div>
              </div>
            </div>

            {/* Trades Table */}
            <div className="bg-slate-900 rounded-xl p-6 mb-8 border border-slate-800">
              <h2 className="text-xl font-bold mb-4">Trade Log</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-slate-400 border-b border-slate-800">
                    <tr>
                      <th className="text-left py-3 px-2">Entry</th>
                      <th className="text-left py-3 px-2">Exit</th>
                      <th className="text-left py-3 px-2">Type</th>
                      <th className="text-right py-3 px-2">Strike</th>
                      <th className="text-right py-3 px-2">Contracts</th>
                      <th className="text-right py-3 px-2">Entry $</th>
                      <th className="text-right py-3 px-2">Exit $</th>
                      <th className="text-right py-3 px-2">P&L</th>
                      <th className="text-right py-3 px-2">%</th>
                      <th className="text-left py-3 px-2">Sentiment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.trades.map((trade, idx) => (
                      <tr key={idx} className="border-b border-slate-800 hover:bg-slate-800/50">
                        <td className="py-3 px-2">{trade.entryTime}</td>
                        <td className="py-3 px-2">{trade.exitTime}</td>
                        <td className="py-3 px-2">
                          <span className={`px-2 py-1 rounded ${trade.type === 'call' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                            {trade.type.toUpperCase()}
                          </span>
                        </td>
                        <td className="py-3 px-2 text-right">{trade.strike.toFixed(2)}</td>
                        <td className="py-3 px-2 text-right">{trade.contracts}</td>
                        <td className="py-3 px-2 text-right">${trade.entryPrice.toFixed(2)}</td>
                        <td className="py-3 px-2 text-right">${trade.exitPrice.toFixed(2)}</td>
                        <td className={`py-3 px-2 text-right font-semibold ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ${trade.pnl.toFixed(2)}
                        </td>
                        <td className={`py-3 px-2 text-right ${trade.pnlPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {trade.pnlPercent.toFixed(1)}%
                        </td>
                        <td className="py-3 px-2">
                          <span className={`px-2 py-1 rounded text-xs ${
                            trade.sentiment === 'bullish' ? 'bg-sky-900 text-sky-300' :
                            trade.sentiment === 'bearish' ? 'bg-orange-900 text-orange-300' :
                            'bg-slate-700 text-slate-300'
                          }`}>
                            {trade.sentiment}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Sentiment Log */}
            <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
              <h2 className="text-xl font-bold mb-4">Sentiment Analysis Log</h2>
              <div className="space-y-3">
                {result.sentimentLog.map((log, idx) => (
                  <div key={idx} className="p-4 bg-slate-800 rounded-lg">
                    <div className="font-semibold text-sky-400 mb-2">
                      {log.time} - {log.sentiment}
                    </div>
                    <ul className="text-sm text-slate-400 space-y-1">
                      {log.reasoning.map((reason, ridx) => (
                        <li key={ridx}>• {reason}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
