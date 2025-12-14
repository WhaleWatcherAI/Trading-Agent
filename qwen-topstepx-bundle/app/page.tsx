'use client';

import { useState, useEffect } from 'react';
import { AnalysisResponse, TradeSignal, NewsItem, OptionsTrade, InstitutionalTrade } from '@/types';

interface DailyData {
  date: string;
  news: NewsItem[];
  optionsTrades: OptionsTrade[];
  institutionalTrades: InstitutionalTrade[];
  lastUpdated: string;
}

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [dailyData, setDailyData] = useState<DailyData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<'scalp' | 'intraday' | 'swing' | 'leap' | 'all'>('all');
  const [symbols, setSymbols] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'trades' | 'news' | 'options' | 'institutional'>('trades');

  // Load stored data on mount
  useEffect(() => {
    fetchDailyData();
  }, []);

  const fetchDailyData = async () => {
    try {
      const response = await fetch('/api/data');
      if (response.ok) {
        const data = await response.json();
        setDailyData(data);
      }
    } catch (err) {
      console.error('Failed to load daily data:', err);
    }
  };

  const runAnalysis = async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        strategy,
        limit: '5',
      });

      if (symbols.trim()) {
        params.append('symbols', symbols.trim().toUpperCase());
      }

      const response = await fetch(`/api/analyze?${params.toString()}`);

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data: any = await response.json();

      // Handle new API format
      if (Array.isArray(data)) {
        // Multi-strategy response - flatten all recommendations
        const allTrades: TradeSignal[] = [];
        data.forEach((result: any) => {
          result.recommendations.forEach((rec: any) => {
            // Convert new format to old TradeSignal format for backward compatibility
            allTrades.push({
              symbol: rec.contract.ticker,
              underlying: rec.symbol,
              contract: `${rec.symbol} ${rec.contract.expiration} ${rec.contract.strike}${rec.contract.type}`,
              strike: rec.contract.strike,
              expiration: rec.contract.expiration,
              type: rec.contract.type.toLowerCase() as 'call' | 'put',
              action: 'buy',
              strategy: rec.strategy,
              currentPrice: rec.contract.currentPrice,
              rating: rec.bullishScore > rec.bearishScore ? rec.bullishScore : -rec.bearishScore,
              confidence: rec.confidence,
              reasoning: rec.reasoning.primaryCatalyst,
              factors: {
                newsImpact: rec.factorBreakdown.news.bullishScore / 10,
                institutionalActivity: rec.factorBreakdown.institutional.bullishScore / 10,
                optionsFlow: rec.factorBreakdown.flow.bullishScore / 10,
                marketTide: rec.factorBreakdown.structure.bullishScore / 10,
                technicals: rec.factorBreakdown.technical.bullishScore / 10,
              },
              timestamp: new Date(),
            });
          });
        });

        setAnalysis({
          trades: allTrades,
          marketOverview: data[0]?.marketContext || { putCallRatio: 1, vix: 20, spy: 0, marketTide: 'neutral', timestamp: new Date() },
          timestamp: new Date(),
        });
      } else if (data.recommendations) {
        // Single strategy response
        const trades = data.recommendations.map((rec: any) => ({
          symbol: rec.contract.ticker,
          underlying: rec.symbol,
          contract: `${rec.symbol} ${rec.contract.expiration} ${rec.contract.strike}${rec.contract.type}`,
          strike: rec.contract.strike,
          expiration: rec.contract.expiration,
          type: rec.contract.type.toLowerCase() as 'call' | 'put',
          action: 'buy',
          strategy: rec.strategy,
          currentPrice: rec.contract.currentPrice,
          rating: rec.bullishScore > rec.bearishScore ? rec.bullishScore : -rec.bearishScore,
          confidence: rec.confidence,
          reasoning: rec.reasoning.primaryCatalyst,
          factors: {
            newsImpact: rec.factorBreakdown.news.bullishScore / 10,
            institutionalActivity: rec.factorBreakdown.institutional.bullishScore / 10,
            optionsFlow: rec.factorBreakdown.flow.bullishScore / 10,
            marketTide: rec.factorBreakdown.structure.bullishScore / 10,
            technicals: rec.factorBreakdown.technical.bullishScore / 10,
          },
          timestamp: new Date(),
        }));

        setAnalysis({
          trades,
          marketOverview: { putCallRatio: 1, vix: 20, spy: 0, marketTide: 'neutral', timestamp: new Date() },
          timestamp: new Date(),
        });
      } else {
        // Old format (fallback)
        setAnalysis(data);
      }

      // Refresh daily data to show newly added items
      await fetchDailyData();
    } catch (err: any) {
      setError(err.message || 'Failed to analyze market');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-2">AI Trading Agent</h1>
          <p className="text-gray-400">Powered by OpenAI, Tradier & Unusual Whales</p>
          {dailyData && (
            <p className="text-sm text-gray-500 mt-2">
              Last Updated: {new Date(dailyData.lastUpdated).toLocaleTimeString()}
            </p>
          )}
        </header>

        <div className="bg-gray-800 rounded-lg p-6 mb-8 shadow-xl">
          <h2 className="text-2xl font-semibold mb-4">Analysis Settings</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium mb-2">Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as any)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2"
                disabled={loading}
              >
                <option value="all">All Strategies (Recommended)</option>
                <option value="scalp">Scalp (15min-4hrs)</option>
                <option value="intraday">Intraday (2hrs-EOD)</option>
                <option value="swing">Swing (2-10 days)</option>
                <option value="leap">LEAP (30-365+ days)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Symbols (Optional - comma separated)
              </label>
              <input
                type="text"
                value={symbols}
                onChange={(e) => setSymbols(e.target.value)}
                placeholder="e.g., AAPL, TSLA, SPY"
                className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2"
                disabled={loading}
              />
            </div>
          </div>

          <button
            onClick={runAnalysis}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white font-semibold py-3 px-6 rounded transition-colors"
          >
            {loading ? 'Analyzing Market...' : 'Run Analysis'}
          </button>
        </div>

        {error && (
          <div className="bg-red-900/50 border border-red-600 rounded-lg p-4 mb-8">
            <p className="font-semibold">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        )}

        {/* Data Stats */}
        {dailyData && (
          <div className="grid grid-cols-3 gap-4 mb-8">
            <div className="bg-gray-800 rounded-lg p-4 text-center">
              <p className="text-gray-400 text-sm">News Items</p>
              <p className="text-3xl font-bold text-blue-400">{dailyData.news.length}</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 text-center">
              <p className="text-gray-400 text-sm">Options Trades</p>
              <p className="text-3xl font-bold text-green-400">{dailyData.optionsTrades.length}</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 text-center">
              <p className="text-gray-400 text-sm">Institutional Trades</p>
              <p className="text-3xl font-bold text-purple-400">{dailyData.institutionalTrades.length}</p>
            </div>
          </div>
        )}

        {analysis && (
          <div className="bg-gray-800 rounded-lg p-6 mb-8 shadow-xl">
            <h2 className="text-2xl font-semibold mb-4">Market Overview</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-gray-400 text-sm">SPY Price</p>
                <p className="text-2xl font-bold">
                  ${analysis.marketOverview.spy?.toFixed(2) || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">VIX</p>
                <p className="text-2xl font-bold">
                  {analysis.marketOverview.vix?.toFixed(2) || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Put/Call Ratio</p>
                <p className="text-2xl font-bold">
                  {analysis.marketOverview.putCallRatio?.toFixed(2) || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Market Tide</p>
                <p className={`text-2xl font-bold ${
                  analysis.marketOverview.marketTide === 'bullish' ? 'text-green-500' :
                  analysis.marketOverview.marketTide === 'bearish' ? 'text-red-500' :
                  'text-gray-400'
                }`}>
                  {analysis.marketOverview.marketTide?.toUpperCase() || 'NEUTRAL'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="bg-gray-800 rounded-t-lg overflow-hidden mb-0">
          <div className="flex border-b border-gray-700">
            <button
              onClick={() => setActiveTab('trades')}
              className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                activeTab === 'trades'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Top Trades {analysis && `(${analysis.trades.length})`}
            </button>
            <button
              onClick={() => setActiveTab('news')}
              className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                activeTab === 'news'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              News Feed {dailyData && `(${dailyData.news.length})`}
            </button>
            <button
              onClick={() => setActiveTab('options')}
              className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                activeTab === 'options'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Options Flow {dailyData && `(${dailyData.optionsTrades.length})`}
            </button>
            <button
              onClick={() => setActiveTab('institutional')}
              className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                activeTab === 'institutional'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Institutional {dailyData && `(${dailyData.institutionalTrades.length})`}
            </button>
          </div>
        </div>

        {/* Tab Content */}
        <div className="bg-gray-800 rounded-b-lg p-6 shadow-xl">
          {activeTab === 'trades' && (
            <div>
              <h2 className="text-2xl font-semibold mb-6">Top 5 Trade Recommendations</h2>
              {!analysis ? (
                <p className="text-gray-400 text-center py-8">
                  Run analysis to see trade recommendations
                </p>
              ) : analysis.trades.length === 0 ? (
                <p className="text-gray-400 text-center py-8">
                  No strong trade signals found. Try different symbols or check back later.
                </p>
              ) : (
                <div className="space-y-6">
                  {analysis.trades.map((trade, index) => (
                    <TradeCard key={index} trade={trade} rank={index + 1} />
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'news' && dailyData && (
            <NewsFeed news={dailyData.news} />
          )}

          {activeTab === 'options' && dailyData && (
            <OptionsFlow trades={dailyData.optionsTrades} />
          )}

          {activeTab === 'institutional' && dailyData && (
            <InstitutionalFeed trades={dailyData.institutionalTrades} />
          )}
        </div>
      </div>
    </div>
  );
}

function TradeCard({ trade, rank }: { trade: TradeSignal; rank: number }) {
  const getRatingColor = (rating: number) => {
    if (rating >= 7) return 'bg-green-600';
    if (rating <= -7) return 'bg-red-600';
    if (rating >= 4) return 'bg-green-500';
    if (rating <= -4) return 'bg-red-500';
    return 'bg-gray-500';
  };

  const getRatingLabel = (rating: number) => {
    const absRating = Math.abs(rating);
    if (rating < 0) {
      return `BEARISH ${absRating}`;
    }
    return `BULLISH ${absRating}`;
  };

  return (
    <div className="bg-gray-700 rounded-lg p-6 hover:bg-gray-650 transition-colors">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-4">
          <div className="text-4xl font-bold text-gray-500">#{rank}</div>
          <div>
            <h3 className="text-2xl font-bold">{trade.underlying}</h3>
            <p className="text-gray-400 text-sm">{trade.contract}</p>
            <p className="text-xs text-gray-500">{new Date(trade.timestamp).toLocaleString()}</p>
          </div>
        </div>

        <div className="text-right">
          <div className={`${getRatingColor(trade.rating)} text-white px-4 py-2 rounded-lg font-bold text-lg mb-2`}>
            {getRatingLabel(trade.rating)}
          </div>
          <p className="text-sm text-gray-400">Confidence: {trade.confidence}%</p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
        <div>
          <p className="text-gray-400 text-xs">Current Price</p>
          <p className="font-semibold">${trade.currentPrice.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-gray-400 text-xs">Strike</p>
          <p className="font-semibold">${trade.strike.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-gray-400 text-xs">Expiration</p>
          <p className="font-semibold">{trade.expiration}</p>
        </div>
        <div>
          <p className="text-gray-400 text-xs">Type</p>
          <p className="font-semibold uppercase">{trade.type}</p>
        </div>
        <div>
          <p className="text-gray-400 text-xs">Strategy</p>
          <p className="font-semibold uppercase">{trade.strategy}</p>
        </div>
        <div>
          <p className="text-gray-400 text-xs">Action</p>
          <p className="font-semibold uppercase text-blue-400">{trade.action}</p>
        </div>
      </div>

      <div className="mb-4">
        <p className="text-sm text-gray-300 leading-relaxed">{trade.reasoning}</p>
      </div>

      <div className="border-t border-gray-600 pt-4">
        <p className="text-xs text-gray-400 mb-2">Factor Breakdown:</p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
          <FactorBadge label="News" value={trade.factors.newsImpact} />
          <FactorBadge label="Institutional" value={trade.factors.institutionalActivity} />
          <FactorBadge label="Options Flow" value={trade.factors.optionsFlow} />
          <FactorBadge label="Market Tide" value={trade.factors.marketTide} />
          <FactorBadge label="Technicals" value={trade.factors.technicals} />
        </div>
      </div>
    </div>
  );
}

function FactorBadge({ label, value }: { label: string; value: number }) {
  const getColor = (val: number) => {
    if (val > 0.3) return 'bg-green-600';
    if (val < -0.3) return 'bg-red-600';
    if (val > 0) return 'bg-green-700';
    if (val < 0) return 'bg-red-700';
    return 'bg-gray-600';
  };

  return (
    <div className={`${getColor(value)} px-2 py-1 rounded text-center`}>
      <p className="font-medium">{label}</p>
      <p className="text-xs">{(value * 100).toFixed(0)}%</p>
    </div>
  );
}

function NewsFeed({ news }: { news: NewsItem[] }) {
  if (news.length === 0) {
    return (
      <p className="text-gray-400 text-center py-8">
        No news data yet. Run analysis to fetch news.
      </p>
    );
  }

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-6">Market News Feed</h2>
      <div className="space-y-4 max-h-[600px] overflow-y-auto">
        {news.map((item, index) => (
          <div key={index} className="bg-gray-700 rounded-lg p-4 hover:bg-gray-650 transition-colors">
            <div className="flex items-start justify-between mb-2">
              <h3 className="text-lg font-semibold flex-1">{item.title}</h3>
              <span className={`px-3 py-1 rounded text-xs font-semibold ml-4 ${
                item.sentiment === 'bullish' ? 'bg-green-600' :
                item.sentiment === 'bearish' ? 'bg-red-600' :
                'bg-gray-600'
              }`}>
                {item.sentiment?.toUpperCase() || 'NEUTRAL'}
              </span>
            </div>
            <p className="text-sm text-gray-400 mb-2">{item.summary}</p>
            <div className="flex items-center justify-between text-xs text-gray-500">
              <div className="flex items-center gap-4">
                <span>{item.source}</span>
                <span>{new Date(item.timestamp).toLocaleString()}</span>
              </div>
              <div className="flex gap-2">
                {item.symbols.slice(0, 5).map(sym => (
                  <span key={sym} className="bg-gray-600 px-2 py-1 rounded">{sym}</span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function OptionsFlow({ trades }: { trades: OptionsTrade[] }) {
  if (trades.length === 0) {
    return (
      <p className="text-gray-400 text-center py-8">
        No options flow data yet. Run analysis to fetch options data.
      </p>
    );
  }

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-6">Large Options Trades</h2>
      <div className="space-y-4 max-h-[600px] overflow-y-auto">
        {trades.map((trade, index) => (
          <div key={index} className="bg-gray-700 rounded-lg p-4 hover:bg-gray-650 transition-colors">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="text-xl font-bold">{trade.underlying}</h3>
                <p className="text-sm text-gray-400">{trade.symbol}</p>
              </div>
              <div className="text-right">
                <span className={`px-3 py-1 rounded font-semibold ${
                  trade.type === 'call' ? 'bg-green-600' : 'bg-red-600'
                }`}>
                  {trade.type.toUpperCase()}
                </span>
                {trade.unusual && (
                  <span className="block mt-2 px-3 py-1 rounded bg-yellow-600 text-xs font-semibold">
                    UNUSUAL
                  </span>
                )}
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-gray-400 text-xs">Strike</p>
                <p className="font-semibold">${trade.strike.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">Premium</p>
                <p className="font-semibold">${trade.premium.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">Volume</p>
                <p className="font-semibold">{trade.volume.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">Side</p>
                <p className="font-semibold uppercase">{trade.side}</p>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500">
              <span>Exp: {trade.expiration}</span>
              <span className="mx-2">•</span>
              <span>{new Date(trade.timestamp).toLocaleString()}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function InstitutionalFeed({ trades }: { trades: InstitutionalTrade[] }) {
  if (trades.length === 0) {
    return (
      <p className="text-gray-400 text-center py-8">
        No institutional trade data yet. Run analysis to fetch data.
      </p>
    );
  }

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-6">Institutional Insider Trading Activity</h2>
      <div className="space-y-4 max-h-[600px] overflow-y-auto">
        {trades.map((trade, index) => (
          <div key={index} className="bg-gray-700 rounded-lg p-4 hover:bg-gray-650 transition-colors">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="text-xl font-bold">{trade.symbol}</h3>
                <p className="text-sm text-gray-400">{trade.institution}</p>
              </div>
              <span className={`px-4 py-2 rounded font-bold text-lg ${
                trade.side === 'buy' ? 'bg-green-600' : 'bg-red-600'
              }`}>
                {trade.side.toUpperCase()}
              </span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-gray-400 text-xs">Shares</p>
                <p className="font-semibold">{trade.shares.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">Price</p>
                <p className="font-semibold">${trade.price.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">Total Value</p>
                <p className="font-semibold text-yellow-400">
                  ${(trade.value / 1_000_000).toFixed(2)}M
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">Time</p>
                <p className="font-semibold text-xs">{new Date(trade.timestamp).toLocaleTimeString()}</p>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500">
              {new Date(trade.timestamp).toLocaleDateString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
