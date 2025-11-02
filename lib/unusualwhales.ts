import axios from 'axios';
import {
  OptionsTrade, InstitutionalTrade, NewsItem, BullBearSignal,
  GreekFlow, SectorFlow, SectorTide, SpotGEX, VolatilityStats
} from '@/types';

const UW_API_KEY = process.env.UNUSUAL_WHALES_API_KEY || '';
const UW_BASE_URL = 'https://api.unusualwhales.com/api';

const uwClient = axios.create({
  baseURL: UW_BASE_URL,
  headers: {
    'Authorization': `Bearer ${UW_API_KEY}`,
    'Accept': 'application/json',
  },
});

// Helper to add delay between requests
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const UW_MAX_PER_MINUTE = 115; // slight buffer below 120/min
const UW_MAX_PER_DAY = 14000; // buffer below 15k/day

let minuteWindowStart = Date.now();
let minuteCount = 0;
let dayWindowStart = Date.now();
let dayCount = 0;

async function ensureRateLimit() {
  const now = Date.now();

  if (now - minuteWindowStart >= 60_000) {
    minuteWindowStart = now;
    minuteCount = 0;
  }

  if (now - dayWindowStart >= 86_400_000) {
    dayWindowStart = now;
    dayCount = 0;
  }

  if (dayCount >= UW_MAX_PER_DAY) {
    throw new Error('Unusual Whales daily quota exhausted (15k/day) – skipping request');
  }

  if (minuteCount >= UW_MAX_PER_MINUTE) {
    const wait = 60_000 - (now - minuteWindowStart) + 50;
    await delay(wait);
    return ensureRateLimit();
  }

  minuteCount += 1;
  dayCount += 1;
}

async function fetchWithRateLimit<T>(
  fetchFn: () => Promise<T>,
  delayMs: number = 600
): Promise<T> {
  await ensureRateLimit();
  await delay(delayMs);
  return fetchFn();
}

// Greek Flow - /group-flow/{flow_group}/greek-flow
export async function getGreekFlow(date?: string): Promise<GreekFlow[]> {
  try {
    const flowGroups = ['technology', 'healthcare', 'energy', 'financial services'];
    const flows: GreekFlow[] = [];

    // Get next Friday for expiry
    const nextFriday = new Date();
    nextFriday.setDate(nextFriday.getDate() + ((5 - nextFriday.getDay() + 7) % 7 || 7));
    const expiry = nextFriday.toISOString().split('T')[0];

    // Fetch sequentially with delays to respect rate limits
    for (const group of flowGroups) {
      try {
        // Don't pass date parameter - endpoint returns latest data
        const result = await uwClient.get(`/group-flow/${group}/greek-flow/${expiry}`);

        const data = result.data?.data || [];
        data.forEach((item: any) => {
          flows.push({
            flow_group: group,
            expiry: item.expiry,
            timestamp: new Date(item.timestamp),
            net_call_premium: parseFloat(item.net_call_premium || 0),
            net_put_premium: parseFloat(item.net_put_premium || 0),
            net_call_volume: parseInt(item.net_call_volume || 0),
            net_put_volume: parseInt(item.net_put_volume || 0),
            dir_delta_flow: parseFloat(item.dir_delta_flow || 0),
            dir_vega_flow: parseFloat(item.dir_vega_flow || 0),
            total_delta_flow: parseFloat(item.total_delta_flow || 0),
            total_vega_flow: parseFloat(item.total_vega_flow || 0),
            volume: parseInt(item.volume || 0),
            transactions: parseInt(item.transactions || 0),
          });
        });

        // Add delay between requests
        if (group !== flowGroups[flowGroups.length - 1]) {
          await delay(600);
        }
      } catch (err: any) {
        console.warn(`Failed to fetch greek flow for ${group}:`, err.response?.status);
      }
    }

    return flows;
  } catch (error) {
    console.error('Error fetching greek flow:', error);
    return [];
  }
}

// Sector Flow - /insider/{sector}/sector-flow
export async function getSectorFlow(date?: string): Promise<SectorFlow[]> {
  try {
    const sectors = ['Technology', 'Healthcare', 'Financial Services', 'Energy', 'Consumer Cyclical', 'Industrials'];
    const flows: SectorFlow[] = [];

    // Fetch sequentially with delays to respect rate limits
    for (const sector of sectors) {
      try {
        const result = await uwClient.get(`/insider/${encodeURIComponent(sector)}/sector-flow`);
        const data = result.data?.data || [];

        data.forEach((item: any) => {
          // The API returns data with sector field already included
          flows.push({
            sector: item.sector || 'Unknown',
            date: item.date,
            buy_sell: item.buy_sell === 'buy' ? 'buy' : 'sell',
            volume: parseInt(item.volume || 0),
            avg_price: parseFloat(item.premium || 0) / parseInt(item.volume || 1), // Calculate avg price from premium/volume
            premium: parseFloat(item.premium || 0),
            transactions: parseInt(item.transactions || 0),
            uniq_insiders: parseInt(item.uniq_insiders || 0),
          });
        });

        // Add delay between requests
        if (sector !== sectors[sectors.length - 1]) {
          await delay(600);
        }
      } catch (err: any) {
        console.warn(`Failed to fetch sector flow for ${sector}:`, err.response?.status);
      }
    }

    return flows;
  } catch (error) {
    console.error('Error fetching sector flow:', error);
    return [];
  }
}

// Sector Tide - /market/{sector}/sector-tide
export async function getSectorTide(date?: string): Promise<SectorTide[]> {
  try {
    const targetDate = date || new Date().toISOString().split('T')[0];
    const sectors = ['Technology', 'Healthcare', 'Financial Services', 'Energy', 'Consumer Cyclical'];
    const tides: SectorTide[] = [];

    // Fetch sequentially with delays to respect rate limits
    for (const sector of sectors) {
      try {
        // Don't pass date parameter - endpoint returns latest data
        const result = await uwClient.get(`/market/${encodeURIComponent(sector)}/sector-tide`);

        const data = result.data?.data || [];
        data.forEach((item: any) => {
          tides.push({
            sector,
            timestamp: new Date(item.timestamp),
            net_call_premium: parseFloat(item.net_call_premium || 0),
            net_put_premium: parseFloat(item.net_put_premium || 0),
            net_volume: parseInt(item.net_volume || 0),
          });
        });

        // Add delay between requests
        if (sector !== sectors[sectors.length - 1]) {
          await delay(600);
        }
      } catch (err: any) {
        console.warn(`Failed to fetch sector tide for ${sector}:`, err.response?.status);
      }
    }

    return tides;
  } catch (error) {
    console.error('Error fetching sector tide:', error);
    return [];
  }
}

// Spot GEX Exposures - /stock/{ticker}/spot-exposures (1-minute data via API)
export async function getSpotGEX(date?: string): Promise<SpotGEX[]> {
  try {
    // Reduced to 4 key tickers to avoid rate limits
    const tickers = ['SPY', 'QQQ', 'AAPL', 'NVDA'];
    const exposures: SpotGEX[] = [];

    // Fetch sequentially with delays to respect rate limits
    for (const ticker of tickers) {
      try {
        // Don't pass date parameter - endpoint doesn't support it, returns latest data
        const response = await uwClient.get(`/stock/${ticker}/spot-exposures`);

        const data = response.data?.data || [];
        if (Array.isArray(data)) {
          data.forEach((item: any) => {
            exposures.push({
              ticker: item.ticker || ticker,
              time: new Date(item.time),
              price: parseFloat(item.price || 0),
              gamma_per_one_percent_move_oi: parseFloat(item.gamma_per_one_percent_move_oi || 0),
              gamma_per_one_percent_move_vol: parseFloat(item.gamma_per_one_percent_move_vol || 0),
              vanna_per_one_percent_move_oi: parseFloat(item.vanna_per_one_percent_move_oi || 0),
              charm_per_one_percent_move_oi: parseFloat(item.charm_per_one_percent_move_oi || 0),
            });
          });
        }

        // Add delay between requests
        if (ticker !== tickers[tickers.length - 1]) {
          await delay(600);
        }
      } catch (err: any) {
        console.warn(`Failed to fetch spot GEX for ${ticker}:`, err.response?.status);
      }
    }

    return exposures;
  } catch (error) {
    console.error('Error fetching spot GEX:', error);
    return [];
  }
}

export interface WhaleFlowAlert {
  ticker: string;
  underlying: string;
  optionType: 'call' | 'put';
  direction: 'bullish' | 'bearish' | 'neutral';
  contracts: number;
  premium: number;
  price: number;
  strike: number;
  expiration: string;
  timestamp: string;
  isSweep: boolean;
}

export async function getWhaleFlowAlerts(options?: {
  symbols?: string[];
  minPremium?: number;
  minContracts?: number;
  lookbackMinutes?: number;
  limit?: number;
}): Promise<WhaleFlowAlert[]> {
  try {
    const {
      symbols = [],
      minPremium = 250_000,
      minContracts = 50,
      lookbackMinutes = 30,
      limit = 200,
    } = options || {};

    const response = await fetchWithRateLimit(() =>
      uwClient.get('/option-trades/flow-alerts', {
        params: {
          limit: Math.min(limit, 500),
        },
      })
    );

    const records = Array.isArray(response.data?.data) ? response.data.data : [];
    const upperSymbols = symbols.map(symbol => symbol.toUpperCase());
    const cutoff = Date.now() - lookbackMinutes * 60 * 1000;

    const alerts: WhaleFlowAlert[] = [];

    records.forEach((item: any) => {
      const ticker =
        (item.ticker || item.symbol || item.underlying_symbol || item.underlying || '').toUpperCase();
      if (!ticker) return;
      if (upperSymbols.length > 0 && !upperSymbols.includes(ticker)) return;

      const contracts = parseInt(
        item.contracts || item.size || item.quantity || item.volume || item.total_volume || 0,
      );
      const premium = parseFloat(
        item.premium ||
          item.cost ||
          item.total_premium ||
          item.notional ||
          item.premium_total ||
          0,
      );

      if (!Number.isFinite(contracts) || contracts <= 0) return;
      if (!Number.isFinite(premium) || premium <= 0) return;
      if (contracts < minContracts && premium < minPremium) return;

      const rawType = (item.option_type || item.type || item.contract_type || '').toLowerCase();
      const optionType = rawType.includes('call') ? 'call' : 'put';

      const rawDirection = (item.direction || item.sentiment || item.side || '').toLowerCase();
      let direction: WhaleFlowAlert['direction'] = 'neutral';
      if (rawDirection.includes('bull')) direction = 'bullish';
      else if (rawDirection.includes('bear')) direction = 'bearish';

      const price = parseFloat(
        item.price ||
          item.fill_price ||
          item.average_price ||
          item.premium_per_contract ||
          premium / Math.max(contracts, 1),
      );

      const strike = parseFloat(item.strike || item.strike_price || item.strike_price_numeric || 0);
      const expiration =
        item.expiration ||
        item.expiration_date ||
        item.expiry ||
        item.expiration_str ||
        '';

      const timestampStr =
        item.timestamp ||
        item.trade_time ||
        item.time ||
        item.executed_at ||
        item.reported_at ||
        null;
      const timestamp = timestampStr ? new Date(timestampStr).toISOString() : new Date().toISOString();
      if (new Date(timestamp).getTime() < cutoff) return;

      alerts.push({
        ticker,
        underlying: ticker,
        optionType,
        direction,
        contracts,
        premium,
        price,
        strike: Number.isFinite(strike) ? strike : 0,
        expiration,
        timestamp,
        isSweep: Boolean(item.is_sweep || item.sweep || item.order_type === 'sweep'),
      });
    });

    return alerts;
  } catch (error) {
    console.error('Error fetching whale flow alerts:', error);
    return [];
  }
}

// Volatility Statistics - /stock/{ticker}/volatility/stats
export async function getVolatilityStats(date?: string, tickers?: string[]): Promise<VolatilityStats[]> {
  try {
    const defaultTickers = ['SPY', 'QQQ', 'AAPL', 'NVDA'];
    const uniqueTickers = Array.from(new Set((tickers && tickers.length ? tickers : defaultTickers).map(t => t.toUpperCase())));
    const stats: VolatilityStats[] = [];

    // Fetch sequentially with delays
    for (const ticker of uniqueTickers) {
      try {
        // Don't pass date parameter - endpoint returns latest data
        const response = await uwClient.get(`/stock/${ticker}/volatility/stats`);

        const data = response.data?.data;
        if (data) {
          stats.push({
            ticker: data.ticker || ticker,
            date: data.date,
            iv: parseFloat(data.iv || 0),
            iv_high: parseFloat(data.iv_high || 0),
            iv_low: parseFloat(data.iv_low || 0),
            iv_rank: parseFloat(data.iv_rank || 0),
            rv: parseFloat(data.rv || 0),
            rv_high: parseFloat(data.rv_high || 0),
            rv_low: parseFloat(data.rv_low || 0),
          });
        }

        // Add delay between requests
        if (ticker !== uniqueTickers[uniqueTickers.length - 1]) {
          await delay(500);
        }
      } catch (err: any) {
        console.warn(`Failed to fetch volatility stats for ${ticker}:`, err.response?.status);
      }
    }

    return stats;
  } catch (error) {
    console.error('Error fetching volatility stats:', error);
    return [];
  }
}

// Institutional Activity - /institution/{name}/activity
export async function getInstitutionalActivity(date?: string): Promise<InstitutionalTrade[]> {
  try {
    const targetDate = date || new Date().toISOString().split('T')[0];
    const institutions = [
      'VANGUARD GROUP INC',
      'BLACKROCK INC',
      'STATE STREET CORP',
      'FIDELITY',
      'JPMORGAN CHASE & CO',
      'GOLDMAN SACHS GROUP INC',
    ];
    const trades: InstitutionalTrade[] = [];

    // Fetch sequentially with delays to respect rate limits
    for (const institution of institutions) {
      try {
        // Don't pass date parameter - endpoint returns latest data
        const result = await uwClient.get(`/institution/${encodeURIComponent(institution)}/activity`, {
          params: {
            limit: 100
          }
        });

        const data = result.data?.data || [];
        data.forEach((item: any) => {
          if (item.ticker && item.units_change !== 0) {
            const unitsChange = parseInt(item.units_change || 0);
            const avgPrice = parseFloat(item.avg_price || 0);

            trades.push({
              symbol: item.ticker,
              shares: Math.abs(unitsChange),
              price: avgPrice,
              value: Math.abs(unitsChange * avgPrice),
              side: unitsChange > 0 ? 'buy' : 'sell',
              institution,
              timestamp: new Date(item.filing_date || item.report_date || Date.now()),
            });
          }
        });

        // Add delay between requests
        if (institution !== institutions[institutions.length - 1]) {
          await delay(600);
        }
      } catch (err: any) {
        console.warn(`Failed to fetch institutional activity for ${institution}:`, err.response?.status);
      }
    }

    return trades;
  } catch (error) {
    console.error('Error fetching institutional activity:', error);
    return [];
  }
}

// News Headlines - /news/headlines
export async function getNewsHeadlines(date?: string): Promise<NewsItem[]> {
  try {
    const response = await uwClient.get('/news/headlines');
    const news = response.data.data || [];

    return news.filter((item: any) => item.headline).map((item: any) => ({
      title: item.headline || 'Untitled',
      summary: item.description || item.summary || item.text || '',
      url: item.url || '#',
      source: item.source || 'Unknown',
      symbols: Array.isArray(item.tickers) ? item.tickers : (item.symbols || []),
      sentiment: (item.sentiment?.toLowerCase() || 'neutral') as 'bullish' | 'bearish' | 'neutral',
      importance: item.is_major ? 0.8 : 0.5,
      timestamp: new Date(item.created_at || item.published_at || item.date || Date.now()),
    }));
  } catch (error) {
    console.error('Error fetching news headlines:', error);
    return [];
  }
}

// Legacy function names for backward compatibility
export async function getUnusualOptionsActivity(date?: string): Promise<OptionsTrade[]> {
  // NOTE: This converts SectorTide to OptionsTrade format
  // Real individual options flow requires paid UW subscription
  return getSectorBasedFlow(date);
}

/**
 * Get sector-based flow (aggregated, not individual trades)
 * This is NOT real unusual options flow - it's sector-level aggregated data
 */
export async function getSectorBasedFlow(date?: string): Promise<OptionsTrade[]> {
  const tides = await getSectorTide(date);

  return tides.map((tide) => {
    const type = Math.abs(tide.net_call_premium) > Math.abs(tide.net_put_premium) ? 'call' : 'put';
    const side: 'bid' | 'ask' | 'mid' = tide.net_volume > 0 ? 'ask' : (tide.net_volume < 0 ? 'bid' : 'mid');

    return {
      symbol: `${tide.sector}-SECTOR`,
      underlying: tide.sector,
      strike: 0,
      expiration: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      type,
      side,
      premium: Math.abs(type === 'call' ? tide.net_call_premium : tide.net_put_premium),
      volume: Math.abs(tide.net_volume),
      openInterest: 0,
      timestamp: tide.timestamp,
      unusual: Math.abs(tide.net_volume) > 50000,
    };
  });
}

export async function getInstitutionalTrades(date?: string): Promise<InstitutionalTrade[]> {
  return getInstitutionalActivity(date);
}

export async function getMarketNews(date?: string): Promise<NewsItem[]> {
  return getNewsHeadlines(date);
}

export function analyzeOptionFlowSentiment(trade: OptionsTrade): {
  signal: BullBearSignal;
  strength: number;
} {
  let signal: BullBearSignal = 'neutral';
  let strength = 0.5;

  if (trade.side === 'mid') {
    return { signal, strength };
  }

  if (trade.type === 'put' && trade.side === 'bid') {
    signal = 'bull';
    strength = 0.7;
  } else if (trade.type === 'call' && trade.side === 'ask') {
    signal = 'bull';
    strength = 0.8;
  } else if (trade.type === 'put' && trade.side === 'ask') {
    signal = 'bear';
    strength = 0.8;
  } else if (trade.type === 'call' && trade.side === 'bid') {
    signal = 'bear';
    strength = 0.7;
  }

  if (trade.unusual) {
    strength = Math.min(1.0, strength + 0.2);
  }

  return { signal, strength };
}

export async function getPutCallRatio(symbol?: string): Promise<number> {
  try {
    const response = await uwClient.get('/market/market-tide');
    const data = response.data.data;

    if (Array.isArray(data) && data.length > 0) {
      const latest = data[data.length - 1];
      const netCallPremium = Math.abs(parseFloat(latest.net_call_premium || 0));
      const netPutPremium = Math.abs(parseFloat(latest.net_put_premium || 0));
      if (netCallPremium > 0) {
        return netPutPremium / netCallPremium;
      }
    }
    return 1.0;
  } catch (error) {
    console.error('Error fetching put/call ratio:', error);
    return 1.0;
  }
}
