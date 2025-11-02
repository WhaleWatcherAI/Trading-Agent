import axios from 'axios';
import {
  StockPrice,
  OptionsTrade,
  TradierBalance,
  TradierPosition,
  TradierOrder,
  TradierOrderStatus,
} from '@/types';

const TRADIER_API_KEY = process.env.TRADIER_API_KEY || '';
const TRADIER_BASE_URL = process.env.TRADIER_BASE_URL || 'https://api.tradier.com/v1';
const TRADIER_ACCOUNT_ID = process.env.TRADIER_ACCOUNT_ID || '';

const tradierClient = axios.create({
  baseURL: TRADIER_BASE_URL,
  headers: {
    'Authorization': `Bearer ${TRADIER_API_KEY}`,
    'Accept': 'application/json',
  },
});

export interface TradierTimesaleBar {
  time: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export async function getStockPrice(symbol: string): Promise<StockPrice> {
  try {
    const response = await tradierClient.get('/markets/quotes', {
      params: { symbols: symbol },
    });

    const quote = response.data.quotes.quote;

    return {
      symbol: quote.symbol,
      price: quote.last,
      change: quote.change,
      changePercent: quote.change_percentage,
      volume: quote.volume,
      timestamp: new Date(),
    };
  } catch (error) {
    console.error(`Error fetching stock price for ${symbol}:`, error);
    throw error;
  }
}

export async function getOptionsChain(symbol: string, expiration?: string): Promise<OptionsTrade[]> {
  try {
    // Get expirations if not provided
    let exp = expiration;
    if (!exp) {
      const expResponse = await tradierClient.get('/markets/options/expirations', {
        params: { symbol },
      });
      exp = expResponse.data.expirations.date[0]; // Get nearest expiration
    }

    const response = await tradierClient.get('/markets/options/chains', {
      params: {
        symbol,
        expiration: exp,
        greeks: true,
      },
    });

    const options = response.data.options.option || [];

    return options.map((opt: any) => ({
      symbol: opt.symbol,
      underlying: opt.underlying,
      strike: opt.strike,
      expiration: opt.expiration_date,
      type: opt.option_type.toLowerCase(),
      side: 'mid', // Tradier doesn't specify, we'll analyze this
      premium: opt.last,
      volume: opt.volume,
      openInterest: opt.open_interest,
      timestamp: new Date(),
      unusual: opt.volume > opt.open_interest * 0.5, // Simple unusual activity detection
    }));
  } catch (error) {
    console.error(`Error fetching options chain for ${symbol}:`, error);
    throw error;
  }
}

export async function getMarketData(): Promise<{ putCallRatio: number; spy: number; vix: number }> {
  try {
    const [spyData, vixData] = await Promise.all([
      tradierClient.get('/markets/quotes', { params: { symbols: 'SPY' } }),
      tradierClient.get('/markets/quotes', { params: { symbols: 'VIX' } }),
    ]);

    const spyQuote = spyData.data.quotes.quote;
    const vixQuote = vixData.data.quotes.quote;

    // Simplified put/call ratio - would need more sophisticated calculation
    const putCallRatio = 1.0; // Placeholder - requires more data from options market

    const vixValue = vixQuote.last || vixQuote.close || vixQuote.prevclose;

    console.log(`📊 Tradier Market Data: SPY=${spyQuote.last}, VIX=${vixValue || 'N/A (markets closed)'}`);

    return {
      putCallRatio,
      spy: spyQuote.last,
      vix: vixValue || null,
    };
  } catch (error: any) {
    console.error('Error fetching market data from Tradier:', error.response?.data || error.message);
    throw error;
  }
}

export async function getOptionQuote(optionSymbol: string) {
  try {
    const response = await tradierClient.get('/markets/quotes', {
      params: { symbols: optionSymbol },
    });

    const quote = response.data.quotes.quote;

    return {
      symbol: quote.symbol,
      bid: quote.bid,
      ask: quote.ask,
      last: quote.last,
      volume: quote.volume,
      openInterest: quote.open_interest,
      bidSize: quote.bidsize,
      askSize: quote.asksize,
    };
  } catch (error) {
    console.error(`Error fetching option quote for ${optionSymbol}:`, error);
    throw error;
  }
}

function requireAccountId(): string {
  if (!TRADIER_ACCOUNT_ID) {
    throw new Error('TRADIER_ACCOUNT_ID not configured');
  }
  return TRADIER_ACCOUNT_ID;
}

export async function getAccountBalances(): Promise<TradierBalance | null> {
  try {
    const accountId = requireAccountId();
    const response = await tradierClient.get(`/accounts/${accountId}/balances`, {
      params: { type: 'brokerage' },
    });

    const balances = response.data?.balances;
    if (!balances) {
      return null;
    }

    return {
      totalCash: parseFloat(balances.cash || 0),
      netValue: parseFloat(balances.total_equity || 0),
      buyingPower: parseFloat(balances.buying_power || 0),
      dayTradingBuyingPower: balances.day_trading_buying_power ? parseFloat(balances.day_trading_buying_power) : undefined,
      maintenanceMargin: balances.maintenance_margin_requirement ? parseFloat(balances.maintenance_margin_requirement) : undefined,
      accruedInterest: balances.accrued_interest ? parseFloat(balances.accrued_interest) : undefined,
      pendingOrdersCount: balances.pending_orders_count ? parseInt(balances.pending_orders_count, 10) : undefined,
      timestamp: new Date().toISOString(),
    };
  } catch (error) {
    console.error('Error fetching Tradier balances', error);
    return null;
  }
}

export async function getAccountPositions(): Promise<TradierPosition[]> {
  try {
    const accountId = requireAccountId();
    const response = await tradierClient.get(`/accounts/${accountId}/positions`);
    const rawPositions = response.data?.positions?.position;
    if (!rawPositions) {
      return [];
    }

    const positionsArray = Array.isArray(rawPositions) ? rawPositions : [rawPositions];
    return positionsArray.map((position: any) => {
      const quantity = parseFloat(position.quantity || 0);
      const costBasis = parseFloat(position.cost_basis || 0);
      const last = parseFloat(position.last_price || 0);
      const marketValue = last * quantity;
      const unrealized = parseFloat(position.unrealized_pl || marketValue - costBasis);

      return {
        symbol: position.symbol,
        quantity,
        costBasis,
        lastPrice: last,
        marketValue,
        unrealizedPL: unrealized,
        unrealizedPLPercent:
          costBasis > 0 ? (unrealized / costBasis) * 100 : 0,
      };
    });
  } catch (error) {
    console.error('Error fetching Tradier positions', error);
    return [];
  }
}

export async function getOpenOrders(): Promise<TradierOrder[]> {
  try {
    const accountId = requireAccountId();
    const response = await tradierClient.get(`/accounts/${accountId}/orders`, {
      params: { status: 'open' },
    });

    const rawOrders = response.data?.orders?.order;
    if (!rawOrders) {
      return [];
    }

    const ordersArray = Array.isArray(rawOrders) ? rawOrders : [rawOrders];
    return ordersArray.map((order: any) => {
      const filled = parseFloat(order.avg_fill_price ? order.filled_quantity || 0 : order.filled_quantity || 0);
      const total = parseFloat(order.quantity || 0);
      const remaining = Math.max(total - filled, 0);

      return {
        id: String(order.id),
        symbol: order.symbol,
        side: (order.side || 'buy').toLowerCase() === 'sell' ? 'sell' : 'buy',
        type: order.type || 'market',
        status: (order.status || 'pending').toLowerCase() as TradierOrderStatus,
        quantity: total,
        filledQuantity: filled,
        remainingQuantity: remaining,
        limitPrice: order.price ? parseFloat(order.price) : undefined,
        stopPrice: order.stop ? parseFloat(order.stop) : undefined,
        submittedAt: order.created_at || new Date().toISOString(),
        updatedAt: order.updated_at,
      };
    });
  } catch (error) {
    console.error('Error fetching Tradier orders', error);
    return [];
  }
}

export async function getHistoricalTimesales(
  symbol: string,
  date: string,
  interval: number = 1,
  sessionFilter: 'all' | 'open' = 'all',
): Promise<TradierTimesaleBar[]> {
  try {
    const response = await tradierClient.get('/markets/timesales', {
      params: {
        symbol,
        interval,
        start: `${date} 09:30`,
        end: `${date} 16:00`,
        session_filter: sessionFilter,
      },
    });

    const segments = response.data?.series?.data;
    if (!segments) {
      return [];
    }

    const rows = Array.isArray(segments) ? segments : [segments];
    return rows
      .map((row: any) => {
        const open = parseFloat(row.open ?? row.price ?? row.last ?? 0);
        const close = parseFloat(row.close ?? row.price ?? row.last ?? 0);
        const high = parseFloat(row.high ?? Math.max(open, close));
        const low = parseFloat(row.low ?? Math.min(open, close));
        const volume = parseFloat(row.volume ?? 0);
        const timestamp = String(row.timestamp ?? `${date} ${row.time ?? ''}`.trim());
        const time = String(row.time ?? timestamp.split(' ')[1] ?? '');

        return {
          time,
          timestamp,
          open,
          high,
          low,
          close,
          volume,
        } as TradierTimesaleBar;
      })
      .filter(bar => Number.isFinite(bar.close) && bar.close > 0);
  } catch (error) {
    console.error(`Error fetching Tradier timesales for ${symbol} on ${date}`, error);
    throw error;
  }
}
