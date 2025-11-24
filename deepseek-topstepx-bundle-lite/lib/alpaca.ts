import 'dotenv/config';

const APCA_KEY = process.env.ALPACA_KEY_ID || process.env.APCA_API_KEY_ID || '';
const APCA_SECRET = process.env.ALPACA_SECRET_KEY || process.env.APCA_API_SECRET_KEY || '';
const TRADING_BASE_URL = (process.env.ALPACA_BASE_URL || 'https://paper-api.alpaca.markets').replace(/\/$/, '');
const DATA_BASE_URL = (process.env.ALPACA_DATA_URL || 'https://data.alpaca.markets').replace(/\/$/, '');

if (!APCA_KEY || !APCA_SECRET) {
  console.warn('[alpaca] Missing API credentials; set ALPACA_KEY_ID / ALPACA_SECRET_KEY');
}

async function alpacaFetch<T>(url: string, init: RequestInit = {}, isDataApi = false): Promise<T> {
  const headers = {
    'APCA-API-KEY-ID': APCA_KEY,
    'APCA-API-SECRET-KEY': APCA_SECRET,
    'Content-Type': 'application/json',
    ...(init.headers || {}),
  };

  const response = await fetch(url, { ...init, headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Alpaca ${isDataApi ? 'data' : 'trading'} API error ${response.status}: ${text}`);
  }
  if (response.status === 204) {
    return {} as T;
  }
  return response.json() as Promise<T>;
}

export interface AlpacaBar {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
  vw?: number;
  n?: number;
}

export interface AlpacaQuote {
  ap: number;
  as: number;
  bp: number;
  bs: number;
  t: string;
}

export interface AlpacaOrderParams {
  symbol: string;
  qty: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  time_in_force?: 'day' | 'gtc';
  limit_price?: number;
  client_order_id?: string;
  extended_hours?: boolean;
}

export interface AlpacaOrder {
  id: string;
  client_order_id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  qty: string;
  filled_qty: string;
  canceled_at: string | null;
  submitted_at: string;
  filled_at: string | null;
  status:
    | 'new'
    | 'partially_filled'
    | 'filled'
    | 'canceled'
    | 'replaced'
    | 'pending_cancel'
    | 'pending_replace'
    | 'accepted'
    | 'pending_new'
    | 'accepted_for_bidding'
    | 'stopped'
    | 'rejected'
    | 'suspended'
    | 'calculated'
    | 'expired';
  limit_price?: number;
  filled_avg_price?: string;
}

export interface AlpacaPosition {
  symbol: string;
  qty: string;
  side: 'long' | 'short';
  avg_entry_price: string;
  market_value: string;
  cost_basis: string;
  unrealized_pl: string;
  unrealized_plpc: string;
  current_price: string;
  lastday_price: string;
  change_today: string;
}

export interface AlpacaAccount {
  id: string;
  account_number: string;
  status: string;
  currency: string;
  cash: string;
  portfolio_value: string;
  equity: string;
  buying_power: string;
  last_equity: string;
  last_maintenance_margin: string;
}

export interface AlpacaOptionContract {
  symbol: string;
  option_type: 'call' | 'put';
  strike_price: number;
  expiration_date: string;
  underlying_symbol?: string;
  ask_price?: number | null;
  bid_price?: number | null;
  last_trade_price?: number | null;
  implied_volatility?: number | null;
}

export interface AlpacaOptionOrderParams {
  symbol: string;
  qty: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  time_in_force?: 'day' | 'gtc';
  limit_price?: number;
  position_effect?: 'open' | 'close';
  client_order_id?: string;
}

export interface AlpacaOptionOrder {
  id: string;
  client_order_id: string;
  symbol: string;
  qty: string;
  filled_qty: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  time_in_force: 'day' | 'gtc';
  position_effect?: 'open' | 'close';
  status:
    | 'new'
    | 'partially_filled'
    | 'filled'
    | 'canceled'
    | 'replaced'
    | 'pending_cancel'
    | 'pending_replace'
    | 'accepted'
    | 'pending_new'
    | 'accepted_for_bidding'
    | 'stopped'
    | 'rejected'
    | 'suspended'
    | 'calculated'
    | 'expired';
  limit_price?: number;
  filled_avg_price?: string;
  submitted_at?: string;
  filled_at?: string | null;
  canceled_at?: string | null;
}

export async function fetchBars(symbol: string, timeframe: string, options: { limit?: number; start?: string; end?: string } = {}): Promise<AlpacaBar[]> {
  const url = new URL(`${DATA_BASE_URL}/v2/stocks/${symbol}/bars`);
  url.searchParams.set('timeframe', timeframe);
  url.searchParams.set('limit', String(options.limit ?? 100));
  url.searchParams.set('adjustment', 'raw');
  url.searchParams.set('feed', process.env.ALPACA_DATA_FEED || 'sip');
  if (options.start) url.searchParams.set('start', options.start);
  if (options.end) url.searchParams.set('end', options.end);

  const data = await alpacaFetch<{ bars?: AlpacaBar[] }>(url.toString(), undefined, true);
  return data.bars ?? [];
}

export async function fetchLatestQuote(symbol: string): Promise<AlpacaQuote | null> {
  const url = `${DATA_BASE_URL}/v2/stocks/${symbol}/quotes/latest`;
  const data = await alpacaFetch<{ quote?: AlpacaQuote }>(url, undefined, true);
  return data.quote ?? null;
}

export async function submitOrder(params: AlpacaOrderParams): Promise<AlpacaOrder> {
  const url = `${TRADING_BASE_URL}/v2/orders`;
  return alpacaFetch<AlpacaOrder>(url, {
    method: 'POST',
    body: JSON.stringify({
      ...params,
      time_in_force: params.time_in_force ?? 'day',
    }),
  });
}

export async function getOrder(orderId: string): Promise<AlpacaOrder> {
  const url = `${TRADING_BASE_URL}/v2/orders/${orderId}`;
  return alpacaFetch<AlpacaOrder>(url);
}

export async function cancelOrder(orderId: string): Promise<void> {
  const url = `${TRADING_BASE_URL}/v2/orders/${orderId}`;
  await alpacaFetch<void>(url, { method: 'DELETE' });
}

export async function listOpenOrders(): Promise<AlpacaOrder[]> {
  const url = `${TRADING_BASE_URL}/v2/orders?status=open`;
  return alpacaFetch<AlpacaOrder[]>(url);
}

export async function getPosition(symbol: string): Promise<AlpacaPosition | null> {
  try {
    const url = `${TRADING_BASE_URL}/v2/positions/${symbol}`;
    return await alpacaFetch<AlpacaPosition>(url);
  } catch (err: any) {
    if (typeof err.message === 'string' && err.message.includes('404')) {
      return null;
    }
    throw err;
  }
}

export async function closePosition(symbol: string, options: { qty?: string; side?: 'long' | 'short' } = {}): Promise<any> {
  const url = new URL(`${TRADING_BASE_URL}/v2/positions/${symbol}`);
  if (options.qty) url.searchParams.set('qty', options.qty);
  if (options.side) url.searchParams.set('side', options.side);
  return alpacaFetch(url.toString(), { method: 'DELETE' });
}

export async function getClock(): Promise<{ timestamp: string; is_open: boolean; next_open: string; next_close: string }> {
  const url = `${TRADING_BASE_URL}/v2/clock`;
  return alpacaFetch(url);
}

export async function fetchOptionExpirations(symbol: string): Promise<string[]> {
  const url = `${DATA_BASE_URL}/v1beta1/options/symbols/${encodeURIComponent(symbol)}/expirations`;
  const data = await alpacaFetch<{ symbol: string; expirations?: string[] }>(url, undefined, true);
  return Array.isArray(data.expirations) ? data.expirations : [];
}

export async function fetchOptionChain(symbol: string, expiration: string, options: { limit?: number } = {}): Promise<AlpacaOptionContract[]> {
  const url = new URL(`${DATA_BASE_URL}/v1beta1/options/chain/${encodeURIComponent(symbol)}`);
  url.searchParams.set('expiration', expiration);
  url.searchParams.set('limit', String(options.limit ?? 250));

  const data = await alpacaFetch<any>(url.toString(), undefined, true);
  const rawOptions: any[] =
    (Array.isArray(data?.options) ? data.options : null) ??
    (Array.isArray(data?.contracts) ? data.contracts : null) ??
    (Array.isArray(data?.data) ? data.data : []) ??
    [];

  return rawOptions
    .map((opt: any): AlpacaOptionContract | null => {
      const optionSymbol = opt?.symbol || opt?.id;
      const optionType = (opt?.option_type || opt?.type || '').toString().toLowerCase();
      const strike = Number(opt?.strike_price ?? opt?.strike);
      if (!optionSymbol || !Number.isFinite(strike) || (optionType !== 'call' && optionType !== 'put')) {
        return null;
      }
      const expirationDate = opt?.expiration_date || opt?.expiration || expiration;
      return {
        symbol: optionSymbol,
        option_type: optionType,
        strike_price: strike,
        expiration_date: expirationDate,
        underlying_symbol: opt?.underlying_symbol ?? opt?.underlying,
        ask_price: numericOrNull(opt?.ask ?? opt?.ask_price),
        bid_price: numericOrNull(opt?.bid ?? opt?.bid_price),
        last_trade_price: numericOrNull(opt?.last ?? opt?.last_price ?? opt?.last_trade_price),
        implied_volatility: numericOrNull(opt?.implied_volatility ?? opt?.implied_vol),
      };
    })
    .filter((opt): opt is AlpacaOptionContract => !!opt);
}

function numericOrNull(value: any): number | null {
  if (value === undefined || value === null || value === '') {
    return null;
  }
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

export async function getAccount(): Promise<AlpacaAccount> {
  const url = `${TRADING_BASE_URL}/v2/account`;
  return alpacaFetch<AlpacaAccount>(url);
}

export async function submitOptionOrder(params: AlpacaOptionOrderParams): Promise<AlpacaOptionOrder> {
  const url = `${TRADING_BASE_URL}/v2/orders`;
  return alpacaFetch<AlpacaOptionOrder>(url, {
    method: 'POST',
    body: JSON.stringify({
      ...params,
      time_in_force: params.time_in_force ?? 'day',
    }),
  });
}

export async function getOptionOrder(orderId: string): Promise<AlpacaOptionOrder> {
  const url = `${TRADING_BASE_URL}/v2/orders/${orderId}`;
  return alpacaFetch<AlpacaOptionOrder>(url);
}

export async function cancelOptionOrder(orderId: string): Promise<void> {
  const url = `${TRADING_BASE_URL}/v2/orders/${orderId}`;
  await alpacaFetch<void>(url, { method: 'DELETE' });
}

export function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function parseBarTime(bar: AlpacaBar): number {
  return new Date(bar.t).getTime();
}
