import crypto from 'crypto';
import { randomUUID } from 'crypto';
import { setTimeout as sleep } from 'timers/promises';

const API_KEY = process.env.COINBASE_API_KEY ?? '';
const API_SECRET = process.env.COINBASE_API_SECRET ?? '';
const API_BASE = (process.env.COINBASE_API_BASE ?? 'https://api.coinbase.com').replace(/\/$/, '');
const API_PREFIX = '/api/v3/brokerage';

if (!API_KEY || !API_SECRET) {
  console.warn('[coinbaseAdvancedTrade] Missing API credentials (COINBASE_API_KEY / COINBASE_API_SECRET)');
}

interface PlaceOrderResponse {
  success: boolean;
  failure_reason?: string;
  error_response?: {
    error: string;
    message?: string;
    error_details?: Record<string, unknown>;
  };
  success_response?: {
    order_id: string;
  };
}

interface HistoricalOrder {
  order_id: string;
  status: string;
  product_id: string;
  side: 'BUY' | 'SELL';
  created_time: string;
  average_filled_price?: string;
  total_filled?: string;
  total_fees?: string;
}

interface HistoricalFillsResponse {
  fills: Array<{
    entry_id: string;
    trade_id: string;
    order_id: string;
    product_id: string;
    price: string;
    size: string;
    commission: string;
    side: 'UNKNOWN_ORDER_SIDE' | 'BUY' | 'SELL';
    time: string;
    is_taker: boolean;
  }>;
  cursor?: string;
  has_next: boolean;
}

export interface AdvancedTradeOrderResult {
  orderId: string;
  fills: HistoricalFillsResponse['fills'];
  filledSize: number;
  filledValue: number;
  fees: number;
  averagePrice: number;
}

function signRequest(method: string, path: string, body: string) {
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const prehash = `${timestamp}${method.toUpperCase()}${path}${body}`;
  const hmac = crypto.createHmac('sha256', Buffer.from(API_SECRET, 'base64'));
  const signature = hmac.update(prehash).digest('base64');
  return { timestamp, signature };
}

async function coinbaseRequest<T>(method: string, path: string, body?: unknown): Promise<T> {
  if (!API_KEY || !API_SECRET) {
    throw new Error('Coinbase credentials are not configured. Set COINBASE_API_KEY / COINBASE_API_SECRET.');
  }

  const requestPath = `${API_PREFIX}${path}`;
  const payload = body ? JSON.stringify(body) : '';
  const { timestamp, signature } = signRequest(method, requestPath, payload);

  const response = await fetch(`${API_BASE}${requestPath}`, {
    method,
    headers: {
      'X-CC-API-KEY': API_KEY,
      'X-CC-API-SIGNATURE': signature,
      'X-CC-API-TIMESTAMP': timestamp,
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: payload || undefined,
  });

  if (!response.ok) {
    const details = await response.text();
    throw new Error(`Coinbase Advanced Trade error ${response.status}: ${details || response.statusText}`);
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json() as Promise<T>;
}

async function pollOrder(orderId: string, timeoutMs = 20_000): Promise<HistoricalOrder> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const response = await coinbaseRequest<{ orders: HistoricalOrder[] }>('GET', `/orders/historical/${orderId}`);
    const order = response.orders?.[0];
    if (order && order.status !== 'OPEN' && order.status !== 'PENDING') {
      return order;
    }
    await sleep(500);
  }
  throw new Error(`Timed out waiting for order ${orderId} to settle`);
}

async function fetchFills(orderId: string): Promise<HistoricalFillsResponse['fills']> {
  const response = await coinbaseRequest<HistoricalFillsResponse>('GET', `/orders/historical/fills?order_id=${orderId}`);
  return response.fills ?? [];
}

function summarizeFills(fills: HistoricalFillsResponse['fills']) {
  let filledSize = 0;
  let filledValue = 0;
  let fees = 0;

  for (const fill of fills) {
    const size = parseFloat(fill.size);
    const price = parseFloat(fill.price);
    const commission = parseFloat(fill.commission);

    if (Number.isFinite(size) && Number.isFinite(price)) {
      filledSize += size;
      filledValue += size * price;
    }
    if (Number.isFinite(commission)) {
      fees += commission;
    }
  }

  const averagePrice = filledSize > 0 ? filledValue / filledSize : 0;
  return { filledSize, filledValue, fees, averagePrice };
}

async function placeOrder(body: any): Promise<AdvancedTradeOrderResult> {
  const response = await coinbaseRequest<PlaceOrderResponse>('POST', '/orders', body);
  if (!response.success || !response.success_response) {
    const reason = response.failure_reason ?? response.error_response?.message ?? 'Unknown failure';
    throw new Error(`Coinbase order failed: ${reason}`);
  }

  const orderId = response.success_response.order_id;
  await pollOrder(orderId);
  const fills = await fetchFills(orderId);
  const summary = summarizeFills(fills);

  return {
    orderId,
    fills,
    ...summary,
  };
}

export async function placeMarketBuy(productId: string, quoteAmountUsd: number): Promise<AdvancedTradeOrderResult> {
  const quote = quoteAmountUsd.toFixed(2);
  return placeOrder({
    client_order_id: randomUUID(),
    product_id: productId,
    side: 'BUY',
    order_configuration: {
      market_market_ioc: {
        quote_size: quote,
      },
    },
  });
}

export async function placeMarketSell(productId: string, baseAmount: number): Promise<AdvancedTradeOrderResult> {
  if (baseAmount <= 0) {
    throw new Error(`Cannot place sell order with size ${baseAmount}`);
  }
  const baseSize = baseAmount.toFixed(8);
  return placeOrder({
    client_order_id: randomUUID(),
    product_id: productId,
    side: 'SELL',
    order_configuration: {
      market_market_ioc: {
        base_size: baseSize,
      },
    },
  });
}
