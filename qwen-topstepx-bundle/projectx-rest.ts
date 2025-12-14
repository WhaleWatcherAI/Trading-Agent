import fetch, { Response } from 'node-fetch';
import { createJwtManager } from './auth/jwt-manager';
import { fetchJwt } from './auth/fetch-jwt';

// Singleton JWT manager - shared across all REST clients to prevent 401 storms
let sharedJwtManager: ReturnType<typeof createJwtManager> | null = null;
function getSharedJwtManager() {
  if (!sharedJwtManager) {
    sharedJwtManager = createJwtManager(fetchJwt);
  }
  return sharedJwtManager;
}

// Singleton REST client
let restClientSingleton: ReturnType<typeof createProjectXRest> | null = null;

/**
 * Get singleton REST client - ensures all code shares the same JWT token
 */
export function getProjectXRest(baseUrl?: string) {
  if (!restClientSingleton) {
    restClientSingleton = createProjectXRest(baseUrl);
  }
  return restClientSingleton;
}

interface BracketConfig {
  ticks: number;
  type: 1 | 2 | 4 | 5 | 6 | 7; // 1=Limit, 2=Market, 4=Stop, 5=TrailingStop, 6=JoinBid, 7=JoinAsk
}

interface PlaceOrderRequest {
  accountId: number;
  contractId: string;
  side: 0 | 1; // 0 = Buy, 1 = Sell
  size: number;
  type: 1 | 2 | 3 | 4 | 5 | 6 | 7; // 1 = Limit, 2 = Market, etc.
  timeInForce: 0 | 1 | 3; // 0 = IOC, 1 = GTC, 3 = FOK (common values, confirm if needed)
  limitPrice?: number; // Required for Limit orders
  stopPrice?: number; // Required for Stop, StopLimit
  stopLossBracket?: BracketConfig;
  takeProfitBracket?: BracketConfig;
}

interface CancelOrderRequest {
  accountId: number;
  orderId: string;
}

interface ModifyOrderRequest {
  accountId: number;
  orderId: string | number;
  size?: number;
  limitPrice?: number | null;
  stopPrice?: number | null;
  trailPrice?: number | null;
}

interface SearchOrdersRequest {
  accountId: number;
  startTimestamp: string;
  endTimestamp?: string | null;
}

interface SearchOpenOrdersRequest {
  accountId: number;
}

export interface ProjectXOrderRecord {
  id: number;
  accountId: number;
  contractId: string;
  symbolId?: string;
  creationTimestamp: string;
  updateTimestamp?: string;
  status: number;
  type: number;
  side: number;
  size: number;
  limitPrice?: number | null;
  stopPrice?: number | null;
  fillVolume?: number | null;
  filledPrice?: number | null;
  customTag?: string | null;
}

interface OrdersResponse {
  orders: ProjectXOrderRecord[];
  success: boolean;
  errorCode: number;
  errorMessage: string | null;
}

export function createProjectXRest(baseUrl?: string) {
  const resolvedBase = baseUrl || process.env.TOPSTEPX_REST_BASE || process.env.TOPSTEPX_BASE_URL;
  if (!resolvedBase) {
    throw new Error('ProjectX REST base URL not set');
  }
  // Use shared JWT manager to prevent 401 storms from multiple clients
  const jwtMgr = getSharedJwtManager();

  async function doFetch(path: string, init: RequestInit, tryRefresh = true): Promise<Response> {
    const jwt = await jwtMgr.getJwt();
    console.log(`[doFetch] Making request to ${resolvedBase}${path}`);

    // Create abort controller for timeout (30 seconds)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const res = await fetch(`${resolvedBase}${path}`, {
        ...init,
        headers: {
          'Authorization': `Bearer ${jwt}`,
          'Content-Type': 'application/json',
          ...(init.headers || {}),
        },
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      console.log(`[doFetch] Received response with status ${res.status}`);
      if (res.status === 401 && tryRefresh) {
        await jwtMgr.markStaleAndRefresh();
        return doFetch(path, init, false);
      }
      return res;
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        console.error(`[doFetch] Request to ${path} timed out after 30 seconds`);
        throw new Error(`Request timeout: ${path}`);
      }
      throw error;
    }
  }

  return {
    placeOrder: async (payload: PlaceOrderRequest) => {
      const res = await doFetch('/api/Order/place', { method: 'POST', body: JSON.stringify(payload) });
      console.log(`[placeOrder] Response status: ${res.status}`);
      console.log(`[placeOrder] Attempting to read response text...`);
      const text = await res.text();
      console.log(`[placeOrder] Response text received (length: ${text.length}). Status: ${res.status}`);
      if (!res.ok) {
        throw new Error(`place ${res.status}: ${text}`);
      }
      console.log(`[placeOrder] Attempting to parse JSON...`);
      console.log(`[placeOrder] Raw response text:`, text);
      try {
        const parsed = JSON.parse(text);
        console.log(`[placeOrder] Parsed response:`, JSON.stringify(parsed));
        return parsed;
      } catch (parseError: any) {
        console.error(`[placeOrder] ❌ JSON PARSE FAILED:`, parseError.message);
        console.error(`[placeOrder] ❌ Raw text that failed to parse:`, text);
        throw new Error(`Failed to parse placeOrder response: ${parseError.message}`);
      }
    },
    cancelOrder: async (payload: CancelOrderRequest) => {
      const res = await doFetch('/api/Order/cancel', { method: 'POST', body: JSON.stringify(payload) });
      console.log(`[cancelOrder] Response status: ${res.status}`);
      console.log(`[cancelOrder] Attempting to read response text...`);
      const text = await res.text();
      console.log(`[cancelOrder] Response text received (length: ${text.length}). Status: ${res.status}`);
      if (!res.ok) throw new Error(`cancel ${res.status}: ${text}`);
      console.log(`[cancelOrder] Attempting to parse JSON...`);
      return JSON.parse(text);
    },
    modifyOrder: async (payload: ModifyOrderRequest) => {
      console.log(`[modifyOrder] Sending payload:`, JSON.stringify(payload));
      const res = await doFetch('/api/Order/modify', { method: 'POST', body: JSON.stringify(payload) });
      console.log(`[modifyOrder] Response status: ${res.status}`);
      const text = await res.text();
      console.log(`[modifyOrder] Response text received (length: ${text.length}). Status: ${res.status}`);
      console.log(`[modifyOrder] Response body:`, text);
      if (!res.ok) {
        throw new Error(`modify ${res.status}: ${text}`);
      }
      return JSON.parse(text);
    },
    searchOrders: async (payload: SearchOrdersRequest): Promise<OrdersResponse> => {
      const res = await doFetch('/api/Order/search', { method: 'POST', body: JSON.stringify(payload) });
      console.log(`[searchOrders] Response status: ${res.status}`);
      const text = await res.text();
      console.log(`[searchOrders] Response text received (length: ${text.length}). Status: ${res.status}`);
      if (!res.ok) {
        throw new Error(`searchOrders ${res.status}: ${text}`);
      }
      return JSON.parse(text);
    },
    searchOpenOrders: async (payload: SearchOpenOrdersRequest): Promise<OrdersResponse> => {
      const res = await doFetch('/api/Order/searchOpen', { method: 'POST', body: JSON.stringify(payload) });
      console.log(`[searchOpenOrders] Response status: ${res.status}`);
      const text = await res.text();
      console.log(`[searchOpenOrders] Response text received (length: ${text.length}). Status: ${res.status}`);
      if (!res.ok) {
        throw new Error(`searchOpenOrders ${res.status}: ${text}`);
      }
      return JSON.parse(text);
    },
    getPositions: async (accountId: number) => {
      const res = await doFetch(`/api/Account/${accountId}/positions`, { method: 'GET' });
      console.log(`[getPositions] Response status: ${res.status}`);
      const text = await res.text();
      // 404 means no positions - return empty array
      if (res.status === 404) return [];
      if (!res.ok) throw new Error(`getPositions ${res.status}: ${text}`);
      return JSON.parse(text);
    },
  };
}
