import fetch, { Response } from 'node-fetch';
import { createJwtManager } from './auth/jwt-manager';
import { fetchJwt } from './auth/fetch-jwt';

interface PlaceOrderRequest {
  request: {
    accountId: number;
    contractId: string;
    side: 0 | 1; // 0 = Buy, 1 = Sell
    size: number;
    type: 1 | 2 | 3 | 4 | 5 | 6 | 7; // 1 = Limit, 2 = Market, etc.
    timeInForce: 0 | 1 | 3; // 0 = IOC, 1 = GTC, 3 = FOK (common values, confirm if needed)
    price?: number; // Required for Limit, StopLimit, etc.
    stopPrice?: number; // Required for Stop, StopLimit
    // Add other potential fields as needed, e.g., stopLossBracket, takeProfitBracket
  }
}

interface CancelOrderRequest {
  accountId: number;
  orderId: string;
}

export function createProjectXRest(baseUrl?: string) {
  const resolvedBase = baseUrl || process.env.TOPSTEPX_REST_BASE || process.env.TOPSTEPX_BASE_URL;
  if (!resolvedBase) {
    throw new Error('ProjectX REST base URL not set');
  }
  const jwtMgr = createJwtManager(fetchJwt);

  async function doFetch(path: string, init: RequestInit, tryRefresh = true): Promise<Response> {
    const jwt = await jwtMgr.getJwt();
    console.log(`[doFetch] Making request to ${resolvedBase}${path}`);
    const res = await fetch(`${resolvedBase}${path}`, {
      ...init,
      headers: {
        'Authorization': `Bearer ${jwt}`,
        'Content-Type': 'application/json',
        ...(init.headers || {}),
      },
    });
    console.log(`[doFetch] Received response with status ${res.status}`);
    if (res.status === 401 && tryRefresh) {
      await jwtMgr.markStaleAndRefresh();
      return doFetch(path, init, false);
    }
    return res;
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
      return JSON.parse(text);
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
    getPositions: async (accountId: number) => {
      const res = await doFetch(`/api/Account/${accountId}/positions`, { method: 'GET' });
      console.log(`[getPositions] Response status: ${res.status}`);
      const text = await res.text();
      if (!res.ok) throw new Error(`getPositions ${res.status}: ${text}`);
      return JSON.parse(text);
    },
  };
}
