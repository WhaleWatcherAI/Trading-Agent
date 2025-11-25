import axios, { AxiosInstance } from 'axios';
import fetch from 'node-fetch';
import { createProjectXRest } from '../projectx-rest';

function getTopstepEnv() {
  return {
    apiKey: process.env.TOPSTEPX_API_KEY || '',
  baseUrl: process.env.TOPSTEPX_BASE_URL || 'https://gateway.topstepx.com',
    username: process.env.TOPSTEPX_USERNAME || '',
  };
}

let sessionToken: string | null = null;
let tokenExpiry: number = 0;
let restClientSingleton: ReturnType<typeof createProjectXRest> | null = null;

export interface TopstepXFuturesBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface FetchFuturesBarsParams {
  contractId: string;  // e.g., 'CON.F.US.MNQ.U25'
  startTime: string;   // ISO 8601 format
  endTime: string;     // ISO 8601 format
  unit?: 1 | 2 | 3 | 4;  // 1=Second, 2=Minute, 3=Hour, 4=Day
  unitNumber?: number;   // e.g., 5 for 5-minute bars
  limit?: number;        // Max bars to return (default: 20000)
  live?: boolean;        // false for sim/demo
}

export interface TopstepXContract {
  id: string;
  name: string;
  description?: string;
  tickSize: number;
  tickValue: number;
  multiplier?: number;
  exchange?: string;
}

export interface TopstepXAccount {
  id: number;
  name: string;
  balance: number;
  canTrade: boolean;
  isVisible: boolean;
}

export function assertTopstepXReady() {
  const { apiKey, username } = getTopstepEnv();
  if (!apiKey || !username) {
    throw new Error('TOPSTEPX_API_KEY and TOPSTEPX_USERNAME must be set in environment before calling TopstepX.');
  }
}

/**
 * Authenticate and get a JWT session token.
 * Token is cached and reused until it expires (24 hours).
 */
export async function authenticate(): Promise<string> {
  assertTopstepXReady();
  const { apiKey, username, baseUrl } = getTopstepEnv();

  // Return cached token if still valid
  if (sessionToken && Date.now() < tokenExpiry) {
    return sessionToken;
  }

  try {
    // Use fetch instead of axios for better proxy compatibility
    const response = await fetch(`${baseUrl}/api/Auth/loginKey`, {
      method: 'POST',
      headers: {
        'Accept': 'text/plain',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userName: username,
        apiKey,
      }),
    });

    if (!response.ok) {
      throw new Error(`Authentication HTTP error: ${response.status}`);
    }

    const data = await response.json();

    if (!data.success || data.errorCode !== 0) {
      throw new Error(`Authentication failed: Error Code ${data.errorCode}`);
    }

    sessionToken = data.token;
    // Token expires in 24 hours, refresh 1 hour before expiry
    tokenExpiry = Date.now() + (23 * 60 * 60 * 1000);

    return sessionToken;
  } catch (error: any) {
    console.error('[topstepx] Authentication failed:', error.message);
    throw error;
  }
}

/**
 * Sleep helper for rate limit backoff
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Get an authenticated axios client with JWT token and retry logic
 */
async function getAuthenticatedClient(): Promise<AxiosInstance> {
  const token = await authenticate();
  const { baseUrl } = getTopstepEnv();

  const client = axios.create({
    baseURL: baseUrl,
    timeout: 30000,
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
  });

  // Add response interceptor for rate limit handling
  client.interceptors.response.use(
    (response) => response,
    async (error) => {
      const config = error.config;

      // Handle 429 rate limit errors with jitter to prevent multi-agent collision
      if (error.response?.status === 429) {
        // Check if we've already retried too many times
        config._retryCount = config._retryCount || 0;

        if (config._retryCount < 5) {
          config._retryCount += 1;

          // Exponential backoff: 2s, 4s, 8s, 16s, 32s + random jitter
          const baseDelayMs = Math.pow(2, config._retryCount) * 1000;
          const jitterMs = Math.floor(Math.random() * 2000); // 0-2000ms random jitter
          const delayMs = baseDelayMs + jitterMs;
          console.log(`[topstepx] Rate limited (429), retrying in ${delayMs}ms (base ${baseDelayMs}ms + jitter ${jitterMs}ms) (attempt ${config._retryCount}/5)`);

          await sleep(delayMs);

          // Retry the request
          return client(config);
        }
      }

      return Promise.reject(error);
    }
  );

  return client;
}

/**
 * Fetch historical bars for a futures contract
 */
export async function fetchTopstepXFuturesBars(
  params: FetchFuturesBarsParams
): Promise<TopstepXFuturesBar[]> {
  assertTopstepXReady();

  const {
    contractId,
    startTime,
    endTime,
    unit = 2,           // Default to minutes
    unitNumber = 1,     // Default to 1-minute bars
    limit = 20000,
    live = false,
  } = params;

  const requestBody = {
    contractId,
    live,
    startTime,
    endTime,
    unit,
    unitNumber,
    limit,
    includePartialBar: false,
  };

  console.log('[topstepx] DEBUG: Fetching historical bars with request:', JSON.stringify(requestBody, null, 2));

  try {
    const client = await getAuthenticatedClient();
    const response = await client.post('/api/History/retrieveBars', requestBody);

    if (!response.data.success || !Array.isArray(response.data.bars)) {
      console.warn('[topstepx] Unexpected response format for history:', response.data);
      return [];
    }

    return response.data.bars.map((bar: any) => ({
      timestamp: bar.t,  // Note: API returns 't' not 'timestamp'
      open: Number(bar.o),
      high: Number(bar.h),
      low: Number(bar.l),
      close: Number(bar.c),
      volume: bar.v != null ? Number(bar.v) : undefined,
    }));
  } catch (error: any) {
    console.error('[topstepx] Failed to fetch futures bars:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Fetch available contracts
 */
export async function fetchTopstepXContracts(live: boolean = false): Promise<TopstepXContract[]> {
  assertTopstepXReady();

  try {
    const client = await getAuthenticatedClient();
    const response = await client.post('/api/Contract/available', { live });

    if (!response.data.success || !Array.isArray(response.data.contracts)) {
      console.warn('[topstepx] Unexpected response format for contracts:', response.data);
      return [];
    }

    return response.data.contracts.map((contract: any) => ({
      id: contract.id,
      name: contract.name,
      description: contract.description,
      tickSize: contract.tickSize,
      tickValue: contract.tickValue,
      multiplier: contract.multiplier,
      exchange: contract.exchange,
    }));
  } catch (error: any) {
    console.error('[topstepx] Failed to fetch contracts:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Fetch trading accounts
 */
export async function fetchTopstepXAccounts(onlyActive: boolean = true): Promise<TopstepXAccount[]> {
  assertTopstepXReady();

  try {
    const client = await getAuthenticatedClient();
    const response = await client.post('/api/Account/search', {
      onlyActiveAccounts: onlyActive,
    });

    if (!response.data.success || !Array.isArray(response.data.accounts)) {
      console.warn('[topstepx] Unexpected response format for accounts:', response.data);
      return [];
    }

    return response.data.accounts.map((acc: any) => ({
      id: acc.id,
      name: acc.name,
      balance: typeof acc.balance === 'number' ? acc.balance : Number(acc.balance ?? 0),
      canTrade: acc.canTrade,
      isVisible: acc.isVisible,
    }));
  } catch (error: any) {
    console.error('[topstepx] Failed to fetch accounts:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Search for a specific contract by ID
 */
export async function fetchTopstepXContract(contractId: string): Promise<TopstepXContract | null> {
  assertTopstepXReady();

  try {
    const client = await getAuthenticatedClient();
    const response = await client.post('/api/Contract/searchById', {
      contractId: contractId,  // API expects 'contractId' not 'id'
    });

    if (!response.data.success || !response.data.contract) {
      return null;
    }

    const contract = response.data.contract;
    return {
      id: contract.id,
      name: contract.name,
      description: contract.description,
      tickSize: contract.tickSize,
      tickValue: contract.tickValue,
      multiplier: contract.multiplier,
      exchange: contract.exchange,
    };
  } catch (error: any) {
    console.error('[topstepx] Failed to fetch contract:', error.response?.data || error.message);
    return null;
  }
}

// Simple cache for contract metadata to avoid repeated API calls
const contractMetadataCache: Map<string, TopstepXContract> = new Map();

// Known contract ID mappings for common symbols (avoids fetching full contract list)
const KNOWN_CONTRACT_IDS: Record<string, string> = {
  'NQZ5': 'CON.F.US.ENQ.Z25',
  'MGCZ5': 'CON.F.US.MGC.Z25',
  'MNQ': 'CON.F.US.MNQ.Z25',
  'MGC': 'CON.F.US.MGC.Z25',
};

export async function fetchTopstepXFuturesMetadata(symbolOrContractId: string): Promise<TopstepXContract | null> {
  assertTopstepXReady();

  const identifier = symbolOrContractId?.trim();
  if (!identifier) {
    return null;
  }

  // Check cache first
  if (contractMetadataCache.has(identifier)) {
    return contractMetadataCache.get(identifier)!;
  }

  const isContractId = identifier.startsWith('CON.');
  if (isContractId) {
    const metadata = await fetchTopstepXContract(identifier);
    if (metadata) {
      contractMetadataCache.set(identifier, metadata);
    }
    return metadata;
  }

  // Try known contract ID mapping first (avoids fetching full list)
  const upper = identifier.toUpperCase();
  const knownContractId = KNOWN_CONTRACT_IDS[upper];
  if (knownContractId) {
    console.log(`[topstepx] Using known contract ID for ${upper}: ${knownContractId}`);
    const metadata = await fetchTopstepXContract(knownContractId);
    if (metadata) {
      contractMetadataCache.set(identifier, metadata);
      contractMetadataCache.set(upper, metadata);
      console.log(`[topstepx] Found contract ${metadata.name}: tickSize=${metadata.tickSize}, tickValue=${metadata.tickValue}`);
      return metadata;
    }
  }

  // Fallback: fetch full contract list (only if known mapping didn't work)
  try {
    const contracts = await fetchTopstepXContracts(false);
    const found = contracts.find(contract => contract.name?.toUpperCase() === upper) || null;
    if (found) {
      contractMetadataCache.set(identifier, found);
      contractMetadataCache.set(upper, found);
      console.log(`[topstepx] Found contract ${found.name}: tickSize=${found.tickSize}, tickValue=${found.tickValue}`);
    }
    return found;
  } catch (error: any) {
    console.error('[topstepx] Failed to fetch contract metadata list:', error.response?.data || error.message);
    return null;
  }
}

/**
 * Select trading account with balance less than 40k
 * Automatically selects appropriate account for trading
 */
export async function selectTradingAccount(maxBalance: number = 40000): Promise<TopstepXAccount | null> {
  assertTopstepXReady();

  try {
    const accounts = await fetchTopstepXAccounts(true);

    // Find accounts with balance < maxBalance
    const eligibleAccounts = accounts.filter(acc =>
      acc.canTrade &&
      acc.isVisible &&
      acc.balance < maxBalance
    );

    if (eligibleAccounts.length === 0) {
      console.warn(`[topstepx] No eligible trading accounts found with balance < $${maxBalance}`);
      return null;
    }

    // Return the account with the highest balance under the limit
    const selectedAccount = eligibleAccounts.sort((a, b) => b.balance - a.balance)[0];

    console.log(`[topstepx] Selected account: ${selectedAccount.name} (ID: ${selectedAccount.id}) with balance $${selectedAccount.balance.toFixed(2)}`);

    return selectedAccount;
  } catch (error: any) {
    console.error('[topstepx] Failed to select trading account:', error.response?.data || error.message);
    return null;
  }
}

/**
 * Order type and side enums
 */
export type OrderSide = 'buy' | 'sell';
export type OrderType = 'market' | 'limit';

/**
 * TopStepX Order Interface
 */
export interface TopstepXOrder {
  contractId: string;
  accountId: number;
  side: OrderSide;
  quantity: number;
  orderType: OrderType;
  limitPrice?: number;  // Required for limit orders
  stopPrice?: number;   // For stop-limit orders
  live?: boolean;       // false for sim/demo
}

/**
 * TopStepX Order Response
 */
export interface TopstepXOrderResponse {
  success: boolean;
  orderId?: string;
  errorCode?: number;
  errorMessage?: string;
  filledQuantity?: number;
  averagePrice?: number;
}

/**
 * Submit an order to TopStepX
 * Supports market and limit orders
 */
export async function submitTopstepXOrder(order: TopstepXOrder): Promise<TopstepXOrderResponse> {
  assertTopstepXReady();

  const {
    contractId,
    accountId,
    side,
    quantity,
    orderType,
    limitPrice,
    stopPrice,
  } = order;

  // Validate order
  if (quantity <= 0) {
    throw new Error('Order quantity must be greater than 0');
  }

  if (orderType === 'limit' && !limitPrice) {
    throw new Error('Limit price is required for limit orders');
  }

  const restOrderType = stopPrice
    ? 4 // Stop market
    : orderType === 'limit'
      ? 1 // Limit
      : 2; // Market

  const restPayload: any = {
    accountId,
    contractId,
    side: side === 'buy' ? 0 : 1,
    size: quantity,
    type: restOrderType,
    timeInForce: 1,
  };

  if (restOrderType === 1 && limitPrice) {
    restPayload.limitPrice = limitPrice;
  }
  if (restOrderType === 4 && stopPrice) {
    restPayload.stopPrice = stopPrice;
  }

  console.log(`[topstepx] Submitting ${orderType.toUpperCase()} ${side.toUpperCase()} order via REST:`, {
    contractId,
    accountId,
    quantity,
    limitPrice: limitPrice || 'N/A',
    stopPrice: stopPrice || 'N/A',
    restOrderType,
  });

  try {
    if (!restClientSingleton) {
      restClientSingleton = createProjectXRest();
    }

    const response = await restClientSingleton.placeOrder(restPayload);

    if (!response || response.success === false) {
      console.error('[topstepx] Order submission failed:', response);
      return {
        success: false,
        errorCode: response?.errorCode,
        errorMessage: response?.errorMessage || 'Order submission failed',
      };
    }

    return {
      success: true,
      orderId: response.orderId ? String(response.orderId) : undefined,
      filledQuantity: response.fillVolume,
      averagePrice: response.averagePrice,
    };
  } catch (error: any) {
    console.error('[topstepx] Failed to submit order:', error?.response?.data || error.message);
    return {
      success: false,
      errorCode: error?.response?.status,
      errorMessage: error?.response?.data?.errorMessage || error.message,
    };
  }
}
