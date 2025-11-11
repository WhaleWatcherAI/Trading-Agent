import axios, { AxiosInstance } from 'axios';

function getTopstepEnv() {
  return {
    apiKey: process.env.TOPSTEPX_API_KEY || '',
    baseUrl: process.env.TOPSTEPX_BASE_URL || 'https://api.topstepx.com',
    username: process.env.TOPSTEPX_USERNAME || '',
  };
}

let sessionToken: string | null = null;
let tokenExpiry: number = 0;

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
    const response = await axios.post(
      `${baseUrl}/api/Auth/loginKey`,
      {
        userName: username,
        apiKey,
      },
      {
        headers: {
          'Accept': 'text/plain',
          'Content-Type': 'application/json',
        },
        timeout: 30000,
      }
    );

    if (!response.data.success || response.data.errorCode !== 0) {
      throw new Error(`Authentication failed: Error Code ${response.data.errorCode}`);
    }

    sessionToken = response.data.token;
    // Token expires in 24 hours, refresh 1 hour before expiry
    tokenExpiry = Date.now() + (23 * 60 * 60 * 1000);

    return sessionToken;
  } catch (error: any) {
    console.error('[topstepx] Authentication failed:', error.response?.data || error.message);
    throw error;
  }
}

/**
 * Get an authenticated axios client with JWT token
 */
async function getAuthenticatedClient(): Promise<AxiosInstance> {
  const token = await authenticate();
  const { baseUrl } = getTopstepEnv();

  return axios.create({
    baseURL: baseUrl,
    timeout: 30000,
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
  });
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

  try {
    const client = await getAuthenticatedClient();
    const response = await client.post('/api/History/retrieveBars', {
      contractId,
      live,
      startTime,
      endTime,
      unit,
      unitNumber,
      limit,
      includePartialBar: false,
    });

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

export async function fetchTopstepXFuturesMetadata(symbolOrContractId: string): Promise<TopstepXContract | null> {
  assertTopstepXReady();

  const identifier = symbolOrContractId?.trim();
  if (!identifier) {
    return null;
  }

  const isContractId = identifier.startsWith('CON.');
  if (isContractId) {
    return fetchTopstepXContract(identifier);
  }

  try {
    const contracts = await fetchTopstepXContracts(false);
    const upper = identifier.toUpperCase();
    return contracts.find(contract => contract.name?.toUpperCase() === upper) || null;
  } catch (error: any) {
    console.error('[topstepx] Failed to fetch contract metadata list:', error.response?.data || error.message);
    return null;
  }
}
