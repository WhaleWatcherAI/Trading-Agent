import { promises as fs } from 'fs';
import * as path from 'path';
import { AlpacaOptionContract } from './alpaca';

const CACHE_VERSION = 1;
const CACHE_FILE = path.resolve(
  process.env.ALPACA_OPTION_CACHE_FILE || './data/alpaca-option-cache.json',
);

interface OptionChainCacheEntry {
  expiration: string;
  updatedAt: string;
  contracts: AlpacaOptionContract[];
}

interface SymbolCacheEntry {
  updatedAt: string;
  expirations: string[];
  chains: Record<string, OptionChainCacheEntry>;
}

interface OptionCacheFile {
  version: number;
  symbols: Record<string, SymbolCacheEntry>;
}

export interface CachedExpirations {
  updatedAt: string;
  expirations: string[];
}

export interface CachedOptionChain {
  expiration: string;
  updatedAt: string;
  contracts: AlpacaOptionContract[];
}

export const DEFAULT_CACHE_MAX_AGE_MS = 72 * 60 * 60 * 1000; // 3 days covers weekends

let cache: OptionCacheFile | null = null;
let loadPromise: Promise<OptionCacheFile> | null = null;

function ensureCacheShape(data: any): OptionCacheFile {
  const base: OptionCacheFile = { version: CACHE_VERSION, symbols: {} };
  if (!data || typeof data !== 'object') {
    return base;
  }

  const symbols: Record<string, SymbolCacheEntry> = {};
  if (data.symbols && typeof data.symbols === 'object') {
    for (const rawKey of Object.keys(data.symbols)) {
      const entry = data.symbols[rawKey];
      if (!entry || typeof entry !== 'object') continue;
      const chains: Record<string, OptionChainCacheEntry> = {};
      if (entry.chains && typeof entry.chains === 'object') {
        for (const chainKey of Object.keys(entry.chains)) {
          const chain = entry.chains[chainKey];
          if (!chain || typeof chain !== 'object') continue;
          if (!Array.isArray(chain.contracts)) continue;
          if (typeof chain.expiration !== 'string' || typeof chain.updatedAt !== 'string') {
            continue;
          }
          chains[chainKey] = {
            expiration: chain.expiration,
            updatedAt: chain.updatedAt,
            contracts: chain.contracts,
          };
        }
      }
      if (!Array.isArray(entry.expirations) || typeof entry.updatedAt !== 'string') {
        continue;
      }
      symbols[rawKey] = {
        updatedAt: entry.updatedAt,
        expirations: entry.expirations,
        chains,
      };
    }
  }

  return { version: CACHE_VERSION, symbols };
}

async function loadCacheFromDisk(): Promise<OptionCacheFile> {
  if (cache) return cache;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    try {
      const raw = await fs.readFile(CACHE_FILE, 'utf8');
      const parsed = JSON.parse(raw);
      cache = ensureCacheShape(parsed);
      return cache;
    } catch (err: any) {
      if (err?.code !== 'ENOENT') {
        console.warn('[option-cache] Failed to read cache file:', err?.message || err);
      }
      cache = { version: CACHE_VERSION, symbols: {} };
      return cache;
    }
  })();

  return loadPromise;
}

async function persistCache(updated: OptionCacheFile): Promise<void> {
  cache = updated;
  await fs.mkdir(path.dirname(CACHE_FILE), { recursive: true });
  await fs.writeFile(CACHE_FILE, JSON.stringify(updated, null, 2), 'utf8');
}

function symbolKey(symbol: string): string {
  return symbol.trim().toUpperCase();
}

function nowIso(): string {
  return new Date().toISOString();
}

function getAgeMs(updatedAt: string): number {
  const updatedTime = new Date(updatedAt).getTime();
  if (!Number.isFinite(updatedTime)) return Number.POSITIVE_INFINITY;
  return Date.now() - updatedTime;
}

export function isCacheEntryFresh(updatedAt: string, maxAgeMs = DEFAULT_CACHE_MAX_AGE_MS): boolean {
  return getAgeMs(updatedAt) <= maxAgeMs;
}

export async function getCachedExpirations(symbol: string): Promise<CachedExpirations | null> {
  const data = await loadCacheFromDisk();
  const entry = data.symbols[symbolKey(symbol)];
  if (!entry || entry.expirations.length === 0) {
    return null;
  }
  return { updatedAt: entry.updatedAt, expirations: entry.expirations };
}

export async function setCachedExpirations(symbol: string, expirations: string[]): Promise<void> {
  const data = await loadCacheFromDisk();
  const key = symbolKey(symbol);
  const existing = data.symbols[key] ?? {
    updatedAt: nowIso(),
    expirations: [],
    chains: {},
  };
  data.symbols[key] = {
    ...existing,
    updatedAt: nowIso(),
    expirations,
  };
  await persistCache(data);
}

export async function getCachedOptionChain(
  symbol: string,
  expiration: string,
): Promise<CachedOptionChain | null> {
  const data = await loadCacheFromDisk();
  const entry = data.symbols[symbolKey(symbol)];
  if (!entry) return null;
  const chain = entry.chains[expiration];
  if (!chain || !Array.isArray(chain.contracts) || chain.contracts.length === 0) {
    return null;
  }
  return { expiration: chain.expiration, updatedAt: chain.updatedAt, contracts: chain.contracts };
}

export async function setCachedOptionChain(
  symbol: string,
  expiration: string,
  contracts: AlpacaOptionContract[],
): Promise<void> {
  const data = await loadCacheFromDisk();
  const key = symbolKey(symbol);
  const entry =
    data.symbols[key] ??
    ({
      updatedAt: nowIso(),
      expirations: [],
      chains: {},
    } as SymbolCacheEntry);

  entry.chains = {
    ...entry.chains,
    [expiration]: {
      expiration,
      updatedAt: nowIso(),
      contracts,
    },
  };
  data.symbols[key] = entry;
  await persistCache(data);
}
