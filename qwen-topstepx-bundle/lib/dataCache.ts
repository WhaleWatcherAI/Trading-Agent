/**
 * Data Cache Service with Background Polling
 *
 * Implements intelligent caching with staleness budgets to avoid hitting
 * API rate limits (120 req/min, 14k/day) on every user request.
 *
 * Strategy:
 * - Background workers poll endpoints at optimal intervals
 * - Chat/analyze endpoints read from cache (fast, no blocking)
 * - Each data type has a staleness budget
 */

interface CacheEntry<T> {
  data: T;
  updatedAt: number;
  source: string;
}

interface StalenessConfig {
  budget: number; // milliseconds
  pollInterval: number; // milliseconds
}

// In-memory cache (upgrade to Redis for production)
const cache = new Map<string, CacheEntry<any>>();

// Staleness budgets per data type (in milliseconds)
export const STALENESS_CONFIG: Record<string, StalenessConfig> = {
  // High priority - poll frequently
  'market_data': { budget: 30_000, pollInterval: 30_000 },        // 30s
  'spot_gex': { budget: 60_000, pollInterval: 60_000 },           // 1min
  'put_call_ratio': { budget: 30_000, pollInterval: 30_000 },     // 30s

  // Medium priority - poll moderately
  'volatility_stats': { budget: 120_000, pollInterval: 120_000 }, // 2min
  'sector_tide': { budget: 180_000, pollInterval: 180_000 },      // 3min
  'greek_flow': { budget: 180_000, pollInterval: 180_000 },       // 3min
  'news_headlines': { budget: 120_000, pollInterval: 120_000 },   // 2min
  'technicals': { budget: 300_000, pollInterval: 300_000 },       // 5min (indicators don't change fast)

  // Low priority - poll rarely
  'institutional': { budget: 600_000, pollInterval: 600_000 },    // 10min
};

/**
 * Get data from cache
 */
export function getCached<T>(key: string): CacheEntry<T> | null {
  return cache.get(key) || null;
}

/**
 * Set data in cache
 */
export function setCache<T>(key: string, data: T, source: string): void {
  cache.set(key, {
    data,
    updatedAt: Date.now(),
    source,
  });
}

/**
 * Check if cached data is stale
 */
export function isStale(key: string, budgetMs: number): boolean {
  const entry = cache.get(key);
  if (!entry) return true;
  return Date.now() - entry.updatedAt > budgetMs;
}

/**
 * Get cached data with staleness info
 */
export function getWithStaleness<T>(key: string, budget: number): {
  data: T | null;
  staleFor: number;
  isFresh: boolean;
} {
  const entry = getCached<T>(key);

  if (!entry) {
    return { data: null, staleFor: Infinity, isFresh: false };
  }

  const age = Date.now() - entry.updatedAt;
  const staleFor = Math.max(0, age - budget);

  return {
    data: entry.data,
    staleFor,
    isFresh: age <= budget,
  };
}

export function getCacheKeys(): string[] {
  return Array.from(cache.keys());
}

/**
 * Clear all cache (useful for testing)
 */
export function clearCache(): void {
  cache.clear();
}

/**
 * Get cache stats
 */
export function getCacheStats(): {
  entries: number;
  keys: string[];
  sizes: Record<string, number>;
} {
  const stats = {
    entries: cache.size,
    keys: Array.from(cache.keys()),
    sizes: {} as Record<string, number>,
  };

  cache.forEach((value, key) => {
    stats.sizes[key] = JSON.stringify(value.data).length;
  });

  return stats;
}

/**
 * Expire old entries (run periodically)
 */
export function expireOldEntries(maxAgeMs: number = 3_600_000): void {
  const now = Date.now();
  const toDelete: string[] = [];

  cache.forEach((entry, key) => {
    if (now - entry.updatedAt > maxAgeMs) {
      toDelete.push(key);
    }
  });

  toDelete.forEach(key => cache.delete(key));

  if (toDelete.length > 0) {
    console.log(`♻️  Expired ${toDelete.length} stale cache entries`);
  }
}
