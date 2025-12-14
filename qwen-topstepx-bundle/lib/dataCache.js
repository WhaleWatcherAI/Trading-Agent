"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.STALENESS_CONFIG = void 0;
exports.getCached = getCached;
exports.setCache = setCache;
exports.isStale = isStale;
exports.getWithStaleness = getWithStaleness;
exports.getCacheKeys = getCacheKeys;
exports.clearCache = clearCache;
exports.getCacheStats = getCacheStats;
exports.expireOldEntries = expireOldEntries;
// In-memory cache (upgrade to Redis for production)
const cache = new Map();
// Staleness budgets per data type (in milliseconds)
exports.STALENESS_CONFIG = {
    // High priority - poll frequently
    'market_data': { budget: 30000, pollInterval: 30000 }, // 30s
    'spot_gex': { budget: 60000, pollInterval: 60000 }, // 1min
    'put_call_ratio': { budget: 30000, pollInterval: 30000 }, // 30s
    // Medium priority - poll moderately
    'volatility_stats': { budget: 120000, pollInterval: 120000 }, // 2min
    'sector_tide': { budget: 180000, pollInterval: 180000 }, // 3min
    'greek_flow': { budget: 180000, pollInterval: 180000 }, // 3min
    'news_headlines': { budget: 120000, pollInterval: 120000 }, // 2min
    'technicals': { budget: 300000, pollInterval: 300000 }, // 5min (indicators don't change fast)
    // Low priority - poll rarely
    'institutional': { budget: 600000, pollInterval: 600000 }, // 10min
};
/**
 * Get data from cache
 */
function getCached(key) {
    return cache.get(key) || null;
}
/**
 * Set data in cache
 */
function setCache(key, data, source) {
    cache.set(key, {
        data,
        updatedAt: Date.now(),
        source,
    });
}
/**
 * Check if cached data is stale
 */
function isStale(key, budgetMs) {
    const entry = cache.get(key);
    if (!entry)
        return true;
    return Date.now() - entry.updatedAt > budgetMs;
}
/**
 * Get cached data with staleness info
 */
function getWithStaleness(key, budget) {
    const entry = getCached(key);
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
function getCacheKeys() {
    return Array.from(cache.keys());
}
/**
 * Clear all cache (useful for testing)
 */
function clearCache() {
    cache.clear();
}
/**
 * Get cache stats
 */
function getCacheStats() {
    const stats = {
        entries: cache.size,
        keys: Array.from(cache.keys()),
        sizes: {},
    };
    cache.forEach((value, key) => {
        stats.sizes[key] = JSON.stringify(value.data).length;
    });
    return stats;
}
/**
 * Expire old entries (run periodically)
 */
function expireOldEntries(maxAgeMs = 3600000) {
    const now = Date.now();
    const toDelete = [];
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
