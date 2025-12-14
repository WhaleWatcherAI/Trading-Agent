import type { RequestOptions } from 'openai/core';

const CACHE_ENABLED = (process.env.DEEPSEEK_CACHE_ENABLED ?? 'true').toLowerCase() !== 'false';
const DEFAULT_MODE = process.env.DEEPSEEK_CACHE_MODE;
const DEFAULT_TTL = process.env.DEEPSEEK_CACHE_TTL;
const DEFAULT_VERSION = process.env.DEEPSEEK_CACHE_VERSION;
const DEFAULT_KEY = process.env.DEEPSEEK_CACHE_KEY;

/**
 * Build request options with DeepSeek cache headers.
 * - Uses signed cache key with optional version suffix.
 * - Set DEEPSEEK_CACHE_ENABLED=false to disable globally.
 * - Bump DEEPSEEK_CACHE_VERSION to bust cache after prompt changes.
 */
export function buildDeepseekCacheOptions(contextKey: string): RequestOptions {
  if (!CACHE_ENABLED) return {};

  const baseKey = DEFAULT_KEY || contextKey;
  const versionedKey = DEFAULT_VERSION ? `${baseKey}:v${DEFAULT_VERSION}` : baseKey;

  const headers: Record<string, string> = {
    'X-Use-Cache': 'true',
    'X-Cache-Key': versionedKey,
  };

  if (DEFAULT_MODE) {
    headers['X-Cache-Mode'] = DEFAULT_MODE;
  }

  if (DEFAULT_TTL) {
    headers['X-Cache-Ttl'] = DEFAULT_TTL;
  }

  return { headers };
}
