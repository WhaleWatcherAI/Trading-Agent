/**
 * Background Polling Service
 *
 * Continuously polls Unusual Whales & Tradier APIs to keep cache fresh
 * without blocking user requests.
 *
 * Rate Limits:
 * - UW: 120 req/min, 14k/day
 * - Strategy: ~10 req/min average (well under limit)
 */

import {
  getMarketData,
  getStockPrice,
} from './tradier';
import {
  getGreekFlow,
  getSpotGEX,
  getVolatilityStats,
  getInstitutionalActivity,
  getNewsHeadlines,
  getSectorTide,
  getPutCallRatio,
} from './unusualwhales';
import {
  getTechnicalIndicators,
} from './technicals';
import {
  setCache,
  STALENESS_CONFIG,
  expireOldEntries,
} from './dataCache';

const ENABLE_POLLING = process.env.ENABLE_POLLING === 'true';

interface PollTask {
  key: string;
  fetcher: () => Promise<any>;
  interval: number;
  lastRun: number;
}

/**
 * Fetch and cache technical indicators for key symbols
 */
async function getAllTechnicals(): Promise<Record<string, any>> {
  const symbols = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA'];
  const results: Record<string, any> = {};

  for (const symbol of symbols) {
    try {
      const indicators = await getTechnicalIndicators(symbol);
      if (indicators) {
        results[symbol] = indicators;
      }
    } catch (error) {
      console.error(`Failed to calculate technicals for ${symbol}:`, error);
    }
  }

  return results;
}

// Poll tasks with their configurations
const POLL_TASKS: PollTask[] = [
  {
    key: 'market_data',
    fetcher: () => getMarketData(),
    interval: STALENESS_CONFIG.market_data.pollInterval,
    lastRun: 0,
  },
  {
    key: 'put_call_ratio',
    fetcher: () => getPutCallRatio(),
    interval: STALENESS_CONFIG.put_call_ratio.pollInterval,
    lastRun: 0,
  },
  {
    key: 'spot_gex',
    fetcher: () => getSpotGEX(),
    interval: STALENESS_CONFIG.spot_gex.pollInterval,
    lastRun: 0,
  },
  {
    key: 'volatility_stats',
    fetcher: () => getVolatilityStats(),
    interval: STALENESS_CONFIG.volatility_stats.pollInterval,
    lastRun: 0,
  },
  {
    key: 'sector_tide',
    fetcher: () => getSectorTide(),
    interval: STALENESS_CONFIG.sector_tide.pollInterval,
    lastRun: 0,
  },
  {
    key: 'greek_flow',
    fetcher: () => getGreekFlow(),
    interval: STALENESS_CONFIG.greek_flow.pollInterval,
    lastRun: 0,
  },
  {
    key: 'news_headlines',
    fetcher: () => getNewsHeadlines(),
    interval: STALENESS_CONFIG.news_headlines.pollInterval,
    lastRun: 0,
  },
  {
    key: 'technicals',
    fetcher: () => getAllTechnicals(),
    interval: STALENESS_CONFIG.technicals.pollInterval,
    lastRun: 0,
  },
  {
    key: 'institutional',
    fetcher: () => getInstitutionalActivity(),
    interval: STALENESS_CONFIG.institutional.pollInterval,
    lastRun: 0,
  },
];

let isRunning = false;
let pollInterval: NodeJS.Timeout | null = null;

/**
 * Execute a single poll task
 */
async function executePollTask(task: PollTask): Promise<void> {
  const now = Date.now();

  // Skip if not yet time to run
  if (now - task.lastRun < task.interval) {
    return;
  }

  try {
    console.log(`🔄 Polling ${task.key}...`);
    const data = await task.fetcher();
    setCache(task.key, data, `poll:${task.key}`);
    task.lastRun = now;
    console.log(`✅ Updated ${task.key} cache`);
  } catch (error: any) {
    console.error(`❌ Failed to poll ${task.key}:`, error.message);
  }
}

/**
 * Main polling loop
 */
async function pollLoop(): Promise<void> {
  if (!isRunning) return;

  // Execute all tasks that are due
  for (const task of POLL_TASKS) {
    if (!isRunning) break;
    await executePollTask(task);

    // Small delay between tasks to avoid bursts
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  // Expire old cache entries every hour
  if (Date.now() % 3_600_000 < 60_000) {
    expireOldEntries();
  }
}

/**
 * Start the polling service
 */
export function startPollingService(intervalMs: number = 10_000): void {
  if (!ENABLE_POLLING) {
    console.log('ℹ️  ENABLE_POLLING not set; skipping background polling service');
    return;
  }
  if (isRunning) {
    console.log('⚠️  Polling service already running');
    return;
  }

  isRunning = true;
  console.log('🚀 Starting background polling service...');
  console.log(`📊 Polling ${POLL_TASKS.length} endpoints (including ${POLL_TASKS.find(t => t.key === 'technicals') ? 'pre-calculated technicals' : 'market data'})`);

  // Run immediately on start
  pollLoop().catch(console.error);

  // Then run every intervalMs
  pollInterval = setInterval(() => {
    pollLoop().catch(console.error);
  }, intervalMs);

  console.log(`✅ Polling service started (checking every ${intervalMs / 1000}s)`);
}

/**
 * Stop the polling service
 */
export function stopPollingService(): void {
  if (!isRunning) {
    console.log('⚠️  Polling service not running');
    return;
  }

  isRunning = false;
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }

  console.log('🛑 Polling service stopped');
}

/**
 * Get polling service status
 */
export function getPollingStatus(): {
  isRunning: boolean;
  tasks: Array<{
    key: string;
    interval: number;
    lastRun: number;
    nextRun: number;
  }>;
} {
  return {
    isRunning,
    tasks: POLL_TASKS.map(task => ({
      key: task.key,
      interval: task.interval,
      lastRun: task.lastRun,
      nextRun: task.lastRun + task.interval,
    })),
  };
}

// Auto-start in production (not during build/test)
if (process.env.NODE_ENV === 'production' || process.env.AUTO_START_POLLING === 'true') {
  startPollingService();
}
