import 'dotenv/config';
import { TwelveDataPriceFeed } from './lib/twelveData';
import { startSmaCrossoverStrategy, SMA_TWELVE_DATA_MAX_MINUTE_BARS } from './run-live-sma-crossover';
import { startMeanReversion5mStrategy } from './run-live-mean-reversion-5min';

const SMA_DEFAULT_SYMBOLS = 'QQQ';
const MR_DEFAULT_SYMBOLS = 'AAPL';
const STOCK_DEFAULT_QTY = '100';

function resolveList(value: string): string[] {
  return value
    .split(',')
    .map(s => s.trim().toUpperCase())
    .filter(Boolean);
}

function ensureEnvDefaults() {
  if (!process.env.SMA_SYMBOLS) {
    process.env.SMA_SYMBOLS = SMA_DEFAULT_SYMBOLS;
  }
  if (!process.env.SMA_TRADE_MODE) {
    process.env.SMA_TRADE_MODE = 'option';
  }
  if (!process.env.SMA_OPTION_CONTRACTS) {
    process.env.SMA_OPTION_CONTRACTS = '1';
  }
  process.env.SMA_USE_TWELVE_DATA = 'true';

  if (!process.env.MR5_SYMBOLS) {
    process.env.MR5_SYMBOLS = MR_DEFAULT_SYMBOLS;
  }
  process.env.MR5_TRADE_MODE = 'stock';
  if (!process.env.MR5_STOCK_SHARES) {
    process.env.MR5_STOCK_SHARES = STOCK_DEFAULT_QTY;
  }
  if (!process.env.MR5_MINUTE_BACKFILL) {
    process.env.MR5_MINUTE_BACKFILL = '600';
  }
}

ensureEnvDefaults();

const smaSymbols = resolveList(process.env.SMA_SYMBOLS || SMA_DEFAULT_SYMBOLS);
const mrSymbols = resolveList(process.env.MR5_SYMBOLS || MR_DEFAULT_SYMBOLS);
const sharedSymbols = Array.from(new Set([...smaSymbols, ...mrSymbols]));

const minuteBackfill = Number(process.env.MR5_MINUTE_BACKFILL || '600');
const maxMinuteBars = Math.max(SMA_TWELVE_DATA_MAX_MINUTE_BARS, minuteBackfill);

if (!process.env.TWELVE_DATA_API_KEY) {
  throw new Error('TWELVE_DATA_API_KEY is required for the shared feed');
}

const sharedFeed = new TwelveDataPriceFeed({
  apiKey: process.env.TWELVE_DATA_API_KEY,
  backupApiKey: process.env.TWELVE_DATA_BACKUP_API_KEY || undefined,
  symbols: sharedSymbols,
  url: process.env.TWELVE_DATA_WS_URL || undefined,
  maxMinuteBars,
});

async function main() {
  console.log('Starting dual live runner:');
  console.log(`  • SMA crossover (options) on: ${smaSymbols.join(', ') || 'n/a'}`);
  console.log(`  • Mean reversion 5m (stocks) on: ${mrSymbols.join(', ') || 'n/a'}`);
  console.log(`Shared Twelve Data feed symbols: ${sharedSymbols.join(', ')}`);

  await sharedFeed.bootstrap();
  sharedFeed.start();

  const smaRunner = startSmaCrossoverStrategy({
    feed: sharedFeed,
    manageProcessSignals: false,
  });
  const meanRunner = startMeanReversion5mStrategy({
    feed: sharedFeed,
    manageProcessSignals: false,
  });

  const shutdownAll = async (reason: 'SIGINT' | 'SIGTERM' | 'external') => {
    console.log(`[dual-runner] Received ${reason}, shutting down strategies...`);
    await Promise.allSettled([
      smaRunner.shutdown(reason),
      meanRunner.shutdown(reason),
    ]);
    try {
      sharedFeed.stop();
    } catch {
      /* noop */
    }
    if (reason !== 'external') {
      process.exit(0);
    }
  };

  process.once('SIGINT', () => {
    shutdownAll('SIGINT').catch(err => {
      console.error('Failed to shutdown cleanly:', err);
      process.exit(1);
    });
  });
  process.once('SIGTERM', () => {
    shutdownAll('SIGTERM').catch(err => {
      console.error('Failed to shutdown cleanly:', err);
      process.exit(1);
    });
  });

  try {
    await Promise.all([smaRunner.task, meanRunner.task]);
  } catch (err) {
    console.error('[dual-runner] underlying task crashed:', err);
    await shutdownAll('external');
    process.exit(1);
  }
}

main().catch(err => {
  console.error('[dual-runner] Fatal error:', err);
  try {
    sharedFeed.stop();
  } catch {
    /* noop */
  }
  process.exit(1);
});
