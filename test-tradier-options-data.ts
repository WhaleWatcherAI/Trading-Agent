import 'dotenv/config';
import {
  fetchTradierOptionExpirations,
  fetchTradierOptionChain,
} from './lib/tradier';
import {
  getCachedExpirations,
  getCachedOptionChain,
  setCachedExpirations,
  setCachedOptionChain,
  isCacheEntryFresh,
  DEFAULT_CACHE_MAX_AGE_MS,
} from './lib/optionCache';

const OPTION_EXPIRATION_EOD_SUFFIX = 'T21:00:00Z';

function parseSymbols(): string[] {
  const cliSymbols = process.argv
    .slice(2)
    .filter(arg => !arg.startsWith('--'))
    .map(s => s.trim().toUpperCase())
    .filter(Boolean);
  if (cliSymbols.length > 0) {
    return cliSymbols;
  }
  return (process.env.MR_SYMBOLS || 'SPY')
    .split(',')
    .map(s => s.trim().toUpperCase())
    .filter(Boolean);
}

function nextViableExpiration(expirations: string[]): string | null {
  const now = Date.now();
  const entries = expirations
    .map(dateStr => {
      const epoch = new Date(`${dateStr}${OPTION_EXPIRATION_EOD_SUFFIX}`).getTime();
      return { dateStr, epoch };
    })
    .filter(entry => Number.isFinite(entry.epoch) && entry.epoch > now)
    .sort((a, b) => a.epoch - b.epoch);

  return entries[0]?.dateStr ?? null;
}

async function testSymbol(symbol: string): Promise<void> {
  console.log(`\n🧪 ${symbol}`);

  const cachedExp = await getCachedExpirations(symbol);
  if (cachedExp) {
    const freshness = isCacheEntryFresh(cachedExp.updatedAt, DEFAULT_CACHE_MAX_AGE_MS)
      ? 'fresh'
      : 'stale';
    console.log(
      `   Cached expirations (${freshness}, ${cachedExp.expirations.length} entries) updated at ${cachedExp.updatedAt}`,
    );
  }

  let expirations: string[] = [];
  try {
    expirations = await fetchTradierOptionExpirations(symbol);
    console.log(`   Tradier expirations: ${expirations.slice(0, 5).join(', ') || 'none returned'}`);
    if (expirations.length > 0) {
      await setCachedExpirations(symbol, expirations);
    }
  } catch (err) {
    console.error(`   ⚠️  Tradier expirations error: ${(err as Error).message}`);
  }

  if (expirations.length === 0) {
    expirations = cachedExp?.expirations ?? [];
    if (expirations.length === 0) {
      console.log('   No expirations available from Tradier or cache');
      return;
    }
    console.log('   Using cached expirations because live Tradier data was unavailable');
  }

  const targetExpiration = nextViableExpiration(expirations) ?? expirations[0];
  if (!targetExpiration) {
    console.log('   No future expirations to test');
    return;
  }

  const cachedChain = await getCachedOptionChain(symbol, targetExpiration);
  if (cachedChain) {
    const freshness = isCacheEntryFresh(cachedChain.updatedAt, DEFAULT_CACHE_MAX_AGE_MS)
      ? 'fresh'
      : 'stale';
    console.log(
      `   Cached chain for ${targetExpiration}: ${cachedChain.contracts.length} contracts (${freshness})`,
    );
  }

  try {
    const chain = await fetchTradierOptionChain(symbol, targetExpiration);
    console.log(
      `   Tradier chain for ${targetExpiration}: ${chain.length} contracts`,
    );
    if (chain.length > 0) {
      await setCachedOptionChain(symbol, targetExpiration, chain);
      const previews = chain.slice(0, 3).map(contract => `${contract.symbol} ${contract.option_type.toUpperCase()} ${contract.strike_price}`);
      console.log(`   Sample contracts: ${previews.join(' | ')}`);
    }
  } catch (err) {
    if (cachedChain) {
      console.error(
        `   ⚠️  Tradier chain fetch failed, using cached data (${(err as Error).message})`,
      );
    } else {
      console.error(
        `   ⚠️  Tradier chain error for ${targetExpiration}: ${(err as Error).message}`,
      );
    }
  }
}

async function main() {
  const symbols = parseSymbols();
  if (symbols.length === 0) {
    console.error('No symbols provided via CLI or MR_SYMBOLS');
    process.exit(1);
  }

  for (const symbol of symbols) {
    await testSymbol(symbol);
  }

  console.log('\n✅ Tradier options data check complete');
}

main().catch(err => {
  console.error('Unexpected error while testing Tradier options data:', err);
  process.exit(1);
});
