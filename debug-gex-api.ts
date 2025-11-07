import axios from 'axios';

const TRADIER_API_KEY = process.env.TRADIER_API_KEY || '';
const TRADIER_BASE_URL = process.env.TRADIER_BASE_URL || 'https://sandbox.tradier.com/v1';

const tradierClient = axios.create({
  baseURL: TRADIER_BASE_URL,
  headers: {
    Authorization: `Bearer ${TRADIER_API_KEY}`,
    Accept: 'application/json',
  },
});

async function testHistoricalOptionsChain() {
  console.log('🔍 Testing if Tradier sandbox returns different options data for different dates:\n');
  
  const symbol = 'SPY';
  const expiration = '2025-11-07';
  
  // Test 1: Current date (no date parameter)
  console.log('Test 1: Options chain WITHOUT date parameter (current)');
  const current = await tradierClient.get('/markets/options/chains', {
    params: { symbol, expiration, greeks: true },
  });
  const currentOptions = current.data?.options?.option || [];
  const currentSample = Array.isArray(currentOptions) ? currentOptions[0] : currentOptions;
  console.log(`  Sample contract: ${currentSample.symbol}`);
  console.log(`  Open Interest: ${currentSample.open_interest}`);
  console.log(`  Volume: ${currentSample.volume}`);
  console.log(`  Last: ${currentSample.last}`);
  console.log(`  Delta: ${currentSample.greeks?.delta}`);
  
  // Test 2: Historical date (Oct 21)
  console.log('\nTest 2: Options chain WITH date=2025-10-21');
  const historical = await tradierClient.get('/markets/options/chains', {
    params: { symbol, expiration, greeks: true, date: '2025-10-21' },
  });
  const historicalOptions = historical.data?.options?.option || [];
  const historicalSample = Array.isArray(historicalOptions) ? historicalOptions[0] : historicalOptions;
  console.log(`  Sample contract: ${historicalSample.symbol}`);
  console.log(`  Open Interest: ${historicalSample.open_interest}`);
  console.log(`  Volume: ${historicalSample.volume}`);
  console.log(`  Last: ${historicalSample.last}`);
  console.log(`  Delta: ${historicalSample.greeks?.delta}`);
  
  // Compare
  console.log('\n🔎 Comparison:');
  if (currentSample.open_interest === historicalSample.open_interest &&
      currentSample.volume === historicalSample.volume &&
      currentSample.last === historicalSample.last) {
    console.log('  ❌ IDENTICAL DATA - Sandbox is ignoring the date parameter!');
    console.log('  The sandbox does NOT support historical options chains.');
  } else {
    console.log('  ✅ Different data - Historical chains are working');
  }
}

testHistoricalOptionsChain().catch(console.error);
