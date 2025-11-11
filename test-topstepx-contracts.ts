import dotenv from 'dotenv';
import { fetchTopstepXContracts, fetchTopstepXContract } from './lib/topstepx';

dotenv.config();

async function testContracts() {
  const symbol = process.argv[2];
  if (!symbol) {
    console.error('Please provide a symbol to search for.');
    process.exit(1);
  }

  console.log(`🔍 Fetching TopstepX Contracts for symbol: ${symbol}...\n`);

  try {
    // Test with demo/sim environment
    console.log('📋 Available Contracts (SIM):');
    const contracts = await fetchTopstepXContracts(false); // false = demo/sim

    console.log(`Found ${contracts.length} contracts:\n`);

    // Filter for E-mini contracts
    const symbolContracts = contracts.filter(c =>
      c.name?.includes(symbol.toUpperCase())
    );

    console.log(`${symbol.toUpperCase()} contracts:`);
    symbolContracts.forEach(contract => {
      console.log(`  - ${contract.id}`);
      console.log(`    Name: ${contract.name}`);
      console.log(`    Description: ${contract.description}`);
      console.log(`    Tick Size: ${contract.tickSize}`);
      console.log(`    Tick Value: $${contract.tickValue}`);
      console.log(`    Multiplier: ${contract.multiplier}`);
      console.log('');
    });

  } catch (error: any) {
    console.error('❌ Error:', error.message);
    if (error.response) {
      console.error('Response:', JSON.stringify(error.response.data, null, 2));
    }
  }
}

testContracts().catch(console.error);
