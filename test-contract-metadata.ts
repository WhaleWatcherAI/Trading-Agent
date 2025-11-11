#!/usr/bin/env tsx
import { fetchTopstepXFuturesMetadata } from './lib/topstepx';

async function testMetadata() {
  const symbols = ['ESZ5', 'NQZ5', 'MESZ5', 'GCZ5', '6EZ5'];

  console.log('Fetching TopstepX contract metadata...\n');

  for (const symbol of symbols) {
    const metadata = await fetchTopstepXFuturesMetadata(symbol);

    if (metadata) {
      console.log(`${symbol} (${metadata.name}):`);
      console.log(`  Contract ID: ${metadata.id}`);
      console.log(`  Tick Size: ${metadata.tickSize}`);
      console.log(`  Tick Value: ${metadata.tickValue}`);
      console.log(`  Multiplier: ${metadata.multiplier}`);

      const calculatedMultiplier = metadata.tickValue && metadata.tickSize
        ? metadata.tickValue / metadata.tickSize
        : metadata.multiplier || 5;

      console.log(`  Calculated Point Multiplier: $${calculatedMultiplier}/point`);
      console.log('');
    } else {
      console.log(`${symbol}: NOT FOUND\n`);
    }
  }
}

testMetadata().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
