#!/usr/bin/env npx tsx
/**
 * Fetch 1-second bars from TopstepX and save to parquet for Chronos testing
 */

import 'dotenv/config';
import { fetchTopstepXFuturesBars, fetchTopstepXFuturesMetadata } from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';

const SYMBOL = process.env.TOPSTEPX_SYMBOL || 'MESZ5';
const CONTRACT_ID = process.env.TOPSTEPX_CONTRACT_ID;
const DAYS_BACK = parseInt(process.env.DAYS_BACK || '3', 10);

async function main() {
  console.log(`Fetching 1-second bars for ${SYMBOL}...`);

  // Get contract ID if not provided
  let contractId = CONTRACT_ID;
  if (!contractId) {
    console.log('Looking up contract ID...');
    const metadata = await fetchTopstepXFuturesMetadata(SYMBOL);
    if (!metadata) {
      throw new Error(`Could not find contract for ${SYMBOL}`);
    }
    contractId = metadata.id;
    console.log(`Found contract: ${contractId}`);
  }

  const endTime = new Date();
  const startTime = new Date(endTime.getTime() - DAYS_BACK * 24 * 60 * 60 * 1000);

  console.log(`Date range: ${startTime.toISOString()} to ${endTime.toISOString()}`);
  console.log(`Fetching ${DAYS_BACK} days of 1-second bars...`);

  const bars = await fetchTopstepXFuturesBars({
    contractId,
    startTime: startTime.toISOString(),
    endTime: endTime.toISOString(),
    unit: 1,        // 1 = Second
    unitNumber: 1,  // 1-second bars
    limit: 500000,  // Max bars
    live: false,
  });

  console.log(`Fetched ${bars.length} bars`);

  if (bars.length === 0) {
    console.error('No bars returned!');
    process.exit(1);
  }

  // Save as JSON for Python to read
  const outputPath = path.join(__dirname, 'ml', 'data', 'bars_1s.json');

  const output = {
    symbol: SYMBOL,
    contractId,
    unit: 'second',
    unitNumber: 1,
    startTime: startTime.toISOString(),
    endTime: endTime.toISOString(),
    barCount: bars.length,
    bars: bars.map(b => ({
      t: b.timestamp,
      o: b.open,
      h: b.high,
      l: b.low,
      c: b.close,
      v: b.volume || 0,
    })),
  };

  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`Saved to ${outputPath}`);

  // Print sample
  console.log('\nSample bars:');
  bars.slice(0, 5).forEach(b => {
    console.log(`  ${b.timestamp}: O=${b.open} H=${b.high} L=${b.low} C=${b.close} V=${b.volume}`);
  });

  console.log('\nNow run:');
  console.log('  cd ml && python3 scripts/chronos_backtest_1s.py');
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
