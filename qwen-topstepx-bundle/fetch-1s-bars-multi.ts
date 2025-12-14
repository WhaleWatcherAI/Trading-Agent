#!/usr/bin/env npx tsx
/**
 * Fetch 1-second bars from TopstepX with multiple API calls
 * Patches together results to get more historical data
 *
 * TopstepX limit is 20,000 bars per call
 * 1 day of 1-second bars (trading hours ~6.5h) = ~23,400 bars
 * So we fetch in 4-hour chunks to stay under limit
 */

import 'dotenv/config';
import { fetchTopstepXFuturesBars, fetchTopstepXFuturesMetadata } from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';

const SYMBOL = process.env.TOPSTEPX_SYMBOL || 'MESZ5';
const CONTRACT_ID = process.env.TOPSTEPX_CONTRACT_ID;
const DAYS_BACK = parseInt(process.env.DAYS_BACK || '14', 10);
const CHUNK_HOURS = 4; // Fetch in 4-hour chunks (4 * 3600 = 14,400 bars max)

interface Bar {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  console.log(`Fetching 1-second bars for ${SYMBOL} (${DAYS_BACK} days)...`);

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

  // Fetch in chunks going backwards from end time
  const allBars: Bar[] = [];
  let currentEnd = new Date(endTime);
  let chunkCount = 0;
  const chunkMs = CHUNK_HOURS * 60 * 60 * 1000;

  while (currentEnd > startTime) {
    const currentStart = new Date(Math.max(currentEnd.getTime() - chunkMs, startTime.getTime()));

    chunkCount++;
    console.log(`\nChunk ${chunkCount}: ${currentStart.toISOString()} to ${currentEnd.toISOString()}`);

    try {
      const bars = await fetchTopstepXFuturesBars({
        contractId,
        startTime: currentStart.toISOString(),
        endTime: currentEnd.toISOString(),
        unit: 1,        // 1 = Second
        unitNumber: 1,  // 1-second bars
        limit: 20000,
        live: false,
      });

      if (bars && bars.length > 0) {
        const formattedBars = bars.map(b => ({
          t: b.timestamp,
          o: b.open,
          h: b.high,
          l: b.low,
          c: b.close,
          v: b.volume || 0,
        }));

        allBars.push(...formattedBars);
        console.log(`  Fetched ${bars.length} bars (total: ${allBars.length})`);
      } else {
        console.log(`  No bars returned for this chunk`);
      }
    } catch (err: any) {
      console.error(`  Error fetching chunk: ${err.message}`);
    }

    // Move to next chunk
    currentEnd = new Date(currentStart.getTime() - 1000); // 1 second before previous chunk start

    // Rate limit - wait between requests
    await sleep(500);
  }

  console.log(`\nTotal bars fetched: ${allBars.length}`);

  if (allBars.length === 0) {
    console.error('No bars returned!');
    process.exit(1);
  }

  // Sort by timestamp (oldest first)
  allBars.sort((a, b) => new Date(a.t).getTime() - new Date(b.t).getTime());

  // Remove duplicates (in case of overlap)
  const seenTimestamps = new Set<string>();
  const uniqueBars = allBars.filter(bar => {
    if (seenTimestamps.has(bar.t)) {
      return false;
    }
    seenTimestamps.add(bar.t);
    return true;
  });

  console.log(`Unique bars after dedup: ${uniqueBars.length}`);

  // Save as JSON for Python to read
  const outputPath = path.join(__dirname, 'ml', 'data', 'bars_1s.json');

  const output = {
    symbol: SYMBOL,
    contractId,
    unit: 'second',
    unitNumber: 1,
    startTime: startTime.toISOString(),
    endTime: endTime.toISOString(),
    barCount: uniqueBars.length,
    bars: uniqueBars,
  };

  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\nSaved to ${outputPath}`);

  // Print sample
  console.log('\nSample bars (first 5):');
  uniqueBars.slice(0, 5).forEach(b => {
    console.log(`  ${b.t}: O=${b.o} H=${b.h} L=${b.l} C=${b.c} V=${b.v}`);
  });

  console.log('\nSample bars (last 5):');
  uniqueBars.slice(-5).forEach(b => {
    console.log(`  ${b.t}: O=${b.o} H=${b.h} L=${b.l} C=${b.c} V=${b.v}`);
  });

  // Calculate expected 1-minute bars
  const expectedMinBars = Math.floor(uniqueBars.length / 60);
  console.log(`\nExpected ~${expectedMinBars} 1-minute bars after aggregation`);
  console.log('\nNow run:');
  console.log('  cd ml && python3 scripts/lstm_vp_cvd_backtest.py --input ../data/bars_1s.json --output ../data/lstm_vp_cvd_results.json');
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
