#!/usr/bin/env npx tsx
import 'dotenv/config';
import { fetchTopstepXFuturesBars, fetchTopstepXFuturesMetadata } from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';

const SYMBOLS = [
  { symbol: 'NQZ5', output: 'bars_1s_nq_today.json' },
  { symbol: 'ESZ5', output: 'bars_1s_es_today.json' },
  { symbol: 'GCG6', output: 'bars_1s_gc_today.json' },
];

const CHUNK_HOURS = 4;

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchSymbol(symbol: string, outputFile: string) {
  console.log(`\nFetching ${symbol}...`);

  const metadata = await fetchTopstepXFuturesMetadata(symbol);
  if (!metadata) throw new Error(`Could not find contract for ${symbol}`);
  const contractId = metadata.id;
  console.log(`Contract: ${contractId}`);

  const endTime = new Date();
  const startTime = new Date(endTime.getTime() - 24 * 60 * 60 * 1000);

  const allBars: any[] = [];
  let currentEnd = new Date(endTime);
  const chunkMs = CHUNK_HOURS * 60 * 60 * 1000;

  while (currentEnd > startTime) {
    const currentStart = new Date(Math.max(currentEnd.getTime() - chunkMs, startTime.getTime()));

    const bars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: currentStart.toISOString(),
      endTime: currentEnd.toISOString(),
      unit: 1,
      unitNumber: 1,
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
      console.log(`  Chunk: ${bars.length} bars (total: ${allBars.length})`);
    }

    currentEnd = new Date(currentStart.getTime() - 1000);
    await sleep(300);
  }

  allBars.sort((a, b) => new Date(a.t).getTime() - new Date(b.t).getTime());
  const seenTimestamps = new Set<string>();
  const uniqueBars = allBars.filter(bar => {
    if (seenTimestamps.has(bar.t)) return false;
    seenTimestamps.add(bar.t);
    return true;
  });

  const output = {
    symbol,
    contractId,
    unit: 1,
    unitNumber: 1,
    startTime: startTime.toISOString(),
    endTime: endTime.toISOString(),
    barCount: uniqueBars.length,
    bars: uniqueBars,
  };

  const outputPath = path.join(__dirname, 'ml', 'data', outputFile);
  fs.writeFileSync(outputPath, JSON.stringify(output));
  console.log(`Saved ${uniqueBars.length} bars to ${outputFile}`);
  return uniqueBars.length;
}

async function main() {
  for (const { symbol, output } of SYMBOLS) {
    await fetchSymbol(symbol, output);
  }
  console.log('\nDone!');
}

main().catch(console.error);
