#!/usr/bin/env npx tsx
/**
 * Fetch 1-second bars from TopstepX across multiple contracts
 * Handles quarterly rollovers to get continuous front-month data
 *
 * MNQ/NQ/MES Contract Schedule:
 * - H = March (expires 3rd Friday of March)
 * - M = June (expires 3rd Friday of June)
 * - U = September (expires 3rd Friday of September)
 * - Z = December (expires 3rd Friday of December)
 *
 * Rollover typically happens ~1 week before expiry when volume shifts
 */

import 'dotenv/config';
import { fetchTopstepXFuturesBars, fetchTopstepXFuturesMetadata } from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';

// Parse command line arguments
const args = process.argv.slice(2);
let BASE_SYMBOL = 'MNQ';  // Default to MNQ
let START_DATE = '2024-12-01';
let END_DATE = '2025-12-07';
let OUTPUT_SUFFIX = '';

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--symbol' && args[i + 1]) {
    BASE_SYMBOL = args[i + 1].toUpperCase();
    i++;
  } else if (args[i] === '--start' && args[i + 1]) {
    START_DATE = args[i + 1];
    i++;
  } else if (args[i] === '--end' && args[i + 1]) {
    END_DATE = args[i + 1];
    i++;
  } else if (args[i] === '--output-suffix' && args[i + 1]) {
    OUTPUT_SUFFIX = args[i + 1];
    i++;
  }
}

const CHUNK_HOURS = 4; // Fetch in 4-hour chunks (4 * 3600 = 14,400 bars max)

interface Bar {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
  contract?: string;  // Track which contract bar came from
}

interface ContractPeriod {
  symbol: string;      // e.g., MNQZ4
  contractId: string;  // e.g., CON.F.US.MNQ.Z24
  startDate: Date;
  endDate: Date;
}

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Get the contract ID format for a given symbol
 * e.g., MNQZ5 -> CON.F.US.MNQ.Z25
 */
function symbolToContractId(symbol: string): string {
  // Extract base (MNQ, NQ, MES, etc) and month+year code (Z5, H5, etc)
  const match = symbol.match(/^([A-Z]+)([HMUZ])(\d+)$/);
  if (!match) {
    throw new Error(`Invalid symbol format: ${symbol}`);
  }

  const [, base, month, yearDigit] = match;
  const year = yearDigit.length === 1 ? `2${yearDigit}` : yearDigit;

  return `CON.F.US.${base}.${month}${year}`;
}

/**
 * Get 3rd Friday of a month (CME futures expiration)
 */
function getThirdFriday(year: number, month: number): Date {
  const date = new Date(year, month, 1);
  // Find first Friday
  while (date.getDay() !== 5) {
    date.setDate(date.getDate() + 1);
  }
  // Add 2 weeks to get 3rd Friday
  date.setDate(date.getDate() + 14);
  return date;
}

/**
 * Get rollover date (about 1 week before expiry when volume typically shifts)
 */
function getRolloverDate(year: number, month: number): Date {
  const expiryDate = getThirdFriday(year, month);
  // Rollover is ~7-10 days before expiry
  const rolloverDate = new Date(expiryDate);
  rolloverDate.setDate(rolloverDate.getDate() - 7);
  return rolloverDate;
}

/**
 * Generate contract periods for a date range
 * Maps each date range to the appropriate front-month contract
 */
function generateContractPeriods(startDate: Date, endDate: Date, baseSymbol: string): ContractPeriod[] {
  const periods: ContractPeriod[] = [];

  // Contract months: H=Mar(2), M=Jun(5), U=Sep(8), Z=Dec(11)
  const contractMonths = [
    { code: 'H', expiryMonth: 2 },   // March
    { code: 'M', expiryMonth: 5 },   // June
    { code: 'U', expiryMonth: 8 },   // September
    { code: 'Z', expiryMonth: 11 },  // December
  ];

  let currentDate = new Date(startDate);

  while (currentDate < endDate) {
    // Find the next contract to expire
    let year = currentDate.getFullYear();
    let contractInfo = null;
    let rollover: Date | null = null;

    // Check current year and next year for the nearest contract
    for (let y = year; y <= year + 1 && !contractInfo; y++) {
      for (const cm of contractMonths) {
        const potentialRollover = getRolloverDate(y, cm.expiryMonth);

        // This contract is active if we're before its rollover date
        if (currentDate < potentialRollover) {
          contractInfo = cm;
          rollover = potentialRollover;
          year = y;
          break;
        }
      }
    }

    if (!contractInfo || !rollover) {
      throw new Error(`Could not determine contract for date: ${currentDate.toISOString()}`);
    }

    // Year code: 2024 -> 4, 2025 -> 5
    const yearCode = year % 10;
    const symbol = `${baseSymbol}${contractInfo.code}${yearCode}`;
    const contractId = symbolToContractId(symbol);

    // Period ends at rollover or end date, whichever comes first
    const periodEnd = new Date(Math.min(rollover.getTime(), endDate.getTime()));

    periods.push({
      symbol,
      contractId,
      startDate: new Date(currentDate),
      endDate: periodEnd,
    });

    // Move to next period (day after rollover)
    currentDate = new Date(rollover);
    currentDate.setDate(currentDate.getDate() + 1);
  }

  return periods;
}

async function fetchContractBars(
  contractId: string,
  startTime: Date,
  endTime: Date,
  contractSymbol: string
): Promise<Bar[]> {
  const allBars: Bar[] = [];
  let currentEnd = new Date(endTime);
  const chunkMs = CHUNK_HOURS * 60 * 60 * 1000;
  let chunkCount = 0;

  while (currentEnd > startTime) {
    const currentStart = new Date(Math.max(currentEnd.getTime() - chunkMs, startTime.getTime()));

    chunkCount++;
    console.log(`  Chunk ${chunkCount}: ${currentStart.toISOString()} to ${currentEnd.toISOString()}`);

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
          contract: contractSymbol,
        }));

        allBars.push(...formattedBars);
        console.log(`    Fetched ${bars.length} bars (chunk total: ${allBars.length})`);
      } else {
        console.log(`    No bars returned for this chunk`);
      }
    } catch (err: any) {
      console.error(`    Error fetching chunk: ${err.message}`);
    }

    // Move to next chunk
    currentEnd = new Date(currentStart.getTime() - 1000); // 1 second before previous chunk start

    // Rate limit - wait between requests
    await sleep(500);
  }

  return allBars;
}

async function main() {
  const startDate = new Date(START_DATE);
  const endDate = new Date(END_DATE);

  console.log(`\n=== Multi-Contract ${BASE_SYMBOL} 1-Second Bar Fetcher ===`);
  console.log(`Date range: ${START_DATE} to ${END_DATE}`);
  console.log(`Base symbol: ${BASE_SYMBOL}\n`);

  // Generate contract periods
  const periods = generateContractPeriods(startDate, endDate, BASE_SYMBOL);

  console.log('Contract periods to fetch:');
  for (const p of periods) {
    console.log(`  ${p.symbol} (${p.contractId}): ${p.startDate.toISOString().split('T')[0]} to ${p.endDate.toISOString().split('T')[0]}`);
  }
  console.log('');

  // Verify each contract exists
  console.log('Verifying contracts...');
  for (const p of periods) {
    const metadata = await fetchTopstepXFuturesMetadata(p.contractId);
    if (!metadata) {
      console.warn(`  ⚠ Contract ${p.contractId} not found - will try symbol lookup`);
      const metadataBySymbol = await fetchTopstepXFuturesMetadata(p.symbol);
      if (metadataBySymbol) {
        p.contractId = metadataBySymbol.id;
        console.log(`  ✓ Found via symbol: ${p.symbol} -> ${p.contractId}`);
      } else {
        console.error(`  ✗ Could not find contract for ${p.symbol}`);
      }
    } else {
      console.log(`  ✓ ${p.symbol} -> ${p.contractId}`);
    }
    await sleep(200);
  }
  console.log('');

  // Fetch bars from each contract
  const allBars: Bar[] = [];

  for (let i = 0; i < periods.length; i++) {
    const p = periods[i];
    console.log(`\n[${i + 1}/${periods.length}] Fetching ${p.symbol} (${p.startDate.toISOString().split('T')[0]} to ${p.endDate.toISOString().split('T')[0]})...`);

    if (!p.contractId) {
      console.log(`  Skipping - no valid contract ID`);
      continue;
    }

    const bars = await fetchContractBars(p.contractId, p.startDate, p.endDate, p.symbol);
    allBars.push(...bars);
    console.log(`  Contract total: ${bars.length} bars (cumulative: ${allBars.length})`);
  }

  console.log(`\n=== Fetch Complete ===`);
  console.log(`Total bars fetched: ${allBars.length}`);

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

  // Count bars per contract
  const contractCounts: Record<string, number> = {};
  for (const bar of uniqueBars) {
    const c = bar.contract || 'unknown';
    contractCounts[c] = (contractCounts[c] || 0) + 1;
  }
  console.log('\nBars per contract:');
  for (const [contract, count] of Object.entries(contractCounts)) {
    console.log(`  ${contract}: ${count.toLocaleString()} bars`);
  }

  // Save as JSON for Python to read
  const outputFilename = OUTPUT_SUFFIX
    ? `bars_1s_${BASE_SYMBOL.toLowerCase()}_${OUTPUT_SUFFIX}.json`
    : `bars_1s_${BASE_SYMBOL.toLowerCase()}_1yr.json`;
  const outputPath = path.join(__dirname, 'ml', 'data', outputFilename);

  const output = {
    symbol: BASE_SYMBOL,
    contracts: periods.map(p => p.symbol),
    unit: 'second',
    unitNumber: 1,
    startTime: startDate.toISOString(),
    endTime: endDate.toISOString(),
    barCount: uniqueBars.length,
    bars: uniqueBars,
  };

  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\nSaved to ${outputPath}`);

  // Print sample
  console.log('\nSample bars (first 5):');
  uniqueBars.slice(0, 5).forEach(b => {
    console.log(`  ${b.t} [${b.contract}]: O=${b.o} H=${b.h} L=${b.l} C=${b.c} V=${b.v}`);
  });

  console.log('\nSample bars (last 5):');
  uniqueBars.slice(-5).forEach(b => {
    console.log(`  ${b.t} [${b.contract}]: O=${b.o} H=${b.h} L=${b.l} C=${b.c} V=${b.v}`);
  });

  // Show rollover points
  console.log('\nContract rollovers in data:');
  let prevContract = uniqueBars[0]?.contract;
  for (const bar of uniqueBars) {
    if (bar.contract !== prevContract) {
      console.log(`  ${prevContract} -> ${bar.contract} at ${bar.t}`);
      prevContract = bar.contract;
    }
  }

  // Calculate expected 1-minute bars
  const expectedMinBars = Math.floor(uniqueBars.length / 60);
  console.log(`\nExpected ~${expectedMinBars.toLocaleString()} 1-minute bars after aggregation`);
  console.log('\nNow run:');
  console.log(`  cd ml && python3 scripts/no_whale_regime_backtest.py --bars data/${outputFilename} --output data/no_whale_regime_results_${BASE_SYMBOL.toLowerCase()}_1yr.json --point-value 2`);
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
