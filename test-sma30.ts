#!/usr/bin/env tsx
import { runRealtimeSMABacktest } from './backtest-sma50-1min-realtime';

async function main() {
  const START = '2025-10-01T00:00:00Z';
  const END = '2025-10-31T23:59:59Z';

  console.log('\nTESTING: MICRO NASDAQ (MNQ) - SMA(30)\n');
  await runRealtimeSMABacktest('CON.F.US.MNQ.Z25', START, END, 30, 14, 0.001, 0.011, 0.37);

  console.log('\n\nTESTING: MICRO E-MINI S&P (MES) - SMA(30)\n');
  await runRealtimeSMABacktest('CON.F.US.MES.Z25', START, END, 30, 14, 0.001, 0.011, 0.37);
}

main().catch(err => {
  console.error('\n✗ Backtest failed:', err.message);
  process.exit(1);
});
