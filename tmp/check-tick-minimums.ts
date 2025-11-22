#!/usr/bin/env tsx

const configs = {
  NQ: { symbol: 'NQZ5', price: 20000, tickSize: 0.25, stopPct: 0.0001, targetPct: 0.0005 },
  MES: { symbol: 'MESZ5', price: 5800, tickSize: 0.25, stopPct: 0.0001, targetPct: 0.0005 },
  MGC: { symbol: 'MGCZ5', price: 2600, tickSize: 0.1, stopPct: 0.0001, targetPct: 0.0005 },
  M6E: { symbol: 'M6EZ5', price: 1.05, tickSize: 0.00001, stopPct: 0.0001, targetPct: 0.0005 },
};

console.log('Checking if current configs meet 4-tick minimum:\n');

for (const [name, cfg] of Object.entries(configs)) {
  const stopDist = cfg.price * cfg.stopPct;
  const targetDist = cfg.price * cfg.targetPct;
  const stopTicks = Math.round(stopDist / cfg.tickSize);
  const targetTicks = Math.round(targetDist / cfg.tickSize);

  console.log(`${name} (${cfg.symbol}):`);
  console.log(`  Price: ${cfg.price.toFixed(2)}, Tick: ${cfg.tickSize}`);
  console.log(`  Stop: ${cfg.stopPct * 100}% = $${stopDist.toFixed(2)} = ${stopTicks} ticks ${stopTicks >= 4 ? '✅' : '❌ TOO SMALL'}`);
  console.log(`  Target: ${cfg.targetPct * 100}% = $${targetDist.toFixed(2)} = ${targetTicks} ticks ${targetTicks >= 4 ? '✅' : '❌ TOO SMALL'}`);
  console.log('');
}
