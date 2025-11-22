#!/usr/bin/env tsx
/**
 * This script applies the native bracket order refactoring to MES, MGC, and M6E strategies
 */

import { readFileSync, writeFileSync } from 'fs';

const files = [
  '/Users/coreycosta/trading-agent/live-topstepx-mes-winner.ts',
  '/Users/coreycosta/trading-agent/live-topstepx-mgc-winner.ts',
  '/Users/coreycosta/trading-agent/live-topstepx-m6e-winner.ts',
];

const OLD_PLACE_BRACKET = `  async placeBracketEntry(
    side: OrderSide,
    stopPrice: number,
    targetPrice: number,
    qty: number,
  ) {
    log(\`[BRACKET] Entry \${side} MARKET, Stop @ \${stopPrice.toFixed(2)}, Target @ \${targetPrice.toFixed(2)}\`);

    const entryResponse = await this.placeMarketIOC(side, qty);
    const entryOrderId = this.resolveOrderId(entryResponse);

    log(\`[BRACKET] Entry market order placed: \${entryOrderId}\`);

    const stopSide: OrderSide = side === 'Buy' ? 'Sell' : 'Buy';
    const targetSide: OrderSide = side === 'Buy' ? 'Sell' : 'Buy';

    const [stopResponse, targetResponse] = await Promise.all([
      this.placeLimitIOC(stopSide, qty, stopPrice),
      this.placeLimitIOC(targetSide, qty, targetPrice),
    ]);

    const stopOrderId = this.resolveOrderId(stopResponse);
    const targetOrderId = this.resolveOrderId(targetResponse);

    log(\`[BRACKET] Stop order placed: \${stopOrderId}, Target order placed: \${targetOrderId}\`);

    return {
      entryOrderId,
      stopOrderId,
      targetOrderId,
      entryFilled: this.isFilledResponse(entryResponse, qty),
      stopFilled: this.isFilledResponse(stopResponse, qty),
      targetFilled: this.isFilledResponse(targetResponse, qty),
    };
  }

  private resolveOrderId(response: any): string | number {
    return response?.orderId ?? response?.id ?? \`topstep-\${Date.now()}\`;
  }`;

const NEW_PLACE_BRACKET = `  async placeBracketEntry(
    side: OrderSide,
    entryPrice: number,
    stopPrice: number,
    targetPrice: number,
    qty: number,
  ) {
    // Calculate stop/target distance in ticks
    const stopDistance = Math.abs(entryPrice - stopPrice);
    const targetDistance = Math.abs(entryPrice - targetPrice);
    const stopTicks = Math.round(stopDistance / this.tickSize);
    const targetTicks = Math.round(targetDistance / this.tickSize);

    log(\`[BRACKET] Entry \${side} MARKET with native brackets:\`);
    log(\`  Entry: \${entryPrice.toFixed(2)}\`);
    log(\`  Stop: \${stopPrice.toFixed(2)} (\${stopTicks} ticks, type 4=Stop)\`);
    log(\`  Target: \${targetPrice.toFixed(2)} (\${targetTicks} ticks, type 1=Limit)\`);

    const payload = {
      accountId: this.accountId,
      contractId: this.contractId,
      side: side === 'Buy' ? 0 : 1,
      size: qty,
      type: 2, // Market entry
      timeInForce: 0, // IOC
      stopLossBracket: {
        ticks: stopTicks,
        type: 4 as const, // Stop order (converts to market when hit)
      },
      takeProfitBracket: {
        ticks: targetTicks,
        type: 1 as const, // Limit order
      },
    };

    log(\`[ORDER] Placing native bracket order: \${JSON.stringify(payload, null, 2)}\`);
    const response = await this.rest.placeOrder(payload);
    const entryOrderId = this.resolveOrderId(response);

    log(\`[BRACKET] Native bracket placed successfully, entry order ID: \${entryOrderId}\`);
    log(\`[BRACKET] Broker will automatically manage OCO for stop/target legs\`);

    return {
      entryOrderId,
      stopOrderId: undefined, // Broker manages these internally
      targetOrderId: undefined, // Broker manages these internally
      entryFilled: this.isFilledResponse(response, qty),
      stopFilled: false,
      targetFilled: false,
    };
  }

  private resolveOrderId(response: any): number {
    const id = response?.orderId ?? response?.id;
    if (typeof id === 'number') return id;
    if (typeof id === 'string') {
      const parsed = parseInt(id, 10);
      if (!isNaN(parsed)) return parsed;
    }
    throw new Error(\`Invalid order ID received: \${JSON.stringify(id)}\`);
  }`;

console.log('Applying native bracket refactoring to 3 strategy files...\n');

for (const file of files) {
  console.log(`Processing: ${file}`);
  const content = readFileSync(file, 'utf8');

  if (!content.includes(OLD_PLACE_BRACKET)) {
    console.log(`  ⚠️  Old pattern not found, skipping`);
    continue;
  }

  const updated = content.replace(OLD_PLACE_BRACKET, NEW_PLACE_BRACKET);
  writeFileSync(file, updated, 'utf8');
  console.log(`  ✅ Updated placeBracketEntry method`);
}

console.log('\n✅ Done! Now manually update the enterPosition calls and remove OCO logic.');
