#!/usr/bin/env tsx
/**
 * Close stuck MGC and MES positions from the bug where fills weren't processed
 */

import { createProjectXRest } from './projectx-rest';

async function closeStuckPositions() {
  const rest = createProjectXRest();
  const accountId = 13056702;

  console.log('Checking for stuck positions...\n');

  try {
    const positions = await rest.getPositions(accountId);
    console.log(`Found ${positions.length} open positions\n`);

    for (const pos of positions) {
      const symbol = pos.contractId.replace('CON.F.US.', '').replace('.Z25', 'Z5');
      console.log(`Position: ${symbol} | Side: ${pos.netQty > 0 ? 'LONG' : 'SHORT'} | Qty: ${Math.abs(pos.netQty)} | PnL: $${pos.unrealizedPnL?.toFixed(2) || '0.00'}`);

      // Close MGC and MES positions
      if (pos.contractId.includes('MGC') || pos.contractId.includes('MES')) {
        console.log(`  -> Closing ${symbol} position...`);

        const side = pos.netQty > 0 ? 1 : 0; // 1 = Sell (close long), 0 = Buy (close short)
        const response = await rest.placeOrder({
          accountId,
          contractId: pos.contractId,
          side,
          size: Math.abs(pos.netQty),
          type: 2, // Market
          timeInForce: 0, // IOC
        });

        if (response.success) {
          console.log(`  ✅ ${symbol} closed successfully! Order ID: ${response.orderId}\n`);
        } else {
          console.log(`  ❌ Failed to close ${symbol}: ${response.errorMessage}\n`);
        }
      }
    }

    console.log('Done!');

  } catch (err: any) {
    console.error('Error:', err.message);
  }
}

closeStuckPositions().catch(console.error);
