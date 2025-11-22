#!/usr/bin/env tsx
/**
 * Test NQ native bracket order (8 ticks stop, 40 ticks target)
 */

import { createProjectXRest } from './projectx-rest';

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testNQBracket() {
  const rest = createProjectXRest();
  const accountId = 13056702;
  const contractId = 'CON.F.US.NQZ5';
  const tickSize = 0.25;

  // NQ config: 0.01% stop = 8 ticks, 0.05% target = 40 ticks
  const stopTicks = 8;
  const targetTicks = 40;

  console.log('=== NQ NATIVE BRACKET TEST ===\n');
  console.log(`Stop: ${stopTicks} ticks (${stopTicks * tickSize} points)`);
  console.log(`Target: ${targetTicks} ticks (${targetTicks * tickSize} points)\n`);

  try {
    const payload = {
      accountId,
      contractId,
      side: 0, // Buy
      size: 1,
      type: 2, // Market
      timeInForce: 0, // IOC
      stopLossBracket: {
        ticks: stopTicks,
        type: 4, // Stop order
      },
      takeProfitBracket: {
        ticks: targetTicks,
        type: 1, // Limit
      },
    };

    console.log('Placing native bracket order...\n');
    const response = await rest.placeOrder(payload);

    console.log('Full response:', JSON.stringify(response, null, 2));

    if (!response.success) {
      console.log(`\n❌ Order rejected: ${response.errorMessage || 'No error message'}`);
      console.log(`Error code: ${response.errorCode}`);
      return;
    }

    console.log(`✅ Order placed! ID: ${response.orderId}\n`);

    // Monitor for 15 seconds
    console.log('Monitoring position...');
    for (let i = 0; i < 15; i++) {
      await sleep(1000);
      const positions = await rest.getPositions(accountId);
      const nqPos = positions.find((p: any) => p.contractId === contractId);

      if (!nqPos || nqPos.netQty === 0) {
        console.log(`[${i + 1}s] ✅ Position FLAT - bracket worked!\n`);
        break;
      } else {
        console.log(`[${i + 1}s] Position: ${nqPos.netQty} contracts, PnL: $${nqPos.unrealizedPnL?.toFixed(2) || '0.00'}`);
      }
    }

    // Cleanup
    const finalPos = await rest.getPositions(accountId);
    const finalNQ = finalPos.find((p: any) => p.contractId === contractId);

    if (finalNQ && finalNQ.netQty !== 0) {
      console.log('\nClosing remaining position...');
      await rest.placeOrder({
        accountId,
        contractId,
        side: finalNQ.netQty > 0 ? 1 : 0,
        size: Math.abs(finalNQ.netQty),
        type: 2,
        timeInForce: 0,
      });
      console.log('✅ Closed\n');
    }

    console.log('=== TEST COMPLETE ===');

  } catch (err: any) {
    console.error('❌ Error:', err.message);
  }
}

testNQBracket().catch(console.error);
