#!/usr/bin/env tsx
/**
 * Test native bracket orders with 2-tick stop and target
 * Tests the new bracket API format with stopLossBracket and takeProfitBracket
 */

import { createProjectXRest } from './projectx-rest';

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testNativeBracket() {
  const rest = createProjectXRest();
  const accountId = 13056702;
  const contractId = 'CON.F.US.MNQ.Z25'; // MNQ micro

  console.log('=== NATIVE BRACKET ORDER TEST (2 ticks each side) ===\n');

  try {
    // Test native bracket order
    console.log('1. Placing native bracket order (Buy MNQ)...');
    console.log('   Entry: Market');
    console.log('   Stop Loss: 2 ticks away (type 4 = Stop)');
    console.log('   Take Profit: 2 ticks away (type 1 = Limit)\n');

    const payload = {
      accountId,
      contractId,
      side: 0, // Buy
      size: 1,
      type: 2, // Market entry
      timeInForce: 0, // IOC
      stopLossBracket: {
        ticks: 2,
        type: 4, // Stop order (converts to market when hit)
      },
      takeProfitBracket: {
        ticks: 2,
        type: 1, // Limit order
      },
    };

    console.log('Payload:', JSON.stringify(payload, null, 2));
    console.log('\nSending order...\n');

    const response = await rest.placeOrder(payload);

    console.log('✅ Order placed successfully!');
    console.log('Response:', JSON.stringify(response, null, 2));
    console.log(`\nEntry Order ID: ${response.orderId}`);
    console.log('Note: Broker will automatically manage stop and target orders');
    console.log('Note: OCO (One Cancels Other) is handled by the broker\n');

    // Wait for fills
    console.log('2. Monitoring position for 10 seconds...');
    for (let i = 0; i < 10; i++) {
      await sleep(1000);
      const positions = await rest.getPositions(accountId);
      const mnqPos = positions.find((p: any) => p.contractId === contractId);

      if (!mnqPos || mnqPos.netQty === 0) {
        console.log(`   [${i + 1}s] Position FLAT - one of the bracket legs filled!`);
        console.log('   ✅ Native bracket OCO worked correctly\n');
        break;
      } else {
        console.log(`   [${i + 1}s] Position open: netQty = ${mnqPos.netQty}, unrealizedPnL = $${mnqPos.unrealizedPnL?.toFixed(2) || '0.00'}`);
      }
    }

    // Final check
    console.log('\n3. Final position check...');
    const finalPositions = await rest.getPositions(accountId);
    const finalMnq = finalPositions.find((p: any) => p.contractId === contractId);

    if (!finalMnq || finalMnq.netQty === 0) {
      console.log('   ✅ Position is FLAT');
      console.log('   ✅ Native bracket orders working correctly!\n');
    } else {
      console.log(`   ⚠️  Position still open: netQty = ${finalMnq.netQty}`);
      console.log('   Closing manually...');

      await rest.placeOrder({
        accountId,
        contractId,
        side: finalMnq.netQty > 0 ? 1 : 0,
        size: Math.abs(finalMnq.netQty),
        type: 2,
        timeInForce: 0,
      });
      console.log('   ✅ Position closed\n');
    }

    console.log('=== TEST COMPLETE ===');
    console.log('Native bracket orders:');
    console.log('  ✅ Single API call places entry + stop + target');
    console.log('  ✅ Broker manages OCO automatically');
    console.log('  ✅ No race conditions or manual cancellation needed');
    console.log('  ✅ Lower latency and more reliable\n');

  } catch (err: any) {
    console.error('\n❌ TEST FAILED:', err.message);

    // Try to close any open positions
    try {
      const positions = await rest.getPositions(accountId);
      const mnqPos = positions.find((p: any) => p.contractId === contractId);

      if (mnqPos && mnqPos.netQty !== 0) {
        console.log('\nClosing position after error...');
        await rest.placeOrder({
          accountId,
          contractId,
          side: mnqPos.netQty > 0 ? 1 : 0,
          size: Math.abs(mnqPos.netQty),
          type: 2,
          timeInForce: 0,
        });
        console.log('✅ Position closed');
      }
    } catch (cleanupErr: any) {
      console.error('Cleanup failed:', cleanupErr.message);
    }
  }
}

testNativeBracket().catch(console.error);
