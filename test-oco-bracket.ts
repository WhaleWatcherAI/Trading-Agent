import { createProjectXRest } from './projectx-rest';

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testOCOBracket() {
  const rest = createProjectXRest();
  const accountId = 13056702;
  const contractId = 'CON.F.US.MNQ.Z25'; // MNQ
  const tickSize = 0.25;
  const ticksAway = 2; // 2 ticks = 0.50 points
  const distance = tickSize * ticksAway;

  console.log('=== OCO BRACKET ORDER TEST (2 ticks each side) ===\n');

  try {
    // Step 1: Place entry market order
    console.log('1. Placing Buy MARKET entry order...');
    const entryOrder = await rest.placeOrder({
      accountId,
      contractId,
      side: 0, // Buy
      size: 1,
      type: 2, // Market
      timeInForce: 1, // GTC
    });

    if (!entryOrder.success) {
      throw new Error(`Entry failed: ${entryOrder.errorMessage}`);
    }

    console.log(`   ✅ Entry order: ${entryOrder.orderId}`);

    // Wait a moment for fill
    await sleep(2000);

    // Estimate entry price (we'll use 25565 as rough estimate)
    // In production, you'd get this from fill event
    const estimatedEntry = 25565;
    const stopPrice = estimatedEntry - distance; // 2 ticks below
    const targetPrice = estimatedEntry + distance; // 2 ticks above

    console.log(`   Estimated entry: ${estimatedEntry}`);
    console.log(`   Stop: ${stopPrice} (2 ticks = $${distance})`);
    console.log(`   Target: ${targetPrice} (2 ticks = $${distance})`);

    // Step 2: Place stop and target (bracket legs)
    console.log('\n2. Placing bracket legs (Stop + Target)...');

    const [stopOrder, targetOrder] = await Promise.all([
      rest.placeOrder({
        accountId,
        contractId,
        side: 1, // Sell (exit)
        size: 1,
        type: 1, // Limit
        timeInForce: 1, // GTC (so it stays open)
        limitPrice: stopPrice,
      }),
      rest.placeOrder({
        accountId,
        contractId,
        side: 1, // Sell (exit)
        size: 1,
        type: 1, // Limit
        timeInForce: 1, // GTC
        limitPrice: targetPrice,
      }),
    ]);

    console.log(`   ✅ Stop order: ${stopOrder.orderId} @ ${stopPrice}`);
    console.log(`   ✅ Target order: ${targetOrder.orderId} @ ${targetPrice}`);

    if (!stopOrder.success || !targetOrder.success) {
      console.log(`   ⚠️ Stop: ${stopOrder.success ? 'OK' : stopOrder.errorMessage}`);
      console.log(`   ⚠️ Target: ${targetOrder.success ? 'OK' : targetOrder.errorMessage}`);
      throw new Error('Bracket orders failed');
    }

    // Step 3: Monitor which fills first
    console.log('\n3. Monitoring for fills (checking every 2 seconds, max 30 seconds)...');
    console.log('   With 2-tick spread, one should fill quickly in active market');

    let stopFilled = false;
    let targetFilled = false;
    let iterations = 0;
    const maxIterations = 15; // 30 seconds

    while (iterations < maxIterations && !stopFilled && !targetFilled) {
      await sleep(2000);
      iterations++;

      // Check positions to see if we're flat (would indicate one filled)
      const positions = await rest.getPositions(accountId);
      const mnqPos = positions.find((p: any) => p.contractId === contractId);

      if (!mnqPos || mnqPos.netQty === 0) {
        console.log(`   [${iterations}] One leg filled! Position now flat.`);

        // Try to determine which filled by checking if orders still exist
        // In real implementation, you'd track via WebSocket events
        console.log('   Checking which leg filled...');

        // Try to cancel both to see which one fails (already filled)
        let stopCancelResult, targetCancelResult;

        try {
          stopCancelResult = await rest.cancelOrder({ accountId, orderId: String(stopOrder.orderId) });
          console.log(`   Stop cancel result: ${JSON.stringify(stopCancelResult)}`);
        } catch (err: any) {
          console.log(`   ⚠️ Stop cancel failed: ${err.message}`);
          stopFilled = true;
        }

        try {
          targetCancelResult = await rest.cancelOrder({ accountId, orderId: String(targetOrder.orderId) });
          console.log(`   Target cancel result: ${JSON.stringify(targetCancelResult)}`);
        } catch (err: any) {
          console.log(`   ⚠️ Target cancel failed: ${err.message}`);
          targetFilled = true;
        }

        break;
      }

      console.log(`   [${iterations}] Position still open (netQty: ${mnqPos?.netQty || 0}), waiting...`);
    }

    if (!stopFilled && !targetFilled) {
      console.log('\n⚠️ No fills after 30 seconds - market may be slow');
      console.log('Cancelling both orders to clean up...');

      try {
        await rest.cancelOrder({ accountId, orderId: String(stopOrder.orderId) });
        console.log('   ✅ Stop cancelled');
      } catch (err: any) {
        console.log(`   ⚠️ Stop cancel: ${err.message}`);
      }

      try {
        await rest.cancelOrder({ accountId, orderId: String(targetOrder.orderId) });
        console.log('   ✅ Target cancelled');
      } catch (err: any) {
        console.log(`   ⚠️ Target cancel: ${err.message}`);
      }

      // Close any remaining position
      const finalPos = await rest.getPositions(accountId);
      const finalMnq = finalPos.find((p: any) => p.contractId === contractId);
      if (finalMnq && finalMnq.netQty !== 0) {
        console.log('\nClosing remaining position...');
        await rest.placeOrder({
          accountId,
          contractId,
          side: finalMnq.netQty > 0 ? 1 : 0, // Opposite side
          size: Math.abs(finalMnq.netQty),
          type: 2,
          timeInForce: 1,
        });
        console.log('   ✅ Position closed');
      }

      return;
    }

    // Step 4: Verify OCO worked (other leg should be cancelled)
    console.log('\n4. Verifying OCO logic...');

    if (stopFilled) {
      console.log('   🔴 STOP filled first (took loss)');
      console.log('   Checking if target was automatically cancelled...');

      // In the real strategy, handlePositionExit would cancel target
      // For this test, we manually verify
      try {
        await rest.cancelOrder({ accountId, orderId: String(targetOrder.orderId) });
        console.log('   ⚠️ Target was NOT cancelled automatically (still active)');
        console.log('   ❌ OCO LOGIC NOT WORKING - this would create unwanted position!');
      } catch (err: any) {
        console.log('   ✅ Target already cancelled/filled (OCO worked)');
      }
    } else if (targetFilled) {
      console.log('   🟢 TARGET filled first (took profit)');
      console.log('   Checking if stop was automatically cancelled...');

      try {
        await rest.cancelOrder({ accountId, orderId: String(stopOrder.orderId) });
        console.log('   ⚠️ Stop was NOT cancelled automatically (still active)');
        console.log('   ❌ OCO LOGIC NOT WORKING - this would create unwanted position!');
      } catch (err: any) {
        console.log('   ✅ Stop already cancelled/filled (OCO worked)');
      }
    }

    // Final position check
    console.log('\n5. Final position check...');
    const finalPositions = await rest.getPositions(accountId);
    const finalMnqPos = finalPositions.find((p: any) => p.contractId === contractId);

    if (!finalMnqPos || finalMnqPos.netQty === 0) {
      console.log('   ✅ Position is FLAT (netQty = 0)');
      console.log('   ✅ No orphaned positions created');
    } else {
      console.log(`   ⚠️ Position NOT flat: netQty = ${finalMnqPos.netQty}`);
      console.log('   ⚠️ There may be an issue - closing now...');

      await rest.placeOrder({
        accountId,
        contractId,
        side: finalMnqPos.netQty > 0 ? 1 : 0,
        size: Math.abs(finalMnqPos.netQty),
        type: 2,
        timeInForce: 1,
      });
      console.log('   ✅ Cleaned up position');
    }

    console.log('\n=== TEST COMPLETE ===');
    console.log('NOTE: This test checks API-level OCO. The real strategy');
    console.log('uses WebSocket events to trigger OCO cancellation in handlePositionExit()');

  } catch (err: any) {
    console.error('\n❌ TEST FAILED:', err.message);
    console.error('Stack:', err.stack);
  }
}

testOCOBracket().catch(console.error);
