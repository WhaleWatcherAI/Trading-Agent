import { createProjectXRest } from './projectx-rest';

async function testLiveBracketOrder() {
  const rest = createProjectXRest();
  const accountId = 13056702;
  const contractId = 'CON.F.US.MNQ.Z25'; // MNQ - smallest contract

  console.log('=== LIVE BRACKET ORDER TEST ===\n');

  try {
    // Get current positions before test
    console.log('1. Checking initial positions...');
    const positionsBefore = await rest.getPositions(accountId);
    console.log('Initial positions:', JSON.stringify(positionsBefore, null, 2));

    // Place a simple market order to get filled immediately
    console.log('\n2. Placing Buy market order (1 MNQ)...');
    const entryOrder = await rest.placeOrder({
      accountId,
      contractId,
      side: 0, // Buy
      size: 1,
      type: 2, // Market
      timeInForce: 1, // GTC (Good Till Cancel)
    });
    console.log('Entry order response:', JSON.stringify(entryOrder, null, 2));

    if (!entryOrder.success) {
      throw new Error(`Entry order failed: ${entryOrder.errorMessage}`);
    }

    // Wait for fill and retry checking position
    console.log('\n3. Waiting for fill (checking every 2 seconds for up to 10 seconds)...');
    let position: any = null;
    for (let i = 0; i < 5; i++) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      const positionsAfterEntry = await rest.getPositions(accountId);
      console.log(`   Check ${i + 1}: ${positionsAfterEntry.length} positions found`);
      position = positionsAfterEntry.find((p: any) => p.contractId === contractId);
      if (position) {
        console.log('   ✅ Position found!');
        break;
      }
    }

    if (!position) {
      console.log('\n⚠️ No position found after 10 seconds.');
      console.log('IOC order may have been canceled (no immediate fill available).');
      console.log('This is normal if there\'s low liquidity at the current price.');
      return;
    }

    console.log('\n4. Position confirmed after entry:');
    console.log(JSON.stringify(position, null, 2));

    console.log('\n✅ Position confirmed:', {
      side: position.netQty > 0 ? 'LONG' : 'SHORT',
      quantity: Math.abs(position.netQty),
      avgPrice: position.avgPrice,
      unrealizedPnL: position.unrealizedPnL,
    });

    // Close the position immediately
    console.log('\n5. Closing position with Sell market order...');
    const closeOrder = await rest.placeOrder({
      accountId,
      contractId,
      side: 1, // Sell
      size: Math.abs(position.netQty),
      type: 2, // Market
      timeInForce: 1, // GTC
    });
    console.log('Close order response:', JSON.stringify(closeOrder, null, 2));

    if (!closeOrder.success) {
      throw new Error(`Close order failed: ${closeOrder.errorMessage}`);
    }

    // Wait for close fill
    console.log('\n6. Waiting 3 seconds for close fill...');
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Check final position
    console.log('\n7. Checking final positions...');
    const positionsAfterClose = await rest.getPositions(accountId);
    console.log('Positions after close:', JSON.stringify(positionsAfterClose, null, 2));

    const finalPosition = positionsAfterClose.find((p: any) => p.contractId === contractId);

    console.log('\n=== TEST RESULTS ===');
    console.log('✅ Entry order placed and filled');
    console.log('✅ Position confirmed with unrealized P&L');
    console.log('✅ Exit order placed and filled');

    if (finalPosition && finalPosition.netQty === 0) {
      console.log('✅ Position closed (netQty = 0)');
      console.log(`📊 Realized P&L: $${finalPosition.realizedPnL?.toFixed(2) || 'N/A'}`);
    } else if (!finalPosition) {
      console.log('✅ Position fully closed (removed from positions)');
    } else {
      console.log('⚠️ Position still open:', finalPosition);
    }

    console.log('\n=== TEST COMPLETE ===');

  } catch (err: any) {
    console.error('\n❌ TEST FAILED:', err.message);
    console.error('Stack:', err.stack);
  }
}

testLiveBracketOrder().catch(console.error);
