#!/usr/bin/env tsx
import { createProjectXRest } from './projectx-rest';

async function testMNQBracket() {
  const rest = createProjectXRest();

  const accountId = 13056702;
  const contractId = 'CON.F.US.MNQ.Z25'; // MNQ (Micro Nasdaq)

  console.log('=== Testing MNQ: 4-Tick Stop / 20-Tick Target Bracket ===\n');
  console.log('MNQ Bracket (Micro Nasdaq):');
  console.log('  Stop:   4 ticks = 1.00 point  = $2');
  console.log('  Target: 20 ticks = 5.00 points = $10');
  console.log('  Risk/Reward: 1:5\n');

  const order = {
    accountId,
    contractId,
    type: 2,  // Market
    side: 0,  // Buy
    size: 1,
    timeInForce: 0, // IOC
    limitPrice: null,
    stopPrice: null,
    trailPrice: null,
    customTag: null,
    stopLossBracket: {
      ticks: -4,  // 4 ticks below (LONG)
      type: 4     // Stop Market
    },
    takeProfitBracket: {
      ticks: 20,  // 20 ticks above (LONG)
      type: 1     // Limit
    }
  };

  console.log('Order payload:');
  console.log(JSON.stringify(order, null, 2));
  console.log('\nSending MNQ bracket order...\n');

  try {
    const response = await rest.placeOrder(order);
    console.log('Response:');
    console.log(JSON.stringify(response, null, 2));

    if (response.success) {
      console.log(`\n✅ MNQ bracket order placed! Order ID: ${response.orderId}`);
      console.log('   Entry: MARKET BUY 1 MNQ');
      console.log('   Stop: 4 ticks below entry (1 point = $2)');
      console.log('   Target: 20 ticks above entry (5 points = $10)');
      console.log('   Expected R:R = 1:5');
    } else {
      console.log(`\n❌ Failed: ${response.errorMessage}`);
    }
  } catch (err: any) {
    console.error('Error:', err.message);
  }
}

testMNQBracket().catch(console.error);
