#!/usr/bin/env tsx
import { createProjectXRest } from './projectx-rest';

async function test2Ticks() {
  const rest = createProjectXRest();

  const accountId = 13056702;
  const contractId = 'CON.F.US.ENQ.Z25'; // NQ

  console.log('=== Testing 2-Tick Bracket ===\n');
  console.log('NQ: 2 ticks = 0.50 points = $10\n');

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
      ticks: -2,  // 2 ticks below
      type: 4
    },
    takeProfitBracket: {
      ticks: 2,  // 2 ticks above
      type: 1
    }
  };

  console.log('Order payload:');
  console.log(JSON.stringify(order, null, 2));
  console.log('\nSending order...\n');

  try {
    const response = await rest.placeOrder(order);
    console.log('Response:');
    console.log(JSON.stringify(response, null, 2));

    if (response.success) {
      console.log(`\n✅ 2-tick bracket accepted! Order ID: ${response.orderId}`);
      console.log('   Stop: 2 ticks = 0.5 points = $10');
      console.log('   Target: 2 ticks = 0.5 points = $10');
    } else {
      console.log(`\n❌ Failed: ${response.errorMessage}`);
    }
  } catch (err: any) {
    console.error('Error:', err.message);
  }
}

test2Ticks().catch(console.error);
