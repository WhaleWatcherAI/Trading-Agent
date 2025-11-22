#!/usr/bin/env tsx
import { createProjectXRest } from './projectx-rest';

async function testBracketDollars() {
  const rest = createProjectXRest();

  const accountId = 13056702;
  const contractId = 'CON.F.US.ENQ.Z25'; // NQ

  console.log('=== Testing Bracket with Dollar Values ===\n');
  console.log('Attempting to use dollar amounts instead of ticks...\n');

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
      ticks: 50,  // Trying $50 (should be interpreted as 50 ticks if ticks-only)
      type: 4
    },
    takeProfitBracket: {
      ticks: 100,  // Trying $100
      type: 1
    }
  };

  console.log('Order payload (using small numbers as test):');
  console.log(JSON.stringify(order, null, 2));
  console.log('\nSending order...\n');

  try {
    const response = await rest.placeOrder(order);
    console.log('Response:');
    console.log(JSON.stringify(response, null, 2));

    if (response.success) {
      console.log(`\n✅ Order accepted! Order ID: ${response.orderId}`);
      console.log('   API interpreted values as TICKS (not dollars)');
      console.log('   Stop: 50 ticks = 12.5 points = $250');
      console.log('   Target: 100 ticks = 25 points = $500');
    } else {
      console.log(`\n❌ Failed: ${response.errorMessage}`);
    }
  } catch (err: any) {
    console.error('Error:', err.message);
  }
}

testBracketDollars().catch(console.error);
