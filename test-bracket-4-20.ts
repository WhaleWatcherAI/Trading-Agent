#!/usr/bin/env tsx
import { createProjectXRest } from './projectx-rest';

async function test4and20Ticks() {
  const rest = createProjectXRest();

  const accountId = 13056702;
  const contractId = 'CON.F.US.ENQ.Z25'; // NQ

  console.log('=== Testing 4-Tick Stop / 20-Tick Target Bracket ===\n');
  console.log('NQ Bracket:');
  console.log('  Stop:   4 ticks = 1.00 point  = $20');
  console.log('  Target: 20 ticks = 5.00 points = $100');
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
  console.log('\nSending bracket order...\n');

  try {
    const response = await rest.placeOrder(order);
    console.log('Response:');
    console.log(JSON.stringify(response, null, 2));

    if (response.success) {
      console.log(`\n✅ Bracket order placed! Order ID: ${response.orderId}`);
      console.log('   Entry: MARKET BUY 1 NQ');
      console.log('   Stop: 4 ticks below entry (1 point = $20)');
      console.log('   Target: 20 ticks above entry (5 points = $100)');
      console.log('   Expected R:R = 1:5');
    } else {
      console.log(`\n❌ Failed: ${response.errorMessage}`);
    }
  } catch (err: any) {
    console.error('Error:', err.message);
  }
}

test4and20Ticks().catch(console.error);
