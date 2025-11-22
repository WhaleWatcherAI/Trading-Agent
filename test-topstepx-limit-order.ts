#!/usr/bin/env tsx
/**
 * Test TopstepX Limit Order Placement
 * Tests that our API client matches the official API spec
 */

import { createProjectXRest } from './projectx-rest';

async function testLimitOrder() {
  const rest = createProjectXRest();

  // Test configuration
  const accountId = 13056702;
  const contractId = 'CON.F.US.MES.Z25'; // MES contract

  console.log('=== TopstepX Limit Order API Test ===\n');
  console.log('Account ID:', accountId);
  console.log('Contract:', contractId);
  console.log('\nThis will place a LIMIT order far from market to avoid fills.\n');

  try {
    // Use approximate current price (MES ~6880)
    const currentPrice = 6880;
    console.log(`Approximate current price: ${currentPrice}\n`);

    // Place limit order 100 points away from market (won't fill)
    const limitPrice = currentPrice - 100;

    console.log('Placing LIMIT BUY order:');
    console.log(`  Side: BUY (0)`);
    console.log(`  Size: 1 contract`);
    console.log(`  Limit Price: ${limitPrice.toFixed(2)} (100 points below market)`);
    console.log(`  Type: 1 (Limit)`);
    console.log(`  Time In Force: 0 (IOC - will cancel immediately if not filled)\n`);

    const orderPayload = {
      accountId,
      contractId,
      type: 1, // 1 = Limit
      side: 0, // 0 = Buy
      size: 1,
      timeInForce: 0, // IOC
      limitPrice: limitPrice,
      stopPrice: null,
      trailPrice: null,
      customTag: null,
      stopLossBracket: null,
      takeProfitBracket: null,
    };

    console.log('Request payload:');
    console.log(JSON.stringify(orderPayload, null, 2));
    console.log('\nSending order...\n');

    const response = await rest.placeOrder(orderPayload);

    console.log('Response:');
    console.log(JSON.stringify(response, null, 2));

    if (response.success) {
      console.log(`\n✅ Limit order placed successfully! Order ID: ${response.orderId}`);
      console.log('   (Order likely cancelled immediately due to IOC and being far from market)');
    } else {
      console.log(`\n❌ Order failed: ${response.errorMessage}`);
      console.log(`   Error code: ${response.errorCode}`);
    }

  } catch (error: any) {
    console.error('\n❌ Error:', error.message);
    console.error('Stack:', error.stack);
  }
}

testLimitOrder().catch(console.error);
