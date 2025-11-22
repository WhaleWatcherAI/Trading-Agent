#!/usr/bin/env tsx
/**
 * Test Bracket Order - Place a single test bracket order
 */

import 'dotenv/config';
import { createProjectXRest } from './projectx-rest';
import { authenticate } from './lib/topstepx';

const ACCOUNT_ID = 13056702;
const CONTRACT_ID = 'CON.F.US.ENQ.Z25'; // NQ
const TICK_SIZE = 0.25;

async function testBracketOrder() {
  console.log('🧪 TEST BRACKET ORDER');
  console.log('='.repeat(50));

  // Authenticate
  console.log('Authenticating...');
  await authenticate();

  // Create REST client
  const rest = createProjectXRest();

  // Test order: BUY 1 contract with 16 tick SL, 24 tick TP
  const payload = {
    accountId: ACCOUNT_ID,
    contractId: CONTRACT_ID,
    side: 0, // Buy
    size: 1,
    type: 2, // Market
    timeInForce: 0,
    limitPrice: null,
    stopPrice: null,
    trailPrice: null,
    customTag: 'TEST-BRACKET',
    stopLossBracket: {
      ticks: -16, // 16 ticks below entry
      type: 4  // Stop Market
    },
    takeProfitBracket: {
      ticks: 24, // 24 ticks above entry
      type: 1  // Limit
    }
  };

  console.log('Placing TEST LONG bracket order:');
  console.log(`  Account: ${ACCOUNT_ID}`);
  console.log(`  Contract: ${CONTRACT_ID}`);
  console.log(`  Size: 1`);
  console.log(`  Stop Loss: -16 ticks`);
  console.log(`  Take Profit: 24 ticks`);
  console.log('');

  try {
    const response = await rest.placeOrder(payload);
    console.log('✅ Order placed successfully!');
    console.log('Response:', JSON.stringify(response, null, 2));

    const orderId = response?.orderId ?? response?.id;
    if (orderId) {
      console.log(`\n📝 Entry Order ID: ${orderId}`);
      console.log('');
      console.log('⏰ Waiting 3 seconds for fill...');
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Check order status
      console.log('\n🔍 Checking order status...');
      try {
        const orders = await rest.getOrders({ accountId: ACCOUNT_ID });
        console.log('Active orders:', JSON.stringify(orders, null, 2));
      } catch (err: any) {
        console.log('Could not fetch orders:', err.message);
      }

      // Immediately flatten to close the test
      console.log('\n🛑 Flattening position (closing test)...');
      try {
        const flattenResponse = await rest.flatten({ accountId: ACCOUNT_ID, contractId: CONTRACT_ID });
        console.log('✅ Position flattened:', JSON.stringify(flattenResponse, null, 2));
      } catch (err: any) {
        console.log('⚠️ Flatten failed (may already be flat):', err.message);
      }
    }

  } catch (err: any) {
    console.error('❌ Order failed:', err.message);
    console.error('Full error:', err);
  }

  console.log('');
  console.log('='.repeat(50));
  console.log('Test complete');
  process.exit(0);
}

testBracketOrder();
