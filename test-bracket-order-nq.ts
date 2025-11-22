#!/usr/bin/env tsx
import { createProjectXRest } from './projectx-rest';

async function testBracketOrder() {
  const rest = createProjectXRest();

  const accountId = 13056702;
  const contractId = 'CON.F.US.ENQ.Z25'; // NQ
  const currentPrice = 25707.11;

  console.log('=== NQ Bracket Order Test ===\n');
  console.log(`Current NQ price: ${currentPrice}`);
  console.log('Bracket: 1% stop and 1% target\n');

  // Calculate 1% in ticks
  // 1% of 25707 = 257.07 points
  // NQ tick size = 0.25
  // 257.07 / 0.25 = 1028 ticks
  // API limit: 1000 ticks max
  const onePercent = currentPrice * 0.01;
  const tickSize = 0.25;
  const ticks = Math.min(1000, Math.round(onePercent / tickSize));

  console.log(`1% = ${onePercent.toFixed(2)} points = ${ticks} ticks`);
  console.log(`Stop @ ${(currentPrice - onePercent).toFixed(2)}`);
  console.log(`Target @ ${(currentPrice + onePercent).toFixed(2)}\n`);

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
      ticks: -ticks,  // Negative for LONG (stop below entry)
      type: 4  // Stop Market (4 = Stop)
    },
    takeProfitBracket: {
      ticks: ticks,  // Positive for LONG (target above entry)
      type: 1  // Limit target
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
      console.log(`   Stop: ${ticks} ticks below entry (~${(currentPrice - onePercent).toFixed(2)})`);
      console.log(`   Target: ${ticks} ticks above entry (~${(currentPrice + onePercent).toFixed(2)})`);
    } else {
      console.log(`\n❌ Failed: ${response.errorMessage}`);
    }
  } catch (err: any) {
    console.error('Error:', err.message);
  }
}

testBracketOrder().catch(console.error);
