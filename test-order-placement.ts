import { createProjectXRest } from './projectx-rest';

async function testOrderPlacement() {
  const rest = createProjectXRest();
  const accountId = 13056702;
  const contractId = 'CON.F.US.MGC.Z25'; // MGC contract

  console.log('\n=== TEST 1: Current format (nested request) ===');
  try {
    const payload1 = {
      request: {
        accountId,
        contractId,
        side: 0, // Buy
        size: 1,
        type: 2, // Market
        timeInForce: 0, // IOC
      }
    };
    console.log('Sending payload:', JSON.stringify(payload1, null, 2));
    const result1 = await rest.placeOrder(payload1 as any);
    console.log('✅ SUCCESS:', result1);
  } catch (err: any) {
    console.log('❌ FAILED:', err.message);
  }

  console.log('\n=== TEST 2: Flat format (no nesting) ===');
  try {
    const payload2 = {
      accountId,
      contractId,
      side: 0, // Buy
      size: 1,
      type: 2, // Market
      timeInForce: 0, // IOC
    };
    console.log('Sending payload:', JSON.stringify(payload2, null, 2));
    // @ts-ignore - testing different format
    const result2 = await rest.placeOrder(payload2 as any);
    console.log('✅ SUCCESS:', result2);
  } catch (err: any) {
    console.log('❌ FAILED:', err.message);
  }

  console.log('\n=== TEST 3: With "request" at root level ===');
  try {
    const payload3 = {
      request: {
        accountId,
        contractId,
        side: 0,
        size: 1,
        type: 2,
        timeInForce: 0,
      }
    };
    console.log('Sending wrapped request directly to API...');
    // Send directly with fetch to bypass placeOrder wrapper
    const jwt = process.env.TOPSTEPX_JWT;
    const response = await fetch('https://api.topstepx.com/api/Order/place', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${jwt}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload3),
    });
    const text = await response.text();
    console.log(`Status: ${response.status}`);
    console.log('Response:', text);
  } catch (err: any) {
    console.log('❌ FAILED:', err.message);
  }

  console.log('\n=== Done ===');
}

testOrderPlacement().catch(console.error);
