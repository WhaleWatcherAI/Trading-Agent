#!/usr/bin/env tsx
import { createProjectXRest } from './projectx-rest';

async function closeOrphanedNQ() {
  const rest = createProjectXRest();
  const accountId = 13056702;
  const contractId = 'CON.F.US.ENQ.Z25';

  console.log('Closing orphaned LONG 3 NQ position...\n');

  try {
    const response = await rest.placeOrder({
      accountId,
      contractId,
      side: 1, // Sell
      size: 3,
      type: 2, // Market
      timeInForce: 0, // IOC
    });

    console.log('Response:', JSON.stringify(response, null, 2));

    if (response.success) {
      console.log('\n✅ Orphaned NQ position closed! Order ID:', response.orderId);
    } else {
      console.log('\n❌ Failed:', response.errorMessage);
    }
  } catch (err: any) {
    console.error('Error:', err.message);
  }
}

closeOrphanedNQ().catch(console.error);
