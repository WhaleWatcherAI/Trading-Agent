#!/usr/bin/env tsx
import { createProjectXRest } from './projectx-rest';

async function closeMES() {
  const rest = createProjectXRest();
  const accountId = 13056702;
  const contractId = 'CON.F.US.MES.Z25';

  console.log('Closing stuck MES position...\n');

  try {
    const response = await rest.placeOrder({
      accountId,
      contractId,
      side: 1, // Sell (close long)
      size: 3,
      type: 2, // Market
      timeInForce: 0, // IOC
    });

    if (response.success) {
      console.log(`✅ MES position closed! Order ID: ${response.orderId}\n`);
    } else {
      console.log(`❌ Failed: ${response.errorMessage}\n`);
    }
  } catch (err: any) {
    console.error('Error:', err.message);
  }
}

closeMES().catch(console.error);
