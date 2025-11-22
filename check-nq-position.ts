#!/usr/bin/env tsx
import 'dotenv/config';
import { createProjectXRest } from './projectx-rest';

async function checkPosition() {
  const rest = createProjectXRest();
  const accountId = 13230351;

  console.log('Checking NQ position and orders for account', accountId);

  try {
    // Check positions
    const positions = await rest.getPositions(accountId);
    console.log('\n📊 POSITIONS:', JSON.stringify(positions, null, 2));

    // Check open orders
    const orders = await rest.searchOpenOrders({ accountId });
    console.log('\n📋 OPEN ORDERS:', JSON.stringify(orders, null, 2));

    // Filter for NQ
    if (orders?.orders) {
      const nqOrders = orders.orders.filter((o: any) => o.contractId === 'CON.F.US.ENQ.Z25');
      console.log('\n🎯 NQ ORDERS ONLY:', JSON.stringify(nqOrders, null, 2));
    }

  } catch (error: any) {
    console.error('Error:', error.message);
  }
}

checkPosition();
