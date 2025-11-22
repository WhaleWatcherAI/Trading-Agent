#!/usr/bin/env tsx
/**
 * Emergency script to cancel ALL open orders
 * Use this to clean up orphaned brackets
 */

import { authenticate } from './lib/topstepx';
import { createProjectXRest } from './projectx-rest';

const ACCOUNT_ID = parseInt(process.env.TOPSTEPX_ACCOUNT_ID || '13230351');

async function main() {
  console.log('🚨 EMERGENCY: Cancelling ALL open orders...\n');

  try {
    // Authenticate
    await authenticate();
    console.log('✅ Authenticated with TopStepX\n');

    // Create REST client
    const rest = createProjectXRest();

    // Fetch all open orders
    console.log(`📋 Fetching open orders for account ${ACCOUNT_ID}...`);
    const response = await rest.searchOpenOrders({ accountId: ACCOUNT_ID });

    if (!response?.success || !response?.orders) {
      console.error('❌ Failed to fetch open orders:', response?.errorMessage);
      return;
    }

    const openOrders = response.orders || [];
    console.log(`Found ${openOrders.length} open order(s)\n`);

    if (openOrders.length === 0) {
      console.log('✅ No open orders to cancel');
      return;
    }

    // Display orders
    console.log('Open Orders:');
    openOrders.forEach((order, i) => {
      const typeStr = order.type === 1 ? 'LIMIT' : order.type === 4 ? 'STOP' : order.type === 2 ? 'MARKET' : 'UNKNOWN';
      const sideStr = order.side === 0 ? 'BUY' : 'SELL';
      const priceStr = order.limitPrice ? `@ ${order.limitPrice}` : order.stopPrice ? `@ ${order.stopPrice}` : '';
      console.log(`  ${i + 1}. Order ${order.id}: ${sideStr} ${order.size} ${typeStr} ${priceStr}`);
    });

    console.log('');
    console.log('🗑️ Cancelling all orders...\n');

    let canceledCount = 0;
    let failedCount = 0;

    for (const order of openOrders) {
      try {
        console.log(`  Cancelling order ${order.id}...`);
        const result = await rest.cancelOrder({
          accountId: ACCOUNT_ID,
          orderId: String(order.id),
        });

        if (result?.success !== false) {
          canceledCount++;
          console.log(`  ✅ Cancelled ${order.id}`);
        } else {
          failedCount++;
          console.error(`  ❌ Failed to cancel ${order.id}: ${result?.errorMessage}`);
        }

        // Small delay between cancellations
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error: any) {
        failedCount++;
        console.error(`  ❌ Error cancelling ${order.id}:`, error?.message || error);
      }
    }

    console.log('');
    console.log('═'.repeat(60));
    console.log(`✅ Successfully cancelled: ${canceledCount}`);
    console.log(`❌ Failed to cancel: ${failedCount}`);
    console.log(`📊 Total processed: ${openOrders.length}`);
    console.log('═'.repeat(60));

    if (failedCount > 0) {
      console.log('\n⚠️ Some orders failed to cancel. Check manually on TopStepX platform.');
      process.exit(1);
    } else {
      console.log('\n✅ All orders cancelled successfully!');
      process.exit(0);
    }

  } catch (error: any) {
    console.error('\n❌ Fatal error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
