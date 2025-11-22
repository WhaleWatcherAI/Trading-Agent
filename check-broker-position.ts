#!/usr/bin/env tsx
/**
 * Check actual broker position, open orders, AND filled orders
 * This reveals if an entry filled but brackets weren't placed
 */

import { authenticate } from './lib/topstepx';
import { createProjectXRest } from './projectx-rest';

const ACCOUNT_ID = parseInt(process.env.TOPSTEPX_ACCOUNT_ID || '13230351');
const CONTRACT_ID = 'CON.F.US.ENQ.Z25'; // NQ

async function main() {
  console.log('🔍 Checking COMPLETE broker state (positions + open + filled orders)...\n');

  try {
    await authenticate();
    const rest = createProjectXRest();

    // 1. Check positions
    console.log('📊 Fetching positions...');
    const positions = await rest.getPositions(ACCOUNT_ID);
    console.log(`Found ${positions?.length || 0} position(s)\n`);

    let nqPosition = null;
    let nqQty = 0;

    if (positions && positions.length > 0) {
      positions.forEach(pos => {
        const qty = (pos.longQuantity || 0) - (pos.shortQuantity || 0);
        if (qty !== 0) {
          console.log(`Position: ${pos.symbol || pos.contractId}`);
          console.log(`  Contract ID: ${pos.contractId}`);
          console.log(`  Net Quantity: ${qty > 0 ? '+' : ''}${qty}`);
          console.log(`  Long: ${pos.longQuantity || 0}, Short: ${pos.shortQuantity || 0}`);
          console.log(`  Avg Price: ${pos.averagePrice || 'N/A'}`);
          console.log('');

          if (pos.contractId === CONTRACT_ID) {
            nqPosition = pos;
            nqQty = qty;
          }
        }
      });
    }

    // 2. Check open orders (pending brackets)
    console.log('📋 Fetching open/pending orders...');
    const ordersResp = await rest.searchOpenOrders({ accountId: ACCOUNT_ID });
    const openOrders = ordersResp?.orders || [];
    console.log(`Found ${openOrders.length} open order(s)\n`);

    const nqOpenOrders = openOrders.filter(o => o.contractId === CONTRACT_ID);

    if (openOrders.length > 0) {
      openOrders.forEach(order => {
        const typeStr = order.type === 1 ? 'LIMIT' : order.type === 4 ? 'STOP' : order.type === 2 ? 'MARKET' : `TYPE_${order.type}`;
        const sideStr = order.side === 0 ? 'BUY' : 'SELL';
        const priceStr = order.limitPrice ? `@ ${order.limitPrice}` : order.stopPrice ? `@ ${order.stopPrice}` : '';

        console.log(`Open Order ${order.id}: ${sideStr} ${order.size} ${typeStr} ${priceStr}`);
        console.log(`  Contract: ${order.contractId}`);
        console.log(`  Status: ${order.status}`);
        console.log('');
      });
    }

    // 3. Check FILLED orders (last 15 minutes to see recent activity)
    console.log('✅ Fetching recent FILLED orders (last 15 min)...');
    const fifteenMin = Date.now() - (15 * 60 * 1000);
    const filledResp = await rest.searchOrders({
      accountId: ACCOUNT_ID,
      startTimestamp: new Date(fifteenMin).toISOString(),
      endTimestamp: new Date().toISOString(),
      pageSize: 100,
    });
    const allOrders = filledResp?.orders || [];
    console.log(`Found ${allOrders.length} orders in last 15 min\n`);

    // Filter for NQ orders
    const recentNqOrders = allOrders
      .filter(o => o.contractId === CONTRACT_ID)
      .sort((a, b) => {
        const aTime = new Date(a.createdAt || a.updatedAt || 0).getTime();
        const bTime = new Date(b.createdAt || b.updatedAt || 0).getTime();
        return bTime - aTime; // newest first
      });

    if (recentNqOrders.length > 0) {
      console.log(`Recent NQ orders (last 15 min):`);
      recentNqOrders.forEach(order => {
        const typeStr = order.type === 1 ? 'LIMIT' : order.type === 4 ? 'STOP' : order.type === 2 ? 'MARKET' : `TYPE_${order.type}`;
        const sideStr = order.side === 0 ? 'BUY' : 'SELL';
        const statusStr = order.status === 2 ? 'FILLED' : order.status === 4 ? 'CANCELED' : order.status === 1 ? 'OPEN' : `STATUS_${order.status}`;
        const priceStr = order.averagePrice ? `@ ${order.averagePrice}` : order.limitPrice ? `@ ${order.limitPrice}` : order.stopPrice ? `@ ${order.stopPrice}` : '';
        const timeStr = order.updatedAt || order.createdAt || '';

        console.log(`  ${timeStr.slice(11, 19)} | Order ${order.id}: ${sideStr} ${order.size} ${typeStr} ${priceStr} | ${statusStr}`);
      });
      console.log('');
    }

    // 4. CRITICAL ANALYSIS - Detect naked positions
    console.log('═'.repeat(60));
    console.log('🔍 NAKED POSITION DETECTION:');
    console.log('═'.repeat(60));

    if (nqQty !== 0) {
      console.log(`🚨 NQ POSITION OPEN: ${nqQty > 0 ? 'LONG' : 'SHORT'} ${Math.abs(nqQty)} @ ${nqPosition?.averagePrice || 'unknown'}`);
      console.log(`🛡️ Protective orders (pending): ${nqOpenOrders.length}`);

      if (nqOpenOrders.length === 0) {
        console.log('\n⚠️⚠️⚠️  CRITICAL: NAKED POSITION - NO PROTECTIVE ORDERS! ⚠️⚠️⚠️');

        // Find the entry order in filled orders
        const filledEntries = recentNqOrders.filter(o =>
          o.status === 2 && // Filled
          o.type === 2 && // Market order (entry)
          ((nqQty > 0 && o.side === 0) || (nqQty < 0 && o.side === 1)) // Matches position direction
        );

        if (filledEntries.length > 0) {
          const entry = filledEntries[0];
          console.log(`\n📍 Found entry order that filled:`);
          console.log(`   Order ${entry.id}: ${entry.side === 0 ? 'BUY' : 'SELL'} ${entry.size} @ ${entry.averagePrice}`);
          console.log(`   Time: ${entry.updatedAt || entry.createdAt}`);

          // Check if brackets were placed but then canceled
          const bracketOrders = recentNqOrders.filter(o =>
            o.id !== entry.id &&
            (o.type === 4 || o.type === 1) && // Stop or Limit
            o.side === (entry.side === 0 ? 1 : 0) // Opposite side (protective)
          );

          if (bracketOrders.length > 0) {
            console.log(`\n⚠️ Found ${bracketOrders.length} bracket order(s) that were placed:`);
            bracketOrders.forEach(b => {
              const statusStr = b.status === 2 ? 'FILLED' : b.status === 4 ? 'CANCELED' : b.status === 1 ? 'OPEN' : `STATUS_${b.status}`;
              console.log(`   Order ${b.id}: ${b.type === 4 ? 'STOP' : 'LIMIT'} | ${statusStr}`);
            });
            console.log(`\n🔥 ISSUE: Brackets were placed but then canceled/filled, leaving naked position!`);
          } else {
            console.log(`\n🔥 ISSUE: Entry filled but NO bracket orders were ever placed!`);
          }
        }

      } else if (nqOpenOrders.length < 2) {
        console.log('\n⚠️ WARNING: INCOMPLETE BRACKETS');
        console.log(`   Expected: 2 orders (stop + target)`);
        console.log(`   Found: ${nqOpenOrders.length}`);

        const hasStop = nqOpenOrders.some(o => o.type === 4);
        const hasTarget = nqOpenOrders.some(o => o.type === 1);

        if (!hasStop) console.log('   ❌ Missing STOP order');
        if (!hasTarget) console.log('   ❌ Missing TARGET order');
      } else {
        console.log('✅ Position properly protected with 2 brackets');
      }
    } else {
      console.log('✅ No NQ position - system is flat');

      // Check if there are orphaned brackets (brackets without position)
      if (nqOpenOrders.length > 0) {
        console.log(`\n⚠️ WARNING: Found ${nqOpenOrders.length} ORPHANED bracket order(s) (no position):`);
        nqOpenOrders.forEach(o => {
          const typeStr = o.type === 1 ? 'LIMIT' : o.type === 4 ? 'STOP' : `TYPE_${o.type}`;
          console.log(`   Order ${o.id}: ${o.side === 0 ? 'BUY' : 'SELL'} ${typeStr}`);
        });
      }
    }
    console.log('═'.repeat(60));

  } catch (error: any) {
    console.error('\n❌ Error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
