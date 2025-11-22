import { createProjectXRest } from './projectx-rest';

async function closePosition() {
  const rest = createProjectXRest();
  const accountId = 13056702;
  const contractId = 'CON.F.US.MNQ.Z25';

  console.log('Closing SHORT position and cancelling open orders...\n');

  // Cancel the open sell order @ 25570
  try {
    console.log('1. Cancelling open sell limit @ 25570 (order 1902119244)...');
    await rest.cancelOrder({ accountId, orderId: '1902119244' });
    console.log('   ✅ Cancelled');
  } catch (err: any) {
    console.log(`   ⚠️ ${err.message}`);
  }

  // Close the SHORT position with a BUY order
  try {
    console.log('\n2. Placing BUY market to close SHORT position...');
    const result = await rest.placeOrder({
      accountId,
      contractId,
      side: 0, // Buy to cover short
      size: 1,
      type: 2, // Market
      timeInForce: 1, // GTC
    });
    console.log(`   ✅ Buy order placed: ${result.orderId}`);
    console.log(`   Position should now be flat`);
  } catch (err: any) {
    console.log(`   ❌ Failed: ${err.message}`);
  }
}

closePosition().catch(console.error);
