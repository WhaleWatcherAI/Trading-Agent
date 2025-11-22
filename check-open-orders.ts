import { createProjectXRest } from './projectx-rest';

async function checkOpenOrders() {
  const restClient = createProjectXRest();
  const accountId = 13056702;

  console.log(`Checking open orders for account ${accountId}...`);

  try {
    const result = await restClient.searchOpenOrders({ accountId });

    console.log(`\nOpen Orders Response:`);
    console.log(`Success: ${result.success}`);
    console.log(`Error Code: ${result.errorCode}`);
    console.log(`Error Message: ${result.errorMessage || 'None'}`);
    console.log(`Number of open orders: ${result.orders?.length || 0}`);

    if (result.orders && result.orders.length > 0) {
      console.log(`\nOrder Details:`);
      result.orders.forEach((order, idx) => {
        console.log(`\nOrder ${idx + 1}:`);
        console.log(`  ID: ${order.id}`);
        console.log(`  Contract: ${order.contractId}`);
        console.log(`  Type: ${order.type} (1=Limit, 2=Market, 4=Stop, 5=TrailingStop)`);
        console.log(`  Side: ${order.side} (0=Buy, 1=Sell)`);
        console.log(`  Size: ${order.size}`);
        console.log(`  Status: ${order.status}`);
        console.log(`  Limit Price: ${order.limitPrice || 'N/A'}`);
        console.log(`  Stop Price: ${order.stopPrice || 'N/A'}`);
        console.log(`  Fill Volume: ${order.fillVolume || 0}`);
        console.log(`  Filled Price: ${order.filledPrice || 'N/A'}`);
        console.log(`  Custom Tag: ${order.customTag || 'N/A'}`);
        console.log(`  Created: ${order.creationTimestamp}`);
      });
    } else {
      console.log('\nNo open orders found.');
    }
  } catch (error) {
    console.error('Error fetching open orders:', error);
  }
}

checkOpenOrders();
