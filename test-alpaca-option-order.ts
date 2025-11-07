import 'dotenv/config';
import { randomUUID } from 'crypto';
import {
  submitOptionOrder,
  cancelOptionOrder,
  getOptionOrder,
  AlpacaOptionOrder,
  AlpacaOptionContract,
} from './lib/alpaca';
import {
  fetchTradierOptionExpirations,
  fetchTradierOptionChain,
} from './lib/tradier';

const OPTION_EXPIRATION_EOD_SUFFIX = 'T21:00:00Z';

async function pickTestContract(symbol: string): Promise<AlpacaOptionContract> {
  const expirations = await fetchTradierOptionExpirations(symbol);
  if (!expirations.length) {
    throw new Error(`Tradier returned no expirations for ${symbol}`);
  }
  const sorted = expirations
    .map(exp => ({ exp, epoch: Date.parse(`${exp}${OPTION_EXPIRATION_EOD_SUFFIX}`) }))
    .filter(item => Number.isFinite(item.epoch))
    .sort((a, b) => a.epoch - b.epoch);
  const target = sorted[0]?.exp ?? expirations[0];
  const chain = await fetchTradierOptionChain(symbol, target);
  if (!chain.length) {
    throw new Error(`Tradier returned no contracts for ${symbol} ${target}`);
  }
  return chain.find(c => c.option_type === 'call') ?? chain[0];
}

function deriveLimitPrice(contract: AlpacaOptionContract): number {
  const ask = contract.ask_price ?? null;
  const bid = contract.bid_price ?? null;
  const last = contract.last_trade_price ?? null;

  let price: number | null = null;
  if (ask && ask > 0) {
    price = ask;
  } else if (last && last > 0) {
    price = last;
  } else if (bid && bid > 0) {
    price = bid * 1.05;
  }
  if (price === null || !Number.isFinite(price) || price <= 0) {
    price = 1;
  }
  const capped = Math.min(price, 500);
  return Number(capped.toFixed(2));
}

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function describeOrder(order: AlpacaOptionOrder): string {
  return [
    `status=${order.status}`,
    `filled=${order.filled_qty}`,
    order.limit_price ? `limit=${order.limit_price}` : null,
    `submitted=${order.submitted_at}`,
  ]
    .filter(Boolean)
    .join(' | ');
}

async function main() {
  const symbol = process.argv[2]?.toUpperCase() || process.env.MR_TEST_SYMBOL || 'SPY';
  console.log(`🧪 Starting Alpaca option order smoke test for ${symbol}`);

  const contract = await pickTestContract(symbol);
  console.log(`   Using contract ${contract.symbol} (${contract.option_type.toUpperCase()} ${contract.strike_price} exp ${contract.expiration_date})`);

  const clientOrderId = `test-${randomUUID()}`;
  const limitPrice = deriveLimitPrice(contract);
  console.log(`   Using limit price $${limitPrice.toFixed(2)}`);
  const order = await submitOptionOrder({
    symbol: contract.symbol,
    qty: '1',
    side: 'buy',
    type: 'limit',
    limit_price: limitPrice,
    time_in_force: 'day',
    position_effect: 'open',
    client_order_id: clientOrderId,
  });

  console.log(`   Submitted order ${order.id} (${describeOrder(order)})`);

  await sleep(2000);

  const refreshed = await getOptionOrder(order.id);
  console.log(`   Current status: ${describeOrder(refreshed)}`);

  if (refreshed.status !== 'filled') {
    console.log('   Canceling test order...');
    await cancelOptionOrder(order.id);
    const finalStatus = await getOptionOrder(order.id);
    console.log(`   Final status after cancel: ${describeOrder(finalStatus)}`);
  } else {
    console.log('   Order filled immediately; leaving position as-is (paper account).');
  }

  console.log('✅ Alpaca option order test complete');
}

main().catch(err => {
  console.error('❌ Test failed:', err);
  process.exit(1);
});
