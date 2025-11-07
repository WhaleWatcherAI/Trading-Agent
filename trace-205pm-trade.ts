import { SMA } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';
import { getOptionsChain } from './lib/tradier';

const SMA_PERIOD = 9;
const COMMISSION = 1.30;
const SLIPPAGE_PERCENT = 0.5;

async function traceTrade() {
  const symbol = 'TSLA';
  const testDate = '2025-11-04';

  console.log('═══════════════════════════════════════════════════');
  console.log('  TRACING 2:05 PM PUT TRADE');
  console.log('═══════════════════════════════════════════════════\n');

  const history = await getHistoricalData(symbol, '1min', 1, testDate);
  const optionsChain = await getOptionsChain(symbol, undefined, testDate);

  const formatTime = (timestamp: string | number) => {
    const date = new Date(Number(timestamp) * 1000);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true, timeZone: 'America/New_York' });
  };

  // Calculate SMA
  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });

  // Find 2:05 PM entry
  const entryIndex = history.findIndex(bar => formatTime(bar.date) === '02:05 PM');

  if (entryIndex === -1) {
    console.log('❌ Could not find 2:05 PM bar');
    return;
  }

  const entryBar = history[entryIndex];
  const previousBar = history[entryIndex - 1];
  const currentSma = smaValues[entryIndex - SMA_PERIOD];
  const previousSma = smaValues[entryIndex - SMA_PERIOD - 1];

  console.log(`Entry Bar (2:05 PM):`);
  console.log(`   Close: ${entryBar.close}`);
  console.log(`   SMA: ${currentSma.toFixed(2)}`);
  console.log(`   Previous close: ${previousBar.close}, Previous SMA: ${previousSma.toFixed(2)}\n`);

  // Find ITM PUT
  const currentPrice = entryBar.close;
  const putOptions = optionsChain.filter(o => o.type === 'put' && o.strike > currentPrice);
  putOptions.sort((a, b) => a.strike - b.strike);
  const bestOption = putOptions[0];

  if (!bestOption) {
    console.log('❌ No PUT option found');
    return;
  }

  console.log(`Selected PUT:`);
  console.log(`   Strike: ${bestOption.strike}`);
  console.log(`   Premium: $${bestOption.premium}`);
  console.log(`   Delta: ${bestOption.greeks?.delta}`);
  console.log(`   Theta: ${bestOption.greeks?.theta}\n`);

  // Entry price calculation
  let underlyingEntryPrice = entryBar.open < currentSma ? entryBar.open : currentSma;
  const priceDiffFromClose = underlyingEntryPrice - currentPrice;
  const premiumAdjustment = (bestOption.greeks?.delta || 0) * priceDiffFromClose;
  const adjustedPremium = bestOption.premium + premiumAdjustment;
  const entryPrice = Math.max(0.01, adjustedPremium * (1 + SLIPPAGE_PERCENT / 100));

  console.log(`Entry Calculation:`);
  console.log(`   Underlying entry: $${underlyingEntryPrice.toFixed(2)}`);
  console.log(`   Option entry premium: $${entryPrice.toFixed(2)}\n`);

  console.log(`Holding trade, watching for exit signal...\n`);

  // Simulate holding and check each bar
  for (let i = entryIndex + 1; i < Math.min(entryIndex + 20, history.length); i++) {
    const bar = history[i];
    const prevBar = history[i - 1];
    const barSma = smaValues[i - SMA_PERIOD];
    const prevSma = smaValues[i - SMA_PERIOD - 1];
    const barPrice = bar.close;

    // Calculate current option value
    const underlyingMove = barPrice - underlyingEntryPrice;
    const minutesHeld = i - entryIndex;
    const thetaDecay = (bestOption.greeks?.theta || 0) * (minutesHeld / 390);
    const optionMove = underlyingMove * (bestOption.greeks?.delta || 0);
    const currentPremium = Math.max(0.01, entryPrice + optionMove + thetaDecay);
    const exitPriceWithSlippage = currentPremium * (1 - SLIPPAGE_PERCENT / 100);
    const grossPnl = (exitPriceWithSlippage - entryPrice) * 100; // × 100 shares per contract!
    const netPnl = grossPnl - COMMISSION;
    const pnlPercent = (netPnl / (entryPrice * 100)) * 100; // % of total capital invested

    // Check for opposite crossover (bullish = exit PUT)
    const crossesUp = prevBar.close <= prevSma && bar.high > barSma;

    const timeStr = formatTime(bar.date);
    const marker = crossesUp ? '🟢 EXIT' : '';

    console.log(`${timeStr} ${marker}`);
    console.log(`   Price: ${barPrice.toFixed(2)}, SMA: ${barSma.toFixed(2)}`);
    console.log(`   Underlying move: ${underlyingMove.toFixed(2)} (${underlyingMove > 0 ? 'UP ⬆️' : 'DOWN ⬇️'})`);
    console.log(`   Option premium: $${currentPremium.toFixed(2)}`);
    console.log(`   P&L: $${netPnl.toFixed(2)} (${pnlPercent.toFixed(1)}%)\n`);

    if (crossesUp) {
      console.log(`═══════════════════════════════════════════════════`);
      console.log(`TRADE EXITED at ${timeStr}:`);
      console.log(`   Entry: $${entryPrice.toFixed(2)} per share × 100 = $${(entryPrice * 100).toFixed(2)}`);
      console.log(`   Exit: $${exitPriceWithSlippage.toFixed(2)} per share × 100 = $${(exitPriceWithSlippage * 100).toFixed(2)}`);
      console.log(`   Underlying: ${underlyingEntryPrice.toFixed(2)} → ${barPrice.toFixed(2)}`);
      console.log(`   Gross P&L: $${grossPnl.toFixed(2)}`);
      console.log(`   Commission: -$${COMMISSION.toFixed(2)}`);
      console.log(`   Net P&L: $${netPnl.toFixed(2)}`);
      console.log(`   Return: ${pnlPercent.toFixed(1)}%`);
      console.log(`   Minutes held: ${minutesHeld}`);
      console.log(`═══════════════════════════════════════════════════`);
      break;
    }
  }
}

traceTrade()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Trace failed:', error);
    process.exit(1);
  });
