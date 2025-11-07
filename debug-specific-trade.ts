import { SMA } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';
import { getOptionsChain } from './lib/tradier';

const SMA_PERIOD = 9;
const COMMISSION = 1.30;
const SLIPPAGE_PERCENT = 0.5;

async function debugTrade() {
  const symbol = 'TSLA';
  const testDate = '2025-11-04'; // Today

  console.log('═══════════════════════════════════════════════════');
  console.log('  DEBUGGING SPECIFIC TRADE');
  console.log('  Expected: PUT opened at 2:05 PM (450.52)');
  console.log('  Expected: PUT closed at 2:14 PM (448.37)');
  console.log('═══════════════════════════════════════════════════\n');

  const history = await getHistoricalData(symbol, '1min', 1, testDate);
  const optionsChain = await getOptionsChain(symbol, undefined, testDate);

  console.log(`Loaded ${history.length} bars of 1-minute data`);
  console.log(`Loaded ${optionsChain.length} options\n`);

  // Convert timestamps to dates and show sample
  const formatTime = (timestamp: string | number) => {
    const date = new Date(Number(timestamp) * 1000);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true, timeZone: 'America/New_York' });
  };

  console.log('First 10 bars:');
  history.slice(0, 10).forEach((bar, i) => {
    console.log(`  ${i}: ${formatTime(bar.date)} - Close: ${bar.close}`);
  });

  // Find the 2:05 PM bar (14:05 EST)
  const targetBar = history.find(bar => {
    const time = formatTime(bar.date);
    return time.includes('2:05') || time.includes('02:05') || time.includes('14:05');
  });

  if (!targetBar) {
    console.log('\n❌ Could not find 2:05 PM bar');
    return;
  }

  console.log(`\n✅ Found 2:05 PM bar:`);
  console.log(`   Time: ${formatTime(targetBar.date)}`);
  console.log(`   Open: ${targetBar.open}`);
  console.log(`   High: ${targetBar.high}`);
  console.log(`   Low: ${targetBar.low}`);
  console.log(`   Close: ${targetBar.close}\n`);

  // Calculate SMA
  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });

  // Find index of 2:05 bar
  const targetIndex = history.findIndex(bar => {
    const time = formatTime(bar.date);
    return time.includes('2:05') || time.includes('02:05') || time.includes('14:05');
  });

  if (targetIndex < SMA_PERIOD) {
    console.log('❌ Not enough data for SMA at 2:05');
    return;
  }

  const currentBar = history[targetIndex];
  const previousBar = history[targetIndex - 1];
  const currentSma = smaValues[targetIndex - SMA_PERIOD];
  const previousSma = smaValues[targetIndex - SMA_PERIOD - 1];

  console.log(`SMA Analysis at 2:05 PM:`);
  console.log(`   Current SMA: ${currentSma}`);
  console.log(`   Previous SMA: ${previousSma}`);
  console.log(`   Previous Close: ${previousBar.close}`);
  console.log(`   Current High: ${currentBar.high}`);
  console.log(`   Current Low: ${currentBar.low}\n`);

  // Check for bearish crossover (PUT entry)
  const crossesDown = previousBar.close >= previousSma && currentBar.low < currentSma;

  console.log(`Crossover Check:`);
  console.log(`   Previous close >= previous SMA? ${previousBar.close} >= ${previousSma} = ${previousBar.close >= previousSma}`);
  console.log(`   Current low < current SMA? ${currentBar.low} < ${currentSma} = ${currentBar.low < currentSma}`);
  console.log(`   ✅ Bearish Crossover (PUT signal): ${crossesDown}\n`);

  if (!crossesDown) {
    console.log('❌ No bearish crossover detected at 2:05 PM');
    return;
  }

  // Find ITM PUT option
  const currentPrice = currentBar.close;
  const putOptions = optionsChain.filter(o => {
    if (o.type !== 'put') return false;
    return o.strike > currentPrice; // ITM for puts
  });

  putOptions.sort((a, b) => a.strike - b.strike); // Closest ITM
  const bestOption = putOptions[0];

  if (!bestOption) {
    console.log('❌ No suitable PUT option found');
    return;
  }

  console.log(`Selected PUT Option:`);
  console.log(`   Strike: ${bestOption.strike}`);
  console.log(`   Premium (at close): $${bestOption.premium}`);
  console.log(`   Delta: ${bestOption.greeks?.delta}`);
  console.log(`   Theta: ${bestOption.greeks?.theta}\n`);

  // Calculate entry price (entry at SMA level)
  let underlyingEntryPrice: number;
  if (currentBar.open < currentSma) {
    underlyingEntryPrice = currentBar.open;
  } else {
    underlyingEntryPrice = currentSma;
  }

  console.log(`Entry Calculation:`);
  console.log(`   Bar open: ${currentBar.open}`);
  console.log(`   SMA: ${currentSma}`);
  console.log(`   Underlying entry price: ${underlyingEntryPrice}\n`);

  // Adjust premium for entry at different price
  const priceDiffFromClose = underlyingEntryPrice - currentPrice;
  const premiumAdjustment = bestOption.greeks?.delta ? priceDiffFromClose * bestOption.greeks.delta : 0;
  const adjustedPremium = bestOption.premium + premiumAdjustment;
  const entryPrice = adjustedPremium * (1 + SLIPPAGE_PERCENT / 100);

  console.log(`Premium Adjustment:`);
  console.log(`   Price diff: ${underlyingEntryPrice} - ${currentPrice} = ${priceDiffFromClose}`);
  console.log(`   Delta: ${bestOption.greeks?.delta}`);
  console.log(`   Adjustment: ${priceDiffFromClose} × ${bestOption.greeks?.delta} = ${premiumAdjustment}`);
  console.log(`   Base premium: $${bestOption.premium}`);
  console.log(`   Adjusted premium: $${bestOption.premium} + ${premiumAdjustment} = $${adjustedPremium}`);
  console.log(`   With slippage (+${SLIPPAGE_PERCENT}%): $${entryPrice.toFixed(2)}\n`);

  // Now simulate holding until 2:14 PM
  const exitBar = history.find(bar => {
    const time = formatTime(bar.date);
    return time.includes('2:14') || time.includes('02:14') || time.includes('14:14');
  });

  if (!exitBar) {
    console.log('❌ Could not find 2:14 PM bar');
    return;
  }

  const exitIndex = history.findIndex(bar => {
    const time = formatTime(bar.date);
    return time.includes('2:14') || time.includes('02:14') || time.includes('14:14');
  });
  const exitPrice_underlying = exitBar.close;

  console.log(`\n═══════════════════════════════════════════════════`);
  console.log(`Exit at 2:14 PM:`);
  console.log(`   Time: ${formatTime(exitBar.date)}`);
  console.log(`   Close: ${exitBar.close}\n`);

  // Calculate P&L
  const underlyingMove = exitPrice_underlying - underlyingEntryPrice;
  console.log(`Underlying Move:`);
  console.log(`   Entry: $${underlyingEntryPrice}`);
  console.log(`   Exit: $${exitPrice_underlying}`);
  console.log(`   Move: ${underlyingMove.toFixed(2)} (${underlyingMove > 0 ? 'UP' : 'DOWN'})\n`);

  // Calculate option price at exit
  const minutesHeld = exitIndex - targetIndex;
  const thetaDecay = (bestOption.greeks?.theta || 0) * (minutesHeld / 390); // 390 minutes in trading day
  const optionMove = underlyingMove * (bestOption.greeks?.delta || 0);
  const estimatedExitPremium = Math.max(0.01, entryPrice + optionMove + thetaDecay);
  const exitPriceWithSlippage = estimatedExitPremium * (1 - SLIPPAGE_PERCENT / 100);

  console.log(`Option P&L Calculation:`);
  console.log(`   Minutes held: ${minutesHeld}`);
  console.log(`   Theta decay: ${thetaDecay.toFixed(4)} (${(bestOption.greeks?.theta || 0)} × ${minutesHeld}/390)`);
  console.log(`   Option move from underlying: ${underlyingMove.toFixed(2)} × ${bestOption.greeks?.delta} = ${optionMove.toFixed(4)}`);
  console.log(`   Entry premium: $${entryPrice.toFixed(2)}`);
  console.log(`   Est exit premium: $${entryPrice.toFixed(2)} + ${optionMove.toFixed(4)} + ${thetaDecay.toFixed(4)} = $${estimatedExitPremium.toFixed(2)}`);
  console.log(`   With exit slippage (-${SLIPPAGE_PERCENT}%): $${exitPriceWithSlippage.toFixed(2)}\n`);

  const grossPnl = exitPriceWithSlippage - entryPrice;
  const netPnl = grossPnl - COMMISSION;

  console.log(`═══════════════════════════════════════════════════`);
  console.log(`FINAL P&L:`);
  console.log(`   Entry: $${entryPrice.toFixed(2)}`);
  console.log(`   Exit: $${exitPriceWithSlippage.toFixed(2)}`);
  console.log(`   Gross P&L: $${grossPnl.toFixed(2)}`);
  console.log(`   Commission: -$${COMMISSION.toFixed(2)}`);
  console.log(`   Net P&L: $${netPnl.toFixed(2)}`);
  console.log(`   Return: ${((netPnl / entryPrice) * 100).toFixed(1)}%`);
  console.log(`═══════════════════════════════════════════════════\n`);

  console.log(`Expected vs Actual:`);
  console.log(`   Expected underlying move: 450.52 → 448.37 = -$2.15`);
  console.log(`   Actual underlying move: ${underlyingEntryPrice.toFixed(2)} → ${exitPrice_underlying.toFixed(2)} = ${underlyingMove.toFixed(2)}`);
  console.log(`   PUT should profit from DOWN move: ${underlyingMove < 0 ? '✅ YES' : '❌ NO'}`);
  console.log(`   Backtest shows profit: ${netPnl > 0 ? '✅ YES' : '❌ NO ($' + netPnl.toFixed(2) + ')'}`);
}

debugTrade()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Debug failed:', error);
    process.exit(1);
  });
