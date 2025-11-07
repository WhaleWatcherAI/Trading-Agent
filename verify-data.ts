import { SMA } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';
import { getOptionsChain } from './lib/tradier';

async function verifyData() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  DATA INTEGRITY CHECK');
  console.log('═══════════════════════════════════════════════════\n');

  const symbol = 'TSLA';
  const testDate = '2025-11-04';

  // Get data
  const history = await getHistoricalData(symbol, '1min', 7, testDate);
  const optionsChain = await getOptionsChain(symbol, undefined, testDate);

  // 1. Check bar timestamps
  console.log('1. CHECKING BAR TIMESTAMPS');
  console.log('─────────────────────────────────────────────────');
  console.log(`Total bars: ${history.length}`);
  console.log(`First 5 bars:`);
  history.slice(0, 5).forEach((bar, i) => {
    const date = new Date(parseInt(bar.date) * 1000);
    console.log(`  Bar ${i}: ${bar.date} → ${date.toISOString()} | Close: $${bar.close.toFixed(2)}`);
  });
  console.log(`Last 5 bars:`);
  history.slice(-5).forEach((bar, i) => {
    const date = new Date(parseInt(bar.date) * 1000);
    console.log(`  Bar ${history.length - 5 + i}: ${bar.date} → ${date.toISOString()} | Close: $${bar.close.toFixed(2)}`);
  });

  // 2. Check options data
  console.log('\n2. CHECKING OPTIONS DATA');
  console.log('─────────────────────────────────────────────────');
  console.log(`Total options: ${optionsChain.length}`);
  const sampleOption = optionsChain.find(o => o.strike === 445 && o.type === 'put');
  if (sampleOption) {
    console.log(`Sample: PUT $445`);
    console.log(`  Premium: $${sampleOption.premium}`);
    console.log(`  Delta: ${sampleOption.greeks?.delta}`);
    console.log(`  Expiration: ${sampleOption.expiration}`);
    console.log(`  Volume: ${sampleOption.volume}, OI: ${sampleOption.openInterest}`);
  }

  // 3. Calculate 9-SMA and find sustained moves
  console.log('\n3. ANALYZING 9-SMA VS PRICE');
  console.log('─────────────────────────────────────────────────');
  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: 9 });

  console.log(`Price range: $${Math.min(...closes).toFixed(2)} - $${Math.max(...closes).toFixed(2)}`);
  console.log(`SMA range: $${Math.min(...smaValues).toFixed(2)} - $${Math.max(...smaValues).toFixed(2)}`);

  // Find periods where price stayed above/below SMA for >10 bars
  console.log('\n4. FINDING SUSTAINED MOVES (>10 bars)');
  console.log('─────────────────────────────────────────────────');

  let aboveCount = 0;
  let belowCount = 0;
  let sustainedMoves = [];
  let currentMove = null;

  for (let i = 9; i < history.length; i++) {
    const close = closes[i];
    const sma = smaValues[i - 9];
    const isAbove = close > sma;
    const isBelow = close < sma;

    if (isAbove) {
      aboveCount++;
      belowCount = 0;

      if (!currentMove || currentMove.direction !== 'above') {
        if (currentMove && currentMove.bars >= 10) {
          sustainedMoves.push(currentMove);
        }
        currentMove = {
          direction: 'above',
          startBar: i,
          startPrice: close,
          startSma: sma,
          bars: 1,
          maxPrice: close,
          minPrice: close,
        };
      } else {
        currentMove.bars++;
        currentMove.maxPrice = Math.max(currentMove.maxPrice, close);
        currentMove.minPrice = Math.min(currentMove.minPrice, close);
      }
    } else if (isBelow) {
      belowCount++;
      aboveCount = 0;

      if (!currentMove || currentMove.direction !== 'below') {
        if (currentMove && currentMove.bars >= 10) {
          sustainedMoves.push(currentMove);
        }
        currentMove = {
          direction: 'below',
          startBar: i,
          startPrice: close,
          startSma: sma,
          bars: 1,
          maxPrice: close,
          minPrice: close,
        };
      } else {
        currentMove.bars++;
        currentMove.maxPrice = Math.max(currentMove.maxPrice, close);
        currentMove.minPrice = Math.min(currentMove.minPrice, close);
      }
    }
  }

  // Add last move if it qualifies
  if (currentMove && currentMove.bars >= 10) {
    sustainedMoves.push(currentMove);
  }

  console.log(`Found ${sustainedMoves.length} sustained moves (>10 bars):\n`);
  sustainedMoves.forEach((move, idx) => {
    const moveSize = move.direction === 'above'
      ? move.maxPrice - move.startSma
      : move.startSma - move.minPrice;

    console.log(`Move #${idx + 1}: ${move.bars} bars ${move.direction.toUpperCase()} SMA`);
    console.log(`  Start: Bar ${move.startBar}, Price: $${move.startPrice.toFixed(2)}, SMA: $${move.startSma.toFixed(2)}`);
    console.log(`  Range: $${move.minPrice.toFixed(2)} - $${move.maxPrice.toFixed(2)}`);
    console.log(`  Max favorable move: $${moveSize.toFixed(2)}`);

    // Estimate what a call/put would have made
    const optionType = move.direction === 'above' ? 'call' : 'put';
    const estimatedDelta = 0.5; // rough estimate
    const optionGain = moveSize * estimatedDelta;
    console.log(`  Est. option gain (50% delta): $${optionGain.toFixed(2)}`);
    console.log(`  After $1.30 commission: $${(optionGain - 1.30).toFixed(2)}`);
    console.log('');
  });

  console.log('\n5. SUMMARY');
  console.log('─────────────────────────────────────────────────');
  console.log(`Total bars: ${history.length}`);
  console.log(`Sustained moves found: ${sustainedMoves.length}`);

  const profitableMoves = sustainedMoves.filter(m => {
    const moveSize = m.direction === 'above' ? m.maxPrice - m.startSma : m.startSma - m.minPrice;
    return (moveSize * 0.5) > 1.30; // Would beat commission
  });

  console.log(`Potentially profitable moves: ${profitableMoves.length}`);

  if (profitableMoves.length > 0) {
    console.log('\n⚠️  ISSUE FOUND: There WERE profitable moves available!');
    console.log('The backtest should have caught at least some of these.');
    console.log('Problem is likely in exit logic or option price calculation.');
  } else {
    console.log('\n✓ Data shows no sustained moves large enough to beat $1.30 commission');
  }

  console.log('\n═══════════════════════════════════════════════════\n');
}

verifyData()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Verification failed:', error);
    process.exit(1);
  });
