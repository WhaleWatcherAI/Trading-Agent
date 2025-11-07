import { getHistoricalData } from './lib/technicals';
import { SMA } from 'technicalindicators';

async function findTrade() {
  const symbol = 'TSLA';
  const testDate = '2025-11-04'; // Today

  const history = await getHistoricalData(symbol, '1min', 1, testDate);

  const formatTime = (timestamp: string | number) => {
    const date = new Date(Number(timestamp) * 1000);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true, timeZone: 'America/New_York' });
  };

  console.log(`Total bars: ${history.length}\n`);

  // Calculate SMA
  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: 9 });

  console.log(`Looking for PUT signals (bearish crossovers) around 2:05 PM (±5 min) near $450...\n`);

  // Look for bars between 2:00 PM and 2:10 PM
  const pmBars = history.filter((bar, i) => {
    if (i < 9) return false; // Need SMA data

    const time = formatTime(bar.date);
    const isPM = time.includes('PM');
    const hour = new Date(Number(bar.date) * 1000).getHours();
    const minute = new Date(Number(bar.date) * 1000).getMinutes();

    // Between 2:00 PM and 2:15 PM (14:00-14:15)
    const inTimeRange = (hour === 14 || hour === 12 || hour === 13) && (minute >= 0 && minute <= 20);

    // Price near 450-452
    const inPriceRange = bar.close >= 449 && bar.close <= 453;

    return inTimeRange && inPriceRange;
  });

  console.log(`Found ${pmBars.length} bars in time/price range:\n`);

  pmBars.forEach((bar, idx) => {
    const i = history.indexOf(bar);
    const previousBar = history[i - 1];
    const currentSma = smaValues[i - 9];
    const previousSma = smaValues[i - 10];

    const crossesDown = previousBar.close >= previousSma && bar.low < currentSma;

    if (crossesDown) {
      console.log(`🔴 SELL SIGNAL at ${formatTime(bar.date)}`);
      console.log(`   Close: ${bar.close}, SMA: ${currentSma.toFixed(2)}`);
      console.log(`   Prev Close: ${previousBar.close}, Prev SMA: ${previousSma.toFixed(2)}\n`);
    } else {
      console.log(`   ${formatTime(bar.date)}: C=${bar.close.toFixed(2)}, SMA=${currentSma.toFixed(2)} (no signal)`);
    }
  });

  console.log(`\n\nShowing ALL bearish crossovers today:\n`);

  for (let i = 9; i < history.length; i++) {
    const bar = history[i];
    const previousBar = history[i - 1];
    const currentSma = smaValues[i - 9];
    const previousSma = smaValues[i - 10];

    const crossesDown = previousBar.close >= previousSma && bar.low < currentSma;

    if (crossesDown) {
      console.log(`🔴 ${formatTime(bar.date)}: Entry=${bar.close.toFixed(2)}, SMA=${currentSma.toFixed(2)}`);
    }
  }
}

findTrade()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Search failed:', error);
    process.exit(1);
  });
