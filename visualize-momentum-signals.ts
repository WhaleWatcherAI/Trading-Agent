import { getHistoricalTimesales } from './lib/tradier';
import { SMA } from 'technicalindicators';

function generateTradingDates(startDate: string, endDate: string): string[] {
  const dates: string[] = [];
  const start = new Date(startDate);
  const end = new Date(endDate);

  const current = new Date(start);
  while (current <= end) {
    const dayOfWeek = current.getDay();
    if (dayOfWeek !== 0 && dayOfWeek !== 6) {
      dates.push(current.toISOString().split('T')[0]);
    }
    current.setDate(current.getDate() + 1);
  }

  return dates;
}

async function visualizeMomentumSignals() {
  const symbol = 'SPY';
  const startDate = '2025-09-16';
  const endDate = '2025-11-04';

  console.log('📊 Visualizing Momentum Signals for SPY\n');
  console.log(`Period: ${startDate} to ${endDate}\n`);

  const tradingDates = generateTradingDates(startDate, endDate);
  const bars: Array<{ date: string; close: number }> = [];

  console.log('Fetching daily data...\n');

  for (const date of tradingDates) {
    try {
      const intradayBars = await getHistoricalTimesales(symbol, date, 5);
      if (intradayBars.length > 0) {
        bars.push({
          date,
          close: intradayBars[intradayBars.length - 1].close,
        });
      }
    } catch (error) {
      // Skip
    }
  }

  console.log(`✓ Got ${bars.length} daily bars\n`);

  // Calculate SMAs
  const closes = bars.map(b => b.close);
  const sma9Values = SMA.calculate({ values: closes, period: 9 });
  const sma20Values = SMA.calculate({ values: closes, period: 20 });

  console.log('=' .repeat(100));
  console.log('Date       | Close   | SMA(9)  | SMA(20) | Position | Signal           | Notes');
  console.log('=' .repeat(100));

  let prevSma9 = 0;
  let prevSma20 = 0;

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const sma9Idx = i - (closes.length - sma9Values.length);
    const sma20Idx = i - (closes.length - sma20Values.length);

    const sma9 = sma9Idx >= 0 ? sma9Values[sma9Idx] : null;
    const sma20 = sma20Idx >= 0 ? sma20Values[sma20Idx] : null;

    let position = '     ';
    let signal = '';
    let notes = '';

    if (sma9 && sma20) {
      // Check for crossover
      if (prevSma9 && prevSma20) {
        // Golden Cross
        if (prevSma9 <= prevSma20 && sma9 > sma20) {
          signal = '🟢 GOLDEN CROSS';
          position = 'CALL ';
          notes = 'BUY CALL';
        }
        // Death Cross
        else if (prevSma9 >= prevSma20 && sma9 < sma20) {
          signal = '🔴 DEATH CROSS';
          position = 'PUT  ';
          notes = 'BUY PUT';
        }
        // Holding trend
        else if (sma9 > sma20) {
          position = 'call ';
          notes = 'bullish trend';
        } else if (sma9 < sma20) {
          position = 'put  ';
          notes = 'bearish trend';
        }
      }

      prevSma9 = sma9;
      prevSma20 = sma20;
    } else {
      notes = 'insufficient data for SMA(20)';
    }

    const dateStr = bar.date;
    const closeStr = bar.close.toFixed(2).padStart(7);
    const sma9Str = sma9 ? sma9.toFixed(2).padStart(7) : '   -   ';
    const sma20Str = sma20 ? sma20.toFixed(2).padStart(7) : '   -   ';
    const signalStr = signal.padEnd(16);

    console.log(`${dateStr} | ${closeStr} | ${sma9Str} | ${sma20Str} | ${position} | ${signalStr} | ${notes}`);
  }

  console.log('=' .repeat(100));

  // Summary
  console.log('\n📈 ANALYSIS:');
  console.log(`  Total bars: ${bars.length}`);
  console.log(`  Bars with SMA(9): ${sma9Values.length}`);
  console.log(`  Bars with SMA(20): ${sma20Values.length} (need 20 bars minimum)`);
  console.log(`  Tradeable bars: ${sma20Values.length} (bars where we can check for crossovers)`);
}

visualizeMomentumSignals().catch(console.error);
