import 'dotenv/config';
import { getHistoricalTimesales } from './lib/tradier';

async function debugNov6() {
  console.log('🔍 Debugging November 6, 2025 data\n');

  try {
    const bars = await getHistoricalTimesales('SPY', '2025-11-06', 1);
    console.log(`✅ Found ${bars.length} 1-minute bars for SPY on 2025-11-06`);

    if (bars.length > 0) {
      console.log('\nFirst 5 bars:');
      bars.slice(0, 5).forEach(bar => {
        console.log(`  ${bar.time} - O:${bar.open} H:${bar.high} L:${bar.low} C:${bar.close} V:${bar.volume}`);
      });

      console.log('\nLast 5 bars:');
      bars.slice(-5).forEach(bar => {
        console.log(`  ${bar.time} - O:${bar.open} H:${bar.high} L:${bar.low} C:${bar.close} V:${bar.volume}`);
      });
    } else {
      console.log('⚠️  No bar data available for this date');
    }
  } catch (error: any) {
    console.error('❌ Error fetching data:', error.message);
  }

  // Also try 15-minute bars
  try {
    const bars15 = await getHistoricalTimesales('SPY', '2025-11-06', 15);
    console.log(`\n✅ Found ${bars15.length} 15-minute bars for SPY on 2025-11-06`);

    if (bars15.length > 0) {
      console.log('\nAll 15-minute bars:');
      bars15.forEach(bar => {
        console.log(`  ${bar.time} - O:${bar.open} H:${bar.high} L:${bar.low} C:${bar.close} V:${bar.volume}`);
      });
    }
  } catch (error: any) {
    console.error('❌ Error fetching 15-min data:', error.message);
  }
}

debugNov6().catch(console.error);
