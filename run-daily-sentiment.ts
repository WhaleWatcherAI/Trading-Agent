import 'dotenv/config';
import { analyzeDailyMarketSentiment } from '@/lib/sentimentAnalyzer';

async function main() {
  const [, , dateArg] = process.argv;

  try {
    const report = await analyzeDailyMarketSentiment(dateArg);
    console.log(JSON.stringify(report, null, 2));
  } catch (error: unknown) {
    console.error('Failed to generate daily sentiment report:', error);
    process.exitCode = 1;
  }
}

void main();
