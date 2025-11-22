import 'dotenv/config';
import { analyzeNewsImpactMultiAsset, filterDailyNews } from './lib/sentimentAnalyzer';
import { getNewsHeadlines } from './lib/unusualwhales';

async function main() {
  try {
    const daysArg = Number(process.argv[2]);
    const windowDays = Number.isFinite(daysArg) && daysArg > 0 ? daysArg : 3;
    const result = await analyzeNewsImpactMultiAsset(undefined, windowDays);

    console.log('Market Score:', result.marketScore.toFixed(1));
    console.log('Market Sentiment:', result.marketSentiment);
    console.log('News Window (days):', result.newsWindowDays);
    console.log('Total News Analyzed:', result.totalNewsAnalyzed);
    console.log('Article Impacts:', result.articleImpacts.length);

    // Reconstruct the selected headline set (same logic as analyzeNewsImpactMultiAsset)
    const allNews = await getNewsHeadlines();
    const windowedNews = filterDailyNews(allNews, undefined, result.newsWindowDays);

    const sortedNews = [...windowedNews].sort((a, b) => {
      if (b.importance !== a.importance) {
        return b.importance - a.importance;
      }

      return b.timestamp.getTime() - a.timestamp.getTime();
    });

    const MAX_HEADLINES = 40;
    const selectedNews = sortedNews.slice(0, MAX_HEADLINES);

    const topSymbols = result.symbolRatings.slice(0, 15);

    console.log('\nTop Symbols (with reasoning and headlines):');
    topSymbols.forEach((s) => {
      console.log('\n----------------------------------------');
      console.log(
        `${s.symbol} (${s.assetClass}) -> Score: ${s.score.toFixed(1)} (${s.sentiment})`,
      );
      console.log(`Reasoning: ${s.reasoning}`);

      const headlineLines: string[] = [];
      s.supportingHeadlines.slice(0, 3).forEach((idx) => {
        if (idx >= 1 && idx <= selectedNews.length) {
          const item = selectedNews[idx - 1];
          headlineLines.push(`#${idx}: ${item.title}`);
        }
      });

      if (headlineLines.length) {
        console.log('Headlines:');
        headlineLines.forEach((line) => console.log(`  - ${line}`));
      } else if (s.supportingHeadlines.length) {
        console.log('Headlines: (indices only)', s.supportingHeadlines);
      } else {
        console.log('Headlines: (none attached)');
      }
    });
  } catch (error: any) {
    console.error('Error running news impact analysis:', error.message || error);
    process.exitCode = 1;
  }
}

void main();
