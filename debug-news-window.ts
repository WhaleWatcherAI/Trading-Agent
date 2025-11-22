import 'dotenv/config';
import { getNewsHeadlines } from './lib/unusualwhales';
import { filterDailyNews } from './lib/sentimentAnalyzer';

async function main() {
  const windowDays = 3;
  const allNews = await getNewsHeadlines();
  const windowed = filterDailyNews(allNews, undefined, windowDays);

  const sorted = [...windowed].sort((a, b) => {
    if (b.importance !== a.importance) {
      return b.importance - a.importance;
    }

    return b.timestamp.getTime() - a.timestamp.getTime();
  });

  const MAX_HEADLINES = 40;
  const selected = sorted.slice(0, MAX_HEADLINES);

  console.log(`Selected headlines over last ${windowDays} day(s): ${selected.length}`);
  selected.forEach((item, idx) => {
    console.log(
      `#${idx + 1} ${item.timestamp.toISOString()} - ${item.title}`,
    );
  });
}

void main();

