import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const symbol = searchParams.get('symbol') || 'SPY';
    const date = searchParams.get('date') || '2025-11-04';

    const fileName = `backtest_sma_sentiment_${symbol}_${date}.json`;
    const filePath = path.join(process.cwd(), fileName);

    console.log(`📂 Loading results from: ${fileName}`);

    try {
      const fileContent = await fs.readFile(filePath, 'utf-8');
      const result = JSON.parse(fileContent);

      console.log(`✅ Successfully loaded results for ${symbol} on ${date}`);
      return NextResponse.json(result);
    } catch (error: any) {
      if (error.code === 'ENOENT') {
        return NextResponse.json(
          { error: `No results found for ${symbol} on ${date}. Please run the backtest first.` },
          { status: 404 }
        );
      }
      throw error;
    }
  } catch (error: any) {
    console.error('Error loading results:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to load results' },
      { status: 500 }
    );
  }
}
