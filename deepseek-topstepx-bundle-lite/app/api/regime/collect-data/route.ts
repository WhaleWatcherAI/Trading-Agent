import { NextRequest, NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import { getOptionsChain, getStockPrice } from '@/lib/tradier';
import { getVolatilityStats, getWhaleFlowAlerts } from '@/lib/unusualwhales';
import { analyzeVolatilityRegime } from '@/lib/regimeAgent';

/**
 * Collect all market data needed for regime backtesting
 * - Options chains with greeks (from Tradier)
 * - Whale flow alerts (from Unusual Whales)
 * - IV stats (from Unusual Whales)
 * - Stock quotes
 * - Regime analysis results
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbols = ['SPY', 'QQQ'], date } = body;

    const targetDate = date || new Date().toISOString().split('T')[0];

    console.log(`📊 Collecting market data for ${targetDate}...`);
    console.log(`   Symbols: ${symbols.join(', ')}`);

    const collectedData: any = {
      date: targetDate,
      symbols: symbols,
      optionsChains: {},
      quotes: {},
      volatilityStats: [],
      whaleFlow: [],
      collectedAt: new Date().toISOString(),
    };

    // 1. Fetch options chains with greeks for each symbol
    console.log('Fetching options chains...');
    for (const symbol of symbols) {
      try {
        const chain = await getOptionsChain(symbol);
        collectedData.optionsChains[symbol] = chain;
        console.log(`  ✅ ${symbol}: ${chain.length} options`);
      } catch (error: any) {
        console.error(`  ❌ ${symbol} options chain failed:`, error.message);
        collectedData.optionsChains[symbol] = [];
      }
    }

    // 2. Fetch stock quotes
    console.log('Fetching stock quotes...');
    for (const symbol of symbols) {
      try {
        const quote = await getStockPrice(symbol);
        collectedData.quotes[symbol] = quote;
        console.log(`  ✅ ${symbol}: $${quote.price}`);
      } catch (error: any) {
        console.error(`  ❌ ${symbol} quote failed:`, error.message);
      }
    }

    // 3. Fetch volatility stats
    console.log('Fetching volatility stats...');
    try {
      const stats = await getVolatilityStats(undefined, symbols);
      collectedData.volatilityStats = stats;
      console.log(`  ✅ Got stats for ${stats.length} symbols`);
    } catch (error: any) {
      console.error(`  ❌ Volatility stats failed:`, error.message);
    }

    // 4. Fetch whale flow alerts
    console.log('Fetching whale flow...');
    try {
      const whaleFlow = await getWhaleFlowAlerts({
        symbols,
        lookbackMinutes: 390, // Full trading day
        limit: 1000,
      });
      collectedData.whaleFlow = whaleFlow;
      console.log(`  ✅ Got ${whaleFlow.length} whale alerts`);
    } catch (error: any) {
      console.error(`  ❌ Whale flow failed:`, error.message);
    }

    // 5. Run regime analysis
    console.log('Running regime analysis...');
    try {
      const modes = ['scalp', 'swing', 'leaps'] as const;
      collectedData.regimeAnalysis = {};

      for (const mode of modes) {
        const analysis = await analyzeVolatilityRegime({ symbols, mode });
        collectedData.regimeAnalysis[mode] = analysis;
        const passed = analysis.universe.filter(u => u.passes).length;
        console.log(`  ✅ ${mode}: ${passed}/${symbols.length} passed Stage 1`);
      }
    } catch (error: any) {
      console.error(`  ❌ Regime analysis failed:`, error.message);
    }

    // 6. Save to file
    const dataDir = path.join(process.cwd(), 'data');
    await fs.mkdir(dataDir, { recursive: true });

    const filePath = path.join(dataDir, `${targetDate}.json`);
    await fs.writeFile(filePath, JSON.stringify(collectedData, null, 2));

    console.log(`✅ Data saved to ${filePath}`);

    return NextResponse.json({
      success: true,
      date: targetDate,
      filePath,
      summary: {
        symbols: symbols.length,
        optionsContracts: Object.values(collectedData.optionsChains).reduce(
          (sum: number, chain: any) => sum + chain.length,
          0,
        ),
        volatilityStats: collectedData.volatilityStats.length,
        whaleAlerts: collectedData.whaleFlow.length,
      },
    });
  } catch (error: any) {
    console.error('Data collection error:', error);
    return NextResponse.json(
      {
        error: 'Failed to collect market data',
        details: error?.message || 'Unknown error',
      },
      { status: 500 },
    );
  }
}
