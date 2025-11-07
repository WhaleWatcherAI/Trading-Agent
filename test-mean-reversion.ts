import 'dotenv/config';
import { backtestMeanReversionMultiple, MeanReversionBacktestResult } from './lib/meanReversionBacktester';
import { promises as fs } from 'fs';

/**
 * Test mean reversion strategy on multiple symbols
 */
async function runMeanReversionBacktests() {
  console.log('🚀 Starting Mean Reversion Backtests\n');
  console.log('═'.repeat(80));

  // Symbols to test - SPY only (for ES/MES futures calculation)
  const symbols = ['SPY'];

  // Recent trading dates (adjusted for Tradier sandbox API limits)
  const dates = [
    '2025-10-27',
    '2025-10-28',
    '2025-10-29',
    '2025-10-30',
    '2025-10-31',
  ];

  console.log(`\n📊 Testing symbols: ${symbols.join(', ')}`);
  console.log(`📅 Testing dates: ${dates.join(', ')}\n`);
  console.log('═'.repeat(80));

  // Run backtests
  const results = await backtestMeanReversionMultiple(symbols, dates, 'intraday', 15, 1);

  // Filter to only positive GEX days (days that were actually traded)
  const tradedResults = results.filter(r => r.regime === 'positive_gex' && r.trades.length > 0);
  const skippedResults = results.filter(r => r.regime !== 'positive_gex' || r.trades.length === 0);

  console.log('\n📈 BACKTEST RESULTS SUMMARY');
  console.log('═'.repeat(80));

  // Group by symbol
  const resultsBySymbol = new Map<string, MeanReversionBacktestResult[]>();
  for (const result of tradedResults) {
    if (!resultsBySymbol.has(result.symbol)) {
      resultsBySymbol.set(result.symbol, []);
    }
    resultsBySymbol.get(result.symbol)!.push(result);
  }

  // Print results for each symbol
  for (const [symbol, symbolResults] of resultsBySymbol.entries()) {
    console.log(`\n💰 ${symbol} - ${symbolResults.length} positive GEX trading days`);
    console.log('─'.repeat(80));

    const allTrades = symbolResults.flatMap(r => r.trades);
    const totalTrades = allTrades.length;
    const winningTrades = allTrades.filter(t => (t.stock?.profit ?? 0) > 0);
    const losingTrades = allTrades.filter(t => (t.stock?.profit ?? 0) <= 0);

    const totalProfit = allTrades.reduce((sum, t) => sum + (t.stock?.profit ?? 0), 0);
    const grossProfit = winningTrades.reduce((sum, t) => sum + (t.stock?.profit ?? 0), 0);
    const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + (t.stock?.profit ?? 0), 0));
    const winRate = totalTrades > 0 ? (winningTrades.length / totalTrades) * 100 : 0;
    const avgWin = winningTrades.length > 0 ? grossProfit / winningTrades.length : 0;
    const avgLoss = losingTrades.length > 0 ? grossLoss / losingTrades.length : 0;
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
    const avgDuration = totalTrades > 0 ? allTrades.reduce((sum, t) => sum + t.durationMinutes, 0) / totalTrades : 0;

    console.log(`  Total Trades: ${totalTrades}`);
    console.log(`  Winning Trades: ${winningTrades.length}`);
    console.log(`  Losing Trades: ${losingTrades.length}`);
    console.log(`  Win Rate: ${winRate.toFixed(2)}%`);
    console.log(`  Total Profit: $${totalProfit.toFixed(2)}`);
    console.log(`  Gross Profit: $${grossProfit.toFixed(2)}`);
    console.log(`  Gross Loss: $${grossLoss.toFixed(2)}`);
    console.log(`  Average Win: $${avgWin.toFixed(2)}`);
    console.log(`  Average Loss: $${avgLoss.toFixed(2)}`);
    console.log(`  Profit Factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);
    console.log(`  Average Duration: ${avgDuration.toFixed(0)} minutes`);

    const optionTrades = allTrades.filter(t => t.option.profit !== null);
    if (optionTrades.length > 0) {
      const optionTotalProfit = optionTrades.reduce((sum, t) => sum + (t.option.profit ?? 0), 0);
      const optionTotalCommission = optionTrades.reduce((sum, t) => sum + (t.option.commission ?? 0), 0);
      const optionTotalSlippage = optionTrades.reduce((sum, t) => sum + (t.option.slippage ?? 0), 0);
      const optionTotalGross = optionTotalProfit + optionTotalCommission + optionTotalSlippage;

      console.log(`\n  Options (per contract):`);
      console.log(`    Gross P&L: $${optionTotalGross.toFixed(2)} | Net P&L: $${optionTotalProfit.toFixed(2)}`);
      console.log(`    Commission: $${optionTotalCommission.toFixed(2)} | Slippage: $${optionTotalSlippage.toFixed(2)}`);

      console.log('  Options Leverage Profiles:');
      [50, 100].forEach(level => {
        const net = optionTotalProfit * level;
        const commission = optionTotalCommission * level;
        const slippage = optionTotalSlippage * level;
        const gross = optionTotalGross * level;
        console.log(
          `    ${level}x → Gross $${gross.toFixed(2)} | Net $${net.toFixed(2)} | Commission $${commission.toFixed(2)} | Slippage $${slippage.toFixed(2)}`
        );
      });
    } else {
      console.log(`\n  Options: No option trades executed`);
    }

    // Print trades by exit reason
    const exitReasons = new Map<string, number>();
    allTrades.forEach(t => {
      exitReasons.set(t.exitReason, (exitReasons.get(t.exitReason) || 0) + 1);
    });

    console.log('\n  Exit Breakdown:');
    for (const [reason, count] of exitReasons.entries()) {
      console.log(`    ${reason}: ${count} (${((count / totalTrades) * 100).toFixed(1)}%)`);
    }

    // Print detailed trades for this symbol
    console.log('\n  Detailed Trades:');
    symbolResults.forEach(dayResult => {
      if (dayResult.trades.length > 0) {
        console.log(`\n    ${dayResult.date} (Net GEX: $${(dayResult.netGex / 1_000_000).toFixed(1)}M)`);
        console.log(`    BB Range: $${dayResult.technicals.bbLower?.toFixed(2)} - $${dayResult.technicals.bbMiddle?.toFixed(2)} - $${dayResult.technicals.bbUpper?.toFixed(2)}`);
        console.log(`    Avg RSI: ${dayResult.technicals.avgRSI?.toFixed(1)}`);
        dayResult.trades.forEach(trade => {
          const entryPrice = trade.stock?.entryPrice ?? 0;
          const exitPrice = trade.stock?.exitPrice ?? 0;
          const profit = trade.stock?.profit ?? 0;
          const profitPct = trade.stock?.profitPct ?? 0;
          const entryRSI = trade.entryRSI ?? null;
          const exitRSI = trade.exitRSI ?? null;
          const profitSign = profit >= 0 ? '+' : '';
          const optionSummary = trade.option?.contract
            ? ` | Option ${trade.option.contract} ${trade.option.profit !== null ? `$${(trade.option.profit ?? 0).toFixed(2)}` : 'n/a'}`
            : '';
          const futuresSummary = trade.futures
            ? ` | ES: ${profitSign}$${trade.futures.esProfit.toFixed(2)} | MES: ${profitSign}$${trade.futures.mesProfit.toFixed(2)}`
            : '';
          console.log(
            `      ${trade.direction.toUpperCase()}: Entry $${entryPrice.toFixed(2)} (RSI ${entryRSI?.toFixed(1) ?? 'n/a'}) → Exit $${exitPrice.toFixed(2)} (RSI ${exitRSI?.toFixed(1) ?? 'n/a'}) = ${profitSign}$${profit.toFixed(2)} (${profitSign}${profitPct.toFixed(2)}%) [${trade.exitReason}]${futuresSummary}${optionSummary}`
          );
        });
      }
    });
  }

  // Print skipped days
  if (skippedResults.length > 0) {
    console.log('\n\n⏭️  SKIPPED DAYS (Negative GEX or No Valid Range)');
    console.log('═'.repeat(80));

    const skippedBySymbol = new Map<string, MeanReversionBacktestResult[]>();
    for (const result of skippedResults) {
      if (!skippedBySymbol.has(result.symbol)) {
        skippedBySymbol.set(result.symbol, []);
      }
      skippedBySymbol.get(result.symbol)!.push(result);
    }

    for (const [symbol, skipped] of skippedBySymbol.entries()) {
      console.log(`\n${symbol}:`);
      skipped.forEach(result => {
        console.log(`  ${result.date}: Net GEX $${(result.netGex / 1_000_000).toFixed(1)}M (${result.regime})`);
        if (result.notes.length > 0) {
          result.notes.forEach(note => console.log(`    - ${note}`));
        }
      });
    }
  }

  // Overall summary
  console.log('\n\n📊 OVERALL SUMMARY');
  console.log('═'.repeat(80));

  const allTrades = tradedResults.flatMap(r => r.trades);
  const totalTrades = allTrades.length;
  const winningTrades = allTrades.filter(t => (t.stock?.profit ?? 0) > 0);
  const losingTrades = allTrades.filter(t => (t.stock?.profit ?? 0) <= 0);

  const totalProfit = allTrades.reduce((sum, t) => sum + (t.stock?.profit ?? 0), 0);
  const grossProfit = winningTrades.reduce((sum, t) => sum + (t.stock?.profit ?? 0), 0);
  const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + (t.stock?.profit ?? 0), 0));
  const winRate = totalTrades > 0 ? (winningTrades.length / totalTrades) * 100 : 0;
  const avgWin = winningTrades.length > 0 ? grossProfit / winningTrades.length : 0;
  const avgLoss = losingTrades.length > 0 ? grossLoss / losingTrades.length : 0;
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;

  console.log(`Total Positive GEX Days: ${tradedResults.length} / ${results.length}`);
  console.log(`Total Trades: ${totalTrades}`);
  console.log(`Win Rate: ${winRate.toFixed(2)}%`);
  console.log(`Total Profit: $${totalProfit.toFixed(2)}`);
  console.log(`Profit Factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);
  console.log(`Average Win: $${avgWin.toFixed(2)} vs Average Loss: $${avgLoss.toFixed(2)}`);

  const overallOptionTrades = allTrades.filter(t => t.option.profit !== null);
  if (overallOptionTrades.length > 0) {
    const optionTotalProfit = overallOptionTrades.reduce((sum, t) => sum + (t.option.profit ?? 0), 0);
    const optionTotalCommission = overallOptionTrades.reduce((sum, t) => sum + (t.option.commission ?? 0), 0);
    const optionTotalSlippage = overallOptionTrades.reduce((sum, t) => sum + (t.option.slippage ?? 0), 0);
    const optionTotalGross = optionTotalProfit + optionTotalCommission + optionTotalSlippage;

    console.log(`\nOptions (per contract):`);
    console.log(`  Gross P&L: $${optionTotalGross.toFixed(2)} | Net P&L: $${optionTotalProfit.toFixed(2)}`);
    console.log(`  Commission: $${optionTotalCommission.toFixed(2)} | Slippage: $${optionTotalSlippage.toFixed(2)}`);

    console.log('Options Leverage Profiles:');
    [50, 100].forEach(level => {
      const net = optionTotalProfit * level;
      const commission = optionTotalCommission * level;
      const slippage = optionTotalSlippage * level;
      const gross = optionTotalGross * level;
      console.log(
        `  ${level}x → Gross $${gross.toFixed(2)} | Net $${net.toFixed(2)} | Commission $${commission.toFixed(2)} | Slippage $${slippage.toFixed(2)}`
      );
    });
  } else {
    console.log(`\nOptions: No option trades executed`);
  }

  // Save results to file
  const outputFile = `backtest_mean_reversion_${new Date().toISOString().split('T')[0]}.json`;
  await fs.writeFile(outputFile, JSON.stringify(results, null, 2));
  console.log(`\n💾 Full results saved to ${outputFile}`);

  console.log('\n✅ Backtest Complete!');
}

// Run the backtests
runMeanReversionBacktests().catch(error => {
  console.error('❌ Backtest failed:', error);
  process.exit(1);
});
