import { backtestMeanReversionMultiple } from './lib/meanReversionBacktester';
import * as fs from 'fs';

async function runMeanReversionBacktests() {
  console.log('🚀 Starting Mean Reversion Backtests (Stock + Options)');
  console.log('\n' + '═'.repeat(80) + '\n');

  const symbols = ['SPY', 'GLD', 'QQQ', 'BAC'];
  const dates = [
    '2025-10-27',
    '2025-10-28',
    '2025-10-29',
    '2025-10-30',
    '2025-10-31',
  ];

  console.log(`📊 Testing symbols: ${symbols.join(', ')}`);
  console.log(`📅 Testing dates: ${dates.join(', ')}`);
  console.log('\n' + '═'.repeat(80) + '\n');

  const results = await backtestMeanReversionMultiple(symbols, dates, 'intraday');

  // Group results by symbol
  const bySymbol = new Map<string, typeof results>();
  results.forEach(result => {
    if (!bySymbol.has(result.symbol)) {
      bySymbol.set(result.symbol, []);
    }
    bySymbol.get(result.symbol)!.push(result);
  });

  console.log('📈 BACKTEST RESULTS SUMMARY');
  console.log('═'.repeat(80));

  // Print results by symbol
  for (const [symbol, symbolResults] of bySymbol.entries()) {
    const positiveDays = symbolResults.filter(r => r.regime === 'positive_gex').length;

    console.log(`\n💰 ${symbol} - ${positiveDays} positive GEX trading days`);
    console.log('─'.repeat(80));

    const allTrades = symbolResults.flatMap(r => r.trades);
    const totalTrades = allTrades.length;
    const winningTrades = allTrades.filter(t => t.stock.profit > 0);
    const winRate = totalTrades > 0 ? (winningTrades.length / totalTrades) * 100 : 0;

    // Stock stats
    const stockTotalProfit = allTrades.reduce((sum, t) => sum + t.stock.profit, 0);
    const stockGrossProfit = winningTrades.reduce((sum, t) => sum + t.stock.profit, 0);
    const stockGrossLoss = Math.abs(allTrades.filter(t => t.stock.profit <= 0).reduce((sum, t) => sum + t.stock.profit, 0));
    const stockAvgWin = winningTrades.length > 0 ? stockGrossProfit / winningTrades.length : 0;
    const stockAvgLoss = (totalTrades - winningTrades.length) > 0 ? stockGrossLoss / (totalTrades - winningTrades.length) : 0;
    const stockProfitFactor = stockGrossLoss > 0 ? stockGrossProfit / stockGrossLoss : stockGrossProfit > 0 ? Infinity : 0;

    // Option stats
    const optionTrades = allTrades.filter(t => t.option.profit !== null);
    const optionWinningTrades = optionTrades.filter(t => t.option.profit! > 0);
    const optionTotalProfit = optionTrades.reduce((sum, t) => sum + (t.option.profit || 0), 0);
    const optionGrossProfit = optionWinningTrades.reduce((sum, t) => sum + t.option.profit!, 0);
    const optionGrossLoss = Math.abs(optionTrades.filter(t => t.option.profit! <= 0).reduce((sum, t) => sum + t.option.profit!, 0));
    const optionAvgWin = optionWinningTrades.length > 0 ? optionGrossProfit / optionWinningTrades.length : 0;
    const optionAvgLoss = (optionTrades.length - optionWinningTrades.length) > 0 ? optionGrossLoss / (optionTrades.length - optionWinningTrades.length) : 0;
    const optionProfitFactor = optionGrossLoss > 0 ? optionGrossProfit / optionGrossLoss : optionGrossProfit > 0 ? Infinity : 0;

    const avgDuration = totalTrades > 0 ? allTrades.reduce((sum, t) => sum + t.durationMinutes, 0) / totalTrades : 0;

    console.log(`  Total Trades: ${totalTrades}`);
    console.log(`  Win Rate: ${winRate.toFixed(2)}%`);
    console.log(`  Average Duration: ${avgDuration.toFixed(0)} minutes`);

    console.log(`\n  📈 STOCK P&L:`);
    console.log(`    Total Profit: $${stockTotalProfit.toFixed(2)}`);
    console.log(`    Profit Factor: ${stockProfitFactor === Infinity ? '∞' : stockProfitFactor.toFixed(2)}`);
    console.log(`    Average Win: $${stockAvgWin.toFixed(2)} | Average Loss: $${stockAvgLoss.toFixed(2)}`);

    if (optionTrades.length > 0) {
      console.log(`\n  📊 OPTIONS P&L: (${optionTrades.length} contracts traded)`);
      console.log(`    Total Profit: $${optionTotalProfit.toFixed(2)}`);
      console.log(`    Profit Factor: ${optionProfitFactor === Infinity ? '∞' : optionProfitFactor.toFixed(2)}`);
      console.log(`    Average Win: $${optionAvgWin.toFixed(2)} | Average Loss: $${optionAvgLoss.toFixed(2)}`);
      console.log(`    💵 Leverage: ${(optionTotalProfit / stockTotalProfit).toFixed(2)}x`);
    } else {
      console.log(`\n  📊 OPTIONS P&L: No options data available`);
    }

    // Print detailed trades
    console.log('\n  Detailed Trades:');
    symbolResults.forEach(dayResult => {
      if (dayResult.trades.length > 0) {
        console.log(`\n    ${dayResult.date} (Net GEX: $${(dayResult.netGex / 1_000_000).toFixed(1)}M)`);
        console.log(`    BB Range: $${dayResult.technicals.bbLower?.toFixed(2)} - $${dayResult.technicals.bbMiddle?.toFixed(2)} - $${dayResult.technicals.bbUpper?.toFixed(2)}`);

        dayResult.trades.forEach(trade => {
          const stockSign = trade.stock.profit >= 0 ? '+' : '';
          const optionSign = (trade.option.profit || 0) >= 0 ? '+' : '';

          if (trade.option.contract) {
            // Both stock and option
            console.log(
              `      ${trade.direction.toUpperCase()}: $${trade.stock.entryPrice.toFixed(2)} → $${trade.stock.exitPrice.toFixed(2)}`
            );
            console.log(
              `        📈 Stock: ${stockSign}$${trade.stock.profit.toFixed(2)} (${stockSign}${trade.stock.profitPct.toFixed(2)}%)`
            );
            console.log(
              `        📊 Option: ${trade.option.strike}${trade.direction === 'long' ? 'C' : 'P'} @$${trade.option.entryPremium?.toFixed(2)} → $${trade.option.exitPremium?.toFixed(2)} = ${optionSign}$${trade.option.profit?.toFixed(2)} (${optionSign}${trade.option.profitPct?.toFixed(2)}%) [${trade.exitReason}]`
            );
            console.log(
              `           Commission: $${trade.option.commission?.toFixed(2)} | Slippage: $${trade.option.slippage?.toFixed(2)}`
            );
          } else {
            // Stock only
            console.log(
              `      ${trade.direction.toUpperCase()}: $${trade.stock.entryPrice.toFixed(2)} → $${trade.stock.exitPrice.toFixed(2)} = ${stockSign}$${trade.stock.profit.toFixed(2)} (${stockSign}${trade.stock.profitPct.toFixed(2)}%) [${trade.exitReason}]`
            );
          }
        });
      }
    });
  }

  // Overall summary
  console.log('\n\n📊 OVERALL SUMMARY');
  console.log('═'.repeat(80));

  const allTrades = results.flatMap(r => r.trades);
  const stockTotalProfit = allTrades.reduce((sum, t) => sum + t.stock.profit, 0);
  const optionTrades = allTrades.filter(t => t.option.profit !== null);
  const optionTotalProfit = optionTrades.reduce((sum, t) => sum + (t.option.profit || 0), 0);
  const winRate = allTrades.length > 0 ? (allTrades.filter(t => t.stock.profit > 0).length / allTrades.length) * 100 : 0;

  console.log(`Total Trades: ${allTrades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(2)}%`);
  console.log(`📈 Stock Total P&L: $${stockTotalProfit.toFixed(2)}`);

  if (optionTrades.length > 0) {
    console.log(`📊 Options Total P&L: $${optionTotalProfit.toFixed(2)} (${optionTrades.length} contracts)`);
    console.log(`💵 Options Leverage: ${(optionTotalProfit / stockTotalProfit).toFixed(2)}x`);
  }

  // Save results
  const timestamp = new Date().toISOString().split('T')[0];
  const filename = `backtest_mean_reversion_dual_${timestamp}.json`;
  fs.writeFileSync(filename, JSON.stringify(results, null, 2));
  console.log(`\n💾 Full results saved to ${filename}`);

  console.log('\n✅ Backtest Complete!');
}

runMeanReversionBacktests().catch(err => {
  console.error('❌ Backtest failed:', err);
  process.exit(1);
});
