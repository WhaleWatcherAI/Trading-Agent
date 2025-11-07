import { backtestMomentumMultiple } from './lib/momentumBacktester';
import * as fs from 'fs';

async function runMomentumBacktests() {
  console.log('🚀 Starting Momentum/Trend Following Backtests');
  console.log('📈 Strategy: SMA(9) x SMA(20) Crossover');
  console.log('🎯 Exit: 1:5 Risk/Reward (Stop -20%, Target +100%)');
  console.log('💡 GEX Filter: DISABLED (sandbox limitation)');
  console.log('\n' + '═'.repeat(80) + '\n');

  const symbols = ['SPY', 'GLD'];

  // Use recent dates only (sandbox limitation - only has ~2 weeks of historical data)
  const startDate = '2025-09-16'; // Start from mid-September
  const endDate = '2025-11-04'; // Up to yesterday

  console.log(`📊 Testing symbols: ${symbols.join(', ')}`);
  console.log(`📅 Period: ${startDate} to ${endDate} (~7 weeks)`);
  console.log('\n' + '═'.repeat(80) + '\n');

  const results = await backtestMomentumMultiple(symbols, startDate, endDate, 'intraday');

  console.log('📈 MOMENTUM BACKTEST RESULTS');
  console.log('═'.repeat(80));

  for (const result of results) {
    console.log(`\n💰 ${result.symbol}`);
    console.log('─'.repeat(80));

    const { summary, trades } = result;

    console.log(`  Total Trades: ${summary.tradeCount}`);
    console.log(`  Win Rate: ${summary.winRate.toFixed(2)}%`);
    console.log(`  Average Hold Time: ${summary.averageDurationDays.toFixed(1)} days`);

    console.log(`\n  📊 OPTIONS P&L:`);
    console.log(`    Total Profit: $${summary.totalProfit.toFixed(2)}`);
    console.log(`    Profit Factor: ${summary.profitFactor === Infinity ? '∞' : summary.profitFactor.toFixed(2)}`);
    console.log(`    Average Win: $${summary.averageWin.toFixed(2)} | Average Loss: $${summary.averageLoss.toFixed(2)}`);
    console.log(`    Total Commission: $${summary.totalCommission.toFixed(2)}`);
    console.log(`    Total Slippage: $${summary.totalSlippage.toFixed(2)}`);

    // Print detailed trades
    console.log('\n  Detailed Trades:');
    trades.forEach(trade => {
      const profitSign = (trade.option.profit || 0) >= 0 ? '+' : '';
      const direction = trade.direction === 'bullish' ? 'CALL' : 'PUT';

      console.log(`\n    ${trade.entryDate} → ${trade.exitDate} (${trade.durationDays} days)`);
      console.log(`    ${direction}: Entry $${trade.entryPrice.toFixed(2)} → Exit $${trade.exitPrice.toFixed(2)}`);
      console.log(`    Entry GEX: $${(trade.entryNetGex / 1_000_000).toFixed(1)}M | Exit GEX: $${(trade.exitNetGex / 1_000_000).toFixed(1)}M`);

      if (trade.option.contract) {
        console.log(
          `    Option: ${trade.option.strike}${trade.option.type === 'call' ? 'C' : 'P'} @$${trade.option.entryPremium?.toFixed(2)} → $${trade.option.exitPremium?.toFixed(2)}`
        );
        console.log(
          `    P&L: ${profitSign}$${trade.option.profit?.toFixed(2)} (${profitSign}${trade.option.profitPct?.toFixed(2)}%) [${trade.exitReason}]`
        );
        console.log(
          `    Costs: Commission $${trade.option.commission?.toFixed(2)} | Slippage $${trade.option.slippage?.toFixed(2)}`
        );
      }
    });

    if (result.notes.length > 0) {
      console.log('\n  Notes:');
      result.notes.forEach(note => console.log(`    - ${note}`));
    }
  }

  // Overall summary
  console.log('\n\n📊 OVERALL SUMMARY');
  console.log('═'.repeat(80));

  const allTrades = results.flatMap(r => r.trades);
  const totalProfit = allTrades.reduce((sum, t) => sum + (t.option.profit || 0), 0);
  const totalCommission = allTrades.reduce((sum, t) => sum + (t.option.commission || 0), 0);
  const totalSlippage = allTrades.reduce((sum, t) => sum + (t.option.slippage || 0), 0);
  const winRate = allTrades.length > 0
    ? (allTrades.filter(t => (t.option.profit || 0) > 0).length / allTrades.length) * 100
    : 0;

  console.log(`Total Trades: ${allTrades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(2)}%`);
  console.log(`Total Profit: $${totalProfit.toFixed(2)}`);
  console.log(`Total Commission: $${totalCommission.toFixed(2)}`);
  console.log(`Total Slippage: $${totalSlippage.toFixed(2)}`);
  console.log(`Net After Costs: $${(totalProfit).toFixed(2)}`);

  // Save results
  const timestamp = new Date().toISOString().split('T')[0];
  const filename = `backtest_momentum_${timestamp}.json`;
  fs.writeFileSync(filename, JSON.stringify(results, null, 2));
  console.log(`\n💾 Full results saved to ${filename}`);

  console.log('\n✅ Momentum Backtest Complete!');
}

runMomentumBacktests().catch(err => {
  console.error('❌ Backtest failed:', err);
  process.exit(1);
});
