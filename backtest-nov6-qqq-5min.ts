import 'dotenv/config';
import { backtestMeanReversion } from './lib/meanReversionBacktester';

async function runNov6QQQBacktest() {
  console.log('🚀 Running Mean Reversion Backtest for QQQ - November 6, 2025 (5-MINUTE BARS)\n');

  const result = await backtestMeanReversion('QQQ', '2025-11-06', 'intraday', 5, 1);

  console.log('\n📊 BACKTEST RESULTS FOR QQQ - November 6, 2025 (5-MIN BARS)');
  console.log('═'.repeat(120));
  console.log(`Net GEX: $${(result.netGex / 1_000_000).toFixed(1)}M (${result.regime})`);
  console.log(`Bollinger Bands: Lower $${result.technicals.bbLower?.toFixed(2)} | Middle $${result.technicals.bbMiddle?.toFixed(2)} | Upper $${result.technicals.bbUpper?.toFixed(2)}`);
  console.log(`Average RSI: ${result.technicals.avgRSI?.toFixed(1)}`);
  console.log(`Total Trades: ${result.trades.length}`);
  console.log('═'.repeat(120));

  if (result.trades.length === 0) {
    console.log('\n⚠️  No trades executed');
    if (result.notes.length > 0) {
      console.log('\nNotes:');
      result.notes.forEach(note => console.log(`  - ${note}`));
    }
    return;
  }

  console.log('\n📋 COMPLETE TRADE LIST:\n');

  result.trades.forEach((trade, idx) => {
    const entryDate = new Date(trade.entryTimestamp);
    const exitDate = new Date(trade.exitTimestamp);

    const entryTime = entryDate.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
    const exitTime = exitDate.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });

    const profitSign = trade.stock.profit >= 0 ? '+' : '';
    const winLoss = trade.stock.profit >= 0 ? '✅ WIN' : '❌ LOSS';

    console.log('═'.repeat(120));
    console.log(`TRADE #${idx + 1} - ${trade.direction.toUpperCase()} - ${winLoss}`);
    console.log('═'.repeat(120));
    console.log(`Entry Time:  ${entryTime}`);
    console.log(`Entry Price: $${trade.stock.entryPrice.toFixed(2)} (RSI: ${trade.entryRSI?.toFixed(1) ?? 'N/A'})`);
    console.log(`Stop Loss:   $${trade.stopLoss?.toFixed(2) ?? 'N/A'}`);
    console.log(`Target:      $${trade.target?.toFixed(2) ?? 'N/A'}`);
    console.log('');
    console.log(`Exit Time:   ${exitTime}`);
    console.log(`Exit Price:  $${trade.stock.exitPrice.toFixed(2)} (RSI: ${trade.exitRSI?.toFixed(1) ?? 'N/A'})`);
    console.log(`Exit Reason: ${trade.exitReason.toUpperCase()}`);
    console.log('');
    console.log(`Duration:    ${trade.durationMinutes} minutes`);
    console.log(`Stock P&L:   ${profitSign}$${trade.stock.profit.toFixed(2)} (${profitSign}${trade.stock.profitPct.toFixed(2)}%)`);

    if (trade.option.contract) {
      console.log('');
      console.log('OPTIONS P&L:');
      console.log(`  Contract:      ${trade.option.contract}`);
      console.log(`  Strike:        $${trade.option.strike?.toFixed(2)}`);
      console.log(`  Entry Premium: $${trade.option.entryPremium?.toFixed(2)}`);
      console.log(`  Exit Premium:  $${trade.option.exitPremium?.toFixed(2)}`);
      console.log(`  Gross P&L:     ${profitSign}$${trade.option.grossProfit?.toFixed(2)}`);
      console.log(`  Commission:    $${trade.option.commission?.toFixed(2)}`);
      console.log(`  Slippage:      $${trade.option.slippage?.toFixed(2)}`);
      console.log(`  Net P&L:       ${profitSign}$${trade.option.profit?.toFixed(2)} (${profitSign}${trade.option.profitPct?.toFixed(2)}%)`);
    }
  });

  // Summary
  console.log('\n\n');
  console.log('═'.repeat(120));
  console.log('📈 SUMMARY STATISTICS');
  console.log('═'.repeat(120));
  console.log(`\nTotal Trades: ${result.summary.tradeCount}`);
  console.log(`Wins: ${result.summary.winCount} | Losses: ${result.summary.lossCount}`);
  console.log(`Win Rate: ${(result.summary.winRate * 100).toFixed(2)}%`);

  console.log(`\n📊 STOCK TRADING (2 units per trade):`);
  console.log(`  Total P&L:      $${result.summary.stock.totalProfit.toFixed(2)}`);
  console.log(`  Gross Profit:   $${result.summary.stock.grossProfit.toFixed(2)}`);
  console.log(`  Gross Loss:     $${result.summary.stock.grossLoss.toFixed(2)}`);
  console.log(`  Average Win:    $${result.summary.stock.averageWin.toFixed(2)}`);
  console.log(`  Average Loss:   $${result.summary.stock.averageLoss.toFixed(2)}`);
  console.log(`  Profit Factor:  ${result.summary.stock.profitFactor === Infinity ? '∞' : result.summary.stock.profitFactor.toFixed(2)}`);

  if (result.summary.options.contractsTraded > 0) {
    console.log(`\n📊 OPTIONS TRADING (${result.summary.options.contractsTraded} contracts):`);
    console.log(`  Total P&L:      $${result.summary.options.totalProfit.toFixed(2)}`);
    console.log(`  Gross Profit:   $${result.summary.options.grossProfit.toFixed(2)}`);
    console.log(`  Gross Loss:     $${result.summary.options.grossLoss.toFixed(2)}`);
    console.log(`  Commission:     $${result.summary.options.totalCommission.toFixed(2)}`);
    console.log(`  Slippage:       $${result.summary.options.totalSlippage.toFixed(2)}`);
    console.log(`  Profit Factor:  ${result.summary.options.profitFactor === Infinity ? '∞' : result.summary.options.profitFactor.toFixed(2)}`);

    console.log(`\n  📈 Leverage Scenarios:`);
    console.log(`    50 contracts:  Net $${result.summary.options.leverage['50x'].netProfit.toFixed(2)} | Gross $${result.summary.options.leverage['50x'].grossProfit.toFixed(2)}`);
    console.log(`    100 contracts: Net $${result.summary.options.leverage['100x'].netProfit.toFixed(2)} | Gross $${result.summary.options.leverage['100x'].grossProfit.toFixed(2)}`);
  }

  console.log(`\n📉 Risk Metrics:`);
  console.log(`  Max Drawdown:   $${result.summary.maxDrawdown.toFixed(2)}`);
  console.log(`  Avg Duration:   ${result.summary.averageDurationMinutes.toFixed(0)} minutes`);

  const exitReasons = new Map<string, number>();
  result.trades.forEach(t => {
    exitReasons.set(t.exitReason, (exitReasons.get(t.exitReason) || 0) + 1);
  });

  console.log(`\n📤 Exit Breakdown:`);
  for (const [reason, count] of exitReasons.entries()) {
    const pct = (count / result.trades.length * 100).toFixed(1);
    console.log(`  ${reason}: ${count} (${pct}%)`);
  }

  if (result.notes.length > 0) {
    console.log('\n📝 Notes:');
    result.notes.forEach(note => console.log(`  - ${note}`));
  }

  console.log('\n✅ Backtest Complete!');
}

runNov6QQQBacktest().catch(error => {
  console.error('❌ Backtest failed:', error);
  process.exit(1);
});
