import 'dotenv/config';
import { backtestMeanReversion } from './lib/meanReversionBacktester';

async function runNov6Backtest() {
  console.log('🚀 Running Mean Reversion Backtest for November 6, 2025\n');

  const result = await backtestMeanReversion('SPY', '2025-11-06', 'intraday', 15, 1);

  console.log('\n📊 BACKTEST RESULTS FOR SPY - November 6, 2025');
  console.log('═'.repeat(100));
  console.log(`Net GEX: $${(result.netGex / 1_000_000).toFixed(1)}M (${result.regime})`);
  console.log(`Bollinger Bands: Lower $${result.technicals.bbLower?.toFixed(2)} | Middle $${result.technicals.bbMiddle?.toFixed(2)} | Upper $${result.technicals.bbUpper?.toFixed(2)}`);
  console.log(`Average RSI: ${result.technicals.avgRSI?.toFixed(1)}`);
  console.log(`Total Trades: ${result.trades.length}`);
  console.log('═'.repeat(100));

  if (result.trades.length === 0) {
    console.log('\n⚠️  No trades executed');
    if (result.notes.length > 0) {
      console.log('\nNotes:');
      result.notes.forEach(note => console.log(`  - ${note}`));
    }
    return;
  }

  console.log('\n📋 TRADE LIST:\n');
  console.log('─'.repeat(100));

  result.trades.forEach((trade, idx) => {
    const entryTime = new Date(trade.entryTimestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });
    const exitTime = new Date(trade.exitTimestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });

    const profitSign = trade.stock.profit >= 0 ? '+' : '';
    const winLoss = trade.stock.profit >= 0 ? '✅ WIN' : '❌ LOSS';

    console.log(`Trade #${idx + 1} - ${trade.direction.toUpperCase()} - ${winLoss}`);
    console.log(`  Entry:  ${entryTime} @ $${trade.stock.entryPrice.toFixed(2)} (RSI: ${trade.entryRSI?.toFixed(1) ?? 'N/A'})`);
    console.log(`  Exit:   ${exitTime} @ $${trade.stock.exitPrice.toFixed(2)} (RSI: ${trade.exitRSI?.toFixed(1) ?? 'N/A'})`);
    console.log(`  P&L:    ${profitSign}$${trade.stock.profit.toFixed(2)} (${profitSign}${trade.stock.profitPct.toFixed(2)}%)`);
    console.log(`  Duration: ${trade.durationMinutes} minutes`);
    console.log(`  Exit Reason: ${trade.exitReason}`);
    console.log(`  Stop Loss: $${trade.stopLoss?.toFixed(2) ?? 'N/A'} | Target: $${trade.target?.toFixed(2) ?? 'N/A'}`);

    if (trade.futures) {
      console.log(`  Futures: ES ${profitSign}$${trade.futures.esProfit.toFixed(2)} | MES ${profitSign}$${trade.futures.mesProfit.toFixed(2)} (${trade.futures.indexPoints.toFixed(2)} SPX pts)`);
    }

    if (trade.option.contract) {
      console.log(`  Option: ${trade.option.contract}`);
      console.log(`    Entry Premium: $${trade.option.entryPremium?.toFixed(2) ?? 'N/A'} | Exit Premium: $${trade.option.exitPremium?.toFixed(2) ?? 'N/A'}`);
      console.log(`    P&L: ${profitSign}$${trade.option.profit?.toFixed(2) ?? 'N/A'} (${profitSign}${trade.option.profitPct?.toFixed(2) ?? 'N/A'}%)`);
      console.log(`    Costs: Commission $${trade.option.commission?.toFixed(2) ?? 'N/A'} | Slippage $${trade.option.slippage?.toFixed(2) ?? 'N/A'}`);
    }

    console.log('─'.repeat(100));
  });

  // Summary
  console.log('\n📈 SUMMARY STATISTICS:');
  console.log('═'.repeat(100));
  console.log(`Total Trades: ${result.summary.tradeCount}`);
  console.log(`Wins: ${result.summary.winCount} | Losses: ${result.summary.lossCount}`);
  console.log(`Win Rate: ${(result.summary.winRate * 100).toFixed(2)}%`);
  console.log(`\nStock Trading:`);
  console.log(`  Total P&L: $${result.summary.stock.totalProfit.toFixed(2)}`);
  console.log(`  Gross Profit: $${result.summary.stock.grossProfit.toFixed(2)}`);
  console.log(`  Gross Loss: $${result.summary.stock.grossLoss.toFixed(2)}`);
  console.log(`  Average Win: $${result.summary.stock.averageWin.toFixed(2)}`);
  console.log(`  Average Loss: $${result.summary.stock.averageLoss.toFixed(2)}`);
  console.log(`  Profit Factor: ${result.summary.stock.profitFactor === Infinity ? '∞' : result.summary.stock.profitFactor.toFixed(2)}`);

  if (result.summary.options.contractsTraded > 0) {
    console.log(`\nOptions Trading (${result.summary.options.contractsTraded} contracts):`);
    console.log(`  Total P&L: $${result.summary.options.totalProfit.toFixed(2)}`);
    console.log(`  Gross Profit: $${result.summary.options.grossProfit.toFixed(2)}`);
    console.log(`  Gross Loss: $${result.summary.options.grossLoss.toFixed(2)}`);
    console.log(`  Commission: $${result.summary.options.totalCommission.toFixed(2)}`);
    console.log(`  Slippage: $${result.summary.options.totalSlippage.toFixed(2)}`);
    console.log(`  Profit Factor: ${result.summary.options.profitFactor === Infinity ? '∞' : result.summary.options.profitFactor.toFixed(2)}`);
    console.log(`\n  Leverage Scenarios:`);
    console.log(`    50x:  Net $${result.summary.options.leverage['50x'].netProfit.toFixed(2)} | Gross $${result.summary.options.leverage['50x'].grossProfit.toFixed(2)}`);
    console.log(`    100x: Net $${result.summary.options.leverage['100x'].netProfit.toFixed(2)} | Gross $${result.summary.options.leverage['100x'].grossProfit.toFixed(2)}`);
  }

  console.log(`\nMax Drawdown: $${result.summary.maxDrawdown.toFixed(2)}`);
  console.log(`Average Trade Duration: ${result.summary.averageDurationMinutes.toFixed(0)} minutes`);

  if (result.notes.length > 0) {
    console.log('\n📝 Notes:');
    result.notes.forEach(note => console.log(`  - ${note}`));
  }

  console.log('\n✅ Backtest Complete!');
}

runNov6Backtest().catch(error => {
  console.error('❌ Backtest failed:', error);
  process.exit(1);
});
