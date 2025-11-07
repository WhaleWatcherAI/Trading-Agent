import { runSmaSentimentBacktest } from './lib/smaSentimentBacktester';

async function main() {
  const symbol = process.argv[2] || 'SPY';
  const date = process.argv[3] || '2025-01-02';

  console.log(`\n🚀 Running SMA Sentiment Strategy Backtest`);
  console.log(`   Symbol: ${symbol}`);
  console.log(`   Date: ${date}`);
  console.log(`   Configuration:`);
  console.log(`     - 5-minute candles`);
  console.log(`     - 20-period SMA`);
  console.log(`     - Sentiment filtering (call/put ratios + options flow)`);
  console.log(`     - Initial capital: $10,000`);
  console.log(`     - Position size: 10% per trade`);
  console.log(`     - Exit after: 30 minutes\n`);

  try {
    const result = await runSmaSentimentBacktest({
      symbol,
      date,
      smaPeriod: 20,
      initialCapital: 10000,
      positionSize: 0.1,
      exitAfterMinutes: 30,
    });

    console.log(`\n${'='.repeat(60)}`);
    console.log(`BACKTEST RESULTS - ${symbol} on ${date}`);
    console.log('='.repeat(60));
    console.log(`\n📊 Performance Metrics:`);
    console.log(`   Initial Capital:     $${result.initialCapital.toFixed(2)}`);
    console.log(`   Final Capital:       $${result.finalCapital.toFixed(2)}`);
    console.log(`   Total Return:        $${result.totalReturn.toFixed(2)} (${result.totalReturnPercent.toFixed(2)}%)`);
    console.log(`\n📈 Trade Statistics:`);
    console.log(`   Total Trades:        ${result.totalTrades}`);
    console.log(`   Winning Trades:      ${result.winningTrades}`);
    console.log(`   Losing Trades:       ${result.losingTrades}`);
    console.log(`   Win Rate:            ${result.winRate.toFixed(2)}%`);
    console.log(`   Profit Factor:       ${result.profitFactor === Infinity ? '∞' : result.profitFactor.toFixed(2)}`);
    console.log(`\n💰 Win/Loss Analysis:`);
    console.log(`   Average Win:         $${result.avgWin.toFixed(2)}`);
    console.log(`   Average Loss:        $${result.avgLoss.toFixed(2)}`);
    console.log(`   Largest Win:         $${result.largestWin.toFixed(2)}`);
    console.log(`   Largest Loss:        $${result.largestLoss.toFixed(2)}`);

    if (result.trades.length > 0) {
      console.log(`\n📋 Trade Details:`);
      console.log('='.repeat(60));
      result.trades.forEach((trade, idx) => {
        const sign = trade.pnl >= 0 ? '+' : '';
        console.log(`\n   Trade #${idx + 1}: ${trade.type.toUpperCase()}`);
        console.log(`     Entry:       ${trade.entryTime} @ $${trade.entryPrice.toFixed(2)}`);
        console.log(`     Exit:        ${trade.exitTime} @ $${trade.exitPrice.toFixed(2)}`);
        console.log(`     Strike:      ${trade.strike.toFixed(2)}`);
        console.log(`     Contracts:   ${trade.contracts}`);
        console.log(`     Sentiment:   ${trade.sentiment} (${(trade.confidence * 100).toFixed(0)}% confidence)`);
        console.log(`     P&L:         ${sign}$${trade.pnl.toFixed(2)} (${sign}${trade.pnlPercent.toFixed(2)}%)`);
        console.log(`     Exit Reason: ${trade.exitReason}`);
      });
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log(`\n✅ Results saved to: backtest_sma_sentiment_${symbol}_${date}.json`);
    console.log(`\n💡 To view the interactive chart, run:`);
    console.log(`   npm run dev`);
    console.log(`   Then visit: http://localhost:3000/sma-sentiment\n`);

  } catch (error: any) {
    console.error(`\n❌ Error running backtest:`, error.message);
    console.error(error);
    process.exit(1);
  }
}

main();
