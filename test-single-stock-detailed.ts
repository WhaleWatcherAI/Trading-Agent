import { SMA } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';
import { getOptionsChain } from './lib/tradier';

interface TradeDetail {
  tradeNumber: number;
  direction: 'CALL' | 'PUT';
  entry: {
    time: string;
    stockPrice: number;
    optionStrike: number;
    optionPremium: number;
    totalCost: number; // premium × 100
    sma: number;
  };
  exit: {
    time: string;
    stockPrice: number;
    optionPremium: number;
    totalValue: number; // premium × 100
    reason: string;
  };
  stockMove: number; // exit - entry stock price
  stockMovePercent: number;
  optionPnL: number; // Net P&L after commission
  optionReturnPercent: number;
  holdMinutes: number;
}

interface BacktestStats {
  symbol: string;
  totalTrades: number;
  winners: number;
  losers: number;
  winRate: number;
  totalPnL: number;
  avgPnL: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  avgHoldMinutes: number;
  profitFactor: number; // Total wins / Total losses
}

const SMA_PERIOD = 9;
const STOP_LOSS_PERCENT = -50;
const TAKE_PROFIT_PERCENT = 100;
const COMMISSION_PER_CONTRACT = 0.65;
const SLIPPAGE_PERCENT = 0.5;

function formatTime(timestamp: string | number): string {
  const date = new Date(Number(timestamp) * 1000);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
    timeZone: 'America/New_York'
  });
}

async function backtestSingleStock(symbol: string): Promise<{ trades: TradeDetail[], stats: BacktestStats }> {
  console.log('═══════════════════════════════════════════════════');
  console.log(`  DETAILED BACKTEST: ${symbol}`);
  console.log('═══════════════════════════════════════════════════');
  console.log(`SMA Period: ${SMA_PERIOD}`);
  console.log(`Date: 2025-11-04`);
  console.log(`Stop Loss: ${STOP_LOSS_PERCENT}%`);
  console.log(`Take Profit: ${TAKE_PROFIT_PERCENT}%`);
  console.log(`Commission: $${COMMISSION_PER_CONTRACT * 2} round-trip`);
  console.log('═══════════════════════════════════════════════════\n');

  const testDate = '2025-11-04';
  const history = await getHistoricalData(symbol, '1min', 1, testDate);
  const optionsChain = await getOptionsChain(symbol, undefined, testDate);

  console.log(`Loaded ${history.length} bars of 1-minute data`);
  console.log(`Loaded ${optionsChain.length} options\n`);

  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });

  const trades: TradeDetail[] = [];
  let currentTrade: any = null;
  let tradeNumber = 0;

  for (let i = SMA_PERIOD; i < history.length; i++) {
    const currentBar = history[i];
    const currentPrice = closes[i];
    const previousPrice = closes[i - 1];
    const currentSma = smaValues[i - SMA_PERIOD];
    const previousSma = smaValues[i - SMA_PERIOD - 1];

    if (!currentTrade) {
      // Entry logic
      const crossesUp = previousPrice <= previousSma && currentBar.high > currentSma;
      const crossesDown = previousPrice >= previousSma && currentBar.low < currentSma;

      if (crossesUp || crossesDown) {
        const direction = crossesUp ? 'CALL' : 'PUT';
        const optionType = crossesUp ? 'call' : 'put';

        const relevantOptions = optionsChain.filter(o => {
          if (o.type !== optionType) return false;
          if (optionType === 'call') return o.strike < currentPrice;
          else return o.strike > currentPrice;
        });

        relevantOptions.sort((a, b) => {
          if (optionType === 'call') return b.strike - a.strike;
          else return a.strike - b.strike;
        });

        if (relevantOptions.length === 0) continue;

        const bestOption = relevantOptions[0];

        let underlyingEntryPrice: number;
        if (crossesUp) {
          underlyingEntryPrice = currentBar.open > currentSma ? currentBar.open : currentSma;
        } else {
          underlyingEntryPrice = currentBar.open < currentSma ? currentBar.open : currentSma;
        }

        const priceDiffFromClose = underlyingEntryPrice - currentPrice;
        const premiumAdjustment = bestOption.greeks?.delta ? priceDiffFromClose * bestOption.greeks.delta : 0;
        const adjustedPremium = bestOption.premium + premiumAdjustment;
        const entryPremium = Math.max(0.01, adjustedPremium * (1 + SLIPPAGE_PERCENT / 100));

        tradeNumber++;
        currentTrade = {
          tradeNumber,
          direction,
          entryIndex: i,
          entryTime: currentBar.date,
          entryStockPrice: underlyingEntryPrice,
          entryPremium: entryPremium,
          strike: bestOption.strike,
          optionType,
          sma: currentSma,
          delta: bestOption.greeks?.delta || 0,
          theta: bestOption.greeks?.theta || 0,
        };
      }
    } else {
      // Exit logic
      const entryType = currentTrade.optionType;
      const entryPremium = currentTrade.entryPremium;

      const oppositeCross = entryType === 'call'
        ? (previousPrice >= previousSma && currentPrice < currentSma)
        : (previousPrice <= previousSma && currentPrice > currentSma);

      const currentOption = optionsChain.find(o => o.strike === currentTrade.strike && o.type === entryType);

      let estimatedCurrentPremium = entryPremium;
      if (currentOption && currentOption.greeks?.delta) {
        const underlyingMove = currentPrice - currentTrade.entryStockPrice;
        const minutesHeld = i - currentTrade.entryIndex;
        const optionMove = underlyingMove * currentOption.greeks.delta;
        const thetaDecay = (currentOption.greeks.theta || 0) * (minutesHeld / 390);
        estimatedCurrentPremium = Math.max(0.01, entryPremium + optionMove + thetaDecay);
      }

      const pnlPercent = ((estimatedCurrentPremium - entryPremium) / entryPremium) * 100;

      let exitReason: string | null = null;
      if (oppositeCross) exitReason = 'Opposite Signal';
      else if (pnlPercent <= STOP_LOSS_PERCENT) exitReason = 'Stop Loss';
      else if (pnlPercent >= TAKE_PROFIT_PERCENT) exitReason = 'Take Profit';
      else if (i === history.length - 1) exitReason = 'End of Day';

      if (exitReason) {
        const exitPremium = estimatedCurrentPremium * (1 - SLIPPAGE_PERCENT / 100);
        const grossPnl = (exitPremium - entryPremium) * 100; // × 100 for contract
        const netPnl = grossPnl - (COMMISSION_PER_CONTRACT * 2);
        const holdMinutes = i - currentTrade.entryIndex;

        const stockMove = currentPrice - currentTrade.entryStockPrice;
        const stockMovePercent = (stockMove / currentTrade.entryStockPrice) * 100;

        const trade: TradeDetail = {
          tradeNumber: currentTrade.tradeNumber,
          direction: currentTrade.direction,
          entry: {
            time: formatTime(currentTrade.entryTime),
            stockPrice: currentTrade.entryStockPrice,
            optionStrike: currentTrade.strike,
            optionPremium: entryPremium,
            totalCost: entryPremium * 100,
            sma: currentTrade.sma,
          },
          exit: {
            time: formatTime(currentBar.date),
            stockPrice: currentPrice,
            optionPremium: exitPremium,
            totalValue: exitPremium * 100,
            reason: exitReason,
          },
          stockMove,
          stockMovePercent,
          optionPnL: netPnl,
          optionReturnPercent: (netPnl / (entryPremium * 100)) * 100,
          holdMinutes,
        };

        trades.push(trade);
        currentTrade = null;
      }
    }
  }

  // Calculate statistics
  const winners = trades.filter(t => t.optionPnL > 0);
  const losers = trades.filter(t => t.optionPnL <= 0);
  const totalPnL = trades.reduce((sum, t) => sum + t.optionPnL, 0);
  const avgPnL = trades.length > 0 ? totalPnL / trades.length : 0;
  const totalWins = winners.reduce((sum, t) => sum + t.optionPnL, 0);
  const totalLosses = Math.abs(losers.reduce((sum, t) => sum + t.optionPnL, 0));
  const avgWin = winners.length > 0 ? totalWins / winners.length : 0;
  const avgLoss = losers.length > 0 ? losers.reduce((sum, t) => sum + t.optionPnL, 0) / losers.length : 0;
  const largestWin = winners.length > 0 ? Math.max(...winners.map(t => t.optionPnL)) : 0;
  const largestLoss = losers.length > 0 ? Math.min(...losers.map(t => t.optionPnL)) : 0;
  const avgHoldMinutes = trades.length > 0 ? trades.reduce((sum, t) => sum + t.holdMinutes, 0) / trades.length : 0;
  const profitFactor = totalLosses > 0 ? totalWins / totalLosses : 0;

  const stats: BacktestStats = {
    symbol,
    totalTrades: trades.length,
    winners: winners.length,
    losers: losers.length,
    winRate: trades.length > 0 ? (winners.length / trades.length) * 100 : 0,
    totalPnL,
    avgPnL,
    avgWin,
    avgLoss,
    largestWin,
    largestLoss,
    avgHoldMinutes,
    profitFactor,
  };

  return { trades, stats };
}

async function run() {
  const symbol = 'TSLA';
  const { trades, stats } = await backtestSingleStock(symbol);

  // Print statistics
  console.log('\n═══════════════════════════════════════════════════');
  console.log('  BACKTEST STATISTICS');
  console.log('═══════════════════════════════════════════════════');
  console.log(`Total Trades: ${stats.totalTrades}`);
  console.log(`Winners: ${stats.winners} | Losers: ${stats.losers}`);
  console.log(`Win Rate: ${stats.winRate.toFixed(1)}%`);
  console.log(`Total P&L: $${stats.totalPnL.toFixed(2)}`);
  console.log(`Average P&L: $${stats.avgPnL.toFixed(2)}`);
  console.log(`Average Win: $${stats.avgWin.toFixed(2)}`);
  console.log(`Average Loss: $${stats.avgLoss.toFixed(2)}`);
  console.log(`Largest Win: $${stats.largestWin.toFixed(2)}`);
  console.log(`Largest Loss: $${stats.largestLoss.toFixed(2)}`);
  console.log(`Average Hold Time: ${stats.avgHoldMinutes.toFixed(1)} minutes`);
  console.log(`Profit Factor: ${stats.profitFactor.toFixed(2)}`);
  console.log('═══════════════════════════════════════════════════\n');

  // Print all trades
  console.log('═══════════════════════════════════════════════════');
  console.log('  TRADE LOG');
  console.log('═══════════════════════════════════════════════════\n');

  trades.forEach(trade => {
    const profit = trade.optionPnL >= 0;
    const emoji = profit ? '✅' : '❌';

    console.log(`${emoji} Trade #${trade.tradeNumber} - ${trade.direction}`);
    console.log(`   ENTRY: ${trade.entry.time}`);
    console.log(`      Stock: $${trade.entry.stockPrice.toFixed(2)} | SMA: $${trade.entry.sma.toFixed(2)}`);
    console.log(`      Option: $${trade.entry.optionStrike} ${trade.direction} @ $${trade.entry.optionPremium.toFixed(2)} per share`);
    console.log(`      Total Cost: $${trade.entry.totalCost.toFixed(2)}`);
    console.log(`   EXIT: ${trade.exit.time} (${trade.holdMinutes} min later)`);
    console.log(`      Stock: $${trade.exit.stockPrice.toFixed(2)} (${trade.stockMove >= 0 ? '+' : ''}${trade.stockMove.toFixed(2)}, ${trade.stockMovePercent >= 0 ? '+' : ''}${trade.stockMovePercent.toFixed(2)}%)`);
    console.log(`      Option: $${trade.exit.optionPremium.toFixed(2)} per share`);
    console.log(`      Total Value: $${trade.exit.totalValue.toFixed(2)}`);
    console.log(`      Reason: ${trade.exit.reason}`);
    console.log(`   P&L: ${trade.optionPnL >= 0 ? '+' : ''}$${trade.optionPnL.toFixed(2)} (${trade.optionReturnPercent >= 0 ? '+' : ''}${trade.optionReturnPercent.toFixed(1)}%)`);
    console.log('');
  });

  console.log('═══════════════════════════════════════════════════');
  console.log('Backtest Complete!');
  console.log('═══════════════════════════════════════════════════');
}

run()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Backtest failed:', error);
    process.exit(1);
  });
