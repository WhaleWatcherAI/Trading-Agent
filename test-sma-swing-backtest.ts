import { SMA } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';
import { getOptionsChain } from './lib/tradier';

const TOP_TICKERS = [
  'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
  'V', 'JPM', 'JNJ', 'WMT', 'MA', 'PG', 'XOM', 'UNH', 'HD', 'CVX', 'DIS',
];

interface Trade {
  symbol: string;
  entry: {
    date: string;
    price: number;
    strike: number;
    type: 'call' | 'put';
    direction: 'bullish' | 'bearish';
    sma: number;
    underlyingPrice: number;
    daysToExpiration: number;
  };
  exit?: {
    date: string;
    price: number;
    reason: 'opposite_signal' | 'end_of_backtest' | 'stop_loss' | 'take_profit';
  };
  pnl?: number;
  pnlPercent?: number;
  daysHeld?: number;
}

interface BacktestResult {
  symbol: string;
  trades: Trade[];
  totalTrades: number;
  winners: number;
  losers: number;
  winRate: number;
  totalPnL: number;
  avgPnL: number;
  avgWin: number;
  avgLoss: number;
  avgDaysHeld: number;
}

const SMA_PERIOD = 50; // Swing trading with 50-SMA (200 requires more data than available)
const MIN_DTE = 5;
const MAX_DTE = 14;
const STOP_LOSS_PERCENT = -50;
const TAKE_PROFIT_PERCENT = 100;
const COMMISSION_PER_CONTRACT = 0.65;
const SLIPPAGE_PERCENT = 0.5;

async function backtestSymbol(symbol: string, startDate: string, endDate: string): Promise<BacktestResult | null> {
  try {
    console.log(`\n📊 Backtesting ${symbol}...`);

    // Get daily historical data (need 50+ days for 50-SMA, get more for backtest window)
    const history = await getHistoricalData(symbol, 'daily', 150);

    if (history.length < SMA_PERIOD + 20) {
      console.log(`  ⚠️  Insufficient data (${history.length} bars)`);
      return null;
    }

    const closes = history.map(d => d.close);
    const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });

    const trades: Trade[] = [];
    let currentTrade: Trade | null = null;

    // Iterate through each day
    for (let i = SMA_PERIOD; i < history.length; i++) {
      const currentBar = history[i];
      const currentPrice = closes[i];
      const previousPrice = closes[i - 1];
      const currentSma = smaValues[i - SMA_PERIOD];
      const previousSma = smaValues[i - SMA_PERIOD - 1];
      const currentDate = currentBar.date;

      // Entry logic
      if (!currentTrade) {
        const crossesUp = previousPrice <= previousSma && currentBar.high > currentSma;
        const crossesDown = previousPrice >= previousSma && currentBar.low < currentSma;

        if (crossesUp || crossesDown) {
          const direction = crossesUp ? 'bullish' : 'bearish';
          const optionType = crossesUp ? 'call' : 'put';

          // Get options chain for this date
          const optionsChain = await getOptionsChain(symbol, undefined, currentDate);

          if (!optionsChain || optionsChain.length === 0) {
            continue;
          }

          // Filter for 5-14 DTE options
          const validOptions = optionsChain.filter(o => {
            if (o.type !== optionType) return false;

            // Calculate DTE
            const expDate = new Date(o.expiration);
            const currDate = new Date(currentDate);
            const dte = Math.ceil((expDate.getTime() - currDate.getTime()) / (1000 * 60 * 60 * 24));

            if (dte < MIN_DTE || dte > MAX_DTE) return false;

            // Filter for ITM options
            if (optionType === 'call') {
              return o.strike < currentPrice;
            } else {
              return o.strike > currentPrice;
            }
          });

          if (validOptions.length === 0) continue;

          // Sort to get closest ITM option
          validOptions.sort((a, b) => {
            if (optionType === 'call') {
              return b.strike - a.strike;
            } else {
              return a.strike - b.strike;
            }
          });

          const bestOption = validOptions[0];

          // Calculate DTE
          const expDate = new Date(bestOption.expiration);
          const currDate = new Date(currentDate);
          const dte = Math.ceil((expDate.getTime() - currDate.getTime()) / (1000 * 60 * 60 * 24));

          // Entry at SMA level
          let underlyingEntryPrice: number;
          if (crossesUp) {
            underlyingEntryPrice = currentBar.open > currentSma ? currentBar.open : currentSma;
          } else {
            underlyingEntryPrice = currentBar.open < currentSma ? currentBar.open : currentSma;
          }

          // Adjust option premium
          const priceDiffFromClose = underlyingEntryPrice - currentPrice;
          const premiumAdjustment = bestOption.greeks?.delta ? priceDiffFromClose * bestOption.greeks.delta : 0;
          const adjustedPremium = bestOption.premium + premiumAdjustment;
          const entryPrice = adjustedPremium * (1 + SLIPPAGE_PERCENT / 100);

          currentTrade = {
            symbol,
            entry: {
              date: currentDate,
              price: Math.max(0.01, entryPrice),
              strike: bestOption.strike,
              type: optionType,
              direction,
              sma: currentSma,
              underlyingPrice: underlyingEntryPrice,
              daysToExpiration: dte,
            },
          };
        }
      }
      // Exit logic
      else {
        const entryType = currentTrade.entry.type;
        const entryPrice = currentTrade.entry.price;

        // Opposite crossover on close
        const oppositeCross = entryType === 'call'
          ? (previousPrice >= previousSma && currentPrice < currentSma)
          : (previousPrice <= previousSma && currentPrice > currentSma);

        // Get options chain to estimate current price
        const optionsChain = await getOptionsChain(symbol, undefined, currentDate);
        const currentOption = optionsChain?.find(o =>
          o.strike === currentTrade.entry.strike &&
          o.type === entryType
        );

        let estimatedCurrentPrice = entryPrice;
        if (currentOption && currentOption.greeks?.delta) {
          const underlyingMove = currentPrice - currentTrade.entry.underlyingPrice;
          const optionMove = underlyingMove * currentOption.greeks.delta;

          // Calculate days held for theta decay
          const entryDate = new Date(currentTrade.entry.date);
          const currentDateObj = new Date(currentDate);
          const daysHeld = Math.ceil((currentDateObj.getTime() - entryDate.getTime()) / (1000 * 60 * 60 * 24));
          const thetaDecay = (currentOption.greeks.theta || 0) * daysHeld;

          estimatedCurrentPrice = Math.max(0.01, entryPrice + optionMove + thetaDecay);
        }

        const pnlPercent = ((estimatedCurrentPrice - entryPrice) / entryPrice) * 100;

        let exitReason: string | null = null;
        if (oppositeCross) exitReason = 'opposite_signal';
        else if (pnlPercent <= STOP_LOSS_PERCENT) exitReason = 'stop_loss';
        else if (pnlPercent >= TAKE_PROFIT_PERCENT) exitReason = 'take_profit';
        else if (i === history.length - 1) exitReason = 'end_of_backtest';

        if (exitReason) {
          const exitPrice = estimatedCurrentPrice * (1 - SLIPPAGE_PERCENT / 100);
          const grossPnl = exitPrice - entryPrice;
          const netPnl = grossPnl - (COMMISSION_PER_CONTRACT * 2);

          const entryDate = new Date(currentTrade.entry.date);
          const exitDate = new Date(currentDate);
          const daysHeld = Math.ceil((exitDate.getTime() - entryDate.getTime()) / (1000 * 60 * 60 * 24));

          currentTrade.exit = { date: currentDate, price: exitPrice, reason: exitReason as any };
          currentTrade.pnl = netPnl;
          currentTrade.pnlPercent = (netPnl / entryPrice) * 100;
          currentTrade.daysHeld = daysHeld;

          trades.push(currentTrade);
          currentTrade = null;
        }
      }
    }

    const winners = trades.filter(t => t.pnl! > 0);
    const losers = trades.filter(t => t.pnl! <= 0);
    const totalPnL = trades.reduce((sum, t) => sum + t.pnl!, 0);
    const avgPnL = trades.length > 0 ? totalPnL / trades.length : 0;
    const avgWin = winners.length > 0 ? winners.reduce((sum, t) => sum + t.pnl!, 0) / winners.length : 0;
    const avgLoss = losers.length > 0 ? losers.reduce((sum, t) => sum + t.pnl!, 0) / losers.length : 0;
    const avgDaysHeld = trades.length > 0 ? trades.reduce((sum, t) => sum + (t.daysHeld || 0), 0) / trades.length : 0;

    const result: BacktestResult = {
      symbol,
      trades,
      totalTrades: trades.length,
      winners: winners.length,
      losers: losers.length,
      winRate: trades.length > 0 ? (winners.length / trades.length) * 100 : 0,
      totalPnL,
      avgPnL,
      avgWin,
      avgLoss,
      avgDaysHeld,
    };

    console.log(`  ✓ Completed: ${result.totalTrades} trades, Win Rate: ${result.winRate.toFixed(1)}%, P&L: $${result.totalPnL.toFixed(2)}, Avg Hold: ${result.avgDaysHeld.toFixed(1)} days`);

    return result;
  } catch (error: any) {
    console.error(`  ✗ Error backtesting ${symbol}:`, error.message);
    return null;
  }
}

async function runBacktest() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  50-SMA SWING TRADING BACKTEST');
  console.log('  (Daily Bars, 5-14 DTE Options)');
  console.log('═══════════════════════════════════════════════════');
  console.log(`SMA Period: ${SMA_PERIOD} (daily bars)`);
  console.log(`Option DTE: ${MIN_DTE}-${MAX_DTE} days`);
  console.log(`Stop Loss: ${STOP_LOSS_PERCENT}%`);
  console.log(`Take Profit: ${TAKE_PROFIT_PERCENT}%`);
  console.log(`Commission: $${COMMISSION_PER_CONTRACT} per contract (round-trip: $${COMMISSION_PER_CONTRACT * 2})`);
  console.log(`Slippage: ${SLIPPAGE_PERCENT}% (entry/exit)`);
  console.log(`Entry: When daily high/low touches SMA`);
  console.log(`Exit: When daily CLOSE crosses back over SMA`);
  console.log(`Backtest Period: Last 150 trading days`);
  console.log('═══════════════════════════════════════════════════\n');

  const results: BacktestResult[] = [];

  for (const symbol of TOP_TICKERS) {
    const result = await backtestSymbol(symbol, '', '');
    if (result) {
      results.push(result);
    }
  }

  // Overall statistics
  console.log('\n═══════════════════════════════════════════════════');
  console.log('  OVERALL RESULTS');
  console.log('═══════════════════════════════════════════════════\n');

  const allTrades = results.flatMap(r => r.trades);
  const totalWinners = results.reduce((sum, r) => sum + r.winners, 0);
  const totalLosers = results.reduce((sum, r) => sum + r.losers, 0);
  const overallPnL = results.reduce((sum, r) => sum + r.totalPnL, 0);
  const overallWinRate = allTrades.length > 0 ? (totalWinners / allTrades.length) * 100 : 0;
  const avgPnLPerTrade = allTrades.length > 0 ? overallPnL / allTrades.length : 0;
  const avgDaysHeld = allTrades.length > 0 ? allTrades.reduce((sum, t) => sum + (t.daysHeld || 0), 0) / allTrades.length : 0;

  console.log(`Total Symbols Tested: ${results.length}`);
  console.log(`Total Trades: ${allTrades.length}`);
  console.log(`Winners: ${totalWinners} | Losers: ${totalLosers}`);
  console.log(`Win Rate: ${overallWinRate.toFixed(1)}%`);
  console.log(`Total P&L: $${overallPnL.toFixed(2)}`);
  console.log(`Avg P&L per Trade: $${avgPnLPerTrade.toFixed(2)}`);
  console.log(`Avg Days Held: ${avgDaysHeld.toFixed(1)} days`);

  // Top performers
  const sortedByPnL = [...results].sort((a, b) => b.totalPnL - a.totalPnL);

  console.log('\n📈 Top 5 Performers:');
  sortedByPnL.slice(0, 5).forEach((r, i) => {
    console.log(`  ${i + 1}. ${r.symbol}: $${r.totalPnL.toFixed(2)} (${r.totalTrades} trades, ${r.winRate.toFixed(1)}% win rate, ${r.avgDaysHeld.toFixed(1)} day avg hold)`);
  });

  console.log('\n📉 Bottom 5 Performers:');
  sortedByPnL.slice(-5).reverse().forEach((r, i) => {
    console.log(`  ${i + 1}. ${r.symbol}: $${r.totalPnL.toFixed(2)} (${r.totalTrades} trades, ${r.winRate.toFixed(1)}% win rate, ${r.avgDaysHeld.toFixed(1)} day avg hold)`);
  });

  // Detailed breakdown
  console.log('\n═══════════════════════════════════════════════════');
  console.log('  DETAILED RESULTS BY SYMBOL');
  console.log('═══════════════════════════════════════════════════\n');

  results.forEach(result => {
    console.log(`${result.symbol}:`);
    console.log(`  Trades: ${result.totalTrades} | Win Rate: ${result.winRate.toFixed(1)}% | Avg Hold: ${result.avgDaysHeld.toFixed(1)} days`);
    console.log(`  P&L: $${result.totalPnL.toFixed(2)} | Avg: $${result.avgPnL.toFixed(2)}`);
    console.log(`  Avg Win: $${result.avgWin.toFixed(2)} | Avg Loss: $${result.avgLoss.toFixed(2)}`);

    if (result.trades.length > 0) {
      console.log(`  Sample Trades:`);
      result.trades.slice(0, 2).forEach((trade, i) => {
        console.log(`    ${i + 1}. ${trade.entry.type.toUpperCase()} $${trade.entry.strike} (${trade.entry.daysToExpiration}DTE) → ${trade.daysHeld} days → ${trade.pnlPercent?.toFixed(1)}% ($${trade.pnl?.toFixed(2)})`);
      });
    }
    console.log('');
  });

  console.log('═══════════════════════════════════════════════════');
  console.log('Backtest Complete!');
  console.log('═══════════════════════════════════════════════════\n');
}

runBacktest()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Backtest failed:', error);
    process.exit(1);
  });
