import { SMA } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';
import { getOptionsChain } from './lib/tradier';

// Top 20 major tickers (removed BRK.B due to symbol format issues)
const TOP_TICKERS = [
  'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
  'V', 'JPM', 'JNJ', 'WMT', 'MA', 'PG', 'XOM', 'UNH', 'HD', 'CVX', 'DIS',
];

interface Trade {
  symbol: string;
  entry: {
    timestamp: string;
    price: number;
    strike: number;
    type: 'call' | 'put';
    direction: 'bullish' | 'bearish';
    sma: number;
    underlyingPrice: number; // Track the underlying price at entry
  };
  exit?: {
    timestamp: string;
    price: number;
    reason: 'opposite_signal' | 'end_of_day' | 'stop_loss' | 'take_profit';
  };
  pnl?: number;
  pnlPercent?: number;
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
}

const SMA_PERIOD = 20; // 20-SMA for medium-term signals
const STOP_LOSS_PERCENT = -50; // Exit if down 50%
const TAKE_PROFIT_PERCENT = 100; // Exit if up 100%
const COMMISSION_PER_CONTRACT = 0.65; // Typical options commission
const SLIPPAGE_PERCENT = 0.5; // 0.5% slippage on entry/exit

async function backtestSymbol(symbol: string, testDate: string): Promise<BacktestResult | null> {
  try {
    console.log(`\n📊 Backtesting ${symbol} on ${testDate}...`);

    // Get 1-minute historical data
    const history = await getHistoricalData(symbol, '1min', 7, testDate);

    if (history.length < SMA_PERIOD + 10) {
      console.log(`  ⚠️  Insufficient data (${history.length} bars)`);
      return null;
    }

    // Get options chain for this date
    const optionsChain = await getOptionsChain(symbol, undefined, testDate);
    if (!optionsChain || optionsChain.length === 0) {
      console.log(`  ⚠️  No options data available`);
      return null;
    }

    const closes = history.map(d => d.close);
    const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });

    const trades: Trade[] = [];
    let currentTrade: Trade | null = null;

    // Iterate through each bar
    for (let i = SMA_PERIOD; i < history.length; i++) {
      const currentBar = history[i];
      const currentPrice = closes[i];
      const previousPrice = closes[i - 1];
      const currentSma = smaValues[i - SMA_PERIOD];
      const previousSma = smaValues[i - SMA_PERIOD - 1];

      // Check for entry signals using intra-bar high/low (tick-level detection)
      if (!currentTrade) {
        // Bullish crossover: previous close below SMA, but current bar high touched/crossed above SMA
        const crossesUp = previousPrice <= previousSma && currentBar.high > currentSma;

        // Bearish crossover: previous close above SMA, but current bar low touched/crossed below SMA
        const crossesDown = previousPrice >= previousSma && currentBar.low < currentSma;

        if (crossesUp || crossesDown) {
          const direction = crossesUp ? 'bullish' : 'bearish';
          const optionType = crossesUp ? 'call' : 'put';

          // Find ITM option closest to current price
          const relevantOptions = optionsChain.filter(o => {
            if (o.type !== optionType) return false;
            if (optionType === 'call') {
              return o.strike < currentPrice;
            } else {
              return o.strike > currentPrice;
            }
          });

          if (relevantOptions.length === 0) continue;

          relevantOptions.sort((a, b) => {
            if (optionType === 'call') {
              return b.strike - a.strike;
            } else {
              return a.strike - b.strike;
            }
          });

          const bestOption = relevantOptions[0];

          // Estimate entry at the crossover point (SMA level) instead of bar close
          // If bar opened past the SMA, use open; otherwise use SMA as entry point
          let underlyingEntryPrice: number;
          if (crossesUp) {
            // Crossed up: enter at SMA or open if already above
            underlyingEntryPrice = currentBar.open > currentSma ? currentBar.open : currentSma;
          } else {
            // Crossed down: enter at SMA or open if already below
            underlyingEntryPrice = currentBar.open < currentSma ? currentBar.open : currentSma;
          }

          // Adjust option premium based on entry at SMA vs current price
          const priceDiffFromClose = underlyingEntryPrice - currentPrice;
          const premiumAdjustment = bestOption.greeks?.delta
            ? priceDiffFromClose * bestOption.greeks.delta
            : 0;
          const adjustedPremium = bestOption.premium + premiumAdjustment; // FIX: Should be + not -

          // Apply slippage - pay the ask on entry
          const entryPrice = adjustedPremium * (1 + SLIPPAGE_PERCENT / 100);

          currentTrade = {
            symbol,
            entry: {
              timestamp: history[i].date,
              price: Math.max(0.01, entryPrice),
              strike: bestOption.strike,
              type: optionType,
              direction,
              sma: currentSma,
              underlyingPrice: underlyingEntryPrice, // Store the underlying entry price
            },
          };
        }
      }
      // Check for exit signals
      else {
        const entryType = currentTrade.entry.type;
        const entryPrice = currentTrade.entry.price;

        // Opposite crossover signal - ONLY on CLOSE crossing back (not intra-bar wicks)
        const oppositeCross = entryType === 'call'
          ? (previousPrice >= previousSma && currentPrice < currentSma)  // Close crossed below SMA
          : (previousPrice <= previousSma && currentPrice > currentSma); // Close crossed above SMA

        // Estimate option price using actual delta from options chain
        // Find the option from the chain
        const currentOption = optionsChain.find(o =>
          o.strike === currentTrade.entry.strike &&
          o.type === entryType
        );

        let estimatedCurrentPrice = entryPrice;
        if (currentOption && currentOption.greeks?.delta) {
          // Use actual delta for price estimation
          // Delta is already signed: positive for calls, negative for puts
          const underlyingMove = currentPrice - currentTrade.entry.underlyingPrice; // FIX: Use entry price, not previous close!
          const optionMove = underlyingMove * currentOption.greeks.delta;  // Don't use Math.abs!

          // Account for theta decay - calculate based on time since entry
          const entryBarIndex = history.findIndex(h => h.date === currentTrade.entry.timestamp);
          const barsHeld = entryBarIndex >= 0 ? (i - entryBarIndex) : 1;
          const thetaDecay = (currentOption.greeks.theta || 0) * (barsHeld / 390);

          estimatedCurrentPrice = Math.max(0.01, entryPrice + optionMove + thetaDecay);
        } else {
          // Fallback to simple estimation if no greeks available
          const priceChange = (currentPrice - closes[i - 1]) / closes[i - 1];
          const estimatedDelta = entryType === 'call' ? 0.5 : -0.5;
          estimatedCurrentPrice = entryPrice * (1 + priceChange * Math.abs(estimatedDelta) * 100);
        }

        const pnlPercent = ((estimatedCurrentPrice - entryPrice) / entryPrice) * 100;

        let exitReason: Trade['exit']['reason'] | null = null;

        if (oppositeCross) {
          exitReason = 'opposite_signal';
        } else if (pnlPercent <= STOP_LOSS_PERCENT) {
          exitReason = 'stop_loss';
        } else if (pnlPercent >= TAKE_PROFIT_PERCENT) {
          exitReason = 'take_profit';
        } else if (i === history.length - 1) {
          exitReason = 'end_of_day';
        }

        if (exitReason) {
          // Apply slippage - sell at the bid on exit
          const exitPrice = estimatedCurrentPrice * (1 - SLIPPAGE_PERCENT / 100);

          // Calculate P&L with commissions (each contract = 100 shares!)
          const grossPnl = (exitPrice - entryPrice) * 100;
          const netPnl = grossPnl - (COMMISSION_PER_CONTRACT * 2); // Entry + exit commission

          currentTrade.exit = {
            timestamp: history[i].date,
            price: exitPrice,
            reason: exitReason,
          };
          currentTrade.pnl = netPnl;
          currentTrade.pnlPercent = (netPnl / entryPrice) * 100;

          trades.push(currentTrade);
          currentTrade = null;
        }
      }
    }

    // Close any open trade at end of day
    if (currentTrade) {
      const lastPrice = closes[closes.length - 1];
      const entryPrice = currentTrade.entry.price;
      const entryType = currentTrade.entry.type;

      // Use actual delta for price estimation
      const currentOption = optionsChain.find(o =>
        o.strike === currentTrade.entry.strike &&
        o.type === entryType
      );

      let estimatedCurrentPrice = entryPrice;
      if (currentOption && currentOption.greeks?.delta) {
        const underlyingMove = lastPrice - closes[closes.length - 2];
        const optionMove = underlyingMove * currentOption.greeks.delta;  // Don't use Math.abs!
        const barsHeld = closes.length - 1 - (currentTrade.entry.timestamp ? history.findIndex(h => h.date === currentTrade.entry.timestamp) : 0);
        const thetaDecay = (currentOption.greeks.theta || 0) * (barsHeld / 390);
        estimatedCurrentPrice = Math.max(0.01, entryPrice + optionMove + thetaDecay);
      } else {
        const priceChange = (lastPrice - closes[closes.length - 2]) / closes[closes.length - 2];
        const estimatedDelta = entryType === 'call' ? 0.5 : -0.5;
        estimatedCurrentPrice = entryPrice * (1 + priceChange * Math.abs(estimatedDelta) * 100);
      }

      // Apply slippage and commissions
      const exitPrice = estimatedCurrentPrice * (1 - SLIPPAGE_PERCENT / 100);
      const grossPnl = exitPrice - entryPrice;
      const netPnl = grossPnl - (COMMISSION_PER_CONTRACT * 2);

      currentTrade.exit = {
        timestamp: history[history.length - 1].date,
        price: exitPrice,
        reason: 'end_of_day',
      };
      currentTrade.pnl = netPnl;
      currentTrade.pnlPercent = (netPnl / entryPrice) * 100;
      trades.push(currentTrade);
    }

    // Calculate statistics
    const winners = trades.filter(t => t.pnl! > 0);
    const losers = trades.filter(t => t.pnl! <= 0);
    const totalPnL = trades.reduce((sum, t) => sum + t.pnl!, 0);
    const avgPnL = trades.length > 0 ? totalPnL / trades.length : 0;
    const avgWin = winners.length > 0 ? winners.reduce((sum, t) => sum + t.pnl!, 0) / winners.length : 0;
    const avgLoss = losers.length > 0 ? losers.reduce((sum, t) => sum + t.pnl!, 0) / losers.length : 0;

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
    };

    console.log(`  ✓ Completed: ${result.totalTrades} trades, Win Rate: ${result.winRate.toFixed(1)}%, P&L: $${result.totalPnL.toFixed(2)}`);

    return result;
  } catch (error: any) {
    console.error(`  ✗ Error backtesting ${symbol}:`, error.message);
    return null;
  }
}

async function runBacktest() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  20-SMA CROSSOVER STRATEGY BACKTEST');
  console.log('  (Tick Entry, Close Exit, Delta-Based Pricing)');
  console.log('═══════════════════════════════════════════════════');
  console.log(`SMA Period: ${SMA_PERIOD}`);
  console.log(`Stop Loss: ${STOP_LOSS_PERCENT}%`);
  console.log(`Take Profit: ${TAKE_PROFIT_PERCENT}%`);
  console.log(`Commission: $${COMMISSION_PER_CONTRACT} per contract (round-trip: $${COMMISSION_PER_CONTRACT * 2})`);
  console.log(`Slippage: ${SLIPPAGE_PERCENT}% (entry/exit)`);
  console.log(`Entry: When intra-bar high/low touches SMA`);
  console.log(`Exit: When CLOSE crosses back over SMA (opposite direction)`);
  console.log(`Option Pricing: Historical premium + delta/theta estimates`);
  console.log(`Test Date: 2025-11-04 (latest available)`);
  console.log('═══════════════════════════════════════════════════\n');

  const results: BacktestResult[] = [];
  const testDate = '2025-11-04';

  for (const symbol of TOP_TICKERS) {
    const result = await backtestSymbol(symbol, testDate);
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

  console.log(`Total Symbols Tested: ${results.length}`);
  console.log(`Total Trades: ${allTrades.length}`);
  console.log(`Winners: ${totalWinners} | Losers: ${totalLosers}`);
  console.log(`Win Rate: ${overallWinRate.toFixed(1)}%`);
  console.log(`Total P&L: $${overallPnL.toFixed(2)}`);
  console.log(`Avg P&L per Trade: $${avgPnLPerTrade.toFixed(2)}`);

  // Top performers
  const sortedByPnL = [...results].sort((a, b) => b.totalPnL - a.totalPnL);

  console.log('\n📈 Top 5 Performers:');
  sortedByPnL.slice(0, 5).forEach((r, i) => {
    console.log(`  ${i + 1}. ${r.symbol}: $${r.totalPnL.toFixed(2)} (${r.totalTrades} trades, ${r.winRate.toFixed(1)}% win rate)`);
  });

  console.log('\n📉 Bottom 5 Performers:');
  sortedByPnL.slice(-5).reverse().forEach((r, i) => {
    console.log(`  ${i + 1}. ${r.symbol}: $${r.totalPnL.toFixed(2)} (${r.totalTrades} trades, ${r.winRate.toFixed(1)}% win rate)`);
  });

  // Detailed trade breakdown
  console.log('\n═══════════════════════════════════════════════════');
  console.log('  DETAILED RESULTS BY SYMBOL');
  console.log('═══════════════════════════════════════════════════\n');

  results.forEach(result => {
    console.log(`${result.symbol}:`);
    console.log(`  Trades: ${result.totalTrades} | Win Rate: ${result.winRate.toFixed(1)}%`);
    console.log(`  P&L: $${result.totalPnL.toFixed(2)} | Avg: $${result.avgPnL.toFixed(2)}`);
    console.log(`  Avg Win: $${result.avgWin.toFixed(2)} | Avg Loss: $${result.avgLoss.toFixed(2)}`);

    if (result.trades.length > 0) {
      console.log(`  Sample Trades:`);
      result.trades.slice(0, 2).forEach((trade, i) => {
        console.log(`    ${i + 1}. ${trade.entry.type.toUpperCase()} $${trade.entry.strike} @ $${trade.entry.price.toFixed(2)} → $${trade.exit?.price.toFixed(2)} (${trade.exit?.reason}) = ${trade.pnlPercent?.toFixed(1)}%`);
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
