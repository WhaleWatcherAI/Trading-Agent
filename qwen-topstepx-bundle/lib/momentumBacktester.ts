import { getHistoricalTimesales } from './tradier';
import {
  generateMomentumSignal,
  checkMomentumExit,
  calculateTrend,
  MomentumSignal,
} from './momentumAgent';
import { calculateGexForSymbol, GexMode } from './gexCalculator';
import { selectMeanReversionOption, OptionContract } from './contractSelector';

export interface MomentumTrade {
  id: string;
  symbol: string;
  direction: 'bullish' | 'bearish';
  entryDate: string;
  exitDate: string;

  // Option trade data
  option: {
    contract: string | null;
    type: 'call' | 'put';
    strike: number | null;
    expiration: string | null;
    entryPremium: number | null;
    exitPremium: number | null;
    profit: number | null;
    profitPct: number | null;
    commission: number | null;
    slippage: number | null;
  };

  entryPrice: number;
  exitPrice: number;
  durationDays: number;
  exitReason: 'opposite_cross' | 'take_profit' | 'stop_loss' | 'end_of_period';
  entryNetGex: number;
  exitNetGex: number;
}

export interface MomentumBacktestResult {
  symbol: string;
  startDate: string;
  endDate: string;
  trades: MomentumTrade[];
  summary: {
    tradeCount: number;
    winCount: number;
    lossCount: number;
    winRate: number;
    totalProfit: number;
    grossProfit: number;
    grossLoss: number;
    averageWin: number;
    averageLoss: number;
    profitFactor: number;
    averageDurationDays: number;
    totalCommission: number;
    totalSlippage: number;
  };
  notes: string[];
}

interface ActivePosition {
  signal: MomentumSignal;
  entryDate: string;
  entryPrice: number;
  optionContract: OptionContract | null;
  optionEntryPremium: number | null;
  entryNetGex: number;
}

/**
 * Generate array of trading dates between start and end (weekdays only)
 */
function generateTradingDates(startDate: string, endDate: string): string[] {
  const dates: string[] = [];
  const start = new Date(startDate);
  const end = new Date(endDate);

  const current = new Date(start);
  while (current <= end) {
    const dayOfWeek = current.getDay();
    // Skip weekends (0 = Sunday, 6 = Saturday)
    if (dayOfWeek !== 0 && dayOfWeek !== 6) {
      dates.push(current.toISOString().split('T')[0]);
    }
    current.setDate(current.getDate() + 1);
  }

  return dates;
}

/**
 * Run momentum/trend following backtest for a single symbol over date range
 */
export async function backtestMomentum(
  symbol: string,
  startDate: string,
  endDate: string,
  mode: GexMode = 'intraday',
): Promise<MomentumBacktestResult> {
  console.log(`\n🔄 Running momentum backtest for ${symbol} from ${startDate} to ${endDate}`);

  const notes: string[] = [];
  const trades: MomentumTrade[] = [];

  // Generate trading dates and fetch intraday 1-min bars
  const tradingDates = generateTradingDates(startDate, endDate);
  const bars: Array<{ date: string; time: string; timestamp: string; close: number }> = [];

  console.log(`  Fetching 1-min intraday data for ${tradingDates.length} trading days...`);

  // Fetch 1-min bars for each day
  for (const date of tradingDates) {
    try {
      const intradayBars = await getHistoricalTimesales(symbol, date, 1); // 1-min bars

      if (intradayBars.length > 0) {
        // Add all 1-min bars to our dataset
        for (const bar of intradayBars) {
          bars.push({
            date,
            time: bar.time,
            timestamp: bar.timestamp,
            close: bar.close,
          });
        }
      } else {
        console.log(`    ⚠️  No intraday data for ${date}`);
      }
    } catch (error: any) {
      // Skip days with no data (holidays, etc.)
      if (!error.message.includes('No timesales data')) {
        console.log(`    ⚠️  Failed to fetch ${date}: ${error.message}`);
      }
    }
  }

  console.log(`  ✓ Got ${bars.length} 1-min bars across ${tradingDates.length} days`);

  if (bars.length === 0) {
    notes.push('No price data available');
  }

  if (bars.length < 20) {
    notes.push(`Insufficient bars (${bars.length}, need at least 20 for SMA20)`);
    return {
      symbol,
      startDate,
      endDate,
      trades: [],
      summary: {
        tradeCount: 0,
        winCount: 0,
        lossCount: 0,
        winRate: 0,
        totalProfit: 0,
        grossProfit: 0,
        grossLoss: 0,
        averageWin: 0,
        averageLoss: 0,
        profitFactor: 0,
        averageDurationDays: 0,
        totalCommission: 0,
        totalSlippage: 0,
      },
      notes,
    };
  }

  // Track active position
  let activePosition: ActivePosition | null = null;

  // Iterate through each 1-min bar
  for (let i = 20; i < bars.length; i++) {
    const bar = bars[i];
    const currentDate = bar.date;
    const currentTime = bar.time;
    const currentTimestamp = bar.timestamp;
    const currentPrice = Number(bar.close);

    if (!Number.isFinite(currentPrice) || currentPrice <= 0) {
      continue;
    }

    // Build price history up to current bar (1-min closes)
    const priceHistory = bars.slice(0, i + 1).map(b => Number(b.close));

    // GEX DISABLED: Sandbox doesn't support historical options chains
    const netGex = 0;

    // Check for exit on active position
    if (activePosition) {
      let shouldExit = false;
      let exitReason: 'opposite_cross' | 'take_profit' | 'stop_loss' = 'opposite_cross';

      // Calculate current option premium
      if (activePosition.optionContract && activePosition.optionEntryPremium) {
        const priceMove = currentPrice - activePosition.entryPrice;
        const delta = activePosition.optionContract.delta;
        const premiumChange = priceMove * Math.abs(delta);

        const currentPremium = activePosition.optionEntryPremium + (
          activePosition.signal.direction === 'bullish' ? premiumChange : -premiumChange
        );

        const profitPct = ((currentPremium - activePosition.optionEntryPremium) / activePosition.optionEntryPremium) * 100;

        // 1:5 Risk/Reward - Stop Loss at -20%, Take Profit at +100%
        if (profitPct <= -20) {
          shouldExit = true;
          exitReason = 'stop_loss';
        } else if (profitPct >= 100) {
          shouldExit = true;
          exitReason = 'take_profit';
        }
      }

      // Check for opposite crossover
      if (!shouldExit) {
        const trend = calculateTrend(priceHistory, 9, 20);
        if (trend) {
          const exitCheck = checkMomentumExit(activePosition.signal.direction, trend);
          if (exitCheck.shouldExit) {
            shouldExit = true;
            exitReason = 'opposite_cross';
          }
        }
      }

      if (shouldExit) {
        // Calculate option P&L
        let optionProfit: number | null = null;
        let optionProfitPct: number | null = null;
        let optionExitPremium: number | null = null;
        let optionCommission: number | null = null;
        let optionSlippage: number | null = null;

        if (activePosition.optionContract && activePosition.optionEntryPremium) {
          const priceMove = currentPrice - activePosition.entryPrice;
          const delta = activePosition.optionContract.delta;
          const premiumChange = priceMove * Math.abs(delta);

          optionExitPremium = activePosition.optionEntryPremium + (
            activePosition.signal.direction === 'bullish' ? premiumChange : -premiumChange
          );

          optionExitPremium = Math.max(0.01, optionExitPremium);

          // Calculate slippage
          const entrySpread = activePosition.optionContract.ask - activePosition.optionContract.bid;
          const exitSpread = entrySpread;
          const entrySlippage = entrySpread * 0.5;
          const exitSlippage = exitSpread * 0.5;
          optionSlippage = (entrySlippage + exitSlippage) * 100;

          // Commission
          optionCommission = 0.08;

          // Net P&L
          const grossProfit = (optionExitPremium - activePosition.optionEntryPremium) * 100;
          optionProfit = grossProfit - optionCommission - optionSlippage;
          optionProfitPct = ((optionExitPremium - activePosition.optionEntryPremium) / activePosition.optionEntryPremium) * 100;
        }

        const durationMs = new Date(currentTimestamp).getTime() - new Date(activePosition.entryDate).getTime();
        const durationDays = durationMs / (1000 * 60 * 60 * 24);

        const trade: MomentumTrade = {
          id: `${symbol}-${activePosition.entryDate}`,
          symbol,
          direction: activePosition.signal.direction,
          entryDate: activePosition.entryDate,
          exitDate: `${currentDate} ${currentTime}`,
          option: {
            contract: activePosition.optionContract?.symbol || null,
            type: activePosition.signal.direction === 'bullish' ? 'call' : 'put',
            strike: activePosition.optionContract?.strike || null,
            expiration: activePosition.optionContract?.expiration || null,
            entryPremium: activePosition.optionEntryPremium,
            exitPremium: optionExitPremium,
            profit: optionProfit,
            profitPct: optionProfitPct,
            commission: optionCommission,
            slippage: optionSlippage,
          },
          entryPrice: activePosition.entryPrice,
          exitPrice: currentPrice,
          durationDays,
          exitReason,
          entryNetGex: activePosition.entryNetGex,
          exitNetGex: netGex,
        };

        trades.push(trade);
        activePosition = null;
      }
    }

    // Check for new entry if no active position
    // TEMPORARILY DISABLED: Negative GEX filter (sandbox data shows all positive GEX)
    if (!activePosition) {
      const signal = generateMomentumSignal(symbol, currentPrice, priceHistory, netGex, {
        fastPeriod: 9,
        slowPeriod: 20,
      });

      if (signal.action === 'buy_call' || signal.action === 'buy_put') {
        console.log(`  🎯 ${currentDate} ${currentTime}: ${signal.action.toUpperCase()} signal (${signal.trend}) @ $${currentPrice.toFixed(2)}`);

        // Try to fetch option contract
        let optionContract: OptionContract | null = null;
        let optionEntryPremium: number | null = null;

        try {
          const direction = signal.action === 'buy_call' ? 'long' : 'short';
          optionContract = await selectMeanReversionOption(
            symbol,
            currentPrice,
            direction,
            currentDate
          );

          if (optionContract) {
            optionEntryPremium = optionContract.mid;
          }
        } catch (error) {
          console.log(`    ⚠️  Could not fetch option for ${symbol} - skipping entry`);
        }

        // Only enter if we successfully got an option
        if (optionContract && optionEntryPremium) {
          console.log(`    ✅ Entered ${signal.direction} position: ${optionContract.symbol} @ $${optionEntryPremium.toFixed(2)}`);
          activePosition = {
            signal,
            entryDate: `${currentDate} ${currentTime}`,
            entryPrice: currentPrice,
            optionContract,
            optionEntryPremium,
            entryNetGex: netGex,
          };
        }
      }
    }
  }

  // Close any remaining position at end of period
  if (activePosition && bars.length > 0) {
    const lastBar = bars[bars.length - 1];
    const exitPrice = Number(lastBar.close);
    const exitDate = `${lastBar.date} ${lastBar.time}`;
    const exitTimestamp = lastBar.timestamp;

    let optionProfit: number | null = null;
    let optionProfitPct: number | null = null;
    let optionExitPremium: number | null = null;
    let optionCommission: number | null = null;
    let optionSlippage: number | null = null;

    if (activePosition.optionContract && activePosition.optionEntryPremium) {
      const priceMove = exitPrice - activePosition.entryPrice;
      const delta = activePosition.optionContract.delta;
      const premiumChange = priceMove * Math.abs(delta);

      optionExitPremium = activePosition.optionEntryPremium + (
        activePosition.signal.direction === 'bullish' ? premiumChange : -premiumChange
      );

      optionExitPremium = Math.max(0.01, optionExitPremium);

      const entrySpread = activePosition.optionContract.ask - activePosition.optionContract.bid;
      const exitSpread = entrySpread;
      const entrySlippage = entrySpread * 0.5;
      const exitSlippage = exitSpread * 0.5;
      optionSlippage = (entrySlippage + exitSlippage) * 100;

      optionCommission = 0.08;

      const grossProfit = (optionExitPremium - activePosition.optionEntryPremium) * 100;
      optionProfit = grossProfit - optionCommission - optionSlippage;
      optionProfitPct = ((optionExitPremium - activePosition.optionEntryPremium) / activePosition.optionEntryPremium) * 100;
    }

    const durationMs = new Date(exitTimestamp).getTime() - new Date(activePosition.entryDate).getTime();
    const durationDays = durationMs / (1000 * 60 * 60 * 24);

    // GEX DISABLED
    const exitNetGex = 0;

    const trade: MomentumTrade = {
      id: `${symbol}-${activePosition.entryDate}`,
      symbol,
      direction: activePosition.signal.direction,
      entryDate: activePosition.entryDate,
      exitDate,
      option: {
        contract: activePosition.optionContract?.symbol || null,
        type: activePosition.signal.direction === 'bullish' ? 'call' : 'put',
        strike: activePosition.optionContract?.strike || null,
        expiration: activePosition.optionContract?.expiration || null,
        entryPremium: activePosition.optionEntryPremium,
        exitPremium: optionExitPremium,
        profit: optionProfit,
        profitPct: optionProfitPct,
        commission: optionCommission,
        slippage: optionSlippage,
      },
      entryPrice: activePosition.entryPrice,
      exitPrice,
      durationDays,
      exitReason: 'end_of_period',
      entryNetGex: activePosition.entryNetGex,
      exitNetGex,
    };

    trades.push(trade);
  }

  // Calculate summary statistics
  const winningTrades = trades.filter(t => (t.option.profit || 0) > 0);
  const losingTrades = trades.filter(t => (t.option.profit || 0) <= 0);

  const totalProfit = trades.reduce((sum, t) => sum + (t.option.profit || 0), 0);
  const grossProfit = winningTrades.reduce((sum, t) => sum + (t.option.profit || 0), 0);
  const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + (t.option.profit || 0), 0));

  const totalCommission = trades.reduce((sum, t) => sum + (t.option.commission || 0), 0);
  const totalSlippage = trades.reduce((sum, t) => sum + (t.option.slippage || 0), 0);

  const summary = {
    tradeCount: trades.length,
    winCount: winningTrades.length,
    lossCount: losingTrades.length,
    winRate: trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0,
    totalProfit,
    grossProfit,
    grossLoss,
    averageWin: winningTrades.length > 0 ? grossProfit / winningTrades.length : 0,
    averageLoss: losingTrades.length > 0 ? grossLoss / losingTrades.length : 0,
    profitFactor: grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0,
    averageDurationDays: trades.length > 0 ? trades.reduce((sum, t) => sum + t.durationDays, 0) / trades.length : 0,
    totalCommission,
    totalSlippage,
  };

  return {
    symbol,
    startDate,
    endDate,
    trades,
    summary,
    notes,
  };
}

/**
 * Run momentum backtest on multiple symbols
 */
export async function backtestMomentumMultiple(
  symbols: string[],
  startDate: string,
  endDate: string,
  mode: GexMode = 'intraday',
): Promise<MomentumBacktestResult[]> {
  const results: MomentumBacktestResult[] = [];

  for (const symbol of symbols) {
    try {
      const result = await backtestMomentum(symbol, startDate, endDate, mode);
      results.push(result);
    } catch (error: any) {
      console.error(`Failed to backtest ${symbol}:`, error.message);
    }
  }

  return results;
}
