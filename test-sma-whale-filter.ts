import { SMA } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';
import { getOptionsChain } from './lib/tradier';

interface WhaleAnalysis {
  callPutRatio: number; // > 1 = more puts, < 1 = more calls
  whaleBias: 'bullish' | 'bearish' | 'neutral';
  whaleCallPremium: number;
  whalePutPremium: number;
  confidence: number; // 0-1 scale
}

interface Trade {
  symbol: string;
  entry: {
    timestamp: string;
    price: number;
    strike: number;
    type: 'call' | 'put';
    direction: 'bullish' | 'bearish';
    sma: number;
    whaleAnalysis: WhaleAnalysis;
  };
  exit?: {
    timestamp: string;
    price: number;
    reason: string;
  };
  pnl?: number;
  pnlPercent?: number;
}

const SMA_PERIOD = 20;
const STOP_LOSS_PERCENT = -50;
const TAKE_PROFIT_PERCENT = 100;
const COMMISSION_PER_CONTRACT = 0.65;
const SLIPPAGE_PERCENT = 0.5;
const WHALE_THRESHOLD = 50000; // $50k+ premium = whale trade
const MIN_WHALE_CONFIDENCE = 0.6; // Only take trades with 60%+ whale confidence

// Analyze whale flow from options chain
function analyzeWhaleFlow(optionsChain: any[]): WhaleAnalysis {
  let totalCallVolume = 0;
  let totalPutVolume = 0;
  let whaleCallPremium = 0;
  let whalePutPremium = 0;

  optionsChain.forEach(option => {
    const volume = option.volume || 0;
    const premium = option.premium || 0;
    const openInterest = option.open_interest || 0;
    const totalPremium = premium * volume * 100; // × 100 for contract multiplier

    if (option.type === 'call') {
      totalCallVolume += volume;
      if (totalPremium >= WHALE_THRESHOLD) {
        whaleCallPremium += totalPremium;
      }
    } else {
      totalPutVolume += volume;
      if (totalPremium >= WHALE_THRESHOLD) {
        whalePutPremium += totalPremium;
      }
    }
  });

  const callPutRatio = totalCallVolume > 0 ? totalPutVolume / totalCallVolume : 1.0;

  // Determine whale bias
  let whaleBias: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  let confidence = 0;

  const totalWhalePremium = whaleCallPremium + whalePutPremium;

  if (totalWhalePremium > 0) {
    const bullishRatio = whaleCallPremium / totalWhalePremium;
    const bearishRatio = whalePutPremium / totalWhalePremium;

    if (bullishRatio > 0.6) {
      whaleBias = 'bullish';
      confidence = bullishRatio;
    } else if (bearishRatio > 0.6) {
      whaleBias = 'bearish';
      confidence = bearishRatio;
    } else {
      confidence = Math.max(bullishRatio, bearishRatio);
    }
  }

  return {
    callPutRatio,
    whaleBias,
    whaleCallPremium,
    whalePutPremium,
    confidence,
  };
}

async function backtestSymbol(symbol: string): Promise<any> {
  try {
    console.log(`\n📊 Backtesting ${symbol} with whale filter...`);

    const testDate = '2025-11-04'; // Use yesterday's date
    const history = await getHistoricalData(symbol, '1min', 1, testDate);
    const optionsChain = await getOptionsChain(symbol, undefined, testDate);

    if (history.length < SMA_PERIOD + 20) {
      console.log(`  ⚠️  Insufficient data`);
      return null;
    }

    const closes = history.map(d => d.close);
    const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });

    const trades: Trade[] = [];
    let currentTrade: Trade | null = null;

    // Analyze overall whale flow for this ticker
    const overallWhaleAnalysis = analyzeWhaleFlow(optionsChain);
    console.log(`  Whale Analysis: ${overallWhaleAnalysis.whaleBias} (${(overallWhaleAnalysis.confidence * 100).toFixed(1)}% confidence)`);
    console.log(`  Call/Put Ratio: ${overallWhaleAnalysis.callPutRatio.toFixed(2)}`);
    console.log(`  Whale Call Premium: $${(overallWhaleAnalysis.whaleCallPremium / 1000).toFixed(1)}k`);
    console.log(`  Whale Put Premium: $${(overallWhaleAnalysis.whalePutPremium / 1000).toFixed(1)}k`);

    for (let i = SMA_PERIOD; i < history.length; i++) {
      const currentBar = history[i];
      const currentPrice = closes[i];
      const previousPrice = closes[i - 1];
      const currentSma = smaValues[i - SMA_PERIOD];
      const previousSma = smaValues[i - SMA_PERIOD - 1];

      if (!currentTrade) {
        const crossesUp = previousPrice <= previousSma && currentBar.high > currentSma;
        const crossesDown = previousPrice >= previousSma && currentBar.low < currentSma;

        if (crossesUp || crossesDown) {
          const direction = crossesUp ? 'bullish' : 'bearish';
          const optionType = crossesUp ? 'call' : 'put';

          // 🐋 WHALE FILTER: Only enter if whales agree with our signal
          const whaleAgrees =
            (direction === 'bullish' && overallWhaleAnalysis.whaleBias === 'bullish') ||
            (direction === 'bearish' && overallWhaleAnalysis.whaleBias === 'bearish');

          const hasConfidence = overallWhaleAnalysis.confidence >= MIN_WHALE_CONFIDENCE;

          if (!whaleAgrees || !hasConfidence) {
            // Skip this trade - whales don't agree
            continue;
          }

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
              whaleAnalysis: overallWhaleAnalysis,
            },
          };
        }
      } else {
        const entryType = currentTrade.entry.type;
        const entryPrice = currentTrade.entry.price;

        const oppositeCross = entryType === 'call'
          ? (previousPrice >= previousSma && currentPrice < currentSma)
          : (previousPrice <= previousSma && currentPrice > currentSma);

        const currentOption = optionsChain.find(o => o.strike === currentTrade.entry.strike && o.type === entryType);

        let estimatedCurrentPrice = entryPrice;
        if (currentOption && currentOption.greeks?.delta) {
          const underlyingMove = currentPrice - previousPrice;
          const optionMove = underlyingMove * currentOption.greeks.delta;
          const thetaDecay = (currentOption.greeks.theta || 0) / 390;
          estimatedCurrentPrice = Math.max(0.01, entryPrice + optionMove + thetaDecay);
        }

        const pnlPercent = ((estimatedCurrentPrice - entryPrice) / entryPrice) * 100;

        let exitReason: string | null = null;
        if (oppositeCross) exitReason = 'opposite_signal';
        else if (pnlPercent <= STOP_LOSS_PERCENT) exitReason = 'stop_loss';
        else if (pnlPercent >= TAKE_PROFIT_PERCENT) exitReason = 'take_profit';
        else if (i === history.length - 1) exitReason = 'end_of_day';

        if (exitReason) {
          const exitPrice = estimatedCurrentPrice * (1 - SLIPPAGE_PERCENT / 100);
          const grossPnl = (exitPrice - entryPrice) * 100; // × 100 for contract multiplier
          const netPnl = grossPnl - (COMMISSION_PER_CONTRACT * 2);

          currentTrade.exit = {
            timestamp: history[i].date,
            price: exitPrice,
            reason: exitReason,
          };
          currentTrade.pnl = netPnl;
          currentTrade.pnlPercent = (netPnl / (entryPrice * 100)) * 100;

          trades.push(currentTrade);
          currentTrade = null;
        }
      }
    }

    const winners = trades.filter(t => t.pnl! > 0);
    const losers = trades.filter(t => t.pnl! <= 0);
    const totalPnL = trades.reduce((sum, t) => sum + t.pnl!, 0);
    const winRate = trades.length > 0 ? (winners.length / trades.length) * 100 : 0;
    const avgWin = winners.length > 0 ? winners.reduce((sum, t) => sum + t.pnl!, 0) / winners.length : 0;
    const avgLoss = losers.length > 0 ? losers.reduce((sum, t) => sum + t.pnl!, 0) / losers.length : 0;

    console.log(`  ✓ Completed: ${trades.length} trades, Win Rate: ${winRate.toFixed(1)}%, P&L: $${totalPnL.toFixed(2)}`);

    return {
      symbol,
      trades,
      totalTrades: trades.length,
      winners: winners.length,
      losers: losers.length,
      winRate,
      totalPnL,
      avgWin,
      avgLoss,
      whaleAnalysis: overallWhaleAnalysis,
    };
  } catch (error: any) {
    console.error(`  ✗ Error backtesting ${symbol}:`, error.message);
    return null;
  }
}

async function runBacktest() {
  const TOP_TICKERS = [
    'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
    'V', 'JPM', 'JNJ', 'WMT', 'MA', 'PG', 'XOM', 'UNH', 'HD', 'CVX', 'DIS',
  ];

  console.log('═══════════════════════════════════════════════════');
  console.log('  20-SMA + WHALE FLOW FILTER BACKTEST');
  console.log('═══════════════════════════════════════════════════');
  console.log(`SMA Period: ${SMA_PERIOD}`);
  console.log(`Whale Threshold: $${(WHALE_THRESHOLD / 1000).toFixed(0)}k premium`);
  console.log(`Min Whale Confidence: ${(MIN_WHALE_CONFIDENCE * 100).toFixed(0)}%`);
  console.log(`Stop Loss: ${STOP_LOSS_PERCENT}%`);
  console.log(`Take Profit: ${TAKE_PROFIT_PERCENT}%`);
  console.log(`Commission: $${COMMISSION_PER_CONTRACT} per contract`);
  console.log('═══════════════════════════════════════════════════');

  const results: any[] = [];

  for (const symbol of TOP_TICKERS) {
    const result = await backtestSymbol(symbol);
    if (result) {
      results.push(result);
    }
  }

  console.log('\n═══════════════════════════════════════════════════');
  console.log('  OVERALL RESULTS');
  console.log('═══════════════════════════════════════════════════\n');

  const totalTrades = results.reduce((sum, r) => sum + r.totalTrades, 0);
  const totalWinners = results.reduce((sum, r) => sum + r.winners, 0);
  const totalLosers = results.reduce((sum, r) => sum + r.losers, 0);
  const overallPnL = results.reduce((sum, r) => sum + r.totalPnL, 0);
  const overallWinRate = totalTrades > 0 ? (totalWinners / totalTrades) * 100 : 0;
  const avgPnLPerTrade = totalTrades > 0 ? overallPnL / totalTrades : 0;

  console.log(`Total Symbols Tested: ${results.length}`);
  console.log(`Total Trades: ${totalTrades}`);
  console.log(`Winners: ${totalWinners} | Losers: ${totalLosers}`);
  console.log(`Win Rate: ${overallWinRate.toFixed(1)}%`);
  console.log(`Total P&L: $${overallPnL.toFixed(2)}`);
  console.log(`Avg P&L per Trade: $${avgPnLPerTrade.toFixed(2)}\n`);

  const sortedByPnL = [...results].sort((a, b) => b.totalPnL - a.totalPnL);

  console.log('📈 Top 5 Performers:');
  sortedByPnL.slice(0, 5).forEach((r, i) => {
    console.log(`  ${i + 1}. ${r.symbol}: $${r.totalPnL.toFixed(2)} (${r.totalTrades} trades, ${r.winRate.toFixed(1)}% win rate)`);
  });

  console.log('\n📉 Bottom 5 Performers:');
  sortedByPnL.slice(-5).reverse().forEach((r, i) => {
    console.log(`  ${i + 1}. ${r.symbol}: $${r.totalPnL.toFixed(2)} (${r.totalTrades} trades, ${r.winRate.toFixed(1)}% win rate)`);
  });

  console.log('\n═══════════════════════════════════════════════════');
  console.log('Backtest Complete!');
  console.log('═══════════════════════════════════════════════════\n');
}

runBacktest()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Backtest failed:', error);
    process.exit(1);
  });
