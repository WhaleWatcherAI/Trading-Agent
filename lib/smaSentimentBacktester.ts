import { promises as fs } from 'fs';
import path from 'path';
import { SMA } from 'technicalindicators';
import { getHistoricalTimesales, TradierTimesaleBar, getOptionsChain } from './tradier';
import { getWhaleFlowAlerts, getPutCallRatio } from './unusualwhales';
import { TradeSignal, OptionsTrade } from '@/types';

export interface SmaSentimentBacktestConfig {
  symbol: string;
  date: string;
  smaPeriod?: number;
  initialCapital?: number;
  positionSize?: number; // Percentage of capital per trade (0-1)
  exitAfterMinutes?: number; // Exit after X minutes
}

export interface BacktestTrade {
  entryTime: string;
  exitTime: string;
  type: 'call' | 'put';
  strike: number;
  expiration: string;
  entryPrice: number;
  exitPrice: number;
  contracts: number;
  entryReason: string;
  exitReason: string;
  pnl: number;
  pnlPercent: number;
  sentiment: string;
  confidence: number;
}

export interface BacktestPricePoint {
  time: string;
  price: number;
  sma: number;
}

export interface BacktestTradeMarker {
  time: string;
  price: number;
  type: 'buy' | 'sell';
  optionType: 'call' | 'put';
  strike: number;
  premium: number;
  sentiment: string;
  reasoning: string[];
}

export interface SmaSentimentBacktestResult {
  symbol: string;
  date: string;
  smaPeriod: number;
  initialCapital: number;
  finalCapital: number;
  totalReturn: number;
  totalReturnPercent: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  profitFactor: number;
  trades: BacktestTrade[];
  priceData: BacktestPricePoint[];
  tradeMarkers: BacktestTradeMarker[];
  sentimentLog: Array<{ time: string; sentiment: string; reasoning: string[] }>;
}

interface SentimentAnalysis {
  overallSentiment: 'bullish' | 'bearish' | 'neutral';
  dailyCallPutRatio: number;
  flowSentiment: 'bullish' | 'bearish' | 'neutral';
  aggregatedSentiment: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  reasoning: string[];
}

async function analyzeSentimentAtTime(
  symbol: string,
  currentTime: Date,
  date: string
): Promise<SentimentAnalysis> {
  const reasoning: string[] = [];

  // Get the day's official call/put ratio
  const dailyCallPutRatio = await getPutCallRatio(symbol);
  reasoning.push(`Daily Call/Put Ratio: ${dailyCallPutRatio.toFixed(2)}`);

  // Get options flow data from last 10 minutes before current time
  const flowAlerts = await getWhaleFlowAlerts({
    symbols: [symbol],
    lookbackMinutes: 10,
    minPremium: 50000,
    date,
  });

  // Filter flow alerts to only include those before current time
  const relevantAlerts = flowAlerts.filter(alert => {
    const alertTime = new Date(alert.timestamp);
    return alertTime <= currentTime;
  });

  // Aggregate sentiment from options flow
  let bullishFlow = 0;
  let bearishFlow = 0;
  let totalFlowValue = 0;

  relevantAlerts.forEach(alert => {
    const value = alert.premium;
    totalFlowValue += value;

    if (alert.optionType === 'call') {
      if (alert.direction === 'bullish') {
        bullishFlow += value;
      } else if (alert.direction === 'bearish') {
        bearishFlow += value;
      }
    } else if (alert.optionType === 'put') {
      if (alert.direction === 'bullish') {
        bearishFlow += value;
      } else if (alert.direction === 'bearish') {
        bullishFlow += value;
      }
    }
  });

  // Calculate flow sentiment
  let flowSentiment: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (totalFlowValue > 0) {
    const bullishPercentage = bullishFlow / totalFlowValue;
    const bearishPercentage = bearishFlow / totalFlowValue;

    if (bullishPercentage > 0.6) {
      flowSentiment = 'bullish';
    } else if (bearishPercentage > 0.6) {
      flowSentiment = 'bearish';
    }

    reasoning.push(`Flow: ${(bullishPercentage * 100).toFixed(0)}% bull, ${(bearishPercentage * 100).toFixed(0)}% bear`);
  }

  // Determine daily sentiment from call/put ratio
  let aggregatedSentiment: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (dailyCallPutRatio < 0.7) {
    aggregatedSentiment = 'bullish';
    reasoning.push('Daily: bullish (more calls)');
  } else if (dailyCallPutRatio > 1.3) {
    aggregatedSentiment = 'bearish';
    reasoning.push('Daily: bearish (more puts)');
  }

  // Combine sentiments
  let overallSentiment: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  let confidence = 0.5;

  if (flowSentiment === 'bullish' && aggregatedSentiment === 'bullish') {
    overallSentiment = 'bullish';
    confidence = 0.9;
    reasoning.push('✓ STRONG BULLISH');
  } else if (flowSentiment === 'bearish' && aggregatedSentiment === 'bearish') {
    overallSentiment = 'bearish';
    confidence = 0.9;
    reasoning.push('✓ STRONG BEARISH');
  } else if (flowSentiment === 'bullish' || aggregatedSentiment === 'bullish') {
    overallSentiment = 'bullish';
    confidence = 0.6;
  } else if (flowSentiment === 'bearish' || aggregatedSentiment === 'bearish') {
    overallSentiment = 'bearish';
    confidence = 0.6;
  }

  return {
    overallSentiment,
    dailyCallPutRatio,
    flowSentiment,
    aggregatedSentiment,
    confidence,
    reasoning,
  };
}

export async function runSmaSentimentBacktest(
  config: SmaSentimentBacktestConfig
): Promise<SmaSentimentBacktestResult> {
  const {
    symbol,
    date,
    smaPeriod = 20,
    initialCapital = 10000,
    positionSize = 0.1,
    exitAfterMinutes = 30,
  } = config;

  console.log(`\n🔄 Starting SMA Sentiment Backtest for ${symbol} on ${date}`);
  console.log(`   SMA Period: ${smaPeriod}, Position Size: ${(positionSize * 100).toFixed(0)}%`);

  // Get 5-minute historical data
  const bars = await getHistoricalTimesales(symbol, date, 5, 'all');

  if (bars.length < smaPeriod + 1) {
    throw new Error(`Insufficient data: got ${bars.length} bars, need at least ${smaPeriod + 1}`);
  }

  console.log(`   Loaded ${bars.length} 5-minute bars`);

  // Calculate SMA for all bars
  const closes = bars.map(b => b.close);
  const smaValues = SMA.calculate({ values: closes, period: smaPeriod });

  // Build price data for visualization
  const priceData: BacktestPricePoint[] = [];
  for (let i = smaPeriod - 1; i < bars.length; i++) {
    priceData.push({
      time: bars[i].time,
      price: bars[i].close,
      sma: smaValues[i - smaPeriod + 1],
    });
  }

  // Backtest logic
  const trades: BacktestTrade[] = [];
  const tradeMarkers: BacktestTradeMarker[] = [];
  const sentimentLog: Array<{ time: string; sentiment: string; reasoning: string[] }> = [];

  let currentCapital = initialCapital;
  let currentPosition: {
    type: 'call' | 'put';
    entryTime: string;
    entryPrice: number;
    contracts: number;
    strike: number;
    expiration: string;
    sentiment: string;
    confidence: number;
    entryIndex: number;
  } | null = null;

  // Iterate through bars looking for crossovers
  for (let i = smaPeriod; i < bars.length; i++) {
    const currentBar = bars[i];
    const previousBar = bars[i - 1];
    const currentSma = smaValues[i - smaPeriod + 1];
    const previousSma = smaValues[i - smaPeriod];

    const currentPrice = currentBar.close;
    const previousPrice = previousBar.close;
    const currentTime = new Date(`${date}T${currentBar.time}`);

    // Check if we need to exit current position
    if (currentPosition) {
      const entryTime = new Date(`${date}T${currentPosition.entryTime}`);
      const minutesInTrade = (currentTime.getTime() - entryTime.getTime()) / (1000 * 60);

      if (minutesInTrade >= exitAfterMinutes) {
        // Exit the trade
        const exitPrice = currentPrice * 0.8; // Simplified: assume option lost some value
        const pnl = (exitPrice - currentPosition.entryPrice) * currentPosition.contracts * 100;
        currentCapital += pnl;

        trades.push({
          entryTime: currentPosition.entryTime,
          exitTime: currentBar.time,
          type: currentPosition.type,
          strike: currentPosition.strike,
          expiration: currentPosition.expiration,
          entryPrice: currentPosition.entryPrice,
          exitPrice,
          contracts: currentPosition.contracts,
          entryReason: `SMA crossover with ${currentPosition.sentiment} sentiment`,
          exitReason: `Time exit (${exitAfterMinutes} min)`,
          pnl,
          pnlPercent: (pnl / (currentPosition.entryPrice * currentPosition.contracts * 100)) * 100,
          sentiment: currentPosition.sentiment,
          confidence: currentPosition.confidence,
        });

        tradeMarkers.push({
          time: currentBar.time,
          price: currentPrice,
          type: 'sell',
          optionType: currentPosition.type,
          strike: currentPosition.strike,
          premium: exitPrice,
          sentiment: currentPosition.sentiment,
          reasoning: [`Exit after ${exitAfterMinutes} min`, `P&L: $${pnl.toFixed(2)}`],
        });

        currentPosition = null;
      }
    }

    // Detect crossovers
    const crossesUp = previousPrice <= previousSma && currentPrice > currentSma;
    const crossesDown = previousPrice >= previousSma && currentPrice < currentSma;

    if (!crossesUp && !crossesDown) {
      continue;
    }

    const direction = crossesUp ? 'bullish' : 'bearish';

    // Analyze sentiment
    const sentiment = await analyzeSentimentAtTime(symbol, currentTime, date);

    sentimentLog.push({
      time: currentBar.time,
      sentiment: `${direction} crossover | Sentiment: ${sentiment.overallSentiment}`,
      reasoning: sentiment.reasoning,
    });

    console.log(`\n   ${currentBar.time}: ${direction.toUpperCase()} crossover detected`);
    console.log(`      Sentiment: ${sentiment.overallSentiment} (${(sentiment.confidence * 100).toFixed(0)}%)`);

    // Check if sentiment aligns
    if (direction !== sentiment.overallSentiment) {
      console.log(`      ✗ Skipped - sentiment mismatch`);
      continue;
    }

    console.log(`      ✓ Trade approved - sentiment aligns`);

    // Don't enter new position if we already have one
    if (currentPosition) {
      console.log(`      ✗ Already in position`);
      continue;
    }

    // Enter new position
    const optionType = crossesUp ? 'call' : 'put';
    const optionsChain = await getOptionsChain(symbol, undefined, date);

    // Find closest ITM option
    const inTheMoneyOptions = optionsChain.filter(o => {
      if (o.type !== optionType) return false;
      if (optionType === 'call') {
        return o.strike < currentPrice;
      } else {
        return o.strike > currentPrice;
      }
    });

    if (inTheMoneyOptions.length === 0) {
      console.log(`      ✗ No ITM options available`);
      continue;
    }

    inTheMoneyOptions.sort((a, b) => {
      if (optionType === 'call') {
        return b.strike - a.strike;
      } else {
        return a.strike - b.strike;
      }
    });

    const bestOption = inTheMoneyOptions[0];
    const entryPrice = bestOption.premium;
    const positionValue = currentCapital * positionSize;
    const contracts = Math.floor(positionValue / (entryPrice * 100));

    if (contracts < 1) {
      console.log(`      ✗ Insufficient capital for 1 contract`);
      continue;
    }

    currentPosition = {
      type: optionType,
      entryTime: currentBar.time,
      entryPrice,
      contracts,
      strike: bestOption.strike,
      expiration: bestOption.expiration,
      sentiment: sentiment.overallSentiment,
      confidence: sentiment.confidence,
      entryIndex: i,
    };

    tradeMarkers.push({
      time: currentBar.time,
      price: currentPrice,
      type: 'buy',
      optionType,
      strike: bestOption.strike,
      premium: entryPrice,
      sentiment: sentiment.overallSentiment,
      reasoning: sentiment.reasoning.slice(0, 3),
    });

    console.log(`      Entry: ${contracts} x ${optionType.toUpperCase()} @ $${entryPrice.toFixed(2)}`);
  }

  // Close any remaining position at market close
  if (currentPosition) {
    const lastBar = bars[bars.length - 1];
    const exitPrice = lastBar.close * 0.7; // Assume option decayed
    const pnl = (exitPrice - currentPosition.entryPrice) * currentPosition.contracts * 100;
    currentCapital += pnl;

    trades.push({
      entryTime: currentPosition.entryTime,
      exitTime: lastBar.time,
      type: currentPosition.type,
      strike: currentPosition.strike,
      expiration: currentPosition.expiration,
      entryPrice: currentPosition.entryPrice,
      exitPrice,
      contracts: currentPosition.contracts,
      entryReason: `SMA crossover with ${currentPosition.sentiment} sentiment`,
      exitReason: 'Market close',
      pnl,
      pnlPercent: (pnl / (currentPosition.entryPrice * currentPosition.contracts * 100)) * 100,
      sentiment: currentPosition.sentiment,
      confidence: currentPosition.confidence,
    });

    tradeMarkers.push({
      time: lastBar.time,
      price: lastBar.close,
      type: 'sell',
      optionType: currentPosition.type,
      strike: currentPosition.strike,
      premium: exitPrice,
      sentiment: currentPosition.sentiment,
      reasoning: ['Market close', `P&L: $${pnl.toFixed(2)}`],
    });
  }

  // Calculate statistics
  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl <= 0);
  const totalWins = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
  const totalLosses = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));

  const result: SmaSentimentBacktestResult = {
    symbol,
    date,
    smaPeriod,
    initialCapital,
    finalCapital: currentCapital,
    totalReturn: currentCapital - initialCapital,
    totalReturnPercent: ((currentCapital - initialCapital) / initialCapital) * 100,
    totalTrades: trades.length,
    winningTrades: winningTrades.length,
    losingTrades: losingTrades.length,
    winRate: trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0,
    avgWin: winningTrades.length > 0 ? totalWins / winningTrades.length : 0,
    avgLoss: losingTrades.length > 0 ? totalLosses / losingTrades.length : 0,
    largestWin: winningTrades.length > 0 ? Math.max(...winningTrades.map(t => t.pnl)) : 0,
    largestLoss: losingTrades.length > 0 ? Math.min(...losingTrades.map(t => t.pnl)) : 0,
    profitFactor: totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? Infinity : 0,
    trades,
    priceData,
    tradeMarkers,
    sentimentLog,
  };

  console.log(`\n✅ Backtest Complete!`);
  console.log(`   Total Trades: ${result.totalTrades}`);
  console.log(`   Win Rate: ${result.winRate.toFixed(1)}%`);
  console.log(`   Total Return: $${result.totalReturn.toFixed(2)} (${result.totalReturnPercent.toFixed(2)}%)`);

  // Save results to file
  const outputPath = path.join(process.cwd(), `backtest_sma_sentiment_${symbol}_${date}.json`);
  await fs.writeFile(outputPath, JSON.stringify(result, null, 2));
  console.log(`   Results saved to: ${outputPath}`);

  return result;
}
