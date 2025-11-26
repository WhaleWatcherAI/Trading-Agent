import { NextRequest, NextResponse } from 'next/server';
import { SMA } from 'technicalindicators';
import { getHistoricalData } from '@/lib/technicals';
import { getOptionsChain, getCallPutRatioAllExpirations } from '@/lib/tradier';
import { getTickerCallPutRatio } from '@/lib/unusualwhales';
import { calculateGexForSymbol } from '@/lib/gexCalculator';
import { analyzeSentiment } from '@/lib/sentimentAnalyzer';

interface TradeDetail {
  tradeNumber: number;
  direction: 'CALL' | 'PUT';
  entry: {
    time: string;
    stockPrice: number;
    optionStrike: number;
    optionPremium: number;
    totalCost: number;
    sma: number;
  };
  exit: {
    time: string;
    stockPrice: number;
    optionPremium: number;
    totalValue: number;
    reason: string;
  };
  stockMove: number;
  stockMovePercent: number;
  optionPnL: number;
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
  profitFactor: number;
  callPutRatio: number;
  bias: string;
  netGex: number;
  hasNegativeGex: boolean;
  sentiment: string;
  sentimentScore: number;
}

const SMA_PERIOD = 20;
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

async function backtestSingleStock(symbol: string, testDate: string): Promise<{ trades: TradeDetail[], stats: BacktestStats }> {
  console.log(`\n🔍 Starting backtest for ${symbol} on date: ${testDate}`);

  const history = await getHistoricalData(symbol, '1min', 2, testDate); // 2 days of data
  console.log(`📈 Retrieved ${history.length} bars of historical data`);

  if (history.length > 0) {
    const firstBar = new Date(Number(history[0].date) * 1000).toLocaleDateString('en-US');
    const lastBar = new Date(Number(history[history.length - 1].date) * 1000).toLocaleDateString('en-US');
    console.log(`📅 Data range: ${firstBar} to ${lastBar}`);
  }

  // Get sentiment analysis from news headlines
  const sentimentAnalysis = await analyzeSentiment(symbol, testDate);

  console.log(`💭 ${symbol} Sentiment: ${sentimentAnalysis.sentiment} (${sentimentAnalysis.score}/10)`);
  console.log(`   Reasoning: ${sentimentAnalysis.reasoning}`);
  console.log(`   News Count: ${sentimentAnalysis.newsCount}`);

  // Get MARKET-WIDE call/put ratio from Unusual Whales (for stats only, not filtering)
  const ratioData = await getTickerCallPutRatio(symbol, testDate);
  const callPutRatio = ratioData.callPutRatio;

  // Calculate GEX for THIS specific ticker
  const gexData = await calculateGexForSymbol(symbol, 'intraday', testDate);
  const netGex = gexData?.summary?.netGex ?? 0;
  const hasNegativeGex = netGex < 0;

  console.log(`⚡ ${symbol} Net GEX: ${netGex.toFixed(2)} (${hasNegativeGex ? 'NEGATIVE - Trade OK ✅' : 'POSITIVE - Skip trades ❌'})`);

  // Filter 1: Only proceed if net GEX is negative
  if (!hasNegativeGex) {
    console.log(`⏭️  Skipping ${symbol} - Net GEX is positive (${netGex.toFixed(2)})`);

    const stats: BacktestStats = {
      symbol,
      totalTrades: 0,
      winners: 0,
      losers: 0,
      winRate: 0,
      totalPnL: 0,
      avgPnL: 0,
      avgWin: 0,
      avgLoss: 0,
      largestWin: 0,
      largestLoss: 0,
      avgHoldMinutes: 0,
      profitFactor: 0,
      callPutRatio,
      bias: ratioData.bias,
      netGex,
      hasNegativeGex,
      sentiment: sentimentAnalysis.sentiment,
      sentimentScore: sentimentAnalysis.score,
    };

    return { trades: [], stats };
  }

  // Filter 2: Only proceed if sentiment score is 8 or higher
  if (sentimentAnalysis.score < 8) {
    console.log(`⏭️  Skipping ${symbol} - Sentiment score too low (${sentimentAnalysis.score}/10, need 8+)`);

    const stats: BacktestStats = {
      symbol,
      totalTrades: 0,
      winners: 0,
      losers: 0,
      winRate: 0,
      totalPnL: 0,
      avgPnL: 0,
      avgWin: 0,
      avgLoss: 0,
      largestWin: 0,
      largestLoss: 0,
      avgHoldMinutes: 0,
      profitFactor: 0,
      callPutRatio,
      bias: ratioData.bias,
      netGex,
      hasNegativeGex,
      sentiment: sentimentAnalysis.sentiment,
      sentimentScore: sentimentAnalysis.score,
    };

    return { trades: [], stats };
  }

  console.log(`✅ ${symbol} passed filters: Negative GEX + Sentiment ${sentimentAnalysis.score}/10`);

  // Get options chain for trading (only nearest expiration)
  const optionsChain = await getOptionsChain(symbol, undefined, testDate);
  console.log(`⚙️  Retrieved ${optionsChain.length} options from nearest expiration`);

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

        // Filter: Only take trades when SMA direction matches sentiment direction
        const signalIsBullish = crossesUp;
        const signalIsBearish = crossesDown;
        const sentimentIsBullish = sentimentAnalysis.sentiment === 'BULLISH';
        const sentimentIsBearish = sentimentAnalysis.sentiment === 'BEARISH';

        if (signalIsBullish && !sentimentIsBullish) {
          console.log(`⏭️  Skipping CALL trade - sentiment is ${sentimentAnalysis.sentiment}, not BULLISH`);
          continue;
        }

        if (signalIsBearish && !sentimentIsBearish) {
          console.log(`⏭️  Skipping PUT trade - sentiment is ${sentimentAnalysis.sentiment}, not BEARISH`);
          continue;
        }

        console.log(`✅ Taking ${direction} trade - SMA ${crossesUp ? 'UP' : 'DOWN'} matches ${sentimentAnalysis.sentiment} sentiment (${sentimentAnalysis.score}/10)`);

        let underlyingEntryPrice: number;
        if (crossesUp) {
          underlyingEntryPrice = currentBar.open > currentSma ? currentBar.open : currentSma;
        } else {
          underlyingEntryPrice = currentBar.open < currentSma ? currentBar.open : currentSma;
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

      // Check GEX zone crossing
      const currentGexZone = getGexZoneForPrice(currentPrice);
      const previousGexZone = currentTrade.previousGexZone;
      const gexWallCrossing = previousGexZone === 'POSITIVE' && (currentGexZone === 'NEUTRAL' || currentGexZone === 'NEGATIVE');

      let exitReason: string | null = null;
      if (oppositeCross) exitReason = 'Opposite Signal';
      else if (gexWallCrossing) exitReason = 'GEX Wall Cross';
      else if (pnlPercent <= STOP_LOSS_PERCENT) exitReason = 'Stop Loss';
      else if (pnlPercent >= TAKE_PROFIT_PERCENT) exitReason = 'Take Profit';
      else if (i === history.length - 1) exitReason = 'End of Day';

      // Update previous GEX zone for next iteration
      currentTrade.previousGexZone = currentGexZone;

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
    callPutRatio,
    bias: isBullishBias ? 'BULLISH' : isBearishBias ? 'BEARISH' : 'NEUTRAL',
    netGex,
    hasNegativeGex,
  };

  console.log(`✅ Backtest complete: ${trades.length} trades found\n`);

  return { trades, stats };
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const symbol = searchParams.get('symbol') || 'TSLA';
    const date = searchParams.get('date') || '2025-11-04';

    const result = await backtestSingleStock(symbol, date);

    return NextResponse.json(result);
  } catch (error: any) {
    console.error('Backtest failed:', error);
    return NextResponse.json(
      { error: error.message || 'Backtest failed' },
      { status: 500 }
    );
  }
}
