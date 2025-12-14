import { NextRequest, NextResponse } from 'next/server';
import { backtestMeanReversionMultiple, MeanReversionTrade } from '@/lib/meanReversionBacktester';

interface TradeDetail {
  tradeNumber: number;
  direction: 'CALL' | 'PUT';
  entry: {
    time: string;
    stockPrice: number;
    optionStrike: number | null;
    optionPremium: number | null;
    totalCost: number | null;
    rsi: number | null;
  };
  exit: {
    time: string;
    stockPrice: number;
    optionPremium: number | null;
    totalValue: number | null;
    reason: string;
  };
  stockMove: number;
  stockMovePercent: number;
  optionPnL: number;
  optionReturnPercent: number | null;
  holdMinutes: number;
  optionContract: string | null;
  date: string;
}

interface BacktestStats {
  symbol: string;
  startDate: string;
  endDate: string;
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
  bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  netGex: number;
  hasNegativeGex: boolean;
}

interface BacktestResponse {
  stats: BacktestStats;
  trades: TradeDetail[];
  rawResults: any[];
}

function formatTimeToEastern(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
    timeZone: 'America/New_York',
  });
}

function getRecentTradingDates(days: number, offsetDays: number = 1): string[] {
  const results: string[] = [];
  const cursor = new Date();
  cursor.setUTCHours(0, 0, 0, 0);
  cursor.setDate(cursor.getDate() - offsetDays);

  while (results.length < days) {
    const day = cursor.getUTCDay();
    if (day !== 0 && day !== 6) {
      results.push(cursor.toISOString().split('T')[0]);
    }
    cursor.setDate(cursor.getDate() - 1);
  }

  return results;
}

function aggregateTrades(trades: TradeDetail[]) {
  const totalPnL = trades.reduce((sum, t) => sum + t.optionPnL, 0);
  const winners = trades.filter(t => t.optionPnL > 0);
  const losers = trades.filter(t => t.optionPnL <= 0);
  const winRate = trades.length > 0 ? (winners.length / trades.length) * 100 : 0;

  const avgWin = winners.length > 0 ? winners.reduce((sum, t) => sum + t.optionPnL, 0) / winners.length : 0;
  const avgLoss = losers.length > 0 ? losers.reduce((sum, t) => sum + t.optionPnL, 0) / losers.length : 0;
  const avgPnL = trades.length > 0 ? totalPnL / trades.length : 0;
  const largestWin = winners.length > 0 ? Math.max(...winners.map(t => t.optionPnL)) : 0;
  const largestLoss = losers.length > 0 ? Math.min(...losers.map(t => t.optionPnL)) : 0;
  const avgHold = trades.length > 0 ? trades.reduce((sum, t) => sum + t.holdMinutes, 0) / trades.length : 0;

  const grossProfit = winners.reduce((sum, t) => sum + t.optionPnL, 0);
  const grossLoss = losers.reduce((sum, t) => sum + t.optionPnL, 0);
  const profitFactor = grossLoss < 0 ? grossProfit / Math.abs(grossLoss) : 0;

  const callCount = trades.filter(t => t.direction === 'CALL').length;
  const putCount = trades.filter(t => t.direction === 'PUT').length;
  const callPutRatio = putCount > 0 ? callCount / putCount : callCount;
  let bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
  if (callCount > putCount) bias = 'BULLISH';
  else if (putCount > callCount) bias = 'BEARISH';

  return {
    totalPnL,
    winners: winners.length,
    losers: losers.length,
    winRate,
    avgWin,
    avgLoss,
    avgPnL,
    largestWin,
    largestLoss,
    avgHoldMinutes: avgHold,
    profitFactor,
    callPutRatio,
    bias,
  };
}

function mapTrade(trade: MeanReversionTrade, date: string, index: number): TradeDetail {
  const direction = trade.direction === 'long' ? 'CALL' : 'PUT';
  const entryPremium = trade.option.entryPremium ?? null;
  const exitPremium = trade.option.exitPremium ?? null;
  const optionPnL = trade.option.profit ?? 0;
  const optionReturn = trade.option.profitPct ?? null;
  const optionStrike = trade.option.strike ?? null;

  return {
    tradeNumber: index + 1,
    direction,
    entry: {
      time: formatTimeToEastern(trade.entryTimestamp),
      stockPrice: trade.stock.entryPrice,
      optionStrike,
      optionPremium: entryPremium,
      totalCost: entryPremium !== null ? entryPremium * 100 : null,
      rsi: trade.entryRSI,
    },
    exit: {
      time: formatTimeToEastern(trade.exitTimestamp),
      stockPrice: trade.stock.exitPrice,
      optionPremium: exitPremium,
      totalValue: exitPremium !== null ? exitPremium * 100 : null,
      reason: trade.exitReason,
    },
    stockMove: trade.stock.exitPrice - trade.stock.entryPrice,
    stockMovePercent: ((trade.stock.exitPrice - trade.stock.entryPrice) / trade.stock.entryPrice) * 100,
    optionPnL,
    optionReturnPercent: optionReturn,
    holdMinutes: trade.durationMinutes,
    optionContract: trade.option.contract,
    date,
  };
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbolParam = searchParams.get('symbol');
    if (!symbolParam) {
      return NextResponse.json({ error: 'Missing symbol parameter' }, { status: 400 });
    }

    const symbol = symbolParam.toUpperCase();
    const days = Number.parseInt(searchParams.get('days') || '2', 10);
    const interval = Number.parseInt(searchParams.get('interval') || '15', 10);
    const intrabar = Number.parseInt(searchParams.get('intrabar') || '1', 10);

    const singleDate = searchParams.get('date');
    const dates = singleDate ? [singleDate] : getRecentTradingDates(Math.max(days, 1));

    const results = await backtestMeanReversionMultiple([symbol], dates, 'intraday', interval, intrabar);

    if (!results || results.length === 0) {
      return NextResponse.json({ error: 'No backtest data generated' }, { status: 404 });
    }

    const trades: TradeDetail[] = [];
    results.forEach(result => {
      result.trades.forEach((trade, idx) => {
        trades.push(mapTrade(trade, result.date, trades.length));
      });
    });

    const statsData = aggregateTrades(trades);
    const earliestDate = results[results.length - 1].date;
    const latestDate = results[0].date;
    const avgNetGex =
      results.reduce((sum, r) => sum + (Number.isFinite(r.netGex) ? r.netGex : 0), 0) / results.length;

    const stats: BacktestStats = {
      symbol,
      startDate: earliestDate,
      endDate: latestDate,
      totalTrades: trades.length,
      winners: statsData.winners,
      losers: statsData.losers,
      winRate: statsData.winRate,
      totalPnL: statsData.totalPnL,
      avgPnL: statsData.avgPnL,
      avgWin: statsData.avgWin,
      avgLoss: statsData.avgLoss,
      largestWin: statsData.largestWin,
      largestLoss: statsData.largestLoss,
      avgHoldMinutes: statsData.avgHoldMinutes,
      profitFactor: statsData.profitFactor,
      callPutRatio: statsData.callPutRatio,
      bias: statsData.bias,
      netGex: avgNetGex,
      hasNegativeGex: avgNetGex < 0,
    };

    const response: BacktestResponse = {
      stats,
      trades,
      rawResults: results,
    };

    return NextResponse.json(response);
  } catch (error: any) {
    console.error('[mean-reversion/backtest] error:', error);
    return NextResponse.json(
      { error: error?.message || 'Mean reversion backtest failed' },
      { status: 500 },
    );
  }
}
