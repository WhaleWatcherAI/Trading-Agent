/**
 * Keltner Channel + RSI Scalping Backtester for TopstepX Futures
 *
 * Strategy:
 * - Entry: RSI < 25 (long) or > 75 (short) + price outside Keltner(20, 1.5)
 * - Exit: Middle Keltner band OR RSI back to 50
 * - Filter: Skip if ADX > 25 (trending market)
 */

import { fetchTopstepXFuturesBars, fetchTopstepXContract } from './topstepx';
import { generateKeltnerScalpSignal, checkKeltnerScalpExit } from './keltnerScalpAgent';
import { RSI } from 'technicalindicators';

export interface KeltnerScalpTrade {
  id: string;
  contractId: string;
  direction: 'long' | 'short';
  entryTimestamp: string;
  exitTimestamp: string;
  entryPrice: number;
  exitPrice: number;
  pointsGained: number;
  ticksGained: number;
  grossProfit: number;
  netProfit: number;
  entryRSI: number | null;
  exitRSI: number | null;
  entryADX: number | null;
  exitReason: 'stop' | 'target' | 'rsi_neutral' | 'end_of_session';
  durationMinutes: number;
}

export interface KeltnerScalpBacktestResult {
  contractId: string;
  contractName: string;
  startDate: string;
  endDate: string;
  barSize: number;
  tickSize: number;
  tickValue: number;
  commission: number;
  trades: KeltnerScalpTrade[];
  summary: {
    totalTrades: number;
    winCount: number;
    lossCount: number;
    winRate: number;
    totalNetProfit: number;
    totalGrossProfit: number;
    totalCommission: number;
    avgWin: number;
    avgLoss: number;
    profitFactor: number;
    avgDurationMinutes: number;
    maxDrawdown: number;
    skippedByADX: number;
  };
}

function buildIntrabarSegments(
  coarseBars: any[],
  intrabarBars: any[],
  coarseMinutes: number,
  intrabarMinutes: number,
): any[][] {
  if (
    coarseMinutes <= intrabarMinutes ||
    intrabarBars === coarseBars ||
    intrabarBars.length === 0
  ) {
    return coarseBars.map(bar => [bar]);
  }

  const segments: any[][] = coarseBars.map(() => []);
  const coarseStartTimes = coarseBars.map(bar => new Date(bar.timestamp).getTime());
  const coarseEndTimes = coarseStartTimes.map((start, idx) => {
    if (idx < coarseStartTimes.length - 1) {
      return coarseStartTimes[idx + 1];
    }
    return start + coarseMinutes * 60 * 1000;
  });

  let coarseIndex = 0;
  for (const miniBar of intrabarBars) {
    const miniTime = new Date(miniBar.timestamp).getTime();
    if (!Number.isFinite(miniTime)) {
      continue;
    }

    while (coarseIndex < coarseBars.length && miniTime >= coarseEndTimes[coarseIndex]) {
      coarseIndex += 1;
    }

    if (coarseIndex >= coarseBars.length) {
      break;
    }

    if (miniTime >= coarseStartTimes[coarseIndex]) {
      segments[coarseIndex].push(miniBar);
    }
  }

  return segments.map((segment, idx) => (segment.length ? segment : [coarseBars[idx]]));
}

/**
 * Backtest Keltner scalping strategy on TopstepX futures
 */
export async function backtestKeltnerScalp(
  contractId: string,
  startDate: string,
  endDate: string,
  barSizeMinutes: number = 5,
  commission: number = 0.62,
): Promise<KeltnerScalpBacktestResult> {
  console.log(`\n⚡ Running Keltner Scalp backtest for ${contractId}`);
  console.log(`   Period: ${startDate} to ${endDate}`);
  console.log(`   Bar size: ${barSizeMinutes} minutes`);

  // Fetch contract details
  const contract = await fetchTopstepXContract(contractId);
  if (!contract) {
    throw new Error(`Contract ${contractId} not found`);
  }

  const tickSize = contract.tickSize;
  const tickValue = contract.tickValue;

  console.log(`   Contract: ${contract.name}`);
  console.log(`   Tick: $${tickSize} = $${tickValue}`);

  // Fetch historical bars
  const bars = await fetchTopstepXFuturesBars({
    contractId,
    startTime: startDate,
    endTime: endDate,
    unit: 2,
    unitNumber: barSizeMinutes,
    limit: 20000,
    live: false,
  });

  if (bars.length === 0) {
    throw new Error(`No historical data for ${contractId}`);
  }

  console.log(`   Loaded ${bars.length} bars`);
  bars.reverse();

  // Fetch 1-minute bars for precise stop/target monitoring
  const intrabarMinutes = 1;
  let intrabarBars: any[] = bars;

  if (barSizeMinutes > intrabarMinutes) {
    try {
      const miniBars = await fetchTopstepXFuturesBars({
        contractId,
        startTime: startDate,
        endTime: endDate,
        unit: 2,
        unitNumber: intrabarMinutes,
        limit: 20000,
        live: false,
      });

      if (miniBars.length) {
        console.log(`   Loaded ${miniBars.length} 1-min bars for stop/target checks`);
        miniBars.reverse();
        intrabarBars = miniBars;
      }
    } catch (error: any) {
      console.warn('   Failed to fetch 1-min bars, using coarse bars');
    }
  }

  const intrabarSegments = buildIntrabarSegments(
    bars,
    intrabarBars,
    barSizeMinutes,
    intrabarMinutes,
  );

  const trades: KeltnerScalpTrade[] = [];
  let activePosition: {
    signal: any;
    entryTimestamp: string;
    entryPrice: number;
    entryRSI: number | null;
    entryADX: number | null;
  } | null = null;

  let tradeIdCounter = 1;
  let skippedByADX = 0;

  // Simulation loop
  for (let i = 30; i < bars.length; i++) {
    const bar = bars[i];
    const barTimestamp = bar.timestamp;
    const barClose = bar.close;
    const priceHistory = bars.slice(0, i + 1).map((b: any) => ({
      high: b.high,
      low: b.low,
      close: b.close,
    }));

    // Calculate current RSI
    const closes = priceHistory.map((p: any) => p.close);
    const rsiValues = RSI.calculate({ values: closes, period: 14 });
    const currentRSI = rsiValues[rsiValues.length - 1] || null;

    // Exit monitoring
    if (activePosition && currentRSI) {
      const intrabars = intrabarSegments[i] || [bar];
      let exitTriggered = false;
      let exitPrice = barClose;
      let exitTimestamp = barTimestamp;
      let exitReason: KeltnerScalpTrade['exitReason'] = 'end_of_session';

      for (const miniBar of intrabars) {
        const miniHigh = miniBar.high ?? miniBar.close;
        const miniLow = miniBar.low ?? miniBar.close;
        const miniClose = miniBar.close;
        const miniTimestamp = miniBar.timestamp;

        // Get current RSI for this minibar
        const miniRsiHistory = bars.slice(0, i + 1).map((b: any) => b.close);
        const miniRsiValues = RSI.calculate({ values: miniRsiHistory, period: 14 });
        const miniRSI = miniRsiValues[miniRsiValues.length - 1] || currentRSI;

        const exitCheck = checkKeltnerScalpExit(
          miniClose,
          miniRSI,
          {
            direction: activePosition.signal.direction,
            entryPrice: activePosition.entryPrice,
            target: activePosition.signal.target,
            stopLoss: activePosition.signal.stopLoss,
          },
          { rsiNeutral: 50 },
        );

        if (exitCheck) {
          exitTriggered = true;
          exitPrice = exitCheck.exitPrice;
          exitReason = exitCheck.reason as any;
          exitTimestamp = miniTimestamp;
          break;
        }
      }

      if (i === bars.length - 1 && !exitTriggered) {
        exitTriggered = true;
        exitReason = 'end_of_session';
        const lastMini = intrabars[intrabars.length - 1];
        exitTimestamp = lastMini?.timestamp ?? barTimestamp;
        exitPrice = lastMini?.close ?? barClose;
      }

      if (exitTriggered) {
        const direction = activePosition.signal.direction;
        const pointsGained =
          direction === 'long'
            ? exitPrice - activePosition.entryPrice
            : activePosition.entryPrice - exitPrice;
        const ticksGained = pointsGained / tickSize;
        const grossProfit = ticksGained * tickValue;
        const netProfit = grossProfit - commission * 2;

        const entryDate = new Date(activePosition.entryTimestamp);
        const exitDate = new Date(exitTimestamp);
        const durationMinutes = (exitDate.getTime() - entryDate.getTime()) / (1000 * 60);

        trades.push({
          id: `${contractId}-${tradeIdCounter++}`,
          contractId,
          direction,
          entryTimestamp: activePosition.entryTimestamp,
          exitTimestamp,
          entryPrice: activePosition.entryPrice,
          exitPrice,
          pointsGained,
          ticksGained,
          grossProfit,
          netProfit,
          entryRSI: activePosition.entryRSI,
          exitRSI: currentRSI,
          entryADX: activePosition.entryADX,
          exitReason,
          durationMinutes,
        });

        activePosition = null;
      }
    }

    // Entry monitoring
    if (!activePosition) {
      const entrySignal = generateKeltnerScalpSignal(
        contractId,
        barClose,
        priceHistory,
        {
          rsiPeriod: 14,
          rsiOversold: 25,
          rsiOverbought: 75,
          keltnerPeriod: 20,
          keltnerMultiplier: 1.5,
          adxPeriod: 14,
          adxThreshold: 25,
          stopLossPercent: 0.002, // 0.2% for scalping
        },
      );

      // Track ADX filter rejections
      if (entrySignal.adx && entrySignal.adx > 25 && entrySignal.direction === 'none') {
        const hasSetup =
          (currentRSI && currentRSI < 25 && barClose < (entrySignal.keltnerLower || 0)) ||
          (currentRSI && currentRSI > 75 && barClose > (entrySignal.keltnerUpper || 0));
        if (hasSetup) {
          skippedByADX++;
        }
      }

      if (entrySignal.direction !== 'none') {
        activePosition = {
          signal: entrySignal,
          entryTimestamp: barTimestamp,
          entryPrice: barClose,
          entryRSI: currentRSI,
          entryADX: entrySignal.adx,
        };
      }
    }
  }

  // Calculate summary
  const wins = trades.filter(t => t.netProfit > 0);
  const losses = trades.filter(t => t.netProfit <= 0);
  const totalNetProfit = trades.reduce((sum, t) => sum + t.netProfit, 0);
  const totalGrossProfit = trades.reduce((sum, t) => sum + t.grossProfit, 0);
  const totalCommission = trades.reduce((sum, t) => sum + commission * 2, 0);
  const avgWin = wins.length > 0 ? wins.reduce((sum, t) => sum + t.netProfit, 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((sum, t) => sum + t.netProfit, 0) / losses.length : 0;
  const winRate = trades.length > 0 ? (wins.length / trades.length) * 100 : 0;
  const profitFactor = Math.abs(avgLoss) > 0 ? avgWin / Math.abs(avgLoss) : 0;
  const avgDurationMinutes =
    trades.length > 0 ? trades.reduce((sum, t) => sum + t.durationMinutes, 0) / trades.length : 0;

  // Max drawdown
  let runningPnL = 0;
  let peak = 0;
  let maxDrawdown = 0;
  for (const trade of trades) {
    runningPnL += trade.netProfit;
    if (runningPnL > peak) {
      peak = runningPnL;
    }
    const drawdown = peak - runningPnL;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }

  console.log(`   ✓ Completed: ${trades.length} trades, ${skippedByADX} skipped by ADX filter`);

  return {
    contractId,
    contractName: contract.name,
    startDate,
    endDate,
    barSize: barSizeMinutes,
    tickSize,
    tickValue,
    commission,
    trades,
    summary: {
      totalTrades: trades.length,
      winCount: wins.length,
      lossCount: losses.length,
      winRate,
      totalNetProfit,
      totalGrossProfit,
      totalCommission,
      avgWin,
      avgLoss,
      profitFactor,
      avgDurationMinutes,
      maxDrawdown,
      skippedByADX,
    },
  };
}
