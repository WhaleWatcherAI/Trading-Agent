/**
 * Mean Reversion Backtest for TopstepX Futures (MES, NQ, ES, etc.)
 * Uses EXACT same logic as stock backtester but for futures contracts
 */

import { fetchTopstepXFuturesBars, fetchTopstepXContract } from './topstepx';
import { generateMeanReversionSignalFromTechnicals, checkMeanReversionExit } from './meanReversionAgent';
import { RSI } from 'technicalindicators';

export interface FuturesTrade {
  id: string;
  contractId: string;
  direction: 'long' | 'short';
  entryTimestamp: string;
  exitTimestamp: string;
  entryPrice: number;
  exitPrice: number;
  pointsGained: number;
  ticksGained: number;
  grossProfit: number;  // Before commission
  netProfit: number;    // After commission
  entryRSI: number | null;
  exitRSI: number | null;
  exitReason: 'stop' | 'target' | 'rsi_neutral' | 'end_of_session' | 'scale';
  durationMinutes: number;
  units: number; // 1 or 2 (for scaling)
  isScaledTrade: boolean; // true if this is the scaled-out half
}

export interface FuturesBacktestResult {
  contractId: string;
  contractName: string;
  startDate: string;
  endDate: string;
  barSize: number;  // minutes
  tickSize: number;
  tickValue: number;
  commission: number;
  trades: FuturesTrade[];
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
    scaledTrades: number;  // Number of times we hit first target
  };
}

const getStopLossPercent = () => 0.001;  // 0.1% stop (default)

function buildIntrabarSegments(
  coarseBars: TopstepXFuturesBar[],
  intrabarBars: TopstepXFuturesBar[],
  coarseMinutes: number,
  intrabarMinutes: number,
): TopstepXFuturesBar[][] {
  if (
    coarseMinutes <= intrabarMinutes ||
    intrabarBars === coarseBars ||
    intrabarBars.length === 0
  ) {
    return coarseBars.map(bar => [bar]);
  }

  const segments: TopstepXFuturesBar[][] = coarseBars.map(() => []);
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
 * Backtest mean reversion strategy on TopstepX futures with scaling
 * EXACT same logic as stock backtest:
 * - Entry: 2 units when RSI oversold/overbought + at BB extreme
 * - First target: BB middle - scale out 1 unit, move stop to just inside middle band
 * - Second target: Opposite outer BB - close remaining 1 unit
 */
export async function backtestFuturesMeanReversion(
  contractId: string,
  startDate: string,
  endDate: string,
  barSizeMinutes: number = 5,
  commission: number = 0.62,  // Per side
): Promise<FuturesBacktestResult> {
  console.log(`\n🔄 Running futures mean reversion backtest for ${contractId}`);
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
    unit: 2,  // Minute
    unitNumber: barSizeMinutes,
    limit: 20000,
    live: false,
  });

  if (bars.length === 0) {
    throw new Error(`No historical data for ${contractId} from ${startDate} to ${endDate}`);
  }

  console.log(`   Loaded ${bars.length} bars`);

  // TopstepX returns bars in descending order (newest first), so reverse them
  bars.reverse();

  const intrabarMinutes = 1;
  let intrabarBars: TopstepXFuturesBar[] = bars;

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
        console.log(`   Loaded ${miniBars.length} intrabar bars for stop/target checks`);
        miniBars.reverse();  // Oldest first
        intrabarBars = miniBars;
      } else {
        console.warn('[topstepx-backtest] Intrabar fetch returned no data, falling back to coarse bars.');
      }
    } catch (error: any) {
      console.warn('[topstepx-backtest] Failed to fetch intrabar bars, falling back to coarse bars:', error.message);
    }
  }

  const intrabarSegments = buildIntrabarSegments(
    bars,
    intrabarBars,
    barSizeMinutes,
    intrabarMinutes,
  );

  const trades: FuturesTrade[] = [];
  const closes = bars.map(b => b.close);
  const highs = bars.map(b => b.high);
  const lows = bars.map(b => b.low);

  let activePosition: {
    signal: any;
    entryTimestamp: string;
    entryPrice: number;
    entryRSI: number | null;
    scaled: boolean;
    units: number;
  } | null = null;

  let tradeIdCounter = 1;
  const stopLossPercent = getStopLossPercent();

  // Simulation loop - EXACT same logic as stock backtest
  for (let i = 20; i < bars.length; i++) {  // Need 20 bars for BB
    const bar = bars[i];
    const barTimestamp = bar.timestamp;
    const barClose = bar.close;
    const priceHistory = closes.slice(0, i + 1);

    // Calculate current RSI
    const rsiValues = RSI.calculate({ values: priceHistory, period: 14 });
    const currentRSI = rsiValues[rsiValues.length - 1] || null;

    // Exit monitoring (if in position)
    if (activePosition) {
      const intrabars = intrabarSegments[i] || [bar];
      let exitTriggered = false;
      let exitPrice = barClose;
      let exitTimestamp = barTimestamp;
      let exitReason: FuturesTrade['exitReason'] = 'end_of_session';

      const direction = activePosition.signal.direction;
      let stopLoss = activePosition.signal.stopLoss;
      let target = activePosition.signal.target;

      for (const miniBar of intrabars) {
        const miniHigh = miniBar.high ?? miniBar.close;
        const miniLow = miniBar.low ?? miniBar.close;
        const miniTimestamp = miniBar.timestamp;

        let reprocessMiniBar = true;
        while (reprocessMiniBar) {
          reprocessMiniBar = false;

          if (!activePosition.scaled && target !== null) {
            const hitTarget =
              (direction === 'long' && miniHigh >= target) ||
              (direction === 'short' && miniLow <= target);

            if (hitTarget) {
              const scaledExitPrice = target;
              const pointsGained = direction === 'long'
                ? scaledExitPrice - activePosition.entryPrice
                : activePosition.entryPrice - scaledExitPrice;
              const ticksGained = pointsGained / tickSize;
              const grossProfit = ticksGained * tickValue;
              const netProfit = grossProfit - (commission * 2);

              const entryDate = new Date(activePosition.entryTimestamp);
              const exitDate = new Date(miniTimestamp);
              const durationMinutes = (exitDate.getTime() - entryDate.getTime()) / (1000 * 60);

              trades.push({
                id: `${contractId}-${tradeIdCounter++}`,
                contractId,
                direction,
                entryTimestamp: activePosition.entryTimestamp,
                exitTimestamp: miniTimestamp,
                entryPrice: activePosition.entryPrice,
                exitPrice: scaledExitPrice,
                pointsGained,
                ticksGained,
                grossProfit,
                netProfit,
                entryRSI: activePosition.entryRSI,
                exitRSI: currentRSI,
                exitReason: 'scale',
                durationMinutes,
                units: 1,
                isScaledTrade: true,
              });

              activePosition.scaled = true;
              activePosition.units = 1;

              if (direction === 'long') {
                const middleBand = target;
                const outerBand = activePosition.signal.bbUpper ?? target;
                activePosition.signal.stopLoss = middleBand * 0.999;
                activePosition.signal.target = outerBand;
              } else {
                const middleBand = target;
                const outerBand = activePosition.signal.bbLower ?? target;
                activePosition.signal.stopLoss = middleBand * 1.001;
                activePosition.signal.target = outerBand;
              }

              stopLoss = activePosition.signal.stopLoss;
              target = activePosition.signal.target;
              reprocessMiniBar = true;
              continue;
            }
          }

          if (direction === 'long') {
            if (stopLoss !== null && miniLow <= stopLoss) {
              exitTriggered = true;
              exitPrice = stopLoss;
              exitReason = 'stop';
              exitTimestamp = miniTimestamp;
            } else if (target !== null && miniHigh >= target) {
              exitTriggered = true;
              exitPrice = target;
              exitReason = 'target';
              exitTimestamp = miniTimestamp;
            }
          } else {
            if (stopLoss !== null && miniHigh >= stopLoss) {
              exitTriggered = true;
              exitPrice = stopLoss;
              exitReason = 'stop';
              exitTimestamp = miniTimestamp;
            } else if (target !== null && miniLow <= target) {
              exitTriggered = true;
              exitPrice = target;
              exitReason = 'target';
              exitTimestamp = miniTimestamp;
            }
          }

          if (exitTriggered) {
            break;
          }
        }

        if (exitTriggered) {
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
        const pointsGained = direction === 'long'
          ? exitPrice - activePosition.entryPrice
          : activePosition.entryPrice - exitPrice;
        const ticksGained = pointsGained / tickSize;
        const grossProfit = ticksGained * tickValue * activePosition.units;
        const netProfit = grossProfit - (commission * 2 * activePosition.units);

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
          exitReason,
          durationMinutes,
          units: activePosition.units,
          isScaledTrade: false,
        });

        activePosition = null;
      }
    }

    // Entry monitoring (if no position) - EXACT same logic as stock backtest
    if (!activePosition) {
      const entrySignal = generateMeanReversionSignalFromTechnicals(
        contractId,
        barClose,
        priceHistory,
        0,  // GEX not used for futures
        {
          rsiPeriod: 14,
          rsiOversold: 30,
          rsiOverbought: 70,
          bbPeriod: 20,
          bbStdDev: 2,
          bbThreshold: 0.005,  // 0.5%
          stopLossPercent,
          targetPercent: 0.02,  // Not used, target is BB middle
        }
      );

      if (entrySignal.direction !== 'none') {
        activePosition = {
          signal: entrySignal,
          entryTimestamp: barTimestamp,
          entryPrice: barClose,
          entryRSI: currentRSI,
          scaled: false,
          units: 2,  // Start with 2 units (EXACT same as stock backtest)
        };
      }
    }
  }

  // Calculate summary statistics
  const wins = trades.filter(t => t.netProfit > 0);
  const losses = trades.filter(t => t.netProfit <= 0);
  const totalNetProfit = trades.reduce((sum, t) => sum + t.netProfit, 0);
  const totalGrossProfit = trades.reduce((sum, t) => sum + t.grossProfit, 0);
  const totalCommission = trades.reduce((sum, t) => sum + (commission * 2 * t.units), 0);
  const avgWin = wins.length > 0 ? wins.reduce((sum, t) => sum + t.netProfit, 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((sum, t) => sum + t.netProfit, 0) / losses.length : 0;
  const winRate = trades.length > 0 ? (wins.length / trades.length) * 100 : 0;
  const profitFactor = Math.abs(avgLoss) > 0 ? avgWin / Math.abs(avgLoss) : 0;
  const avgDurationMinutes = trades.length > 0
    ? trades.reduce((sum, t) => sum + t.durationMinutes, 0) / trades.length
    : 0;

  // Calculate max drawdown
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

  const scaledTrades = trades.filter(t => t.isScaledTrade).length;

  console.log(`   ✓ Completed: ${trades.length} trades, ${scaledTrades} scaled`);

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
      scaledTrades,
    },
  };
}
