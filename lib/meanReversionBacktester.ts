import { getHistoricalTimesales, TradierTimesaleBar } from './tradier';
import { fetchTwelveDataBars } from './twelveDataRest';
import {
  generateMeanReversionSignal,
  generateMeanReversionSignalFromTechnicals,
  checkMeanReversionExit,
  calculateTechnicals,
  MeanReversionSignal,
} from './meanReversionAgent';
import { GexMode } from './gexCalculator';
import { RSI, SMA } from 'technicalindicators';
import { selectMeanReversionOption, OptionContract } from './contractSelector';
import { fetchTopstepXFuturesBars, fetchTopstepXContract, TopstepXFuturesBar } from './topstepx';

const TWELVE_DATA_API_KEY = process.env.TWELVE_DATA_API_KEY || '';
const USE_TWELVE_DATA = (process.env.USE_TWELVE_DATA ?? 'true').toLowerCase() !== 'false';

export interface MeanReversionTrade {
  id: string;
  symbol: string;
  direction: 'long' | 'short';
  entryTimestamp: string;
  exitTimestamp: string;

  // Stock trade data
  stock: {
    entryPrice: number;
    exitPrice: number;
    profit: number;
    profitPct: number;
  };

  // Futures trade data (ES/MES) - calculated if symbol is SPY
  futures?: {
    esProfit: number; // ES P&L ($50/point)
    mesProfit: number; // MES P&L ($5/point)
    indexPoints: number; // SPX index points moved
    commission: number; // $2 round-trip
  };

  // Options trade data (if available)
  option: {
    contract: string | null; // Option symbol
    strike: number | null;
    expiration: string | null;
    entryPremium: number | null; // Premium paid/received
    exitPremium: number | null;
    profit: number | null; // Per contract profit (net of commission)
    profitPct: number | null;
    commission: number | null; // Total commission (entry + exit = $0.08)
    slippage: number | null; // Slippage cost based on spread
    grossProfit: number | null; // Per contract profit before costs
  };

  stopLoss: number | null;
  target: number | null;
  entryRSI: number | null;
  exitRSI: number | null;
  durationMinutes: number;
  exitReason: 'stop' | 'target' | 'rsi_neutral' | 'end_of_day';
  netGex: number;
}

export interface MeanReversionBacktestResult {
  symbol: string;
  date: string;
  mode: GexMode;
  netGex: number;
  regime: 'positive_gex' | 'negative_gex';
  technicals: {
    bbUpper: number | null;
    bbMiddle: number | null;
    bbLower: number | null;
    avgRSI: number | null;
  };
  trades: MeanReversionTrade[];
  summary: {
    tradeCount: number;
    winCount: number;
    lossCount: number;
    winRate: number;

    // Stock P&L
    stock: {
      totalProfit: number;
      grossProfit: number;
      grossLoss: number;
      averageWin: number;
      averageLoss: number;
      profitFactor: number;
    };

    // Options P&L
    options: {
      totalProfit: number;
      grossProfit: number;
      grossLoss: number;
      averageWin: number;
      averageLoss: number;
      profitFactor: number;
      contractsTraded: number;
      totalCommission: number;
      totalSlippage: number;
      leverage: OptionsLeverageProfiles;
    };

    maxDrawdown: number;
    averageDurationMinutes: number;
  };
  notes: string[];
}

interface OptionsLeverageProfile {
  netProfit: number;
  grossProfit: number;
  commission: number;
  slippage: number;
}

type OptionsLeverageProfiles = Record<'50x' | '100x', OptionsLeverageProfile>;

interface ActivePosition {
  signal: MeanReversionSignal;
  entryTimestamp: string;
  entryBar: TradierTimesaleBar;
  entryPrice: number;
  optionContract: any | null; // OptionContract from contract selector
  optionEntryPremium: number | null;
  scaled: boolean; // Has position been scaled (taken partial profit)?
  units: number; // Number of units still active (starts at 2, becomes 1 after scaling)
}

async function fetchBarsWithPreference(
  symbol: string,
  date: string,
  intervalMinutes: number,
  notes: string[],
): Promise<TradierTimesaleBar[]> {
  const canUseTwelve = TWELVE_DATA_API_KEY && USE_TWELVE_DATA;
  if (canUseTwelve) {
    try {
      const tdBars = await fetchTwelveDataBars(symbol, date, intervalMinutes, TWELVE_DATA_API_KEY);
      if (tdBars.length > 0) {
        return tdBars.map(bar => ({
          time: bar.time,
          timestamp: bar.timestamp,
          open: bar.open,
          high: bar.high,
          low: bar.low,
          close: bar.close,
          volume: bar.volume,
        }));
      }
      notes.push(`Twelve Data returned no ${intervalMinutes}m bars for ${symbol} on ${date}`);
    } catch (error: any) {
      notes.push(`Twelve Data fetch failed: ${error?.message ?? error}`);
    }
  }

  return getHistoricalTimesales(symbol, date, intervalMinutes);
}

function getStopLossPercent(_: string): number {
  return 0.001;
}

function createEmptyLeverageProfiles(): OptionsLeverageProfiles {
  return {
    '50x': { netProfit: 0, grossProfit: 0, commission: 0, slippage: 0 },
    '100x': { netProfit: 0, grossProfit: 0, commission: 0, slippage: 0 },
  };
}

function parseTimestamp(timeStr: string, date: string): Date {
  const trimmed = (timeStr || '').trim();
  if (!trimmed) return new Date(NaN);

  let withDate = trimmed;
  if (/^\d{2}:\d{2}:\d{2}$/.test(trimmed)) {
    withDate = `${date}T${trimmed}`;
  } else if (trimmed.includes(' ')) {
    withDate = trimmed.replace(' ', 'T');
  } else if (!trimmed.includes('T')) {
    withDate = `${date}T${trimmed}`;
  }

  // Add timezone if missing
  if (!/[+-]\d{2}:\d{2}$/.test(withDate) && !withDate.endsWith('Z')) {
    const year = parseInt(date.slice(0, 4), 10);
    const month = parseInt(date.slice(5, 7), 10);
    const day = parseInt(date.slice(8, 10), 10);

    // Calculate DST boundaries for US Eastern Time
    // DST starts: Second Sunday of March at 2:00 AM
    // DST ends: First Sunday of November at 2:00 AM
    const getDSTStart = (year: number): Date => {
      const march = new Date(year, 2, 1); // March 1st
      const firstSunday = 7 - march.getDay();
      const secondSunday = firstSunday + 7;
      return new Date(year, 2, secondSunday);
    };

    const getDSTEnd = (year: number): Date => {
      const november = new Date(year, 10, 1); // November 1st
      const firstSunday = 7 - november.getDay();
      return new Date(year, 10, firstSunday);
    };

    const dstStart = getDSTStart(year);
    const dstEnd = getDSTEnd(year);
    const currentDate = new Date(year, month - 1, day);

    const isDST = currentDate >= dstStart && currentDate < dstEnd;
    const offset = isDST ? '-04:00' : '-05:00';
    withDate = `${withDate}${offset}`;
  }

  return new Date(withDate);
}

function minutesBetween(start: string, end: string): number {
  const delta = new Date(end).getTime() - new Date(start).getTime();
  if (!Number.isFinite(delta)) return 0;
  return Math.max(1, Math.round(delta / 60000));
}

/**
 * Run mean reversion backtest for a single symbol on a specific date
 */
export async function backtestMeanReversion(
  symbol: string,
  date: string,
  mode: GexMode = 'intraday',
  intervalMinutes: number = 1,
  intrabarIntervalMinutes: number = 1,
): Promise<MeanReversionBacktestResult> {
  console.log(`\n🔄 Running mean reversion backtest for ${symbol} on ${date} (${mode} mode)`);

const notes: string[] = [];
  const trades: MeanReversionTrade[] = [];

  // Fetch intraday price bars first (need for price history)
  let bars: TradierTimesaleBar[] = [];
  try {
    bars = await fetchBarsWithPreference(symbol, date, intervalMinutes, notes);
    if (bars.length === 0) {
      notes.push('No intraday price data available');
    }
  } catch (error: any) {
    notes.push(`Failed to fetch price data: ${error.message}`);
  }

  if (bars.length < 20) {
    notes.push(`Insufficient bars (${bars.length}, need at least 20)`);
    return {
      symbol,
      date,
      mode,
      netGex: 0,
      regime: 'negative_gex',
      technicals: { bbUpper: null, bbMiddle: null, bbLower: null, avgRSI: null },
      trades: [],
      summary: {
        tradeCount: 0,
        winCount: 0,
        lossCount: 0,
        winRate: 0,
        stock: {
          totalProfit: 0,
          grossProfit: 0,
          grossLoss: 0,
          averageWin: 0,
          averageLoss: 0,
          profitFactor: 0,
        },
        options: {
          totalProfit: 0,
          grossProfit: 0,
          grossLoss: 0,
          averageWin: 0,
          averageLoss: 0,
          profitFactor: 0,
          contractsTraded: 0,
          totalCommission: 0,
          totalSlippage: 0,
          leverage: createEmptyLeverageProfiles(),
        },
        maxDrawdown: 0,
        averageDurationMinutes: 0,
      },
      notes,
    };
  }

  // Fetch higher resolution bars for intrabar simulation when needed
  let intrabarBars: TradierTimesaleBar[] = bars;
  if (intervalMinutes > intrabarIntervalMinutes) {
    try {
      intrabarBars = await fetchBarsWithPreference(
        symbol,
        date,
        intrabarIntervalMinutes,
        notes,
      );
    } catch (error: any) {
      notes.push(`Failed to fetch intrabar data: ${error.message}`);
      intrabarBars = bars;
    }
  }

  const intervalMs = intervalMinutes * 60_000;
  const parseBarTime = (bar: TradierTimesaleBar): number => {
    const parsed = parseTimestamp(bar.time || String(bar.timestamp), date);
    return parsed.getTime();
  };

  const intrabarSegments: TradierTimesaleBar[][] = bars.map(() => []);

  if (intervalMinutes > intrabarIntervalMinutes && intrabarBars !== bars) {
    const sortedIntrabars = intrabarBars
      .slice()
      .sort((a, b) => parseBarTime(a) - parseBarTime(b));

    const barStartTimes = bars.map(parseBarTime);
    let barIndex = 0;

    for (const miniBar of sortedIntrabars) {
      const miniTime = parseBarTime(miniBar);
      if (!Number.isFinite(miniTime)) {
        continue;
      }

      while (
        barIndex < barStartTimes.length &&
        miniTime >= barStartTimes[barIndex] + intervalMs
      ) {
        barIndex += 1;
      }

      if (barIndex >= barStartTimes.length) {
        break;
      }

      const nextStart = barIndex < barStartTimes.length - 1
        ? barStartTimes[barIndex + 1]
        : barStartTimes[barIndex] + intervalMs;

      if (miniTime >= barStartTimes[barIndex] && miniTime < nextStart) {
        intrabarSegments[barIndex].push(miniBar);
      }
    }
  }

  for (let i = 0; i < intrabarSegments.length; i += 1) {
    if (intrabarSegments[i].length === 0) {
      intrabarSegments[i] = [bars[i]];
    }
  }

  // Generate initial signal to check GEX regime
  const initialPriceHistory = bars.slice(0, 20).map(b => Number(b.close));
  let signal: MeanReversionSignal;
  const stopLossPercent = getStopLossPercent(symbol);
  try {
    signal = await generateMeanReversionSignal({
      symbol,
      date,
      mode,
      rsiPeriod: 14,
      rsiOversold: 30,
      rsiOverbought: 70,
      bbPeriod: 20,
      bbStdDev: 2,
      bbThreshold: 0.005, // 0.5%
      stopLossPercent,
      targetPercent: 0.02, // 2%
    }, initialPriceHistory);
  } catch (error: any) {
    notes.push(`Failed to generate signal: ${error.message}`);
    return {
      symbol,
      date,
      mode,
      netGex: 0,
      regime: 'negative_gex',
      technicals: { bbUpper: null, bbMiddle: null, bbLower: null, avgRSI: null },
      trades: [],
      summary: {
        tradeCount: 0,
        winCount: 0,
        lossCount: 0,
        winRate: 0,
        stock: {
          totalProfit: 0,
          grossProfit: 0,
          grossLoss: 0,
          averageWin: 0,
          averageLoss: 0,
          profitFactor: 0,
        },
        options: {
          totalProfit: 0,
          grossProfit: 0,
          grossLoss: 0,
          averageWin: 0,
          averageLoss: 0,
          profitFactor: 0,
          contractsTraded: 0,
          totalCommission: 0,
          totalSlippage: 0,
          leverage: createEmptyLeverageProfiles(),
        },
        maxDrawdown: 0,
        averageDurationMinutes: 0,
      },
      notes,
    };
  }

  // Check if we have a positive GEX day (DISABLED - trading all days)
  // if (signal.regime !== 'positive_gex') {
  //   notes.push(`Skipped: Net GEX is ${(signal.netGex / 1_000_000).toFixed(1)}M (not positive)`);
  //   return {
  //     symbol,
  //     date,
  //     mode,
  //     netGex: signal.netGex,
  //     regime: signal.regime,
  //     technicals: {
  //       bbUpper: signal.bbUpper,
  //       bbMiddle: signal.bbMiddle,
  //       bbLower: signal.bbLower,
  //       avgRSI: signal.rsi,
  //     },
  //     trades: [],
  //     summary: {
  //       tradeCount: 0,
  //       winCount: 0,
  //       lossCount: 0,
  //       winRate: 0,
  //       stock: {
  //         totalProfit: 0,
  //         grossProfit: 0,
  //         grossLoss: 0,
  //         averageWin: 0,
  //         averageLoss: 0,
  //         profitFactor: 0,
  //       },
  //       options: {
  //         totalProfit: 0,
  //         grossProfit: 0,
  //         grossLoss: 0,
  //         averageWin: 0,
  //         averageLoss: 0,
  //         profitFactor: 0,
  //         contractsTraded: 0,
  //         totalCommission: 0,
  //         totalSlippage: 0,
  //         leverage: createEmptyLeverageProfiles(),
  //       },
  //       maxDrawdown: 0,
  //       averageDurationMinutes: 0,
  //     },
  //     notes,
  //   };
  // }

  notes.push(`Trading on ${signal.regime} day: Net GEX ${(signal.netGex / 1_000_000).toFixed(1)}M`);

  // Simulate trading throughout the day
  let activePosition: ActivePosition | null = null;
  let equity = 0;
  let peakEquity = 0;
  let maxDrawdown = 0;

  const allRSIValues: number[] = [];

  const closeActivePosition = (
    exitPrice: number,
    exitTimestamp: string,
    exitReason: 'stop' | 'target' | 'rsi_neutral' | 'end_of_day',
    exitRSI: number | null,
  ) => {
    if (!activePosition) {
      return;
    }

    const entryPrice = Number.isFinite(activePosition.entryPrice)
      ? activePosition.entryPrice
      : Number(activePosition.entryBar.close);
    if (!Number.isFinite(entryPrice) || entryPrice <= 0) {
      activePosition = null;
      return;
    }

    // Calculate profit including scaling
    let stockProfit = activePosition.signal.direction === 'long'
      ? exitPrice - entryPrice
      : entryPrice - exitPrice;

    // If position was scaled, add the profit from the first unit at middle band
    if (activePosition.scaled && activePosition.signal.target) {
      const firstUnitExit = activePosition.signal.direction === 'long'
        ? activePosition.signal.target / 0.99 // Original middle band before adjustment
        : activePosition.signal.target / 1.01;

      const firstUnitProfit = activePosition.signal.direction === 'long'
        ? firstUnitExit - entryPrice
        : entryPrice - firstUnitExit;

      // Total profit = 1 unit at middle + 1 unit at current exit
      stockProfit = firstUnitProfit + stockProfit;
    } else {
      // No scaling, multiply by 2 units
      stockProfit = stockProfit * 2;
    }

    const stockProfitPct = (stockProfit / (entryPrice * 2)) * 100;

    let optionProfit: number | null = null;
    let optionProfitPct: number | null = null;
    let optionExitPremium: number | null = null;
    let optionCommission: number | null = null;
    let optionSlippage: number | null = null;
    let optionGrossProfit: number | null = null;

    if (activePosition.optionContract && activePosition.optionEntryPremium) {
      const priceMove = exitPrice - entryPrice;
      const delta = activePosition.optionContract.delta;
      const premiumChange = priceMove * Math.abs(delta);

      optionExitPremium = activePosition.optionEntryPremium + (
        activePosition.signal.direction === 'long' ? premiumChange : -premiumChange
      );

      optionExitPremium = Math.max(0.01, optionExitPremium);

      const entrySpread = activePosition.optionContract.ask - activePosition.optionContract.bid;
      const exitSpread = entrySpread;
      const entrySlippage = entrySpread * 0.5;
      const exitSlippage = exitSpread * 0.5;
      optionSlippage = (entrySlippage + exitSlippage) * 100;

      optionCommission = 0.08;

      const grossProfit = (optionExitPremium - activePosition.optionEntryPremium) * 100;
      optionGrossProfit = grossProfit;
      optionProfit = grossProfit - optionCommission - optionSlippage;
      optionProfitPct = ((optionExitPremium - activePosition.optionEntryPremium) / activePosition.optionEntryPremium) * 100;
    }

    const durationMinutes = minutesBetween(activePosition.entryTimestamp, exitTimestamp);

    // Calculate futures P&L if symbol is SPY
    let futuresData: {
      esProfit: number;
      mesProfit: number;
      indexPoints: number;
      commission: number;
    } | undefined;

    if (symbol === 'SPY') {
      // SPY price movement in dollars
      const spyMove = Math.abs(exitPrice - entryPrice);
      // Convert SPY dollar move to SPX index points (SPY ≈ SPX/10)
      const indexPoints = spyMove * 10;
      // Calculate ES P&L: $50 per index point, 2 units (for scaling strategy)
      const esGrossProfit = indexPoints * 50 * 2;
      // Calculate MES P&L: $5 per index point, 2 units
      const mesGrossProfit = indexPoints * 5 * 2;
      // Commission: $1 per side = $2 round-trip per contract, 2 contracts
      const commission = 2 * 2; // $4 total for 2 contracts

      // Net profit (apply direction)
      const profitMultiplier = activePosition.signal.direction === 'long'
        ? (exitPrice > entryPrice ? 1 : -1)
        : (exitPrice < entryPrice ? 1 : -1);

      futuresData = {
        esProfit: (esGrossProfit * profitMultiplier) - commission,
        mesProfit: (mesGrossProfit * profitMultiplier) - commission,
        indexPoints: indexPoints,
        commission: commission,
      };
    }

    const trade: MeanReversionTrade = {
      id: `${symbol}-${activePosition.entryTimestamp}`,
      symbol,
      direction: activePosition.signal.direction,
      entryTimestamp: activePosition.entryTimestamp,
      exitTimestamp,
      stock: {
        entryPrice,
        exitPrice,
        profit: stockProfit,
        profitPct: stockProfitPct,
      },
      futures: futuresData,
      option: {
        contract: activePosition.optionContract?.symbol || null,
        strike: activePosition.optionContract?.strike || null,
        expiration: activePosition.optionContract?.expiration || null,
        entryPremium: activePosition.optionEntryPremium,
        exitPremium: optionExitPremium,
        profit: optionProfit,
        profitPct: optionProfitPct,
        commission: optionCommission,
        slippage: optionSlippage,
        grossProfit: optionGrossProfit,
      },
      stopLoss: activePosition.signal.stopLoss,
      target: activePosition.signal.target,
      entryRSI: activePosition.signal.rsi,
      exitRSI,
      durationMinutes,
      exitReason,
      netGex: signal.netGex,
    };

    trades.push(trade);

    equity += stockProfit;
    peakEquity = Math.max(peakEquity, equity);
    maxDrawdown = Math.max(maxDrawdown, peakEquity - equity);

    activePosition = null;
  };

  for (let i = 20; i < bars.length; i++) {
    const bar = bars[i];
    const barDate = parseTimestamp(bar.time || String(bar.timestamp), date);
    const barTime = barDate.getTime();
    if (!Number.isFinite(barTime)) {
      continue;
    }

    const barTimestamp = barDate.toISOString();
    const currentPrice = Number(bar.close);
    if (!Number.isFinite(currentPrice) || currentPrice <= 0) {
      continue;
    }

    const priceHistory = bars
      .slice(0, i + 1)
      .map(b => Number(b.close))
      .filter(price => Number.isFinite(price));

    if (priceHistory.length < 20) {
      continue;
    }

    const rsiValues = RSI.calculate({ values: priceHistory, period: 14 });
    const currentRSI = rsiValues[rsiValues.length - 1];
    if (Number.isFinite(currentRSI)) {
      allRSIValues.push(currentRSI);
    }

    const minuteBars = intrabarSegments[i] && intrabarSegments[i].length > 0
      ? intrabarSegments[i]
      : [bar];

    // Exit monitoring - check every 1-min bar for exits
    for (const minuteBar of minuteBars) {
      if (!activePosition) {
        continue;
      }

      const minuteDate = parseTimestamp(minuteBar.time || String(minuteBar.timestamp), date);
      const minuteTime = minuteDate.getTime();
      if (!Number.isFinite(minuteTime)) {
        continue;
      }

      const minuteTimestamp = minuteDate.toISOString();
      const minuteClose = Number(minuteBar.close);
      if (!Number.isFinite(minuteClose) || minuteClose <= 0) {
        continue;
      }

      // Exit monitoring for active positions
      if (activePosition) {
        const barHigh = Number(minuteBar.high);
        const barLow = Number(minuteBar.low);
        const barOpen = Number(minuteBar.open);
        const referencePrice = Number.isFinite(barOpen) && barOpen > 0 ? barOpen : minuteClose;

        let exitTriggered = false;
        let exitReason: 'stop' | 'target' | 'rsi_neutral' = 'rsi_neutral';
        let exitPrice = minuteClose;

        const considerExit = (reason: 'stop' | 'target', price: number) => {
          if (!Number.isFinite(price)) {
            return;
          }
          if (!exitTriggered) {
            exitTriggered = true;
            exitReason = reason;
            exitPrice = price;
            return;
          }

          const existingDistance = Math.abs(referencePrice - exitPrice);
          const candidateDistance = Math.abs(referencePrice - price);
          if (candidateDistance < existingDistance) {
            exitReason = reason;
            exitPrice = price;
          }
        };

        const direction = activePosition.signal.direction;
        let stopLoss = activePosition.signal.stopLoss;
        let target = activePosition.signal.target;

        // Check for scaling opportunity at middle band
        if (!activePosition.scaled && target !== null) {
          const hitTarget = (direction === 'long' && Number.isFinite(barHigh) && barHigh >= target) ||
                           (direction === 'short' && Number.isFinite(barLow) && barLow <= target);

          if (hitTarget) {
            // Scale out: take profit on half, let other half run
            activePosition.scaled = true;
            activePosition.units = 1;

            // Adjust stop loss near middle band and set target to opposite outer band
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

            // Update local variables
            stopLoss = activePosition.signal.stopLoss;
            target = activePosition.signal.target;

            // Don't exit yet, continue with new stop/target
            continue;
          }
        }

        if (direction === 'long') {
          if (typeof stopLoss === 'number' && Number.isFinite(barLow) && barLow <= stopLoss) {
            considerExit('stop', stopLoss);
          }
          if (typeof target === 'number' && Number.isFinite(barHigh) && barHigh >= target) {
            considerExit('target', target);
          }
        } else if (direction === 'short') {
          if (typeof stopLoss === 'number' && Number.isFinite(barHigh) && barHigh >= stopLoss) {
            considerExit('stop', stopLoss);
          }
          if (typeof target === 'number' && Number.isFinite(barLow) && barLow <= target) {
            considerExit('target', target);
          }
        }

        if (!exitTriggered) {
          const exitCheck = checkMeanReversionExit(minuteClose, activePosition.signal, currentRSI);
          if (exitCheck.shouldExit && exitCheck.exitReason !== 'none') {
            exitTriggered = true;
            exitReason = exitCheck.exitReason as 'stop' | 'target' | 'rsi_neutral';
            exitPrice = exitCheck.exitPrice;
          }
        }

        if (exitTriggered) {
          closeActivePosition(
            exitPrice,
            minuteTimestamp,
            exitReason,
            Number.isFinite(currentRSI) ? currentRSI : null,
          );
        }
      }
    }

    // Entry monitoring - check AFTER 15-min bar closes
    if (!activePosition) {
      const entrySignal = generateMeanReversionSignalFromTechnicals(
        symbol,
        currentPrice,
        priceHistory,
        signal.netGex,
        {
          rsiPeriod: 14,
          rsiOversold: 30,
          rsiOverbought: 70,
          bbPeriod: 20,
          bbStdDev: 2,
          bbThreshold: 0.005,
          stopLossPercent,
          targetPercent: 0.02,
        }
      );

      if (entrySignal.direction !== 'none') {
        let optionContract: OptionContract | null = null;
        let optionEntryPremium: number | null = null;
        const derivedEntryPrice = Number.isFinite(entrySignal.entryPrice)
          ? Number(entrySignal.entryPrice)
          : currentPrice;

        try {
          optionContract = await selectMeanReversionOption(
            symbol,
            currentPrice,
            entrySignal.direction,
            date
          );

          if (optionContract) {
            optionEntryPremium = optionContract.mid;
          }
        } catch (error) {
          console.log(`Could not fetch option for ${symbol} - continuing with stock only`);
        }

        activePosition = {
          signal: entrySignal,
          entryTimestamp: barTimestamp,
          entryBar: { ...bar },
          entryPrice: derivedEntryPrice,
          optionContract,
          optionEntryPremium,
          scaled: false,
          units: 2, // Start with 2 units
        };
      }
    }
  }

  if (activePosition && bars.length > 0) {
    const lastBar = bars[bars.length - 1];
    const exitPrice = Number(lastBar.close);
    if (Number.isFinite(exitPrice) && exitPrice > 0) {
      const lastDate = parseTimestamp(lastBar.time || String(lastBar.timestamp), date);
      const timestamp = Number.isFinite(lastDate.getTime())
        ? lastDate.toISOString()
        : activePosition.entryTimestamp;

      const finalPriceHistory = bars
        .map(b => Number(b.close))
        .filter(price => Number.isFinite(price));
      const finalRSIValues = RSI.calculate({ values: finalPriceHistory, period: 14 });
      const finalRSI = finalRSIValues[finalRSIValues.length - 1];

      closeActivePosition(
        exitPrice,
        timestamp,
        'end_of_day',
        Number.isFinite(finalRSI) ? finalRSI : null,
      );
    } else {
      activePosition = null;
    }
  }

  // Calculate summary statistics
  const tradeCount = trades.length;
  const winCount = trades.filter(t => t.stock.profit > 0).length;
  const lossCount = trades.filter(t => t.stock.profit <= 0).length;
  const winRate = tradeCount > 0 ? winCount / tradeCount : 0;

  // Stock P&L statistics
  const stockTotalProfit = trades.reduce((sum, t) => sum + t.stock.profit, 0);
  const stockGrossProfit = trades.filter(t => t.stock.profit > 0).reduce((sum, t) => sum + t.stock.profit, 0);
  const stockGrossLoss = Math.abs(trades.filter(t => t.stock.profit <= 0).reduce((sum, t) => sum + t.stock.profit, 0));
  const stockAverageWin = winCount > 0 ? stockGrossProfit / winCount : 0;
  const stockAverageLoss = lossCount > 0 ? stockGrossLoss / lossCount : 0;
  const stockProfitFactor = stockGrossLoss > 0 ? stockGrossProfit / stockGrossLoss : stockGrossProfit > 0 ? Infinity : 0;

  // Option P&L statistics
  const optionTrades = trades.filter(t => t.option.profit !== null);
  const optionWinCount = optionTrades.filter(t => t.option.profit! > 0).length;
  const optionLossCount = optionTrades.filter(t => t.option.profit! <= 0).length;
  const optionTotalProfit = optionTrades.reduce((sum, t) => sum + (t.option.profit || 0), 0);
  const optionGrossProfit = optionTrades.filter(t => t.option.profit! > 0).reduce((sum, t) => sum + t.option.profit!, 0);
  const optionGrossLoss = Math.abs(optionTrades.filter(t => t.option.profit! <= 0).reduce((sum, t) => sum + t.option.profit!, 0));
  const optionAverageWin = optionWinCount > 0 ? optionGrossProfit / optionWinCount : 0;
  const optionAverageLoss = optionLossCount > 0 ? optionGrossLoss / optionLossCount : 0;
  const optionProfitFactor = optionGrossLoss > 0 ? optionGrossProfit / optionGrossLoss : optionGrossProfit > 0 ? Infinity : 0;
  const optionTotalCommission = optionTrades.reduce((sum, t) => sum + (t.option.commission || 0), 0);
  const optionTotalSlippage = optionTrades.reduce((sum, t) => sum + (t.option.slippage || 0), 0);
  const optionTotalGrossPnL = optionTotalProfit + optionTotalCommission + optionTotalSlippage;

  const optionLeverageProfiles: OptionsLeverageProfiles = {
    '50x': {
      netProfit: optionTotalProfit * 50,
      grossProfit: optionTotalGrossPnL * 50,
      commission: optionTotalCommission * 50,
      slippage: optionTotalSlippage * 50,
    },
    '100x': {
      netProfit: optionTotalProfit * 100,
      grossProfit: optionTotalGrossPnL * 100,
      commission: optionTotalCommission * 100,
      slippage: optionTotalSlippage * 100,
    },
  };

  const averageDurationMinutes = tradeCount > 0
    ? trades.reduce((sum, t) => sum + t.durationMinutes, 0) / tradeCount
    : 0;

  // Calculate average RSI across the day
  const avgRSI = allRSIValues.length > 0
    ? allRSIValues.reduce((sum, val) => sum + val, 0) / allRSIValues.length
    : null;

  return {
    symbol,
    date,
    mode,
    netGex: signal.netGex,
    regime: signal.regime,
    technicals: {
      bbUpper: signal.bbUpper,
      bbMiddle: signal.bbMiddle,
      bbLower: signal.bbLower,
      avgRSI,
    },
    trades,
    summary: {
      tradeCount,
      winCount,
      lossCount,
      winRate,
      stock: {
        totalProfit: stockTotalProfit,
        grossProfit: stockGrossProfit,
        grossLoss: stockGrossLoss,
        averageWin: stockAverageWin,
        averageLoss: stockAverageLoss,
        profitFactor: stockProfitFactor,
      },
      options: {
        totalProfit: optionTotalProfit,
        grossProfit: optionGrossProfit,
        grossLoss: optionGrossLoss,
        averageWin: optionAverageWin,
        averageLoss: optionAverageLoss,
        profitFactor: optionProfitFactor,
        contractsTraded: optionTrades.length,
        totalCommission: optionTotalCommission,
        totalSlippage: optionTotalSlippage,
        leverage: optionLeverageProfiles,
      },
      maxDrawdown,
      averageDurationMinutes,
    },
    notes,
  };
}

/**
 * Run backtests across multiple symbols and dates
 */
export async function backtestMeanReversionMultiple(
  symbols: string[],
  dates: string[],
  mode: GexMode = 'intraday',
  intervalMinutes: number = 1,
  intrabarIntervalMinutes: number = 1,
): Promise<MeanReversionBacktestResult[]> {
  const results: MeanReversionBacktestResult[] = [];

  for (const symbol of symbols) {
    for (const date of dates) {
      try {
        const result = await backtestMeanReversion(
          symbol,
          date,
          mode,
          intervalMinutes,
          intrabarIntervalMinutes,
        );
        results.push(result);
      } catch (error: any) {
        console.error(`Failed to backtest ${symbol} on ${date}:`, error.message);
        // Continue with other symbols/dates
      }
    }
  }

  return results;
}
