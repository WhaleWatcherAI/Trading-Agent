import { SMA } from 'technicalindicators';
import { getOptionsChain } from './lib/tradier';
import { getHistoricalData } from './lib/technicals';

interface Trade {
  symbol: string;
  entry: { timestamp: string; price: number; strike: number; type: 'call' | 'put'; direction: 'bullish' | 'bearish'; sma: number };
  exit?: { timestamp: string; price: number; reason: string };
  pnl?: number;
  pnlPercent?: number;
}

const SMA_PERIOD = 9;
const COMMISSION = 1.30;
const SLIPPAGE_PERCENT = 0.5;
const STOP_LOSS_PERCENT = -50;
const TAKE_PROFIT_PERCENT = 100;

async function backtestTSLA() {
  const symbol = 'TSLA';
  const testDate = '2025-11-04';

  console.log('Testing TSLA with FIXED option pricing...\n');

  const history = await getHistoricalData(symbol, '1min', 7, testDate);
  const optionsChain = await getOptionsChain(symbol, undefined, testDate);

  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });

  const trades: Trade[] = [];
  let currentTrade: Trade | null = null;

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

        const relevantOptions = optionsChain.filter(o => {
          if (o.type !== optionType) return false;
          if (optionType === 'call') return o.strike < currentPrice;
          else return o.strike > currentPrice;
        });

        relevantOptions.sort((a, b) => {
          if (optionType === 'call') return b.strike - a.strike;
          else return a.strike - b.strike;
        });

        if (relevantOptions.length > 0) {
          const bestOption = relevantOptions[0];

          let underlyingEntryPrice: number;
          if (crossesUp) {
            underlyingEntryPrice = currentBar.open > currentSma ? currentBar.open : currentSma;
          } else {
            underlyingEntryPrice = currentBar.open < currentSma ? currentBar.open : currentSma;
          }

          const priceDiffFromClose = underlyingEntryPrice - currentPrice;
          const premiumAdjustment = bestOption.greeks?.delta ? priceDiffFromClose * bestOption.greeks.delta : 0;
          const adjustedPremium = bestOption.premium + premiumAdjustment; // FIXED!
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
            },
          };
        }
      }
    } else {
      const entryType = currentTrade.entry.type;
      const entryPrice = currentTrade.entry.price;

      const oppositeCross = entryType === 'call'
        ? (previousPrice >= previousSma && currentBar.low < currentSma)
        : (previousPrice <= previousSma && currentBar.high > currentSma);

      const currentOption = optionsChain.find(o => o.strike === currentTrade.entry.strike && o.type === entryType);

      let estimatedCurrentPrice = entryPrice;
      if (currentOption && currentOption.greeks?.delta) {
        const underlyingMove = currentPrice - closes[i - 1];
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
        const grossPnl = exitPrice - entryPrice;
        const netPnl = grossPnl - COMMISSION;

        currentTrade.exit = { timestamp: history[i].date, price: exitPrice, reason: exitReason };
        currentTrade.pnl = netPnl;
        currentTrade.pnlPercent = (netPnl / entryPrice) * 100;
        trades.push(currentTrade);
        currentTrade = null;
      }
    }
  }

  const winners = trades.filter(t => t.pnl! > 0);
  const losers = trades.filter(t => t.pnl! <= 0);
  const totalPnL = trades.reduce((sum, t) => sum + t.pnl!, 0);
  const winRate = trades.length > 0 ? (winners.length / trades.length) * 100 : 0;

  console.log(`Total trades: ${trades.length}`);
  console.log(`Winners: ${winners.length} | Losers: ${losers.length}`);
  console.log(`Win rate: ${winRate.toFixed(1)}%`);
  console.log(`Total P&L: $${totalPnL.toFixed(2)}`);
  console.log(`Avg P&L: $${(totalPnL / trades.length).toFixed(2)}\n`);

  if (winners.length > 0) {
    console.log('Sample winning trades:');
    winners.slice(0, 5).forEach((t, i) => {
      console.log(`  ${i + 1}. ${t.entry.type.toUpperCase()} $${t.entry.strike} → ${t.pnlPercent!.toFixed(1)}% ($${t.pnl!.toFixed(2)})`);
    });
  }
}

backtestTSLA()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Test failed:', error);
    process.exit(1);
  });
