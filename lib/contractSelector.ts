import axios from 'axios';
import { Strategy, STRATEGY_PARAMS, StrategyParams } from './strategyEngine';

const TRADIER_API_KEY = process.env.TRADIER_API_KEY || '';
const TRADIER_BASE_URL = process.env.TRADIER_BASE_URL || 'https://api.tradier.com/v1';

const tradierClient = axios.create({
  baseURL: TRADIER_BASE_URL,
  headers: {
    'Authorization': `Bearer ${TRADIER_API_KEY}`,
    'Accept': 'application/json',
  },
});

export interface OptionContract {
  symbol: string; // Option symbol (e.g., "AAPL251121C00150000")
  underlying: string; // Stock symbol
  strike: number;
  expiration: string; // YYYY-MM-DD
  type: 'call' | 'put';
  bid: number;
  ask: number;
  last: number;
  mid: number;
  volume: number;
  openInterest: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  impliedVolatility: number;
  daysToExpiration: number;
}

export interface ContractRecommendation {
  contract: OptionContract;
  reasoning: string;
  alternativeContracts: OptionContract[];
}

/**
 * Fetch available expiration dates for a symbol
 */
async function getExpirationDates(symbol: string): Promise<string[]> {
  try {
    const response = await tradierClient.get('/markets/options/expirations', {
      params: { symbol, includeAllRoots: false },
    });

    const expirations = response.data.expirations.date;
    return Array.isArray(expirations) ? expirations : [expirations];
  } catch (error) {
    console.error(`Error fetching expirations for ${symbol}:`, error);
    return [];
  }
}

/**
 * Fetch options chain for a specific expiration
 */
async function getOptionsChain(symbol: string, expiration: string): Promise<any[]> {
  try {
    const response = await tradierClient.get('/markets/options/chains', {
      params: {
        symbol,
        expiration,
        greeks: true,
      },
    });

    const options = response.data.options?.option;
    if (!options) return [];

    return Array.isArray(options) ? options : [options];
  } catch (error) {
    console.error(`Error fetching options chain for ${symbol} ${expiration}:`, error);
    return [];
  }
}

/**
 * Calculate days to expiration
 */
function calculateDTE(expirationDate: string): number {
  const exp = new Date(expirationDate);
  const now = new Date();
  const diffTime = exp.getTime() - now.getTime();
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return Math.max(0, diffDays);
}

/**
 * Find expirations that match the strategy's DTE range
 */
function filterExpirationsByDTE(
  expirations: string[],
  dteRange: { min: number; max: number }
): string[] {
  return expirations.filter(exp => {
    const dte = calculateDTE(exp);
    return dte >= dteRange.min && dte <= dteRange.max;
  });
}

/**
 * Parse option data into OptionContract format
 */
function parseOptionContract(opt: any, underlying: string): OptionContract {
  const dte = calculateDTE(opt.expiration_date);

  return {
    symbol: opt.symbol,
    underlying,
    strike: parseFloat(opt.strike),
    expiration: opt.expiration_date,
    type: opt.option_type.toLowerCase() as 'call' | 'put',
    bid: parseFloat(opt.bid) || 0,
    ask: parseFloat(opt.ask) || 0,
    last: parseFloat(opt.last) || 0,
    mid: ((parseFloat(opt.bid) || 0) + (parseFloat(opt.ask) || 0)) / 2,
    volume: parseInt(opt.volume) || 0,
    openInterest: parseInt(opt.open_interest) || 0,
    delta: parseFloat(opt.greeks?.delta) || 0,
    gamma: parseFloat(opt.greeks?.gamma) || 0,
    theta: parseFloat(opt.greeks?.theta) || 0,
    vega: parseFloat(opt.greeks?.vega) || 0,
    impliedVolatility: parseFloat(opt.greeks?.smv_vol) || 0,
    daysToExpiration: dte,
  };
}

/**
 * Filter contracts by liquidity requirements
 */
function filterByLiquidity(
  contracts: OptionContract[],
  params: StrategyParams
): OptionContract[] {
  return contracts.filter(c => {
    const hasLiquidity = c.openInterest >= params.minOpenInterest && c.volume >= params.minVolume;
    const hasValidPrice = c.bid > 0 && c.ask > 0 && c.ask < 100; // Reasonable price range
    const hasTightSpread = (c.ask - c.bid) / c.mid < 0.3; // Spread < 30% of mid

    return hasLiquidity && hasValidPrice && hasTightSpread;
  });
}

/**
 * Find the best strike based on delta target and strike bias
 */
function findBestStrike(
  contracts: OptionContract[],
  currentPrice: number,
  targetDelta: number,
  direction: 'bullish' | 'bearish',
  strikeBias: 'ATM' | 'ITM' | 'OTM'
): OptionContract | null {
  if (contracts.length === 0) return null;

  // Filter by option type (calls for bullish, puts for bearish)
  const filteredByType = contracts.filter(c =>
    direction === 'bullish' ? c.type === 'call' : c.type === 'put'
  );

  if (filteredByType.length === 0) return null;

  // Find ATM strike
  const atmStrike = filteredByType.reduce((prev, curr) =>
    Math.abs(curr.strike - currentPrice) < Math.abs(prev.strike - currentPrice) ? curr : prev
  );

  // Apply strike bias
  let candidates: OptionContract[] = [];

  if (strikeBias === 'ATM') {
    // Find strikes within 2% of ATM
    candidates = filteredByType.filter(c =>
      Math.abs(c.strike - atmStrike.strike) / atmStrike.strike < 0.02
    );
  } else if (strikeBias === 'ITM') {
    // For calls: strike < current, for puts: strike > current
    if (direction === 'bullish') {
      candidates = filteredByType.filter(c => c.strike < currentPrice && c.strike >= currentPrice * 0.90);
    } else {
      candidates = filteredByType.filter(c => c.strike > currentPrice && c.strike <= currentPrice * 1.10);
    }
  } else if (strikeBias === 'OTM') {
    // For calls: strike > current, for puts: strike < current
    if (direction === 'bullish') {
      candidates = filteredByType.filter(c => c.strike > currentPrice && c.strike <= currentPrice * 1.15);
    } else {
      candidates = filteredByType.filter(c => c.strike < currentPrice && c.strike >= currentPrice * 0.85);
    }
  }

  if (candidates.length === 0) {
    candidates = [atmStrike]; // Fallback to ATM
  }

  // Find contract with delta closest to target
  const best = candidates.reduce((prev, curr) => {
    const prevDeltaDiff = Math.abs(Math.abs(prev.delta) - targetDelta);
    const currDeltaDiff = Math.abs(Math.abs(curr.delta) - targetDelta);
    return currDeltaDiff < prevDeltaDiff ? curr : prev;
  });

  return best;
}

/**
 * Main function to select the best option contract
 */
export async function selectBestContract(
  symbol: string,
  currentPrice: number,
  direction: 'bullish' | 'bearish',
  strategy: Strategy
): Promise<ContractRecommendation | null> {
  try {
    const params = STRATEGY_PARAMS[strategy];

    // Get all available expirations
    const allExpirations = await getExpirationDates(symbol);

    // Filter by DTE range
    const validExpirations = filterExpirationsByDTE(allExpirations, params.dteRange);

    if (validExpirations.length === 0) {
      console.warn(`No valid expirations found for ${symbol} with DTE ${params.dteRange.min}-${params.dteRange.max}`);
      return null;
    }

    // Try each expiration until we find good contracts
    let bestContract: OptionContract | null = null;
    let alternativeContracts: OptionContract[] = [];
    let selectedExpiration = '';

    for (const expiration of validExpirations.slice(0, 5)) { // Try first 5 valid expirations
      const chainData = await getOptionsChain(symbol, expiration);
      const contracts = chainData.map(opt => parseOptionContract(opt, symbol));

      // Filter by liquidity
      const liquidContracts = filterByLiquidity(contracts, params);

      if (liquidContracts.length > 0) {
        // Find best strike
        const best = findBestStrike(
          liquidContracts,
          currentPrice,
          params.targetDelta,
          direction,
          params.strikeBias
        );

        if (best) {
          bestContract = best;
          selectedExpiration = expiration;

          // Find 2-3 alternative strikes
          const alternatives = liquidContracts
            .filter(c => c.symbol !== best.symbol && c.type === best.type)
            .sort((a, b) => {
              // Sort by delta similarity
              const aDiff = Math.abs(Math.abs(a.delta) - params.targetDelta);
              const bDiff = Math.abs(Math.abs(b.delta) - params.targetDelta);
              return aDiff - bDiff;
            })
            .slice(0, 3);

          alternativeContracts = alternatives;
          break;
        }
      }
    }

    if (!bestContract) {
      console.warn(`No suitable contracts found for ${symbol} ${strategy} ${direction}`);
      return null;
    }

    // Generate reasoning
    const reasoning = generateContractReasoning(bestContract, params, currentPrice, direction);

    return {
      contract: bestContract,
      reasoning,
      alternativeContracts,
    };
  } catch (error) {
    console.error(`Error selecting contract for ${symbol}:`, error);
    return null;
  }
}

/**
 * Generate human-readable reasoning for contract selection
 */
function generateContractReasoning(
  contract: OptionContract,
  params: StrategyParams,
  currentPrice: number,
  direction: 'bullish' | 'bearish'
): string {
  const moneyness =
    contract.type === 'call'
      ? contract.strike > currentPrice
        ? 'OTM'
        : contract.strike < currentPrice
        ? 'ITM'
        : 'ATM'
      : contract.strike < currentPrice
      ? 'OTM'
      : contract.strike > currentPrice
      ? 'ITM'
      : 'ATM';

  const distanceFromSpot = ((Math.abs(contract.strike - currentPrice) / currentPrice) * 100).toFixed(1);

  const reasoning = `
Selected ${contract.expiration} ${contract.strike} ${contract.type.toUpperCase()} (${moneyness}, ${distanceFromSpot}% from spot)
Delta: ${contract.delta.toFixed(2)} (target: ${params.targetDelta.toFixed(2)})
Premium: $${contract.mid.toFixed(2)} (bid: $${contract.bid.toFixed(2)}, ask: $${contract.ask.toFixed(2)})
Liquidity: ${contract.openInterest.toLocaleString()} OI, ${contract.volume.toLocaleString()} volume
Greeks: Gamma ${contract.gamma.toFixed(3)}, Theta ${contract.theta.toFixed(2)}, IV ${(contract.impliedVolatility * 100).toFixed(1)}%
DTE: ${contract.daysToExpiration} days (${params.strategy.toUpperCase()} optimal range)
  `.trim();

  return reasoning;
}

/**
 * Get current stock price from Tradier
 */
export async function getStockPrice(symbol: string): Promise<number> {
  try {
    const response = await tradierClient.get('/markets/quotes', {
      params: { symbols: symbol },
    });

    const quote = response.data.quotes.quote;
    const price = parseFloat(quote.last) || 0;
    console.log(`💰 ${symbol} price: $${price}`);
    return price;
  } catch (error: any) {
    console.error(`❌ Error fetching stock price for ${symbol}:`, error.response?.data || error.message);
    return 0;
  }
}

/**
 * Get multiple contract recommendations for a symbol (for multi-strategy output)
 */
export async function getMultiStrategyContracts(
  symbol: string,
  direction: 'bullish' | 'bearish',
  strategies: Strategy[] = ['scalp', 'intraday', 'swing', 'leap']
): Promise<Record<Strategy, ContractRecommendation | null>> {
  const currentPrice = await getStockPrice(symbol);

  if (currentPrice === 0) {
    console.error(`Could not fetch price for ${symbol}`);
    return {} as Record<Strategy, ContractRecommendation | null>;
  }

  const results: Record<string, ContractRecommendation | null> = {};

  // Fetch contracts for each strategy sequentially (to avoid rate limits)
  for (const strategy of strategies) {
    const recommendation = await selectBestContract(symbol, currentPrice, direction, strategy);
    results[strategy] = recommendation;

    // Small delay to respect rate limits
    await new Promise(resolve => setTimeout(resolve, 300));
  }

  return results as Record<Strategy, ContractRecommendation | null>;
}

/**
 * Select ATM/ITM option with nearest expiration for mean reversion trading
 * Simplified version for backtesting
 */
export async function selectMeanReversionOption(
  symbol: string,
  currentPrice: number,
  direction: 'long' | 'short',
  date?: string
): Promise<OptionContract | null> {
  try {
    // Get all available expirations
    const allExpirations = await getExpirationDates(symbol);

    if (allExpirations.length === 0) {
      return null;
    }

    // Sort by date to get nearest expiration
    const sortedExpirations = allExpirations.sort((a, b) =>
      new Date(a).getTime() - new Date(b).getTime()
    );

    // Try first 3 nearest expirations
    for (const expiration of sortedExpirations.slice(0, 3)) {
      const chainData = await getOptionsChain(symbol, expiration);
      const contracts = chainData.map(opt => parseOptionContract(opt, symbol));

      // Filter by option type
      const optionType = direction === 'long' ? 'call' : 'put';
      const filteredByType = contracts.filter(c => c.type === optionType);

      if (filteredByType.length === 0) continue;

      // Find ATM strike
      const atmStrike = filteredByType.reduce((prev, curr) =>
        Math.abs(curr.strike - currentPrice) < Math.abs(prev.strike - currentPrice) ? curr : prev
      );

      // Look for ATM or closest ITM option
      let bestOption: OptionContract | null = null;

      if (direction === 'long') {
        // For calls: prefer strike <= currentPrice (ATM or ITM)
        const atmOrItm = filteredByType.filter(c => c.strike <= currentPrice);
        bestOption = atmOrItm.length > 0
          ? atmOrItm.reduce((prev, curr) => curr.strike > prev.strike ? curr : prev)
          : atmStrike;
      } else {
        // For puts: prefer strike >= currentPrice (ATM or ITM)
        const atmOrItm = filteredByType.filter(c => c.strike >= currentPrice);
        bestOption = atmOrItm.length > 0
          ? atmOrItm.reduce((prev, curr) => curr.strike < prev.strike ? curr : prev)
          : atmStrike;
      }

      // Ensure option has valid pricing
      if (bestOption && bestOption.bid > 0 && bestOption.ask > 0 && bestOption.mid > 0) {
        return bestOption;
      }
    }

    return null;
  } catch (error) {
    console.error(`Error selecting mean reversion option for ${symbol}:`, error);
    return null;
  }
}
