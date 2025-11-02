import axios from 'axios';

const TRADIER_API_KEY = process.env.TRADIER_API_KEY || '';
const TRADIER_BASE_URL = process.env.TRADIER_BASE_URL || 'https://sandbox.tradier.com/v1';

const tradierClient = axios.create({
  baseURL: TRADIER_BASE_URL,
  headers: {
    Authorization: `Bearer ${TRADIER_API_KEY}`,
    Accept: 'application/json',
  },
});

export type GexMode = 'intraday' | 'swing' | 'leaps';

export interface GexByStrike {
  strike: number;
  callGex: number;
  putGex: number;
  netGex: number;
  callGexPerDollar: number;
  putGexPerDollar: number;
  netGexPerDollar: number;
  oi: number;
  volume: number;
}

export interface GexOptionContract {
  strike: number;
  expiration: string;
  type: 'call' | 'put';
  volume: number;
  openInterest: number;
  last: number;
  bid: number;
  ask: number;
  impliedVol?: number;
}

export interface GexExpirationSummary {
  expiration: string;
  dte: number;
  totalCallGex: number;
  totalPutGex: number;
  netGex: number;
  netGexPerDollar: number;
}

export interface GexCalculationResult {
  symbol: string;
  stockPrice: number;
  gexData: GexByStrike[];
  gammaWall: number;
  maxGex: number;
  summary: {
    totalCallGex: number;
    totalPutGex: number;
    netGex: number;
    totalCallGexPerDollar: number;
    totalPutGexPerDollar: number;
    netGexPerDollar: number;
  };
  expirations: string[];
  expirationDetails: { date: string; dte: number }[];
  expirationSummaries: GexExpirationSummary[];
  source: string;
  calculatedAt: string;
  contracts: GexOptionContract[];
}

const DTE_RANGES: Record<GexMode, { min: number; max: number }> = {
  intraday: { min: 3, max: 7 },
  swing: { min: 10, max: 20 },
  leaps: { min: 45, max: 400 },
};

export async function calculateGexForSymbol(symbol: string, mode: GexMode = 'intraday'): Promise<GexCalculationResult> {
  console.log(`📊 Calculating GEX profile for ${symbol} (${mode})`);

  const quoteResponse = await tradierClient.get('/markets/quotes', {
    params: { symbols: symbol },
  });

  const quote = quoteResponse.data?.quotes?.quote;
  if (!quote) {
    throw new Error(`No quote data returned for ${symbol}`);
  }

  const stockPrice = parseFloat(quote.last || quote.close || quote.prevclose || 0);
  if (!Number.isFinite(stockPrice) || stockPrice <= 0) {
    throw new Error(`Invalid stock price for ${symbol}`);
  }

  const expirationsResponse = await tradierClient.get('/markets/options/expirations', {
    params: {
      symbol,
      includeAllRoots: true,
      strikes: false,
    },
  });

  const expirations = expirationsResponse.data?.expirations?.date;
  if (!expirations || (Array.isArray(expirations) && expirations.length === 0)) {
    throw new Error(`No options expirations found for ${symbol}`);
  }

  const MS_PER_DAY = 1000 * 60 * 60 * 24;
  const today = new Date();
  const todayUTC = Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate());

  const normalizedExpirations = (Array.isArray(expirations) ? expirations : [expirations])
    .map((dateStr: string) => {
      const expDate = new Date(`${dateStr}T16:00:00Z`);
      if (Number.isNaN(expDate.getTime())) return null;
      const timeDiff = expDate.getTime() - todayUTC;
      const dte = Math.max(0, Math.round(timeDiff / MS_PER_DAY));
      return { date: dateStr, expDate, dte };
    })
    .filter((item): item is { date: string; expDate: Date; dte: number } => !!item)
    .sort((a, b) => a.expDate.getTime() - b.expDate.getTime());

  if (normalizedExpirations.length === 0) {
    throw new Error(`Unable to normalize expirations for ${symbol}`);
  }

  const range = DTE_RANGES[mode];
  let filtered = normalizedExpirations;
  if (range) {
    const candidates = normalizedExpirations.filter(({ dte }) => dte >= range.min && dte <= range.max);
    if (candidates.length > 0) {
      filtered = candidates;
    }
  }

  const MAX_EXPIRATIONS = 3;
  const selectedExpirations = filtered.slice(0, MAX_EXPIRATIONS);

  if (selectedExpirations.length === 0) {
    throw new Error(`No expirations within configured DTE window for ${symbol} (${mode})`);
  }

  const CONTRACT_MULTIPLIER = 100;
  const gexByStrike: Record<number, GexByStrike> = {};
  const contracts: GexOptionContract[] = [];
  const expirationTotals: Record<string, GexExpirationSummary> = {};
  const dteLookup = Object.fromEntries(selectedExpirations.map(item => [item.date, item.dte]));

  for (const { date: expiration } of selectedExpirations) {
    const chainResponse = await tradierClient.get('/markets/options/chains', {
      params: {
        symbol,
        expiration,
        greeks: true,
      },
    });

    const options = chainResponse.data?.options?.option;
    if (!options) continue;

    const optionsArray = Array.isArray(options) ? options : [options];

    optionsArray.forEach((option: any) => {
      const strike = parseFloat(option.strike);
      const openInterest = parseInt(option.open_interest || 0);
      const gamma = parseFloat(option.greeks?.gamma || 0);
      const optionType = option.option_type;
      const volume = parseInt(option.volume || 0);
      const last = parseFloat(option.last || 0);
      const bid = parseFloat(option.bid || 0);
      const ask = parseFloat(option.ask || 0);
      const iv = option.greeks?.mid_iv ? parseFloat(option.greeks.mid_iv) : undefined;

      if (!Number.isFinite(strike) || strike <= 0) return;

      contracts.push({
        strike,
        expiration,
        type: optionType === 'call' ? 'call' : 'put',
        volume,
        openInterest,
        last,
        bid,
        ask,
        impliedVol: iv,
      });

      if (!expirationTotals[expiration]) {
        expirationTotals[expiration] = {
          expiration,
          dte: dteLookup[expiration] ?? 0,
          totalCallGex: 0,
          totalPutGex: 0,
          netGex: 0,
          netGexPerDollar: 0,
        };
      }

      if (!gexByStrike[strike]) {
        gexByStrike[strike] = {
          strike,
          callGex: 0,
          putGex: 0,
          netGex: 0,
          callGexPerDollar: 0,
          putGexPerDollar: 0,
          netGexPerDollar: 0,
          oi: 0,
          volume: 0,
        };
      }

      const deltaSharesPerDollar = gamma * openInterest * CONTRACT_MULTIPLIER;
      const notionalPerDollar = deltaSharesPerDollar * stockPrice;
      const notionalPerPercent = deltaSharesPerDollar * (stockPrice * stockPrice * 0.01);

      if (optionType === 'call') {
        const absPercent = Math.abs(notionalPerPercent);
        const absDollar = Math.abs(notionalPerDollar);
        gexByStrike[strike].callGex += absPercent;
        gexByStrike[strike].callGexPerDollar += absDollar;
        gexByStrike[strike].netGex += notionalPerPercent;
        gexByStrike[strike].netGexPerDollar += notionalPerDollar;
        expirationTotals[expiration].totalCallGex += absPercent;
        expirationTotals[expiration].netGex += notionalPerPercent;
        expirationTotals[expiration].netGexPerDollar += notionalPerDollar;
      } else if (optionType === 'put') {
        const absPercent = Math.abs(notionalPerPercent);
        const absDollar = Math.abs(notionalPerDollar);
        gexByStrike[strike].putGex += absPercent;
        gexByStrike[strike].putGexPerDollar += absDollar;
        gexByStrike[strike].netGex -= notionalPerPercent;
        gexByStrike[strike].netGexPerDollar -= notionalPerDollar;
        expirationTotals[expiration].totalPutGex += absPercent;
        expirationTotals[expiration].netGex -= notionalPerPercent;
        expirationTotals[expiration].netGexPerDollar -= notionalPerDollar;
      }

      gexByStrike[strike].oi += openInterest;
      gexByStrike[strike].volume += volume;
    });
  }

  const gexData = Object.values(gexByStrike)
    .sort((a, b) => a.strike - b.strike)
    .filter(item => item.oi > 0);

  if (gexData.length === 0) {
    throw new Error(`No valid GEX data for ${symbol}`);
  }

  const maxGex = Math.max(...gexData.map(item => Math.abs(item.netGex)));
  const gammaWall = gexData.find(item => Math.abs(item.netGex) === maxGex)?.strike || stockPrice;

  return {
    symbol,
    stockPrice,
    gexData,
    gammaWall,
    maxGex,
    summary: {
      totalCallGex: gexData.reduce((sum, item) => sum + item.callGex, 0),
      totalPutGex: gexData.reduce((sum, item) => sum + item.putGex, 0),
      netGex: gexData.reduce((sum, item) => sum + item.netGex, 0),
      totalCallGexPerDollar: gexData.reduce((sum, item) => sum + item.callGexPerDollar, 0),
      totalPutGexPerDollar: gexData.reduce((sum, item) => sum + item.putGexPerDollar, 0),
      netGexPerDollar: gexData.reduce((sum, item) => sum + item.netGexPerDollar, 0),
    },
    expirations: selectedExpirations.map(({ date }) => date),
    expirationDetails: selectedExpirations.map(({ date, dte }) => ({ date, dte })),
    expirationSummaries: selectedExpirations
      .map(({ date }) => expirationTotals[date])
      .filter((summary): summary is GexExpirationSummary => Boolean(summary)),
    source: `Tradier Options Chain (${mode} mode)`,
    calculatedAt: new Date().toISOString(),
    contracts,
  };
}
