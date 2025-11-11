export interface StockPrice {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: Date;
}

export interface OptionsTrade {
  symbol: string;
  underlying: string;
  strike: number;
  expiration: string;
  type: 'call' | 'put';
  side: 'bid' | 'ask' | 'mid';
  premium: number;
  volume: number;
  openInterest: number;
  timestamp: Date;
  unusual: boolean;
  greeks?: {
    delta?: number;
    gamma?: number;
    theta?: number;
    vega?: number;
    rho?: number;
    phi?: number;
    bid_iv?: number;
    mid_iv?: number;
    ask_iv?: number;
    smv_vol?: number;
  };
}

export interface InstitutionalTrade {
  symbol: string;
  shares: number;
  price: number;
  value: number;
  side: 'buy' | 'sell';
  institution: string;
  timestamp: Date;
}

export interface NewsItem {
  title: string;
  summary: string;
  url: string;
  source: string;
  symbols: string[];
  sentiment?: 'bullish' | 'bearish' | 'neutral';
  importance: number; // 0-1
  timestamp: Date;
}

export interface MarketData {
  putCallRatio: number;
  vix: number;
  spy: number;
  marketTide: 'bullish' | 'bearish' | 'neutral';
  timestamp: Date;
}

export interface TradeSignal {
  symbol: string;
  underlying: string;
  contract: string; // Full contract description
  strike: number;
  expiration: string;
  type: 'call' | 'put';
  action: 'buy' | 'sell';
  strategy: 'scalp' | 'intraday' | 'swing' | 'leap' | 'sma_crossover';
  currentPrice: number;
  targetPrice?: number;
  stopLoss?: number;
  rating: number; // 1-10, negative for bearish
  confidence: number; // 0-100
  reasoning: string;
  factors: {
    newsImpact: number;
    institutionalActivity: number;
    optionsFlow: number;
    marketTide: number;
    technicals: number;
  };
  timestamp: Date;
}

export interface AnalysisRequest {
  symbols?: string[];
  strategy?: 'scalp' | 'intraday' | 'swing' | 'leap' | 'all';
  limit?: number;
  threshold?: number; // Minimum score threshold (default: 7)
}

export interface AnalysisResponse {
  trades: TradeSignal[];
  marketOverview: MarketData;
  timestamp: Date;
}

export type BullBearSignal = 'bull' | 'bear' | 'neutral';

export interface OptionFlowSignal {
  signal: BullBearSignal;
  strength: number; // 0-1
  reasoning: string;
}

// New types for all Unusual Whales endpoints
export interface GreekFlow {
  flow_group: string;
  expiry: string;
  timestamp: Date;
  net_call_premium: number;
  net_put_premium: number;
  net_call_volume: number;
  net_put_volume: number;
  dir_delta_flow: number;
  dir_vega_flow: number;
  total_delta_flow: number;
  total_vega_flow: number;
  volume: number;
  transactions: number;
}

export interface SectorFlow {
  sector: string;
  date: string;
  buy_sell: 'buy' | 'sell';
  volume: number;
  avg_price: number;
  premium: number;
  transactions: number;
  uniq_insiders: number;
}

export interface SectorTide {
  sector: string;
  timestamp: Date;
  net_call_premium: number;
  net_put_premium: number;
  net_volume: number;
}

export interface SpotGEX {
  ticker: string;
  time: Date;
  price: number;
  gamma_per_one_percent_move_oi: number;
  gamma_per_one_percent_move_vol: number;
  vanna_per_one_percent_move_oi: number;
  charm_per_one_percent_move_oi: number;
}

export interface VolatilityStats {
  ticker: string;
  date: string;
  iv: number;
  iv_high: number;
  iv_low: number;
  iv_rank: number;
  rv: number;
  rv_high: number;
  rv_low: number;
}

export type LiquidityTier = 'large' | 'mid_small';

export interface RegimeWhaleTrade {
  optionType: 'call' | 'put';
  direction: 'bullish' | 'bearish';
  contracts: number;
  premium: number;
  strike: number;
  expiration: string;
  midpointPrice: number;
  timestamp?: string;
}

export interface RegimeStage1Metrics {
  marketCap?: number | null;
  averageVolume?: number | null;
  optionsVolume?: number | null;
  openInterest?: number | null;
  ivRank?: number | null;
  ivDelta15m?: number | null;
  ivDelta30m?: number | null;
  volumeToOi?: number | null;
}

export interface RegimeStage1Thresholds {
  ivRank: number;
  volumeToOi: number;
  whaleContracts: number;
  whalePremium: number;
}

export interface RegimeStage1Result {
  symbol: string;
  tier: LiquidityTier;
  passes: boolean;
  metrics: RegimeStage1Metrics;
  thresholds: RegimeStage1Thresholds;
  whaleTrades: RegimeWhaleTrade[];
  failedCriteria: string[];
  notes: string[];
}

export type GammaRegime = 'expansion' | 'pinning';

export interface RegimeStage2Summary {
  symbol: string;
  price: number;
  netGex: number;
  netGexPerDollar: number;
  totalCallGex: number;
  totalPutGex: number;
  regime: GammaRegime;
  gammaWall: number;
  gammaFlipLevel?: number;
  slope: 'rising' | 'falling' | 'flat';
  slopeStrength: 'weak' | 'moderate' | 'strong';
  gammaFlipDistance?: number;
  dominantExpirations: RegimeExpirationContribution[];
  trendNarrative: string;
  regimeTransition: 'stable' | 'flip_to_expansion' | 'flip_to_pinning';
  recentSlopeDelta?: number;
  expirations: { date: string; dte: number }[];
  mode: 'scalp' | 'swing' | 'leaps';
}

export interface RegimeExpirationContribution {
  expiration: string;
  dte: number;
  netGex: number;
  totalCallGex: number;
  totalPutGex: number;
}

export interface RegimeGexLevel {
  strike: number;
  netGex: number;
  callGex: number;
  putGex: number;
  netGexPerDollar: number;
  callGexPerDollar: number;
  putGexPerDollar: number;
  oi: number;
  volume: number;
  classification: 'call_wall' | 'put_zone' | 'neutral';
}

export interface RegimeStage3Profile {
  symbol: string;
  gammaWall: number;
  callWalls: RegimeWallDetail[];
  putZones: RegimeWallDetail[];
  profile: RegimeGexLevel[];
  gammaFlipLevel?: number;
  priceInteraction?: 'above_call_wall' | 'below_put_wall' | 'inside_range' | null;
  rangeOutlook: 'range_bound' | 'breakout_watch' | 'downside_risk' | 'upside_risk';
}

export interface RegimeWallDetail extends RegimeGexLevel {
  strength: number;
  distancePct: number;
  isNearPrice: boolean;
  zScore: number;
}

export type RegimeTradeStatus =
  | 'watching'
  | 'entered'
  | 'scaled'
  | 'target_hit'
  | 'stopped'
  | 'expired'
  | 'cancelled';

export interface RegimeTradeLifecycle {
  id: string;
  symbol: string;
  direction: 'long' | 'short';
  strategy: 'scalp' | 'swing' | 'leaps';
  status: RegimeTradeStatus;
  positionSize: 'full' | 'half';
  triggerLevel: number;
  entryPrice: number;
  stopLoss: number;
  firstTarget: number;
  secondaryTarget?: number;
  riskPerShare: number;
  rMultipleAchieved: number;
  timeframeMinutes: number;
  timerExpiry?: string;
  enteredAt?: string;
  lastUpdated: string;
  addOnDone: boolean;
  scaledAt?: string;
  exits: Array<{ type: 'target' | 'stop' | 'time'; price: number; timestamp: string }>;
  history: Array<{ timestamp: string; status: RegimeTradeStatus; note: string }>;
  nextAction: string;
  whaleConfirmation?: RegimeWhaleTrade | null;
  hedgeActive?: boolean;
  hedgeNote?: string;
}

export type RegimeBias = 'bullish' | 'bearish' | 'balanced';
export type RegimeBiasStrength = 'low' | 'medium' | 'high';

export interface BacktestPricePoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface BacktestStrikeHighlight {
  strike: number;
  expiration: string;
  premium: number;
  volume: number;
  openInterest: number;
}

export interface BacktestTradeMarker {
  type: 'entry' | 'exit';
  tradeId: string;
  direction: 'long' | 'short';
  timestamp: string;
  price: number;
  reason: string;
}

export interface BacktestTrade {
  id: string;
  direction: 'long' | 'short';
  entryTimestamp: string;
  exitTimestamp: string;
  entryPrice: number;
  exitPrice: number;
  durationMinutes: number;
  profit: number;
  profitPct: number;
  tradeCost: number;
  exitReason: 'target' | 'flip' | 'stop' | 'flat' | 'end_of_day';
  entryReason: string;
}

export interface SectorTimelinePoint {
  timestamp: string;
  callPremium: number;
  putPremium: number;
  netPremium: number;
  callVolume: number;
  putVolume: number;
  totalTrades: number;
  whaleTrades: RegimeWhaleTrade[];
  bias: RegimeBias;
  strength: RegimeBiasStrength;
  cumulativeCallPremium: number;
  cumulativePutPremium: number;
  topCallHighlight?: BacktestStrikeHighlight;
  topPutHighlight?: BacktestStrikeHighlight;
  price?: BacktestPricePoint;
  markers?: BacktestTradeMarker[];
}

export interface SectorBacktestSummary {
  minutes: number;
  totalCallPremium: number;
  totalPutPremium: number;
  netPremium: number;
  whaleTrades: number;
  dominantBias: RegimeBias;
  biasCounts: Record<RegimeBias, number>;
  regimeTransitions: number;
  dominantExpirations: Array<{
    expiration: string;
    callPremium: number;
    putPremium: number;
    netPremium: number;
    trades: number;
  }>;
  priceChangePct?: number;
  topWhales: RegimeWhaleTrade[];
  tradeCount: number;
  winRate?: number;
  totalProfit: number;
  grossProfit: number;
  grossLoss: number;
  maxDrawdown?: number;
  averageDurationMinutes?: number;
}

export interface EquityCurvePoint {
  timestamp: string;
  equity: number;
}

export interface SectorBacktestResult {
  sector: string;
  mappedSymbol?: string;
  timeline: SectorTimelinePoint[];
  priceSource: 'tradier' | 'unavailable';
  summary: SectorBacktestSummary;
  trades: BacktestTrade[];
  equityCurve: EquityCurvePoint[];
}

export interface AggregatedBacktestSummary {
  sectors: number;
  totalCallPremium: number;
  totalPutPremium: number;
  netPremium: number;
  whaleTrades: number;
  dominantBias: RegimeBias;
  biasCounts: Record<RegimeBias, number>;
  regimeTransitions: number;
  totalProfit: number;
  tradeCount: number;
  winRate?: number;
}

export interface RegimeBacktestResult {
  date: string;
  mode: 'scalp' | 'swing' | 'leaps';
  resolutionMinutes: number;
  dataSources: {
    flow: string;
    prices: 'tradier' | 'unavailable';
  };
  sectors: SectorBacktestResult[];
  aggregated: AggregatedBacktestSummary;
  notes: string[];
}

export interface RegimeTradeSignal {
  id: string;
  symbol: string;
  action: 'buy' | 'sell';
  direction: 'long' | 'short';
  strategy: 'scalp' | 'swing' | 'leaps';
  regime: GammaRegime;
  positionSize: 'full' | 'half';
  entry: {
    price: number;
    triggerLevel: number;
    triggerType: 'breakout' | 'breakdown' | 'fade' | 'range-reversion';
  };
  stopLoss: number;
  firstTarget: number;
  secondaryTarget?: number;
  rationale: string[];
  whaleConfirmation?: RegimeWhaleTrade | null;
  riskPerShare?: number;
  timeframeMinutes?: number;
}

export interface VolatilityRegimeAnalysis {
  symbol: string;
  price: number;
  timestamp: string;
  mode: 'scalp' | 'swing' | 'leaps';
  stage1: RegimeStage1Result;
  stage2: RegimeStage2Summary;
  stage3: RegimeStage3Profile;
  tradeSignals: RegimeTradeSignal[];
  activeTrades: RegimeTradeLifecycle[];
}

export interface VolatilityRegimeResponse {
  symbols: string[];
  mode: 'scalp' | 'swing' | 'leaps';
  universe: RegimeStage1Result[];
  analyses: VolatilityRegimeAnalysis[];
  activeTrades: RegimeTradeLifecycle[];
  generatedAt: string;
}

export interface TradierBalance {
  totalCash: number;
  netValue: number;
  buyingPower: number;
  dayTradingBuyingPower?: number;
  maintenanceMargin?: number;
  accruedInterest?: number;
  pendingOrdersCount?: number;
  timestamp: string;
}

export interface TradierPosition {
  symbol: string;
  quantity: number;
  costBasis: number;
  lastPrice: number;
  marketValue: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
}

export type TradierOrderStatus =
  | 'open'
  | 'filled'
  | 'cancelled'
  | 'expired'
  | 'rejected'
  | 'pending';

export interface TradierOrder {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: string;
  status: TradierOrderStatus;
  quantity: number;
  filledQuantity: number;
  remainingQuantity: number;
  limitPrice?: number;
  stopPrice?: number;
  submittedAt: string;
  updatedAt?: string;
}

export interface AccountSnapshot {
  balances: TradierBalance | null;
  positions: TradierPosition[];
  orders: TradierOrder[];
  fetchedAt: string;
}
