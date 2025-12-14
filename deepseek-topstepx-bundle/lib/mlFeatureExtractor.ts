import { FuturesMarketData } from './openaiTradingAgent';

export interface MlFeatureSnapshot {
  symbol: string;
  timestamp: string;
  features: Record<string, number | null>;
}

function safeNum(value: any): number | null {
  const num = typeof value === 'string' ? Number(value) : value;
  return Number.isFinite(num) ? Number(num) : null;
}

function estimateTickSizeFromMarket(marketData: FuturesMarketData): number {
  const poc = marketData.volumeProfile?.poc;
  const distTicks = marketData.marketStats?.distance_to_poc_ticks;
  if (poc && distTicks && distTicks > 0) {
    const tick = Math.abs(marketData.currentPrice - poc) / distTicks;
    if (Number.isFinite(tick) && tick > 0) {
      return tick;
    }
  }
  return 0.25;
}

export function buildMlFeatureSnapshot(marketData: FuturesMarketData): MlFeatureSnapshot {
  const tickSize = estimateTickSizeFromMarket(marketData);
  const currentPrice = marketData.currentPrice ?? 0;
  const poc = marketData.volumeProfile?.poc ?? currentPrice;
  const vah = marketData.volumeProfile?.vah ?? currentPrice;
  const val = marketData.volumeProfile?.val ?? currentPrice;

  const wall = marketData.microstructure?.nearestRestingWallInDirection;
  const flow = marketData.flowSignals;
  const volRegime = marketData.marketStats?.volatilityRegime;
  const reversal = marketData.reversalScores;

  const absorption = marketData.absorption || [];
  const absorptionBid = absorption
    .filter(a => a.side === 'bid')
    .map(a => a.strength)
    .filter(Number.isFinite);
  const absorptionAsk = absorption
    .filter(a => a.side === 'ask')
    .map(a => a.strength)
    .filter(Number.isFinite);
  const exhaustion = marketData.exhaustion || [];
  const exhaustionBid = exhaustion
    .filter(a => a.side === 'bid')
    .map(a => a.strength)
    .filter(Number.isFinite);
  const exhaustionAsk = exhaustion
    .filter(a => a.side === 'ask')
    .map(a => a.strength)
    .filter(Number.isFinite);

  const marketState = (marketData.marketState?.state || '').toLowerCase();
  const connection = marketData.connectionHealth;

  const features: Record<string, number | null> = {
    current_price: safeNum(currentPrice),
    dist_to_poc_ticks: safeNum((currentPrice - poc) / tickSize),
    dist_to_vah_ticks: safeNum((currentPrice - vah) / tickSize),
    dist_to_val_ticks: safeNum((currentPrice - val) / tickSize),
    distance_to_poc_ticks_smoothed: safeNum(marketData.marketStats?.distance_to_poc_ticks),
    session_range_ticks: safeNum(marketData.marketStats?.session_range_ticks),
    session_range_percentile: safeNum(marketData.marketStats?.session_range_percentile),
    time_above_value_sec: safeNum(marketData.marketStats?.time_above_value_sec),
    time_in_value_sec: safeNum(marketData.marketStats?.time_in_value_sec),
    time_below_value_sec: safeNum(marketData.marketStats?.time_below_value_sec),
    cvd_slope_5min: safeNum(marketData.marketStats?.cvd_slope_5min),
    cvd_slope_15min: safeNum(marketData.marketStats?.cvd_slope_15min),
    atr5m: safeNum(marketData.marketStats?.atr5m),
    current_range_ticks: safeNum(marketData.marketStats?.currentRangeTicks),

    cvd_value: safeNum(marketData.cvd?.value),
    delta_last_1m: safeNum(flow?.deltaLast1m),
    delta_last_5m: safeNum(flow?.deltaLast5m),
    cvd_slope_short: safeNum(flow?.cvdSlopeShort),
    cvd_slope_long: safeNum(flow?.cvdSlopeLong),
    cvd_divergence_is_strong: flow?.cvdDivergence === 'strong' ? 1 : 0,
    cvd_divergence_is_weak: flow?.cvdDivergence === 'weak' ? 1 : 0,

    absorption_bid_strength_max: absorptionBid.length ? Math.max(...absorptionBid) : null,
    absorption_ask_strength_max: absorptionAsk.length ? Math.max(...absorptionAsk) : null,
    exhaustion_bid_strength_max: exhaustionBid.length ? Math.max(...exhaustionBid) : null,
    exhaustion_ask_strength_max: exhaustionAsk.length ? Math.max(...exhaustionAsk) : null,

    nearest_wall_distance_ticks: wall && Number.isFinite(wall.distance) ? safeNum(wall.distance) : null,
    nearest_wall_size: safeNum(wall?.size),
    nearest_wall_is_ask: wall ? (wall.side === 'ask' ? 1 : 0) : null,
    weak_wall_detected: marketData.microstructure?.weakWallDetected ? 1 : 0,

    vol_regime_low: volRegime === 'low' ? 1 : 0,
    vol_regime_normal: volRegime === 'normal' ? 1 : 0,
    vol_regime_high: volRegime === 'high' ? 1 : 0,

    reversal_long: safeNum(reversal?.long),
    reversal_short: safeNum(reversal?.short),

    order_flow_confirmed: marketData.orderFlowConfirmed ? 1 : 0,

    market_state_up: marketState.includes('up') ? 1 : 0,
    market_state_down: marketState.includes('down') ? 1 : 0,
    market_state_balance: marketState.includes('balance') ? 1 : 0,
    market_state_breakout_fail: marketState.includes('failed_breakout') ? 1 : 0,

    poc_cross_last_5m: safeNum(marketData.pocCrossStats?.count_last_5min),
    poc_cross_last_15m: safeNum(marketData.pocCrossStats?.count_last_15min),
    poc_cross_last_30m: safeNum(marketData.pocCrossStats?.count_last_30min),
    poc_time_since_last_cross_sec: safeNum(marketData.pocCrossStats?.time_since_last_cross_sec),

    performance_win_rate: safeNum(marketData.performance?.win_rate),
    performance_profit_factor: safeNum(marketData.performance?.profit_factor),
    performance_trade_count: safeNum(marketData.performance?.trade_count),

    // Connection / gap awareness
    seconds_since_market_hub_event: safeNum(connection?.lastMarketHubEventAgoSec),
    seconds_since_market_hub_disconnect: safeNum(connection?.lastMarketHubDisconnectAgoSec),
    market_hub_connected: connection?.marketHubState === 'connected' ? 1 : 0,
    market_hub_reconnecting: connection?.marketHubState === 'reconnecting' ? 1 : 0,
  };

  return {
    symbol: marketData.symbol,
    timestamp: marketData.timestamp,
    features,
  };
}
