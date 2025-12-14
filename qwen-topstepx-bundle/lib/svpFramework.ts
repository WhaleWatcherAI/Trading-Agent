/**
 * SVP (Session Volume Profile) Framework
 *
 * Pure auction theory approach:
 * - Daily bias from POC migration
 * - Intraday location rules (fade value edges)
 * - LVN/HVN geometry for entries/targets
 * - No chart timeframe dependency - profile anchored to RTH session
 */

export interface SessionProfile {
  date: string;
  poc: number;              // Point of Control
  vah: number;              // Value Area High
  val: number;              // Value Area Low
  vaw: number;              // Value Area Width
  lvns: number[];           // Low Volume Nodes
  hvns: number[];           // High Volume Nodes
  totalVolume: number;
  nodes: Array<{ price: number; volume: number }>;
}

export type Bias = 'bull' | 'bear' | 'neutral';

export interface BiasCalculation {
  bias: Bias;
  pocDrift: number;         // ΔPOC from previous session
  driftGate: number;        // Threshold for significant drift
  prevPOC: number;
  twoDaysAgoPOC: number;
}

export interface TradeOpportunity {
  side: 'long' | 'short' | null;
  entryLVN: number | null;
  entryLimit: number | null;
  tp1_POC: number;
  tp2_Edge: number;
  stopLoss: number;
  insideDeadZone: boolean;
  reasoning: string;
}

/**
 * Calculate session volume profile from bars
 * This should be called with RTH session bars only
 */
export function calculateSessionProfile(
  bars: Array<{ high: number; low: number; volume: number; timestamp: string }>,
  tickSize: number,
  valueAreaPercent: number = 0.70
): SessionProfile {
  if (bars.length === 0) {
    throw new Error('No bars provided for session profile');
  }

  const sessionDate = new Date(bars[0].timestamp).toISOString().split('T')[0];

  // Find price range
  const allPrices = bars.flatMap(bar => [bar.high, bar.low]);
  const minPrice = Math.min(...allPrices);
  const maxPrice = Math.max(...allPrices);

  // Build volume-by-price distribution
  const priceMap = new Map<number, number>();

  for (const bar of bars) {
    const barRange = bar.high - bar.low;
    const volumePerTick = barRange > 0 ? bar.volume / (barRange / tickSize) : bar.volume;

    // Distribute volume across price levels
    let currentPrice = bar.low;
    while (currentPrice <= bar.high) {
      const roundedPrice = Math.round(currentPrice / tickSize) * tickSize;
      priceMap.set(roundedPrice, (priceMap.get(roundedPrice) || 0) + volumePerTick);
      currentPrice += tickSize;
    }
  }

  // Sort nodes by price
  const nodes = Array.from(priceMap.entries())
    .map(([price, volume]) => ({ price, volume }))
    .sort((a, b) => a.price - b.price);

  const totalVolume = nodes.reduce((sum, node) => sum + node.volume, 0);

  // POC = highest volume price
  const pocNode = nodes.reduce((max, node) =>
    node.volume > max.volume ? node : max
  );
  const poc = pocNode.price;

  // Value Area = 70% of volume around POC
  const targetVolume = totalVolume * valueAreaPercent;
  let vaVolume = pocNode.volume;
  let vahIndex = nodes.indexOf(pocNode);
  let valIndex = nodes.indexOf(pocNode);

  while (vaVolume < targetVolume && (vahIndex < nodes.length - 1 || valIndex > 0)) {
    const volumeAbove = vahIndex < nodes.length - 1 ? nodes[vahIndex + 1].volume : 0;
    const volumeBelow = valIndex > 0 ? nodes[valIndex - 1].volume : 0;

    if (volumeAbove >= volumeBelow && vahIndex < nodes.length - 1) {
      vahIndex++;
      vaVolume += volumeAbove;
    } else if (valIndex > 0) {
      valIndex--;
      vaVolume += volumeBelow;
    } else {
      break;
    }
  }

  const vah = nodes[vahIndex].price;
  const val = nodes[valIndex].price;
  const vaw = vah - val;

  // LVNs = local volume minima (troughs)
  const lvns: number[] = [];
  const avgVolume = totalVolume / nodes.length;
  const lvnThreshold = avgVolume * 0.5;

  for (let i = 1; i < nodes.length - 1; i++) {
    const node = nodes[i];
    const prevNode = nodes[i - 1];
    const nextNode = nodes[i + 1];

    if (node.volume < prevNode.volume &&
        node.volume < nextNode.volume &&
        node.volume < lvnThreshold) {
      lvns.push(node.price);
    }
  }

  // HVNs = local volume maxima (peaks)
  const hvns: number[] = [];
  const hvnThreshold = avgVolume * 1.5;

  for (let i = 1; i < nodes.length - 1; i++) {
    const node = nodes[i];
    const prevNode = nodes[i - 1];
    const nextNode = nodes[i + 1];

    if (node.volume > prevNode.volume &&
        node.volume > nextNode.volume &&
        node.volume > hvnThreshold) {
      hvns.push(node.price);
    }
  }

  return {
    date: sessionDate,
    poc,
    vah,
    val,
    vaw,
    lvns,
    hvns,
    totalVolume,
    nodes,
  };
}

/**
 * Calculate daily bias from POC migration
 */
export function calculateBias(
  todayProfile: SessionProfile,
  yesterdayProfile: SessionProfile | null,
  twoDaysAgoProfile: SessionProfile | null,
  historicalProfiles: SessionProfile[]
): BiasCalculation {
  if (!yesterdayProfile) {
    return {
      bias: 'neutral',
      pocDrift: 0,
      driftGate: 0,
      prevPOC: todayProfile.poc,
      twoDaysAgoPOC: todayProfile.poc,
    };
  }

  // Calculate drift gate from recent VAW median
  const last5VAWs = historicalProfiles
    .slice(-5)
    .map(p => p.vaw)
    .sort((a, b) => a - b);
  const medianVAW = last5VAWs[Math.floor(last5VAWs.length / 2)] || todayProfile.vaw;
  const driftGate = 0.15 * medianVAW; // Tune: 0.10–0.20

  const pocDrift = todayProfile.poc - yesterdayProfile.poc;
  const prevPOC = yesterdayProfile.poc;
  const twoDaysAgoPOC = twoDaysAgoProfile?.poc || prevPOC;

  // Bias logic
  let bias: Bias = 'neutral';

  // Bull: ΔPOC₁ > gate and POCᵈ ≥ POCᵈ⁻²
  if (pocDrift > driftGate && todayProfile.poc >= twoDaysAgoPOC) {
    bias = 'bull';
  }
  // Bear: ΔPOC₁ < −gate and POCᵈ ≤ POCᵈ⁻²
  else if (pocDrift < -driftGate && todayProfile.poc <= twoDaysAgoPOC) {
    bias = 'bear';
  }

  return {
    bias,
    pocDrift,
    driftGate,
    prevPOC,
    twoDaysAgoPOC,
  };
}

/**
 * Evaluate trade opportunity at current price
 */
export function evaluateTradeOpportunity(
  currentPrice: number,
  bias: Bias,
  todayProfile: SessionProfile,
  tickSize: number
): TradeOpportunity {
  const deadZone = 0.10 * todayProfile.vaw;

  // POC dead-zone check
  const insideDeadZone = Math.abs(currentPrice - todayProfile.poc) <= deadZone;

  if (insideDeadZone) {
    return {
      side: null,
      entryLVN: null,
      entryLimit: null,
      tp1_POC: todayProfile.poc,
      tp2_Edge: todayProfile.vah,
      stopLoss: currentPrice,
      insideDeadZone: true,
      reasoning: 'Inside POC dead-zone',
    };
  }

  // Find nearest LVN
  const nearestLVN = findNearestLVN(currentPrice, todayProfile.lvns);
  const lvnProximity = Math.max(3 * tickSize, 0.15 * todayProfile.vaw);

  let side: 'long' | 'short' | null = null;
  let entryLimit: number | null = null;
  let tp1_POC = todayProfile.poc;
  let tp2_Edge = todayProfile.vah;
  let stopLoss = currentPrice;
  let reasoning = '';

  if (bias === 'bull') {
    // Prefer longs at VAL taps or LVN troughs below/at VAL
    const nearVAL = Math.abs(currentPrice - todayProfile.val) <= lvnProximity;
    const lvnBelowVAL = nearestLVN !== null && nearestLVN <= todayProfile.val &&
                        Math.abs(currentPrice - nearestLVN) <= lvnProximity;

    if (nearVAL || lvnBelowVAL) {
      side = 'long';
      entryLimit = nearestLVN || currentPrice;
      tp2_Edge = findTargetWithHVN(currentPrice, todayProfile.vah, todayProfile.hvns, 'long');
      stopLoss = entryLimit - 0.01 * todayProfile.vaw; // Stop BELOW entry
      reasoning = `Bull bias: Long at ${nearVAL ? 'VAL' : 'LVN below VAL'}`;
    } else {
      reasoning = 'Bull bias: Not at VAL or LVN';
    }
  } else if (bias === 'bear') {
    // Prefer shorts at VAH taps or LVN peaks above/at VAH
    const nearVAH = Math.abs(currentPrice - todayProfile.vah) <= lvnProximity;
    const lvnAboveVAH = nearestLVN !== null && nearestLVN >= todayProfile.vah &&
                        Math.abs(currentPrice - nearestLVN) <= lvnProximity;

    if (nearVAH || lvnAboveVAH) {
      side = 'short';
      entryLimit = nearestLVN || currentPrice;
      tp1_POC = todayProfile.poc;
      tp2_Edge = findTargetWithHVN(currentPrice, todayProfile.val, todayProfile.hvns, 'short');
      stopLoss = entryLimit + 0.01 * todayProfile.vaw; // Stop ABOVE entry
      reasoning = `Bear bias: Short at ${nearVAH ? 'VAH' : 'LVN above VAH'}`;
    } else {
      reasoning = 'Bear bias: Not at VAH or LVN';
    }
  } else {
    // Neutral: Fade edges inside value toward POC
    const nearVAH = Math.abs(currentPrice - todayProfile.vah) <= lvnProximity;
    const nearVAL = Math.abs(currentPrice - todayProfile.val) <= lvnProximity;

    if (nearVAH) {
      side = 'short';
      entryLimit = nearestLVN || currentPrice;
      tp1_POC = todayProfile.poc;
      tp2_Edge = todayProfile.val;
      stopLoss = entryLimit + 0.01 * todayProfile.vaw; // Stop ABOVE entry
      reasoning = 'Neutral: Fade VAH toward POC';
    } else if (nearVAL) {
      side = 'long';
      entryLimit = nearestLVN || currentPrice;
      tp1_POC = todayProfile.poc;
      tp2_Edge = todayProfile.vah;
      stopLoss = entryLimit - 0.01 * todayProfile.vaw; // Stop BELOW entry
      reasoning = 'Neutral: Fade VAL toward POC';
    } else {
      reasoning = 'Neutral: Not at value edges';
    }
  }

  return {
    side,
    entryLVN: nearestLVN,
    entryLimit,
    tp1_POC,
    tp2_Edge,
    stopLoss,
    insideDeadZone: false,
    reasoning,
  };
}

/**
 * Find nearest LVN to current price
 */
function findNearestLVN(price: number, lvns: number[]): number | null {
  if (lvns.length === 0) return null;

  return lvns.reduce((closest, lvn) => {
    const distCurrent = Math.abs(lvn - price);
    const distClosest = closest === null ? Infinity : Math.abs(closest - price);
    return distCurrent < distClosest ? lvn : closest;
  }, null as number | null);
}

/**
 * Find target considering HVN obstacles
 */
function findTargetWithHVN(
  entryPrice: number,
  edgeTarget: number,
  hvns: number[],
  side: 'long' | 'short'
): number {
  // Find HVNs between entry and edge target
  const hvnsBetween = hvns.filter(hvn =>
    side === 'long'
      ? hvn > entryPrice && hvn < edgeTarget
      : hvn < entryPrice && hvn > edgeTarget
  );

  if (hvnsBetween.length === 0) {
    return edgeTarget;
  }

  // Return nearest HVN (obstacle)
  const nearestHVN = hvnsBetween.reduce((closest, hvn) => {
    const distCurrent = Math.abs(hvn - entryPrice);
    const distClosest = Math.abs(closest - entryPrice);
    return distCurrent < distClosest ? hvn : closest;
  });

  return nearestHVN;
}

/**
 * Check if we're inside prior session's value area (for gap context)
 */
export function isInsidePriorValue(
  currentPrice: number,
  priorProfile: SessionProfile | null
): boolean {
  if (!priorProfile) return true;
  return currentPrice >= priorProfile.val && currentPrice <= priorProfile.vah;
}
