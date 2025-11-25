/**
 * Volume Profile - Session RTH + Rolling Micro-Composite
 *
 * Provides POC, VAH, VAL, LVN, HVN identification for mean-reversion entry quality
 */

export interface VolumeNode {
  price: number;
  volume: number;
}

export interface VolumeProfileResult {
  poc: number;              // Point of Control (highest volume price)
  vah: number;              // Value Area High (70% volume upper bound)
  val: number;              // Value Area Low (70% volume lower bound)
  vaw: number;              // Value Area Width (VAH - VAL)
  lvns: number[];           // Low Volume Nodes (troughs)
  hvns: number[];           // High Volume Nodes (peaks)
  totalVolume: number;
  nodes: VolumeNode[];
}

export interface EdgeScore {
  score: number;            // 0-3
  touchesVAH_VAL: boolean;  // +1 if near VAH/VAL
  nearLVN: boolean;         // +1 if within 4-8 ticks of LVN
  farFromPOC: boolean;      // +1 if distance to POC ≥ 0.4×VAW
  passesFilter: boolean;    // score >= 2 AND not inside ±0.15×VAW of POC
}

/**
 * Calculate volume profile from bars
 */
export function calculateVolumeProfile(
  bars: Array<{ high: number; low: number; volume: number }>,
  tickSize: number,
  valueAreaPercent: number = 0.70
): VolumeProfileResult {
  if (bars.length === 0) {
    throw new Error('No bars provided for volume profile calculation');
  }

  // Find price range
  const allPrices = bars.flatMap(bar => [bar.high, bar.low]);
  const minPrice = Math.min(...allPrices);
  const maxPrice = Math.max(...allPrices);

  // Create volume nodes at each tick level
  const priceMap = new Map<number, number>();

  for (const bar of bars) {
    const barRange = bar.high - bar.low;
    const volumePerTick = barRange > 0 ? bar.volume / (barRange / tickSize) : bar.volume;

    // Distribute volume across price levels in the bar
    let currentPrice = bar.low;
    while (currentPrice <= bar.high) {
      const roundedPrice = Math.round(currentPrice / tickSize) * tickSize;
      priceMap.set(roundedPrice, (priceMap.get(roundedPrice) || 0) + volumePerTick);
      currentPrice += tickSize;
    }
  }

  // Sort nodes by price
  const nodes: VolumeNode[] = Array.from(priceMap.entries())
    .map(([price, volume]) => ({ price, volume }))
    .sort((a, b) => a.price - b.price);

  const totalVolume = nodes.reduce((sum, node) => sum + node.volume, 0);

  // Find POC (Point of Control - highest volume price)
  const pocNode = nodes.reduce((max, node) =>
    node.volume > max.volume ? node : max
  );
  const poc = pocNode.price;

  // Calculate Value Area (70% of volume around POC)
  const targetVolume = totalVolume * valueAreaPercent;
  let vaVolume = pocNode.volume;
  let vahIndex = nodes.indexOf(pocNode);
  let valIndex = nodes.indexOf(pocNode);

  // Expand value area up and down from POC
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

  // Identify LVNs (Low Volume Nodes - local minima)
  const lvns: number[] = [];
  const avgVolume = totalVolume / nodes.length;
  const lvnThreshold = avgVolume * 0.5; // LVNs are <50% of average volume

  for (let i = 1; i < nodes.length - 1; i++) {
    const node = nodes[i];
    const prevNode = nodes[i - 1];
    const nextNode = nodes[i + 1];

    // Local minimum and below threshold
    if (node.volume < prevNode.volume &&
        node.volume < nextNode.volume &&
        node.volume < lvnThreshold) {
      lvns.push(node.price);
    }
  }

  // Identify HVNs (High Volume Nodes - local maxima)
  const hvns: number[] = [];
  const hvnThreshold = avgVolume * 1.5; // HVNs are >150% of average volume

  for (let i = 1; i < nodes.length - 1; i++) {
    const node = nodes[i];
    const prevNode = nodes[i - 1];
    const nextNode = nodes[i + 1];

    // Local maximum and above threshold
    if (node.volume > prevNode.volume &&
        node.volume > nextNode.volume &&
        node.volume > hvnThreshold) {
      hvns.push(node.price);
    }
  }

  return {
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
 * Calculate edge score for a potential entry
 */
export function calculateEdgeScore(
  price: number,
  profile: VolumeProfileResult,
  tickSize: number
): EdgeScore {
  let score = 0;
  const touchThreshold = 2 * tickSize; // Within 2 ticks
  const lvnSearchTicks = { min: 4, max: 8 }; // 4-8 ticks away

  // +1 if price touches VAH/VAL
  const touchesVAH_VAL =
    Math.abs(price - profile.vah) <= touchThreshold ||
    Math.abs(price - profile.val) <= touchThreshold;
  if (touchesVAH_VAL) score++;

  // +1 if within 4-8 ticks of an LVN
  const nearLVN = profile.lvns.some(lvn => {
    const distance = Math.abs(price - lvn);
    const tickDistance = distance / tickSize;
    return tickDistance >= lvnSearchTicks.min && tickDistance <= lvnSearchTicks.max;
  });
  if (nearLVN) score++;

  // +1 if distance to POC ≥ 0.4×VAW
  const distanceToPOC = Math.abs(price - profile.poc);
  const farFromPOC = distanceToPOC >= 0.4 * profile.vaw;
  if (farFromPOC) score++;

  // Final filter: score >= 2 AND not inside ±0.15×VAW of POC
  const pocExclusionZone = 0.15 * profile.vaw;
  const insidePOCZone = distanceToPOC < pocExclusionZone;
  const passesFilter = score >= 2 && !insidePOCZone;

  return {
    score,
    touchesVAH_VAL,
    nearLVN,
    farFromPOC,
    passesFilter,
  };
}

/**
 * Find optimal limit order price at/inside LVN
 */
export function findLimitOrderPrice(
  currentPrice: number,
  side: 'long' | 'short',
  profile: VolumeProfileResult,
  tickSize: number
): number | null {
  // Find nearest LVN in the direction of mean reversion
  const relevantLVNs = side === 'long'
    ? profile.lvns.filter(lvn => lvn <= currentPrice) // Long: LVN below current price
    : profile.lvns.filter(lvn => lvn >= currentPrice); // Short: LVN above current price

  if (relevantLVNs.length === 0) {
    return null; // No suitable LVN
  }

  // Get closest LVN
  const nearestLVN = relevantLVNs.reduce((closest, lvn) => {
    const distCurrent = Math.abs(lvn - currentPrice);
    const distClosest = Math.abs(closest - currentPrice);
    return distCurrent < distClosest ? lvn : closest;
  });

  // Place limit just inside the LVN (1 tick inside for better fill probability)
  return side === 'long'
    ? nearestLVN + tickSize  // Long: 1 tick above LVN
    : nearestLVN - tickSize; // Short: 1 tick below LVN
}

/**
 * Determine final TP: outer band OR nearest HVN/POC if closer
 */
export function findOptimalTP(
  entryPrice: number,
  side: 'long' | 'short',
  outerBand: number,
  profile: VolumeProfileResult,
  tickSize: number
): number {
  // Direction: long targets higher, short targets lower
  const targetDirection = side === 'long' ? 1 : -1;

  // Check if POC is between entry and outer band
  const pocBetween = side === 'long'
    ? profile.poc > entryPrice && profile.poc < outerBand
    : profile.poc < entryPrice && profile.poc > outerBand;

  // Find HVNs between entry and outer band
  const hvnsBetween = profile.hvns.filter(hvn =>
    side === 'long'
      ? hvn > entryPrice && hvn < outerBand
      : hvn < entryPrice && hvn > outerBand
  );

  // If POC is closer, use it
  if (pocBetween) {
    const distanceToPOC = Math.abs(profile.poc - entryPrice);
    const distanceToOuter = Math.abs(outerBand - entryPrice);
    if (distanceToPOC < distanceToOuter * 0.8) { // POC is <80% of distance to outer
      return profile.poc;
    }
  }

  // If nearest HVN is significantly closer, use it
  if (hvnsBetween.length > 0) {
    const nearestHVN = hvnsBetween.reduce((closest, hvn) => {
      const distCurrent = Math.abs(hvn - entryPrice);
      const distClosest = Math.abs(closest - entryPrice);
      return distCurrent < distClosest ? hvn : closest;
    });

    const distanceToHVN = Math.abs(nearestHVN - entryPrice);
    const distanceToOuter = Math.abs(outerBand - entryPrice);
    if (distanceToHVN < distanceToOuter * 0.7) { // HVN is <70% of distance to outer
      return nearestHVN;
    }
  }

  // Default to outer band
  return outerBand;
}
