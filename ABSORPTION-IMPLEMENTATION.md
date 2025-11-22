# BTC Dashboard Absorption Detection Implementation

## Overview
Absorption detection identifies when large aggressive trades hit resting orderbook walls but price fails to break through - a key reversal/support/resistance signal.

## Implementation Status: IN PROGRESS

### Phase 1: Data Structures ✅
- Added `absorptionCanvas` and `absorptionCtx`
- Added `absorptionZones` array to store detected absorption events
- Added `priceZones` Map for tracking active zones
- Configuration:
  - `ZONE_RANGE_BTC = 5` ($5 range)
  - `ZONE_WINDOW_MS = 3000` (3 second window)
  - `ABSORPTION_MIN_VOL = 0.5` (minimum 0.5 BTC)

### Phase 2: Canvas Setup ✅
- Added `#absorptionCanvas` to HTML
- Added CSS z-index layering (z-index: 25, above whales)
- Added to initCharts() call
- Added to resize handler
- Added to render subscription

### Phase 3: Functions Needed (TODO)

1. **initAbsorptionCanvas()**
   - Initialize canvas and context
   - Set initial dimensions

2. **resizeAbsorptionCanvas()**
   - Match chart container size

3. **getZoneKey(price)**
   - Round price to zone bucket
   - Return zone identifier string

4. **processAggressiveTrade(price, size, side)**
   - Get zone from price
   - Update aggBuyVol or aggSellVol
   - Track priceMin/priceMax
   - Call detectAbsorption() after window

5. **detectAbsorption(zoneKey, zone)**
   - Check if lots of aggressive volume
   - Check if price didn't break
   - Check if wall refilled
   - Calculate absorption score
   - Add to absorptionZones if criteria met

6. **renderAbsorption()**
   - Clear canvas
   - For each absorptionZone:
     - Convert price to y-coordinate
     - Draw hollow rectangle (buy = blue, sell = red)
     - Add label with score/volume

7. **cleanupOldZones()**
   - Remove zones older than window
   - Remove old absorption markers

### Phase 4: Integration Points

1. **aggressive_trade socket handler**
   - Call processAggressiveTrade() for each trade
   - This feeds the zone tracking

2. **orderbook socket handler**
   - Track wallBefore/wallAfter for zones
   - Used to detect if wall refilled

3. **Periodic cleanup**
   - setInterval to cleanup old zones every second

## Visualization

**Aggressive buying absorbed (resistance)**:
- Red hollow rectangle
- Positioned at zone price
- Label: "↓ X.XX BTC" (aggressive buy volume absorbed)
- Meaning: Strong selling resistance prevented price from rising

**Aggressive selling absorbed (support)**:
- Blue hollow rectangle
- Positioned at zone price
- Label: "↑ X.XX BTC" (aggressive sell volume absorbed)
- Meaning: Strong buying support prevented price from falling

## Next Steps
1. Add initAbsorptionCanvas() and resizeAbsorptionCanvas()
2. Add zone helper functions
3. Add processAggressiveTrade() logic
4. Add detectAbsorption() scoring logic
5. Add renderAbsorption() visual output
6. Integrate into aggressive_trade handler
7. Test and tune thresholds

## Files Modified
- `/Users/coreycosta/trading-agent/public/btc-dashboard.html`
