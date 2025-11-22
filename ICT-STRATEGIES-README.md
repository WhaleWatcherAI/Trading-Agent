# ICT Trading Strategies - 1-Minute Backtests

This document describes three ICT (Inner Circle Trader) style strategies implemented as 1-minute chart backtests using the TopstepX futures data framework.

## Overview

All three strategies:
- Run on **1-minute bars** (configurable via environment variables)
- Use **Fair Value Gaps (FVG)** as core entry mechanisms
- Include realistic **slippage and commission models** from `slip-config.json`
- Implement **session filters** (primarily NY session: 09:30-11:30 and 13:30-15:30 ET)
- Support **scaling out** at TP1 and moving stops to breakeven
- Track detailed performance metrics (PnL, win rate, profit factor, drawdown)

---

## Strategy #1: Liquidity-Sweep + FVG Return

**File:** `backtest-ict-liquidity-sweep-fvg.ts`

### Core Concept (ICT 2022)
Wait for price to raid obvious resting liquidity (prior day's H/L), then fade back into the range at the FVG 50% midpoint.

### Entry Logic
1. **Detect Liquidity Sweep:**
   - **Sell-side sweep (for long):** Today's low breaks yesterday's low, then closes back above within N bars (≤5)
   - **Buy-side sweep (for short):** Today's high breaks yesterday's high, then closes back below within N bars

2. **Detect FVG (3-bar pattern):**
   - **Bullish FVG:** `low[t] > high[t-2]`
   - **Bearish FVG:** `high[t] < low[t-2]`
   - **Midpoint:** Average of the gap (50% fill level)

3. **Entry:**
   - **Long:** After sell-side sweep → first bullish FVG → limit entry at 50% midpoint
   - **Short:** After buy-side sweep → first bearish FVG → limit entry at 50% midpoint

### Risk/Targets
- **Stop Loss:** Just beyond sweep extreme (default: 2 ticks outside swept low/high)
- **TP1:** 1R (scale out 50% of position)
- **TP2:** 2R (exit remaining position)
- **Stop Management:** Move to breakeven after TP1 hit

### Configuration (Environment Variables)

```bash
# Symbol & Contract
ICT_SWEEP_SYMBOL=NQZ5                    # Default: NQZ5
ICT_SWEEP_CONTRACT_ID=CON.F.US.ENQ.Z25   # Optional, auto-resolved from symbol

# Date Range
ICT_SWEEP_START=2025-01-01T00:00:00Z     # Default: 90 days ago
ICT_SWEEP_END=2025-11-11T00:00:00Z       # Default: now

# Strategy Parameters
ICT_SWEEP_RECLAIM_BARS=5                 # Max bars to reclaim after sweep (default: 5)
ICT_SWEEP_SL_BUFFER=2                    # Stop loss buffer in ticks (default: 2)
ICT_SWEEP_TP1=1                          # TP1 R-multiple (default: 1)
ICT_SWEEP_TP2=2                          # TP2 R-multiple (default: 2)
ICT_SWEEP_CONTRACTS=2                    # Number of contracts (default: 2)
ICT_SWEEP_SCALE_PERCENT=0.5              # Scale out % at TP1 (default: 0.5 = 50%)
ICT_SWEEP_COMMISSION=1.40                # Commission per side USD (auto-inferred if not set)
```

### Example Usage

```bash
# Run with defaults (NQZ5, last 90 days)
npx tsx backtest-ict-liquidity-sweep-fvg.ts

# Custom date range
ICT_SWEEP_START="2025-11-01T00:00:00Z" ICT_SWEEP_END="2025-11-11T00:00:00Z" \
npx tsx backtest-ict-liquidity-sweep-fvg.ts

# Different symbol (ES futures)
ICT_SWEEP_SYMBOL=ESZ5 ICT_SWEEP_CONTRACTS=3 \
npx tsx backtest-ict-liquidity-sweep-fvg.ts
```

---

## Strategy #2: BOS/CHOCH + FVG (Trend-Following)

**File:** `backtest-ict-bos-choch-fvg.ts`

### Core Concept
Confirm Break of Structure (BOS) or Change of Character (CHOCH) using swing high/low breaks, then take the next FVG in that direction at 50% fill. More momentum-friendly than Strategy #1.

### Entry Logic
1. **Detect BOS (Break of Structure):**
   - Use pivot detection (default: 3-bar pivots) to identify swing points
   - **Bullish BOS:** Current high breaks recent swing high
   - **Bearish BOS:** Current low breaks recent swing low

2. **Detect FVG after BOS:**
   - After bullish BOS → wait for bullish FVG (within N bars, default: 10)
   - After bearish BOS → wait for bearish FVG

3. **Entry:**
   - Enter at 50% FVG midpoint
   - Direction: same as BOS direction

### Risk/Targets
- **Stop Loss:** Just beyond the swing low/high that triggered BOS (+ buffer ticks)
- **TP1:** 1R (scale out 50%)
- **TP2:** 2R (exit remaining)
- **Stop Management:** Move to breakeven after TP1

### Configuration (Environment Variables)

```bash
# Symbol & Contract
ICT_BOS_SYMBOL=NQZ5
ICT_BOS_CONTRACT_ID=CON.F.US.ENQ.Z25

# Date Range
ICT_BOS_START=2025-01-01T00:00:00Z
ICT_BOS_END=2025-11-11T00:00:00Z

# Strategy Parameters
ICT_BOS_PIVOT_LEN=3                      # Pivot length for swing detection (default: 3)
ICT_BOS_FVG_LOOKBACK=10                  # Max bars to find FVG after BOS (default: 10)
ICT_BOS_SL_BUFFER=2                      # Stop loss buffer in ticks (default: 2)
ICT_BOS_TP1=1                            # TP1 R-multiple (default: 1)
ICT_BOS_TP2=2                            # TP2 R-multiple (default: 2)
ICT_BOS_CONTRACTS=2                      # Number of contracts (default: 2)
ICT_BOS_SCALE_PERCENT=0.5                # Scale out % at TP1 (default: 0.5)
ICT_BOS_COMMISSION=1.40
```

### Example Usage

```bash
# Run with defaults
npx tsx backtest-ict-bos-choch-fvg.ts

# Longer pivot length (more significant swings)
ICT_BOS_PIVOT_LEN=5 npx tsx backtest-ict-bos-choch-fvg.ts

# Different symbol (Gold futures)
ICT_BOS_SYMBOL=GCZ5 ICT_BOS_CONTRACTS=1 \
npx tsx backtest-ict-bos-choch-fvg.ts
```

---

## Strategy #3: Power of Three (PO3) Lite

**File:** `backtest-ict-po3-lite.ts`

### Core Concept
ICT's session-based model tracking Accumulation → Manipulation → Distribution across three sessions:
1. **Asia (20:00-00:00 ET):** Accumulation - establish range
2. **London (02:00-05:00 ET):** Manipulation - sweep one side of Asia range
3. **NY (09:30-11:30, 13:30-15:30 ET):** Distribution - trade back into/through range

### Entry Logic
1. **Track Asia Range:**
   - Record high/low during Asia session (20:00-00:00 ET)

2. **Detect London Manipulation:**
   - **Bullish manipulation:** London sweeps Asia low (breaks below)
   - **Bearish manipulation:** London sweeps Asia high (breaks above)

3. **NY Distribution Entry:**
   - After **bullish** London manipulation → look for **bullish FVG** in NY
   - After **bearish** London manipulation → look for **bearish FVG** in NY
   - Enter at FVG 50% midpoint

### Risk/Targets
- **Stop Loss:** Just beyond London sweep extreme (+ buffer ticks)
- **TP1:** Asia range midpoint (scale out 50%)
- **TP2:** Opposite side of Asia range (full target)
- **Stop Management:** Move to breakeven after TP1

### Configuration (Environment Variables)

```bash
# Symbol & Contract
ICT_PO3_SYMBOL=NQZ5
ICT_PO3_CONTRACT_ID=CON.F.US.ENQ.Z25

# Date Range
ICT_PO3_START=2025-01-01T00:00:00Z
ICT_PO3_END=2025-11-11T00:00:00Z

# Strategy Parameters
ICT_PO3_SL_BUFFER=2                      # Stop loss buffer in ticks (default: 2)
ICT_PO3_CONTRACTS=2                      # Number of contracts (default: 2)
ICT_PO3_SCALE_PERCENT=0.5                # Scale out % at TP1 (default: 0.5)
ICT_PO3_COMMISSION=1.40
```

### Example Usage

```bash
# Run with defaults
npx tsx backtest-ict-po3-lite.ts

# Different symbol (Euro futures)
ICT_PO3_SYMBOL=6EZ5 ICT_PO3_CONTRACTS=2 \
npx tsx backtest-ict-po3-lite.ts

# Custom date range
ICT_PO3_START="2025-10-01T00:00:00Z" ICT_PO3_END="2025-11-11T00:00:00Z" \
npx tsx backtest-ict-po3-lite.ts
```

---

## Common Features

### Fair Value Gap (FVG) Detection
All strategies use the same 3-bar FVG detection:
- **Bullish FVG:** `low[t] > high[t-2]` (gap down, creates buy zone)
- **Bearish FVG:** `high[t] < low[t-2]` (gap up, creates sell zone)
- **Entry:** 50% midpoint of the gap

### Slippage & Commission Model
Realistic fills from `slip-config.json`:
- **Entry (aggressive):** `mid ± (0.5×spread + σ_entry)`
- **TP (passive/agg mix):** `mid ∓ E_tp_ticks` where `E_tp_ticks = (1-p)×(spread + σ_tp)`
- **Stop (adverse):** `trigger ∓ σ_stop`
- **Fees:** Applied per side from config (e.g., NQ = $1.40/side, ES = $1.40/side)

### Performance Metrics
Each backtest reports:
- Total trades, wins, losses, win rate
- Net PnL (after fees/slippage), gross profit, gross loss
- Average win, average loss, profit factor
- Maximum drawdown
- Exit reason breakdown (tp1, tp2, stop, session, end_of_data)
- Recent trades with detailed entry/exit info

---

## Supported Symbols

All strategies work with any TopstepX futures contract. Common examples:

| Symbol | Description | Default Commission/Side |
|--------|-------------|------------------------|
| **NQZ5** | Nasdaq-100 E-mini Dec 2025 | $1.40 |
| **MNQ** | Micro Nasdaq-100 | $0.37 |
| **ESZ5** | S&P 500 E-mini Dec 2025 | $1.40 |
| **MES** | Micro S&P 500 | $0.37 |
| **GCZ5** | Gold Dec 2025 | $2.40 |
| **MGC** | Micro Gold | $0.86 |
| **6EZ5** | Euro FX Dec 2025 | $1.62 |
| **M6E** | Micro Euro FX | $0.35 |

*(Commissions auto-inferred from `slip-config.json` if not specified)*

---

## Tips for Optimization

1. **Start with short date ranges** (7-30 days) to iterate quickly
2. **Adjust pivot lengths** (Strategy #2) for different market structures
3. **Tune reclaim bars** (Strategy #1) based on symbol volatility
4. **Test different R-multiples** for TP1/TP2 based on risk tolerance
5. **Compare all three** on the same dataset to see which fits current market regime
6. **Walk-forward test** across multiple months to avoid curve-fitting

---

## Example: Multi-Symbol Backtest

Run all three strategies on NQ, ES, and GC:

```bash
# Strategy #1: Liquidity-Sweep + FVG
for symbol in NQZ5 ESZ5 GCZ5; do
  echo "Running Liquidity-Sweep on $symbol..."
  ICT_SWEEP_SYMBOL=$symbol npx tsx backtest-ict-liquidity-sweep-fvg.ts
done

# Strategy #2: BOS/CHOCH + FVG
for symbol in NQZ5 ESZ5 GCZ5; do
  echo "Running BOS/CHOCH on $symbol..."
  ICT_BOS_SYMBOL=$symbol npx tsx backtest-ict-bos-choch-fvg.ts
done

# Strategy #3: PO3 Lite
for symbol in NQZ5 ESZ5 GCZ5; do
  echo "Running PO3 on $symbol..."
  ICT_PO3_SYMBOL=$symbol npx tsx backtest-ict-po3-lite.ts
done
```

---

## Next Steps

1. **Backtest on historical data** (3-12 months)
2. **Forward test** on recent/live data
3. **Combine with filters** (ATR, volume, session bias)
4. **Paper trade** before going live
5. **Implement live versions** using your existing `live-topstepx-*.ts` framework

---

## Quick Start Summary

```bash
# Test Strategy #1 on NQ (last 7 days)
ICT_SWEEP_START="2025-11-04T00:00:00Z" ICT_SWEEP_END="2025-11-11T00:00:00Z" \
npx tsx backtest-ict-liquidity-sweep-fvg.ts

# Test Strategy #2 on ES (last 30 days)
ICT_BOS_SYMBOL=ESZ5 ICT_BOS_START="2025-10-12T00:00:00Z" \
npx tsx backtest-ict-bos-choch-fvg.ts

# Test Strategy #3 on Gold (last 90 days)
ICT_PO3_SYMBOL=GCZ5 npx tsx backtest-ict-po3-lite.ts
```

---

**Happy trading! Remember: these are educational tools. Always validate with paper trading before risking real capital.**
