# Volatility Regime Strategy Build Progress

This document tracks how the current implementation lines up with the full **Volatility Regime-Based Options Strategy Agent (CLI Design)** specification.

## Stage 1 – Universe Filtering

| Requirement | Status | Notes |
| --- | --- | --- |
| Liquidity tiers (Large Cap vs Mid/Small) based on market cap, average shares, options volume | **Partial** | Tiering logic is present, but relies on limited quote metadata and approximated options volume from the Tradier chain snapshot. Needs robust fundamentals data (market cap, float, average shares) and aggregated options flow. |
| Filter IV Rank (IVR) thresholds (<=0.35 large, <=0.25 small) | **Partial** | Uses UnusualWhales `getVolatilityStats` IV rank when available; gaps when ticker not in cached list. No fallback to alternative data provider. |
| Volume-to-OI Ratio filter (>=1.5 large, >=2.0 small) | **Partial** | Ratio computed from Tradier chain volume/OI per run; requires intraday aggregate to avoid single snapshot noise. |
| IVΔ (15–30m) > 0 | **Partial** | Rolling 2h IV history maintains 15m/30m deltas when UW volatility stats refresh; still requires guaranteed periodic snapshots and secondary provider. |
| Whale trades thresholds (contracts/premium by tier) | **Partial** | Pulls Unusual Whales flow alerts with tier-aware thresholds and 30m lookback; needs richer direction metadata and redundancy if API unavailable. |
| Watchlist segmentation & reporting | **Partial** | UI shows pass/fail per ticker with key metrics, but no separation by tier or rejection list with reasons. |

## Stage 2 – Regime Detection

| Requirement | Status | Notes |
| --- | --- | --- |
| Net GEX calculation by summing gamma exposure across expiries (short vs long) | **Partial** | Summation implemented per strike using Tradier greeks. Needs validation of sign conventions (puts vs calls) and dealer positioning assumptions. |
| Mode-based expiry windows (Scalp 3–7 DTE, Swing 10–20, LEAPS far-dated) | **Implemented** | Mode toggles map to GEX calculator. LEAPS mode lacks bespoke analytics beyond expiry range. |
| Regime labeling: Expansion (net <= 0) vs Pinning (>0) | **Implemented** | Labeling and slope caching exist. Requires verification with real data to ensure sign handling. |
| Gamma flip monitoring and slope tracking | **Partial** | Flip level estimated via adjacent strike interpolation; slope now uses 15/30m history with narratives but still lacks cross-session persistence and adaptive smoothing. |
| Display of dominant expiries and gamma flip context | **Partial** | Dashboard now highlights top expiration contributors and flip distance, still needs narrative guidance (e.g., “Gamma flip expected at…” commentary). |

## Stage 3 – Per-Strike GEX Wall Logic

| Requirement | Status | Notes |
| --- | --- | --- |
| GEX histogram highlighting call walls / put zones | **Partial** | Chart overlays and panel expose wall strength, distance, z-scores, and latest HVN/POC confluence; still needs intraday structural alerts and streaming updates. |
| Overlay price interaction and inflection logic | **Partial** | Basic classification (above/below) but no integration with live price feed movements. |
| Trade entry validation aligning regime, IVΔ, price vs wall, whale confirmation | **Stub** | Signals generated if Stage 1 passes and simple heuristics align. Lacks verification of IVΔ or whale flow and doesn’t enforce timing window. |

## Position Management Rules

| Requirement | Status | Notes |
| --- | --- | --- |
| Position sizing by IVR (full vs half) | **Partial** | Applied in signal creation but dependent on incomplete IVR data. |
| Confidence adjustments via volume profile confluence | **Missing** | Volume profile/HVN/LVN data not fetched; logic absent. |
| LEAPS hedge rule for tail protection | **Missing** | No logic to add OTM hedge contracts. |
| Stop, exit timing, add-on, and scale-out logic | **Partial** | Lifecycle engine now tracks trigger->entry transitions, +1R scale-outs, add-on attempts, time stops, and exits; still heuristic without live price feed or volume-profile confluence. |

## Display & Output

| Requirement | Status | Notes |
| --- | --- | --- |
| Universe scan panel with metrics and reasons | **Partial** | Implemented high-level metrics; needs tier grouping and rejection list w/ reasons. |
| Regime detection panel with color-coded labels | **Partial** | UI shows values but lacks color-coded badges and narrative overlays. |
| Per-strike GEX chart (ASCII/visual) | **Implemented** | Basic textual chart provided; could use richer visualization or export. |
| Trade signals with rationale (entry, stop, targets) | **Partial** | Generated but depend on placeholder inputs; missing live timestamps and validity windows. |
| Output modes (JSON, markdown report, CLI logs) | **Partial** | JSON toggle available client-side; no markdown report generator or CLI log streaming. |
| Chart overlays (TradingView/matplotlib exports) | **Missing** | Not implemented. Needs data shaping for external overlay tools. |

## Data Sources & Integration

| Requirement | Status | Notes |
| --- | --- | --- |
| Options IV data (real IV rank, IVΔ) | **Partial** | Pulls UW volatility stats and derives 15m/30m deltas via cached history; must ensure consistent cadence and add fallback. |
| Options flow / whale data stream | **Partial** | Integrated UW flow alerts; needs enhanced filtering (bid/ask, sweeps) and alternate feed for resiliency. |
| Gamma calculations validated with external service | **Partial** | In-house calculations implemented; no cross-check with SpotGamma/MenthorQ. |
| Volume profile data for confluence | **Missing** | No volume profile API integration. |
| Real-time price quotes for triggers | **Partial** | Uses Tradier last price from snapshot; no real-time updates or streaming. |
| Modular CLI flags (--mode, --universe, --refresh, etc.) | **Partial** | API supports symbols/mode; no CLI script or refresh interval. |

## Remaining Major Tasks

1. **Data Fidelity**
   - Harden IVΔ sampling cadence with scheduled refresh and fallback provider.
   - Enrich whale flow context (bid/ask aggression, sweep detection) and add redundant feed.
   - Ingest volume profile/HVN-LVN data.

2. **Analytics Refinement**
   - Validate gamma sign conventions and include delta-hedging assumptions.
   - Persist slope/transition state across sessions and calibrate narratives.
   - Enhance wall alerting with structural change detection (e.g., z-score thresholds over time).

3. **Trade Lifecycle Automation**
   - Upgrade lifecycle loop with streaming price/volume data (current client refresh is 60s poll) and broker integration.
   - Add LEAPS hedging logic and volume-profile-based confidence adjustments.
   - Persist trade logs externally (JSON/markdown export or database) for auditability.

4. **Dash & UX Enhancements**
   - Add tier-based grouping, pass/fail breakdown, color-coded regime labels.
   - Provide deeper rationale text and visual overlays.
   - Surface CLI/automation entry points (e.g., Node CLI with refresh cadence).

5. **Testing & Verification**
   - Establish integration tests against mocked Tradier/UW responses.
   - Create validation scripts comparing calculated GEX with known benchmarks.

This status will be updated as each requirement is completed. Contributions should reference the missing items above to ensure the final agent matches the original specification.
