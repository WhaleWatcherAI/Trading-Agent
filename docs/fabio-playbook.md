# Fabio Playbook – Agent Specification

This document encodes the core strategy and philosophy for a future AI trading agent. It mirrors the provided specification and is intended as canonical reference for training and implementation.

---

## 1. High-level model for the agent

**Core idea:**  
The market is an auction oscillating between:

- **Balance:** price rotating around fair value (value area around POC).
- **Imbalance:** one side (buyers or sellers) is clearly aggressive, pushing away from prior value to find a new fair value.

The model has three decision layers:

1. **Market State** – Is the market balanced or out of balance?
2. **Location** – Where is price relative to important Volume Profile levels (POC, value area, LVNs)?
3. **Aggression** – Is there clear order-flow confirmation (big prints, footprints, CVD)?

If all 3 align → trade; if any is missing → no trade.

**Two setups:**

- **Setup 1 – Trend Model:** Out-of-balance → continuation toward new balance.
- **Setup 2 – Mean Reversion Model:** Failed move out-of-balance → snap back to balance (POC).

---

## 2. Definitions the agent must understand

### 2.1 Balance vs Out-of-Balance

**Balanced market:**

- Price rotates around a prior value area/POC.
- The Volume Profile for the session or previous day shows a bell-shaped distribution.
- Trades mostly occur inside the value area (≈ 70% of the time).
- Breakouts often fail.

**Out-of-balance market:**

- Price cleanly displaces away from prior value area.
- Candles have direction and momentum (impulse legs).
- Volume Profile is skewed; price is trading outside prior value area for a sustained period.
- The auction is searching for a new area of fair value.

### 2.2 Volume Profile structures

**POC (Point of Control):**

- Price level with highest traded volume in the chosen profile.
- Represents fair value or center of balance.

**Value Area:**

- Region containing ≈ 70% of traded volume around POC.
- Edges often called VAH (Value Area High) and VAL (Value Area Low).

**LVN (Low Volume Node):**

- Price level or small zone with noticeably lower volume relative to surrounding prices on the profile.
- Represents an area where price moved through quickly, often a future reaction zone.
- In this playbook, LVNs inside an impulse leg or reclaim leg are primary entry locations.

### 2.3 Order Flow / Aggression

The agent should interpret **aggression** as market orders hitting the book with clear directional bias, visible via:

- **Big prints / bubbles:** unusually large trades at or near a key level.
- **Footprint imbalance:** significantly more aggressive buys vs sells, or vice versa, at price levels (e.g., large delta at a price in a footprint chart).

**CVD (Cumulative Volume Delta):**

- Upwards sloping → net aggressive buying.
- Downwards sloping → net aggressive selling.
- Strong, sustained CVD in trade direction = confirmation.

### 2.4 Instruments, style, sessions

- **Instruments:** Futures (e.g., NASDAQ, ES, etc.).
- **Trading style:** Intraday scalping.

**Preferred sessions:**

- **Trend Model:** New York session, especially NASDAQ/ES.
- **Mean Reversion Model:** London session or compressed summer conditions in US session.

---

## 3. Global risk & trade management rules

These apply to both setups:

**Risk per trade:**

- 0.25% to 0.5% of account balance.

**Stop placement:**

- Always just beyond the aggressive print or zone that triggered entry.
- Add 1–2 ticks buffer beyond obvious swing high/low to avoid random slippage.

**Break-even rule (Trend Model emphasis):**

- If CVD shows strong pressure in trade direction, move stop to break-even earlier than usual.

**Invalidation logic:**

- If you are wrong, you should be wrong immediately.
- Never widen the stop once in the trade.

**Targets (default):**

- Primary target = balance POC (from the relevant profile).
- Do not stretch to the other side of the range by default.
- On rare, strong trend days you may trail beyond POC, but this is the exception.

---

## 4. Setup 1 – Trend Model (Out-of-Balance → New Balance)

### 4.1 When the agent is allowed to look for this setup

**Conditions:**

- Market is out of balance (away from previous value area/POC).
- Price shows displacement and momentum away from prior value.
- Session = New York preferred (avoid London open).
- Not in clear chop or tight range; structure should show a strong directional move.

If any of the above fails → no Trend Model trades.

### 4.2 Step 1 – Identify the impulse leg

Find the impulse leg that:

- Broke structure (e.g., broke out of prior balance or cleared support/resistance).
- Is clearly directional (up for longs, down for shorts).

Mark:

- Start of the impulse.
- End of the impulse (current high for up move, current low for down move).

### 4.3 Step 2 – Location via Volume Profile on the impulse leg

- Apply a Volume Profile only to this impulse leg.
- Identify LVNs inside that move.
- Mark those LVNs as potential reaction zones.
- Do not place blind limit orders at LVNs.
- Instead, set alerts slightly before price reaches each LVN to prepare for order flow evaluation.

### 4.4 Step 3 – Execution trigger (order flow)

When price retests an LVN in the direction of the main trend:

**For longs in an uptrend:**

- Price moves down into an LVN formed during the prior up impulse.
- At/near the LVN, look for buy aggression:
  - Big buy prints or
  - Buy footprint imbalances (bid-ask delta strongly favoring buys at that level).

**For shorts in a downtrend:**

- Price moves up into an LVN formed during prior down impulse.
- At/near the LVN, look for sell aggression:
  - Big sell prints or
  - Sell footprint imbalances.

**Entry rule:**

- Enter only after aggression appears in the direction of the trend at or very close to the LVN.
- No aggression = no trade, even if LVN is hit.

### 4.5 Step 4 – Risk management (Trend Model specific)

**Stop loss:**

- Place just beyond the aggressive footprint or prints that justified the entry.
- Add 1–2 ticks buffer before the obvious swing high/low (to avoid random wicks taking you out).

**Break-even rule:**

- If CVD turns strongly in favor of your trade direction and confirms flowing order flow, move stop to break-even earlier.
- The goal is to protect capital while giving room for continuation.

### 4.6 Step 5 – Targeting

- Main target = previous balance POC (the POC of the balance the market is moving toward).
- Exit full position at that POC under normal conditions.

**Optional advanced behavior:**

- On rare, strong trend days, allow a trailing component beyond POC based on continued displacement and order flow.
- Base rule: close at POC, because ≈70% of the time price reverses around balance.

---

## 5. Setup 2 – Mean Reversion Model (Failed Breakout → Back Into Balance)

### 5.1 When the agent is allowed to look for this setup

**Conditions:**

- Market is in balance or consolidation.
- A clear balance area is defined, often using the previous day’s profile as the reference.
- Price attempts to break out of balance (above VAH or below VAL).
- The breakout fails (returns back inside balance).

**Best environments:**

- London session, or
- Compressed summer conditions with frequent failed breaks and rotations.

### 5.2 Step 1 – Market State

Use previous day’s profile or current session’s early structure to define:

- Balance POC.
- Value Area High (VAH) and Low (VAL).

Confirm that price has spent time rotating around this POC – evidence of balance.

Wait for price to push out of balance (move beyond VAH or VAL).

### 5.3 Step 2 – Detect failed breakout and reclaim

Agent logic:

1. Detect first breakout attempt beyond VAH/VAL:
   - Price moves outside value area but fails to follow through (no sustained displacement).
2. If price re-enters the value area, mark this as potential failure.
3. Watch for a second attempt that also fails to sustain outside balance:
   - Price gets rejected again and comes back inside value.

When the second failure confirms that the market cannot sustain outside balance, the mean reversion model is active.

Then:

- Identify the reclaim leg:
  - From the point price re-enters balance back toward POC.
  - Apply Volume Profile on this reclaim leg and mark LVNs within it.

### 5.4 Step 3 – Execution trigger inside reclaimed balance

On the pullback into an LVN of the reclaim leg:

**For a short mean reversion (failed upside breakout):**

- Price initially broke above VAH, failed, then re-entered balance.
- Price pulls back upward into an LVN formed during the reclaim down leg.
- At that LVN, check order flow:
  - Buyers are still trying to hit, but they fail to push price higher.
  - Aggression on the buy side dies out or gets absorbed.
- Enter short when price is firmly back inside balance and order flow confirms lack of follow-through.

**For a long mean reversion (failed downside breakout):**

- Same logic mirrored: failed break below VAL, re-entry into balance, reclaim leg up, then pullback into LVN with sell aggression failing.

**Key:**  
Entry is taken after reclaim and during the pullback into the reclaim LVN with order-flow confirming failure to continue the breakout.

### 5.5 Step 4 – Risk management (Mean Reversion specific)

**Stop loss:**

- Place just beyond the aggressive level or print that formed your entry logic.
- Add 1–2 ticks buffer beyond the failed high/low.

**Invalidation rule:**

- If price quickly moves back outside balance and holds beyond your LVN or failed high–low, you are wrong.
- Do not widen stops.

### 5.6 Step 5 – Targeting

- Target = POC inside the balance area.
- Exit full position at the POC.
- Do not automatically aim for the opposite side of the range; that is considered stretching unless conditions are exceptional.

---

## 6. Pros and Cons – Meta-rules for the agent

### 6.1 Pros (Why this model is used)

- **Clear structure:** Decisions always go:
  - Market state (balanced / out-of-balance),
  - Location (LVN / POC / value edge),
  - Aggression (order flow).
- **Adaptability:**
  - Trend Model for trending, out-of-balance days.
  - Mean Reversion Model for balanced, range-bound days.
- **Tight risk:**
  - Small stops just beyond aggression zones.
- **High sample size:**
  - Frequent opportunities across sessions allow statistical edge to express.
- **Built-in discipline:**
  - "No alignment, no trade" reduces overtrading.
- **Scalable R:R:**
  - Normal trades can yield R:R from ~1:2.5 to ~1:5 or more; rare trend days can go higher.

### 6.2 Cons (Behavior the agent should expect)

- **Low streaky win rate possible:**
  - Multiple small stopouts in choppy conditions are normal.
- **High attention requirement:**
  - Requires continuous monitoring of order flow and volume profile during sessions.
- **Psychological stress with size:**
  - Larger sizes still follow same rules, but emotional load increases.

---

## 7. Encoded example trades

These examples illustrate how to store concrete trades for training.

### Example 1 – Trend Continuation Long (New York, Out-of-Balance Uptrend)

```json
{
  "example_id": "trend_model_long_01",
  "model": "trend_continuation",
  "session": "NewYork",
  "market_state": "out_of_balance_uptrend",
  "context": {
    "description": "Price broke out of prior balance and is strongly trending higher.",
    "prior_balance_poc": "below current price",
    "impulse_leg": {
      "direction": "up",
      "start": "breakout point from balance",
      "end": "current swing high before pullback"
    },
    "volume_profile": {
      "applied_to": "impulse_leg",
      "lvns": ["lvn_1", "lvn_2"]
    }
  },
  "entry_logic": {
    "retest_location": "lvn_1",
    "order_flow": {
      "at_lvn": {
        "aggression": "strong_buy",
        "footprint_signal": "buy_imbalance",
        "big_trades": true
      }
    },
    "side": "long",
    "entry_reason": "Price pulled back into LVN within upward impulse leg; big buy aggression confirmed continuation."
  },
  "risk_management": {
    "stop_type": "hard_stop",
    "stop_location": "a few ticks below aggressive buy prints and local swing low",
    "risk_per_trade_pct": [0.25, 0.5],
    "breakeven_rule": "If CVD continues strongly upwards after entry, move stop to breakeven early."
  },
  "exit_logic": {
    "planned_target": "next_balance_poc",
    "actual_exit_reason": "sellers showed notable aggression against the move; exit to protect profits",
    "note": "Illustrates optional discretionary exit before POC when opposing aggression appears."
  }
}
```

### Example 2 – Mean Reversion Short (London, Failed Upside Breakout)

```json
{
  "example_id": "mean_reversion_short_01",
  "model": "mean_reversion",
  "session": "London",
  "market_state": "balanced_with_failed_breakout_above",
  "context": {
    "description": "Market rotating around prior balance POC; two failed attempts to break above value.",
    "balance_reference": "previous_day_profile",
    "value_area": {
      "vah": "above_poc",
      "val": "below_poc",
      "poc": "center_of_range"
    },
    "breakouts": [
      {
        "direction": "up",
        "attempt": 1,
        "result": "failed",
        "comment": "price could not sustain outside value"
      },
      {
        "direction": "up",
        "attempt": 2,
        "result": "failed",
        "comment": "price rejected again and re-entered balance"
      }
    ],
    "reclaim_leg": {
      "direction": "down",
      "profile": {
        "lvns": ["reclaim_lvn_1"]
      }
    }
  },
  "entry_logic": {
    "retest_location": "reclaim_lvn_1",
    "order_flow": {
      "at_lvn": {
        "buyer_aggression": "initially_present_but_fades",
        "follow_through": "absent",
        "interpretation": "buyers cannot push through; failed breakout confirmed"
      }
    },
    "side": "short",
    "entry_reason": "After second failed breakout above balance, price re-entered value; short taken on pullback into LVN of reclaim leg with buyers failing."
  },
  "risk_management": {
    "stop_type": "hard_stop",
    "stop_location": "just above failed high, with 1–2 tick buffer",
    "risk_per_trade_pct": [0.25, 0.5],
    "invalidation": "price holds above failed high and outside balance"
  },
  "exit_logic": {
    "target": "balance_poc",
    "exit_type": "full",
    "description": "Price rotated back into value and hit POC cleanly. No attempt to stretch to opposite end of range."
  }
}
```

---

## 8. Compact rule summary for implementation

Mini checklist the agent should follow:

1. **Determine market state:**
   - If price oscillates around POC → balanced.
   - If price strongly displaces away from prior value → out_of_balance.

2. **If out_of_balance → Trend Model:**
   - Identify latest impulse that broke structure.
   - Build Volume Profile on that impulse.
   - Mark LVNs and wait for retest.
   - On LVN retest: require matching directional aggression (footprint, big prints, CVD).
   - Enter with stop just beyond aggression and 1–2 ticks buffer.
   - Target: next balance POC; consider breakeven/trail with strong CVD.

3. **If in balance with failed breakout → Mean Reversion Model:**
   - Define balance via prior day or current profile (POC, VAH, VAL).
   - Detect one or more failed attempts to trade outside VAH/VAL.
   - After reclaim back inside value, define reclaim leg and LVNs.
   - On pullback into LVN, confirm failed aggression (no follow-through).
   - Enter back toward POC with stop beyond failed high/low.
   - Target: POC of balance.

4. **Global:**
   - Risk per trade: 0.25–0.5% of account.
   - Never widen stops.
   - No trade if market state, location, and aggression are not all aligned.

