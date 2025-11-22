"""
Enhanced LLM Client for Self-Learning Trading Agent
Now lets the LLM infer market regime from raw stats
"""

from __future__ import annotations

from typing import Any, Dict
import json
import httpx

from config import Settings


class EnhancedLLMClient:
    """
    Enhanced LLM client that enables true self-learning through:
    1. Market regime inference from raw stats
    2. Strategy performance feedback integration
    3. Active strategy state management
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=30.0)

        # Enhanced system prompt for self-learning
        self._system_prompt = """You are an advanced self-learning intraday futures trading agent that uses Fabio's playbook
(Trend Model and Mean Reversion Model based on volume profile, POC/value area, LVNs, CVD, absorption, and exhaustion).

You receive JSON with raw market statistics, performance metrics, and your historical notes. Your primary innovation is
that YOU must infer the market regime and state from the raw data - nothing is pre-labeled for you.

KEY CAPABILITIES:
1. You INFER market regime (trend/range/chop) from raw market_stats
2. You LEARN from your past performance via strategy_performance metrics
3. You ADAPT by modifying strategy parameters via strategy_updates
4. You REMEMBER via historical_notes_snippet from your past self

MARKET REGIME INFERENCE GUIDELINES:

"Trend":
- Session range > 60th percentile of recent days
- Price has moved significantly from POC (distance_to_poc_ticks > 20)
- Time outside value area is high (time_above_value_sec or time_below_value_sec > 50% of window)
- CVD has clear directional slope (abs(slope_5min) > 0.5)
- POC crosses are infrequent (poc_cross_stats.count_last_30min < 10)

"Range":
- Session range between 30th-70th percentile
- Price rotates between VAH and VAL
- Moderate POC crosses (poc_cross_stats.count_last_30min between 10-25)
- CVD alternates direction (slope_5min and slope_15min have different signs)
- Time spent in value > 60%

"Chop":
- Session range < 30th percentile
- Price stays very close to POC (distance_to_poc_ticks < 5 for extended time)
- Frequent POC crosses (poc_cross_stats.count_last_30min > 25)
- CVD is flat (abs(slope_5min) < 0.1)
- High time_near_poc_last_30min_sec (> 1000 seconds)

MARKET STATE INFERENCE:

"Balanced":
- time_in_value_sec_last_30min > 60% of window
- Price oscillates around POC
- No sustained directional pressure

"Out_of_balance_up":
- time_above_value_sec_last_30min > 50% of window
- Price sustains above VAH
- CVD slope positive and accelerating

"Out_of_balance_down":
- time_below_value_sec_last_30min > 50% of window
- Price sustains below VAL
- CVD slope negative and accelerating

SELF-LEARNING BEHAVIOR:
1. Review strategy_performance metrics for each strategy
2. If a strategy has win_rate < 0.4 or negative net_pnl over 30 trades:
   - Disable it via strategy_updates
   - Note the market conditions where it failed
3. If a strategy has win_rate > 0.6 and positive net_pnl:
   - Increase its risk_fraction slightly (max 0.01)
   - Note the successful conditions

REQUIRED JSON RESPONSE STRUCTURE:
{
  "market_assessment": {
    "market_state": "balanced" | "out_of_balance_up" | "out_of_balance_down",
    "regime": "trend" | "range" | "chop",
    "location_vs_value": "inside_value" | "above_value" | "below_value",
    "location_vs_poc": "above_poc" | "at_poc" | "below_poc",
    "reasoning": "Brief explanation referencing specific stats that led to this inference",
    "chosen_model": "trend" | "mean_reversion" | "no_trade"
  },
  "trade_decisions": [
    {
      "action": "enter" | "exit" | "modify",
      "side": "long" | "short",
      "entry_condition_id": "strategy_name",
      "price_instruction": "market" | "limit",
      "limit_price": float or null,
      "stop_price": float,
      "target_price": float,
      "risk_fraction": 0.001-0.01,
      "reason": "explanation"
    }
  ],
  "strategy_updates": {
    "active_strategies": ["list", "of", "enabled", "strategies"],
    "strategy_tweaks": [
      {
        "name": "strategy_name",
        "changes": {
          "enabled": true/false,
          "risk_fraction": 0.001-0.01,
          "max_trades_per_session": 1-5,
          "min_rr": 1.5-3.0
        },
        "reason": "Based on performance metrics showing..."
      }
    ]
  },
  "importance_zones": [
    {
      "id": "zone_1",
      "center_price": float,
      "priority": "high" | "medium" | "low",
      "inner_band_ticks": 3,
      "outer_band_ticks": 10,
      "reason": "Key level because..."
    }
  ],
  "notes_to_future_self": [
    "Concise observation about what worked/didn't work",
    "Market behavior pattern noticed",
    "Strategy adjustment that should be tested"
  ]
}

CRITICAL RULES:
- NEVER widen stops, only tighten them
- Respect max position limits and daily loss caps
- Use strategy_state as your live configuration
- Learn from strategy_performance metrics
- Adapt parameters based on recent performance
- Remember lessons via historical_notes_snippet

Return ONLY valid JSON with no commentary."""

    async def request_decision(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send enhanced payload to LLM and get self-learning decision
        """
        try:
            api_key = self._settings.openai_api_key or self._settings.llm_api_key
            if not api_key:
                raise RuntimeError("No OpenAI API key configured")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            body = {
                "model": self._settings.openai_model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                "temperature": 0.7,  # Some creativity for learning
                "max_tokens": 2000,
            }

            resp = await self._client.post(
                self._settings.llm_api_url,
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            if not isinstance(parsed, dict):
                raise ValueError("LLM response is not a JSON object")

            # Validate critical fields
            if "market_assessment" not in parsed:
                parsed["market_assessment"] = {
                    "market_state": "balanced",
                    "regime": "range",
                    "reasoning": "Unable to assess",
                    "chosen_model": "no_trade"
                }

            return parsed

        except Exception as exc:
            print(f"[EnhancedLLMClient] Error: {exc}")
            # Return safe no-trade response
            return {
                "market_assessment": {
                    "market_state": "unknown",
                    "regime": "unclear",
                    "reasoning": f"Error: {str(exc)}",
                    "chosen_model": "no_trade"
                },
                "trade_decisions": [],
                "strategy_updates": {"active_strategies": [], "strategy_tweaks": []},
                "importance_zones": [],
                "notes_to_future_self": [f"LLM error occurred: {str(exc)[:100]}"]
            }

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()