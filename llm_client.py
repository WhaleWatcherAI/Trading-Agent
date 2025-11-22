from __future__ import annotations

from typing import Any, Dict
import json

import httpx

from config import Settings


class LLMClient:
    """
    HTTP client for the external LLM trader agent.

    The LLM is expected to:
      - Accept JSON payloads.
      - Return JSON decisions adhering to the agreed contract.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=15.0)

        # System prompt is baked into the client so the agent runs autonomously.
        # It encodes Fabio's playbook and the JSON contract the engine expects.
        self._system_prompt = (
            "You are an intraday futures trading agent that ONLY uses Fabio's playbook "
            "(Trend Model and Mean Reversion Model based on volume profile, POC/value area, "
            "LVNs, CVD, absorption, and exhaustion). You do NOT use ICT/SMC/BOS/FVG or any "
            "other frameworks.\n\n"
            "You receive a JSON object describing current market state, features, positions, "
            "strategy_state, importance_zones, and historical_notes_snippet. You must:\n"
            "- Assess market_state and regime using the supplied derived_state and profiles.\n"
            "- Choose either 'trend', 'mean_reversion', or 'no_trade' in market_assessment.chosen_model.\n"
            "- Emit trade_decisions to enter/manage/exit positions consistent with Fabio's rules:\n"
            "  * Trend Model: only when out_of_balance_up/down, in trend regime, and location vs value/POC\n"
            "    plus orderflow (cvd_trend, absorption/exhaustion, big prints) confirm continuation.\n"
            "  * Mean Reversion Model: only when market is balanced, after failed attempts to leave value "
            "    and reclaim back inside, using LVNs and orderflow failure to fade back to POC.\n"
            "- Use strategy_state as the live configuration for your strategies and update it via "
            "  strategy_updates.strategy_tweaks (enabled flags, max_trades_per_session, min_rr, "
            "  risk_fraction, etc.).\n"
            "- Never widen stops; only move stops closer to entry. Respect the engine's hard risk rails.\n"
            "- Use notes_to_future_self to record very short lessons based on recent behavior. These "
            "  will be fed back to you as historical_notes_snippet on later calls.\n\n"
            "You MUST respond with a single JSON object matching this high-level structure:\n"
            "{\n"
            '  \"market_assessment\": {\n'
            '    \"market_state\": \"balanced\" | \"out_of_balance_up\" | \"out_of_balance_down\" | \"unknown\",\n'
            '    \"regime\": \"trend\" | \"range\" | \"unclear\",\n'
            '    \"chosen_model\": \"trend\" | \"mean_reversion\" | \"no_trade\"\n'
            "  },\n"
            '  \"trade_decisions\": [ ... ],\n'
            '  \"importance_zones\": [ ... ],\n'
            '  \"strategy_updates\": { \"active_strategies\": [...], \"strategy_tweaks\": [ ... ] },\n'
            '  \"calculation_requests\": [ ... ],\n'
            '  \"notes_to_future_self\": [ ... ]\n'
            "}\n"
            "Return ONLY valid JSON with no extra commentary."
        )

    async def request_decision(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON payload to the LLM trader agent and return the parsed JSON response.

        On any error or invalid response, a safe no‑trade response is returned.
        """
        try:
            api_key = self._settings.openai_api_key or self._settings.llm_api_key
            if not api_key:
                raise RuntimeError("No OpenAI API key configured (OPENAI_API_KEY or LLM_API_KEY).")

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
            }

            resp = await self._client.post(
                self._settings.llm_api_url,
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            # OpenAI chat responses: data["choices"][0]["message"]["content"] is a JSON string
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("LLM response content is not a JSON object")
            return parsed
        except Exception as exc:
            # Log and return safe fallback
            print(f"[LLM] Error during decision request: {exc}")
            return {
                "market_assessment": {
                    "market_state": "unknown",
                    "regime": "unclear",
                    "chosen_model": "no_trade",
                },
                "trade_decisions": [],
                "importance_zones": [],
                "strategy_updates": {},
                "calculation_requests": [],
                "notes_to_future_self": [],
            }
