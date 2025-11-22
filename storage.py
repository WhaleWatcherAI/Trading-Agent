from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
import json
import os


def save_llm_exchange(
    symbol: str,
    prompt: Dict[str, Any],
    response: Dict[str, Any],
    path: str = "llm_io_log.jsonl",
) -> None:
    """
    Append a single JSONL record containing the prompt and response.

    Only this file is persisted; all higher‑level state (strategy_state,
    historical notes) can be reconstructed by replaying it.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "prompt": prompt,
        "response": response,
    }
    line = json.dumps(record, ensure_ascii=False)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

