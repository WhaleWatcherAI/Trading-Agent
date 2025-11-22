from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import os


TradingMode = Literal["live_trading", "paper_trading"]


@dataclass
class Settings:
    """
    Global configuration for the LLM‑driven TopstepX trading agent.

    Values are primarily loaded from environment variables with safe defaults
    so the app can be run in paper_trading mode without credentials.
    """

    # Connection
    topstepx_api_key: str
    topstepx_username: str
    topstepx_account_id: str
    topstepx_market_hub_url: str
    topstepx_user_hub_url: str | None
    topstepx_rest_base_url: str

    # LLM (generic + OpenAI)
    llm_api_url: str
    llm_api_key: str
    openai_api_key: str
    openai_model: str

    # Trading
    mode: TradingMode
    symbol: str
    account_balance: float
    risk_per_trade_fraction: float
    min_risk_fraction: float
    max_risk_fraction: float
    max_daily_loss_fraction: float

    # LLM scheduling
    llm_decision_interval_default_sec: int
    llm_decision_interval_outer_band_sec: int
    llm_decision_interval_inner_band_sec: int

    llm_log_path: str = "llm_io_log.jsonl"

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Construct Settings from environment variables with pragmatic defaults.
        """
        def getenv(name: str, default: str) -> str:
            return os.getenv(name, default)

        def getenv_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                return default

        def getenv_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        mode_env = getenv("TRADING_MODE", "paper_trading")
        mode: TradingMode = "live_trading" if mode_env == "live_trading" else "paper_trading"

        return cls(
            # Connection
            topstepx_api_key=getenv("TOPSTEPX_API_KEY", ""),
            topstepx_username=getenv("TOPSTEPX_USERNAME", ""),
            topstepx_account_id=getenv("TOPSTEPX_ACCOUNT_ID", ""),
            topstepx_market_hub_url=getenv("TOPSTEPX_MARKET_HUB_URL", "wss://rtc.topstepx.com/hubs/market"),
            topstepx_user_hub_url=os.getenv("TOPSTEPX_USER_HUB_URL"),
            topstepx_rest_base_url=getenv("TOPSTEPX_REST_BASE_URL", "https://api.topstepx.com"),
            # LLM
            llm_api_url=getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
            llm_api_key=getenv("LLM_API_KEY", ""),
            openai_api_key=getenv("OPENAI_API_KEY", getenv("LLM_API_KEY", "")),
            openai_model=getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            # Trading
            mode=mode,
            symbol=getenv("TRADING_SYMBOL", "NQZ5"),
            account_balance=getenv_float("ACCOUNT_BALANCE", 50000.0),
            risk_per_trade_fraction=getenv_float("RISK_PER_TRADE_FRACTION", 0.0025),
            min_risk_fraction=getenv_float("MIN_RISK_FRACTION", 0.001),
            max_risk_fraction=getenv_float("MAX_RISK_FRACTION", 0.005),
            max_daily_loss_fraction=getenv_float("MAX_DAILY_LOSS_FRACTION", 0.03),
            # LLM scheduling
            llm_decision_interval_default_sec=getenv_int("LLM_DECISION_INTERVAL_DEFAULT_SEC", 60),
            llm_decision_interval_outer_band_sec=getenv_int("LLM_DECISION_INTERVAL_OUTER_BAND_SEC", 30),
            llm_decision_interval_inner_band_sec=getenv_int("LLM_DECISION_INTERVAL_INNER_BAND_SEC", 10),
        )


settings = Settings.from_env()
