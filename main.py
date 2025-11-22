from __future__ import annotations

import argparse
import asyncio

from config import Settings
from engine import run_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM‑driven TopstepX trading agent")
    parser.add_argument(
        "--mode",
        choices=["live_trading", "paper_trading"],
        default=None,
        help="Trading mode (default from env TRADING_MODE, otherwise paper_trading)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Instrument symbol (e.g. NQZ5). Overrides TRADING_SYMBOL.",
    )
    return parser.parse_args()


def build_settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if args.mode is not None:
        settings.mode = args.mode  # type: ignore[assignment]
    if args.symbol is not None:
        settings.symbol = args.symbol
    return settings


if __name__ == "__main__":
    cli_args = parse_args()
    app_settings = build_settings_from_args(cli_args)
    asyncio.run(run_agent(app_settings))

