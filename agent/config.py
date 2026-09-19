"""Configuration. Everything comes from the environment (or Streamlit secrets)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date


def _secret(name: str, default: str = "") -> str:
    """Environment first, then Streamlit secrets when running in the cloud."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


@dataclass
class Settings:
    monday_token: str = field(default_factory=lambda: _secret("MONDAY_API_TOKEN"))
    deals_board_id: str = field(default_factory=lambda: _secret("MONDAY_DEALS_BOARD_ID"))
    work_orders_board_id: str = field(
        default_factory=lambda: _secret("MONDAY_WORK_ORDERS_BOARD_ID"))

    anthropic_key: str = field(default_factory=lambda: _secret("ANTHROPIC_API_KEY"))
    model: str = field(default_factory=lambda: _secret("ANTHROPIC_MODEL", "claude-sonnet-4-5"))

    page_size: int = 250
    max_retries: int = 4
    cache_ttl: int = 300           # seconds; boards change slowly
    max_tool_turns: int = 8

    @property
    def today(self) -> date:
        """Overridable 'today' so demos and tests are reproducible."""
        override = _secret("AGENT_TODAY")
        if override:
            from .normalize import parse_date
            return parse_date(override) or date.today()
        return date.today()

    def missing(self) -> list[str]:
        required = {
            "MONDAY_API_TOKEN": self.monday_token,
            "MONDAY_DEALS_BOARD_ID": self.deals_board_id,
            "MONDAY_WORK_ORDERS_BOARD_ID": self.work_orders_board_id,
            "ANTHROPIC_API_KEY": self.anthropic_key,
        }
        return [name for name, value in required.items() if not value]


settings = Settings()
