"""Configuration and fiscal-calendar helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime

try:
    import streamlit as st
except Exception:  # pragma: no cover - allows non-streamlit use
    st = None


def _secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment."""
    if st is not None:
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:
            pass
    return os.environ.get(name, default)


@dataclass
class Settings:
    monday_api_token: str
    work_orders_board_id: str
    deals_board_id: str
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    cache_ttl_seconds: int = 300

    @property
    def is_configured(self) -> bool:
        return bool(
            self.monday_api_token
            and self.work_orders_board_id
            and self.deals_board_id
            and self.gemini_api_key
        )

    def missing(self) -> list[str]:
        names = {
            "MONDAY_API_TOKEN": self.monday_api_token,
            "WORK_ORDERS_BOARD_ID": self.work_orders_board_id,
            "DEALS_BOARD_ID": self.deals_board_id,
            "GEMINI_API_KEY": self.gemini_api_key,
        }
        return [k for k, v in names.items() if not v]


def load_settings() -> Settings:
    return Settings(
        monday_api_token=_secret("MONDAY_API_TOKEN"),
        work_orders_board_id=_secret("WORK_ORDERS_BOARD_ID"),
        deals_board_id=_secret("DEALS_BOARD_ID"),
        gemini_api_key=_secret("GEMINI_API_KEY"),
        gemini_model=_secret("GEMINI_MODEL", "gemini-2.5-flash"),
    )


# --- Fiscal calendar -------------------------------------------------------
# Skylark is an Indian company: amounts are in rupees and invoice numbers read
# "SDPL/FY25-26/916". So "this quarter" means the Indian fiscal quarter
# (Apr-Jun = Q1), not the calendar quarter. This is documented in the
# Decision Log and surfaced to the user in every date-scoped answer.

FY_QUARTER_MONTHS = {
    1: (4, 6),
    2: (7, 9),
    3: (10, 12),
    4: (1, 3),
}


def fiscal_year_of(d: date) -> int:
    """FY2026 = 1 Apr 2025 to 31 Mar 2026 (labelled FY25-26)."""
    return d.year + 1 if d.month >= 4 else d.year


def fiscal_quarter_of(d: date) -> int:
    for q, (start, end) in FY_QUARTER_MONTHS.items():
        if start <= d.month <= end:
            return q
    return 4


def fiscal_quarter_bounds(fy: int, quarter: int) -> tuple[date, date]:
    start_month, end_month = FY_QUARTER_MONTHS[quarter]
    year = fy - 1 if start_month >= 4 else fy
    start = date(year, start_month, 1)
    if end_month == 12:
        end = date(year, 12, 31)
    else:
        next_month = date(year, end_month + 1, 1)
        end = date(next_month.year, next_month.month, 1)
        end = date(end.year, end.month, 1)
        end = end.replace(day=1)
        from datetime import timedelta

        end = end - timedelta(days=1)
    return start, end


def fy_label(fy: int) -> str:
    return f"FY{str(fy - 1)[2:]}-{str(fy)[2:]}"


def today() -> date:
    override = _secret("TODAY_OVERRIDE")
    if override:
        try:
            return datetime.strptime(override, "%Y-%m-%d").date()
        except ValueError:
            pass
    return date.today()
