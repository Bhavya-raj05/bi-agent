"""
Value-level cleaning for messy monday.com data.

Every function here is total: it never raises on bad input, it returns None and
lets the caller decide. Coverage (how much of a column survived parsing) is
tracked by store.py and surfaced to the user as a caveat.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------

_NULL_TOKENS = {
    "", "-", "--", "n/a", "na", "nan", "nil", "none", "null", "tbd", "tba",
    "not applicable", "#n/a", "#value!", "#ref!", "?", ".",
}


def clean_text(value: Any) -> str | None:
    """Trim, collapse whitespace, map spreadsheet null-ish tokens to None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in _NULL_TOKENS:
        return None
    return text or None


def title_key(value: Any) -> str:
    """Aggressive key for matching column titles: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def canonical_label(value: Any) -> str | None:
    """Case/spacing-insensitive label with original-ish casing preserved.

    'BIlled' and 'billed ' both become 'Billed', so groupbys don't split.
    """
    text = clean_text(value)
    if text is None:
        return None
    return " ".join(word.capitalize() if word.isupper() or word.islower() else word
                    for word in text.split())


# --------------------------------------------------------------------------
# numbers and money
# --------------------------------------------------------------------------

_CURRENCY_NOISE = re.compile(r"[₹$,\s]|(?i:rs\.?|inr|lacs?|lakhs?|crores?|cr\b)")


def parse_number(value: Any) -> float | None:
    """Parse a number out of anything. Handles '₹ 1,23,456.78', '(500)' = -500."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    text = clean_text(value)
    if text is None:
        return None

    negative = text.startswith("(") and text.endswith(")")
    multiplier = 1.0
    lowered = text.lower()
    if re.search(r"\b(cr|crores?)\b", lowered):
        multiplier = 1e7
    elif re.search(r"\b(lacs?|lakhs?)\b", lowered):
        multiplier = 1e5

    text = _CURRENCY_NOISE.sub("", text).strip("()")
    match = re.search(r"-?\d*\.?\d+", text)
    if not match:
        return None
    try:
        number = float(match.group()) * multiplier
    except ValueError:
        return None
    return -number if negative else number


# Unit families seen in 'Quantities as per PO'. Everything in a family is
# converted to the family's base unit so quantities are comparable.
_UNIT_ALIASES = {
    "ha": ("hectare", 1.0), "hect": ("hectare", 1.0), "hectare": ("hectare", 1.0),
    "hectares": ("hectare", 1.0),
    "acr": ("hectare", 0.404686), "acre": ("hectare", 0.404686),
    "acres": ("hectare", 0.404686),
    "sqkm": ("hectare", 100.0), "km2": ("hectare", 100.0),
    "km": ("km", 1.0), "kms": ("km", 1.0), "kilometer": ("km", 1.0),
    "kilometers": ("km", 1.0), "kilometre": ("km", 1.0), "kilometres": ("km", 1.0),
    "mw": ("mw", 1.0), "kw": ("mw", 0.001),
    "month": ("month", 1.0), "months": ("month", 1.0), "mnths": ("month", 1.0),
    "visit": ("visit", 1.0), "visits": ("visit", 1.0),
    "location": ("location", 1.0), "locations": ("location", 1.0),
    "site": ("location", 1.0), "sites": ("location", 1.0),
    "mine": ("location", 1.0), "mines": ("location", 1.0),
    "tower": ("unit", 1.0), "towers": ("unit", 1.0),
    "nos": ("unit", 1.0), "no": ("unit", 1.0), "unit": ("unit", 1.0),
    "units": ("unit", 1.0), "qty": ("unit", 1.0),
}


def parse_quantity(value: Any) -> tuple[float | None, str | None]:
    """'5360 HA' -> (5360.0, 'hectare'); '2057 Acr' -> (832.4, 'hectare');
    '3956HA' -> (3956.0, 'hectare'); bare '600' -> (600.0, None).

    Quantities without a unit are NOT assumed to be hectares - they stay
    unitless so they are never silently summed with areas.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = parse_number(value)
        return (number, None)
    text = clean_text(value)
    if text is None:
        return (None, None)

    match = re.match(r"^\s*(-?\d[\d,]*\.?\d*)\s*([A-Za-z][A-Za-z\.\s]*)?", text)
    if not match:
        return (parse_number(text), None)

    number = parse_number(match.group(1))
    unit_raw = (match.group(2) or "").strip().lower().replace(".", "").replace(" ", "")
    if not unit_raw:
        return (number, None)

    family, factor = _UNIT_ALIASES.get(unit_raw, (None, 1.0))
    if family is None:
        return (number, unit_raw)          # unknown unit: keep it, don't convert
    return (None if number is None else round(number * factor, 4), family)


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------

_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d-%b-%y", "%d.%m.%Y",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
)

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12,
    "december": 12,
}


def parse_date(value: Any) -> date | None:
    """Parse a date from a monday text value, ISO string, or Excel serial.

    Ambiguous d/m vs m/d: day-first is tried first because the source is an
    Indian operations tracker. Documented in DECISION_LOG.md.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Excel serial date (1900 system). Guard the plausible range.
        if 20000 < float(value) < 60000:
            from datetime import timedelta
            return (datetime(1899, 12, 30) + timedelta(days=float(value))).date()
        return None

    text = clean_text(value)
    if text is None:
        return None
    text = text.split("T")[0] if re.match(r"^\d{4}-\d{2}-\d{2}T", text) else text
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_month_name(value: Any) -> int | None:
    """'June', 'Dec', 'december ' -> month number. Bare month names carry no
    year; resolve_fiscal_month() attaches one."""
    text = clean_text(value)
    if text is None:
        return None
    return _MONTHS.get(text.lower().strip(".")[:9]) or _MONTHS.get(text.lower()[:3])


# --------------------------------------------------------------------------
# Indian fiscal calendar (April - March), confirmed by invoice numbers
# formatted 'SDPL/FY25-26/431'.
# --------------------------------------------------------------------------

def fiscal_year(day: date) -> int:
    """FY label start year: 2025-06-01 and 2026-02-01 both -> 2025 (FY25-26)."""
    return day.year if day.month >= 4 else day.year - 1


def fiscal_quarter(day: date) -> int:
    """Q1 = Apr-Jun ... Q4 = Jan-Mar."""
    return ((day.month - 4) % 12) // 3 + 1


def fiscal_label(day: date) -> str:
    start = fiscal_year(day)
    return f"FY{str(start)[2:]}-{str(start + 1)[2:]} Q{fiscal_quarter(day)}"


def fiscal_quarter_bounds(reference: date, offset: int = 0) -> tuple[date, date]:
    """Start/end dates of the fiscal quarter containing `reference`,
    shifted by `offset` quarters (-1 = previous quarter)."""
    quarter_index = (fiscal_year(reference) * 4 + (fiscal_quarter(reference) - 1)) + offset
    year, quarter = divmod(quarter_index, 4)
    start_month = 4 + quarter * 3
    start_year = year
    if start_month > 12:
        start_month -= 12
        start_year += 1
    start = date(start_year, start_month, 1)
    end_month, end_year = start_month + 3, start_year
    if end_month > 12:
        end_month -= 12
        end_year += 1
    from datetime import timedelta
    return start, date(end_year, end_month, 1) - timedelta(days=1)


def resolve_fiscal_month(month_name: Any, reference: date) -> date | None:
    """'June' with reference 2026-09-19 -> 2025-06-01, because June falls in the
    fiscal year that is currently open. Bare month names in the tracker always
    refer to the current FY."""
    month = parse_month_name(month_name)
    if month is None:
        return None
    start_year = fiscal_year(reference)
    return date(start_year if month >= 4 else start_year + 1, month, 1)


# --------------------------------------------------------------------------
# controlled vocabularies
# --------------------------------------------------------------------------

SECTOR_CANON = {
    "mining": "Mining", "mines": "Mining", "mine": "Mining",
    "powerline": "Powerline", "power line": "Powerline", "power": "Powerline",
    "transmission": "Powerline", "t&d": "Powerline",
    "renewables": "Renewables", "renewable": "Renewables", "solar": "Renewables",
    "wind": "Renewables", "energy": "Renewables",
    "railways": "Railways", "railway": "Railways", "rail": "Railways",
    "construction": "Construction", "infra": "Construction",
    "infrastructure": "Construction",
    "tender": "Tender", "tenders": "Tender",
    "dsp": "DSP", "aviation": "Aviation", "manufacturing": "Manufacturing",
    "security and surveillance": "Security & Surveillance",
    "security & surveillance": "Security & Surveillance",
    "surveillance": "Security & Surveillance", "security": "Security & Surveillance",
    "others": "Others", "other": "Others", "misc": "Others",
}

# "energy sector" in a founder's question is not a literal column value.
SECTOR_SYNONYMS = {
    "energy": ["Powerline", "Renewables"],
    "power": ["Powerline", "Renewables"],
    "utilities": ["Powerline", "Renewables"],
    "infra": ["Construction", "Railways"],
    "infrastructure": ["Construction", "Railways"],
}


def canonical_sector(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return SECTOR_CANON.get(text.lower(), text.title())


def expand_sector_query(value: str) -> list[str]:
    """Map a user's word for a sector onto the sector values that exist."""
    key = (value or "").strip().lower()
    if key in SECTOR_SYNONYMS:
        return SECTOR_SYNONYMS[key]
    canonical = canonical_sector(value)
    return [canonical] if canonical else []


DEAL_STATUS_CANON = {
    "open": "Open", "won": "Won", "dead": "Dead", "lost": "Dead",
    "on hold": "On Hold", "hold": "On Hold", "onhold": "On Hold",
}


def canonical_deal_status(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return DEAL_STATUS_CANON.get(text.lower(), text.title())


_STAGE_PATTERN = re.compile(r"^\s*([A-Z])\s*[\.\)]\s*(.+)$")

# Stage letters encode funnel order in the source data. Rows without a letter
# ('Project Completed') are appended after the lettered ladder.
_UNLETTERED_STAGE_RANK = {"project completed": 99}


def parse_stage(value: Any) -> tuple[str | None, int | None]:
    """'E. Proposal/Commercials Sent' -> ('Proposal/Commercials Sent', 5)."""
    text = clean_text(value)
    if text is None:
        return (None, None)
    match = _STAGE_PATTERN.match(text)
    if match:
        letter, label = match.group(1), match.group(2).strip()
        return (label, ord(letter.upper()) - ord("A") + 1)
    return (text, _UNLETTERED_STAGE_RANK.get(text.lower()))


# Stages that mean the deal is commercially committed, used for "won pipeline".
WON_STAGE_RANKS = {7, 8, 10, 11, 99}   # G,H,J,K + Project Completed
LOST_STAGE_RANKS = {12, 14, 15}        # L,N,O

PROBABILITY_WEIGHTS = {"high": 0.8, "medium": 0.5, "low": 0.2}


def probability_weight(value: Any) -> float | None:
    text = clean_text(value)
    if text is None:
        return None
    return PROBABILITY_WEIGHTS.get(text.lower())


def is_repeated_header(row: dict[str, Any]) -> bool:
    """Spreadsheet exports sometimes carry a repeated header row into the data.
    Detect it by cells whose value equals their own column title."""
    matches = sum(
        1 for key, value in row.items()
        if isinstance(value, str) and title_key(value) == title_key(key) and value.strip()
    )
    return matches >= 3
