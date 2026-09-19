"""
Data resilience layer.

Everything that comes back from monday.com is treated as untrusted text. This
module turns it into two tidy DataFrames plus a DataQualityReport that the
agent attaches to its answers.

The guiding rule: never silently invent a value. A field that cannot be parsed
becomes NaN/None and is counted in the quality report, so any figure derived
from it can be reported with its coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

# --- vocabularies ----------------------------------------------------------

# Free-text sector words a founder might use -> sector values in the data.
# "energy" is the one in the brief's example query and does not exist as a
# sector value, so it fans out to the two sectors that are actually energy.
SECTOR_SYNONYMS: dict[str, list[str]] = {
    "energy": ["Renewables", "Powerline"],
    "power": ["Powerline", "Renewables"],
    "solar": ["Renewables"],
    "wind": ["Renewables"],
    "renewable": ["Renewables"],
    "renewables": ["Renewables"],
    "transmission": ["Powerline"],
    "powerline": ["Powerline"],
    "power line": ["Powerline"],
    "utilities": ["Powerline", "Renewables"],
    "mining": ["Mining"],
    "mines": ["Mining"],
    "rail": ["Railways"],
    "railway": ["Railways"],
    "railways": ["Railways"],
    "construction": ["Construction"],
    "infra": ["Construction", "Railways"],
    "infrastructure": ["Construction", "Railways"],
    "manufacturing": ["Manufacturing"],
    "aviation": ["Aviation"],
    "security": ["Security and Surveillance"],
    "surveillance": ["Security and Surveillance"],
    "dsp": ["DSP"],
    "tender": ["Tender"],
    "others": ["Others"],
    "other": ["Others"],
}

# Invoice / billing status strings in the raw data, mapped to a clean set.
# Note 'BIlled' - a real typo in the source; we fold it in rather than
# pretending the data is clean.
BILLING_STATUS_CANON = {
    "billed": "Billed",
    "bilٍled": "Billed",
    "fully billed": "Billed",
    "partially billed": "Partially Billed",
    "not billed yet": "Not Billed",
    "not billable": "Not Billable",
    "update required": "Unknown",
    "stuck": "Stuck",
}

EXECUTION_STATUS_CANON = {
    "completed": "Completed",
    "partial completed": "Partially Completed",
    "ongoing": "Ongoing",
    "executed until current month": "Ongoing",
    "not started": "Not Started",
    "pause / struck": "Paused",
    "details pending from client": "Blocked",
}

DEAL_STATUS_CANON = {
    "open": "Open",
    "won": "Won",
    "dead": "Lost",
    "on hold": "On Hold",
}

MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%Y-%m-%d %H:%M:%S",
]

NUMERIC_RE = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")

# Rows whose values echo the column header - the repeated-header-row problem.
HEADER_ECHO_FIELDS = ["Deal Status", "Deal Stage", "Sector/service"]


# --- quality reporting -----------------------------------------------------


@dataclass
class DataQualityReport:
    board: str
    rows_received: int = 0
    rows_dropped_header_echo: int = 0
    rows_dropped_duplicate: int = 0
    rows_dropped_empty: int = 0
    unparsed_dates: dict[str, int] = field(default_factory=dict)
    unparsed_numbers: dict[str, int] = field(default_factory=dict)
    null_counts: dict[str, int] = field(default_factory=dict)
    unmapped_values: dict[str, set] = field(default_factory=dict)

    @property
    def rows_kept(self) -> int:
        return (
            self.rows_received
            - self.rows_dropped_header_echo
            - self.rows_dropped_duplicate
            - self.rows_dropped_empty
        )

    def coverage(self, column: str) -> float:
        """Fraction of kept rows that have a usable value in `column`."""
        if self.rows_kept <= 0:
            return 0.0
        nulls = self.null_counts.get(column, 0)
        return max(0.0, (self.rows_kept - nulls) / self.rows_kept)

    def summary_lines(self) -> list[str]:
        lines = [f"{self.board}: {self.rows_kept} usable rows of {self.rows_received} fetched."]
        dropped = []
        if self.rows_dropped_header_echo:
            dropped.append(f"{self.rows_dropped_header_echo} repeated header rows")
        if self.rows_dropped_duplicate:
            dropped.append(f"{self.rows_dropped_duplicate} duplicates")
        if self.rows_dropped_empty:
            dropped.append(f"{self.rows_dropped_empty} empty rows")
        if dropped:
            lines.append("Dropped " + ", ".join(dropped) + ".")
        thin = {
            col: self.coverage(col)
            for col in self.null_counts
            if self.coverage(col) < 0.75
        }
        for col, cov in sorted(thin.items(), key=lambda kv: kv[1])[:6]:
            lines.append(f"'{col}' is only {cov:.0%} populated.")
        return lines


# --- scalar parsers --------------------------------------------------------


def parse_number(value: Any) -> float | None:
    """'₹1,23,456.78' -> 123456.78 ; '5360 HA' -> 5360.0 ; '' -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None if pd.isna(value) else float(value)
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "-", "nil", "none", "tbd"}:
        return None
    match = NUMERIC_RE.search(text.replace("\u20b9", ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "-", "none", "tbd"}:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
        return None if pd.isna(parsed) else parsed.date()
    except Exception:
        return None


def parse_month(value: Any) -> int | None:
    if value is None:
        return None
    return MONTH_NAMES.get(str(value).strip().lower())


def canon(value: Any, mapping: dict[str, str], report: DataQualityReport | None = None,
          field_name: str = "") -> str | None:
    if value is None or str(value).strip() == "":
        return None
    key = str(value).strip().lower()
    if key in mapping:
        return mapping[key]
    # Prefix match handles 'Billed- Visit 7' style variants.
    for prefix, canonical in mapping.items():
        if key.startswith(prefix):
            return canonical
    if report is not None and field_name:
        report.unmapped_values.setdefault(field_name, set()).add(str(value).strip())
    return str(value).strip()


def split_multi(value: Any) -> list[str]:
    """'Topography Survey: RGB, Hydrology' -> ['Topography Survey: RGB', 'Hydrology']"""
    if value is None or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def resolve_sectors(text: str) -> list[str]:
    """Map a founder's free-text sector word onto actual sector values."""
    key = (text or "").strip().lower()
    if key in SECTOR_SYNONYMS:
        return SECTOR_SYNONYMS[key]
    for word, sectors in SECTOR_SYNONYMS.items():
        if word in key:
            return sectors
    return [text.strip().title()] if text else []


# --- row-level cleanup -----------------------------------------------------


def _is_header_echo(row: dict) -> bool:
    for field_name in HEADER_ECHO_FIELDS:
        if str(row.get(field_name, "")).strip() == field_name:
            return True
    return False


def _is_empty(row: dict) -> bool:
    values = [v for k, v in row.items() if not k.startswith("__")]
    return all(v is None or str(v).strip() == "" for v in values)


def _pick(row: dict, *candidates: str) -> Any:
    """Return the first present column among candidates (title drift tolerant)."""
    for name in candidates:
        if name in row and row[name] not in (None, ""):
            return row[name]
    lowered = {k.strip().lower(): v for k, v in row.items()}
    for name in candidates:
        v = lowered.get(name.strip().lower())
        if v not in (None, ""):
            return v
    return None


# --- board normalisers -----------------------------------------------------


def normalize_work_orders(rows: Iterable[dict]) -> tuple[pd.DataFrame, DataQualityReport]:
    rows = list(rows)
    report = DataQualityReport(board="Work Orders", rows_received=len(rows))

    records = []
    seen = set()
    for row in rows:
        if _is_empty(row):
            report.rows_dropped_empty += 1
            continue
        if _is_header_echo(row):
            report.rows_dropped_header_echo += 1
            continue

        serial = _pick(row, "Name", "Serial #", "__name")
        key = (serial, _pick(row, "Deal name masked"), _pick(row, "Date of PO/LOI"))
        if key in seen:
            report.rows_dropped_duplicate += 1
            continue
        seen.add(key)

        po_date = parse_date(_pick(row, "Date of PO/LOI"))
        records.append(
            {
                "work_order_id": serial,
                "deal_name": _pick(row, "Deal name masked"),
                "customer": _pick(row, "Customer Name Code"),
                "sector": (_pick(row, "Sector") or "").strip() or None,
                "nature_of_work": _pick(row, "Nature of Work"),
                "work_types": split_multi(_pick(row, "Type of Work")),
                "work_type_raw": _pick(row, "Type of Work"),
                "owner": _pick(row, "BD/KAM Personnel code"),
                "execution_status": canon(
                    _pick(row, "Execution Status"), EXECUTION_STATUS_CANON,
                    report, "Execution Status",
                ),
                "billing_status": canon(
                    _pick(row, "Invoice Status", "Billing Status"),
                    BILLING_STATUS_CANON, report, "Invoice Status",
                ),
                "wo_status": _pick(row, "WO Status (billed)"),
                "software_platform": _pick(
                    row,
                    "Is any Skylark software platform part of the client deliverables in this deal?",
                ),
                "po_date": po_date,
                "start_date": parse_date(_pick(row, "Probable Start Date")),
                "end_date": parse_date(_pick(row, "Probable End Date")),
                "delivery_date": parse_date(_pick(row, "Data Delivery Date")),
                "last_invoice_date": parse_date(_pick(row, "Last invoice date")),
                "order_value": parse_number(
                    _pick(row, "Amount in Rupees (Excl of GST) (Masked)")
                ),
                "order_value_incl_gst": parse_number(
                    _pick(row, "Amount in Rupees (Incl of GST) (Masked)")
                ),
                "billed_value": parse_number(
                    _pick(row, "Billed Value in Rupees (Excl of GST.) (Masked)")
                ),
                "collected_amount": parse_number(
                    _pick(row, "Collected Amount in Rupees (Incl of GST.) (Masked)")
                ),
                "to_be_billed": parse_number(
                    _pick(row, "Amount to be billed in Rs. (Exl. of GST) (Masked)")
                ),
                "receivable": parse_number(_pick(row, "Amount Receivable (Masked)")),
                "ar_priority": (
                    str(_pick(row, "AR Priority account") or "").strip().lower() == "priority"
                ),
                "qty_po": parse_number(_pick(row, "Quantities as per PO")),
                "qty_billed": parse_number(_pick(row, "Quantity billed (till date)")),
                "qty_balance": parse_number(_pick(row, "Balance in quantity")),
                "billing_month": parse_month(_pick(row, "Actual Billing Month")),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df, report

    for column in df.columns:
        if column == "work_types":
            report.null_counts[column] = int((df[column].str.len() == 0).sum())
        else:
            report.null_counts[column] = int(df[column].isna().sum())

    return df, report


def normalize_deals(rows: Iterable[dict]) -> tuple[pd.DataFrame, DataQualityReport]:
    rows = list(rows)
    report = DataQualityReport(board="Deals", rows_received=len(rows))

    records = []
    seen = set()
    for row in rows:
        if _is_empty(row):
            report.rows_dropped_empty += 1
            continue
        if _is_header_echo(row):
            report.rows_dropped_header_echo += 1
            continue

        deal_name = _pick(row, "Deal Name")
        client = _pick(row, "Client Code")
        created = parse_date(_pick(row, "Created Date"))
        value = parse_number(_pick(row, "Masked Deal value"))
        key = (deal_name, client, created, value, _pick(row, "Deal Stage"))
        if key in seen:
            report.rows_dropped_duplicate += 1
            continue
        seen.add(key)

        stage_raw = str(_pick(row, "Deal Stage") or "").strip()
        stage_match = re.match(r"^([A-Z])\.\s*(.+)$", stage_raw)
        if stage_match:
            stage_order = ord(stage_match.group(1)) - 64
            stage_label = stage_match.group(2).strip()
        else:
            stage_order = None
            stage_label = stage_raw or None

        records.append(
            {
                "deal_name": deal_name,
                "owner": _pick(row, "Owner code"),
                "client": client,
                "status": canon(_pick(row, "Deal Status"), DEAL_STATUS_CANON,
                                report, "Deal Status"),
                "stage_label": stage_label,
                "stage_order": stage_order,
                "probability": (_pick(row, "Closure Probability") or None),
                "deal_value": value,
                "sector": (_pick(row, "Sector/service") or "").strip() or None,
                "product": _pick(row, "Product deal"),
                "created_date": created,
                "tentative_close_date": parse_date(_pick(row, "Tentative Close Date")),
                "actual_close_date": parse_date(_pick(row, "Close Date (A)")),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df, report

    for column in df.columns:
        report.null_counts[column] = int(df[column].isna().sum())

    return df, report
