"""
One-off import helper: turns the two supplied .xlsx files into CSVs that
monday.com will accept, and prints the column-type mapping to use in the
import wizard.

This runs ONCE, before the agent exists. The agent never reads these files -
it queries monday.com live. Only structurally invalid rows are removed here
(repeated header rows pasted into the data). Value-level messiness is left
intact on purpose, so the agent's cleaning layer is genuinely exercised.

    python scripts/prepare_monday_csvs.py path/to/xlsx_folder
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

RENAMES = {
    "Is any Skylark software platform part of the client deliverables in this deal?":
        "Skylark Platform",
    "Last executed month of recurring project": "Last Executed Month",
}

# Column types to pick in monday.com's import wizard.
DEALS_TYPES = {
    "Deal Name": "Name (item name)", "Owner code": "Status", "Client Code": "Text",
    "Deal Status": "Status", "Close Date (A)": "Date",
    "Closure Probability": "Status", "Masked Deal value": "Numbers",
    "Tentative Close Date": "Date", "Deal Stage": "Status",
    "Product deal": "Status", "Sector/service": "Status", "Created Date": "Date",
}
WORK_ORDER_TYPES = {
    "Deal name masked": "Name (item name)", "Customer Name Code": "Text",
    "Serial #": "Text", "Nature of Work": "Status", "Last Executed Month": "Text",
    "Execution Status": "Status", "Sector": "Status", "Type of Work": "Text",
    "Document Type": "Status", "BD/KAM Personnel code": "Status",
    "Skylark Platform": "Status", "Invoice Status": "Status",
    "WO Status (billed)": "Status", "Billing Status": "Status",
    "AR Priority account": "Status", "latest invoice no.": "Text",
    "Quantities as per PO": "Text",
    "__dates__": "Date  (Data Delivery Date, Date of PO/LOI, Probable Start/End Date, "
                 "Last invoice date, Collection Date)",
    "__numbers__": "Numbers (every Amount / Billed / Collected / Receivable / Quantity "
                   "column)",
}


def _is_repeated_header(row: pd.Series) -> bool:
    matches = sum(1 for column, value in row.items()
                  if isinstance(value, str) and value.strip().lower() == str(column).strip().lower())
    return matches >= 3


def _to_csv_value(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if pd.isna(value):
        return ""
    return value


def prepare(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    before = len(frame)
    frame = frame[~frame.apply(_is_repeated_header, axis=1)].copy()
    header_rows = before - len(frame)
    frame = frame.rename(columns=RENAMES)
    frame = frame.dropna(axis=1, how="all")          # drop entirely empty columns
    frame = frame.map(_to_csv_value)
    print(f"{label}: {before} rows -> {len(frame)} "
          f"({header_rows} repeated header rows removed, "
          f"{len(frame.columns)} columns kept)")
    return frame


def main(source_dir: str = ".") -> None:
    source = Path(source_dir)
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)

    deals = pd.read_excel(source / "Deal_funnel_Data.xlsx")
    work = pd.read_excel(source / "Work_Order_Tracker_Data.xlsx", header=1)

    prepare(deals, "Deals").to_csv(out / "deals_monday_import.csv", index=False)
    prepare(work, "Work orders").to_csv(out / "work_orders_monday_import.csv", index=False)

    print("\nColumn types for the monday.com import wizard")
    print("-- Deals board --")
    for column, kind in DEALS_TYPES.items():
        print(f"  {column:<24} {kind}")
    print("-- Work Orders board --")
    for column, kind in WORK_ORDER_TYPES.items():
        print(f"  {column:<24} {kind}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
