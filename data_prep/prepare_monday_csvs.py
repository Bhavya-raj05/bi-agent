"""
prepare_monday_csvs.py
----------------------
ONE-TIME import preparation. This script is NOT part of the running agent.

The agent queries monday.com live via GraphQL (see src/monday_client.py) and
never reads these CSVs at runtime. This script exists only to get the two
supplied spreadsheets into monday.com in a shape that monday can import
without mangling.

What it fixes:
  Work Orders
    - real header sits on row 2 (row 1 is a title band)
    - drops 4 columns that are 100% empty
    - splits "5360 HA" style quantities into value + unit columns
    - normalises month names (Dec/June -> December/June)
  Deals
    - removes repeated header rows embedded mid-data (rows 50, 179)
    - removes exact duplicate rows
    - strips the "A. " / "E. " ordering prefix into a separate rank column

It deliberately does NOT invent values for missing data. Blanks stay blank so
the agent can report coverage honestly.

Usage:
    python data_prep/prepare_monday_csvs.py \
        --work-orders Work_Order_Tracker_Data.xlsx \
        --deals Deal_funnel_Data.xlsx \
        --outdir data_prep/out
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

EMPTY_WO_COLUMNS = [
    "Expected Billing Month",
    "Actual Collection Month",
    "Collection status",
    "Collection Date",
]

MONTH_CANON = {
    "jan": "January", "january": "January",
    "feb": "February", "february": "February",
    "mar": "March", "march": "March",
    "apr": "April", "april": "April",
    "may": "May",
    "jun": "June", "june": "June",
    "jul": "July", "july": "July",
    "aug": "August", "august": "August",
    "sep": "September", "sept": "September", "september": "September",
    "oct": "October", "october": "October",
    "nov": "November", "november": "November",
    "dec": "December", "december": "December",
}

QUANTITY_RE = re.compile(r"^\s*([0-9][0-9,]*\.?[0-9]*)\s*([A-Za-z%²/ ]*)\s*$")


def canon_month(value):
    if pd.isna(value):
        return None
    return MONTH_CANON.get(str(value).strip().lower(), str(value).strip())


def split_quantity(value):
    """'5360 HA' -> (5360.0, 'HA');  600 -> (600.0, None);  junk -> (None, raw)."""
    if pd.isna(value):
        return (None, None)
    if isinstance(value, (int, float)):
        return (float(value), None)
    match = QUANTITY_RE.match(str(value))
    if not match:
        return (None, str(value).strip())
    number = float(match.group(1).replace(",", ""))
    unit = (match.group(2) or "").strip().upper() or None
    return (number, unit)


def prepare_work_orders(path: Path) -> pd.DataFrame:
    # header=1 because row 0 of the sheet is a title band, not column names
    df = pd.read_excel(path, header=1)
    df = df.dropna(how="all")

    dropped = [c for c in EMPTY_WO_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"  dropped {len(dropped)} fully-empty columns: {dropped}")

    for col in ("Last executed month of recurring project", "Actual Billing Month"):
        if col in df.columns:
            df[col] = df[col].map(canon_month)

    for col in ("Quantity by Ops", "Quantities as per PO", "Balance in quantity"):
        if col in df.columns:
            parsed = df[col].map(split_quantity)
            df[col] = [p[0] for p in parsed]
            unit_col = f"{col} (unit)"
            units = [p[1] for p in parsed]
            if any(u for u in units):
                df[unit_col] = units

    # Serial # is the natural unique key; make it the monday item name.
    df = df.rename(columns={"Serial #": "Name"})
    cols = ["Name"] + [c for c in df.columns if c != "Name"]
    df = df[cols]

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")

    print(f"  work orders: {len(df)} rows x {len(df.columns)} columns")
    return df


def prepare_deals(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    before = len(df)

    # Repeated header rows pasted into the middle of the data.
    junk = df["Deal Stage"].astype(str).str.strip() == "Deal Stage"
    df = df[~junk]
    print(f"  removed {int(junk.sum())} repeated header rows")

    df = df.drop_duplicates()
    print(f"  removed {before - int(junk.sum()) - len(df)} exact duplicate rows")

    # "E. Proposal/Commercials Sent" -> rank 5, label "Proposal/Commercials Sent"
    def stage_parts(value):
        if pd.isna(value):
            return (None, None)
        text = str(value).strip()
        match = re.match(r"^([A-Z])\.\s*(.+)$", text)
        if match:
            return (ord(match.group(1)) - 64, match.group(2).strip())
        return (None, text)

    parts = df["Deal Stage"].map(stage_parts)
    df["Stage Order"] = [p[0] for p in parts]
    df["Stage Label"] = [p[1] for p in parts]

    # monday needs a Name column; Deal Name repeats, so build a stable key.
    df = df.reset_index(drop=True)
    df.insert(0, "Name", [f"DEAL-{i + 1:03d} {n}" for i, n in enumerate(df["Deal Name"].fillna("Unnamed"))])

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        else:
            # mixed object columns holding real datetimes
            df[col] = df[col].map(
                lambda v: v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else v
            )

    print(f"  deals: {len(df)} rows x {len(df.columns)} columns")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-orders", required=True)
    parser.add_argument("--deals", required=True)
    parser.add_argument("--outdir", default="data_prep/out")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Work Orders:")
    wo = prepare_work_orders(Path(args.work_orders))
    wo_path = outdir / "work_orders_monday.csv"
    wo.to_csv(wo_path, index=False)

    print("Deals:")
    deals = prepare_deals(Path(args.deals))
    deals_path = outdir / "deals_monday.csv"
    deals.to_csv(deals_path, index=False)

    print(f"\nWrote {wo_path} and {deals_path}")
    print("Import these two files into monday.com as separate boards.")


if __name__ == "__main__":
    main()
