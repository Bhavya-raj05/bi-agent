"""
Turns raw monday.com items into two clean pandas DataFrames plus a data-quality
report. This is the only place that knows about monday's item shape; everything
downstream sees canonical fields.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from . import normalize as nz
from .config import settings
from .monday_client import MondayClient, RawBoard
from .schema import DEALS_SCHEMA, WORK_ORDERS_SCHEMA, BoardSchema, coerce

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    deals: pd.DataFrame
    work_orders: pd.DataFrame
    quality: dict[str, Any]
    as_of: date

    def caveats_for(self, board: str, fields: list[str]) -> list[str]:
        """Human-readable coverage warnings for the fields an answer relies on."""
        out = []
        coverage = self.quality[board]["coverage"]
        for name in fields:
            pct = coverage.get(name)
            if pct is not None and pct < 95:
                out.append(f"`{name}` is populated on {pct:.0f}% of {board.replace('_',' ')}")
        return out


def _rows_from_board(raw: RawBoard, schema: BoardSchema) -> tuple[list[dict], dict]:
    """Flatten monday items to {canonical_field: parsed_value}, tracking what
    could not be mapped or parsed."""
    titles = raw.column_titles
    unmapped: set[str] = set()
    unparsed: dict[str, int] = {}
    rows: list[dict] = []
    dropped_header_rows = 0

    for item in raw.items:
        raw_row: dict[str, Any] = {"monday_item_id": item["id"]}
        by_title: dict[str, Any] = {}

        # The item's own Name column holds the first spreadsheet column on import.
        name_field = schema.fields[0]
        by_title[name_field.aliases[0] if name_field.aliases else name_field.name] = item.get("name")

        for cell in item.get("column_values", []):
            title = titles.get(cell["id"], cell["id"])
            by_title[title] = cell.get("text")

        if nz.is_repeated_header(by_title):
            dropped_header_rows += 1
            continue

        for title, value in by_title.items():
            field_obj = schema.match(title)
            if field_obj is None:
                if nz.clean_text(value) is not None:
                    unmapped.add(title)
                    raw_row[f"extra::{title}"] = nz.clean_text(value)
                continue

            if field_obj.kind == "stage":
                label, rank = nz.parse_stage(value)
                raw_row[field_obj.name] = label
                raw_row[f"{field_obj.name}_rank"] = rank
            elif field_obj.kind == "quantity":
                amount, unit = nz.parse_quantity(value)
                raw_row[field_obj.name] = amount
                raw_row[f"{field_obj.name}_unit"] = unit
            elif field_obj.kind == "month":
                raw_row[field_obj.name] = nz.resolve_fiscal_month(value, settings.today)
                raw_row[f"{field_obj.name}_raw"] = nz.clean_text(value)
            else:
                parsed = coerce(field_obj, value)
                raw_row[field_obj.name] = parsed
                if parsed is None and nz.clean_text(value) is not None:
                    unparsed[field_obj.name] = unparsed.get(field_obj.name, 0) + 1

        rows.append(raw_row)

    diagnostics = {
        "unmapped_columns": sorted(unmapped),
        "unparsed_values": unparsed,
        "dropped_header_rows": dropped_header_rows,
    }
    return rows, diagnostics


def _finalize_deals(df: pd.DataFrame, today: date) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates(
        subset=[c for c in ("deal_name", "client_code", "deal_stage", "deal_value",
                            "created_date") if c in df],
        keep="first",
    ).copy()

    df["probability_weight"] = df.get("closure_probability").map(nz.probability_weight) \
        if "closure_probability" in df else None
    df["weighted_value"] = df["deal_value"] * df["probability_weight"] \
        if "deal_value" in df else None

    rank = df.get("deal_stage_rank")
    df["is_open"] = df.get("deal_status").eq("Open") if "deal_status" in df else False
    df["is_won"] = (df.get("deal_status").eq("Won")) | (rank.isin(nz.WON_STAGE_RANKS)) \
        if rank is not None else df.get("deal_status").eq("Won")
    df["is_lost"] = (df.get("deal_status").eq("Dead")) | (rank.isin(nz.LOST_STAGE_RANKS)) \
        if rank is not None else df.get("deal_status").eq("Dead")

    for col in ("created_date", "tentative_close_date", "close_date_actual"):
        if col in df:
            df[f"{col}_fq"] = df[col].map(lambda d: nz.fiscal_label(d) if isinstance(d, date) else None)

    if "tentative_close_date" in df:
        df["is_stale"] = df["tentative_close_date"].map(
            lambda d: isinstance(d, date) and d < today) & df["is_open"]
    return df


def _finalize_work_orders(df: pd.DataFrame, today: date) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["work_order_id"], keep="first").copy() \
        if "work_order_id" in df else df.copy()

    order = df.get("order_value_ex_gst")
    billed = df.get("billed_ex_gst")
    if order is not None and billed is not None:
        df["billed_pct"] = (billed / order.replace(0, pd.NA) * 100).astype("Float64")
        # billed_ex is sparse; fall back to the inc-GST pair when it is missing.
        fallback = df.get("billed_inc_gst") / df.get("order_value_inc_gst").replace(0, pd.NA) * 100
        df["billed_pct"] = df["billed_pct"].fillna(fallback.astype("Float64"))

    if "collected_inc_gst" in df and "billed_inc_gst" in df:
        df["collection_pct"] = (df["collected_inc_gst"]
                                / df["billed_inc_gst"].replace(0, pd.NA) * 100).astype("Float64")
        df["uncollected_inc_gst"] = (df["billed_inc_gst"].fillna(0)
                                     - df["collected_inc_gst"].fillna(0)).clip(lower=0)

    if "receivable" in df:
        df["has_receivable"] = df["receivable"].fillna(0) > 0

    # Over-billing shows up as a negative "amount to be billed".
    if "to_bill_ex_gst" in df:
        df["is_overbilled"] = df["to_bill_ex_gst"].fillna(0) < -1

    for col in ("po_date", "last_invoice_date", "end_date", "data_delivery_date"):
        if col in df:
            df[f"{col}_fq"] = df[col].map(
                lambda d: nz.fiscal_label(d) if isinstance(d, date) else None)

    if "end_date" in df and "execution_status" in df:
        df["is_overdue"] = df.apply(
            lambda r: isinstance(r["end_date"], date) and r["end_date"] < today
            and r["execution_status"] not in ("Completed", None),
            axis=1,
        )
    return df


def _coverage(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    return {c: round(float(df[c].notna().mean() * 100), 1)
            for c in df.columns if not c.startswith("extra::")}


def build_dataset(client: MondayClient | None = None, force: bool = False) -> Dataset:
    """Fetch both boards live from monday.com and return clean frames."""
    client = client or MondayClient()
    today = settings.today

    deals_raw = client.fetch_board(settings.deals_board_id, force=force)
    work_raw = client.fetch_board(settings.work_orders_board_id, force=force)

    deal_rows, deal_diag = _rows_from_board(deals_raw, DEALS_SCHEMA)
    work_rows, work_diag = _rows_from_board(work_raw, WORK_ORDERS_SCHEMA)

    deals = _finalize_deals(pd.DataFrame(deal_rows), today)
    work_orders = _finalize_work_orders(pd.DataFrame(work_rows), today)

    quality = {
        "as_of": today.isoformat(),
        "deals": {
            "board_name": deals_raw.name,
            "rows_fetched": len(deals_raw.items),
            "rows_usable": int(len(deals)),
            "duplicates_removed": len(deal_rows) - len(deals),
            "coverage": _coverage(deals),
            **deal_diag,
        },
        "work_orders": {
            "board_name": work_raw.name,
            "rows_fetched": len(work_raw.items),
            "rows_usable": int(len(work_orders)),
            "duplicates_removed": len(work_rows) - len(work_orders),
            "coverage": _coverage(work_orders),
            **work_diag,
        },
    }
    return Dataset(deals=deals, work_orders=work_orders, quality=quality, as_of=today)
