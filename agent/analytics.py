"""
Deterministic analytics over the cleaned frames.

Every function returns a plain dict: numbers, a small amount of supporting
detail, and an explicit `caveats` list. The LLM never does arithmetic - it picks
a function, reads the result, and writes the narrative. That keeps the numbers
auditable and stops the model inventing figures.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from . import normalize as nz
from .store import Dataset

MAX_ROWS = 25


# ---------------------------------------------------------------- helpers

def _money(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


def resolve_period(period: str | None, start: str | None, end: str | None,
                   today: date) -> tuple[date | None, date | None, str]:
    """Turn 'this quarter' / 'last 90 days' / explicit ISO dates into bounds."""
    if start or end:
        s, e = nz.parse_date(start), nz.parse_date(end)
        return s, e, f"{s or 'beginning'} to {e or 'today'}"

    key = (period or "").strip().lower()
    if not key or key in ("all", "all time", "alltime", "ever"):
        return None, None, "all time"
    if key in ("this quarter", "current quarter", "quarter", "qtd"):
        s, e = nz.fiscal_quarter_bounds(today, 0)
        return s, e, f"{nz.fiscal_label(today)} ({s} to {e})"
    if key in ("last quarter", "previous quarter"):
        s, e = nz.fiscal_quarter_bounds(today, -1)
        return s, e, f"previous fiscal quarter ({s} to {e})"
    if key in ("next quarter",):
        s, e = nz.fiscal_quarter_bounds(today, 1)
        return s, e, f"next fiscal quarter ({s} to {e})"
    if key in ("this fiscal year", "this year", "fy", "ytd", "fytd"):
        fy = nz.fiscal_year(today)
        return date(fy, 4, 1), date(fy + 1, 3, 31), f"FY{str(fy)[2:]}-{str(fy+1)[2:]}"
    if key in ("last fiscal year", "last year"):
        fy = nz.fiscal_year(today) - 1
        return date(fy, 4, 1), date(fy + 1, 3, 31), f"FY{str(fy)[2:]}-{str(fy+1)[2:]}"
    if key in ("this month", "month", "mtd"):
        s = today.replace(day=1)
        return s, today, f"{s:%B %Y} to date"
    for days, label in ((30, "last 30 days"), (60, "last 60 days"),
                        (90, "last 90 days"), (180, "last 180 days")):
        if key in (label, label.replace(" ", ""), f"{days}d", f"last {days}d"):
            return today - timedelta(days=days), today, label
    return None, None, f"all time (could not interpret period '{period}')"


def _filter_dates(df: pd.DataFrame, column: str, start: date | None,
                  end: date | None) -> tuple[pd.DataFrame, int]:
    """Filter on a date column, reporting how many rows had no date at all."""
    if column not in df or (start is None and end is None):
        return df, 0
    values = df[column]
    undated = int(values.isna().sum())
    mask = values.notna()
    if start:
        mask &= values.map(lambda d: isinstance(d, date) and d >= start)
    if end:
        mask &= values.map(lambda d: isinstance(d, date) and d <= end)
    return df[mask], undated


def date_range(df: pd.DataFrame, column: str) -> tuple[date | None, date | None]:
    if column not in df or df.empty:
        return None, None
    values = [v for v in df[column] if isinstance(v, date)]
    return (min(values), max(values)) if values else (None, None)


def _empty_period_note(before: pd.DataFrame, after: pd.DataFrame, column: str,
                       start: date | None, end: date | None) -> list[str]:
    """A period filter that returns nothing is usually a stale board, not a real
    zero. Say which is which instead of reporting a confident 0."""
    if len(after) or not len(before) or (start is None and end is None):
        return []
    earliest, latest = date_range(before, column)
    if earliest is None:
        return [f"No records in scope have a {column.replace('_',' ')} at all, so the "
                "period filter could not be applied - this is a data gap, not a zero."]
    return [f"Nothing falls in the requested window. The board only has "
            f"{column.replace('_',' ')} values between {earliest} and {latest}, so this "
            "period is outside what has been recorded - treat it as 'not yet updated' "
            "rather than 'no activity'."]


def _apply_filters(df: pd.DataFrame, sector=None, owner=None, client=None,
                   stage=None, status=None, extra: dict | None = None) -> pd.DataFrame:
    out = df
    if sector and "sector" in out:
        wanted = []
        for token in ([sector] if isinstance(sector, str) else list(sector)):
            wanted.extend(nz.expand_sector_query(token))
        out = out[out["sector"].isin(wanted)]
    if owner and "owner_code" in out:
        owners = [owner] if isinstance(owner, str) else list(owner)
        out = out[out["owner_code"].isin(owners)]
    if client:
        column = "client_code" if "client_code" in out else "customer_code"
        if column in out:
            clients = [client] if isinstance(client, str) else list(client)
            out = out[out[column].isin(clients)]
    if stage and "deal_stage" in out:
        stages = [stage] if isinstance(stage, str) else list(stage)
        keys = {nz.title_key(s) for s in stages}
        out = out[out["deal_stage"].map(lambda v: v is not None and (
            nz.title_key(v) in keys or any(k in nz.title_key(v) for k in keys)))]
    if status:
        column = "deal_status" if "deal_status" in out else "execution_status"
        if column in out:
            wanted = {nz.title_key(s) for s in
                      ([status] if isinstance(status, str) else list(status))}
            out = out[out[column].map(lambda v: v is not None and nz.title_key(v) in wanted)]
    for column, value in (extra or {}).items():
        if column in out and value is not None:
            out = out[out[column] == value]
    return out


def _breakdown(df: pd.DataFrame, by: str, value_column: str | None) -> list[dict]:
    if by not in df or df.empty:
        return []
    grouped = df.groupby(by, dropna=False)
    rows = []
    for key, chunk in grouped:
        row = {"group": "Unspecified" if key is None or pd.isna(key) else str(key),
               "count": int(len(chunk))}
        if value_column and value_column in chunk:
            row["value"] = _money(chunk[value_column].sum())
            row["value_known_for"] = int(chunk[value_column].notna().sum())
        rows.append(row)
    rows.sort(key=lambda r: (r.get("value") or 0, r["count"]), reverse=True)
    return rows


# ---------------------------------------------------------------- tools

def describe_data(data: Dataset) -> dict:
    """What fields and values exist, so the model filters with real values."""
    def distinct(df, column, limit=25):
        if column not in df:
            return []
        values = [v for v in df[column].dropna().unique().tolist()]
        return sorted(map(str, values))[:limit]

    def window(df, column):
        earliest, latest = date_range(df, column)
        return f"{earliest} to {latest}" if earliest else "none recorded"

    return {
        "as_of": data.as_of.isoformat(),
        "fiscal_year": "April-March (Indian FY); 'this quarter' uses it",
        "date_coverage": {
            "deals.created_date": window(data.deals, "created_date"),
            "deals.tentative_close_date": window(data.deals, "tentative_close_date"),
            "work_orders.po_date": window(data.work_orders, "po_date"),
            "work_orders.last_invoice_date": window(data.work_orders, "last_invoice_date"),
        },
        "deals": {
            "rows": int(len(data.deals)),
            "sectors": distinct(data.deals, "sector"),
            "stages": distinct(data.deals, "deal_stage", 30),
            "statuses": distinct(data.deals, "deal_status"),
            "owners": distinct(data.deals, "owner_code"),
            "products": distinct(data.deals, "product"),
        },
        "work_orders": {
            "rows": int(len(data.work_orders)),
            "sectors": distinct(data.work_orders, "sector"),
            "execution_statuses": distinct(data.work_orders, "execution_status"),
            "nature_of_work": distinct(data.work_orders, "nature_of_work"),
            "types_of_work": distinct(data.work_orders, "type_of_work", 40),
            "invoice_statuses": distinct(data.work_orders, "invoice_status"),
            "owners": distinct(data.work_orders, "owner_code"),
        },
    }


def data_quality_report(data: Dataset) -> dict:
    report = {"as_of": data.as_of.isoformat(), "boards": {}}
    for board in ("deals", "work_orders"):
        info = data.quality[board]
        coverage = info["coverage"]
        sparse = sorted(((k, v) for k, v in coverage.items() if v < 60),
                        key=lambda kv: kv[1])
        report["boards"][board] = {
            "board_name": info["board_name"],
            "rows_fetched": info["rows_fetched"],
            "rows_usable": info["rows_usable"],
            "duplicates_removed": info["duplicates_removed"],
            "repeated_header_rows_dropped": info["dropped_header_rows"],
            "columns_not_recognised": info["unmapped_columns"],
            "values_that_failed_parsing": info["unparsed_values"],
            "sparse_fields_pct_populated": dict(sparse[:15]),
            "empty_fields": [k for k, v in coverage.items() if v == 0],
        }
    return report


def pipeline_summary(data: Dataset, sector=None, owner=None, stage=None,
                     status="Open", period=None, start_date=None, end_date=None,
                     date_field="tentative_close_date", group_by="sector") -> dict:
    start, end, period_label = resolve_period(period, start_date, end_date, data.as_of)
    scoped = _apply_filters(data.deals, sector=sector, owner=owner, stage=stage,
                            status=status)
    df, undated = _filter_dates(scoped, date_field, start, end)

    value = df["deal_value"] if "deal_value" in df else pd.Series(dtype=float)
    weighted = df["weighted_value"] if "weighted_value" in df else pd.Series(dtype=float)
    known = int(value.notna().sum())

    caveats = _empty_period_note(scoped, df, date_field, start, end)
    if len(df) and known < len(df):
        caveats.append(
            f"{len(df) - known} of {len(df)} matching deals have no deal value, so the "
            f"total covers {known/len(df)*100:.0f}% of the pipeline by count."
        )
    if undated:
        caveats.append(f"{undated} deals have no {date_field.replace('_',' ')} and were "
                       "excluded from the period filter.")
    if "probability_weight" in df and len(df):
        weighted_known = int(df["probability_weight"].notna().sum())
        if weighted_known < len(df):
            caveats.append(
                f"Closure probability is set on only {weighted_known} deals; the weighted "
                "figure covers those alone (High=0.8, Medium=0.5, Low=0.2)."
            )
    if "is_stale" in df and len(df):
        stale = int(df["is_stale"].fillna(False).sum())
        if stale:
            caveats.append(f"{stale} of these deals have a tentative close date already in "
                           "the past - the funnel may be out of date.")

    return {
        "period": period_label,
        "filters": {"sector": sector, "owner": owner, "stage": stage, "status": status,
                    "date_field": date_field},
        "deal_count": int(len(df)),
        "total_value": _money(value.sum()) if known else 0.0,
        "weighted_value": _money(weighted.sum()) if len(weighted) else 0.0,
        "median_deal_value": _money(value.median()) if known else None,
        "largest_deal_value": _money(value.max()) if known else None,
        "value_coverage_pct": round(known / len(df) * 100, 1) if len(df) else 0.0,
        "breakdown": _breakdown(df, group_by, "deal_value")[:15],
        "by_stage": _breakdown(df, "deal_stage", "deal_value")[:15],
        "matched_before_period_filter": int(len(scoped)),
        "caveats": caveats,
    }


def funnel_snapshot(data: Dataset, sector=None, owner=None) -> dict:
    df = _apply_filters(data.deals, sector=sector, owner=owner)
    open_deals = df[df.get("is_open", False) == True]  # noqa: E712
    stages = []
    for rank, chunk in open_deals.groupby("deal_stage_rank", dropna=False):
        label = chunk["deal_stage"].dropna().iloc[0] if chunk["deal_stage"].notna().any() else "Unspecified"
        stages.append({
            "stage": label,
            "order": None if pd.isna(rank) else int(rank),
            "deals": int(len(chunk)),
            "value": _money(chunk["deal_value"].sum()),
        })
    stages.sort(key=lambda s: (s["order"] is None, s["order"] or 0))
    won, lost = int(df.get("is_won", pd.Series(dtype=bool)).sum()), int(df.get("is_lost", pd.Series(dtype=bool)).sum())
    decided = won + lost
    return {
        "open_deals": int(len(open_deals)),
        "open_value": _money(open_deals["deal_value"].sum()),
        "stages": stages,
        "won": won,
        "lost_or_dead": lost,
        "win_rate_pct": round(won / decided * 100, 1) if decided else None,
        "caveats": ["Win rate counts only deals that reached a decision "
                    f"({decided} of {len(df)} deals); the rest are still open or on hold."]
        if decided else ["No closed deals in scope, so win rate is not meaningful."],
    }


def revenue_summary(data: Dataset, sector=None, owner=None, period=None,
                    start_date=None, end_date=None, date_field="po_date",
                    group_by="sector") -> dict:
    start, end, period_label = resolve_period(period, start_date, end_date, data.as_of)
    scoped = _apply_filters(data.work_orders, sector=sector, owner=owner)
    df, undated = _filter_dates(scoped, date_field, start, end)

    def total(column):
        return _money(df[column].sum()) if column in df else None

    order_value = total("order_value_ex_gst") or 0.0
    billed = total("billed_ex_gst") or 0.0
    billed_inc = total("billed_inc_gst") or 0.0
    collected = total("collected_inc_gst") or 0.0
    receivable = total("receivable") or 0.0

    caveats = _empty_period_note(scoped, df, date_field, start, end)
    if "billed_ex_gst" in df and len(df):
        known = int(df["billed_ex_gst"].notna().sum())
        if known < len(df):
            caveats.append(
                f"Billed value (excl. GST) is blank on {len(df)-known} of {len(df)} work "
                "orders; billed totals are a floor, not a certainty."
            )
    if "collected_inc_gst" in df and len(df):
        known = int(df["collected_inc_gst"].notna().sum())
        caveats.append(f"Collections are recorded on {known} of {len(df)} work orders.")
    if undated:
        caveats.append(f"{undated} work orders have no {date_field.replace('_',' ')} "
                       "and fall outside the period filter.")
    if "is_overbilled" in df:
        over = int(df["is_overbilled"].fillna(False).sum())
        if over:
            caveats.append(f"{over} work orders are billed above order value (negative "
                           "amount-to-bill) - likely scope additions not reflected in the PO.")

    return {
        "period": period_label,
        "filters": {"sector": sector, "owner": owner, "date_field": date_field},
        "work_orders": int(len(df)),
        "order_book_ex_gst": order_value,
        "billed_ex_gst": billed,
        "billed_inc_gst": billed_inc,
        "collected_inc_gst": collected,
        "outstanding_receivable": receivable,
        "unbilled_ex_gst": _money(df["to_bill_ex_gst"].sum()) if "to_bill_ex_gst" in df else None,
        "billed_pct_of_order_book": round(billed / order_value * 100, 1) if order_value else None,
        "collected_pct_of_billed": round(collected / billed_inc * 100, 1) if billed_inc else None,
        "breakdown": _breakdown(df, group_by, "order_value_ex_gst")[:15],
        "caveats": caveats,
    }


def delivery_summary(data: Dataset, sector=None, owner=None, period=None,
                     start_date=None, end_date=None, group_by="execution_status") -> dict:
    start, end, period_label = resolve_period(period, start_date, end_date, data.as_of)
    scoped = _apply_filters(data.work_orders, sector=sector, owner=owner)
    df, undated = _filter_dates(scoped, "start_date", start, end)

    overdue = df[df.get("is_overdue", False) == True] if "is_overdue" in df else df.iloc[0:0]  # noqa: E712
    status_counts = _breakdown(df, "execution_status", "order_value_ex_gst")

    caveats = _empty_period_note(scoped, df, "start_date", start, end)
    if undated:
        caveats.append(f"{undated} work orders have no probable start date and were "
                       "excluded from the period filter.")
    if "execution_status" in df:
        missing = int(df["execution_status"].isna().sum())
        if missing:
            caveats.append(f"{missing} work orders have no execution status.")
    caveats.append("'Overdue' means the probable end date has passed and execution status "
                   "is not Completed.")

    return {
        "period": period_label,
        "work_orders": int(len(df)),
        "by_status": status_counts,
        "breakdown": _breakdown(df, group_by, "order_value_ex_gst")[:15],
        "overdue_count": int(len(overdue)),
        "overdue_value_ex_gst": _money(overdue["order_value_ex_gst"].sum()) if len(overdue) else 0.0,
        "overdue_examples": overdue.sort_values("order_value_ex_gst", ascending=False)
        [["work_order_id", "deal_name", "sector", "end_date", "execution_status",
          "order_value_ex_gst"]].head(10).astype(str).to_dict("records") if len(overdue) else [],
        "caveats": caveats,
    }


def receivables_summary(data: Dataset, sector=None, owner=None,
                        min_amount: float = 0.0) -> dict:
    df = data.work_orders
    df = _apply_filters(df, sector=sector, owner=owner)
    if "receivable" not in df:
        return {"error": "No receivable column found on the work orders board.",
                "caveats": []}
    outstanding = df[df["receivable"].fillna(0) > max(min_amount, 0)]

    today = data.as_of
    def bucket(row):
        reference = row.get("last_invoice_date") or row.get("po_date")
        if not isinstance(reference, date):
            return "No invoice date"
        days = (today - reference).days
        for limit, label in ((30, "0-30 days"), (60, "31-60 days"),
                             (90, "61-90 days"), (180, "91-180 days")):
            if days <= limit:
                return label
        return "180+ days"

    aging: dict[str, dict] = {}
    for _, row in outstanding.iterrows():
        key = bucket(row)
        entry = aging.setdefault(key, {"work_orders": 0, "amount": 0.0})
        entry["work_orders"] += 1
        entry["amount"] = round(entry["amount"] + float(row["receivable"] or 0), 2)

    priority = outstanding[outstanding.get("ar_priority").notna()] \
        if "ar_priority" in outstanding else outstanding.iloc[0:0]

    return {
        "as_of": today.isoformat(),
        "work_orders_with_receivable": int(len(outstanding)),
        "total_receivable": _money(outstanding["receivable"].sum()),
        "aging": aging,
        "priority_accounts": int(len(priority)),
        "priority_receivable": _money(priority["receivable"].sum()) if len(priority) else 0.0,
        "top_exposures": outstanding.sort_values("receivable", ascending=False)
        [["work_order_id", "deal_name", "customer_code", "sector", "receivable",
          "last_invoice_date", "invoice_status"]].head(10).astype(str).to_dict("records"),
        "caveats": [
            "Aging is measured from last invoice date, falling back to PO date where no "
            "invoice date is recorded - it is an approximation, not a ledger.",
            "Collection status, collection date and actual collection month are empty on "
            "the board, so nothing here is reconciled against actual receipts.",
        ],
    }


def sector_performance(data: Dataset, sectors=None) -> dict:
    """Both boards side by side: what is being sold vs what is being delivered."""
    deals = _apply_filters(data.deals, sector=sectors)
    work = _apply_filters(data.work_orders, sector=sectors)
    rows = {}
    for name, chunk in deals.groupby("sector", dropna=False):
        key = "Unspecified" if pd.isna(name) else str(name)
        open_chunk = chunk[chunk.get("is_open", False) == True]  # noqa: E712
        rows.setdefault(key, {})["sector"] = key
        rows[key].update({
            "open_deals": int(len(open_chunk)),
            "open_pipeline_value": _money(open_chunk["deal_value"].sum()),
            "won_deals": int(chunk.get("is_won", pd.Series(dtype=bool)).sum()),
        })
    for name, chunk in work.groupby("sector", dropna=False):
        key = "Unspecified" if pd.isna(name) else str(name)
        rows.setdefault(key, {"sector": key, "open_deals": 0,
                              "open_pipeline_value": 0.0, "won_deals": 0})
        rows[key].update({
            "work_orders": int(len(chunk)),
            "order_book_ex_gst": _money(chunk["order_value_ex_gst"].sum()),
            "billed_ex_gst": _money(chunk["billed_ex_gst"].sum()),
            "receivable": _money(chunk["receivable"].sum()),
        })
    ordered = sorted(rows.values(),
                     key=lambda r: (r.get("order_book_ex_gst") or 0) +
                                   (r.get("open_pipeline_value") or 0), reverse=True)
    return {
        "sectors": ordered,
        "caveats": ["Deals and work orders are joined on sector, not on a shared key: the "
                    "boards have no reliable common identifier (deal names are masked and "
                    "repeat across rows), so this is a sector-level comparison, not a "
                    "deal-to-work-order match."],
    }


def list_records(data: Dataset, board: str = "deals", sector=None, owner=None,
                 stage=None, status=None, sort_by=None, descending=True,
                 limit: int = 10, columns=None) -> dict:
    df = data.deals if board == "deals" else data.work_orders
    df = _apply_filters(df, sector=sector, owner=owner, stage=stage, status=status)
    default_sort = "deal_value" if board == "deals" else "order_value_ex_gst"
    sort_column = sort_by if sort_by in df.columns else default_sort
    if sort_column in df:
        df = df.sort_values(sort_column, ascending=not descending, na_position="last")

    default_columns = (["deal_name", "client_code", "sector", "deal_stage", "deal_status",
                        "deal_value", "closure_probability", "tentative_close_date", "owner_code"]
                       if board == "deals" else
                       ["work_order_id", "deal_name", "customer_code", "sector",
                        "type_of_work", "execution_status", "order_value_ex_gst",
                        "billed_ex_gst", "receivable", "end_date"])
    use = [c for c in (columns or default_columns) if c in df.columns]
    limit = max(1, min(int(limit), MAX_ROWS))
    return {
        "board": board,
        "matched": int(len(df)),
        "returned": min(limit, len(df)),
        "sorted_by": sort_column,
        "records": df[use].head(limit).astype(str).to_dict("records"),
        "caveats": [] if len(df) <= limit else
        [f"Showing {limit} of {len(df)} matching records."],
    }


def leadership_brief(data: Dataset, period: str = "this quarter") -> dict:
    """Pre-packaged board/leadership update: pipeline, delivery, cash, risks."""
    pipeline = pipeline_summary(data, period=period, status="Open")
    all_open = pipeline_summary(data, period=None, status="Open")
    funnel = funnel_snapshot(data)
    revenue = revenue_summary(data, period=None)
    delivery = delivery_summary(data, period=None)
    receivables = receivables_summary(data)
    sectors = sector_performance(data)

    risks = []
    if delivery["overdue_count"]:
        risks.append(f"{delivery['overdue_count']} work orders past their probable end date "
                     f"({delivery['overdue_value_ex_gst']:,.0f} INR of order value).")
    if receivables.get("total_receivable"):
        old = receivables["aging"].get("180+ days", {})
        if old.get("amount"):
            risks.append(f"{old['amount']:,.0f} INR receivable is more than 180 days past "
                         "its last invoice date.")
    stale = int(data.deals.get("is_stale", pd.Series(dtype=bool)).fillna(False).sum())
    if stale:
        risks.append(f"{stale} open deals have a tentative close date in the past - funnel "
                     "hygiene issue before any forecast is credible.")

    return {
        "period": pipeline["period"],
        "as_of": data.as_of.isoformat(),
        "pipeline_closing_in_period": {
            "deals": pipeline["deal_count"],
            "value": pipeline["total_value"],
            "weighted_value": pipeline["weighted_value"],
        },
        "total_open_pipeline": {
            "deals": all_open["deal_count"],
            "value": all_open["total_value"],
            "top_sectors": all_open["breakdown"][:5],
        },
        "funnel": {"win_rate_pct": funnel["win_rate_pct"], "won": funnel["won"],
                   "lost_or_dead": funnel["lost_or_dead"],
                   "stages": funnel["stages"][:8]},
        "delivery": {"work_orders": delivery["work_orders"],
                     "by_status": delivery["by_status"],
                     "overdue": delivery["overdue_count"]},
        "cash": {"order_book_ex_gst": revenue["order_book_ex_gst"],
                 "billed_ex_gst": revenue["billed_ex_gst"],
                 "collected_inc_gst": revenue["collected_inc_gst"],
                 "receivable": receivables.get("total_receivable"),
                 "collected_pct_of_billed": revenue["collected_pct_of_billed"]},
        "sector_view": sectors["sectors"][:6],
        "risks": risks,
        "caveats": sorted(set(pipeline["caveats"] + revenue["caveats"]
                              + receivables["caveats"]))[:6],
    }
