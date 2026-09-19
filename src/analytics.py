"""
Analytics layer.

Every number the agent reports is computed here, in pandas. The LLM chooses
*which* analysis to run and with what filters; it never does arithmetic. That
keeps figures reproducible and removes the main hallucination surface.

Each analysis returns an AnalysisResult carrying its own caveats, so coverage
warnings travel with the number instead of being bolted on afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from .config import (
    fiscal_quarter_bounds,
    fiscal_quarter_of,
    fiscal_year_of,
    fy_label,
    today,
)
from .normalize import resolve_sectors

OPEN_STAGE_MAX = 7  # stages A-G are pre-win; H onward means work order received


@dataclass
class Filters:
    sectors: list[str] = field(default_factory=list)
    sector_term: str | None = None
    owners: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    clients: list[str] = field(default_factory=list)
    period: str | None = None          # "current_quarter" | "last_quarter" | "fy" | "all"
    date_from: date | None = None
    date_to: date | None = None
    top_n: int = 10

    def resolved_sectors(self) -> list[str]:
        if self.sectors:
            out: list[str] = []
            for s in self.sectors:
                out.extend(resolve_sectors(s))
            return sorted(set(out))
        if self.sector_term:
            return resolve_sectors(self.sector_term)
        return []


@dataclass
class AnalysisResult:
    title: str
    metrics: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    caveats: list[str] = field(default_factory=list)
    period_label: str | None = None

    def to_prompt_text(self) -> str:
        lines = [f"ANALYSIS: {self.title}"]
        if self.period_label:
            lines.append(f"PERIOD: {self.period_label}")
        for key, value in self.metrics.items():
            lines.append(f"  {key}: {value}")
        if self.table is not None and not self.table.empty:
            lines.append("TABLE:")
            lines.append(self.table.head(25).to_string(index=False))
        for caveat in self.caveats:
            lines.append(f"CAVEAT: {caveat}")
        return "\n".join(lines)


# --- helpers ---------------------------------------------------------------


def rupees(amount: float | None) -> str:
    """Indian-format currency, in crore/lakh since founder figures are large."""
    if amount is None or pd.isna(amount):
        return "n/a"
    sign = "-" if amount < 0 else ""
    amount = abs(float(amount))
    if amount >= 1_00_00_000:
        return f"{sign}Rs {amount / 1_00_00_000:.2f} Cr"
    if amount >= 1_00_000:
        return f"{sign}Rs {amount / 1_00_000:.2f} L"
    return f"{sign}Rs {amount:,.0f}"


def resolve_period(period: str | None, reference: date | None = None) -> tuple[date | None, date | None, str]:
    reference = reference or today()
    fy = fiscal_year_of(reference)
    quarter = fiscal_quarter_of(reference)

    if period in (None, "all"):
        return None, None, "all time"
    if period == "current_quarter":
        start, end = fiscal_quarter_bounds(fy, quarter)
        return start, end, f"Q{quarter} {fy_label(fy)} ({start} to {end})"
    if period == "last_quarter":
        prev_q = quarter - 1 or 4
        prev_fy = fy if quarter > 1 else fy - 1
        start, end = fiscal_quarter_bounds(prev_fy, prev_q)
        return start, end, f"Q{prev_q} {fy_label(prev_fy)} ({start} to {end})"
    if period == "current_fy":
        start, _ = fiscal_quarter_bounds(fy, 1)
        _, end = fiscal_quarter_bounds(fy, 4)
        return start, end, f"{fy_label(fy)} ({start} to {end})"
    if period == "last_fy":
        start, _ = fiscal_quarter_bounds(fy - 1, 1)
        _, end = fiscal_quarter_bounds(fy - 1, 4)
        return start, end, f"{fy_label(fy - 1)} ({start} to {end})"
    if period == "ytd":
        start, _ = fiscal_quarter_bounds(fy, 1)
        return start, reference, f"{fy_label(fy)} to date ({start} to {reference})"
    return None, None, "all time"


def _apply_common(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    out = df
    sectors = filters.resolved_sectors()
    if sectors and "sector" in out.columns:
        out = out[out["sector"].isin(sectors)]
    if filters.owners and "owner" in out.columns:
        out = out[out["owner"].isin(filters.owners)]
    if filters.clients:
        col = "client" if "client" in out.columns else "customer"
        if col in out.columns:
            out = out[out[col].isin(filters.clients)]
    return out


def _apply_dates(df: pd.DataFrame, column: str, start: date | None, end: date | None) -> tuple[pd.DataFrame, int]:
    """Filter by date range. Returns (filtered, rows_excluded_for_missing_date)."""
    if start is None and end is None:
        return df, 0
    missing = int(df[column].isna().sum())
    out = df[df[column].notna()]
    if start is not None:
        out = out[out[column] >= start]
    if end is not None:
        out = out[out[column] <= end]
    return out, missing


def _coverage_caveat(df: pd.DataFrame, column: str, label: str) -> str | None:
    if df.empty:
        return None
    known = int(df[column].notna().sum())
    total = len(df)
    if known == total:
        return None
    pct = known / total if total else 0
    return (
        f"{total - known} of {total} records have no {label}; "
        f"figures below cover the {known} that do ({pct:.0%})."
    )


def _date_range_caveat(df: pd.DataFrame, column: str, label: str) -> str | None:
    """When a period filter empties the result, say what the data actually covers.

    The supplied dataset runs to early 2026, so a genuine 'this quarter'
    question can legitimately match nothing. Saying so beats returning a
    silent zero.
    """
    dates = df[column].dropna()
    if dates.empty:
        return f"No record on this board carries a usable {label}."
    return (
        f"Nothing falls in that window. The board's {label} values run from "
        f"{min(dates)} to {max(dates)}, so the period you asked about sits "
        f"outside the data."
    )


# --- analyses --------------------------------------------------------------


def pipeline_health(deals: pd.DataFrame, filters: Filters) -> AnalysisResult:
    start, end, label = resolve_period(filters.period)
    scoped = _apply_common(deals, filters)

    date_column = "tentative_close_date"
    scoped, missing_dates = _apply_dates(scoped, date_column, start, end)

    open_deals = scoped[scoped["status"] == "Open"]
    result = AnalysisResult(
        title="Pipeline health",
        period_label=label,
        metrics={
            "deals in scope": len(scoped),
            "open deals": len(open_deals),
            "won deals": int((scoped["status"] == "Won").sum()),
            "lost deals": int((scoped["status"] == "Lost").sum()),
            "on hold": int((scoped["status"] == "On Hold").sum()),
            "open pipeline value (where known)": rupees(open_deals["deal_value"].sum()),
            "weighted pipeline (High=0.8, Med=0.5, Low=0.2)": rupees(
                _weighted_value(open_deals)
            ),
        },
    )

    if not open_deals.empty:
        by_stage = (
            open_deals.groupby("stage_label", dropna=False)
            .agg(deals=("deal_name", "size"), value=("deal_value", "sum"))
            .sort_values("deals", ascending=False)
            .reset_index()
        )
        by_stage["value"] = by_stage["value"].map(rupees)
        result.table = by_stage

    if scoped.empty and (start or end):
        note = _date_range_caveat(deals, date_column, "tentative close date")
        if note:
            result.caveats.append(note)
    caveat = _coverage_caveat(open_deals, "deal_value", "deal value")
    if caveat:
        result.caveats.append(caveat)
    if missing_dates:
        result.caveats.append(
            f"{missing_dates} deals have no tentative close date and were excluded "
            f"from the {label} window."
        )
    if filters.sector_term and filters.resolved_sectors():
        result.caveats.append(
            f"'{filters.sector_term}' is not a sector value in the data; "
            f"read as {', '.join(filters.resolved_sectors())}."
        )
    return result


def _weighted_value(deals: pd.DataFrame) -> float:
    weights = {"High": 0.8, "Medium": 0.5, "Low": 0.2}
    subset = deals[deals["deal_value"].notna()]
    if subset.empty:
        return 0.0
    return float(
        sum(
            row.deal_value * weights.get(row.probability, 0.3)
            for row in subset.itertuples()
        )
    )


def sector_performance(deals: pd.DataFrame, work_orders: pd.DataFrame,
                       filters: Filters) -> AnalysisResult:
    start, end, label = resolve_period(filters.period)
    scoped_deals, _ = _apply_dates(_apply_common(deals, filters),
                                   "created_date", start, end)
    scoped_wo, _ = _apply_dates(_apply_common(work_orders, filters),
                                "po_date", start, end)

    deal_side = (
        scoped_deals.groupby("sector", dropna=False)
        .agg(
            deals=("deal_name", "size"),
            won=("status", lambda s: int((s == "Won").sum())),
            lost=("status", lambda s: int((s == "Lost").sum())),
            pipeline_value=("deal_value", "sum"),
        )
    )
    wo_side = (
        scoped_wo.groupby("sector", dropna=False)
        .agg(work_orders=("work_order_id", "size"), order_value=("order_value", "sum"))
    )
    merged = deal_side.join(wo_side, how="outer").fillna(0).reset_index()
    merged["win_rate"] = merged.apply(
        lambda r: f"{r['won'] / (r['won'] + r['lost']):.0%}"
        if (r["won"] + r["lost"]) > 0 else "n/a",
        axis=1,
    )
    merged["pipeline_value"] = merged["pipeline_value"].map(rupees)
    merged["order_value"] = merged["order_value"].map(rupees)
    merged = merged.sort_values("deals", ascending=False)

    result = AnalysisResult(
        title="Sector performance (deals and work orders)",
        period_label=label,
        table=merged,
        metrics={"sectors covered": len(merged)},
    )
    if scoped_deals.empty and scoped_wo.empty and (start or end):
        note = _date_range_caveat(deals, "created_date", "deal created date")
        if note:
            result.caveats.append(note)
    caveat = _coverage_caveat(scoped_deals, "deal_value", "deal value")
    if caveat:
        result.caveats.append(caveat)
    result.caveats.append(
        "Sector labels differ across boards: Deals also uses 'Tender' and 'DSP', "
        "which describe a route to market rather than an industry."
    )
    return result


def revenue_summary(work_orders: pd.DataFrame, filters: Filters) -> AnalysisResult:
    start, end, label = resolve_period(filters.period)
    scoped = _apply_common(work_orders, filters)
    scoped, missing = _apply_dates(scoped, "po_date", start, end)

    booked = scoped["order_value"].sum()
    billed = scoped["billed_value"].sum()
    collected = scoped["collected_amount"].sum()
    outstanding = scoped["to_be_billed"].sum()

    result = AnalysisResult(
        title="Revenue and billing",
        period_label=label,
        metrics={
            "work orders in scope": len(scoped),
            "order book (excl GST)": rupees(booked),
            "billed to date (excl GST)": rupees(billed),
            "collected (incl GST)": rupees(collected),
            "yet to bill (excl GST)": rupees(outstanding),
            "billing ratio": f"{billed / booked:.0%}" if booked else "n/a",
        },
    )

    by_sector = (
        scoped.groupby("sector", dropna=False)
        .agg(orders=("work_order_id", "size"), booked=("order_value", "sum"),
             billed=("billed_value", "sum"))
        .sort_values("booked", ascending=False)
        .reset_index()
    )
    by_sector["booked"] = by_sector["booked"].map(rupees)
    by_sector["billed"] = by_sector["billed"].map(rupees)
    result.table = by_sector

    if scoped.empty and (start or end):
        note = _date_range_caveat(work_orders, "po_date", "PO date")
        if note:
            result.caveats.append(note)
    for column, human in [
        ("billed_value", "billed value"),
        ("collected_amount", "collection figure"),
    ]:
        caveat = _coverage_caveat(scoped, column, human)
        if caveat:
            result.caveats.append(caveat)
    if missing:
        result.caveats.append(
            f"{missing} work orders have no PO date and fall outside any period filter."
        )
    negative = scoped[scoped["to_be_billed"] < 0]
    if not negative.empty:
        result.caveats.append(
            f"{len(negative)} work orders show a negative amount-to-bill, i.e. billed "
            f"above order value. Treated as over-billing, not cleaned away."
        )
    return result


def receivables(work_orders: pd.DataFrame, filters: Filters) -> AnalysisResult:
    scoped = _apply_common(work_orders, filters)
    outstanding = scoped[scoped["receivable"].fillna(0) > 0]

    result = AnalysisResult(
        title="Receivables",
        metrics={
            "work orders with a receivable": len(outstanding),
            "total receivable": rupees(outstanding["receivable"].sum()),
            "flagged AR priority": int(outstanding["ar_priority"].sum()),
            "priority receivable": rupees(
                outstanding[outstanding["ar_priority"]]["receivable"].sum()
            ),
        },
    )

    if not outstanding.empty:
        table = (
            outstanding.sort_values("receivable", ascending=False)
            .head(filters.top_n)[
                ["work_order_id", "deal_name", "customer", "sector",
                 "receivable", "last_invoice_date", "ar_priority"]
            ]
            .copy()
        )
        table["receivable"] = table["receivable"].map(rupees)
        result.table = table

    result.caveats.append(
        "Receivable is taken as reported on the board; there is no payment-due-date "
        "column, so true ageing buckets cannot be computed."
    )
    return result


def operations_summary(work_orders: pd.DataFrame, filters: Filters) -> AnalysisResult:
    start, end, label = resolve_period(filters.period)
    scoped = _apply_common(work_orders, filters)
    scoped, _ = _apply_dates(scoped, "po_date", start, end)

    by_status = (
        scoped.groupby("execution_status", dropna=False)
        .agg(work_orders=("work_order_id", "size"), value=("order_value", "sum"))
        .sort_values("work_orders", ascending=False)
        .reset_index()
    )
    by_status["value"] = by_status["value"].map(rupees)

    reference = today()
    overdue = scoped[
        scoped["end_date"].notna()
        & (scoped["end_date"] < reference)
        & (~scoped["execution_status"].isin(["Completed"]))
    ]

    result = AnalysisResult(
        title="Operational status",
        period_label=label,
        table=by_status,
        metrics={
            "work orders in scope": len(scoped),
            "past probable end date and not completed": len(overdue),
            "value at risk in those": rupees(overdue["order_value"].sum()),
            "blocked on client": int((scoped["execution_status"] == "Blocked").sum()),
            "paused": int((scoped["execution_status"] == "Paused").sum()),
        },
    )
    if scoped.empty and (start or end):
        note = _date_range_caveat(work_orders, "po_date", "PO date")
        if note:
            result.caveats.append(note)
    caveat = _coverage_caveat(scoped, "end_date", "probable end date")
    if caveat:
        result.caveats.append(caveat)
    result.caveats.append(
        "'Probable end date' is a plan date, not a contractual deadline, so overdue "
        "here means schedule slip rather than breach."
    )
    return result


def deal_to_delivery(deals: pd.DataFrame, work_orders: pd.DataFrame,
                     filters: Filters) -> AnalysisResult:
    """Cross-board view: won deals against work orders actually raised."""
    scoped_deals = _apply_common(deals, filters)
    won = scoped_deals[scoped_deals["status"] == "Won"]
    scoped_wo = _apply_common(work_orders, filters)

    wo_names = set(scoped_wo["deal_name"].dropna().astype(str).str.strip().str.lower())
    won_names = won["deal_name"].dropna().astype(str).str.strip().str.lower()
    matched = int(won_names.isin(wo_names).sum())

    result = AnalysisResult(
        title="Won deals vs work orders raised",
        metrics={
            "won deals": len(won),
            "won deals with a matching work order": matched,
            "won deals with no matching work order": len(won) - matched,
            "work orders in scope": len(scoped_wo),
            "won deal value (where known)": rupees(won["deal_value"].sum()),
            "work order book": rupees(scoped_wo["order_value"].sum()),
        },
    )
    result.caveats.append(
        "The two boards share no join key. Matching is on masked deal name only, "
        "and those names repeat across clients, so treat the match count as "
        "indicative rather than exact."
    )
    caveat = _coverage_caveat(won, "deal_value", "deal value")
    if caveat:
        result.caveats.append(caveat)
    return result


def top_records(deals: pd.DataFrame, work_orders: pd.DataFrame,
                filters: Filters, board: str = "deals") -> AnalysisResult:
    if board == "work_orders":
        scoped = _apply_common(work_orders, filters)
        if filters.statuses:
            scoped = scoped[scoped["execution_status"].isin(filters.statuses)]
        table = (
            scoped.sort_values("order_value", ascending=False)
            .head(filters.top_n)[
                ["work_order_id", "deal_name", "customer", "sector",
                 "execution_status", "order_value", "po_date"]
            ]
            .copy()
        )
        table["order_value"] = table["order_value"].map(rupees)
        title = "Largest work orders"
    else:
        scoped = _apply_common(deals, filters)
        if filters.statuses:
            scoped = scoped[scoped["status"].isin(filters.statuses)]
        table = (
            scoped.sort_values("deal_value", ascending=False)
            .head(filters.top_n)[
                ["deal_name", "client", "owner", "sector", "status",
                 "stage_label", "deal_value", "tentative_close_date"]
            ]
            .copy()
        )
        table["deal_value"] = table["deal_value"].map(rupees)
        title = "Largest deals"

    result = AnalysisResult(title=title, table=table,
                            metrics={"records shown": len(table)})
    value_col = "order_value" if board == "work_orders" else "deal_value"
    caveat = _coverage_caveat(scoped, value_col, "value")
    if caveat:
        result.caveats.append(
            caveat + " Records without a value cannot be ranked and are omitted."
        )
    return result


def owner_performance(deals: pd.DataFrame, filters: Filters) -> AnalysisResult:
    start, end, label = resolve_period(filters.period)
    scoped, _ = _apply_dates(_apply_common(deals, filters), "created_date", start, end)

    table = (
        scoped.groupby("owner", dropna=False)
        .agg(
            deals=("deal_name", "size"),
            won=("status", lambda s: int((s == "Won").sum())),
            lost=("status", lambda s: int((s == "Lost").sum())),
            open_deals=("status", lambda s: int((s == "Open").sum())),
            value=("deal_value", "sum"),
        )
        .reset_index()
    )
    table["win_rate"] = table.apply(
        lambda r: f"{r['won'] / (r['won'] + r['lost']):.0%}"
        if (r["won"] + r["lost"]) > 0 else "n/a",
        axis=1,
    )
    table["value"] = table["value"].map(rupees)
    table = table.sort_values("deals", ascending=False)

    result = AnalysisResult(title="Performance by BD owner", period_label=label,
                            table=table, metrics={"owners": len(table)})
    result.caveats.append(
        "Owner codes are masked and one code may cover more than one person."
    )
    return result


def leadership_update(deals: pd.DataFrame, work_orders: pd.DataFrame,
                      filters: Filters) -> AnalysisResult:
    """Composite pack for a board or investor update."""
    filters.period = filters.period or "current_quarter"
    pipeline = pipeline_health(deals, filters)
    revenue = revenue_summary(work_orders, filters)
    ops = operations_summary(work_orders, filters)
    ar = receivables(work_orders, filters)
    sectors = sector_performance(deals, work_orders, filters)

    merged_metrics: dict[str, Any] = {}
    for part in (pipeline, revenue, ops, ar):
        for key, value in part.metrics.items():
            merged_metrics[f"{part.title} - {key}"] = value

    result = AnalysisResult(
        title="Leadership update pack",
        period_label=pipeline.period_label,
        metrics=merged_metrics,
        table=sectors.table,
    )
    seen = set()
    for part in (pipeline, revenue, ops, ar, sectors):
        for caveat in part.caveats:
            if caveat not in seen:
                seen.add(caveat)
                result.caveats.append(caveat)
    return result


ANALYSIS_REGISTRY = {
    "pipeline_health": "Open pipeline: counts, value, weighted value, stage breakdown.",
    "sector_performance": "Deals and work orders compared across sectors, with win rates.",
    "revenue_summary": "Order book, billed, collected, yet-to-bill, by sector.",
    "receivables": "Outstanding receivables and AR priority accounts.",
    "operations_summary": "Execution status, schedule slip, blocked and paused work.",
    "deal_to_delivery": "Won deals matched against work orders actually raised.",
    "top_records": "Ranked list of the largest deals or work orders.",
    "owner_performance": "Deal counts, win rate and value by BD/KAM owner.",
    "leadership_update": "Composite pack: pipeline, revenue, ops and AR together.",
}


def run_analysis(name: str, deals: pd.DataFrame, work_orders: pd.DataFrame,
                 filters: Filters, board: str = "deals") -> AnalysisResult:
    if name == "pipeline_health":
        return pipeline_health(deals, filters)
    if name == "sector_performance":
        return sector_performance(deals, work_orders, filters)
    if name == "revenue_summary":
        return revenue_summary(work_orders, filters)
    if name == "receivables":
        return receivables(work_orders, filters)
    if name == "operations_summary":
        return operations_summary(work_orders, filters)
    if name == "deal_to_delivery":
        return deal_to_delivery(deals, work_orders, filters)
    if name == "top_records":
        return top_records(deals, work_orders, filters, board=board)
    if name == "owner_performance":
        return owner_performance(deals, filters)
    if name == "leadership_update":
        return leadership_update(deals, work_orders, filters)
    raise ValueError(f"Unknown analysis: {name}")
