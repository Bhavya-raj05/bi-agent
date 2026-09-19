"""Tool definitions handed to Claude, and the dispatcher that runs them."""
from __future__ import annotations

import json
from typing import Any, Callable

from . import analytics
from .store import Dataset

_SECTOR = {
    "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
    "description": "Sector filter. Accepts board values (Mining, Powerline, Renewables, "
                   "Railways, Construction, Tender, DSP, Aviation, Manufacturing, "
                   "Security & Surveillance, Others) and loose words like 'energy', which "
                   "expands to Powerline + Renewables.",
}
_PERIOD = {
    "type": "string",
    "description": "Relative period: 'this quarter', 'last quarter', 'next quarter', "
                   "'this fiscal year', 'this month', 'last 30/60/90 days', or 'all'. "
                   "Fiscal quarters run April-March.",
}
_OWNER = {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
          "description": "Owner code(s) such as OWNER_003."}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "describe_data",
        "description": "List the fields and the actual distinct values present on both "
                       "boards. Call this first whenever you are unsure what a filter value "
                       "should be, instead of guessing a sector or stage name.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "data_quality_report",
        "description": "Row counts, duplicates removed, unrecognised columns, sparse and "
                       "empty fields. Use when the user asks how reliable the data is, or "
                       "when an answer hinges on a field you suspect is sparse.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "pipeline_summary",
        "description": "Sales pipeline from the Deals board: deal counts, total and "
                       "probability-weighted value, breakdowns. The main tool for 'how's "
                       "our pipeline' questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": _SECTOR,
                "owner": _OWNER,
                "stage": {"anyOf": [{"type": "string"},
                                    {"type": "array", "items": {"type": "string"}}],
                          "description": "Funnel stage name or fragment, e.g. 'Negotiations'."},
                "status": {"type": "string",
                           "description": "Deal status: Open, Won, Dead, On Hold. "
                                          "Defaults to Open. Pass null for all."},
                "period": _PERIOD,
                "start_date": {"type": "string", "description": "ISO date, overrides period."},
                "end_date": {"type": "string", "description": "ISO date, overrides period."},
                "date_field": {"type": "string",
                               "enum": ["tentative_close_date", "created_date",
                                        "close_date_actual"],
                               "description": "Which date the period filters on. Use "
                                              "tentative_close_date for 'closing this "
                                              "quarter', created_date for 'new deals'."},
                "group_by": {"type": "string",
                             "enum": ["sector", "deal_stage", "owner_code", "product",
                                      "closure_probability", "client_code"]},
            },
        },
    },
    {
        "name": "funnel_snapshot",
        "description": "Stage-by-stage view of open deals plus win rate. Use for funnel "
                       "health and conversion questions.",
        "input_schema": {"type": "object",
                         "properties": {"sector": _SECTOR, "owner": _OWNER}},
    },
    {
        "name": "revenue_summary",
        "description": "Money from the Work Orders board: order book, billed, collected, "
                       "unbilled and receivable. Use for revenue, billing and collections "
                       "questions. Note this is execution/billing data, not booked P&L.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": _SECTOR, "owner": _OWNER, "period": _PERIOD,
                "start_date": {"type": "string"}, "end_date": {"type": "string"},
                "date_field": {"type": "string",
                               "enum": ["po_date", "last_invoice_date", "end_date",
                                        "data_delivery_date"],
                               "description": "po_date for when work was ordered, "
                                              "last_invoice_date for billing activity."},
                "group_by": {"type": "string",
                             "enum": ["sector", "type_of_work", "nature_of_work",
                                      "owner_code", "customer_code", "platform",
                                      "invoice_status"]},
            },
        },
    },
    {
        "name": "delivery_summary",
        "description": "Operational execution from the Work Orders board: status mix, "
                       "overdue projects and their value. Use for 'how is delivery going' "
                       "or 'what is slipping'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": _SECTOR, "owner": _OWNER, "period": _PERIOD,
                "start_date": {"type": "string"}, "end_date": {"type": "string"},
                "group_by": {"type": "string",
                             "enum": ["execution_status", "sector", "type_of_work",
                                      "nature_of_work", "owner_code", "customer_code"]},
            },
        },
    },
    {
        "name": "receivables_summary",
        "description": "Outstanding receivables with an aging profile and the largest "
                       "exposures. Use for cash, collections and AR questions.",
        "input_schema": {
            "type": "object",
            "properties": {"sector": _SECTOR, "owner": _OWNER,
                           "min_amount": {"type": "number"}},
        },
    },
    {
        "name": "sector_performance",
        "description": "Cross-board comparison: open pipeline from Deals against order "
                       "book, billing and receivables from Work Orders, by sector. Use when "
                       "the question spans selling and delivering.",
        "input_schema": {"type": "object", "properties": {"sectors": _SECTOR}},
    },
    {
        "name": "list_records",
        "description": "Return individual deals or work orders matching filters. Use when "
                       "the user asks 'which ones' or wants named examples.",
        "input_schema": {
            "type": "object",
            "properties": {
                "board": {"type": "string", "enum": ["deals", "work_orders"]},
                "sector": _SECTOR, "owner": _OWNER,
                "stage": {"type": "string"}, "status": {"type": "string"},
                "sort_by": {"type": "string",
                            "description": "Field to sort on, e.g. deal_value, receivable."},
                "descending": {"type": "boolean"},
                "limit": {"type": "integer", "description": "Max 25."},
            },
            "required": ["board"],
        },
    },
    {
        "name": "leadership_brief",
        "description": "One composite call that assembles a leadership/board update: "
                       "pipeline, funnel health, delivery, cash position, sector view and "
                       "risks. Use when the user asks for an update, a summary for the "
                       "board, or 'what should I tell leadership'.",
        "input_schema": {"type": "object", "properties": {"period": _PERIOD}},
    },
]

_DISPATCH: dict[str, Callable] = {
    "describe_data": analytics.describe_data,
    "data_quality_report": analytics.data_quality_report,
    "pipeline_summary": analytics.pipeline_summary,
    "funnel_snapshot": analytics.funnel_snapshot,
    "revenue_summary": analytics.revenue_summary,
    "delivery_summary": analytics.delivery_summary,
    "receivables_summary": analytics.receivables_summary,
    "sector_performance": analytics.sector_performance,
    "list_records": analytics.list_records,
    "leadership_brief": analytics.leadership_brief,
}


def run_tool(name: str, arguments: dict, data: Dataset) -> str:
    """Execute a tool and return a JSON string for the model.

    Tool failures are returned as data, not raised: the model can then explain
    the limitation to the user instead of the whole turn collapsing.
    """
    function = _DISPATCH.get(name)
    if function is None:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    try:
        result = function(data, **{k: v for k, v in (arguments or {}).items()
                                   if v is not None})
    except TypeError as exc:
        return json.dumps({"error": f"Bad arguments for {name}: {exc}"})
    except Exception as exc:                      # noqa: BLE001 - surfaced to the model
        return json.dumps({"error": f"{name} failed: {type(exc).__name__}: {exc}"})
    return json.dumps(result, default=str)
