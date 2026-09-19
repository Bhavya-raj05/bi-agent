"""
Maps monday.com column titles onto a stable canonical schema.

The agent never depends on monday column IDs (which differ per workspace) or on
exact titles (which people rename). It matches on a normalized title key with an
alias list, so 'Sector', 'sector/service' and 'Sector / Service' all land on the
same canonical field. Unmapped columns are kept under their raw title and
reported by the data-quality tool, so nothing is silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from .normalize import (
    canonical_deal_status,
    canonical_label,
    canonical_sector,
    clean_text,
    parse_date,
    parse_number,
    parse_quantity,
    title_key,
)

Kind = Literal["text", "label", "number", "money", "date", "sector", "status",
               "stage", "quantity", "month"]


@dataclass(frozen=True)
class Field:
    name: str                      # canonical field name
    kind: Kind
    aliases: tuple[str, ...]       # human titles this field may appear under
    description: str = ""
    parser: Callable | None = None


def _f(name, kind, *aliases, description="", parser=None):
    return Field(name, kind, tuple(aliases), description, parser)


DEAL_FIELDS: list[Field] = [
    _f("deal_name", "text", "Deal Name", "Name", "Deal", description="Masked deal/account name"),
    _f("owner_code", "label", "Owner code", "BD/KAM Personnel code", "Owner",
       description="Masked BD/KAM owner, e.g. OWNER_001"),
    _f("client_code", "label", "Client Code", "Customer Code", "Account Code",
       description="Masked client, e.g. COMPANY089"),
    _f("deal_status", "status", "Deal Status", "Status",
       description="Open / Won / Dead / On Hold"),
    _f("close_date_actual", "date", "Close Date (A)", "Actual Close Date", "Closed On"),
    _f("closure_probability", "label", "Closure Probability", "Probability", "Confidence",
       description="High / Medium / Low"),
    _f("deal_value", "money", "Masked Deal value", "Deal Value", "Value", "Amount",
       description="Deal value in INR (masked)"),
    _f("tentative_close_date", "date", "Tentative Close Date", "Expected Close Date",
       "Close Date"),
    _f("deal_stage", "stage", "Deal Stage", "Stage",
       description="Lettered funnel stage, A. Lead Generated ... O. Not Relevant"),
    _f("product", "label", "Product deal", "Product", "Product Mix"),
    _f("sector", "sector", "Sector/service", "Sector", "Sector / Service", "Industry"),
    _f("created_date", "date", "Created Date", "Create Date", "Created"),
]

WORK_ORDER_FIELDS: list[Field] = [
    _f("deal_name", "text", "Deal name masked", "Deal Name", "Name"),
    _f("customer_code", "label", "Customer Name Code", "Customer Code", "Client Code"),
    _f("work_order_id", "text", "Serial #", "Serial No", "WO ID", "Serial",
       description="Work-order identifier, e.g. SDPLDEAL-075"),
    _f("nature_of_work", "label", "Nature of Work",
       description="One time Project / Monthly Contract / Annual Rate Contract / PoC"),
    _f("last_executed_month", "month", "Last executed month of recurring project"),
    _f("execution_status", "label", "Execution Status",
       description="Completed / Ongoing / Not Started / Partial Completed / Pause / struck"),
    _f("data_delivery_date", "date", "Data Delivery Date", "Delivery Date"),
    _f("po_date", "date", "Date of PO/LOI", "PO Date", "Date of PO"),
    _f("document_type", "label", "Document Type",
       description="Purchase Order / LOA/LOI / Email Confirmation"),
    _f("start_date", "date", "Probable Start Date", "Start Date"),
    _f("end_date", "date", "Probable End Date", "End Date"),
    _f("owner_code", "label", "BD/KAM Personnel code", "Owner code", "Owner"),
    _f("sector", "sector", "Sector", "Sector/service", "Industry"),
    _f("type_of_work", "label", "Type of Work", "Work Type", "Service"),
    _f("platform", "label",
       "Is any Skylark software platform part of the client deliverables in this deal?",
       "Skylark Platform", "Platform",
       description="NONE / SPECTRA / DMO / SPECTRA + DMO"),
    _f("last_invoice_date", "date", "Last invoice date", "Invoice Date"),
    _f("last_invoice_no", "text", "latest invoice no.", "Invoice No", "Latest Invoice No"),
    _f("order_value_ex_gst", "money", "Amount in Rupees (Excl of GST) (Masked)",
       "Order Value Excl GST", "Amount Excl GST"),
    _f("order_value_inc_gst", "money", "Amount in Rupees (Incl of GST) (Masked)",
       "Order Value Incl GST", "Amount Incl GST"),
    _f("billed_ex_gst", "money", "Billed Value in Rupees (Excl of GST.) (Masked)",
       "Billed Excl GST"),
    _f("billed_inc_gst", "money", "Billed Value in Rupees (Incl of GST.) (Masked)",
       "Billed Incl GST"),
    _f("collected_inc_gst", "money",
       "Collected Amount in Rupees (Incl of GST.) (Masked)", "Collected"),
    _f("to_bill_ex_gst", "money", "Amount to be billed in Rs. (Exl. of GST) (Masked)"),
    _f("to_bill_inc_gst", "money", "Amount to be billed in Rs. (Incl. of GST) (Masked)"),
    _f("receivable", "money", "Amount Receivable (Masked)", "Receivable", "AR"),
    _f("ar_priority", "label", "AR Priority account", "AR Priority"),
    _f("quantity_ops", "number", "Quantity by Ops"),
    _f("quantity_po", "quantity", "Quantities as per PO", "Quantity as per PO"),
    _f("quantity_billed", "number", "Quantity billed (till date)"),
    _f("quantity_balance", "number", "Balance in quantity"),
    _f("invoice_status", "label", "Invoice Status"),
    _f("expected_billing_month", "month", "Expected Billing Month"),
    _f("actual_billing_month", "month", "Actual Billing Month"),
    _f("actual_collection_month", "month", "Actual Collection Month"),
    _f("wo_status", "label", "WO Status (billed)", "WO Status"),
    _f("collection_status", "label", "Collection status"),
    _f("collection_date", "date", "Collection Date"),
    _f("billing_status", "label", "Billing Status"),
]


@dataclass
class BoardSchema:
    key: str
    fields: list[Field]
    lookup: dict[str, Field] = field(init=False)

    def __post_init__(self):
        self.lookup = {}
        for f in self.fields:
            for alias in (f.name, *f.aliases):
                self.lookup.setdefault(title_key(alias), f)

    def match(self, column_title: str) -> Field | None:
        return self.lookup.get(title_key(column_title))

    def describe(self) -> list[dict]:
        return [{"field": f.name, "type": f.kind, "about": f.description}
                for f in self.fields]


DEALS_SCHEMA = BoardSchema("deals", DEAL_FIELDS)
WORK_ORDERS_SCHEMA = BoardSchema("work_orders", WORK_ORDER_FIELDS)


_PARSERS = {
    "text": clean_text,
    "label": canonical_label,
    "number": parse_number,
    "money": parse_number,
    "date": parse_date,
    "sector": canonical_sector,
    "status": canonical_deal_status,
}


def coerce(field_obj: Field, raw):
    """Apply the parser for a field's kind. 'stage', 'quantity' and 'month' are
    handled in store.py because they produce more than one output column."""
    parser = field_obj.parser or _PARSERS.get(field_obj.kind)
    return parser(raw) if parser else clean_text(raw)
