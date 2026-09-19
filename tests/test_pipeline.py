"""
Offline tests. They build monday.com-shaped payloads from the import CSVs and
run them through the real client/store/analytics path, so the cleaning and
analytics layers can be verified without a token or a network call.

    python -m pytest tests/ -q      (or just: python tests/test_pipeline.py)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import analytics, normalize as nz                       # noqa: E402
from agent.monday_client import RawBoard                           # noqa: E402
from agent.schema import DEALS_SCHEMA, WORK_ORDERS_SCHEMA          # noqa: E402
from agent.store import (Dataset, _finalize_deals,                 # noqa: E402
                         _finalize_work_orders, _rows_from_board, _coverage)

DATA = Path(__file__).resolve().parent.parent / "data"
TODAY = date(2026, 9, 19)


def fake_board(csv_path: Path, name: str) -> RawBoard:
    """Shape a CSV like a monday.com board response: item name + text cells."""
    frame = pd.read_csv(csv_path, dtype=str).fillna("")
    titles = list(frame.columns)
    columns = [{"id": f"col{i}", "title": t, "type": "text"}
               for i, t in enumerate(titles[1:], start=1)]
    items = []
    for index, row in frame.iterrows():
        items.append({
            "id": str(index),
            "name": row[titles[0]],
            "column_values": [{"id": f"col{i}", "text": row[t], "type": "text"}
                              for i, t in enumerate(titles[1:], start=1)],
        })
    return RawBoard(board_id="0", name=name, columns=columns, items=items)


def build_offline_dataset() -> Dataset:
    deals_raw = fake_board(DATA / "deals_monday_import.csv", "Deals")
    work_raw = fake_board(DATA / "work_orders_monday_import.csv", "Work Orders")
    deal_rows, deal_diag = _rows_from_board(deals_raw, DEALS_SCHEMA)
    work_rows, work_diag = _rows_from_board(work_raw, WORK_ORDERS_SCHEMA)
    deals = _finalize_deals(pd.DataFrame(deal_rows), TODAY)
    work = _finalize_work_orders(pd.DataFrame(work_rows), TODAY)
    quality = {
        "as_of": TODAY.isoformat(),
        "deals": {"board_name": "Deals", "rows_fetched": len(deals_raw.items),
                  "rows_usable": len(deals), "duplicates_removed": len(deal_rows) - len(deals),
                  "coverage": _coverage(deals), **deal_diag},
        "work_orders": {"board_name": "Work Orders", "rows_fetched": len(work_raw.items),
                        "rows_usable": len(work), "duplicates_removed": len(work_rows) - len(work),
                        "coverage": _coverage(work), **work_diag},
    }
    return Dataset(deals=deals, work_orders=work, quality=quality, as_of=TODAY)


# ------------------------------------------------------------------ units

def test_parse_quantity_units():
    assert nz.parse_quantity("5360 HA") == (5360.0, "hectare")
    assert nz.parse_quantity("3956HA") == (3956.0, "hectare")
    assert nz.parse_quantity("2057 Acr")[1] == "hectare"
    assert nz.parse_quantity("24 Months") == (24.0, "month")
    assert nz.parse_quantity("600") == (600.0, None)
    assert nz.parse_quantity("") == (None, None)


def test_parse_money():
    assert nz.parse_number("₹ 1,23,456.78") == 123456.78
    assert nz.parse_number("(500)") == -500.0
    assert nz.parse_number("2.5 Cr") == 25000000.0
    assert nz.parse_number("n/a") is None


def test_fiscal_calendar():
    assert nz.fiscal_year(date(2026, 2, 1)) == 2025
    assert nz.fiscal_quarter(date(2026, 9, 19)) == 2
    start, end = nz.fiscal_quarter_bounds(date(2026, 9, 19))
    assert (start, end) == (date(2026, 7, 1), date(2026, 9, 30))


def test_stage_and_sector():
    assert nz.parse_stage("E. Proposal/Commercials Sent") == ("Proposal/Commercials Sent", 5)
    assert nz.canonical_sector("mining") == "Mining"
    assert set(nz.expand_sector_query("energy")) == {"Powerline", "Renewables"}


# ------------------------------------------------------------------ pipeline

def test_dataset_loads_and_cleans():
    data = build_offline_dataset()
    assert len(data.deals) > 300
    assert len(data.work_orders) > 150
    assert data.quality["deals"]["duplicates_removed"] > 0     # 12 dupes in source
    assert not data.deals["sector"].isin(["Sector/service"]).any()


def test_tools_return_numbers_and_caveats():
    data = build_offline_dataset()
    for result in (
        analytics.pipeline_summary(data, sector="energy", period="this quarter"),
        analytics.revenue_summary(data),
        analytics.delivery_summary(data),
        analytics.receivables_summary(data),
        analytics.sector_performance(data),
        analytics.funnel_snapshot(data),
        analytics.leadership_brief(data),
        analytics.data_quality_report(data),
        analytics.describe_data(data),
        analytics.list_records(data, board="work_orders", limit=5),
    ):
        assert isinstance(result, dict) and result


if __name__ == "__main__":
    data = build_offline_dataset()
    print(f"deals={len(data.deals)} work_orders={len(data.work_orders)}")
    for name, fn in [
        ("energy pipeline this quarter",
         lambda: analytics.pipeline_summary(data, sector="energy", period="this quarter")),
        ("all open pipeline", lambda: analytics.pipeline_summary(data)),
        ("revenue", lambda: analytics.revenue_summary(data)),
        ("delivery", lambda: analytics.delivery_summary(data)),
        ("receivables", lambda: analytics.receivables_summary(data)),
        ("leadership brief", lambda: analytics.leadership_brief(data)),
    ]:
        print("\n===", name)
        import json
        print(json.dumps(fn(), indent=2, default=str)[:1600])
