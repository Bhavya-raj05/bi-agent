"""Offline harness: feeds the prepared CSVs through the same normalize +
analytics path the live monday.com client would, so the logic is verified
against the real data without needing an API token."""
import pandas as pd
from src.normalize import normalize_work_orders, normalize_deals
from src.analytics import Filters, run_analysis, ANALYSIS_REGISTRY
from src.agent import Agent

def rows(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.to_dict("records")

wo, wo_rep = normalize_work_orders(rows("data_prep/out/work_orders_monday.csv"))
dl, dl_rep = normalize_deals(rows("data_prep/out/deals_monday.csv"))
print("work orders:", wo.shape, "| deals:", dl.shape)
for r in (wo_rep, dl_rep):
    for line in r.summary_lines(): print("  ", line)

agent = Agent(api_key="")  # forces fallback planner
qs = ["How's our pipeline looking for the energy sector this quarter?",
      "What's our total order book and how much have we billed?",
      "Which sector has the best win rate?",
      "How much are we owed and who are the priority accounts?",
      "Which work orders are running late?",
      "Give me a leadership update",
      "Top 5 largest deals",
      "How are our BD owners doing?"]
for q in qs:
    print("\n" + "="*70); print("Q:", q)
    ans, res, plan = agent.answer(q, dl, wo)
    print("-> analysis:", plan.analysis, "| period:", plan.filters.period, "| sector:", plan.filters.sector_term)
    print(ans[:900])
