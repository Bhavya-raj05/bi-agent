"""
The agent: query understanding and narration.

Two LLM calls per question, with a deterministic fallback for both:

  1. PLAN  - turn the founder's question into {analysis, filters} JSON, or ask
             a clarifying question if the question is genuinely ambiguous.
  2. NARRATE - turn the computed AnalysisResult into prose.

The LLM never sees raw board rows and never produces a figure. It sees the
question (for planning) and the computed metrics table (for narration). If
Gemini is unavailable, a keyword planner and a plain formatter take over, so
the app degrades rather than dies.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime

from .analytics import ANALYSIS_REGISTRY, AnalysisResult, Filters, run_analysis
from .config import today

try:
    # Current unified SDK. The older `google-generativeai` package reached
    # end-of-life in Nov 2025, so this project uses `google-genai`.
    from google import genai
except Exception:  # pragma: no cover
    genai = None


VALID_PERIODS = [
    "current_quarter", "last_quarter", "current_fy", "last_fy", "ytd", "all",
]


@dataclass
class Plan:
    analysis: str
    filters: Filters
    board: str = "deals"
    clarifying_question: str | None = None
    interpretation: str = ""


PLANNER_PROMPT = """You are the query planner for a business intelligence agent \
used by the founders of Skylark Drones, a drone services company in India.

You convert a business question into a JSON plan. You never answer the question \
yourself and you never state any figure.

Available analyses:
{analyses}

Data you can filter on:
- sectors (Work Orders board): Mining, Renewables, Railways, Powerline, Construction, Others
- sectors (Deals board): the above plus Tender, DSP, Security and Surveillance, Aviation, Manufacturing
- deal statuses: Open, Won, Lost, On Hold
- execution statuses: Completed, Partially Completed, Ongoing, Not Started, Paused, Blocked
- owners: OWNER_001 through OWNER_008

Periods use the INDIAN FISCAL YEAR (April to March). Q1 is Apr-Jun. \
Today is {today}.

Return ONLY a JSON object, no markdown fences, with these keys:
{{
  "analysis": one of {analysis_names},
  "board": "deals" or "work_orders" (only matters for top_records),
  "filters": {{
     "sector_term": the user's own sector word if they used one, else null,
     "owners": [],
     "statuses": [],
     "clients": [],
     "period": one of {periods},
     "top_n": integer, default 10
  }},
  "interpretation": one short sentence describing how you read the question,
  "clarifying_question": a question to ask the user, or null
}}

Rules:
- Put the user's own sector wording in sector_term verbatim (e.g. "energy"). \
The system maps it to real sector values and tells the user how it mapped.
- Only set clarifying_question when the question cannot be planned at all. \
If a reasonable default exists, use the default and say so in interpretation. \
Prefer answering over asking.
- If no period is stated, use "all".
- "This quarter" means the current Indian fiscal quarter.
- Questions about money already earned or invoiced -> revenue_summary. \
Questions about future or in-flight deals -> pipeline_health.
- Requests for a board update, investor update, monthly review or summary \
for leadership -> leadership_update.

Question: {question}
"""


NARRATOR_PROMPT = """You are a business intelligence analyst briefing the \
founders of Skylark Drones, an Indian drone services company.

Below is a computed analysis. Every figure in it was calculated \
deterministically from live monday.com data.

Write a concise answer to the founder's question:
- Lead with the direct answer in one or two sentences.
- Then give the two or three numbers that matter most. Use the figures exactly \
as given; never recalculate, never round differently, never invent a number.
- Add one line of interpretation: what this suggests or what to watch.
- Finish with the data caveats, plainly, under a short heading. Do not soften \
them. If a caveat says coverage is partial, make clear the figure is a floor, \
not a total.
- No preamble. No markdown headings above level 3. Under 200 words.

Founder's question: {question}

{analysis}
"""


class Agent:
    def __init__(self, api_key: str = "", model: str = "gemini-2.5-flash"):
        self.model_name = model
        self.llm = None
        if api_key and genai is not None:
            try:
                self.llm = genai.Client(api_key=api_key)
            except Exception:
                self.llm = None

    def _generate(self, prompt: str) -> str:
        response = self.llm.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return (response.text or "").strip()

    @property
    def llm_available(self) -> bool:
        return self.llm is not None

    # --- planning ----------------------------------------------------------

    def plan(self, question: str) -> Plan:
        if self.llm is None:
            return self._fallback_plan(question)

        prompt = PLANNER_PROMPT.format(
            analyses="\n".join(f"- {k}: {v}" for k, v in ANALYSIS_REGISTRY.items()),
            analysis_names=list(ANALYSIS_REGISTRY),
            periods=VALID_PERIODS,
            today=today().isoformat(),
            question=question,
        )
        try:
            payload = _extract_json(self._generate(prompt))
        except Exception:
            return self._fallback_plan(question)

        if not payload:
            return self._fallback_plan(question)

        analysis = payload.get("analysis")
        if analysis not in ANALYSIS_REGISTRY:
            return self._fallback_plan(question)

        raw_filters = payload.get("filters") or {}
        period = raw_filters.get("period")
        if period not in VALID_PERIODS:
            period = "all"

        filters = Filters(
            sector_term=raw_filters.get("sector_term") or None,
            owners=_as_list(raw_filters.get("owners")),
            statuses=_as_list(raw_filters.get("statuses")),
            clients=_as_list(raw_filters.get("clients")),
            period=period,
            top_n=int(raw_filters.get("top_n") or 10),
        )
        return Plan(
            analysis=analysis,
            filters=filters,
            board=payload.get("board") or "deals",
            clarifying_question=payload.get("clarifying_question") or None,
            interpretation=payload.get("interpretation") or "",
        )

    def _fallback_plan(self, question: str) -> Plan:
        """Keyword planner used when Gemini is unavailable or returns junk."""
        text = question.lower()

        if any(w in text for w in ("board update", "leadership", "investor",
                                   "monthly update", "summary for", "brief")):
            analysis = "leadership_update"
        elif any(w in text for w in ("receivable", "owed", "owe us", "outstanding",
                                     "collect", "ar ", "priority account", "unpaid")):
            analysis = "receivables"
        elif any(w in text for w in ("late", "overdue", "behind schedule", "slip",
                                     "delayed", "execution", "delivery", "ongoing",
                                     "operational", "ops ", "stuck", "blocked")):
            analysis = "operations_summary"
        elif "pipeline" in text or "funnel" in text:
            analysis = "pipeline_health"
        elif any(w in text for w in ("revenue", "billed", "billing", "invoice",
                                     "order book", "earned", "collected")):
            analysis = "revenue_summary"
        elif any(w in text for w in ("owner", "bd ", "kam", "rep ", "salesperson")):
            analysis = "owner_performance"
        elif any(w in text for w in ("top ", "largest", "biggest", "list ")):
            analysis = "top_records"
        elif any(w in text for w in ("win rate", "sector", "vertical", "segment",
                                     "compare", "performance")):
            analysis = "sector_performance"
        elif any(w in text for w in ("won deal", "conversion", "delivery gap")):
            analysis = "deal_to_delivery"
        else:
            analysis = "pipeline_health"

        if "last quarter" in text:
            period = "last_quarter"
        elif "this quarter" in text or "current quarter" in text:
            period = "current_quarter"
        elif "this year" in text or "ytd" in text or "year to date" in text:
            period = "ytd"
        elif "last year" in text:
            period = "last_fy"
        else:
            period = "all"

        sector_term = None
        for word in ("energy", "mining", "powerline", "power", "renewable",
                     "railway", "rail", "construction", "solar", "aviation",
                     "manufacturing", "surveillance", "tender"):
            if word in text:
                sector_term = word
                break

        board = "work_orders" if "work order" in text else "deals"
        top_match = re.search(r"top\s+(\d{1,3})", text)
        top_n = int(top_match.group(1)) if top_match else 10

        return Plan(
            analysis=analysis,
            filters=Filters(sector_term=sector_term, period=period, top_n=top_n),
            board=board,
            interpretation="Planned without the language model (keyword match).",
        )

    # --- narration ---------------------------------------------------------

    def narrate(self, question: str, result: AnalysisResult) -> str:
        if self.llm is None:
            return _plain_answer(result)
        try:
            text = self._generate(
                NARRATOR_PROMPT.format(
                    question=question, analysis=result.to_prompt_text()
                )
            )
            return text or _plain_answer(result)
        except Exception:
            return _plain_answer(result)

    # --- end to end --------------------------------------------------------

    def answer(self, question: str, deals, work_orders) -> tuple[str, AnalysisResult | None, Plan]:
        plan = self.plan(question)
        if plan.clarifying_question:
            return plan.clarifying_question, None, plan

        result = run_analysis(
            plan.analysis, deals, work_orders, plan.filters, board=plan.board
        )
        return self.narrate(question, result), result, plan


# --- helpers ---------------------------------------------------------------


def _as_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _plain_answer(result: AnalysisResult) -> str:
    """Deterministic formatting used when the LLM is unavailable."""
    lines = [f"**{result.title}**"]
    if result.period_label:
        lines.append(f"Period: {result.period_label}")
    lines.append("")
    for key, value in result.metrics.items():
        lines.append(f"- {key}: **{value}**")
    if result.caveats:
        lines.append("")
        lines.append("**Data caveats**")
        for caveat in result.caveats:
            lines.append(f"- {caveat}")
    lines.append("")
    lines.append("_Answer generated without the language model._")
    return "\n".join(lines)
