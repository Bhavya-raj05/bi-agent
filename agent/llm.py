"""The conversational layer: Claude plans, the tools compute, Claude narrates."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

import anthropic

from .config import settings
from .store import Dataset
from .tools import TOOL_SPECS, run_tool

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the business intelligence agent for Skylark Drones. You answer questions \
from founders and executives using two live monday.com boards: a Deals board (sales \
funnel) and a Work Orders board (project execution, billing and collections).

How to work:
- Never compute numbers yourself. Every figure you state must come from a tool result \
in this conversation. If a tool did not give you a number, say you do not have it.
- Call `describe_data` before guessing what a filter value should be. Sector, stage and \
status values are controlled vocabularies.
- Prefer one well-chosen tool call over many. For a general update, `leadership_brief` \
already assembles the whole picture.
- If a question genuinely cannot be narrowed down (two plausible readings that give \
different answers), ask one short clarifying question. Otherwise state your \
interpretation in a clause and answer - founders want the answer, not an interview.

Data reality you must respect:
- The fiscal year runs April to March. "This quarter" means the current fiscal quarter.
- Money is in INR and is masked, so absolute values are directionally useful but not \
audited. Values are excluding GST unless the field name says otherwise.
- Many fields are sparsely populated. Tool results carry a `caveats` list. If a caveat \
materially affects the answer, state it in one short sentence - do not dump the list.
- Deal values are missing on roughly half of deals. Any pipeline total is a floor. Say so \
when it matters.
- The two boards have no reliable shared key, so never claim a specific deal became a \
specific work order.

How to answer:
- Lead with the number and what it means. Then one or two lines of context: what is \
driving it, what is concentrated where, what looks off.
- Use plain prose. Short tables only when comparing more than three things. Format INR \
readably (₹4.2 Cr, ₹38.5 L) with the raw figure in brackets where precision matters.
- Flag anomalies you notice - stale close dates, over-billing, concentration risk - even \
when not asked. That judgement is why a founder is talking to you instead of a filter.
- Never invent a field, a client, or a trend the data does not support.
"""


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    error: str | None = None


class BIAgent:
    def __init__(self, dataset: Dataset, model: str | None = None):
        self.dataset = dataset
        self.model = model or settings.model
        self.client = anthropic.Anthropic(api_key=settings.anthropic_key)

    def _context_note(self) -> str:
        q = self.dataset.quality
        return (f"Today is {self.dataset.as_of.isoformat()}. Boards loaded: "
                f"{q['deals']['rows_usable']} deals, "
                f"{q['work_orders']['rows_usable']} work orders.")

    def ask(self, question: str, history: list[dict] | None = None) -> AgentReply:
        """Run the tool loop until the model produces a final answer."""
        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": question})
        used: list[dict] = []

        for _ in range(settings.max_tool_turns):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=SYSTEM_PROMPT + "\n\n" + self._context_note(),
                    tools=TOOL_SPECS,
                    messages=messages,
                )
            except anthropic.APIStatusError as exc:
                return AgentReply(
                    text="I could not reach the language model just now. The monday.com "
                         "data is loaded, so please retry in a moment.",
                    tool_calls=used, error=str(exc))
            except Exception as exc:                       # noqa: BLE001
                return AgentReply(text="Something went wrong generating the answer.",
                                  tool_calls=used, error=str(exc))

            blocks = response.content
            tool_uses = [b for b in blocks if b.type == "tool_use"]
            if not tool_uses:
                text = "\n".join(b.text for b in blocks if b.type == "text").strip()
                messages.append({"role": "assistant", "content": blocks})
                return AgentReply(text=text or "(no answer produced)", tool_calls=used)

            messages.append({"role": "assistant", "content": blocks})
            results = []
            for call in tool_uses:
                log.info("tool %s %s", call.name, call.input)
                payload = run_tool(call.name, call.input, self.dataset)
                used.append({"name": call.name, "input": call.input,
                             "output": json.loads(payload)})
                results.append({"type": "tool_result", "tool_use_id": call.id,
                                "content": payload})
            messages.append({"role": "user", "content": results})

        return AgentReply(
            text="That question needed more analysis steps than I allow in one turn. "
                 "Try narrowing it - one sector, one period, one metric.",
            tool_calls=used)


def conversation_history(turns: list[dict], limit: int = 6) -> list[dict]:
    """Trim UI history to plain user/assistant text turns for the next request."""
    trimmed = [t for t in turns if t.get("role") in ("user", "assistant") and t.get("content")]
    return [{"role": t["role"], "content": t["content"]} for t in trimmed[-limit:]]
