"""
Skylark BI Agent - Streamlit front end.

Run locally:   streamlit run app.py
Hosted:        Streamlit Community Cloud, secrets set in the dashboard.
"""

from __future__ import annotations

import traceback

import pandas as pd
import streamlit as st

from src.agent import Agent
from src.analytics import Filters, leadership_update
from src.config import load_settings
from src.monday_client import MondayClient, MondayError
from src.normalize import normalize_deals, normalize_work_orders

st.set_page_config(page_title="Skylark BI Agent", page_icon="📊", layout="wide")

SAMPLE_QUESTIONS = [
    "How's our pipeline looking for the energy sector this quarter?",
    "What's our total order book and how much of it have we billed?",
    "Which sector has the best win rate?",
    "How much are we owed right now, and who are the priority accounts?",
    "Which work orders are running late?",
    "Give me a leadership update for this quarter.",
]


@st.cache_data(ttl=300, show_spinner=False)
def load_boards(token: str, wo_board: str, deals_board: str):
    """Fetch and normalise both boards. Cached for 5 minutes."""
    client = MondayClient(token)
    wo_rows = client.fetch_board_rows(wo_board)
    deal_rows = client.fetch_board_rows(deals_board)
    work_orders, wo_report = normalize_work_orders(wo_rows)
    deals, deal_report = normalize_deals(deal_rows)
    return work_orders, deals, wo_report, deal_report


def main():
    settings = load_settings()

    st.title("Skylark BI Agent")
    st.caption(
        "Ask a business question. Answers are computed live from monday.com - "
        "no cached exports, no hardcoded data."
    )

    if not settings.monday_api_token or not settings.work_orders_board_id \
            or not settings.deals_board_id:
        st.error("monday.com is not configured. Missing: " + ", ".join(settings.missing()))
        st.info(
            "Set these in `.streamlit/secrets.toml` locally, or in the app's "
            "Secrets panel on Streamlit Community Cloud. See README.md."
        )
        st.stop()

    try:
        with st.spinner("Loading boards from monday.com..."):
            work_orders, deals, wo_report, deal_report = load_boards(
                settings.monday_api_token,
                settings.work_orders_board_id,
                settings.deals_board_id,
            )
    except MondayError as exc:
        st.error(f"Could not reach monday.com.\n\n{exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error loading data: {exc}")
        st.code(traceback.format_exc())
        st.stop()

    if work_orders.empty and deals.empty:
        st.warning("Both boards came back empty. Check the board IDs.")
        st.stop()

    agent = Agent(settings.gemini_api_key, settings.gemini_model)

    with st.sidebar:
        st.subheader("Connection")
        st.success(f"Work Orders: {len(work_orders)} rows")
        st.success(f"Deals: {len(deals)} rows")
        if agent.llm_available:
            st.success(f"Gemini: {settings.gemini_model}")
        else:
            st.warning(
                "No Gemini key - running in deterministic mode. "
                "Analyses still work; answers are formatted, not written."
            )
        if st.button("Refresh from monday.com"):
            st.cache_data.clear()
            st.rerun()

        st.subheader("Data quality")
        for report in (wo_report, deal_report):
            with st.expander(report.board, expanded=False):
                for line in report.summary_lines():
                    st.write("- " + line)
                if report.unmapped_values:
                    st.caption("Unrecognised values kept as-is:")
                    for field_name, values in report.unmapped_values.items():
                        st.caption(f"{field_name}: {', '.join(sorted(values)[:6])}")

    tab_chat, tab_update, tab_data = st.tabs(["Ask", "Leadership update", "Underlying data"])

    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.write("**Try:**")
        cols = st.columns(3)
        for index, question in enumerate(SAMPLE_QUESTIONS):
            if cols[index % 3].button(question, key=f"sample_{index}"):
                st.session_state.pending = question

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("table") is not None:
                    st.dataframe(message["table"], use_container_width=True)

        typed = st.chat_input("Ask about pipeline, revenue, delivery or receivables...")
        question = typed or st.session_state.pop("pending", None)

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            table = None
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer, result, plan = agent.answer(question, deals, work_orders)
                    except Exception as exc:
                        answer = (
                            f"I hit an error running that analysis: {exc}\n\n"
                            "Try rephrasing, or pick one of the sample questions."
                        )
                        result, plan = None, None

                st.markdown(answer)
                table = result.table if result is not None else None
                if table is not None and not table.empty:
                    st.dataframe(table, use_container_width=True)
                if plan is not None and plan.interpretation:
                    st.caption(f"Read as: {plan.interpretation}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "table": table}
            )

    with tab_update:
        st.subheader("Leadership update")
        st.caption(
            "Interpreted as: a quarter-scoped pack covering pipeline, revenue, "
            "delivery and receivables, with every data caveat stated up front, "
            "ready to paste into a board email."
        )
        period = st.selectbox(
            "Period", ["current_quarter", "last_quarter", "ytd", "current_fy", "all"], index=0
        )
        if st.button("Generate"):
            result = leadership_update(deals, work_orders, Filters(period=period))
            st.markdown(f"### {result.title} - {result.period_label}")
            metrics = pd.DataFrame(
                {"Metric": list(result.metrics), "Value": list(result.metrics.values())}
            )
            st.dataframe(metrics, use_container_width=True, hide_index=True)
            if result.table is not None:
                st.markdown("**By sector**")
                st.dataframe(result.table, use_container_width=True, hide_index=True)
            st.markdown("**Data caveats**")
            for caveat in result.caveats:
                st.markdown(f"- {caveat}")
            if agent.llm_available:
                with st.spinner("Writing the narrative..."):
                    st.markdown("**Narrative**")
                    st.markdown(
                        agent.narrate("Write a leadership update for the period.", result)
                    )

    with tab_data:
        st.caption("Normalised view of what the agent is reasoning over.")
        st.markdown("**Work Orders**")
        st.dataframe(work_orders, use_container_width=True)
        st.markdown("**Deals**")
        st.dataframe(deals, use_container_width=True)


if __name__ == "__main__":
    main()
