# Skylark BI Agent

A conversational business intelligence agent over two monday.com boards —
**Work Orders** (project execution) and **Deals** (sales pipeline). It answers
founder-level questions in plain language, computes every figure from live
monday.com data, and states the data-quality caveats that sit behind each
number.

Built for the Skylark Drones technical assignment.

---

## What it does

Ask *"How's our pipeline looking for the energy sector this quarter?"* and the
agent will:

1. Fetch both boards live from monday.com via GraphQL (never a cached export)
2. Normalise the messy fields — dates in mixed formats, `5360 HA` quantities,
   `BIlled` vs `Billed`, repeated header rows pasted mid-data
3. Recognise that **"energy" is not a sector in the data** and map it to
   Renewables + Powerline, then say so in the answer
4. Compute the numbers in pandas
5. Have Gemini write the narrative — over the computed figures only

The LLM never sees a raw board row and never produces a number.

---

## Architecture

```
                 ┌────────────────────────────────────────────┐
  monday.com ───►│ monday_client.py   GraphQL, cursor paging,  │
   (2 boards)    │                    retries, 401/429 handling│
                 └────────────────┬───────────────────────────┘
                                  │ rows keyed by column TITLE
                 ┌────────────────▼───────────────────────────┐
                 │ normalize.py    resilience layer            │
                 │  · drops header-echo + duplicate rows       │
                 │  · multi-format date + currency parsing     │
                 │  · canonical status/sector vocabularies     │
                 │  · emits a DataQualityReport                │
                 └────────────────┬───────────────────────────┘
                                  │ two tidy DataFrames
   question ─────► agent.py ──────┤
                   (plan)         │
                      │           ▼
                      │   ┌──────────────────────────────────┐
                      │   │ analytics.py   9 named analyses, │
                      └──►│ all arithmetic in pandas, each   │
                          │ result carries its own caveats   │
                          └───────────────┬──────────────────┘
                                          │ AnalysisResult
                          agent.py (narrate) ──► answer + table
                                          │
                                      app.py (Streamlit)
```

| File | Role |
|---|---|
| `src/monday_client.py` | Read-only GraphQL client. Cursor pagination, exponential backoff on 429/5xx, explicit messages for auth and complexity errors. |
| `src/normalize.py` | Turns untrusted board text into two clean DataFrames plus a `DataQualityReport`. Never invents a value. |
| `src/analytics.py` | Nine deterministic analyses. Each returns metrics, an optional table, and its own caveats. |
| `src/agent.py` | Gemini plans (question → analysis + filters) and narrates (figures → prose). Keyword fallback for both if the LLM is unavailable. |
| `src/config.py` | Secrets and the Indian fiscal-year calendar. |
| `app.py` | Streamlit UI: chat, leadership update tab, data-quality sidebar. |
| `data_prep/prepare_monday_csvs.py` | One-time import prep. **Not used at runtime.** |
| `tests_offline.py` | Runs the full normalize → analyse → answer path against the prepared CSVs, so the logic is verifiable without a token. |

---

## Setup

### 1. monday.com

1. Create a free monday.com account.
2. **Import the boards.** In the prepared folder `data_prep/out/` there are two
   CSVs ready to import. In monday.com: *Add → Import data → Excel/CSV*, upload
   `work_orders_monday.csv`, name the board **Work Orders**. Repeat with
   `deals_monday.csv` as **Deals**.
   - Set the first column as the item name when prompted.
   - Recommended column types are in the table below; monday's auto-detection
     gets most of them right and the agent reads column *text* regardless, so a
     wrong type degrades display, not correctness.
3. **Get the board IDs.** Open each board; the URL ends in
   `/boards/1234567890` — that number is the board ID.
4. **Get an API token.** Profile picture → *Developers* → *API token* → copy.
   Admins can also find it under *Administration → Connections → Personal API
   token*. API access works on the free plan; the daily call limit there is
   1,000 requests, which is ample since the agent caches for 5 minutes.

Recommended column types:

| Column group | Type |
|---|---|
| Serial #, Deal Name, Client/Customer Code, Owner code | Text |
| Sector, Nature of Work, Execution Status, Invoice Status, Deal Status, Deal Stage, Closure Probability, Document Type | Status or Dropdown |
| All `... Date` columns | Date |
| All `Amount ...`, `Masked Deal value`, `Quantity ...` columns | Numbers |
| Type of Work | Dropdown, multi-select |

### 2. Gemini key

Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).
The app defaults to `gemini-2.5-flash`, which is on the free tier.

### 3. Run it

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# fill in the four values
streamlit run app.py
```

### 4. Deploy (hosted prototype)

1. Push this folder to a GitHub repo. `.gitignore` already excludes
   `secrets.toml` — check it never gets committed.
2. Go to [share.streamlit.io](https://share.streamlit.io), *New app*, point it
   at the repo, main file `app.py`.
3. *Advanced settings → Secrets*: paste the contents of your
   `secrets.toml`.
4. Deploy. The resulting `*.streamlit.app` URL is the hosted prototype — it
   needs no local setup to test.

---

## Verifying without monday.com

```bash
python tests_offline.py
```

This feeds the prepared CSVs through the same normalise → analyse → answer path
the live client uses, with the LLM disabled, and prints every analysis. It is
how the logic was validated during development.

---

## A note on dates in the sample data

Deal and work-order activity in the supplied dataset runs from roughly
**August 2024 to April 2026**. Since "this quarter" resolves against the real
current date, a genuine *this quarter* question can correctly return nothing.
The agent says so explicitly and reports the date range the data actually
covers, rather than returning a silent zero. To demo against a populated
quarter, set `TODAY_OVERRIDE` in secrets.

## Scope

Read-only, as specified. The client issues only `boards` and `items_page`
queries; there is no mutation anywhere in the codebase.
