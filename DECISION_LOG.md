# Decision Log — Skylark BI Agent

## 1. Key assumptions

**Fiscal calendar is Indian (April–March).** Amounts are in rupees and invoice
numbers read `SDPL/FY25-26/916`, so "this quarter" means the fiscal quarter, not
the calendar one. Q1 is Apr–Jun. Every date-scoped answer prints the resolved
window (e.g. `Q3 FY25-26 (2025-10-01 to 2025-12-31)`) so the user can see the
interpretation rather than guess at it.

**"Energy" is not a sector.** The brief's own example question asks about the
energy sector, and no such value exists in the data. Rather than return zero
rows, the agent maps free-text sector words onto real values — energy →
Renewables + Powerline — and states the mapping in the answer. Roughly twenty
synonyms are handled this way.

**Excluding-GST figures are the revenue basis.** Both inclusive and exclusive
columns exist. Order book, billed and yet-to-bill use excl-GST; collections use
incl-GST because that is the only form the collection column takes. Each metric
label says which.

**The two boards have no join key.** Deal names are masked, repeat across
clients, and there is no deal ID on the work order board. Cross-board matching
is done on masked deal name and labelled as indicative, not exact.

**Negative "amount to be billed" is over-billing, not corruption.** Six work
orders bill above order value. These are surfaced as a caveat and left in the
totals rather than clipped to zero.

**Missing values are never imputed.** 181 of 346 deals have no deal value. Any
pipeline figure is therefore a floor, and the agent reports the coverage
percentage alongside it every time.

## 2. Trade-offs chosen, and why

**The LLM plans and narrates; pandas computes.** Gemini turns a question into
`{analysis, filters}` JSON and later turns computed figures into prose. It never
sees a board row and never does arithmetic. This costs some flexibility — only
nine analyses exist, so a truly novel question gets mapped to the nearest one —
but it removes the main hallucination surface. A founder acting on a fabricated
revenue number is a worse failure than a slightly-off analysis choice, and the
printed "Read as:" line makes a wrong mapping visible.

Text-to-SQL or code generation would have been more flexible. It would also let
a model invent a figure that looks plausible, and in a six-hour build there was
no time to construct the guardrails that approach needs.

**Every analysis carries its own caveats.** Coverage warnings are attached where
the number is computed, not bolted on in the prompt, so they cannot be dropped
by a narrator that decides they are not interesting. The narrator prompt
explicitly forbids softening them.

**Columns addressed by title, not monday column ID.** Re-importing a board
generates new column IDs. Reading `column_values { column { title } }` means a
re-import does not break the agent. Slightly more verbose queries, much less
fragile.

**Graceful degradation over failure.** No Gemini key, a quota exhaustion, or a
malformed model response all fall back to a keyword planner and a deterministic
formatter. The app keeps answering; it just stops writing prose. Similarly, a
monday 429 or 5xx retries with backoff, and a 401 produces a specific message
about token regeneration rather than a stack trace.

**Import prep is a separate one-time script.** `prepare_monday_csvs.py` fixes
what monday's importer cannot — a header on row 2, repeated header rows, four
empty columns — and is never imported by the app. The runtime still normalises
everything again, because a human editing a board can reintroduce any of it.

**Streamlit.** Free hosting, a public link, no local setup for the reviewer, and
chat UI in about forty lines. The cost is limited control over layout and a
rerun-per-interaction model. For a prototype judged on data handling rather than
front-end craft, that was the right trade.

## 3. How I interpreted "the agent should help prepare data for leadership updates"

Read as: **a founder should be able to get, in one click, the numbers they would
otherwise spend an afternoon assembling for a board email — with the caveats
already attached, so nobody presents a partial figure as a total.**

Implemented as a dedicated tab producing a period-scoped composite pack:
pipeline health, revenue and billing, operational status, and receivables, plus
a sector breakdown, plus a de-duplicated list of every data caveat across all
four, plus an optional written narrative.

The deliberate choice is that the caveats are part of the deliverable, not a
footnote. Given that half the deals carry no value, a leadership pack that
reported pipeline without saying so would be actively misleading. The pack says
"figures cover the 50% of deals that have a value" in the body.

## 4. What I'd do differently with more time

- **Real conversation memory.** Each question is planned independently, so
  "and how about mining?" as a follow-up does not inherit the previous filters.
  A short rolling context in the planner prompt would fix this.
- **An evaluation set.** Twenty question/expected-analysis pairs run against the
  planner, to measure routing accuracy rather than trusting spot checks.
- **A proper join between the boards.** Even a fuzzy match on client code plus
  date proximity would beat name matching, and would allow a real
  deal-to-delivery cycle-time metric.
- **Trend over time.** Everything is point-in-time. Quarter-over-quarter
  movement is what a founder actually wants, and it needs either historical
  snapshots or monday's activity log.
- **Narrower LLM output validation.** The planner's JSON is validated against
  the analysis registry, but filter values are not checked against the
  vocabularies that actually exist on the boards; an invented owner code
  silently returns an empty result rather than a correction.
- **Ageing buckets for receivables.** There is no payment-due-date column, so
  true 30/60/90 ageing is impossible from this data. Adding one column to the
  board would make it trivial.
