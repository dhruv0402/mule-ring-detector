# 03 — App Flow

Two flows matter here: the **offline pipeline** that produces everything, and the **analyst-facing app** that reads it.

## 1. Offline pipeline flow

```
download_data.sh ──► ingest ──► split ──► features ──► train baselines (A, B)
                                              │                │
                                              │                ▼
                                              └──────► train GNN (C)
                                                               │
                         validation scores ──► pick threshold ◄┘
                                                               │
                                       test scores ──► extract rings ──► evaluate
                                                               │              │
                                                               ▼              ▼
                                                     reports/rings.parquet   reports/metrics.json
                                                     reports/explanations.parquet
                                                               │
                                                               ▼
                                                        Streamlit app
                                                     (+ optional Neo4j load)
```

Each arrow is a `make` target and writes a file, so any stage can be re-run alone.

## 2. App screens

### Screen 1 — Overview
- Headline numbers for the selected dataset and model: transactions scored, alerts raised, rings found.
- Mini results table: baseline vs graph-feature GBM vs GNN (minority F1, PR-AUC).
- Short "how to read this" note and a visible data disclaimer ("synthetic data").

### Screen 2 — Ring queue (the main screen)
- Table of predicted rings, ranked by ring score.
- Columns: rank, ring score, pattern guess, accounts, transactions, total amount, time span.
- Filters: pattern type, minimum size, minimum score, date range.
- **Alert budget slider:** "show top K rings" with live precision / recall at that K (computed from ground truth, clearly labelled as evaluation-only).
- Clicking a row opens Screen 3.

### Screen 3 — Ring explorer
- Directed graph of the ring: nodes = accounts, edges = transfers (thickness ≈ amount, colour ≈ risk score; tooltip with time, amount, currency, payment format).
- Side panel: auto-generated summary (pattern, size, amount, duration) and the top reasons (SHAP-driven for member transactions).
- Timeline strip showing when the transfers happened.
- Evaluation-mode toggle: overlay ground-truth laundering edges to show what the system got right and wrong (hidden by default so the analyst view stays realistic).
- Button: "Open in Neo4j" / "Export ring" (if the graph DB layer is enabled).

### Screen 4 — Account profile
- Opened by clicking any account in a ring.
- Shows that account's history: in/out counts, amounts, counterparties, pass-through behaviour, other rings it appears in.

### Screen 5 — Model comparison
- Results table for all models and datasets with mean ± std over seeds.
- Precision-recall curves overlaid.
- Per-pattern recall chart (which ring types are caught vs missed).
- Ablation table (raw → +account features → +graph features → GNN).

### Screen 6 — Data & limitations
- Which datasets, versions and splits were used and why.
- Plain statement of limits: synthetic data, simulated labels, no real-world validation, only batch scoring, false positives cost analyst time.
- Citations.

## 3. User journey (the 2-minute demo script)

1. Land on Overview → see the results table and the "synthetic data" note.
2. Go to Ring queue → set alert budget to top 20 rings.
3. Open the top ring → see the cycle drawn out, read the summary and reasons.
4. Toggle ground-truth overlay → show it matches the true laundering pattern.
5. Click an account → see its profile and that it appears in only that ring.
6. Open Model comparison → show baseline vs GNN and the per-pattern chart, and point at the weakest pattern type.
7. Open Limitations → close on honesty.

## 4. State and data loading

- App loads precomputed artefacts only (`reports/` + a small processed bundle). No training in the app.
- Cache loaded tables with Streamlit's caching.
- Selected dataset, model, filters and selected ring live in session state so navigation keeps context.

## 5. Error and empty states

| Situation | Behaviour |
|-----------|-----------|
| Artefacts missing | Show which `make` target to run |
| No rings above filters | Friendly empty state with filter reset |
| Very large ring (> max size) | Show the community-split version, with a note |
| Neo4j not configured | Hide Neo4j buttons; app still fully works |
| Evaluation overlay requested on a dataset without ring ground truth | Disable the toggle with a tooltip |

## 6. Optional: Neo4j demo queries

Show a small "Investigator queries" panel with three canned queries (accounts shared between rings, shortest path between two flagged accounts, all counterparties of a ring). Only build this if the Neo4j layer exists.
