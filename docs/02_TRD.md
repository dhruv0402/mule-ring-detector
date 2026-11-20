# 02 — Technical Requirements Document (TRD)

## 1. Architecture at a glance

```
 Raw files (IBM AML / SAML-D / Elliptic)
            │
            ▼
   ┌──────────────────┐
   │ 1. Ingest        │  CSV → standard schema → Parquet cache; parse ring ground truth
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │ 2. Temporal split│  60 / 20 / 20 by timestamp (train / val / test)
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │ 3. Features      │  causal account features + graph-pattern features
   └───┬──────────┬───┘
       │          │
       ▼          ▼
 ┌──────────┐ ┌───────────────┐
 │ 4a. GBM  │ │ 4b. GNN (PyG) │   edge classification, same split
 │ A: raw   │ │ GIN / PNA-style│
 │ B: +graph│ │ + ego IDs etc. │
 └────┬─────┘ └──────┬────────┘
      └──────┬───────┘
             ▼
   ┌──────────────────┐
   │ 5. Threshold     │  chosen on validation set only
   └────────┬─────────┘
            ▼
   ┌──────────────────┐
   │ 6. Ring extractor│  flagged edges → subgraph → components/communities → pattern label → ring score
   └────────┬─────────┘
            ▼
   ┌──────────────────┐        ┌────────────────────┐
   │ 7. Evaluation    │        │ 8. Explanations    │
   │ edge + ring level│        │ SHAP + ring summary│
   └────────┬─────────┘        └─────────┬──────────┘
            └────────────┬───────────────┘
                         ▼
            reports/*.json, *.parquet
                         ▼
        ┌─────────────────────────────┐
        │ 9. Streamlit app (+ Neo4j)  │
        └─────────────────────────────┘
```

## 2. Technology choices

| Layer | Choice | Why / note |
|-------|--------|-----------|
| Language | Python 3.10+ | Your existing stack |
| Data wrangling | pandas + PyArrow (Parquet); Polars optional | pandas is fine for HI-Small if dtypes are set; Polars if memory gets tight |
| Graph analysis | NetworkX | Exploration, ring extraction on the *small* flagged subgraph. Not for whole-graph computation |
| Tabular models | LightGBM (primary), XGBoost (comparison) | Strong baselines on imbalanced tabular data |
| GNN | PyTorch + PyTorch Geometric | Edge-feature-aware message passing, neighbour sampling |
| Explainability | SHAP | TreeExplainer for the GBMs |
| Metrics | scikit-learn | Average precision, PR curves |
| App | Streamlit; pyvis or Plotly for graph drawing | Fast to build, free to host |
| Graph DB (optional) | Neo4j (Aura Free or local Docker) | Only for the demo/query layer |
| Config | YAML + a small loader | One source of truth |
| Testing | pytest | Core logic only |
| Reproducibility | fixed seeds, `data_manifest.json`, `make` targets | |

Pin exact versions in `requirements.txt` after your first successful install (`pip freeze`). PyTorch Geometric must match your PyTorch/CUDA build; follow its install instructions rather than guessing.

## 3. Data model

### 3.1 Standard transaction schema (all datasets map into this)

| Column | Type | Notes |
|--------|------|-------|
| `edge_id` | int64 | Row index after sorting by time (stable) |
| `ts` | datetime64 | Transaction time |
| `src` | string/int | Sender account id = `bank + "_" + account` |
| `dst` | string/int | Receiver account id (same construction) |
| `amount_sent` / `amount_recv` | float | In their own currencies |
| `cur_sent` / `cur_recv` | category | Currency |
| `pay_format` | category | Payment type (wire, ACH, cheque, credit card, ...) |
| `src_bank` / `dst_bank` | category | |
| `label` | int8 | 1 = laundering |
| `ring_id` | int, nullable | Ground-truth ring this edge belongs to (evaluation only, from patterns file) |
| `pattern_type` | category, nullable | Ground-truth pattern name (evaluation only) |

Account ids are integer-encoded for modelling; the mapping is saved so the app can show originals.

### 3.2 Ring ground-truth table

`ring_id, pattern_type, accounts (set), edge_ids (set), n_accounts, n_edges, start_ts, end_ts`

### 3.3 Predicted ring table (output)

`pred_ring_id, accounts, edge_ids, pattern_guess, ring_score, n_accounts, total_amount, start_ts, end_ts, top_reasons`

## 4. Pipeline stages in detail

### 4.1 Ingest
- Read CSVs with explicit dtypes and names (the IBM files repeat the header name "Account" twice, so assign column names explicitly).
- Sort by timestamp; assign stable `edge_id`.
- Write Parquet to `data/processed/<dataset>/`.
- Parse the patterns file to attach `ring_id` / `pattern_type` to edges (evaluation columns only).
- Write `reports/data_manifest.json` (file names, row counts, checksums, dataset version).

### 4.2 Temporal split
- Sort by `ts`; cut at the 60th and 80th percentile of time (or of row count; pick one and document it).
- Fit nothing on validation/test. Any encoders or scalers are fit on train only.
- Graph for GNN training uses edges up to the end of the training window; validation/test edges are added cumulatively when scoring later windows (mirror the published protocol; document exactly what you do).

### 4.3 Feature engineering (all causal: only information strictly before the transaction's timestamp)

**Edge-level (basic)**
- log(amount), currency pair, same-currency flag, payment format, same-bank flag, hour/day-of-week.

**Account behaviour (computed on history before time t)**
- Count/sum of transactions sent and received so far; unique counterparties so far.
- Time since the account's previous transaction (sent and received).
- Amount relative to that account's historical average (z-score or ratio).
- Account "pass-through" behaviour: share of recently received money that was sent onward within a short window.
- In-degree vs out-degree imbalance.

**Graph-pattern features (the part that moves the needle)**
- Windowed fan-in and fan-out counts around the edge's endpoints (several window sizes, e.g. 1h/1d/full history).
- Cycle participation: does this edge close a directed cycle of length 2–6 within a time window? (Implement 2- and 3-cycles first, longer cycles if time allows.)
- Scatter-gather indicator: many distinct intermediaries between the same source and destination within a window.
- Bidirectional flow between the pair; repeated transfers between the same pair.
- Neighbourhood stats: mean/max of neighbours' degree and amounts (1-hop, optionally 2-hop).

Implementation note: use time-sorted groupby operations (`cumcount`, `cumsum`, `shift`, rolling windows) for the per-account features. Cycle and scatter-gather features are the expensive ones; compute them on a time-bounded sliding window, and test on a small slice before running on the whole dataset.

**Leakage test (required):** a unit test builds a tiny synthetic transaction table and asserts that changing any *future* transaction does not change the features of an earlier one.

### 4.4 Models

**A. Raw-feature GBM.** LightGBM on edge-level (basic) features only.
**B. Graph-feature GBM.** LightGBM (and XGBoost for comparison) on basic + account + graph-pattern features. Handle imbalance with class weighting or `scale_pos_weight`, tuned on validation. Early stopping on validation PR-AUC.
**C. GNN (edge classification).**
- Directed multigraph, accounts as nodes, transactions as edges with feature vectors.
- Start from a GIN-style edge-aware message-passing model; then add the three adaptations shown to matter for directed multigraphs: **reverse message passing** (separate parameters for incoming and outgoing edges), **port numbering** (distinguish parallel/neighbouring edges), and **ego IDs** (mark the endpoints of the edge being classified). Optionally edge-feature updates between layers.
- Edge classification head: concatenate the two endpoint embeddings and the edge embedding → MLP → logit.
- Neighbour sampling (mini-batch) so it fits in memory; class-weighted loss; early stopping on validation F1/PR-AUC.
- Before trusting your implementation, **run IBM's official Multi-GNN code once** on HI-Small and confirm you can reproduce the published number. Then compare your model to it.

### 4.5 Thresholding
Choose the decision threshold that maximises F1 on the *validation* set (and also report results at fixed alert budgets). Freeze it for test.

### 4.6 Ring extraction

Input: test-period edges with scores and a threshold.
1. **Flag** edges with score ≥ threshold (optionally a slightly lower second threshold to "grow" rings: include neighbouring edges that are individually weaker but attached to a flagged edge).
2. Build an undirected view of the subgraph induced by flagged edges' endpoints (keep direction and edge attributes for later).
3. Take **connected components** as candidate rings.
4. If a component is too large (set a max size, e.g. a few hundred accounts), split it with community detection (Louvain/label propagation) and keep sub-communities.
5. **Pattern label** each ring using rules on its directed structure: cycle (strongly connected component with a directed cycle), fan-in (one sink with many sources), fan-out (one source, many sinks), scatter-gather (source → many intermediaries → one sink), stack/bipartite (layered or two-sided), else "other".
6. **Ring score:** a documented aggregate (e.g. mean of top-n edge scores weighted by amount, plus a bonus for recognised structure). Keep it simple and justify it; test two variants in the ablation.
7. Output the predicted-ring table (§3.3).

### 4.7 Explanations
- SHAP values for the GBM per flagged edge; top-k features stored with each alert.
- Ring summary: template-based natural language from structure + top drivers, e.g. "12 accounts, directed cycle, $X moved in 3 hours; drivers: unusually fast pass-through, repeated transfers between the same pair." Template, not LLM, so every statement is traceable.

### 4.8 Evaluation
Defined fully in `05_EVALUATION_PLAN.md`. The evaluation module is dataset-agnostic and takes only scores, labels and (optionally) the ring ground-truth table.

## 5. Optional graph database layer (Neo4j)

Purpose: demo and ad-hoc investigation queries, not model compute.

```
(:Account {id, bank})
(:Account)-[:SENT {edge_id, ts, amount, currency, format, score, label}]->(:Account)
(:Ring {ring_id, pattern, score})
(:Account)-[:IN_RING]->(:Ring)
```

Only flagged rings and their immediate neighbourhoods are loaded (thousands of nodes, not millions). Example queries to show: all accounts in ring X; accounts that appear in more than one ring; shortest path between two flagged accounts; accounts that both receive from and send to the same ring.

If time is short, skip Neo4j and draw rings directly from the Parquet outputs in Streamlit. Neo4j earns its place only if the demo actually uses the queries.

## 6. Application layer (Streamlit)
See `03_APP_FLOW.md`. The app reads precomputed artefacts from `reports/` and `data/processed/` and does **not** retrain anything at runtime. A small, licence-checked demo extract is used for hosted deployment.

## 7. Repository and configuration
- Layout: see README.
- `configs/config.yaml` controls dataset name, paths, split fractions, window sizes, model hyperparameters, thresholds, seeds.
- `Makefile` targets: `data`, `features`, `train-baseline`, `train-gnn`, `rings`, `evaluate`, `app`, `test`.

## 8. Performance and resource plan
- Develop and debug on a **10% time-slice or sampled subgraph**; run the full HI-Small only when the pipeline is stable.
- Parquet + categorical dtypes; avoid Python loops over rows.
- NetworkX only on flagged subgraphs.
- GNN: neighbour sampling, mixed precision if on GPU, checkpoint every epoch.

## 9. Testing
- Unit tests: temporal split (no overlap, ordered), causal features (future-invariance), metrics (known toy cases), pattern parser (synthetic sample), ring extraction (planted cycle/fan-in recovered), ring-overlap metric.
- Smoke test: whole pipeline on a tiny synthetic dataset in CI.

## 10. Deployment
- Streamlit Community Cloud from GitHub. Ship a small processed demo bundle; never ship raw data.
- Document how to regenerate everything locally from the download script.

## 11. Security and licensing
- API tokens only in environment variables or `~/.kaggle/`; `.gitignore` covers data, tokens and model checkpoints.
- Cite all datasets and papers in the README.
