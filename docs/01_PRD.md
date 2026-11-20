# 01 — Product Requirements Document (PRD)

**Project:** Money Mule Ring Detector — catching whole fraud rings, not single transactions
**Owner:** Dhruv Gourisaria
**Status:** Draft v1 (pre-build)

---

## 1. Problem

Anti-money-laundering (AML) systems have historically flagged **individual transactions** using fixed rules (amount over a threshold, watch-listed country, and so on). Launderers know the rules, so they split money into many small, innocuous-looking transfers and route it through chains of "mule" accounts. Looked at one at a time, every transaction looks normal. Looked at together, the *shape* of the money flow (a fan-out, a loop that returns to its origin, a scatter then gather) gives it away.

Analysts then face two practical problems: the base rate of laundering is tiny (roughly one transaction in a thousand or fewer in the benchmark data), so naive models drown them in false alarms, and a flagged transaction on its own does not tell them *who else is involved*.

## 2. Vision

A system that reads a transaction history as a **network**, scores transactions and accounts for laundering risk using both transaction details and graph structure, and then groups the flagged activity into **candidate rings** that an analyst can inspect, with a plain-language reason for each.

## 3. Goals

| # | Goal | How we know it worked |
|---|------|-----------------------|
| G1 | Detect laundering transactions better than a tabular-only model | Graph-aware models beat the raw-feature baseline on minority-class F1 and PR-AUC (see 05_EVALUATION_PLAN) |
| G2 | Go beyond per-transaction scoring to **ring-level detection** | Ring recall and ring precision are measured against ground-truth patterns |
| G3 | Be honest and reproducible | Temporal split, causal features, fixed seeds, documented limitations |
| G4 | Be demonstrable | A working app where a user can open a flagged ring and see its structure and the reasons |
| G5 | Show real engineering judgement | A fair baseline-vs-GNN comparison and ablation, with the result reported whichever way it goes |

## 4. Non-goals

- Not a production AML system and not a claim of real-world accuracy. The data is synthetic (plus one real crypto graph).
- Not a regulatory/SAR filing tool.
- No real-time streaming infrastructure (batch scoring only; real-time is discussed as future work).
- No attempt to de-anonymise anyone.

## 5. Users

1. **AML analyst (the product persona).** Wants a short, ranked queue of suspicious rings with reasons, not thousands of isolated alerts.
2. **Reviewer / interviewer / evaluator (the real audience).** Wants to see a rigorous evaluation, clean code, and a demo they can click through in a couple of minutes.
3. **Dhruv (builder).** Wants a project that is finishable, explainable end to end, and defensible under questioning.

## 6. User stories

- As an analyst, I see a ranked list of suspected rings with a risk score, the number of accounts, total money moved and the detected pattern (e.g. cycle, fan-in).
- As an analyst, I can open one ring and see the accounts and transfers drawn as a directed graph with amounts and timing.
- As an analyst, I can see *why* a transaction or ring was flagged (top contributing features and structural pattern).
- As an analyst, I can set an alert budget ("review the top K per day") and see the precision and recall I would get at that budget.
- As a reviewer, I can see baseline vs graph-feature vs GNN results side by side, on HI-Small and LI-Small, with variance.
- As a reviewer, I can read the limitations section and see that the author understands them.

## 7. Functional requirements

**Data**
- FR1. Ingest IBM AML (HI/LI, Small/Medium) transaction, account and pattern files; ingest SAML-D; optionally Elliptic.
- FR2. Normalise to a common schema (see TRD §3) and cache as Parquet.
- FR3. Parse the patterns file into a ring ground-truth table.

**Modelling**
- FR4. Chronological train/validation/test split.
- FR5. Causal feature engineering: account-level behaviour and graph-pattern features.
- FR6. Baseline A: gradient-boosted model on raw transaction features.
- FR7. Baseline B: gradient-boosted model on raw + graph features.
- FR8. Model C: graph neural network for edge (transaction) classification on the directed multigraph.
- FR9. Threshold selection on validation data only.

**Ring detection**
- FR10. Build a candidate-ring graph from high-scoring transactions and extract connected groups (with community splitting for oversized groups).
- FR11. Label each ring with a structural pattern (cycle, fan-in, fan-out, scatter-gather, other) using rules.
- FR12. Score and rank rings.

**Explainability**
- FR13. Per-transaction feature attributions for the gradient-boosted model (SHAP).
- FR14. Per-ring plain-language summary generated from structure and top drivers.

**Evaluation**
- FR15. Edge-level metrics: PR-AUC, minority-class precision/recall/F1, precision@K, recall at fixed alert budgets.
- FR16. Ring-level metrics: ring recall, ring precision, per-pattern recall, analyst workload.
- FR17. Ablation table and multi-seed variance.

**Application**
- FR18. Streamlit app: overview, alert queue, ring explorer, account profile, model comparison, limitations page.
- FR19. Optional Neo4j export/import of flagged rings for graph queries and visualisation.

## 8. Non-functional requirements

- **Reproducibility:** one config file, fixed seeds, pinned dataset version, a `make` target for each stage.
- **Leakage safety:** automated test asserting that no feature uses data from the future.
- **Performance:** HI-Small processed end to end on a 16 GB laptop; GNN training feasible on a free Colab/Kaggle GPU.
- **Code quality:** typed, tested core functions (split, metrics, causal features, ring extraction), README that gets a new user from clone to demo.
- **Privacy/licensing:** no raw dataset files committed; no credentials committed.

## 9. Success metrics

Anchored to published results on the same data (see 04_DATA_PLAN §2). These are targets to aim at, not promises.

| Metric | Minimum (project is credible) | Target |
|--------|------------------------------|--------|
| HI-Small minority-class F1, raw-feature baseline | Reproduce ~20% | Within a few points of published |
| HI-Small minority-class F1, graph features + GBM | > 50% | Low-to-mid 60s |
| HI-Small minority-class F1, GNN | Within ~5 points of graph-feature GBM | Match or modestly exceed it |
| LI-Small | Report honestly (published results are low) | Beat the raw baseline by a clear margin |
| Ring recall (HI-Small) | Measured and reported | Report per pattern type; identify weak patterns |
| Demo | App runs from a clean clone | Under 2 minutes to find and explain a ring |

## 10. Scope tiers

- **MVP (must ship):** FR1–FR9, FR15, baselines A and B, simple ring extraction (FR10), Streamlit alert queue + ring view, README with honest results.
- **Should:** GNN (Model C), ring-level metrics (FR16), SHAP (FR13), SAML-D second dataset, ablation table.
- **Stretch:** Neo4j layer, Elliptic/Elliptic++ real-data experiment, ring pattern classifier beyond rules, Elliptic2 subgraph experiment, real-time scoring sketch.

## 11. Assumptions and constraints

- Data is synthetic; conclusions about real banks are explicitly out of scope.
- Work is done on a personal laptop (assume ~16 GB RAM) with optional free cloud GPU.
- Solo project; the build must be finishable in a few weeks, so MVP is protected first.

## 12. Key risks (short list; full table in 06_ROADMAP_AND_RISKS)

- GNN does not beat the tuned GBM. **Mitigation:** this is a valid, reportable finding; the baseline-first design makes the project robust to it.
- Leakage inflating results. **Mitigation:** causal-feature unit test, temporal split, validation-only thresholds.
- Scope creep (Neo4j, three datasets, subgraph models). **Mitigation:** tiered scope above.
- LI-Small scores look weak. **Mitigation:** explain the base-rate difference and compare with published LI numbers.
