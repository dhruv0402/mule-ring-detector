# 05 — Evaluation Plan

The evaluation is the part of this project that decides whether it reads as a toy or as real work. Everything below is designed so that the numbers can be defended.

## 1. Principles

1. **Time-ordered split only.** 60% earliest transactions train, next 20% validation, last 20% test.
2. **Causal features only.** Nothing from the future may influence a feature.
3. **Validation picks, test reports.** Thresholds and hyperparameters are chosen on validation; test is touched once per final model.
4. **Accuracy is banned.** With ~0.1% positives, a model that predicts "never laundering" scores about 99.9% accuracy. It is meaningless here.
5. **Report variance.** Mean ± std over 3–5 seeds for anything stochastic.
6. **Report what didn't work.** A negative result with a clear explanation is a good result.

## 2. Edge-level metrics (per transaction)

| Metric | Why |
|--------|-----|
| **PR-AUC (average precision)** | Threshold-free, the right summary for heavy imbalance |
| **Minority-class F1** | The metric used by the published benchmarks, so you can compare |
| **Minority-class precision and recall** (at the validation-chosen threshold) | Shows the trade-off behind the F1 |
| **Precision@K / Recall@K** at fixed alert budgets | How an analyst actually uses a model: "I can review K alerts" |
| ROC-AUC (secondary only) | Often looks great under imbalance; do not lead with it |

## 3. Ring-level metrics (the differentiator)

Ground truth comes from the IBM patterns file: each true ring has a set of accounts.

**Matching rule.** A predicted ring *matches* a true ring if the overlap of their account sets satisfies Jaccard ≥ 0.5 (use 0.3 as a looser secondary setting, and report both). One true ring may be matched by at most one predicted ring (greedy by overlap) to avoid double counting.

| Metric | Definition |
|--------|------------|
| **Ring recall** | matched true rings ÷ all true rings in the test period |
| **Ring precision** | predicted rings that match a true ring ÷ all predicted rings |
| **Per-pattern recall** | ring recall broken out by pattern type (fan-in, fan-out, cycle, scatter-gather, ...) |
| **Account-level coverage** | fraction of true ring accounts that appear in any predicted ring |
| **Analyst workload** | accounts to review per true ring found at a given alert budget |

Rings that straddle the train/test boundary: assign a ring to the test set only if its *first* transaction falls in the test window, and say so.

## 4. Baselines and ablation

| ID | Model | Features |
|----|-------|----------|
| M0 | Majority / random | none (sanity floor) |
| M1 | LightGBM | raw edge features |
| M2 | LightGBM | + account behaviour features |
| M3 | LightGBM | + graph-pattern features |
| M4 | XGBoost | same as M3 (second GBM for robustness) |
| M5 | GNN (plain edge-aware GIN) | raw edge features |
| M6 | GNN + reverse message passing + ports + ego IDs | raw edge features |
| M7 (optional) | GNN + graph-pattern features as edge inputs, or GNN embeddings fed into GBM | combined |

The headline question is **M3 vs M6**. Whichever wins, explain why, using the per-pattern breakdown (published work suggests GNNs help most on cycles and scatter-gather).

## 5. Datasets and what each answers

| Dataset | Question |
|---------|----------|
| IBM HI-Small | Does the pipeline work and beat the baseline? |
| IBM LI-Small | Does it hold up at a realistic, lower laundering rate? |
| IBM HI-Medium | Does it scale and generalise? |
| SAML-D | Does it generalise to a different generator, and which typologies fail? (typology column used for analysis only) |
| Elliptic / Elliptic++ (optional) | Does graph structure still help on real labelled data? (node-level task; reported separately) |

## 6. Sanity checks before believing any number

- **Reproduce a published anchor** (IBM's Multi-GNN code on HI-Small) to confirm your split and metric code agree with the literature.
- **Shuffle test:** train with shuffled labels; PR-AUC should collapse to the base rate. If it doesn't, you have leakage.
- **Future-invariance test:** perturb future rows; earlier features must not change.
- **Feature importance review:** if one feature dominates absurdly (e.g. something that encodes the label), investigate.
- **Single-seed vs multi-seed gap:** a big gap means the result is fragile.

## 7. Statistical reporting

- Table cells: `mean ± std (n seeds)`.
- For the key M3-vs-M6 comparison, include a paired comparison over seeds, or a bootstrap confidence interval over test-set days/blocks. Do not claim a win inside the noise.

## 8. Results table templates (fill in as you go)

**Edge level**

| Model | Dataset | PR-AUC | Min-F1 | Precision | Recall | P@K |
|-------|---------|--------|--------|-----------|--------|-----|
| M1 | HI-Small | | | | | |
| M3 | HI-Small | | | | | |
| M6 | HI-Small | | | | | |
| ... | LI-Small | | | | | |

**Ring level**

| Model | Dataset | Ring recall (J≥0.5) | Ring precision | Fan-in | Fan-out | Cycle | Scatter-gather |
|-------|---------|--------------------|----------------|--------|---------|-------|----------------|

## 9. Error analysis (required section in the README)

- Show 2–3 rings the system caught and 2–3 it missed, with the reason (too small, too slow, too spread out in time, looks like normal business activity).
- Show the most common false-positive structure.
- State which pattern types are weakest and what you'd try next.

## 10. Limitations to state plainly

- Synthetic data: labels come from the simulator, which makes the task cleaner than reality (real labels are late, partial and noisy).
- The simulator's laundering patterns are finite and known; real launderers adapt.
- Batch evaluation, not a live deployment; no analyst feedback loop.
- A real bank would care about cost-weighted metrics (false positives cost analyst time); the alert-budget view is a first step toward that.
