# 06 — Roadmap, Risks and Interview Notes

## 1. Phased roadmap

I haven't assumed a deadline, so this is organised in phases with an exit check each. If you have a fixed number of weeks, give each phase roughly an equal slice and protect Phases 1 to 4 (that is the MVP).

| Phase | What you build | Exit check (don't move on until true) |
|-------|---------------|---------------------------------------|
| **0. Setup** | Repo, environment, Kaggle token, download HI-Small, data manifest | `make data` works; row counts match the dataset page |
| **1. Data + EDA** | Ingest to Parquet, patterns parser, ring ground-truth table, EDA notebook (class balance, amount distributions by currency, pattern types and sizes, time spread) | Ring table loads; you can describe each pattern type in your own words |
| **2. Baselines** | Temporal split, causal basic + account features, LightGBM M1/M2, metrics module, future-invariance test | M1 lands near the published raw-feature number; shuffle test collapses to base rate |
| **3. Graph features + GBM** | Fan-in/out, windowed degree, cycle, scatter-gather features; M3/M4; SHAP | M3 is clearly above M1; feature importances make sense |
| **4. Rings + app (MVP done)** | Ring extraction, pattern rules, ring score, ring-level metrics, Streamlit queue + explorer | You can demo end to end from a clean clone |
| **5. GNN** | Run IBM Multi-GNN once to reproduce its number; then your PyG model M5/M6 | Your GNN is within a few points of the reference, or you can explain the gap |
| **6. Generalise** | LI-Small, HI-Medium, SAML-D runs; ablation table; multi-seed variance | Results tables filled with mean ± std |
| **7. Polish** | README with results and error analysis, limitations page, optional Neo4j, optional Elliptic, short write-up | A stranger can understand and reproduce it |

**Order of cuts if time runs out:** drop Neo4j first, then Elliptic, then SAML-D, then HI-Medium. Never drop the baselines, the temporal split, or the ring-level evaluation.

## 2. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Data leakage inflates scores | High (common bug) | High | Causal features by construction, future-invariance test, shuffle test, temporal split |
| GNN fails to beat tuned GBM | Medium | Low if reported well | Baseline-first design; analyse per pattern; the finding itself is publishable in a README |
| LI-Small scores look poor | High | Medium | Explain base-rate difference; compare with published LI numbers; show transfer from HI if you try it |
| HI-Small is too heavy for the laptop | Medium | Medium | Parquet, categorical dtypes, work on a time slice first, Colab/Kaggle for GNN |
| Cycle / scatter-gather features too slow | Medium | Medium | Sliding time windows, cap lengths, start with 2- and 3-cycles, test on a slice |
| Patterns file format differs from what the parser expects | Medium | Low | Open the real file first; fix the parser; unit-test it on a snippet |
| Ring extraction merges everything into one giant component | Medium | Medium | Max-size cap + community detection; use a higher threshold for "seed" edges |
| Scope creep | High | High | Tiered scope in PRD; cut order above |
| Dataset version changes | Low | Low | Record version and checksums in the manifest |
| Overclaiming | Medium | High (credibility) | Limitations page; "synthetic data" wording everywhere |

## 3. Definition of done

- [ ] `make data features train-baseline rings evaluate app` runs from a clean clone (with datasets downloaded)
- [ ] Temporal split and causal-feature tests pass
- [ ] Results tables filled for HI-Small and LI-Small, with variance
- [ ] Ring-level metrics reported, with per-pattern breakdown
- [ ] Error analysis written (caught and missed rings)
- [ ] README states limitations plainly
- [ ] App runs and the 2-minute demo script works
- [ ] No raw data, tokens or checkpoints committed

## 4. Questions you should be ready to answer (interview prep)

1. *Why a graph approach instead of rules or a flat classifier?* Because the signal is in the relationships: individual transfers look normal; their arrangement does not.
2. *Why is accuracy useless here?* ~0.1% positives. A constant "no" is 99.9% accurate.
3. *Why a temporal split?* Random splits leak future behaviour, and in deployment you always predict forward in time.
4. *Did the GNN beat the gradient-boosted model?* (Your honest answer, with the per-pattern reason.)
5. *How do you know there's no leakage?* Future-invariance test, shuffle test, validation-only thresholds, reproduced a published anchor.
6. *What would change with real bank data?* Late and noisy labels, many more benign structures that look like rings (payroll, marketplaces, family transfers), concept drift, privacy limits, and the cost of false positives.
7. *How would you deploy it?* Batch nightly scoring first; incremental graph updates and streaming features later; analyst feedback to recalibrate thresholds.
8. *What is the difference between edge-level and ring-level evaluation, and why does it matter?* An analyst investigates rings, not transactions; catching 3 of 10 edges in a ring may still surface the ring, while catching many unrelated edges does not.

## 5. Possible extensions after the MVP

- Pretrain on HI, fine-tune on LI (published work shows this helps on low-illicit data).
- Federated / multi-bank view: each bank sees only its own edges, versus a shared graph.
- Temporal GNN or time-encoded edges.
- Elliptic2 subgraph classification as a real-data ring experiment.
- Active-learning loop that simulates analyst feedback.
