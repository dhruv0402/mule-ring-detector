# 04 — Data Plan

> The honest starting point: **there is no public dataset of real bank transactions with confirmed money-laundering rings.** Banks cannot release them. So "good data" for this project means (a) the best *labelled synthetic* benchmarks, (b) one *real* labelled crypto graph as a reality check, and (c) a disciplined way of using them so the results mean something.
>
> Everything below was gathered from the dataset pages and papers listed at the bottom. Before you rely on a number, open the source and confirm it against the version you actually download. Dataset pages get updated.

---

## 1. The datasets, ranked by how useful they are for this project

| # | Dataset | Role in project | Label level | Size (as published) | Access |
|---|---------|-----------------|-------------|---------------------|--------|
| 1 | **IBM AML (AMLworld) — HI-Small** | Primary development set | Transaction (edge) + ring files | ~5.08M transactions, ~515K accounts, 5,177 laundering (0.102%) | Kaggle |
| 2 | **IBM AML — LI-Small** | The "hard, realistic" test | Transaction (edge) + ring files | ~4.0K laundering transactions, ~1 in 1,942 | Kaggle |
| 3 | **IBM AML — HI-Medium** | Scale + generalisation check | Transaction (edge) + ring files | ~35K laundering transactions, ~1 in 905 | Kaggle |
| 4 | **SAML-D** | Second, independent generator (different typologies, countries, payment types) | Transaction + typology name | 9,504,852 transactions, 0.1039% suspicious, 28 typologies (11 normal, 17 suspicious) | Kaggle |
| 5 | **Elliptic (Elliptic1) / Elliptic++** | Real-world sanity check on a *real* labelled graph | Node (transaction or wallet) | Elliptic++: 203K transactions, 822K wallet addresses | Kaggle / Google Drive |
| 6 | **Elliptic2** (stretch) | Ring-level ("subgraph") classification on real data | Subgraph | 122K labelled subgraphs inside a 49M-node, 196M-edge background graph | elliptic.co / Kaggle |

### Why IBM AML is the backbone

- The generator simulates people, companies and banks, and models the full laundering cycle (placement, layering, integration), so laundering is hidden inside realistic background activity rather than bolted on.
- It ships **three files per dataset**: the transactions, an accounts table, and a **patterns file** that lists every injected laundering scheme as a group of transactions. That patterns file is the single most valuable thing for you, because it gives **ring-level ground truth** (see §3). Most student projects only evaluate per-transaction and never use it.
- Pattern types in the benchmark include fan-out, fan-in, cycles, scatter-gather, gather-scatter, stack, bipartite and random structures.
- It is published with a proper paper (Altman et al., NeurIPS 2023 Datasets & Benchmarks), comes with reference baselines, and is under a Community Data License Agreement. You can cite it and compare against it.
- **HI vs LI:** both groups share the same generator; HI has more laundering, LI has less (so patterns are harder to find). LI is the more realistic base rate and the more honest test. In the published benchmarks, models score far lower on LI than on HI, so do not be alarmed when your LI numbers look small.

**Use order:** build everything on HI-Small → check on LI-Small → check scale/generalisation on HI-Medium. Do not start with Medium or Large; they are slow to iterate on and add nothing until the pipeline is solid.

### Why SAML-D is the best second dataset

It was built from the literature and interviews with AML specialists, includes geography, high-risk countries and payment types, and labels each suspicious transaction with a **typology name**. That lets you answer a question IBM's data answers less directly: *which kinds of laundering does my system catch and which does it miss?* Because it comes from a different generator, a model that holds up on both is much more convincing than one tuned to a single dataset.

Caveats: it is synthetic, the typologies cannot cover every real-world technique, and its flagged-transaction ratio is very low (about 0.1%). Do not feed the typology column to the model as a feature; use it only to *analyse* results (see leakage rules, §4).

### Why Elliptic is still worth adding

Everything above is synthetic. Elliptic is derived from real Bitcoin transactions with real (analyst-derived) illicit/licit labels. It differs from your main task: its labels are on **nodes** and the data is **crypto**, not bank transfers, and a large share of nodes are unlabeled. So treat it as a *separate, secondary experiment* ("does the graph pipeline still help on real data?"), not a replacement benchmark. Elliptic++ adds wallet-address-level data and richer features on top of the original.

**Elliptic2** is the real-data answer to "rings, not single transactions": its labels are on whole **subgraphs**. It is also very large, so only attempt it as a stretch goal once everything else is done, and plan for a sampled subset.

---

## 2. What published results tell you to expect (your sanity anchors)

These are *minority-class F1* scores (F1 on the laundering class only) reported in the papers. Splits, preprocessing and dataset versions differ between papers, so use them as **ballpark anchors, not exact targets**.

| Approach | HI-Small | LI-Small | Source |
|----------|---------:|---------:|--------|
| LightGBM, raw transaction features only | ~21% | ~2% | Graph Feature Preprocessor paper |
| LightGBM + graph-pattern features | ~63% | ~21% | Reported in benchmark tables |
| XGBoost + graph-pattern features | ~63% | ~27% | Reported in benchmark tables |
| Multi-GIN (GNN with ego IDs etc.) | ~59% | ~24% | Egressy et al., AAAI 2024 |
| Multi-PNA (GNN) | ~66% | ~30% | Egressy et al., AAAI 2024 |
| Multi-PNA + edge updates | ~68% | ~33% | Egressy et al., AAAI 2024 |

What this table is really telling you:

1. **Graph structure is what matters.** Raw per-transaction features give about 21% F1 on HI-Small; adding graph-pattern features roughly triples it. A plain gradient-boosted model with good graph features is competitive with GNNs, and one paper reports it beating the best GNN baseline on several settings. This is exactly why the baseline-first plan in the PRD is non-negotiable.
2. **LI is much harder.** Expect low scores and say so openly.
3. **A well-engineered GNN can edge ahead**, particularly on cyclic and scatter-gather structures, but the gap is modest.

**Realistic target for this project:** reproduce the "LightGBM + graph features" tier on HI-Small (low-to-mid 60s F1), get a GNN into the same neighbourhood, and report LI-Small honestly. Beating every published number is not the goal. A clean, leak-free, reproducible result with a ring-level evaluation is.

---

## 3. The feature that makes your project different: ring-level ground truth

The IBM patterns file lists each laundering attempt as a block of transactions, bracketed by BEGIN/END lines that name the pattern type. From that you can build a ground-truth table of rings: `ring_id, pattern_type, set_of_accounts, set_of_transactions`.

This allows metrics the published benchmarks mostly do not report:

- **Ring recall:** what fraction of true rings did the system recover (see 05_EVALUATION_PLAN.md for the exact overlap rule)?
- **Per-pattern recall:** fan-in vs cycle vs scatter-gather, etc.
- **Ring precision / analyst workload:** how many predicted rings are real, and how many accounts must an analyst review?

*Parser note:* the block format above is my understanding of the file. Open the file you download and confirm the format first; the parser in `src/mulering/data/ingest.py` is written against that format and covered by a unit test on a synthetic sample, but check it against the real file before trusting it.

---

## 4. Rules for using this data so the results are trustworthy

1. **Split by time, never randomly.** Use the chronological 60/20/20 split (train/validation/test) that the benchmark papers use. Random splits leak future behaviour and inflate scores.
2. **Compute every feature causally.** A feature for a transaction at time *t* may only use transactions strictly before *t*. This is the most common silent bug in AML projects.
3. **Pick the decision threshold on the validation set only**, then freeze it for the test set.
4. **Never use these as model inputs:** the `Is Laundering` label (obviously), the pattern/typology name, or anything derived from the patterns file. They exist for *evaluation and analysis*, not training.
5. **Account IDs:** build account identity as `bank + account number` (the account number alone is not unique across banks).
6. **Currencies:** the data mixes currencies and there are no exchange rates shipped with it. Use log-transformed amounts plus currency as a categorical feature, and add an "amount relative to this account's history" feature. Do not pretend amounts are directly comparable across currencies.
7. **Report variance.** Use 3–5 seeds for stochastic models and report mean ± std.
8. **Do not redistribute the raw datasets** in your GitHub repo. Keep them in `data/raw/` (git-ignored) and download via the script. Check the licence terms before publishing any derived sample.
9. **Pin the dataset version** (record the Kaggle version number and file checksums in `reports/data_manifest.json`) so your results are reproducible.

---

## 5. Download & setup

```bash
# one-time: install Kaggle CLI and put your API token in ~/.kaggle/ (never commit it)
pip install kaggle
bash scripts/download_data.sh ibm      # IBM AML (all six datasets; start with HI-Small)
bash scripts/download_data.sh samld    # SAML-D
bash scripts/download_data.sh elliptic # original Elliptic (Elliptic++ and Elliptic2 are manual, see below)
```

Manual steps:
- **Elliptic++:** download from the project's GitHub page (the data is hosted on Google Drive).
- **Elliptic2:** request via elliptic.co/elliptic2 or the Kaggle page; follow the official guide repo for the expected file layout.

Disk and memory guidance: HI-Small is a few hundred MB as CSV; convert once to Parquet (the pipeline does this) and work from Parquet. Plan on 16 GB RAM for HI-Small with pandas; use chunked reads or Polars if you are tighter. Use a free Colab or Kaggle GPU for GNN training if your laptop has none.

---

## 6. Reference implementations to study (not copy blindly)

- **IBM Multi-GNN** (official code for the AAAI 2024 paper, Apache-2.0): the reference for GNN training on this exact data. Run it once to reproduce the published HI-Small number; that gives you a trusted yardstick for your own GNN.
- **Elliptic2 official guide repo** (MIT-IBM): data layout and subgraph-learning examples.
- **Elliptic++ repo:** tutorial notebooks for transaction and actor classification.
- **Graph Feature Preprocessor paper:** a clear list of the graph-pattern features (fan-in/out, cycles, scatter-gather, temporal degree features) that lift gradient-boosted models. Implement your own version; do not depend on the paper's library.

---

## 7. Sources

- Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023 (arXiv 2306.16424); Kaggle: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`
- Egressy et al., *Provably Powerful Graph Neural Networks for Directed Multigraphs*, AAAI 2024 (arXiv 2306.11586); code: `github.com/IBM/Multi-GNN`
- *Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection* (ACM ICAIF 2024, DOI 10.1145/3677052.3698674)
- Oztas et al., *Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset* (IEEE ICEBE 2023); Kaggle: `berkanoztas/synthetic-transaction-monitoring-dataset-aml`
- Bellei et al., *The Shape of Money Laundering: Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset* (arXiv 2404.19109); `github.com/MITIBMxGraph/Elliptic2`
- Elliptic++ (arXiv 2306.06108); `github.com/git-disl/EllipticPlusPlus`; original Elliptic: Kaggle `ellipticco/elliptic-data-set`
