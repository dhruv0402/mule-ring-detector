# Money Mule Ring Detector

Catching whole fraud rings, not single transactions. The system reads a transaction history as a directed network, scores each transfer using transaction details plus graph structure, and then groups flagged activity into candidate **rings** an analyst can inspect.

> **Status:** pre-build. Docs and scaffold are complete; modelling code is built phase by phase (see `docs/06_ROADMAP_AND_RISKS.md`).
> **Data:** synthetic benchmarks (plus an optional real Bitcoin graph). Nothing here claims real-world bank accuracy.

## What's in the box

| Path | Contents |
|------|----------|
| `docs/01_PRD.md` | Product requirements: goals, users, requirements, scope tiers, success metrics |
| `docs/02_TRD.md` | Technical design: architecture, schema, features, models, ring extraction, Neo4j layer |
| `docs/03_APP_FLOW.md` | Pipeline flow, app screens, 2-minute demo script, empty/error states |
| `docs/04_DATA_PLAN.md` | **Dataset research**: which data to use, why, expected results, leakage rules, download steps |
| `docs/05_EVALUATION_PLAN.md` | Edge-level and ring-level metrics, baselines, ablations, sanity checks |
| `docs/06_ROADMAP_AND_RISKS.md` | Phases with exit checks, risk register, interview prep |
| `src/mulering/` | Working: ingest, patterns parser, temporal split, causal features, metrics, ring extraction, LightGBM baseline, Streamlit stub. To build: graph-pattern features, GNN |
| `tests/` | Split, future-invariance (leakage), metrics, parser, ring-matching and ring-extraction tests |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put your Kaggle token in ~/.kaggle/kaggle.json, then:
bash scripts/download_data.sh ibm      # start with HI-Small
make test                              # should pass before you touch real data
make ingest                            # CSV -> Parquet + ring ground truth
```

PyTorch and PyTorch Geometric are installed separately (match your CPU/CUDA build) when you reach the GNN phase.

## Datasets

IBM AML (HI/LI, Small/Medium) is the backbone; SAML-D is the independent second generator; Elliptic / Elliptic++ / Elliptic2 are optional real-data checks. Details, sizes, links and caveats are in `docs/04_DATA_PLAN.md`. Raw data is git-ignored and must be downloaded by script.

## Results

*(fill in as you go; templates in `docs/05_EVALUATION_PLAN.md`)*

## Limitations

Synthetic data; simulator-defined laundering patterns; batch evaluation only; no analyst feedback loop. See `docs/05_EVALUATION_PLAN.md` §10.

## Citations

- Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023
- Egressy et al., *Provably Powerful Graph Neural Networks for Directed Multigraphs*, AAAI 2024
- Oztas et al., *Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset*, IEEE ICEBE 2023
- Bellei et al., *The Shape of Money Laundering ... Elliptic2*, 2024; Elliptic++ (arXiv 2306.06108)
