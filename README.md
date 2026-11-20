# Money Mule Ring Detector

**Graph-Based Detection of Money Mule Rings in Financial Transaction Networks**  
*An end-to-end machine learning and graph analytics framework for financial crime investigation.*

[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen)](tests/test_core.py)
[![Dataset](https://img.shields.io/badge/Dataset-IBM_AML_HI--Small-blue)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![UI](https://img.shields.io/badge/UI-Streamlit_&_PyVis-orange)](src/mulering/app/streamlit_app.py)

---

## The Problem

Traditional Anti-Money Laundering (AML) monitoring systems inspect transactions in isolation using static heuristics (threshold amounts, entity watch-lists). Laundering syndicates evade these per-transaction checks by structuring funds across chains of "mule" accounts. Because illicit transfers constitute only ~0.1% of transactions in benchmark data, per-transaction models either miss coordinated activity or generate high false-positive volumes that overwhelm compliance investigators.

Furthermore, compliance analysts investigate **cases**, not single database rows. Scoring transactions in isolation fails to surface the underlying criminal network.

This project designs, evaluates, and deploys a **ring-level detection system** evaluated on the official **IBM AML Benchmark (`HI-Small`: 5,078,345 transactions, 5,177 illicit transfers, 370 ground-truth rings)** using IBM's AMLworld simulator (Altman et al., NeurIPS 2023).

---

## System Architecture

```
[ Raw Transactions (5.07M rows) ]
               │
               ▼
[ Chronological Split (60% Train / 20% Val / 20% Test) ]
               │
               ▼
[ Causal Feature Engine (Future-Invariant Historical Graph) ]
               │
       ┌───────┴───────┐
       ▼               ▼
[ Model M1: Raw ] [ Model M2: Causal Graph ]
       │               │
       └───────┬───────┘
               ▼
[ Validation Threshold Selection (Max Minority F1) ]
               │
               ▼
[ Candidate Ring Extraction (Connected Components + Louvain) ]
               │
               ▼
[ Structural Topology Classifier (Cycle, Fan-In, Fan-Out, Scatter-Gather, Stack) ]
               │
       ┌───────┴──────────────────────┐
       ▼                              ▼
[ Streamlit Analyst Dashboard ] [ Automated FinCEN SAR Generator ]
(Interactive PyVis Network Graph) (One-Click Regulatory Briefs)
```

---

## Empirical Benchmark Results

Evaluated on the official IBM AML `HI-Small` test partition (**1,015,669 transactions; 1,797 illicit transfers**):

### 1. Transaction-Level Performance

| Model | Architecture | Test PR-AUC | Minority F1 | Precision | Recall | P @ 1,000 | P @ 5,000 |
|---|---|---|---|---|---|---|---|
| **M1: Baseline** | LightGBM (Raw Tabular) | 0.0055 | 0.0132 | 0.0067 | 0.6144 | 0.0060 | 0.0038 |
| **M2: Graph-Augmented** | LightGBM (Causal Graph) | **0.0456** | **0.0967** | **0.0513** | **0.8570** | **0.0430** | **0.0332** |

> **Key Finding:** Adding causal graph features yielded an **8.3x improvement in PR-AUC** (0.0055 $\to$ 0.0456) and increased test recall from 61.4% to 85.7%. At a fixed alert budget of $K = 1,000$, Model M2 delivers **7.2x higher precision** than the raw baseline.

---

### 2. Ring-Level Detection Performance

Evaluated against the **192 true laundering rings** active during the test horizon:
- **Candidate Rings Extracted:** 11,270 components
- **Ring Recall @ Jaccard $\ge 0.5$:** **40.10%** (77 of 192 true rings recovered)
- **Ring Recall @ Jaccard $\ge 0.3$:** **50.52%** (97 of 192 true rings recovered)
- **Ring Precision @ Jaccard $\ge 0.5$:** **0.68%** (77 / 11,270)

> **Operational Insight:** Thresholding on raw edge probabilities causes graph over-segmentation, leading to high candidate volume (11,270 clusters). This demonstrates why **Ring Risk Scoring** and **Alert Budget Controls (Top K)** are essential for human investigators.

---

### 3. Detection Breakdown by Laundering Topology

| Laundering Pattern | True Rings in Test | Detected (Jaccard $\ge 0.5$) | Recall ($\ge 0.5$) | Detected (Jaccard $\ge 0.3$) | Recall ($\ge 0.3$) | Behavioral Mechanism |
|---|---|---|---|---|---|---|
| **CYCLE** | 25 | 14 | 56.0% | 20 | **80.0%** | Circular transfers create strong reciprocal flow signals. |
| **RANDOM** | 17 | 11 | 64.7% | 14 | **82.4%** | Irregular pairings trigger low historical pair counts and amount anomalies. |
| **SCATTER-GATHER** | 23 | 11 | 47.8% | 14 | **60.9%** | Dispersion and reconsolidation flagged via intermediary pass-through flow balance. |
| **GATHER-SCATTER** | 39 | 20 | 51.3% | 23 | **59.0%** | Aggregation into staging accounts followed by disbursement caught via degree ratios. |
| **FAN-OUT** | 21 | 7 | 33.3% | 12 | **57.1%** | Structuring/smurfing flagged via elevated source counterparty counts. |
| **FAN-IN** | 20 | 11 | 55.0% | 11 | **55.0%** | Funneling into collector accounts flagged via elevated destination source counts. |
| **BIPARTITE** | 21 | 2 | 9.5% | 2 | **9.5%** | Disconnected two-sided transfers mimic standard merchant clearing. |
| **STACK** | 26 | 1 | 3.8% | 1 | **3.8%** | Linear daisy-chains ($A \to B \to C \to D$) lack circularity or branching; primary blind spot for local features. |

> **Theoretical Implication:** Local graph features excel at cycles (80%) and fan structures (~58%), but struggle with linear stacks (3.85%). This structural blind spot provides strong empirical motivation for multi-hop Graph Neural Networks over simple 1-hop tabular features.

---

## Methodological Rigor

1. **Strict Chronological Splitting:**
   - 60% Train (3,047,007 rows), 20% Validation (1,015,669 rows), 20% Test (1,015,669 rows) ordered strictly by timestamp. Random splitting is prohibited.
2. **Causal Features by Construction:**
   - Every feature computed at timestamp $t$ uses exclusively transactions occurring prior to $t$. Verified via unit tests (`test_features_are_future_invariant`).
3. **Validation-Only Decision Thresholding:**
   - Decision boundaries are tuned on validation minority F1 and frozen before evaluating on test data.

---

## Quick Start & Reproduction

### Prerequisites
```bash
git clone https://github.com/dhruv0402/mule-ring-detector.git
cd mule-ring-detector

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run Tests
```bash
make test
```

### Download Benchmark & Run Complete Pipeline
```bash
# 1. Download official IBM AML dataset (requires ~/.kaggle/access_token or kaggle auth)
make data

# 2. Ingest, extract causal features, train baselines, extract rings, and evaluate
make ingest features train-baseline rings evaluate

# 3. Launch the interactive Streamlit analyst dashboard
make app
```

*(Note: For instant local testing without downloading the 5M dataset, run `make mock-data` first to generate a synthetic benchmark matching the IBM schema).*

---

## Interactive Analyst Dashboard

The Streamlit interface (`src/mulering/app/streamlit_app.py`) provides:
- **Prioritized Case Queue:** Candidate rings ranked by composite risk score with dynamic alert budget controls.
- **Interactive Network Visualizer:** Rendered with PyVis, displaying directional flow arrows, transaction volumes, and timestamps.
- **Automated SAR Generator:** Formats detected rings into formal FinCEN-style Suspicious Activity Report narratives, exportable with one click.
- **Account Profile Inspector:** Deep-dive inspection of any suspect entity's incoming/outgoing volume and counterparty diversity.

---

## Honest Limitations

1. **Synthetic Simulator Nuances:** Data originates from IBM's AMLworld generator (Altman et al., 2023). While realistic, real banking data features noisier labels, delayed SAR outcomes, and benign business structures (e.g., payroll resembling fan-out).
2. **Ring Precision:** Connected component extraction produces 11,270 clusters for 192 true rings. Alert budget controls and ring ranking are required to filter noise.
3. **Absence of Higher-Order Multi-Hop Features:** The current pipeline implements 1-hop degree and 2-cycles, but omits 3-hop and 4-hop cycle detection algorithms. Adding multi-hop cycle searches is the designated path to published benchmark parity (~60% F1).

---

## Citations

- Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023.
- Egressy et al., *Provably Powerful Graph Neural Networks for Directed Multigraphs*, AAAI 2024.
