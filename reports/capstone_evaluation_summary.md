# Capstone Evaluation Summary: Graph-Based Money Mule Ring Detection

**Project:** Graph-Based Detection of Money Mule Rings in Financial Transaction Networks  
**Author:** Dhruv Gourisaria  
**Repository:** [github.com/dhruv0402/mule-ring-detector](https://github.com/dhruv0402/mule-ring-detector)  
**Dataset:** Official IBM AML Benchmark (`HI-Small`: 5,078,345 transactions, 5,177 illicit, 370 ground-truth rings)  
**Date:** November 2026  

---

## 1. Executive Summary

Traditional rule-based Anti-Money Laundering (AML) monitoring flags transactions in isolation using static heuristics (threshold amounts, watch-listed entities). Laundering syndicates exploit this blind spot by structuring large sums across chains of mule accounts, keeping individual transactions benign. With an illicit base rate of only ~0.1%, conventional systems either fail to detect coordinated laundering or flood compliance investigators with overwhelming false alarms.

This capstone project implements and rigorously evaluates an end-to-end graph-aware detection framework that:
1. Represents 5.07 million financial transactions as a chronological directed multigraph.
2. Implements causal graph features and a Directed Multigraph Graph Neural Network (GNN).
3. Clusters flagged high-risk activity into candidate mule rings and classifies their structural topology (cycles, fan-in, fan-out, scatter-gather, stacks).
4. Delivers an interactive analyst dashboard in Streamlit with interactive PyVis network graph exploration.

---

## 2. Experimental Rigor & Leakage Prevention

To ensure scientific and engineering validity, this project enforces three strict methodological guardrails:

1. **Strict Chronological Splitting:**
   - **Train:** 3,047,007 transactions (60%) — 2,298 laundering transfers.
   - **Validation:** 1,015,669 transactions (20%) — 1,082 laundering transfers.
   - **Test:** 1,015,669 transactions (20%) — 1,797 laundering transfers.
   - No random splitting is permitted, preventing temporal target leakage.
2. **Causal Features by Construction:**
   - All historical metrics (running volumes, distinct counterparties, net flow balance, reciprocal transfers) are computed using only transactions occurring strictly before timestamp $t$.
   - Automated unit tests (`tests/test_core.py`) verify future-invariance: modifying future rows has zero effect on earlier feature vectors.
3. **Validation-Only Decision Thresholding:**
   - Decision thresholds are selected to maximize minority-class F1 on the validation set, and frozen before evaluating the test set.

---

## 3. Benchmark Results: Transaction-Level (Edge) Evaluation

On the official IBM HI-Small test set (1,015,669 transactions; 1,797 illicit):

| Model | Architecture | Test PR-AUC | Minority F1 | Test Precision | Test Recall | Precision @ 20 |
|---|---|---|---|---|---|---|
| **M1: Baseline** | LightGBM (Raw Tabular Features) | 0.0055 | 0.0132 | 0.0067 | 0.6144 | 0.0500 |
| **M2: Feature-Engineered** | LightGBM (Causal Graph Features) | **0.0456** | **0.0967** | **0.0513** | **0.8570** | **0.1000** |

### Key Findings:
- **Graph Feature Lift:** Adding causal graph features (reciprocal flows, fan-in/fan-out ratios, and net balance dynamics) produced an **8.3x increase in PR-AUC** (from 0.0055 to 0.0456) and lifted recall from 61.4% to 85.7%.
- **Raw Feature Limitations:** Standard transaction attributes (amount, currency match, payment format, hour) fail to distinguish illicit transfers because launderers deliberately mimic benign distribution parameters.

---

## 4. Ring-Level Evaluation (Investigator-Centric)

AML investigators review *cases*, not isolated table rows. This project is evaluated directly against the 192 true laundering rings active during the test horizon.

### Ring Recovery Metrics:
- **True Rings Active in Test Horizon:** 192
- **Candidate Rings Extracted:** 11,270
- **Ring Recall (Jaccard $\ge 0.5$):** **40.10%** (77 of 192 complete rings recovered)
- **Ring Recall (Jaccard $\ge 0.3$):** **50.52%** (97 of 192 rings recovered)

### Recall Breakdown by Laundering Pattern:
| Laundering Topology | Planted in Test | Successfully Detected | Pattern Recall Rate | Analytical Insight |
|---|---|---|---|---|
| **CYCLE** | 25 | 20 | **80.00%** | Closed loops produce strong reciprocal signals that graph features detect easily. |
| **RANDOM** | 17 | 14 | **82.35%** | Highly active non-standard counterparty pairs surface rapidly. |
| **SCATTER-GATHER** | 23 | 14 | **60.87%** | Intermediary pass-through behavior flags the dispersion and consolidation phases. |
| **GATHER-SCATTER** | 39 | 23 | **58.97%** | Concentration into staging accounts before secondary dispersion is largely caught. |
| **FAN-OUT** | 21 | 12 | **57.14%** | Structuring/smurfing behavior flagged via elevated out-degree counterparty ratios. |
| **FAN-IN** | 20 | 11 | **55.00%** | Aggregation into collector accounts detected via high in-degree ratios. |
| **BIPARTITE** | 21 | 2 | **9.52%** | Disconnected two-sided transfers mimic standard commercial vendor clearing. |
| **STACK** | 26 | 1 | **3.85%** | Linear daisy-chains ($A \to B \to C \to D$) lack branching or circularity; primary blind spot for local features. |

> **Key Capstone Insight for Examiners:** The per-topology breakdown exposes a critical insight: *local graph features excel at cycles (80%) and fan structures (~58%), but struggle with linear stacks (3.85%)*. This provides strong theoretical justification for multi-hop GNN message passing over simple 1-hop tabular features.

---

## 5. Analyst Dashboard & Demo Architecture

The interactive Streamlit application (`src/mulering/app/streamlit_app.py`) provides:
1. **Prioritized Case Queue:** Ranked list of candidate mule rings with alert budget controls (Top $K$ slider).
2. **Interactive Network Graph:** Powered by PyVis, allowing investigators to visually trace money flows with directional arrows, transaction amounts, and timestamps.
3. **Topology Classification & Explanation:** Plain-language case briefs explaining why the ring was flagged (e.g., *"Structural cycle pattern involving 4 accounts moving $38,000 over 12 hours"*).
4. **Model Performance & Rigor Tab:** Live scorecard showing edge-level and ring-level evaluation metrics.

---

## 6. How to Run

```bash
# 1. Download official benchmark data
make data

# 2. Ingest, extract causal features, train baselines, extract rings, and evaluate
make ingest features train-baseline rings evaluate

# 3. Launch the interactive analyst app
make app
```
