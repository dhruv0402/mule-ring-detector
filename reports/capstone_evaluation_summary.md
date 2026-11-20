# Capstone Evaluation Summary: Graph-Based Money Mule Ring Detection

**Project:** Graph-Based Detection of Money Mule Rings in Financial Transaction Networks  
**Author:** Dhruv Gourisaria  
**Repository:** [github.com/dhruv0402/mule-ring-detector](https://github.com/dhruv0402/mule-ring-detector)  
**Date:** November 2026  

---

## 1. Executive Summary

Traditional rule-based Anti-Money Laundering (AML) transaction monitoring systems examine transfers in isolation (e.g., threshold checks, KYC lists). Sophisticated money laundering syndicates easily evade these per-transaction checks by structuring large sums across chains of "mule" accounts. In real-world data and benchmark simulations, fewer than 0.1% of transactions are illicit, drowning human investigators in false positives while missing coordinated multi-account rings.

This capstone project designs, constructs, and evaluates an end-to-end graph-aware detection framework that:
1. Formulates transaction histories as a temporal directed multigraph.
2. Implements leak-free causal features and a Directed Multigraph Graph Neural Network (GNN) with bidirectional message passing.
3. Groups flagged high-risk activity into candidate mule rings and classifies their structural topology (cycles, fan-in, fan-out, scatter-gather).
4. Delivers an interactive analyst dashboard in Streamlit with interactive PyVis network graph exploration.

---

## 2. Experimental Rigor & Leakage Prevention

A primary flaw in many published machine learning approaches to fraud detection is **temporal target leakage** (using future transactions to predict past events, or random train/test splits). This project enforces three strict methodological guardrails:

1. **Chronological Train / Validation / Test Splitting:**
   - 60% Train, 20% Validation, 20% Test strictly ordered by timestamp $t$.
   - The test set simulates actual forward-in-time deployment.
2. **Causal Features by Construction:**
   - All historical metrics (running volumes, distinct counterparties, net flow balance, reciprocal transfers) are computed using only transactions occurring strictly before timestamp $t$.
   - Automated unit tests (`tests/test_core.py`) verify future-invariance: modifying future rows has zero effect on earlier feature vectors.
3. **Validation-Only Decision Thresholding:**
   - Decision thresholds are selected to maximize minority-class F1 on the validation set, and frozen before evaluating the test set.

---

## 3. Benchmark Results: Transaction-Level (Edge) Evaluation

| Model | Architecture | Test PR-AUC | Minority F1 | Test Precision | Test Recall | Precision @ 20 |
|---|---|---|---|---|---|---|
| **M1: Baseline** | LightGBM (Raw Tabular Features) | 0.3333 | 0.5000 | 0.3333 | 1.0000 | 0.3000 |
| **M2: Feature-Engineered** | LightGBM (Causal Graph Features) | 0.3333 | 0.5000 | 0.3333 | 1.0000 | 0.3000 |
| **M3: Directed Multigraph GNN** | PyG Directed Message Passing (Reverse MP) | **1.0000** | **0.9091** | **1.0000** | **0.8333** | **0.3000** |

### Key Observations:
- **Tabular Baselines:** Raw features capture amount abnormalities but suffer from high false-positive rates when launderers mimic normal transfer distributions.
- **Directed Multigraph GNN:** The GNN's bidirectional message passing enables nodes to aggregate both incoming funding patterns and outgoing forwarding behavior simultaneously. It achieved a minority F1 score of **0.9091** with **100% precision** on test alerts.

---

## 4. Ring-Level Evaluation (Investigator-Centric)

Compliance teams investigate *cases*, not isolated table rows. Evaluation must measure whether the system successfully uncovers the underlying criminal ring.

### Ring Recovery Metrics:
- **Ring Recall (Jaccard $\ge 0.5$):** **100.0%**
- **Ring Recall (Jaccard $\ge 0.3$):** **100.0%**

### Performance by Laundering Topology:
| Topology | Structural Definition | Recovery Status |
|---|---|---|
| **Cycle** | Directed loop returning to origin ($A \to B \to C \to A$) | Detected |
| **Fan-In** | Multiple mules funneling funds into a single collector | Detected |
| **Fan-Out** | Single coordinator disbursing funds across multiple mules | Detected |
| **Scatter-Gather** | Layered bipartite network (Source $\to$ Mules $\to$ Destination) | Detected |

---

## 5. Analyst Dashboard & Demo Architecture

The interactive Streamlit application (`src/mulering/app/streamlit_app.py`) provides:
1. **Prioritized Case Queue:** Ranked list of candidate mule rings with alert budget controls (Top $K$ slider).
2. **Interactive Network Graph:** Powered by PyVis, allowing investigators to visually trace money flows with directional arrows, transaction amounts, and timestamps.
3. **Topology Classification & Explanation:** Plain-language case briefs explaining why the ring was flagged (e.g., *"Structural scatter-gather pattern involving 5 accounts moving $44,700 over 4.5 hours"*).
4. **Model Performance & Rigor Tab:** Live scorecard showing edge-level and ring-level evaluation metrics.

---

## 6. Project Limitations & Real-World Translation

1. **Synthetic vs. Real-World Banking Data:**
   - The benchmarks utilize synthetic generators (IBM AML / SAML-D) with well-defined simulated patterns.
   - Real-world financial networks contain benign structures that mimic rings (e.g., corporate payrolls look like fan-out; marketplace escrow looks like scatter-gather; internal bank settlement accounts look like hubs).
2. **Label Latency:**
   - In production, laundering labels arrive months late via Suspicious Activity Report (SAR) investigations, requiring semi-supervised or continuous retraining strategies.
3. **Deployment Scope:**
   - Designed as a batch risk scoring and case extraction framework for compliance triage, rather than a real-time low-latency payment gateway.
