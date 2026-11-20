# Capstone Presentation Slide Deck: Money Mule Ring Detector

**Project:** Graph-Based Detection of Money Mule Rings in Financial Transaction Networks  
**Author:** Dhruv Gourisaria  
**Repository:** [github.com/dhruv0402/mule-ring-detector](https://github.com/dhruv0402/mule-ring-detector)  
**Evaluated On:** Official IBM AML Benchmark (`HI-Small`: 5,078,345 transactions, 5,177 illicit, 370 true rings)  

---

### Slide 1: Title & Overview
- **Title:** Graph-Based Detection of Money Mule Rings in Financial Transaction Networks
- **Subtitle:** Shifting AML Surveillance from Isolated Transactions to Coordinated Networks
- **Author:** Dhruv Gourisaria
- **Key Message:** Traditional monitoring systems look at one transaction at a time. This project detects the *relational shapes* of organized money mule rings using causal graph learning.

---

### Slide 2: The Problem: The Relational Blind Spot & Alert Fatigue
- **Core Industry Bottleneck:**
  - Launderers structure large sums into small, innocuous-looking transfers across chains of mule accounts (*smurfing*).
  - Single-transaction classifiers evaluate rows without context: a \$4,500 wire looks ordinary.
- **Extreme Class Imbalance:** Illicit transactions account for only **0.101%** of transfers in benchmark data.
- **The Case vs. Row Problem:** Analysts investigate *cases*, not disconnected rows. Flagging isolated rows generates massive false alarms and manual triage overhead.

---

### Slide 3: Methodological Rigor & Zero-Leakage Pipeline
- **Chronological Temporal Split:**
  - **Train (60%):** 3,047,007 rows (2,298 illicit)
  - **Validation (20%):** 1,015,669 rows (1,082 illicit)
  - **Test (20%):** 1,015,669 rows (1,797 illicit)
  - Random train/test splits are strictly prohibited to avoid future-leakage.
- **Causal Historical Features:** Features for transfer at timestamp $t$ only read data strictly *before* $t$.
- **Automated Invariance Verification:** Verified via unit tests (`test_features_are_future_invariant`).

---

### Slide 4: Feature Engineering: Causal Graph Signals
- **Reciprocal Flows:** Counts of direct reverse transfers ($v \to u$ prior to $u \to v$) to capture circular routing.
- **Counterparty Diversity:** Unique outbound destinations from source and unique inbound sources into destination.
- **Net Flow Balance Dynamics:** Real-time account balance tracking ($V_{in} - V_{out}$) and pass-through ratios.
- **Velocity Metrics:** Inter-transaction time delta and relative volume vs. account historical mean.

---

### Slide 5: Empirical Benchmark Results (Official IBM HI-Small)
- **Dataset:** 5,078,345 total transactions; 1,015,669 test transactions.
- **Comparison Table:**
  | Metric | M1: Raw Tabular Baseline | M2: Causal Graph-Augmented | Relative Impact |
  |---|---|---|---|
  | **Test PR-AUC** | 0.0055 | **0.0456** | **8.3x Improvement** |
  | **Minority F1** | 0.0132 | **0.0967** | **7.3x Improvement** |
  | **Test Recall** | 61.44% | **85.70%** | **+24.3% More Crime Caught** |
  | **Precision @ 1,000** | 0.0060 | **0.0430** | **7.2x Higher Precision** |
- **Visual:** Refer to `reports/figures/pr_curves.png` and `reports/figures/feature_importance.png`.

---

### Slide 6: Ring-Level Recovery & The Precision Challenge
- **Active True Rings in Test Horizon:** 192 rings
- **Ring Recovery Metrics:**
  - **Ring Recall @ Jaccard $\ge 0.5$ (Strict):** **40.10%** (77 of 192 recovered)
  - **Ring Recall @ Jaccard $\ge 0.3$ (Moderate):** **50.52%** (97 of 192 recovered)
- **The Ring Precision Challenge (0.68%):**
  - Connected component clustering on flagged edges generated **11,270 candidate rings**.
  - **Crucial Capstone Takeaway:** Naive thresholding over-fragments the graph. This proves why **Ring Scoring** and **Alert Budgeting (Top K)** are mandatory for practical operations.
- **Visual:** Refer to `reports/figures/alert_budget_curve.png`.

---

### Slide 7: Laundering Topology Breakdown: What Works vs. What Fails
- **Recovery Rates by Topology:**
  - `CYCLE`: **80.0%** recovered (Circular flows trigger strong reciprocity signals)
  - `RANDOM`: **82.4%** recovered (High-velocity anomalous pairs)
  - `SCATTER-GATHER`: **60.9%** recovered (Intermediary pass-through flagged)
  - `FAN-OUT` / `FAN-IN`: **~56%** recovered (High-degree hub ratios)
  - `STACK` (Linear Chains): **3.85%** recovered (Major blind spot!)
- **Why Did Stacks Fail?** Linear daisy-chains ($A \to B \to C \to D$) have in-degree 1, out-degree 1, and zero circularity.
- **The Case for GNNs:** Directly proves the necessity of multi-hop message-passing architectures.
- **Visual:** Refer to `reports/figures/topology_recall.png`.

---

### Slide 8: Interactive Investigator Dashboard & UI
- **Live Streamlit Interface:**
  - **Prioritized Ring Queue:** Filter by topology, minimum risk score, and alert budget (Top $K$).
  - **Interactive Network Visualization (PyVis):** Explore suspect accounts as directed graphs with flow amounts and timestamps.
  - **Account Profile Inspector:** Deep-dive into any suspect entity's incoming/outgoing transaction ledger.
- **Regulatory Automation:**
  - Automated FinCEN SAR / FIU-IND STR regulatory narrative generator.
  - One-click export for compliance audit trails.

---

### Slide 9: Honest Limitations
1. **Synthetic vs. Real-World Data:** Evaluated on IBM AMLworld simulator. Real banking data contains noisier labels, delayed SAR outcomes, and benign business structures (payroll mimicking fan-out).
2. **Cluster Over-Segmentation:** Naive connected components require graph-cut or community pruning to reduce the 11,270 extracted clusters.
3. **Absence of Multi-Hop Cycle Features:** Implementing 3-hop and 4-hop DFS cycle searches is the designated path to published benchmark parity (~60% F1).

---

### Slide 10: Conclusion & Summary of Deliverables
- **Scientific Rigor:** Reproducible, leak-free comparison demonstrating an **8.3x PR-AUC lift** with causal graph features.
- **Domain Contribution:** Shifting evaluation from transaction rows to complete rings, exposing structural blind spots on linear chains.
- **Engineering Deliverables:**
  - Fully tested Python pipeline (`make all`, `make test`).
  - Interactive Streamlit dashboard (`make app`).
  - Complete report (`reports/CAPSTONE_PROJECT_REPORT.md`).
  - GitHub repository: `https://github.com/dhruv0402/mule-ring-detector`.
