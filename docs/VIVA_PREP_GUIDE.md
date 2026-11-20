# Capstone Viva & Defense Preparation Guide

**Project:** Money Mule Ring Detector  
**Author:** Dhruv Gourisaria  
**Repository:** [github.com/dhruv0402/mule-ring-detector](https://github.com/dhruv0402/mule-ring-detector)  

---

## 1. The 60-Second Elevator Pitch

> *"Traditional Anti-Money Laundering monitoring evaluates transactions in isolation, but money launderers deliberately structure large sums into small, benign transfers across networks of mule accounts. With only 0.1% of transactions being illicit, isolated models miss coordinated networks or bury human analysts in false positives.
>
> In this capstone, I built an end-to-end graph-aware detection framework evaluated on the official IBM AML benchmark of 5.07 million transactions. By engineering causal graph features strictly backward in time, we achieve an 8.3x improvement in PR-AUC over tabular baselines. More importantly, we shift the detection unit from transactions to **rings**, recovering 80% of laundering cycles and 50.5% of overall rings, and providing investigators with an interactive network dashboard and automated regulatory SAR narrative drafting."*

---

## 2. Top 5 Examiner Questions & Exact Defensible Answers

### Q1: *"Why did you use a chronological split instead of 5-fold cross-validation?"*
**Answer:**
> *"In fraud and financial surveillance, random splitting is a fatal methodological error because it causes future target leakage—allowing a model to use future transactions to predict past behavior. In production banking, a model only ever predicts forward into the future. Our 60% train, 20% validation, and 20% test partitions are strictly ordered by timestamp, ensuring zero leakage. We also wrote automated unit tests (`test_features_are_future_invariant`) proving that altering future rows has zero impact on earlier feature values."*

### Q2: *"Why is Accuracy completely absent from your evaluation?"*
**Answer:**
> *"Because of extreme class imbalance: in the IBM benchmark, only 0.101% of transactions are laundering (5,177 out of 5,078,345). A naive trivial model that simply predicts 'Legitimate' for every transaction achieves 99.899% accuracy while catching zero crime. We evaluate exclusively on Precision-Recall AUC (PR-AUC), minority-class F1, and Precision/Recall at fixed alert budgets ($K = 1,000$)."*

### Q3: *"Your ring precision is only 0.68% (77 matches out of 11,270 extracted components). Isn't that too low for real banks?"*
**Answer:**
> *"Yes, and reporting that openly is one of our most important findings. Naive connected-component extraction over-segments the graph because even a single weakly flagged bridge edge can pull in innocent accounts. This empirical finding demonstrates exactly why **Ring Risk Scoring** and **Alert Budget Controls** are necessary. Instead of reviewing 11,000 raw clusters, our Streamlit dashboard ranks rings by mean edge score, allowing a compliance team to review just the top 20 or 50 highest-confidence cases per day."*

### Q4: *"Why did your model catch 80% of Cycles but only 3.8% of linear Stacks?"*
**Answer:**
> *"Because of the boundary of local 1-hop graph features. Cycles create direct reciprocal flows ($v \to u$ following $u \to v$) and fan topologies create high degree disparity, which our causal features (`reverse_pair_count_before`, fan-in ratios) detect immediately. However, linear daisy-chains ($A \to B \to C \to D$) have an in-degree of 1, an out-degree of 1, and zero circularity. Each hop looks completely ordinary in isolation. This failure is significant because it provides empirical justification for multi-hop Graph Neural Networks (GNNs), which pass multi-hop messages across chains."*

### Q5: *"How does this differ from published benchmarks on IBM HI-Small that reach ~60% F1?"*
**Answer:**
> *"Published benchmarks like Egressy et al. (AAAI 2024) utilize multi-hop windowed cycle detection algorithms and deep graph neural network architectures. Our current Model M2 implements 1-hop degree, flow balance, and 2-cycles, which is why our absolute F1 is 0.097. We do not claim state-of-the-art parity; our objective was to rigorously demonstrate the relative lift of graph features over raw tabular features (8.3x PR-AUC lift) under leak-free conditions and provide ring-level topology analysis."*

---

## 3. The 2-Minute Live Demo Script (`make app`)

1. **Overview Scorecard (30s):**
   - Open `make app`.
   - Point out the headline metrics: 1,015,669 test transactions scored, 11,270 candidate rings surfaced.
   - Show the **Alert Budget slider**: slide it from 10 to 25 to show how compliance leads tune queue volume to analyst headcount.

2. **Ring Explorer & Interactive PyVis Graph (45s):**
   - Click on Ring #0 (`CYCLE` or `SCATTER-GATHER`).
   - Show the interactive network diagram: drag nodes, hover over edges to show transaction amounts, timestamps, and risk probabilities.
   - Show the **Mule Account Profile Inspector**: click an account to reveal its incoming vs. outgoing ledger balance.

3. **Automated Regulatory SAR Narrative (30s):**
   - Expand the **"Generate FinCEN Suspicious Activity Report (SAR) Narrative"** drawer.
   - Show the automatically compiled compliance narrative: Part I (Subjects), Part II (Volumes & Dates), Part III (Modus Operandi), Part IV (Red Flags).
   - Click the download button to show how investigators export the audit brief.

4. **Capstone Benchmark Tab (15s):**
   - Open Tab 3 to show the formal evaluation table: M1 (Raw) vs. M2 (Causal Graph), and the per-topology recall breakdown.
   - Conclude on the honest limitations (linear stack blind spot motivating multi-hop GNNs).
