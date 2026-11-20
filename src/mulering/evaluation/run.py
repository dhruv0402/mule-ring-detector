"""Run comprehensive evaluation on test set: transaction-level and ring-level metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from mulering.config import load_config
from mulering.evaluation.metrics import (
    best_f1_threshold,
    minority_prf,
    pr_auc,
    precision_at_k,
    recall_at_k,
    ring_match,
)


def run_evaluation(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    reports_dir = Path(cfg.get("reports_dir", "reports"))
    processed_dir = Path(cfg["dataset"]["processed_dir"])

    preds_file = reports_dir / "test_predictions.parquet"
    rings_pred_file = reports_dir / "rings.parquet"
    rings_truth_file = processed_dir / "rings_truth.pkl"
    thresh_file = reports_dir / "threshold.json"

    if not preds_file.exists():
        raise FileNotFoundError(f"Missing {preds_file}. Run 'make train-baseline' first.")

    te_df = pd.read_parquet(preds_file)
    y_true = te_df["label"].to_numpy()
    score_m1 = te_df["score_m1"].to_numpy()
    score_m2 = te_df["score_m2"].to_numpy()
    score_gnn = te_df["score_gnn"].to_numpy() if "score_gnn" in te_df.columns else None

    # Load thresholds
    th_m1, th_m2, th_gnn = 0.5, 0.5, 0.5
    if thresh_file.exists():
        with open(thresh_file) as f:
            t_data = json.load(f)
            th_m1 = t_data.get("m1_raw", {}).get("threshold", 0.5)
            th_m2 = t_data.get("m2_graph", {}).get("threshold", 0.5)
            th_gnn = t_data.get("m3_gnn", {}).get("threshold", 0.5)

    # 1. Edge-level Evaluation
    edge_metrics = {
        "m1_raw": {
            "name": "LightGBM (Tabular Raw)",
            "pr_auc": pr_auc(y_true, score_m1),
            "threshold": th_m1,
            **minority_prf(y_true, score_m1, th_m1),
            "p_at_20": precision_at_k(y_true, score_m1, 20),
            "r_at_20": recall_at_k(y_true, score_m1, 20),
            "p_at_50": precision_at_k(y_true, score_m1, 50),
            "r_at_50": recall_at_k(y_true, score_m1, 50),
        },
        "m2_graph": {
            "name": "LightGBM (Causal Graph)",
            "pr_auc": pr_auc(y_true, score_m2),
            "threshold": th_m2,
            **minority_prf(y_true, score_m2, th_m2),
            "p_at_20": precision_at_k(y_true, score_m2, 20),
            "r_at_20": recall_at_k(y_true, score_m2, 20),
            "p_at_50": precision_at_k(y_true, score_m2, 50),
            "r_at_50": recall_at_k(y_true, score_m2, 50),
        },
    }

    if score_gnn is not None:
        edge_metrics["m3_gnn"] = {
            "name": "Directed Multigraph GNN",
            "pr_auc": pr_auc(y_true, score_gnn),
            "threshold": th_gnn,
            **minority_prf(y_true, score_gnn, th_gnn),
            "p_at_20": precision_at_k(y_true, score_gnn, 20),
            "r_at_20": recall_at_k(y_true, score_gnn, 20),
            "p_at_50": precision_at_k(y_true, score_gnn, 50),
            "r_at_50": recall_at_k(y_true, score_gnn, 50),
        }

    # 2. Ring-level Evaluation
    ring_metrics = {}
    if rings_truth_file.exists() and rings_pred_file.exists():
        truth_rings_df = pd.read_pickle(rings_truth_file)
        pred_rings_df = pd.read_parquet(rings_pred_file)

        # Filter true rings present in the test window
        test_edge_ids = set(te_df["edge_id"])
        test_truth = truth_rings_df[truth_rings_df["edge_ids"].apply(lambda e: len(e & test_edge_ids) > 0)].copy()

        true_acct_sets = [set(a) for a in test_truth["accounts"]]
        pred_acct_sets = [set(a) for a in pred_rings_df["accounts"]]

        match_50 = ring_match(true_acct_sets, pred_acct_sets, thresh=0.5)
        match_30 = ring_match(true_acct_sets, pred_acct_sets, thresh=0.3)

        # Per-pattern recall breakdown
        matched_true_indices = {ti for ti, _, _ in match_30["matched"]}
        pattern_breakdown = {}
        for idx, row in test_truth.reset_index(drop=True).iterrows():
            ptype = row["pattern_type"]
            if ptype not in pattern_breakdown:
                pattern_breakdown[ptype] = {"total": 0, "detected": 0}
            pattern_breakdown[ptype]["total"] += 1
            if idx in matched_true_indices:
                pattern_breakdown[ptype]["detected"] += 1

        for ptype, counts in pattern_breakdown.items():
            counts["recall"] = counts["detected"] / max(counts["total"], 1)

        ring_metrics = {
            "n_true_rings_test": len(test_truth),
            "n_pred_rings": len(pred_rings_df),
            "jaccard_0.5": {
                "ring_recall": match_50["ring_recall"],
                "ring_precision": match_50["ring_precision"],
                "n_matched": len(match_50["matched"]),
            },
            "jaccard_0.3": {
                "ring_recall": match_30["ring_recall"],
                "ring_precision": match_30["ring_precision"],
                "n_matched": len(match_30["matched"]),
            },
            "pattern_recall": pattern_breakdown,
        }

    full_results = {
        "edge_level": edge_metrics,
        "ring_level": ring_metrics,
    }

    out_json = reports_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(full_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 65)
    print("                     CAPSTONE BENCHMARK REPORT")
    print("=" * 65)
    print("--- 1. Transaction-Level Metrics ---")
    print(f"{'Model':<25} {'PR-AUC':<10} {'F1':<10} {'Prec':<10} {'Rec':<10} {'P@20':<10}")
    print("-" * 65)
    for key, m in edge_metrics.items():
        print(f"{m['name']:<25} {m['pr_auc']:<10.4f} {m['f1']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['p_at_20']:<10.4f}")

    print("\n--- 2. Ring-Level Metrics (Test Set) ---")
    if ring_metrics:
        print(f"True Rings in Test: {ring_metrics['n_true_rings_test']}, Predicted Rings: {ring_metrics['n_pred_rings']}")
        print(f"Ring Recall @ Jaccard 0.5: {ring_metrics['jaccard_0.5']['ring_recall']:.2%}")
        print(f"Ring Precision @ Jaccard 0.5: {ring_metrics['jaccard_0.5']['ring_precision']:.2%}")
        print(f"Ring Recall @ Jaccard 0.3: {ring_metrics['jaccard_0.3']['ring_recall']:.2%}")
        print("\nRecall by Laundering Pattern:")
        for ptype, counts in ring_metrics.get("pattern_recall", {}).items():
            print(f"  - {ptype:<16}: {counts['detected']}/{counts['total']} ({counts['recall']:.2%})")
    print("=" * 65)
    print(f"Saved evaluation metrics to {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AML detector.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_evaluation(args.config)
