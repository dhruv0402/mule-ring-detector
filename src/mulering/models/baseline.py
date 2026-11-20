"""LightGBM baselines (M1: raw features, M2: + account/graph features)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from mulering.config import load_config
from mulering.data.splits import temporal_split
from mulering.evaluation.metrics import best_f1_threshold, minority_prf, pr_auc


RAW_FEATURES = [
    "log_amount_sent",
    "log_amount_recv",
    "same_currency",
    "same_bank",
    "hour",
    "dow",
]

GRAPH_FEATURES = RAW_FEATURES + [
    "src_out_count_before",
    "src_out_sum_before",
    "dst_in_count_before",
    "dst_in_sum_before",
    "pair_count_before",
    "src_hours_since_last_out",
    "amount_vs_src_mean",
    "src_in_sum_before",
    "src_in_count_before",
    "dst_out_sum_before",
    "dst_out_count_before",
    "reverse_pair_count_before",
    "src_distinct_dst_before",
    "dst_distinct_src_before",
    "src_flow_balance",
    "dst_flow_balance",
    "src_fanout_ratio",
    "dst_fanin_ratio",
]


def train_lgbm(X_tr, y_tr, X_va, y_va, params: dict | None = None, seed: int = 42):
    pos = max(int(np.sum(y_tr)), 1)
    neg = len(y_tr) - pos
    scale_weight = float(neg / pos) if pos > 0 else 1.0

    p = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "scale_pos_weight": scale_weight,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        p.update(params)

    model = lgb.LGBMClassifier(**p)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def run_training(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    reports_dir = Path(cfg.get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    features_path = processed_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run 'make features' first.")

    print(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)

    # Chronological train/val/test split
    fractions = tuple(cfg["split"]["fractions"])
    tr_df, va_df, te_df = temporal_split(df, fractions=fractions)
    print(f"Split sizes: Train={len(tr_df):,}, Val={len(va_df):,}, Test={len(te_df):,}")
    print(f"Laundering cases: Train={int(tr_df['label'].sum())}, Val={int(va_df['label'].sum())}, Test={int(te_df['label'].sum())}")

    y_tr, y_va, y_te = tr_df["label"].to_numpy(), va_df["label"].to_numpy(), te_df["label"].to_numpy()

    # Train M1 (Raw Features)
    print("\n--- Training Model M1: Tabular Raw Features ---")
    X_tr_m1, X_va_m1, X_te_m1 = tr_df[RAW_FEATURES], va_df[RAW_FEATURES], te_df[RAW_FEATURES]
    m1 = train_lgbm(X_tr_m1, y_tr, X_va_m1, y_va, cfg.get("model", {}).get("lightgbm"), seed=cfg.get("seed", 42))

    preds_va_m1 = m1.predict_proba(X_va_m1)[:, 1]
    preds_te_m1 = m1.predict_proba(X_te_m1)[:, 1]
    th_m1, f1_va_m1 = best_f1_threshold(y_va, preds_va_m1)
    prf_te_m1 = minority_prf(y_te, preds_te_m1, th_m1)
    auc_te_m1 = pr_auc(y_te, preds_te_m1)
    print(f"M1 Test: PR-AUC = {auc_te_m1:.4f}, F1 = {prf_te_m1['f1']:.4f} (Precision={prf_te_m1['precision']:.4f}, Recall={prf_te_m1['recall']:.4f})")

    # Train M2 (Graph-Augmented Features)
    print("\n--- Training Model M2: Causal Graph-Augmented Features ---")
    available_graph_feats = [c for c in GRAPH_FEATURES if c in tr_df.columns]
    X_tr_m2, X_va_m2, X_te_m2 = tr_df[available_graph_feats], va_df[available_graph_feats], te_df[available_graph_feats]
    m2 = train_lgbm(X_tr_m2, y_tr, X_va_m2, y_va, cfg.get("model", {}).get("lightgbm"), seed=cfg.get("seed", 42))

    preds_va_m2 = m2.predict_proba(X_va_m2)[:, 1]
    preds_te_m2 = m2.predict_proba(X_te_m2)[:, 1]
    th_m2, f1_va_m2 = best_f1_threshold(y_va, preds_va_m2)
    prf_te_m2 = minority_prf(y_te, preds_te_m2, th_m2)
    auc_te_m2 = pr_auc(y_te, preds_te_m2)
    print(f"M2 Test: PR-AUC = {auc_te_m2:.4f}, F1 = {prf_te_m2['f1']:.4f} (Precision={prf_te_m2['precision']:.4f}, Recall={prf_te_m2['recall']:.4f})")

    # Save threshold and evaluation artifacts
    thresh_info = {
        "m1_raw": {"threshold": float(th_m1), "val_f1": float(f1_va_m1), "test_pr_auc": float(auc_te_m1), "test_f1": float(prf_te_m1["f1"])},
        "m2_graph": {"threshold": float(th_m2), "val_f1": float(f1_va_m2), "test_pr_auc": float(auc_te_m2), "test_f1": float(prf_te_m2["f1"])},
        "primary_model": "m2_graph",
        "primary_threshold": float(th_m2),
    }
    with open(reports_dir / "threshold.json", "w") as f:
        json.dump(thresh_info, f, indent=2)

    # Save test predictions table for ring extraction & downstream analysis
    te_out = te_df.copy()
    te_out["score_m1"] = preds_te_m1
    te_out["score_m2"] = preds_te_m2
    te_out["score"] = preds_te_m2  # primary score
    te_out.to_parquet(reports_dir / "test_predictions.parquet", index=False)

    # Save models
    joblib.dump(m1, reports_dir / "model_m1.joblib")
    joblib.dump(m2, reports_dir / "model_m2.joblib")
    print(f"\nArtifacts saved to {reports_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_training(args.config)
