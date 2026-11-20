"""Feature engineering pipeline: loads transactions, applies causal transforms, and writes features.parquet."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from mulering.config import load_config
from mulering.features.causal import add_basic, add_account_history, add_graph_features


def build_features(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    trans_file = processed_dir / "transactions.parquet"
    if not trans_file.exists():
        raise FileNotFoundError(f"Missing {trans_file}. Run 'make ingest' first.")

    print(f"Loading transactions from {trans_file}...")
    df = pd.read_parquet(trans_file)

    print("Building causal features...")
    df = add_basic(df)
    df = add_account_history(df)
    df = add_graph_features(df)

    out_file = processed_dir / "features.parquet"
    df.to_parquet(out_file, index=False)
    print(f"Features saved to {out_file} (shape: {df.shape})")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build features for AML detection.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    build_features(args.config)
