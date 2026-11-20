"""Turn flagged edges into candidate rings and give each a structural pattern guess."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import networkx as nx
import pandas as pd
from mulering.config import load_config


def classify_pattern(g: nx.DiGraph) -> str:
    """Rule-based structural label for a ring's directed subgraph."""
    n = g.number_of_nodes()
    if n < 2:
        return "other"
    if any(len(c) > 1 for c in nx.strongly_connected_components(g)):
        return "cycle"
    ins = dict(g.in_degree()); outs = dict(g.out_degree())
    sinks = [v for v in g if outs[v] == 0]; sources = [v for v in g if ins[v] == 0]
    if len(sinks) == 1 and len(sources) >= 3 and g.number_of_edges() >= len(sources):
        mids = [v for v in g if ins[v] and outs[v]]
        return "scatter-gather" if mids and len(sources) == 1 else "fan-in"
    if len(sources) == 1 and len(sinks) >= 3:
        return "fan-out"
    if len(sources) == 1 and len(sinks) == 1 and n >= 4:
        return "scatter-gather"
    return "other"


def extract_rings(flagged: pd.DataFrame, max_accounts: int = 300) -> list[dict]:
    """flagged: rows with src, dst, score, amount_sent, ts, edge_id. Returns ring dicts."""
    und = nx.Graph()
    und.add_edges_from(zip(flagged["src"], flagged["dst"]))
    rings = []
    for comp in nx.connected_components(und):
        groups = [comp]
        if len(comp) > max_accounts:
            try:
                import community as community_louvain  # python-louvain
                part = community_louvain.best_partition(und.subgraph(comp), random_state=0)
                by = {}
                for node, c in part.items():
                    by.setdefault(c, set()).add(node)
                groups = list(by.values())
            except ImportError:
                pass
        for grp in groups:
            if len(grp) < 2:
                continue
            sub = flagged[flagged["src"].isin(grp) & flagged["dst"].isin(grp)]
            if sub.empty:
                continue
            g = nx.DiGraph(); g.add_edges_from(zip(sub["src"], sub["dst"]))
            pattern = classify_pattern(g)
            duration_hrs = (sub["ts"].max() - sub["ts"].min()).total_seconds() / 3600.0
            total_amt = float(sub["amount_sent"].sum())
            reasons = f"Structural {pattern} pattern involving {len(grp)} accounts moving ${total_amt:,.2f} over {duration_hrs:.1f} hours."

            rings.append({
                "accounts": set(grp),
                "edge_ids": set(sub["edge_id"]),
                "pattern_guess": pattern,
                "ring_score": float(sub["score"].nlargest(5).mean()),
                "n_accounts": len(grp),
                "total_amount": total_amt,
                "start_ts": sub["ts"].min(),
                "end_ts": sub["ts"].max(),
                "top_reasons": reasons,
            })
    rings.sort(key=lambda r: -r["ring_score"])
    for i, r in enumerate(rings):
        r["pred_ring_id"] = i
    return rings


def run_extraction(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    reports_dir = Path(cfg.get("reports_dir", "reports"))
    preds_file = reports_dir / "test_predictions.parquet"
    thresh_file = reports_dir / "threshold.json"

    if not preds_file.exists():
        raise FileNotFoundError(f"Missing {preds_file}. Run 'make train-baseline' first.")

    df = pd.read_parquet(preds_file)

    threshold = 0.5
    if thresh_file.exists():
        with open(thresh_file) as f:
            t_data = json.load(f)
            threshold = t_data.get("primary_threshold", 0.5)

    print(f"Loaded {len(df):,} test transactions.")
    print(f"Using alert threshold: {threshold:.4f}")

    flagged = df[df["score"] >= threshold].copy()
    if len(flagged) < 2:
        print("Warning: Threshold yielded fewer than 2 edges. Falling back to top 1% score threshold.")
        q = df["score"].quantile(0.99)
        flagged = df[df["score"] >= q].copy()

    print(f"Flagged {len(flagged):,} suspicious edges.")
    max_accts = cfg.get("rings", {}).get("max_ring_accounts", 300)
    rings = extract_rings(flagged, max_accounts=max_accts)
    print(f"Extracted {len(rings):,} candidate mule rings.")

    # Convert sets to lists for Parquet storage
    rows = []
    for r in rings:
        row = dict(r)
        row["accounts"] = list(r["accounts"])
        row["edge_ids"] = list(r["edge_ids"])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = reports_dir / "rings.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved candidate rings to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mule rings from scored transactions.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_extraction(args.config)
