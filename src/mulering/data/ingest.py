"""Ingest IBM AML files into a standard schema and parse ring ground truth.

NOTE: the patterns-file format below (BEGIN/END LAUNDERING ATTEMPT blocks containing
transaction rows) is my understanding of the IBM files. Open the real file and confirm
before trusting the parser.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# The IBM CSV repeats the header name "Account", so we assign names explicitly.
IBM_COLS = [
    "ts", "src_bank", "src_acct", "dst_bank", "dst_acct",
    "amount_recv", "cur_recv", "amount_sent", "cur_sent", "pay_format", "label",
]


def load_ibm_transactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=0, names=IBM_COLS, dtype={
        "src_bank": "string", "src_acct": "string", "dst_bank": "string", "dst_acct": "string",
        "cur_recv": "category", "cur_sent": "category", "pay_format": "category", "label": "int8",
    })
    df["ts"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
    df["src"] = df["src_bank"] + "_" + df["src_acct"]
    df["dst"] = df["dst_bank"] + "_" + df["dst_acct"]
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df.insert(0, "edge_id", df.index.astype("int64"))
    return df


def parse_patterns(path: str | Path) -> list[dict]:
    """Return a list of rings: {ring_id, pattern_type, rows: [list of csv fields]}."""
    rings, cur, rid = [], None, 0
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                head = line.split(" - ", 1)[1] if " - " in line else "UNKNOWN"
                ptype = head.split(":")[0].strip()
                cur = {"ring_id": rid, "pattern_type": ptype, "rows": []}
                rid += 1
            elif line.startswith("END LAUNDERING ATTEMPT"):
                if cur is not None:
                    rings.append(cur)
                cur = None
            elif cur is not None:
                cur["rows"].append(line.split(","))
    return rings


def attach_ring_ids(df: pd.DataFrame, rings: list[dict]) -> pd.DataFrame:
    """Add evaluation-only columns ring_id / pattern_type by matching transaction rows.

    Matching key: (ts, src, dst, amount_sent). Evaluation use only: never a model input.
    """
    recs = []
    for r in rings:
        for f in r["rows"]:
            if len(f) < 11:
                continue
            recs.append({
                "ts": pd.to_datetime(f[0], format="%Y/%m/%d %H:%M"),
                "src": f"{f[1]}_{f[2]}", "dst": f"{f[3]}_{f[4]}",
                "amount_sent": float(f[7]),
                "ring_id": r["ring_id"], "pattern_type": r["pattern_type"],
            })
    key = ["ts", "src", "dst", "amount_sent"]
    m = pd.DataFrame(recs).drop_duplicates(key)
    out = df.merge(m, on=key, how="left")
    return out


def ring_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.dropna(subset=["ring_id"]).groupby("ring_id")
    rows = []
    for rid, d in g:
        accts = set(d["src"]) | set(d["dst"])
        rows.append({
            "ring_id": int(rid), "pattern_type": d["pattern_type"].iloc[0],
            "accounts": accts, "edge_ids": set(d["edge_id"]),
            "n_accounts": len(accts), "n_edges": len(d),
            "start_ts": d["ts"].min(), "end_ts": d["ts"].max(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from mulering.config import load_config
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(ap.parse_args().config)["dataset"]
    raw = Path(cfg["raw_dir"]); out = Path(cfg["processed_dir"]); out.mkdir(parents=True, exist_ok=True)
    df = load_ibm_transactions(raw / cfg["trans_file"])
    rings = parse_patterns(raw / cfg["patterns_file"])
    df = attach_ring_ids(df, rings)
    df.to_parquet(out / "transactions.parquet", index=False)
    ring_table(df).to_pickle(out / "rings_truth.pkl")
    print(f"{len(df):,} transactions, {int(df['label'].sum()):,} laundering, {len(rings):,} true rings")
