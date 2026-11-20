"""Causal (no-future-leak) features. A row's features use only rows strictly before it.

Input must be sorted by time. Rows sharing an identical timestamp are treated in file order;
if you need strict time ordering, break ties with edge_id and document it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_basic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount_sent"] = np.log1p(out["amount_sent"])
    out["log_amount_recv"] = np.log1p(out["amount_recv"])
    out["same_currency"] = (out["cur_sent"].astype(str) == out["cur_recv"].astype(str)).astype("int8")
    out["same_bank"] = (out["src_bank"].astype(str) == out["dst_bank"].astype(str)).astype("int8")
    out["hour"] = out["ts"].dt.hour.astype("int8")
    out["dow"] = out["ts"].dt.dayofweek.astype("int8")
    return out


def add_account_history(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account running counts/sums BEFORE the current transaction (strictly past only)."""
    out = df.copy()
    out["src_out_count_before"] = out.groupby("src").cumcount()
    out["src_out_sum_before"] = out.groupby("src")["amount_sent"].cumsum() - out["amount_sent"]
    out["dst_in_count_before"] = out.groupby("dst").cumcount()
    out["dst_in_sum_before"] = out.groupby("dst")["amount_recv"].cumsum() - out["amount_recv"]
    # how many times this exact (src,dst) pair transacted before
    out["pair_count_before"] = out.groupby(["src", "dst"]).cumcount()
    # time since the sender's previous outgoing transaction (hours); NaN for first
    prev = out.groupby("src")["ts"].shift(1)
    out["src_hours_since_last_out"] = (out["ts"] - prev).dt.total_seconds() / 3600.0
    # amount relative to sender's historical mean outgoing amount
    mean_before = out["src_out_sum_before"] / out["src_out_count_before"].replace(0, np.nan)
    out["amount_vs_src_mean"] = out["amount_sent"] / mean_before
    return out


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal graph structure features: reciprocal flows (2-cycles), fan-in/fan-out ratios,

    flow balances, and counterparty diversity strictly before the current transaction.
    """
    out = df.copy()
    n = len(df)

    acct_in_sum: dict[str, float] = {}
    acct_in_count: dict[str, int] = {}
    acct_out_sum: dict[str, float] = {}
    acct_out_count: dict[str, int] = {}
    acct_unique_dst: dict[str, set[str]] = {}
    acct_unique_src: dict[str, set[str]] = {}
    pair_counts: dict[tuple[str, str], int] = {}

    src_in_sum_before = np.zeros(n, dtype=np.float64)
    src_in_count_before = np.zeros(n, dtype=np.int32)
    dst_out_sum_before = np.zeros(n, dtype=np.float64)
    dst_out_count_before = np.zeros(n, dtype=np.int32)
    reverse_pair_count_before = np.zeros(n, dtype=np.int32)
    src_distinct_dst_before = np.zeros(n, dtype=np.int32)
    dst_distinct_src_before = np.zeros(n, dtype=np.int32)

    srcs = out["src"].to_numpy()
    dsts = out["dst"].to_numpy()
    sent_amts = out["amount_sent"].to_numpy(dtype=np.float64)
    recv_amts = out["amount_recv"].to_numpy(dtype=np.float64)

    for i in range(n):
        u = srcs[i]
        v = dsts[i]
        amt_s = sent_amts[i]
        amt_r = recv_amts[i]

        # Extract values strictly BEFORE updating with row i
        src_in_sum_before[i] = acct_in_sum.get(u, 0.0)
        src_in_count_before[i] = acct_in_count.get(u, 0)
        dst_out_sum_before[i] = acct_out_sum.get(v, 0.0)
        dst_out_count_before[i] = acct_out_count.get(v, 0)

        reverse_pair_count_before[i] = pair_counts.get((v, u), 0)

        src_distinct_dst_before[i] = len(acct_unique_dst.get(u, ()))
        dst_distinct_src_before[i] = len(acct_unique_src.get(v, ()))

        # Update running state
        acct_out_sum[u] = acct_out_sum.get(u, 0.0) + amt_s
        acct_out_count[u] = acct_out_count.get(u, 0) + 1
        if u not in acct_unique_dst:
            acct_unique_dst[u] = {v}
        else:
            acct_unique_dst[u].add(v)

        acct_in_sum[v] = acct_in_sum.get(v, 0.0) + amt_r
        acct_in_count[v] = acct_in_count.get(v, 0) + 1
        if v not in acct_unique_src:
            acct_unique_src[v] = {u}
        else:
            acct_unique_src[v].add(u)

        pair_counts[(u, v)] = pair_counts.get((u, v), 0) + 1

    out["src_in_sum_before"] = src_in_sum_before
    out["src_in_count_before"] = src_in_count_before
    out["dst_out_sum_before"] = dst_out_sum_before
    out["dst_out_count_before"] = dst_out_count_before
    out["reverse_pair_count_before"] = reverse_pair_count_before
    out["src_distinct_dst_before"] = src_distinct_dst_before
    out["dst_distinct_src_before"] = dst_distinct_src_before

    out["src_flow_balance"] = out["src_in_sum_before"] - out["src_out_sum_before"]
    out["dst_flow_balance"] = out["dst_in_sum_before"] - out["dst_out_sum_before"]

    out["src_fanout_ratio"] = out["src_distinct_dst_before"] / (out["src_out_count_before"] + 1)
    out["dst_fanin_ratio"] = out["dst_distinct_src_before"] / (out["dst_in_count_before"] + 1)

    return out

