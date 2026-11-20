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
