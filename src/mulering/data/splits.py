"""Chronological train/val/test split. Never random."""
from __future__ import annotations
import pandas as pd


def temporal_split(df: pd.DataFrame, fractions=(0.6, 0.2, 0.2), time_col="ts"):
    assert abs(sum(fractions) - 1.0) < 1e-9
    d = df.sort_values(time_col, kind="stable")
    n = len(d)
    i1 = int(n * fractions[0]); i2 = int(n * (fractions[0] + fractions[1]))
    return d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
