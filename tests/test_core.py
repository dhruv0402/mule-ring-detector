import numpy as np, pandas as pd, pytest
from mulering.data.splits import temporal_split
from mulering.data.ingest import parse_patterns, attach_ring_ids, ring_table, IBM_COLS
from mulering.features.causal import add_basic, add_account_history
from mulering.evaluation.metrics import (best_f1_threshold, minority_prf, precision_at_k,
                                         recall_at_k, ring_match, pr_auc)
from mulering.rings.extract import extract_rings, classify_pattern
import networkx as nx


def toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-09-01", periods=n, freq="h")
    accts = [f"B1_A{i}" for i in range(6)]
    df = pd.DataFrame({
        "edge_id": range(n), "ts": ts,
        "src": rng.choice(accts, n), "dst": rng.choice(accts, n),
        "src_bank": "B1", "dst_bank": "B1",
        "amount_sent": rng.uniform(10, 1000, n), "amount_recv": rng.uniform(10, 1000, n),
        "cur_sent": "USD", "cur_recv": "USD", "label": 0,
    })
    return df


def test_temporal_split_is_ordered_and_disjoint():
    df = toy()
    tr, va, te = temporal_split(df)
    assert tr["ts"].max() <= va["ts"].min() and va["ts"].max() <= te["ts"].min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_features_are_future_invariant():
    df = toy()
    base = add_account_history(add_basic(df))
    cut = 30
    df2 = df.copy()
    df2.loc[cut + 1:, "amount_sent"] *= 100          # perturb the FUTURE only
    df2.loc[cut + 1:, "amount_recv"] *= 100
    pert = add_account_history(add_basic(df2))
    cols = ["src_out_count_before", "src_out_sum_before", "dst_in_count_before",
            "dst_in_sum_before", "pair_count_before", "src_hours_since_last_out", "amount_vs_src_mean"]
    pd.testing.assert_frame_equal(base.loc[:cut, cols], pert.loc[:cut, cols])


def test_metrics_known_case():
    y = np.array([0, 0, 0, 1, 1, 0]); s = np.array([.1, .2, .3, .9, .8, .4])
    assert pr_auc(y, s) == pytest.approx(1.0)
    t, f1 = best_f1_threshold(y, s)
    assert f1 == pytest.approx(1.0)
    assert minority_prf(y, s, t)["recall"] == 1.0
    assert precision_at_k(y, s, 2) == 1.0 and recall_at_k(y, s, 2) == 1.0


def test_ring_match_and_jaccard():
    true = [{"a", "b", "c"}, {"x", "y"}]
    pred = [{"a", "b", "c", "d"}, {"q", "r"}]
    r = ring_match(true, pred, 0.5)
    assert r["ring_recall"] == 0.5 and r["ring_precision"] == 0.5


def test_patterns_parser_and_attach(tmp_path):
    f = tmp_path / "pat.txt"
    f.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT\n"
        "2022/09/01 00:20,011,AAA,012,BBB,100.0,US Dollar,100.0,US Dollar,Wire,1\n"
        "2022/09/01 00:25,011,AAA,013,CCC,200.0,US Dollar,200.0,US Dollar,Wire,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n")
    rings = parse_patterns(f)
    assert len(rings) == 1 and rings[0]["pattern_type"] == "FAN-OUT" and len(rings[0]["rows"]) == 2
    df = pd.DataFrame({
        "edge_id": [0, 1], "ts": pd.to_datetime(["2022-09-01 00:20", "2022-09-01 00:25"]),
        "src": ["011_AAA", "011_AAA"], "dst": ["012_BBB", "013_CCC"],
        "amount_sent": [100.0, 200.0], "label": [1, 1]})
    out = attach_ring_ids(df, rings)
    assert out["ring_id"].notna().all()
    t = ring_table(out)
    assert t.loc[0, "n_accounts"] == 3


def test_extract_recovers_planted_cycle_and_fanin():
    rows = []
    cyc = ["c1", "c2", "c3"]
    for i in range(3):
        rows.append(("cyc_" + cyc[i], cyc[i], cyc[(i + 1) % 3]))
    for i in range(4):
        rows.append(("fan", f"s{i}", "sink"))
    flagged = pd.DataFrame({
        "edge_id": range(len(rows)), "src": [r[1] for r in rows], "dst": [r[2] for r in rows],
        "score": 0.9, "amount_sent": 100.0, "ts": pd.Timestamp("2022-09-01")})
    rings = extract_rings(flagged)
    kinds = sorted(r["pattern_guess"] for r in rings)
    assert kinds == ["cycle", "fan-in"]
