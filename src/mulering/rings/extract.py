"""Turn flagged edges into candidate rings and give each a structural pattern guess."""
from __future__ import annotations
import networkx as nx
import pandas as pd


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
            rings.append({
                "accounts": set(grp), "edge_ids": set(sub["edge_id"]),
                "pattern_guess": classify_pattern(g),
                "ring_score": float(sub["score"].nlargest(5).mean()),
                "n_accounts": len(grp), "total_amount": float(sub["amount_sent"].sum()),
                "start_ts": sub["ts"].min(), "end_ts": sub["ts"].max(),
            })
    rings.sort(key=lambda r: -r["ring_score"])
    for i, r in enumerate(rings):
        r["pred_ring_id"] = i
    return rings
