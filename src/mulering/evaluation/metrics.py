"""Edge-level and ring-level metrics. Accuracy is deliberately absent."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y_true, scores) -> float:
    return float(average_precision_score(y_true, scores))


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    """Choose threshold maximising minority-class F1. Call on VALIDATION data only."""
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]), float(f1[i])


def minority_prf(y_true, scores, threshold: float) -> dict:
    y = np.asarray(y_true).astype(bool); pred = np.asarray(scores) >= threshold
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def precision_at_k(y_true, scores, k: int) -> float:
    idx = np.argsort(-np.asarray(scores))[:k]
    return float(np.asarray(y_true)[idx].mean())


def recall_at_k(y_true, scores, k: int) -> float:
    y = np.asarray(y_true); idx = np.argsort(-np.asarray(scores))[:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def ring_match(true_rings: list[set], pred_rings: list[set], thresh: float = 0.5) -> dict:
    """Greedy one-to-one matching of predicted to true rings by account-set Jaccard."""
    pairs = sorted(
        ((jaccard(t, p), ti, pi) for ti, t in enumerate(true_rings) for pi, p in enumerate(pred_rings)),
        reverse=True,
    )
    used_t, used_p, matched = set(), set(), []
    for j, ti, pi in pairs:
        if j < thresh:
            break
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti); used_p.add(pi); matched.append((ti, pi, j))
    return {
        "ring_recall": len(used_t) / max(len(true_rings), 1),
        "ring_precision": len(used_p) / max(len(pred_rings), 1),
        "matched": matched,
    }
