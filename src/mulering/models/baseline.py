"""LightGBM baselines (A: raw features, B: + account/graph features)."""
from __future__ import annotations
import lightgbm as lgb
import numpy as np


def train_lgbm(X_tr, y_tr, X_va, y_va, params: dict | None = None, seed: int = 42):
    pos = max(int(np.sum(y_tr)), 1)
    p = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63,
             scale_pos_weight=(len(y_tr) - pos) / pos, random_state=seed, n_jobs=-1)
    p.update(params or {})
    model = lgb.LGBMClassifier(**p)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model
