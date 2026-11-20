"""PyTorch Geometric Edge Classifier for Directed Multigraphs.

Implements reverse message passing (incoming and outgoing aggregation) with edge attributes
and edge classification head for money mule detection under extreme class imbalance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from mulering.config import load_config
from mulering.data.splits import temporal_split
from mulering.evaluation.metrics import best_f1_threshold, minority_prf, pr_auc


EDGE_FEATURE_COLS = [
    "log_amount_sent",
    "log_amount_recv",
    "same_currency",
    "same_bank",
    "hour",
    "dow",
    "src_fanout_ratio",
    "dst_fanin_ratio",
    "reverse_pair_count_before",
]


class DirectedMultigraphLayer(nn.Module):
    """Message passing layer separating incoming and outgoing edges for directed multigraphs."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_in = nn.Sequential(
            nn.Linear(node_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.msg_out = nn.Sequential(
            nn.Linear(node_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.combine = nn.Sequential(
            nn.Linear(node_dim + 2 * out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        if edge_index.size(1) == 0:
            return x

        src, dst = edge_index[0], edge_index[1]

        # Incoming messages (src -> dst aggregated at dst)
        in_inputs = torch.cat([x[src], edge_attr], dim=-1)
        m_in = self.msg_in(in_inputs)
        agg_in = torch.zeros(num_nodes, m_in.size(-1), device=x.device)
        agg_in.index_add_(0, dst, m_in)

        # Outgoing messages (dst -> src aggregated at src)
        out_inputs = torch.cat([x[dst], edge_attr], dim=-1)
        m_out = self.msg_out(out_inputs)
        agg_out = torch.zeros(num_nodes, m_out.size(-1), device=x.device)
        agg_out.index_add_(0, src, m_out)

        out = self.combine(torch.cat([x, agg_in, agg_out], dim=-1))
        return out


class DirectedGNNEdgeClassifier(nn.Module):
    """Complete GNN model: node encoder + directed message passing + edge classification head."""

    def __init__(self, num_nodes: int, edge_feat_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        self.layers = nn.ModuleList([
            DirectedMultigraphLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        query_edge_index: torch.Tensor,
        query_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = self.node_emb.num_embeddings
        all_node_ids = torch.arange(num_nodes, device=edge_index.device)
        h = self.node_emb(all_node_ids)

        # Projected edge features for message passing
        edge_h = self.edge_proj(edge_attr) if edge_attr.size(0) > 0 else edge_attr

        # Message passing layers with residual connections
        for layer in self.layers:
            h_next = layer(h, edge_index, edge_h)
            h = h + h_next

        # Classify query edges: concat(src_emb, dst_emb, query_edge_attr)
        q_src, q_dst = query_edge_index[0], query_edge_index[1]
        q_edge_h = self.edge_proj(query_edge_attr)
        pair_h = torch.cat([h[q_src], h[q_dst], q_edge_h], dim=-1)
        logits = self.classifier(pair_h).squeeze(-1)
        return logits


def train_gnn(config_path: str = "configs/config.yaml", epochs: int = 40):
    cfg = load_config(config_path)
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    reports_dir = Path(cfg.get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    features_path = processed_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run 'make features' first.")

    print(f"Loading data from {features_path} for GNN training...")
    df = pd.read_parquet(features_path)

    # Encode node IDs
    all_accounts = sorted(list(set(df["src"]).union(set(df["dst"]))))
    acct2idx = {a: i for i, a in enumerate(all_accounts)}
    num_nodes = len(all_accounts)
    print(f"Total graph nodes (accounts): {num_nodes:,}")

    df["src_idx"] = df["src"].map(acct2idx)
    df["dst_idx"] = df["dst"].map(acct2idx)

    # Scale edge features
    feat_cols = [c for c in EDGE_FEATURE_COLS if c in df.columns]
    scaler = StandardScaler()
    edge_feats_norm = scaler.fit_transform(df[feat_cols].fillna(0.0).values)

    # Chronological temporal split
    fractions = tuple(cfg["split"]["fractions"])
    tr_df, va_df, te_df = temporal_split(df, fractions=fractions)
    n_tr, n_va, n_te = len(tr_df), len(va_df), len(te_df)

    i_tr_end = n_tr
    i_va_end = n_tr + n_va

    # Convert to tensors
    src_all = torch.tensor(df["src_idx"].to_numpy(), dtype=torch.long)
    dst_all = torch.tensor(df["dst_idx"].to_numpy(), dtype=torch.long)
    edge_index_all = torch.stack([src_all, dst_all], dim=0)
    edge_attr_all = torch.tensor(edge_feats_norm, dtype=torch.float32)
    labels_all = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)

    # Slices for temporal graphs
    train_edges = edge_index_all[:, :i_tr_end]
    train_attr = edge_attr_all[:i_tr_end]
    train_labels = labels_all[:i_tr_end]

    val_edges = edge_index_all[:, :i_va_end]
    val_attr = edge_attr_all[:i_va_end]
    val_query_edges = edge_index_all[:, i_tr_end:i_va_end]
    val_query_attr = edge_attr_all[i_tr_end:i_va_end]
    val_labels = labels_all[i_tr_end:i_va_end]

    test_query_edges = edge_index_all[:, i_va_end:]
    test_query_attr = edge_attr_all[i_va_end:]
    test_labels = labels_all[i_va_end:]

    # Class weighting for extreme imbalance
    n_pos = max(int(train_labels.sum().item()), 1)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = DirectedGNNEdgeClassifier(
        num_nodes=num_nodes,
        edge_feat_dim=len(feat_cols),
        hidden_dim=32,
        num_layers=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    print("\n--- Training Model M3: Directed Multigraph GNN ---")
    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_edges, train_attr, train_edges, train_attr)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_logits = model(val_edges, val_attr, val_query_edges, val_query_attr)
            val_scores = torch.sigmoid(val_logits).cpu().numpy()
            y_va_np = val_labels.cpu().numpy()
            v_auc = pr_auc(y_va_np, val_scores)

        if v_auc > best_val_auc:
            best_val_auc = v_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss.item():.4f} | Val PR-AUC: {v_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final scoring on validation and test sets
    model.eval()
    with torch.no_grad():
        final_val_logits = model(val_edges, val_attr, val_query_edges, val_query_attr)
        final_val_scores = torch.sigmoid(final_val_logits).cpu().numpy()

        test_graph_edges = edge_index_all  # cumulative test graph
        test_graph_attr = edge_attr_all
        test_logits = model(test_graph_edges, test_graph_attr, test_query_edges, test_query_attr)
        test_scores = torch.sigmoid(test_logits).cpu().numpy()

    y_va_np = val_labels.cpu().numpy()
    y_te_np = test_labels.cpu().numpy()

    th_gnn, f1_va_gnn = best_f1_threshold(y_va_np, final_val_scores)
    prf_te_gnn = minority_prf(y_te_np, test_scores, th_gnn)
    auc_te_gnn = pr_auc(y_te_np, test_scores)

    print(f"\nGNN Test Performance:")
    print(f"  - PR-AUC: {auc_te_gnn:.4f}")
    print(f"  - Minority F1: {prf_te_gnn['f1']:.4f} (Precision: {prf_te_gnn['precision']:.4f}, Recall: {prf_te_gnn['recall']:.4f})")

    # Update threshold.json
    thresh_path = reports_dir / "threshold.json"
    thresh_data = {}
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresh_data = json.load(f)

    thresh_data["m3_gnn"] = {
        "threshold": float(th_gnn),
        "val_f1": float(f1_va_gnn),
        "test_pr_auc": float(auc_te_gnn),
        "test_f1": float(prf_te_gnn["f1"]),
    }
    with open(thresh_path, "w") as f:
        json.dump(thresh_data, f, indent=2)

    # Update test_predictions.parquet
    preds_path = reports_dir / "test_predictions.parquet"
    if preds_path.exists():
        te_preds_df = pd.read_parquet(preds_path)
        te_preds_df["score_gnn"] = test_scores
        te_preds_df.to_parquet(preds_path, index=False)

    # Save model weights
    torch.save(model.state_dict(), reports_dir / "model_gnn.pt")
    print(f"Saved GNN model and scores to {reports_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch Geometric directed multigraph GNN.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    train_gnn(args.config, epochs=args.epochs)
