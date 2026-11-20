"""GNN edge classifier (TO BUILD in Phase 5).

Plan (see docs/02_TRD.md section 4.4):
  - PyTorch Geometric, directed multigraph, accounts = nodes, transactions = edges.
  - Start from an edge-aware GIN; add reverse message passing, port numbering, ego IDs.
  - Edge head: concat(endpoint embeddings, edge embedding) -> MLP -> logit.
  - Neighbour sampling, class-weighted loss, early stopping on validation.
  - FIRST reproduce IBM's official Multi-GNN result on HI-Small to validate your split/metrics.
"""
def train_gnn(*args, **kwargs):
    raise NotImplementedError("Implement in Phase 5; see module docstring and docs/02_TRD.md")
