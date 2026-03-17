import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool


def _make_gine_mlp(hidden_dim: int):
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GINEVirtualNodeClassifier(nn.Module):
    """
    GINEConv + Virtual Node + BatchNorm for graph-level binary classification.
    Frontend: direct input_proj(in_dim -> hidden_dim), no feature slicing.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        pooling: str = "mean",
        edge_attr_dim: int = 2,
    ):
        super().__init__()
        assert pooling in ["mean", "add"]

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_attr_dim, hidden_dim)

        self.convs = nn.ModuleList(
            [GINEConv(_make_gine_mlp(hidden_dim)) for _ in range(num_layers)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        self.vn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr

        x = self.input_proj(x)
        e = self.edge_encoder(edge_attr)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        virtualnode_emb = x.new_zeros((num_graphs, self.hidden_dim))

        for i in range(self.num_layers):
            x = x + virtualnode_emb[batch]
            x = self.convs[i](x, edge_index, e)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if i != self.num_layers - 1:
                vn_update = global_add_pool(x, batch)
                virtualnode_emb = virtualnode_emb + self.vn_mlp(vn_update)

        graph_emb = (
            global_add_pool(x, batch)
            if self.pooling == "add"
            else global_mean_pool(x, batch)
        )
        logits = self.classifier(graph_emb).view(-1)
        return logits
