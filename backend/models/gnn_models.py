import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GINConv,
    GATConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)

def get_pooling_layer(pooling_name: str):
    pooling_name = pooling_name.lower()
    if pooling_name == "sum" or pooling_name == "add":
        return global_add_pool
    elif pooling_name == "max":
        return global_max_pool
    else:
        return global_mean_pool

class GCNNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 7,
        hidden_dim: int = 64,
        out_dim: int = 2,
        num_layers: int = 2,
        pooling: str = "mean",
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = get_pooling_layer(pooling)
        self.pooling_name = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 32 else hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 32 else hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        node_embeddings = h
        graph_embedding = self.pooling(node_embeddings, batch)
        logits = self.classifier(graph_embedding)
        return logits

    def get_embeddings(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        graph_embedding = self.pooling(h, batch)
        return h, graph_embedding

class GraphSAGENet(nn.Module):
    def __init__(
        self,
        in_dim: int = 7,
        hidden_dim: int = 64,
        out_dim: int = 2,
        num_layers: int = 2,
        pooling: str = "mean",
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = get_pooling_layer(pooling)
        self.pooling_name = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        else:
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr='mean'))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 32 else hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 32 else hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        node_embeddings = h
        graph_embedding = self.pooling(node_embeddings, batch)
        logits = self.classifier(graph_embedding)
        return logits

    def get_embeddings(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        graph_embedding = self.pooling(h, batch)
        return h, graph_embedding

class GINNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 7,
        hidden_dim: int = 64,
        out_dim: int = 2,
        num_layers: int = 2,
        pooling: str = "sum",
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = get_pooling_layer(pooling)
        self.pooling_name = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            dim_in = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(dim_in, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 32 else hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 32 else hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        node_embeddings = h
        graph_embedding = self.pooling(node_embeddings, batch)
        logits = self.classifier(graph_embedding)
        return logits

    def get_embeddings(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        graph_embedding = self.pooling(h, batch)
        return h, graph_embedding

class GATNet(nn.Module):
    def __init__(
        self,
        in_dim: int = 7,
        hidden_dim: int = 64,
        out_dim: int = 2,
        num_layers: int = 2,
        heads: int = 4,
        pooling: str = "mean",
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = get_pooling_layer(pooling)
        self.pooling_name = pooling

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        else:
            self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 32 else hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 32 else hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        node_embeddings = h
        graph_embedding = self.pooling(node_embeddings, batch)
        logits = self.classifier(graph_embedding)
        return logits

    def get_embeddings(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        graph_embedding = self.pooling(h, batch)
        return h, graph_embedding

def build_model(
    model_name: str,
    in_dim: int = 7,
    hidden_dim: int = 64,
    out_dim: int = 2,
    num_layers: int = 2,
    pooling: str = "mean",
    dropout: float = 0.2
):
    name = model_name.upper()
    if name == "GCN":
        return GCNNet(in_dim, hidden_dim, out_dim, num_layers, pooling, dropout)
    elif name in ["SAGE", "GRAPHSAGE"]:
        return GraphSAGENet(in_dim, hidden_dim, out_dim, num_layers, pooling, dropout)
    elif name == "GIN":
        return GINNet(in_dim, hidden_dim, out_dim, num_layers, pooling if pooling != "mean" else "sum", dropout)
    elif name == "GAT":
        return GATNet(in_dim, hidden_dim, out_dim, num_layers, heads=4, pooling=pooling, dropout=dropout)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
