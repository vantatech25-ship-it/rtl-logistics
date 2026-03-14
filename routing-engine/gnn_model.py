"""
gnn_model.py
PyTorch Geometric Graph Neural Network for RTL Logistics route scoring.
Trains a GraphSAGE model on hub features to predict congestion scores.

Architecture:
  Input: Hub node features (load_factor, connectivity, lat, lon)
  2x GraphSAGE convolution layers
  Output: Per-hub congestion score (regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1. Model definition
# ─────────────────────────────────────────────────────────────────

class LogisticsGNN(nn.Module):
    """
    GraphSAGE model that predicts a congestion score for each hub node.
    Features per node: [load_factor, connectivity_norm, lat_norm, lon_norm]
    """
    def __init__(self, in_channels: int = 4, hidden: int = 32, out_channels: int = 1):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head  = nn.Linear(hidden, out_channels)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.head(x))  # output 0–1 congestion score


# ─────────────────────────────────────────────────────────────────
# 2. Data helpers — convert the Rust engine's hub data → PyG graph
# ─────────────────────────────────────────────────────────────────

def hubs_to_pyg_graph(hub_features: list, edges: list[tuple[int,int]]) -> Data:
    """
    hub_features: list of dicts from GET /hubs (load_factor, connectivity, etc.)
    edges: list of (from_index, to_index) integer pairs
    """
    node_feats = []
    labels     = []

    for h in hub_features:
        lat = h.get("latitude", 0.0)
        lon = h.get("longitude", 0.0)
        node_feats.append([
            float(h["load_factor"]),
            float(h["connectivity"]) / 10.0,
            (float(lat) + 90.0)  / 180.0,
            (float(lon) + 180.0) / 360.0,
        ])
        labels.append(float(h["congestion_score"]))

    x      = torch.tensor(node_feats, dtype=torch.float)
    y      = torch.tensor(labels,     dtype=torch.float).unsqueeze(1)
    ei     = torch.tensor(edges,      dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=ei, y=y)


# ─────────────────────────────────────────────────────────────────
# 3. Training loop
# ─────────────────────────────────────────────────────────────────

def train(model: nn.Module, data: Data, epochs: int = 200, lr: float = 0.01) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    losses = []

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 20 == 0:
            print(f"Epoch {epoch:>3} | Loss: {loss.item():.6f}")

    return losses


# ─────────────────────────────────────────────────────────────────
# 4. Inference — replace heuristic in gnn/mod.rs via sidecar call
# ─────────────────────────────────────────────────────────────────

def predict_congestion(model: nn.Module, data: Data) -> list[float]:
    """Returns per-hub congestion scores as a Python float list."""
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
    return preds.squeeze(1).tolist()


def save_model(model: nn.Module, path: str = "gnn_weights.pt") -> None:
    torch.save(model.state_dict(), path)
    print(f"[GNN] Model saved → {path}")


def load_model(path: str = "gnn_weights.pt", **kwargs) -> LogisticsGNN:
    model = LogisticsGNN(**kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────
# 5. Demo training on synthetic RTL graph
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic hub features (mirrors Rust demo network)
    hub_data = [
        {"load_factor": 0.75, "connectivity": 3, "latitude": -26.2041, "longitude":  28.0473, "congestion_score": 0.51},
        {"load_factor": 0.55, "connectivity": 2, "latitude": -33.9249, "longitude":  18.4241, "congestion_score": 0.39},
        {"load_factor": 0.80, "connectivity": 2, "latitude": -29.8587, "longitude":  31.0218, "congestion_score": 0.55},
        {"load_factor": 0.40, "connectivity": 3, "latitude": -33.9608, "longitude":  25.6022, "congestion_score": 0.30},
    ]
    edge_list = [(0,1),(0,2),(0,3),(1,3),(2,3)]  # mirrors Rust EDGES

    graph = hubs_to_pyg_graph(hub_data, edge_list)
    model = LogisticsGNN(in_channels=4, hidden=32, out_channels=1)

    print("=== Training GNN ===")
    losses = train(model, graph, epochs=200)

    preds = predict_congestion(model, graph)
    print("\n=== Congestion Predictions ===")
    for hub, pred in zip(hub_data, preds):
        print(f"  Hub load={hub['load_factor']} | predicted_cong={pred:.4f} | actual={hub['congestion_score']}")

    save_model(model)
