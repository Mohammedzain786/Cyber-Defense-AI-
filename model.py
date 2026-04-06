import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


# 🔹 Node Encoder (GAT)
class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden=64, out=128, heads=8):
        super().__init__()

        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=0.3)
        self.gat2 = GATConv(hidden * heads, hidden, heads=heads, dropout=0.3)
        self.gat3 = GATConv(hidden * heads, out, heads=1, concat=False)

        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden * heads)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(self.bn1(x))

        x = self.gat2(x, edge_index)
        x = F.elu(self.bn2(x))

        x = self.gat3(x, edge_index)
        return x


# 🔹 Graph-Level Classifier (FIXED)
class GraphClassifier(nn.Module):
    def __init__(self, node_encoder, num_classes=4):
        super().__init__()

        self.encoder = node_encoder
        self.gcn = GCNConv(128, 64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, data):
        # Step 1: Encode nodes
        x = self.encoder(data.x, data.edge_index)

        # ✅ Step 2: Apply GCN BEFORE pooling (THIS IS THE FIX)
        x = F.relu(self.gcn(x, data.edge_index))

        # Step 3: Pool to graph level
        x = global_mean_pool(x, data.batch)

        # Step 4: Classification
        return self.classifier(x)


# 🔹 Node-Level Recommender
class NodeRecommender(nn.Module):
    def __init__(self, node_encoder, num_actions=5):
        super().__init__()

        self.encoder = node_encoder

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions)
        )

    def forward(self, data):
        x = self.encoder(data.x, data.edge_index)
        return self.head(x)