import torch
import argparse

from model import NodeEncoder, GraphClassifier, NodeRecommender
from train import train_epoch
# from preprocessing import preprocess_data   # (you can connect later)


# 🔹 Dummy dataset (temporary)
def create_dummy_loader():
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    x = torch.rand((10, 10))  # 10 nodes, 10 features
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5]
    ], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.y = torch.randint(0, 5, (10,))          # node labels
    data.graph_y = torch.randint(0, 4, (1,))     # graph label
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    loader = DataLoader([data], batch_size=1)
    return loader


# 🔹 Argument parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    return parser.parse_args()


# 🔹 Main execution
if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 🔸 Data (temporary loader)
    train_loader = create_dummy_loader()

    # 🔸 Models
    encoder = NodeEncoder(in_channels=10).to(device)
    model_g = GraphClassifier(encoder).to(device)
    model_n = NodeRecommender(encoder).to(device)

    # 🔸 Optimizers
    opt_g = torch.optim.Adam(model_g.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_n = torch.optim.Adam(model_n.parameters(), lr=args.lr, weight_decay=1e-4)

    # 🔸 Training loop
    for epoch in range(args.epochs):
        loss = train_epoch(model_g, model_n, train_loader, opt_g, opt_n, device)

        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    print("✅ Training Complete")