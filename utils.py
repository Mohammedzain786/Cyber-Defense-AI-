import torch
from torch_geometric.loader import DataLoader


# 🔹 Create Data Loaders
def get_loaders(dataset, batch_size=32):
    """
    Splits dataset into train / val / test
    """

    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size)
    test_loader  = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# 🔹 Simple Evaluation
def evaluate(model_g, model_n, loader, device):
    model_g.eval()
    model_n.eval()

    correct_g = 0
    total_g   = 0

    correct_n = 0
    total_n   = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Graph-level
            out_g = model_g(data)
            pred_g = out_g.argmax(dim=1)
            correct_g += (pred_g == data.graph_y).sum().item()
            total_g += data.graph_y.size(0)

            # Node-level
            out_n = model_n(data)
            pred_n = out_n.argmax(dim=1)
            correct_n += (pred_n == data.y).sum().item()
            total_n += data.y.size(0)

    acc_g = correct_g / total_g if total_g > 0 else 0
    acc_n = correct_n / total_n if total_n > 0 else 0

    return acc_g, acc_n


# 🔹 Save model checkpoints
def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


# 🔹 Load model checkpoints
def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model