import torch
import torch.nn.functional as F


def train_epoch(model_g, model_n, loader, opt_g, opt_n, device):
    """
    model_g → GraphClassifier
    model_n → NodeRecommender
    loader  → DataLoader
    opt_g   → optimizer for graph model
    opt_n   → optimizer for node model
    """

    model_g.train()
    model_n.train()

    total_loss = 0

    for data in loader:
        data = data.to(device)

        # 🔹 Graph-level prediction
        out_g = model_g(data)
        loss_g = F.cross_entropy(out_g, data.graph_y)

        # 🔹 Node-level prediction
        out_n = model_n(data)
        loss_n = F.cross_entropy(out_n, data.y)

        # 🔹 Combined loss (from your PDF)
        loss = 0.4 * loss_g + 0.6 * loss_n

        # 🔹 Backprop
        opt_g.zero_grad()
        opt_n.zero_grad()

        loss.backward()

        # 🔹 Gradient clipping (important for stability)
        torch.nn.utils.clip_grad_norm_(model_g.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model_n.parameters(), 1.0)

        opt_g.step()
        opt_n.step()

        total_loss += loss.item()

    return total_loss / len(loader)