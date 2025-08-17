import os
import torch
from torch import nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn.models import SchNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET = 0  # QM9 has 19 targets; 0 = dipole moment (mu)

def split(dataset, r_train=0.8, r_val=0.1, seed=42):
    torch.manual_seed(seed)
    n = len(dataset)
    idx = torch.randperm(n)
    n_tr = int(n * r_train)
    n_va = int(n * (r_train + r_val))
    return dataset[idx[:n_tr]], dataset[idx[n_tr:n_va]], dataset[idx[n_va:]]

def main():
    print(f"Device: {DEVICE}")
    path = os.path.join("data", "QM9")
    ds = QM9(path)

    # Standardize target (in-place; fine for a quick POC)
    y = ds.data.y[:, TARGET]
    y_mean, y_std = y.mean(), y.std()
    ds.data.y[:, TARGET] = (y - y_mean) / (y_std + 1e-8)

    train_ds, val_ds, test_ds = split(ds)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

    # Version-agnostic SchNet: do not pass out_channels/num_outputs
    model = SchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=3,
        num_gaussians=50,
        cutoff=10.0,
        max_num_neighbors=32
    ).to(DEVICE)

    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    def run(loader, train: bool = False):
        model.train() if train else model.eval()
        tot_loss, n_graphs = 0.0, 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = batch.to(DEVICE)
                # IMPORTANT: pass batch.batch for graph-level pooling inside SchNet
                pred = model(batch.z, batch.pos, batch.batch)
                # Ensure shape [B, 1]
                if pred.dim() == 1:
                    pred = pred.unsqueeze(1)
                y_t = batch.y[:, TARGET:TARGET+1]
                loss = loss_fn(pred, y_t)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                tot_loss += loss.item() * batch.num_graphs
                n_graphs += batch.num_graphs
        # RMSE on standardized target
        return (tot_loss / max(n_graphs, 1)) ** 0.5

    best_val = float("inf")
    for epoch in range(1, 51):
        train_rmse = run(train_loader, train=True)
        val_rmse = run(val_loader, train=False)
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), "schnet_qm9.pt")
        print(f"Epoch {epoch:03d} | train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}")

    # Test with best checkpoint
    model.load_state_dict(torch.load("schnet_qm9.pt", map_location=DEVICE))
    test_rmse = run(test_loader, train=False)
    print(f"Test RMSE (standardized): {test_rmse:.4f}")

if __name__ == "__main__":
    main()
