import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append(os.path.dirname(__file__))

from tinyswitchgru import TinySwitchGRU

class SeqDS(Dataset):
    def __init__(self, x, y, m):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.m = torch.from_numpy(m).float()
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, i):
        return self.x[i], self.y[i], self.m[i]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def l1_recurrent(model, lam):
    if lam <= 0:
        return 0.0
    s = 0.0
    for p in model.recurrent_params():
        s = s + p.abs().sum()
    return lam * s

def run_epoch(model, loader, opt, device, l1_lam):
    ce = nn.CrossEntropyLoss(reduction="none")
    train = opt is not None
    model.train(train)

    total_loss = 0.0
    total_correct = 0.0
    total_count = 0.0

    for xb, yb, mb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)

        logits = model(xb, ctx=None, mask=mb)

        B, T, C = logits.shape
        loss_per = ce(logits.view(B*T, C), yb.view(B*T)).view(B, T)
        denom = mb.sum().clamp_min(1.0)
        loss = (loss_per * mb).sum() / denom
        loss = loss + l1_recurrent(model, l1_lam)

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = logits.argmax(-1)
            correct = ((pred == yb).float() * mb).sum()
            total_loss += loss.item() * denom.item()
            total_correct += correct.item()
            total_count += denom.item()

    return total_loss / total_count, total_correct / total_count

def fit_eval(
    train_loader,
    test_loader,
    device,
    hidden_dim=2,
    readout="diag",
    lr=5e-3,
    l1_lam=0.0,
    epochs=15,
    seed=0,
):
    set_seed(seed)
    model = TinySwitchGRU(
        input_dim=3,
        hidden_dim=hidden_dim,
        num_contexts=1,
        out_dim=2,
        readout=readout,
        use_mask=True,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_te_acc = -1.0
    best_te_loss = 1e9
    best_ep = -1

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, device, l1_lam)
        te_loss, te_acc = run_epoch(model, test_loader, None, device, 0.0)

        if te_acc > best_te_acc:
            best_te_acc = te_acc
            best_te_loss = te_loss
            best_ep = ep

    return best_te_loss, best_te_acc, best_ep

def format_row(cols, widths):
    out = []
    for c, w in zip(cols, widths):
        s = str(c)
        if len(s) > w:
            s = s[:w]
        out.append(s.ljust(w))
    return " | ".join(out)

def main():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    base = "/Users/gabrielnixonraj/Desktop/ibl_project"
    data_path = os.path.join(base, "data", "ibl_sequences_T800.npz")
    data = np.load(data_path)
    x = data["x"]
    y = data["y"]
    m = data["mask"]

    ds = SeqDS(x, y, m)
    n_train = int(0.8 * len(ds))
    n_test = len(ds) - n_train
    train_ds, test_ds = random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(0))

    tr_ld = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    te_ld = DataLoader(test_ds, batch_size=128, shuffle=False, drop_last=False)

    configs = [
        {
            "name": "Vanilla (paper-style TinyGRU, diag)",
            "hidden_dim": 2,
            "readout": "diag",
            "lr": 5e-3,
            "l1_lam": 0.0,
            "epochs": 15,
            "seed": 0,
        },
        {
            "name": "Vanilla + recurrent L1 (1e-4)",
            "hidden_dim": 2,
            "readout": "diag",
            "lr": 5e-3,
            "l1_lam": 1e-4,
            "epochs": 15,
            "seed": 0,
        },
        {
            "name": "Paper variant: diag readout (you ran)",
            "hidden_dim": 2,
            "readout": "diag",
            "lr": 5e-3,
            "l1_lam": 0.0,
            "epochs": 15,
            "seed": 0,
        },
        {
            "name": "Paper variant: fc readout",
            "hidden_dim": 2,
            "readout": "fc",
            "lr": 5e-3,
            "l1_lam": 0.0,
            "epochs": 15,
            "seed": 0,
        },
    ]

    rows = []
    t0 = time.time()
    for cfg in configs:
        te_loss, te_acc, best_ep = fit_eval(
            tr_ld, te_ld, device=device,
            hidden_dim=cfg["hidden_dim"],
            readout=cfg["readout"],
            lr=cfg["lr"],
            l1_lam=cfg["l1_lam"],
            epochs=cfg["epochs"],
            seed=cfg["seed"],
        )
        rows.append([
            cfg["name"],
            f"H={cfg['hidden_dim']}",
            cfg["readout"],
            f"{cfg['lr']:.0e}",
            f"{cfg['l1_lam']:.0e}",
            cfg["epochs"],
            f"{te_loss:.4f}",
            f"{te_acc:.3f}",
            best_ep
        ])

    headers = ["Model", "Hidden", "Readout", "LR", "L1", "Epochs", "BestTestLoss", "BestTestAcc", "BestEp"]
    widths = [44, 6, 6, 5, 6, 6, 12, 11, 6]

    print("\nDEVICE:", device)
    print(format_row(headers, widths))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for r in rows:
        print(format_row(r, widths))
    print("\nTotal time (s):", round(time.time() - t0, 2))

if __name__ == "__main__":
    main()
