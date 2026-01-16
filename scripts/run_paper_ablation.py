import os
import sys
import time
import csv
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

def fit_eval(train_loader, test_loader, device, hidden_dim, readout, lr, l1_lam, epochs, seed):
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
        _ = run_epoch(model, train_loader, opt, device, l1_lam)
        te_loss, te_acc = run_epoch(model, test_loader, None, device, 0.0)
        if te_acc > best_te_acc:
            best_te_acc = te_acc
            best_te_loss = te_loss
            best_ep = ep

    return best_te_loss, best_te_acc, best_ep

def mean_std(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0

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

    hidden_dims = [1, 2, 3, 4]
    readouts = ["diag", "fc"]
    l1s = [0.0, 1e-5, 1e-4]
    seeds = [0, 1, 2]

    lr = 5e-3
    epochs = 15

    rows = []
    t0 = time.time()

    for H in hidden_dims:
        for r in readouts:
            for l1 in l1s:
                losses = []
                accs = []
                best_eps = []
                for sd in seeds:
                    te_loss, te_acc, best_ep = fit_eval(
                        tr_ld, te_ld, device=device,
                        hidden_dim=H, readout=r, lr=lr, l1_lam=l1, epochs=epochs, seed=sd
                    )
                    losses.append(te_loss)
                    accs.append(te_acc)
                    best_eps.append(best_ep)

                m_loss, s_loss = mean_std(losses)
                m_acc, s_acc = mean_std(accs)
                m_ep, _ = mean_std(best_eps)

                rows.append({
                    "hidden_dim": H,
                    "readout": r,
                    "l1": l1,
                    "lr": lr,
                    "epochs": epochs,
                    "seeds": ",".join(map(str, seeds)),
                    "best_test_loss_mean": m_loss,
                    "best_test_loss_std": s_loss,
                    "best_test_acc_mean": m_acc,
                    "best_test_acc_std": s_acc,
                    "best_ep_mean": m_ep,
                })

                print(f"done: H={H} readout={r} l1={l1:.0e} | acc {m_acc:.3f}±{s_acc:.3f}")

    rows_sorted = sorted(rows, key=lambda d: d["best_test_acc_mean"], reverse=True)

    out_csv = os.path.join(base, "scripts", "ablations_tiny_switchgru.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(rows_sorted)

    out_md = os.path.join(base, "scripts", "ablations_tiny_switchgru.md")
    with open(out_md, "w") as f:
        f.write(f"DEVICE: {device}\n\n")
        f.write("| Hidden | Readout | L1 | Best Test Acc (mean±std) | Best Test Loss (mean±std) |\n")
        f.write("|---:|:---:|---:|---:|---:|\n")
        for d in rows_sorted:
            f.write(
                f"| {d['hidden_dim']} | {d['readout']} | {d['l1']:.0e} | "
                f"{d['best_test_acc_mean']:.3f}±{d['best_test_acc_std']:.3f} | "
                f"{d['best_test_loss_mean']:.4f}±{d['best_test_loss_std']:.4f} |\n"
            )

    headers = ["Hidden", "Readout", "L1", "BestAcc(mean±std)", "BestLoss(mean±std)"]
    widths = [6, 6, 6, 16, 18]

    print("\nDEVICE:", device)
    print(format_row(headers, widths))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for d in rows_sorted[:12]:
        print(format_row([
            d["hidden_dim"],
            d["readout"],
            f"{d['l1']:.0e}",
            f"{d['best_test_acc_mean']:.3f}±{d['best_test_acc_std']:.3f}",
            f"{d['best_test_loss_mean']:.4f}±{d['best_test_loss_std']:.4f}",
        ], widths))

    print("\nWrote:")
    print(out_csv)
    print(out_md)
    print("\nTotal time (s):", round(time.time() - t0, 2))

if __name__ == "__main__":
    main()
