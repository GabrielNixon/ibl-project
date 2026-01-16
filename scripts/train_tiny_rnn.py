import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

NPZ = "/Users/gabrielnixonraj/Desktop/ibl_project/data/ibl_sequences_T800.npz"
OUT = "/Users/gabrielnixonraj/Desktop/ibl_project/meta/tiny_rnn_metrics.npz"

class SeqDS(Dataset):
    def __init__(self, x, y, m):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.m = torch.from_numpy(m).float()
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i], self.m[i]

class TinyRNN(nn.Module):
    def __init__(self, d_in=3, h=4):
        super().__init__()
        self.rnn = nn.GRU(d_in, h, batch_first=True)
        self.out = nn.Linear(h, 2)
    def forward(self, x):
        z, _ = self.rnn(x)
        return self.out(z)

def main():
    d = np.load(NPZ, allow_pickle=True)
    x, y, m = d["x"], d["y"], d["mask"]

    n = x.shape[0]
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]

    tr_ds = SeqDS(x[tr], y[tr], m[tr])
    te_ds = SeqDS(x[te], y[te], m[te])

    tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=64, shuffle=False)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = TinyRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    def run(loader, train=False):
        model.train(train)
        total_loss, total_correct, total_count = 0.0, 0.0, 0.0
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            logits = model(xb)
            loss = loss_fn(logits.reshape(-1, 2), yb.reshape(-1))
            mask = mb.reshape(-1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                pred = logits.argmax(-1)
                correct = ((pred == yb).float() * mb).sum()
                count = mb.sum()
                total_loss += float(loss) * float(count)
                total_correct += float(correct)
                total_count += float(count)
        return total_loss / total_count, total_correct / total_count

    for ep in range(15):
        tr_loss, tr_acc = run(tr_ld, train=True)
        te_loss, te_acc = run(te_ld, train=False)
        print(
            f"epoch {ep+1:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"test loss {te_loss:.4f} acc {te_acc:.3f}"
        )

if __name__ == "__main__":
    main()
