import os
import sys
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


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("device:", device)

    base = "/Users/gabrielnixonraj/Desktop/ibl_project"
    data_path = os.path.join(base, "data", "ibl_sequences_T800.npz")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}")

    data = np.load(data_path)

    x = data["x"]
    y = data["y"]
    m = data["mask"]

    print("Loaded data:")
    print("x:", x.shape, "y:", y.shape, "m:", m.shape)

    ds = SeqDS(x, y, m)

    n_train = int(0.8 * len(ds))
    n_test = len(ds) - n_train
    train_ds, test_ds = random_split(
        ds, [n_train, n_test],
        generator=torch.Generator().manual_seed(0)
    )

    tr_ld = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    te_ld = DataLoader(test_ds, batch_size=128, shuffle=False, drop_last=False)

    model = TinySwitchGRU(
        input_dim=3,
        hidden_dim=2,
        num_contexts=1,
        out_dim=2,
        readout="diag",
        use_mask=True,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    ce = nn.CrossEntropyLoss(reduction="none")

    def run(loader, train: bool):
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
            loss_per = ce(
                logits.view(B * T, C),
                yb.view(B * T)
            ).view(B, T)

            denom = mb.sum().clamp_min(1.0)
            loss = (loss_per * mb).sum() / denom

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
