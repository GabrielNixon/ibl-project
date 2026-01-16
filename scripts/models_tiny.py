import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGRU(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=1, readout="full"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(in_dim, hidden_dim)
        if readout == "diag":
            self.beta = nn.Parameter(torch.zeros(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(2))
            self.readout = "diag"
        else:
            self.Wo = nn.Linear(hidden_dim, 2)
            self.readout = "full"

    def forward(self, x, mask):
        B, T, D = x.shape
        h = x.new_zeros(B, self.hidden_dim)
        logits = []
        for t in range(T):
            xt = x[:, t, :]
            mt = mask[:, t].unsqueeze(-1)
            h_new = self.gru(xt, h)
            h = mt * h_new + (1.0 - mt) * h
            if self.readout == "diag":
                s = h * self.beta.unsqueeze(0)
                s2 = torch.stack([s.sum(dim=1), -s.sum(dim=1)], dim=1) + self.bias.unsqueeze(0)
                logits.append(s2)
            else:
                logits.append(self.Wo(h))
        return torch.stack(logits, dim=1)

    def recurrent_params(self):
        ps = []
        for n, p in self.gru.named_parameters():
            if "weight_hh" in n:
                ps.append(p)
        return ps


class SwitchingGRU1(nn.Module):
    def __init__(self, hidden_dim=1, n_switch=6, stim_dim=1, readout="full"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_switch = n_switch
        self.stim_in = nn.Linear(stim_dim, 3 * hidden_dim, bias=True)

        self.W_hr = nn.Parameter(torch.empty(n_switch, hidden_dim, hidden_dim))
        self.W_hz = nn.Parameter(torch.empty(n_switch, hidden_dim, hidden_dim))
        self.W_hn = nn.Parameter(torch.empty(n_switch, hidden_dim, hidden_dim))

        self.b_hr = nn.Parameter(torch.zeros(n_switch, hidden_dim))
        self.b_hz = nn.Parameter(torch.zeros(n_switch, hidden_dim))
        self.b_hn = nn.Parameter(torch.zeros(n_switch, hidden_dim))

        nn.init.orthogonal_(self.W_hr)
        nn.init.orthogonal_(self.W_hz)
        nn.init.orthogonal_(self.W_hn)

        if readout == "diag":
            self.beta = nn.Parameter(torch.zeros(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(2))
            self.readout = "diag"
        else:
            self.Wo = nn.Linear(hidden_dim, 2)
            self.readout = "full"

    def forward(self, stim, switch_id, mask):
        B, T, _ = stim.shape
        h = stim.new_zeros(B, self.hidden_dim)
        logits = []
        for t in range(T):
            mt = mask[:, t].unsqueeze(-1)
            sid = switch_id[:, t].clamp(0, self.n_switch - 1)

            inp = self.stim_in(stim[:, t, :])
            bir, biz, bin_ = torch.chunk(inp, 3, dim=1)

            W_hr = self.W_hr[sid]
            W_hz = self.W_hz[sid]
            W_hn = self.W_hn[sid]

            b_hr = self.b_hr[sid]
            b_hz = self.b_hz[sid]
            b_hn = self.b_hn[sid]

            hr = (W_hr @ h.unsqueeze(-1)).squeeze(-1) + b_hr
            hz = (W_hz @ h.unsqueeze(-1)).squeeze(-1) + b_hz
            hn = (W_hn @ h.unsqueeze(-1)).squeeze(-1) + b_hn

            r = torch.sigmoid(bir + hr)
            z = torch.sigmoid(biz + hz)
            n = torch.tanh(bin_ + r * hn)

            h_new = (1.0 - z) * n + z * h
            h = mt * h_new + (1.0 - mt) * h

            if self.readout == "diag":
                s = h * self.beta.unsqueeze(0)
                s2 = torch.stack([s.sum(dim=1), -s.sum(dim=1)], dim=1) + self.bias.unsqueeze(0)
                logits.append(s2)
            else:
                logits.append(self.Wo(h))
        return torch.stack(logits, dim=1)

    def recurrent_params(self):
        return [self.W_hr, self.W_hz, self.W_hn]
