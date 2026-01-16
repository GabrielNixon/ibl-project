import torch
import torch.nn as nn

class TinySwitchGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2,
        num_contexts: int = 1,
        out_dim: int = 2,
        readout: str = "fc",   # "fc" or "diag"
        use_mask: bool = True,
    ):
        super().__init__()
        assert readout in ["fc", "diag"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_contexts = num_contexts
        self.out_dim = out_dim
        self.readout = readout
        self.use_mask = use_mask

        self.W_ir = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.W_iz = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.W_in = nn.Parameter(torch.empty(hidden_dim, input_dim))

        self.b_ir = nn.Parameter(torch.zeros(hidden_dim))
        self.b_iz = nn.Parameter(torch.zeros(hidden_dim))
        self.b_in = nn.Parameter(torch.zeros(hidden_dim))

        self.W_hr = nn.Parameter(torch.empty(num_contexts, hidden_dim, hidden_dim))
        self.W_hz = nn.Parameter(torch.empty(num_contexts, hidden_dim, hidden_dim))
        self.W_hn = nn.Parameter(torch.empty(num_contexts, hidden_dim, hidden_dim))

        self.b_hr = nn.Parameter(torch.zeros(num_contexts, hidden_dim))
        self.b_hz = nn.Parameter(torch.zeros(num_contexts, hidden_dim))
        self.b_hn = nn.Parameter(torch.zeros(num_contexts, hidden_dim))

        if readout == "fc":
            self.Wo = nn.Linear(hidden_dim, out_dim)
        else:
            assert out_dim == 2
            self.beta = nn.Parameter(torch.zeros(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(2))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_ir)
        nn.init.xavier_uniform_(self.W_iz)
        nn.init.xavier_uniform_(self.W_in)

        nn.init.xavier_uniform_(self.W_hr)
        nn.init.xavier_uniform_(self.W_hz)
        nn.init.xavier_uniform_(self.W_hn)

        if self.readout == "fc":
            nn.init.xavier_uniform_(self.Wo.weight)
            nn.init.zeros_(self.Wo.bias)
        else:
            nn.init.zeros_(self.beta)
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,              # [B, T, D]
        ctx: torch.Tensor = None,     # [B, T] long in [0..K-1] OR [B, T, K] one-hot/soft
        mask: torch.Tensor = None,    # [B, T] in {0,1} (optional)
        h0: torch.Tensor = None,      # [B, H] (optional)
    ):
        B, T, D = x.shape
        assert D == self.input_dim

        if h0 is None:
            h = x.new_zeros(B, self.hidden_dim)
        else:
            h = h0

        if self.use_mask:
            if mask is None:
                mt = x.new_ones(B)
                use_mask = False
            else:
                use_mask = True

        if self.num_contexts == 1:
            if ctx is None:
                ctx = x.new_zeros(B, T, dtype=torch.long)
        else:
            assert ctx is not None

        logits = []
        for t in range(T):
            xt = x[:, t, :]

            ir = (self.W_ir @ xt.unsqueeze(-1)).squeeze(-1) + self.b_ir
            iz = (self.W_iz @ xt.unsqueeze(-1)).squeeze(-1) + self.b_iz
            in_ = (self.W_in @ xt.unsqueeze(-1)).squeeze(-1) + self.b_in

            if ctx.dim() == 2:
                kt = ctx[:, t].long()
                W_hr = self.W_hr[kt]
                W_hz = self.W_hz[kt]
                W_hn = self.W_hn[kt]
                b_hr = self.b_hr[kt]
                b_hz = self.b_hz[kt]
                b_hn = self.b_hn[kt]
            else:
                ct = ctx[:, t, :].to(x.dtype)
                W_hr = torch.einsum("bk,kij->bij", ct, self.W_hr)
                W_hz = torch.einsum("bk,kij->bij", ct, self.W_hz)
                W_hn = torch.einsum("bk,kij->bij", ct, self.W_hn)
                b_hr = torch.einsum("bk,kj->bj", ct, self.b_hr)
                b_hz = torch.einsum("bk,kj->bj", ct, self.b_hz)
                b_hn = torch.einsum("bk,kj->bj", ct, self.b_hn)

            if ctx.dim() == 2:
                hr = (W_hr @ h.unsqueeze(-1)).squeeze(-1) + b_hr
                hz = (W_hz @ h.unsqueeze(-1)).squeeze(-1) + b_hz
                hn = (W_hn @ h.unsqueeze(-1)).squeeze(-1) + b_hn
            else:
                hr = torch.einsum("bij,bj->bi", W_hr, h) + b_hr
                hz = torch.einsum("bij,bj->bi", W_hz, h) + b_hz
                hn = torch.einsum("bij,bj->bi", W_hn, h) + b_hn

            r = torch.sigmoid(ir + hr)
            z = torch.sigmoid(iz + hz)
            n = torch.tanh(in_ + r * hn)

            h_new = (1.0 - z) * n + z * h

            if self.use_mask and use_mask:
                mt = mask[:, t].to(h.dtype)
                h = mt.unsqueeze(-1) * h_new + (1.0 - mt.unsqueeze(-1)) * h
            else:
                h = h_new

            if self.readout == "diag":
                s = h * self.beta.unsqueeze(0)
                s2 = torch.stack([s.sum(dim=1), -s.sum(dim=1)], dim=1) + self.bias.unsqueeze(0)
                logits.append(s2)
            else:
                logits.append(self.Wo(h))

        return torch.stack(logits, dim=1)

    def recurrent_params(self):
        return [self.W_hr, self.W_hz, self.W_hn]
