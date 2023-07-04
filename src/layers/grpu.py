import torch
import torch.nn as nn


class GRU(nn.Module):
    """
    Implementation of the GRU layer in a neural network.
    The calculation of the hidden state is performed using the following formulas:

    z_t = sigmoid(W_z @ x_t + b_iz + W_hz @ h_(t-1) + b_hz)
    r_t = sigmoid(W_r @ x_t + b_ir + W_hr @ h_(t-1) + b_hr)
    n_t = tanh(W_in @ x_t + b_in + r_t * (W_hh @ h_(t-1) + b_hh))
    h_t = (1 - z_t) * n_t + z_t * h_(t-1)

    where:
    - h_t is the hidden state at time t
    - x_t is the input at time t
    - h_(t-1) is the hidden state from the previous layer at time (t-1)
    - r_t is the reset gate
    - z_t is the update gate
    - n_t is the new gate

    The weight initialization type can be specified ('orthogonal', 'uniform').
    """
    def __init__(self, input_dim: int, hidden_dim: int, init_type: str = 'orthogonal'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_zh = nn.Parameter(torch.rand(hidden_dim, hidden_dim))
        self.b_zh = nn.Parameter(torch.rand(1, hidden_dim))
        self.W_zx = nn.Parameter(torch.rand(input_dim, hidden_dim))
        self.b_zx = nn.Parameter(torch.rand(1, hidden_dim))

        self.W_rh = nn.Parameter(torch.rand(hidden_dim, hidden_dim))
        self.b_rh = nn.Parameter(torch.rand(1, hidden_dim))
        self.W_rx = nn.Parameter(torch.rand(input_dim, hidden_dim))
        self.b_rx = nn.Parameter(torch.rand(1, hidden_dim))

        self.W_nh = nn.Parameter(torch.rand(hidden_dim, hidden_dim))
        self.b_nh = nn.Parameter(torch.rand(1, hidden_dim))
        self.W_nx = nn.Parameter(torch.rand(input_dim, hidden_dim))
        self.b_nx = nn.Parameter(torch.rand(1, hidden_dim))

        self.init_parameters(init_type)

    def init_parameters(self, init_type: str):
        std = 1.0 / self.hidden_dim**0.5
        for param in self.parameters():
            if init_type == 'uniform':
                nn.init.uniform_(param, -std, std)
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(param)
            else:
                raise NotImplementedError

    def forward(self, x: torch.Tensor, hidden=None):
        # x = [batch_size, seq_len, embed_dim]
        device = x.device
        if hidden is None:
            hidden = torch.zeros((x.size(0), self.hidden_dim), device=device)
        for idx in range(x.size(1)):
            z = torch.sigmoid(
                x[:, idx] @ self.W_zx + self.b_zx + hidden @ self.W_zh + self.b_zh
            )
            r = torch.sigmoid(
                x[:, idx] @ self.W_rx + self.b_rx + hidden @ self.W_rh + self.b_rh
            )
            n = torch.tanh(
                x[:, idx] @ self.W_nx + self.b_nx + r * (hidden @ self.W_nh + self.b_nh)
            )
            hidden = (1 - z) * n + z * hidden
        return hidden