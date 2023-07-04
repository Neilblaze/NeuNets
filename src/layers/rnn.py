import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    Implementation of the RNN layer for a neural network.
    The calculation of the hidden state is done using the formula:

    h_t = tanh(x_t @ W_ih + b_ih + h_(t-1) @ W_ih + b_hh)

    h_t - hidden state at time step t
    x_t - input at time step t
    h_(t-1) - hidden state of the previous layer at time step (t-1)

    The weight initialization type can be specified as 'orthogonal' or 'uniform'.
    """
    def __init__(self, input_dim: int, hidden_dim: int, init_type: str = 'orthogonal'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_ih = nn.Parameter(torch.rand(input_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.rand(1, hidden_dim))
        self.W_hh = nn.Parameter(torch.rand(hidden_dim, hidden_dim))
        self.b_hh = nn.Parameter(torch.rand(1, hidden_dim))

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
            hidden = torch.tanh(
                x[:, idx] @ self.W_ih + self.b_ih + hidden @ self.W_hh + self.b_hh
            )
        return hidden