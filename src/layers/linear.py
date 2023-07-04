from typing import Any

import torch.nn as nn
import torch


class LinearFunction(torch.autograd.Function):
    """
    Function for implementing the forward and backward pass
    of a linear (fully-connected) layer in a neural network.
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Classic linear implementation X @ W.T + b
        :param ctx: context for storing derivatives during the forward pass
        :param args: arguments in the order [X, W, b]
        :param kwargs:
        :return:
        """
        inputs = args[0]
        weight = args[1]
        bias = args[2]

        ctx.save_for_backward(inputs, weight, bias)

        return inputs @ weight.T + bias

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Backpropagation of gradients for X @ W.T + b

        dL/dX = dL/dz * dz/dX = dL/dz * W
        dL/dW = dL/dz * dz/dW = dL/dz * X
        dL/db = dL/dz * dz/db = dL/dz * 1

        :param ctx: context with saved values from the forward pass
        :param grad_outputs: incoming gradient
        :return: gradients for dX, dW, db
        """
        inputs, weight, bias = ctx.saved_tensors

        der_inputs = grad_outputs[0] @ weight
        der_weight = grad_outputs[0].T @ inputs
        der_bias = grad_outputs[0].sum(axis=0)

        return der_inputs, der_weight, der_bias


class Linear(nn.Module):
    """
    Implementation of a linear (fully-connected) layer in a neural network.
    To override the backward method, you need to create your own function and call its methods.

    The weights are initialized with a normal distribution multiplied by 1e-3.
    It is important to initialize the bias with zero or a very small value.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 1e-3)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.linear = LinearFunction.apply

    def forward(self, x):
        return self.linear(x, self.weight, self.bias)