from typing import Any

import torch
import torch.nn as nn


class ReLUFunction(torch.autograd.Function):
    """
    Function for implementing the forward and backward pass of the 
    ReLU activation function in a neural network.
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Implementation of the forward pass of the ReLU activation function: ReLU(X) = max(X, 0), where the max operation is applied element-wise.
        :param ctx: context for saving derivatives during the forward pass
        :param args: arguments in the order [X,]
        :param kwargs:
        :return: ReLU values of the input
        """
        inputs = args[0]
        ctx.save_for_backward(inputs)
        return torch.max(inputs, torch.zeros_like(inputs))

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Backward propagation of gradients for ReLU(X) = max(X, 0)
        dL/dX = dL/dz * dz/dX = dL/dz * I[x > 0], where I is the indicator function.

        :param ctx: context with saved values from the forward pass
        :param grad_outputs: incoming gradient
        :return: gradients with respect to dX
        """
        inputs = ctx.saved_tensors[0]
        mask = torch.where(inputs > 0, torch.ones_like(inputs), torch.zeros_like(inputs))
        return grad_outputs[0] * mask


class ReLU(nn.Module):
    """
    Implementation of the ReLU activation function for a neural network layer.
    To override the backward method, you need to create your own function and call its methods.
    """
    def __init__(self):
        super().__init__()
        self.relu = ReLUFunction.apply

    def forward(self, x):
        return self.relu(x)