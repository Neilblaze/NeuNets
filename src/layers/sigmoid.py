from typing import Any

import torch
import torch.nn as nn


class SigmoidFunction(torch.autograd.Function):
    """
    Function for implementing the forward and backward pass of the 
    Sigmoid activation function in a neural network layer.
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Implementation of the forward pass for the Sigmoid activation function: 
        Sigmoid(X) = 1 / (1 + exp(-X)), where the exp operation is applied element-wise.
        
        :param ctx: context for saving derivatives during the forward pass
        :param args: arguments in the order [X,]
        :param kwargs:
        :return: values of the Sigmoid function applied to the input
        """
        inputs = args[0]
        sigmoid = 1 / (1 + torch.exp(-inputs))
        ctx.save_for_backward(sigmoid)
        return sigmoid

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Backward propagation of gradients for Sigmoid(X) = 1 / (1 + exp(-X))

        dL/dX = dL/dz * dz/dX = dL/dz * sigmoid * (1 - sigmoid), where sigmoid is the sigmoid function applied to X.

        :param ctx: context with saved values from the forward pass
        :param grad_outputs: incoming gradient
        :return: gradients with respect to dX
        """
        sigmoid = ctx.saved_tensors[0]
        return grad_outputs[0] * sigmoid * (1 - sigmoid)


class Sigmoid(nn.Module):
    """
    Implementation of the Sigmoid activation function for a neural network layer.
    To override the backward method, you need to create your own function and call its methods.
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = SigmoidFunction.apply

    def forward(self, x):
        return self.sigmoid(x)