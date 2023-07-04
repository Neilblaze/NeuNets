from typing import Any

import torch
import torch.nn as nn


class DropoutFunction(torch.autograd.Function):
    """
    Function for implementing the forward and backward pass
    of the Dropout layer in a neural network.
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Implementation of the forward pass of Dropout with a dropout probability of p.
        :param ctx: context for saving derivatives during the forward pass
        :param args: arguments in the order [X, p]
        :param kwargs:
        :return:
        """
        inputs = args[0]
        probability = args[1]
        dropout_mask = torch.where(torch.rand(inputs.shape) < probability,
                                   torch.zeros_like(inputs),
                                   torch.ones_like(inputs)
                                   )
        ctx.save_for_backward(dropout_mask)
        return inputs * dropout_mask

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Backward propagation of gradients for Dropout.

        dL/dX = dL/dz * dz/dX = dL/dz * dropout_mask

        :param ctx: context with saved values from the forward pass
        :param grad_outputs: incoming gradient
        :return: gradients with respect to dX
        """
        dropout_mask = ctx.saved_tensors[0]
        return grad_outputs[0] * dropout_mask, None


class Dropout(nn.Module):
    def __init__(self, probability: float = 0.5):
        """
        Implementation of the Dropout layer for a neural network.
        To override the backward method, you need to create your own function
        and call its methods.
        Neurons are zeroed out with a probability of `probability`.
        """
        super().__init__()
        assert (probability >= 0) and (probability < 1)
        self.probability = probability
        self.dropout = DropoutFunction.apply

    def forward(self, x):
        """
        The theoretical implementation involves the calculation of:
            - during training: y = f(WX) * m
            - during testing: y = (1 - p) * f(WX)
        Here, `m` is a masking matrix consisting of {0, 1} to disable neurons.
        Each element takes the value 0 with a probability of p = self.probability
        For practical implementation, we divide both expressions by (1 - p)
        :param x:
        :return:
        """
        if self.training:
            # Normalize the output during training, as the neuron receives an average of (1 - p) information
            return self.dropout(x, self.probability) / (1 - self.probability)
        else:
            # Disable Dropout during testing
            return self.dropout(x, 0)