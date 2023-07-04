from typing import Any

import torch
import torch.nn as nn


class LogSoftmaxFunction(torch.autograd.Function):
    """
    Function for implementing the forward and backward pass of the 
    LogSoftmax layer in a neural network.
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Softmax(X) = exp(x_i) / sum(exp(x_j)), where i = 1...K (number of classes)
        To avoid overflow in the exponentials, we use the Log-Sum-Exp trick.
        We can either use the built-in function torch.logsumexp or implement it ourselves:
            x_max = torch.max(x)
            s = torch.exp(x - x_max).sum()
            lse = torch.log(s)
        We obtain softmax(X) = exp(x_i) / sum(exp(x_j)) = exp(x_i - lse(x))
            -> log(softmax(x)) = log(exp(x_i - lse)) = x_i - lse
        :param ctx: context for saving derivatives during the forward pass
        :param args: arguments in the order [X,]
        :param kwargs:
        :return:
        """
        x = args[0]
        lse = torch.logsumexp(x, dim=-1).view(-1, 1)
        log_softmax = x - lse
        ctx.save_for_backward(log_softmax)
        return log_softmax


    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Backpropagation of gradients for LogSoftmax(X)
        For our specific loss function, we can compute all calculations in matrix form.
        Therefore, we assume that the argument grad_outputs[0] is a matrix with only one non-zero value in each row (not necessarily one).

        dL/dX = dL/dz * dz/dX = dL/dz * (1 - exp(logsoftmax(x)))
            z = x - lse(x)
            dz/dx = 1 - exp(x_i) / sum(exp(x_j)) = 1 - softmax(x) = 1 - exp(logsoftmax(x))

        :param ctx: context with saved values during the forward pass
        :param grad_outputs: incoming gradient
        :return: gradients with respect to dX
        """
        log_softmax = ctx.saved_tensors[0]
        grad_output = grad_outputs[0]
        result = grad_output - torch.sum(grad_output, dim=-1).view(-1, 1) * torch.exp(log_softmax)
        return result


class LogSoftmax(nn.Module):
    """
    Implementation of LogSoftmax layer in a neural network.
    To override the backward method, you need to create your 
    own function and call its methods.
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmaxFunction.apply

    def forward(self, x):
        return self.log_softmax(x)