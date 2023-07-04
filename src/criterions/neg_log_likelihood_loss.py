from typing import Any

import torch
import torch.nn as nn


class NLLLossFunction(torch.autograd.Function):
    """
    Function for implementing forward and backward pass of the 
    Negative Log Likelihood Loss for neural networks
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Negative Log-Likelihood Loss: L(activation, target) = - sum(target[i] * log(activation[i])) / M -> min,
        where target[i] = 1 if the object belongs to class i, otherwise target[i] = 0.
        This expression can be rewritten as:
        NLLLoss(activation, target) = - sum(log(activation[i]|target[i] = 1)) / M
            activation - outputs of the linear layer after LogSoftmax()
            target - tensor with class labels (tensor([0, 2, 1, ...]))
            M - number of objects in the dataset
        :param ctx: context for storing derivatives during forward pass
        :param args: arguments in the order [activation, target]
        :param kwargs:
        :return:
        """
        activation = args[0]
        target = args[1]
        ctx.save_for_backward(activation, target)
        # It can be done using torch.gather(activation, 1, target.to(int).view(-1, 1))
        # Equivalent in numpy: np.take_along_axis()
        predicted = activation[range(activation.shape[0]), target]
        return - predicted.mean()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Backpropagation of the gradient for Negative Log Likelihood Loss:

        In the logic of the operation, the incoming gradient is a scalar value. 
        However, subsequent layers expect the gradient to be in the form of a matrix. 
        Therefore, a tensor with non-zero values at the indices 
        specified by the target is required.

        dL/d(activation) = dL/dz * dz/d(activation) = dL/dz * (- 1 / M)
        M - number of samples in the dataset
        dL/d(target) = None, since target is the class labels (const)

        :param ctx: Context with saved values from the forward pass
        :param grad_outputs: Incoming gradient
        :return: Gradients with respect to d(activation), d(target)
        """
        activation, target = ctx.saved_tensors[0], ctx.saved_tensors[1]
        grad_matrix = torch.zeros_like(activation)
        grad_matrix[range(activation.shape[0]), target] = -1
        return grad_outputs[0] * grad_matrix / len(target), None


class NLLLoss(nn.Module):
    """
    Implementation of the Negative Log Likelihood Loss function for a neural network.
    To override the backward method, it is necessary to create your own function
    and call its methods.
    """
    def __init__(self):
        super().__init__()
        self.nllloss = NLLLossFunction.apply

    def forward(self, activation, target):
        return self.nllloss(activation, target)