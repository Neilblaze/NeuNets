from typing import Any

import torch
import torch.nn as nn


class FocalLossFunction(torch.autograd.Function):
    """
    Function for implementing forward and backward passes 
    of the Focal Loss function for a neural network.
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Formula: L(activation, target) = - (1 - activation)^(gamma) * log(activation) -> min,
        where target[i] = 1 if the object belongs to the i-th class, otherwise target[i] = 0.
            activation - outputs of the linear layer after Softmax()
            target - tensor with class labels (tensor([0, 2, 1, ...]))
            gamma - "tightness" parameter (as gamma increases, the function drops faster when saturated)
            M - number of objects in the dataset
        :param ctx: context for saving derivatives during the forward pass
        :param args: arguments in the order [activation, target, gamma]
        :param kwargs:
        :return:
        """
        activation = args[0]
        target = args[1]
        gamma = args[2]

        ctx.save_for_backward(activation, target, torch.tensor(gamma, requires_grad=False))
        target_mask = (range(activation.shape[0]), target)

        predicted = ((1 - activation[target_mask]) ** gamma) * torch.log(activation)[target_mask]

        return - predicted.mean()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Backpropagation of gradients for Focal Loss:
        
        According to the logic of the operation, the incoming gradient is a scalar, 
        but subsequent layers expect the gradient in the form of a matrix, 
        so we need a tensor with non-zero values at the target indices.

        dL/d(activation) = dL/dz * dz/d(activation) =
            = dL/dz * (gamma * (1 - activation)^(gamma - 1) * log(activation) - ((1 - activation)^(gamma)) / activation)
            M - number of objects in the dataset
        dL/d(target) = None, as target is class labels (const)
        dL/d(gamma) = None, as gamma is constant

        :param ctx: context with saved values from the forward pass
        :param grad_outputs: incoming gradient
        :return: gradients for d(activation), d(target), d(gamma)
        """
        activation = ctx.saved_tensors[0]
        target = ctx.saved_tensors[1]
        gamma = ctx.saved_tensors[2]

        grad_matrix = torch.zeros_like(activation)
        target_mask = (range(activation.shape[0]), target)

        grad_matrix[target_mask] = (
                gamma * ((1 - activation[target_mask]) ** (gamma - 1)) * torch.log(activation[target_mask]) -
                ((1 - activation[target_mask]) ** gamma) / activation[target_mask]
        )

        return grad_outputs[0] * grad_matrix / len(target), None, None


class FocalLoss(nn.Module):
    """
    Implementation of the FocalLoss layer for a neural network.
    To override the backward method, it is necessary to create a custom function and call its methods.

    The SoftMax operation is applied internally in the forward pass.
    """
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.softmax = nn.Softmax(dim=-1)
        self.focal_loss = FocalLossFunction.apply

    def forward(self, activation, target):
        """
        Forward pass of Focal Loss
        :param activation: output after Linear layer
        :param target: class labels
        :return:
        """
        return self.focal_loss(self.softmax(activation), target, self.gamma)