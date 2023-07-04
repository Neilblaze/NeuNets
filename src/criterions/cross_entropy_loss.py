import torch.nn as nn

from src.layers.log_softmax import LogSoftmax
from src.criterions.neg_log_likelihood_loss import NLLLoss


class CrossEntropyLoss(nn.Module):
    """
    Implementation of the CrossEntropyLoss function for a neural network.
    To override the backward method, it is necessary to create your own function
    and call its methods.

    In PyTorch, the implementation is as follows: CrossEntropyLoss() = LogSoftmax() + NLLLoss(),
    so the criterion takes the outputs from the linear layer and applies LogSoftmax()
    internally.
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.nllloss = NLLLoss()

    def forward(self, activation, target):
        return self.nllloss(self.log_softmax(activation), target)