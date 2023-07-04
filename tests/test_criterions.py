import unittest
import torch
from torch.autograd import gradcheck

from src.criterions.neg_log_likelihood_loss import NLLLossFunction
from src.criterions.focal_loss import FocalLossFunction


class TestCriterions(unittest.TestCase):

    def test_nllloss_autograd(self):
        neg_log_loss_func = NLLLossFunction.apply
        x = torch.randn(3, 4, requires_grad=True).double()
        y = torch.randint(4, (3,))
        self.assertTrue(gradcheck(neg_log_loss_func, (x, y)))

    def test_focal_loss_autograd(self):
        focal_loss_func = FocalLossFunction.apply
        x = torch.randn(3, 4, requires_grad=True).double()
        y = torch.randint(4, (3,))
        gamma = 2
        self.assertTrue(gradcheck(focal_loss_func, (torch.nn.functional.softmax(x, dim=-1), y, gamma)))