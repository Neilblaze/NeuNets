import unittest
import torch
from torch.autograd import gradcheck

from src.layers.linear import LinearFunction
from src.layers.relu import ReLUFunction
from src.layers.sigmoid import SigmoidFunction
from src.layers.dropout import DropoutFunction
from src.layers.log_softmax import LogSoftmaxFunction


class TestLayers(unittest.TestCase):
    def test_linear_autograd(self):
        lin_func = LinearFunction.apply
        x = torch.randn(2, 3, requires_grad=True).double()
        w = torch.randn(5, 3, requires_grad=True).double()
        b = torch.zeros(5, requires_grad=True).double()
        self.assertTrue(gradcheck(lin_func, (x, w, b)))

    def test_relu_autograd(self):
        relu_func = ReLUFunction.apply
        x = torch.randn(2, 3, requires_grad=True).double()
        self.assertTrue(gradcheck(relu_func, x))

    def test_sigmoid_autograd(self):
        sigmoid_func = SigmoidFunction.apply
        x = torch.randn(2, 3, requires_grad=True).double()
        self.assertTrue(gradcheck(sigmoid_func, x))

    def test_dropout_autograd(self):
        drop_func = DropoutFunction.apply
        x = torch.randn(2, 3, requires_grad=True).double()
        probability = 0.0
        self.assertTrue(gradcheck(drop_func, (x, probability)))

    def test_log_softmax_autograd(self):
        log_softmax_func = LogSoftmaxFunction.apply
        x = torch.randn(2, 3, requires_grad=True).double()
        self.assertTrue(gradcheck(log_softmax_func, x))