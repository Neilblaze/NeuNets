from typing import Optional, Callable

import torch


class SGD(torch.optim.Optimizer):
    """
    Implementation of the classic Stochastic Gradient Descent (SGD) method for neural networks.
    """
    def __init__(self, params, lr: float = 1e-3):
        """
        Gradient Descent method:
        :param params: model parameters for gradient update
        :param lr: learning rate
        """
        opt_params = {
            'lr': lr,
        }
        super().__init__(params, opt_params)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        """
        Calculation of iterative weight update using the formula:
        w_(t+1) = w_(t) - lr * grad(L(w_(t))),
            w_(t+1) - weight values at the next iteration (t+1)
            w_(t) - weight values at iteration t
            lr - learning rate
            grad(*) - gradient of *
            L(w_(t)) - loss function of w_(t)
        :param closure:
        :return:
        """
        for group in self.param_groups:
            lr = group['lr']
            for param in group['params']:
                if param.grad is not None:
                    param.data += (- lr * param.grad)
        return