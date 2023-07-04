from typing import Optional, Callable

import torch


class AdaGrad(torch.optim.Optimizer):
    """
    Implementation of the AdaGrad optimization method for a neural network.
    """
    def __init__(self, params, lr: float = 1e-2, eps: float = 1e-6):
        """
        AdaGrad method.
        :param params: model parameters for gradient update
        :param lr: learning rate
        :param eps: protection against division by zero
        """
        # Since `params` is a generator, it needs to be saved for iteration in multiple calls.
        model_params = list(params)
        opt_params = {
            'lr': lr,
            'eps': eps,
        }
        self.v = [torch.zeros_like(param) for param in model_params]
        super().__init__(model_params, opt_params)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        """
        Calculation of iterative weight update using the formula:
        v_(t+1) = v_(t) + grad(L(w_(t)))**2
        w_(t+1) = w_(t) - lr * grad(L(w_(t))) / sqrt(v_(t+1) + eps),
            w_(t+1) - values of the weights at the next iteration (t+1)
            w_(t) - values of the weights at iteration t
            v_(t+1) - convergence rate normalization at step (t+1)
            v_(0) - matrix of zeros with the dimension of grad(L(w_(t)))
            lr - learning rate
            eps - small constant for protection against division by zero
            grad(*) - gradient of *
            L(w_(t)) - loss function of w_(t)
        :param closure:
        :return:
        """
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            for i, param in enumerate(group['params']):
                if param.grad is not None:
                    self.v[i] += torch.square(param.grad)
                    param.data += (- lr * param.grad / torch.sqrt(self.v[i] + eps))
        return