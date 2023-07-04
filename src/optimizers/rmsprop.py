from typing import Optional, Callable

import torch


class RMSprop(torch.optim.Optimizer):
    """
    Implementation of the RMSprop method for neural networks.
    The method is an enhancement of AdaGrad by adding a moving average 
    for the parameter v.
    """
    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.8, eps: float = 1e-6):
        """
        RMSprop method: AdaGrad + Moving Average
        :param params: model parameters for gradient update
        :param lr: learning rate
        :param alpha: coefficient for moving average
        :param eps: protection against division by zero
        """
        assert (alpha > 0) and (alpha < 1.0)
        # Since params is a generator, it needs to be saved for traversal in multiple calls.
        model_params = list(params)
        opt_params = {
            'lr': lr,
            'alpha': alpha,
            'eps': eps,
        }
        self.v = [torch.zeros_like(param) for param in model_params]
        super().__init__(model_params, opt_params)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        """
        Calculation of iterative weight updates using the formula:
        v_(t+1) = alpha * v_(t) + (1 - alpha) * grad(L(w_(t)))**2
        w_(t+1) = w_(t) - lr * grad(L(w_(t))) / sqrt(v_(t+1) + eps),
            w_(t+1) - values of weights at the next iteration (t+1)
            w_(t) - values of weights at iteration t
            v_(t+1) - normalization of the convergence rate at step (t+1)
            v_(0) - matrix of zeros with the same dimensions as grad(L(w_(t)))
            lr - learning rate
            alpha - influence of the previous value of v
            eps - small constant for protection against division by zero
            grad(*) - gradient of *
            L(w_(t)) - loss function of w_(t)
        :param closure:
        :return:
        """
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            for i, param in enumerate(group['params']):
                if param.grad is not None:
                    self.v[i] = alpha * self.v[i] + (1 - alpha) * torch.square(param.grad)
                    param.data += (- lr * param.grad / torch.sqrt(self.v[i] + eps))
        return