"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            u_w = 0 if self.u.get(w) is None else self.u.get(w)
            u_w = self.momentum * u_w + (1 - self.momentum) * (w.grad.detach() + w.detach() * self.weight_decay)
            w.cached_data = w.cached_data - self.lr * u_w.cached_data
            self.u[w] = u_w

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for w in self.params:
            m_w = 0 if self.m.get(w) is None else self.m.get(w)
            v_w = 0 if self.v.get(w) is None else self.v.get(w)
            m_w = self.beta1 * m_w + (1 - self.beta1) * (w.grad.detach() + w.detach() * self.weight_decay)
            v_w = self.beta2 * v_w + (1 - self.beta2) * (w.grad.detach() + w.detach() * self.weight_decay) ** 2
            
            self.m[w] = m_w
            self.v[w] = v_w
            
            m_w = self.m[w].cached_data / (1 - self.beta1 ** self.t)
            v_w = self.v[w].cached_data / (1 - self.beta2 ** self.t)
            
            w.cached_data = w.cached_data - self.lr * m_w / ((v_w ** 0.5) + self.eps)
