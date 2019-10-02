import torch
import torch.nn as nn

from torch.autograd import Function

class Bernoulli(Function):
    
    @staticmethod
    def forward(ctx, x):
        return torch.bernoulli(x)

    @staticmethod
    def backward(ctx, gradient):
        return gradient

class Estimate(Function):

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, gradient):
        return gradient

bernoulli = Bernoulli.apply
estimate = Estimate.apply

class StochasticNeuron(nn.Module):
    
    def __init__(self):
        super(StochasticNeuron, self).__init__()

    def __init__(self, variant='StraightThrough'):
        super(StochasticNeuron, self).__init__()
        self.variant = variant
        # self.l = nn.Linear(1, 1)

    def forward(self, x):
        p = torch.sigmoid(x)

        if self.variant == 'StraightThrough':
            turned_off = bernoulli(p)
        # use REINFORCE
        else:
            turned_off = estimate(p)

        return torch.mul(x, turned_off)


"""
x = torch.randn((32, 100))
neuron = StochasticNeuron()
print(neuron(x))
"""