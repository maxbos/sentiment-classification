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

bernoulli = Bernoulli.apply

class StochasticNeuron(nn.Module):

    def forward(self, x):
        p = torch.sigmoid(x)
        turned_off = bernoulli(p)
        return torch.mul(x, turned_off)

"""
x = torch.randn((32, 100))
neuron = StochasticNeuron()
print(neuron(x))
"""