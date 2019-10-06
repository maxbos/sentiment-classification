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


class BinaryNeuron(nn.Module):

    def __init__(self, variant):
        super(BinaryNeuron, self).__init__()
        assert variant in ['D-ST', 'S-ST'], 'variant should be D-ST or S-ST'
        self.variant = variant

    def forward(self, x):
        p = torch.sigmoid(x)

        if self.variant == 'S-ST':
            turned_off = bernoulli(p)
        # use D-ST
        else:
            turned_off = estimate(p)

        return torch.mul(x, turned_off)
