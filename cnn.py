import torch
import torch.nn as nn
import torch.nn.functional as F
from stochastic_neuron import StochasticNeuron

class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        ste,
        embedding_dim,
        filter_amount,
        kernel_sizes,
        classes_amount,
        dropout,
        pad_idx
    ):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.ste = StochasticNeuron('REINFORCE') if ste == 'REINFORCE' else StochasticNeuron()
        self.ste_type = ste

        kernels = []
        for size in kernel_sizes:
            k = nn.Conv2d(
                in_channels=1,
                out_channels=filter_amount,
                kernel_size=(size, embedding_dim)
            ) 
            kernels.append(k)

        self.kernels = nn.ModuleList(kernels)
        self.attention = nn.Linear(len(kernel_sizes) * filter_amount, classes_amount)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        if self.ste_type in ['REINFORCE', 'ST']:
            embeddings = self.ste(self.embedding(text))
        else:
            embeddings = self.embedding(text)

        embedded = embeddings.unsqueeze(1)
  
        convoluted = []
        for kernel in self.kernels:
            convoluted.append(F.relu(kernel(embedded)).squeeze(3))

        maximum = []
        for c in convoluted:
            maximum.append(F.max_pool1d(c, c.shape[2]).squeeze(2))

        concatenated = torch.cat(maximum, dim=1)

        logits = self.dropout(concatenated)
        return self.attention(logits)
  