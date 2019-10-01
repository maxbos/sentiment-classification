import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from stochastic_neuron import StochasticNeuron


class PositionalEncoding(nn.Module):
  """
  Positional encoding.
  """

  def __init__(self, d_model, dropout=0.1, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)


class Model(nn.Module):
  """"""

  def __init__(
    self, ntokens, d_model, nhead, num_layers, device,
    n_output=2,
  ):
    super(Model, self).__init__()
    self.embedding = nn.Embedding(ntokens, d_model)
    self.pos_encoder = PositionalEncoding(d_model)
    self.ste = StochasticNeuron()
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.fc = nn.Linear(d_model, n_output)

  def forward(self, input):
    """"""
    embedded = self.embedding(input)
    pos_encoded = self.pos_encoder(embedded)
    out = self.ste(pos_encoded)
    out = self.transformer_encoder(out)
    print('out transformer', out.size())
    out = self.fc(out)
    return F.sigmoid(out)
