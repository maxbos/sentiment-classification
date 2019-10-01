import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from stochastic_neuron import StochasticNeuron


class PositionalEncoding(nn.Module):
  """
  Positional encoding.
  """

  def __init__(self, d_model, dropout=0.0, max_len=5000):
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
    d_mpool, n_output=2,
  ):
    super(Model, self).__init__()

    self.d_model = d_model
    self.d_mpool = d_mpool

    self.embedding = nn.Embedding(ntokens, d_model)
    self.pos_encoder = PositionalEncoding(d_model)
    self.ste = StochasticNeuron()
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    # self.maxpool = nn.AdaptiveMaxPool2d((d_mpool, d_mpool))
    self.maxpool = nn.AdaptiveAvgPool2d((d_mpool, d_mpool))
    self.fc1 = nn.Linear(d_mpool*d_mpool, 256)
    self.fc2 = nn.Linear(256, n_output)

  def forward(self, input):
    """"""
    embedded = self.embedding(input)
    pos_encoded = self.pos_encoder(embedded)
    # out = self.ste(pos_encoded)
    out = self.transformer_encoder(pos_encoded)
    out = self.maxpool(out)
    out = out.view(-1, self.d_mpool*self.d_mpool)
    out = self.fc1(out)
    out = self.fc2(out)
    return F.logsigmoid(out)
