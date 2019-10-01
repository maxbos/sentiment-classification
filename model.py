import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
  """"""

  def __init__(self, d_model, nhead, num_layers, pretrained_embeddings, device):
    self.embedding = nn.Embedding.from_pretrained(
      pretrained_embeddings, freeze=False).to(device)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.fc = nn.Linear(100, 2)

  def forward(self, input):
    """"""
    embedded = self.embedding(input)
    out = self.ste(embedded)
    out = self.transformer_encoder(out)
    out = self.fc(out)
    return F.sigmoid(out)
