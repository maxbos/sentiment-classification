import argparse
import time
import torch
import os
import math
import numpy as np
import torchtext
import torchtext.data as data
# import torchtext.vocab.GloVe as GloVe
import torchtext.datasets as datasets
from torchtext.vocab import GloVe

from model import Model


def main():
  print('Starting process...')
  
  # set up fields
  TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True)
  LABEL = torchtext.data.Field(sequential=False)

  # make splits for data
  train_split, test_split = datasets.IMDB.splits(TEXT, LABEL)

  # build the vocabulary
  TEXT.build_vocab(train_split, vectors=GloVe(name='6B', dim=ARGS.d_model))
  LABEL.build_vocab(train_split)

  # make iterator for splits
  train_iter, test_iter = torchtext.data.BucketIterator.splits(
      (train_split, test_split), batch_size=ARGS.batch_size, device=0)
  
  ntokens = len(TEXT.vocab.stoi)
  model = Model(
    ntokens=ntokens,
    d_model=ARGS.d_model, nhead=ARGS.nhead, num_layers=ARGS.num_layers,
    device=DEVICE,
  )

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())

  for epoch in range(1, ARGS.epochs + 1):
    train(epoch, ntokens, model, train_iter, criterion, optimizer)


def train(epoch, ntokens, model, train_iter, criterion, optimizer):
  """"""
  model.train()
  start_time = time.time()
  loss = 0.
  for i, batch in enumerate(train_iter):
    data, targets = batch
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output.view(-1, ntokens), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    total_loss += loss.item()
    log_interval = 200
    if i % log_interval == 0 and i > 0:
      cur_loss = total_loss / log_interval
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches | '
            'ms/batch {:5.2f} | '
            'loss {:5.2f} | ppl {:8.2f}'.format(
              epoch, i, len(train_iter),
              elapsed * 1000 / log_interval,
              cur_loss, math.exp(cur_loss)))
      total_loss = 0
      start_time = time.time()


def evaluate(model, loss, test_iter):
  loss = 0.
  model.eval()
  for i, batch in test_iter:
    loss += model(batch)
  return loss / (i + 1)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', default=30, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=4, type=int,
                      help='batch size')
  parser.add_argument('--d_model', default=50, type=int,
                      help='embedding size')
  parser.add_argument('--nhead', default=2, type=int,
                      help='heads in multi head attention')
  parser.add_argument('--num_layers', default=6, type=int,
                      help='amount of layers transformer')

  ARGS = parser.parse_args()
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  main()
