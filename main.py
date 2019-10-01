import argparse
import torch
import os
import numpy as np
import torchtext
import torchtext.data as data
# import torchtext.vocab.GloVe as GloVe
import torchtext.datasets as datasets

from model import Model


def main():
  print('Starting process...')

  TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
  LABEL = data.Field(sequential=False)

  # make splits for data
  train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

  # build the vocabulary
  TEXT.build_vocab(train_data) # vectors=GloVe(name='6B', dim=300))
  LABEL.build_vocab(train_data)

  # make iterator for splits
  train_iter, test_iter = data.BucketIterator.splits(
      (train_data, test_data), batch_size=1, device=0)

  model = Model(
    ntokens=len(TEXT.vocab.stoi),
    d_model=ARGS.d_model, nhead=ARGS.nhead, num_layers=ARGS.num_layers,
    device=DEVICE,
  )

  cross_entropy = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters())
  
  print("wwwwww", train_iter)
  for epoch in range(1, ARGS.epochs + 1):
    train(train_iter, model, cross_entropy)
    evaluate(model, cross_entropy, test_iter)

def evaluate(model, loss, test_iter):
  loss = 0.
  model.eval()
  for i, batch in test_iter:
    loss += loss(model(batch))

  return loss / (i + 1)

def train(train_iter, model, loss_function):
  model.train()
  loss = 0.
  print(train_iter)
  for i, batch in enumerate(train_iter):

    # dont know what the second part of the tuple is
    probs = model(batch.text[0]) # batch.label)
    print(probs)
    print(batch.label)
    print(loss_function(probs, batch.label))

    print(probs.size())

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
