import argparse
import torch
import os
import numpy as np
import torchtext
import torchtext.datasets as datasets

from model import Model


def main():
  print('Starting process...')

  train_iter, test_iter = datasets.IMDB.iters(batch_size=ARGS.batch_size)
  
  model = Model(
    ntokens=200,
    d_model=ARGS.d_model, nhead=ARGS.nhead, num_layers=ARGS.num_layers,
    device=DEVICE,
  )
  model.to(DEVICE)

  cross_entropy = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())
  
  for epoch in range(1, ARGS.epochs + 1):
    train(train_iter)
    evaluate(test_iter)

def evaluate(model, test_iter):
  loss = 0.
  model.eval()
  for i, batch in test_iter:
    loss += model(batch)

  return loss / (i + 1)

def train(train_iter):
  model.train()
  loss = 0.
  print(train_iter)
  for i, batch in enumerate(train_iter):
    print(batch)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', default=30, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=4, type=int,
                      help='batch size')
  parser.add_argument('--d_model', default=300, type=int,
                      help='embedding size')
  parser.add_argument('--nhead', default=2, type=int,
                      help='heads in multi head attention')
  parser.add_argument('--num_layers', default=6, type=int,
                      help='amount of layers transformer')

  ARGS = parser.parse_args()
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  main()
