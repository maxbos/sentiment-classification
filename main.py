import argparse
import torch
import os
import numpy as np

from model import Model


def main():
  print('Starting process...')

  model = Model()
  model.to(DEVICE)

  cross_entropy = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())
  
  for epoch in range(ARGS.epochs):
    


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', default=30, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=4, type=int,
                      help='batch size')
  parser.add_argument('--device', default='cpu', type=str,
                      help='device')
  parser.add_argument('--data_dir', default='data', type=str,
                      help='directory where the data is stored')

  ARGS = parser.parse_args()
  DEVICE = torch.device(ARGS.device)
  main()
