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
  
  SEED = 1234
  torch.manual_seed(SEED)

  # set up fields
  TEXT = torchtext.data.Field(tokenize='spacy', batch_first=True)
  LABEL = torchtext.data.LabelField(dtype=torch.float)

  # make splits for data
  train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

  # build the vocabulary
  MAX_VOCAB_SIZE = 25_000
  TEXT.build_vocab(
    train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d",
    unk_init=torch.Tensor.normal_,
  )
  LABEL.build_vocab(train_data)

  # make iterator for splits
  train_iter, test_iter = torchtext.data.BucketIterator.splits(
      (train_data, test_data), batch_size=ARGS.batch_size, device=0)
  
  ntokens = len(TEXT.vocab)
  PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
  model = Model(
    ntokens=ntokens,
    d_model=ARGS.d_model, nhead=ARGS.nhead, num_layers=ARGS.num_layers,
    d_mpool=ARGS.d_mpool,
    pad_idx=PAD_IDX,
    device=DEVICE,
  )

  pretrained_embeddings = TEXT.vocab.vectors
  model.embedding.weight.data.copy_(pretrained_embeddings)
  UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
  model.embedding.weight.data[UNK_IDX] = torch.zeros(ARGS.d_model)
  model.embedding.weight.data[PAD_IDX] = torch.zeros(ARGS.d_model)

  criterion = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters())

  for epoch in range(1, ARGS.epochs + 1):
    train(epoch, ntokens, model, train_iter, criterion, optimizer)


# def accuracy(output, labels):
#   predicted = output.argmax(-1)
#   return (predicted == (labels-1).long()).float().mean().item()
def accuracy(output, labels):
  rounded_preds = torch.round(torch.sigmoid(output))
  correct = (rounded_preds == labels).float() #convert into float for division 
  acc = correct.sum() / len(correct)
  return acc


def train(epoch, ntokens, model, train_iter, criterion, optimizer):
  """"""
  model.train()
  start_time = time.time()
  total_loss = 0.
  for i, batch in enumerate(train_iter):
    optimizer.zero_grad()
    output = model(batch.text).squeeze(1)
    loss = criterion(output, batch.label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    
    total_loss += loss.item()
    log_interval = 3
    if i % log_interval == 0 and i > 0:
      cur_loss = total_loss / log_interval
      elapsed = time.time() - start_time
      train_accuracy = accuracy(output, batch.label)
      print('| epoch {:3d} | {:5d}/{:5d} batches | '
            'ms/batch {:5.2f} | '
            'loss {:5.5f} | train acc {:5.2f}'.format(
              epoch, i, len(train_iter),
              elapsed * 1000 / log_interval,
              cur_loss, train_accuracy))
      total_loss = 0.
      start_time = time.time()


# def evaluate(model, loss, test_iter):
#   model.eval()
#   with torch.no_grad():
    


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', default=4, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=32, type=int,
                      help='batch size')
  parser.add_argument('--d_model', default=100, type=int,
                      help='embedding size')
  parser.add_argument('--nhead', default=4, type=int,
                      help='heads in multi head attention')
  parser.add_argument('--num_layers', default=2, type=int,
                      help='amount of layers transformer')
  parser.add_argument('--d_mpool', default=64, type=int,
                      help='number of vectors in sequence pooling')

  ARGS = parser.parse_args()
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  main()
