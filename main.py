import argparse
import time
import torch
import os
import random
from tqdm import tqdm
import numpy as np
import torchtext
import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import GloVe

from model import Model
from cnn import CNN


def main():
  print('Starting process...')
  
  SEED = 1234
  torch.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True

  # set up fields
  TEXT = torchtext.data.Field(tokenize='spacy', batch_first=True)
  LABEL = torchtext.data.LabelField(dtype=torch.float)

  # make splits for data
  train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
  train_data, valid_data = train_data.split(random_state=random.seed(SEED))

  # build the vocabulary
  MAX_VOCAB_SIZE = 25_000
  TEXT.build_vocab(
    train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d",
    unk_init=torch.Tensor.normal_,
  )
  LABEL.build_vocab(train_data)

  # make iterator for splits
  train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
      (train_data, valid_data, test_data), batch_size=ARGS.batch_size, device=DEVICE)
  
  vocab_size = len(TEXT.vocab)
  N_FILTERS = 100
  FILTER_SIZES = [3,4,5]
  OUTPUT_DIM = 1
  DROPOUT = 0.5
  PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

  model = CNN(vocab_size, ARGS.embed_dim, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

  pretrained_embeddings = TEXT.vocab.vectors
  model.embedding.weight.data.copy_(pretrained_embeddings)
  UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
  model.embedding.weight.data[UNK_IDX] = torch.zeros(ARGS.embed_dim)
  model.embedding.weight.data[PAD_IDX] = torch.zeros(ARGS.embed_dim)

  criterion = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters())

  model.to(DEVICE)
  criterion.to(DEVICE)

  best_valid_loss = float('inf')
  for epoch in range(1, ARGS.epochs + 1):
    start_time = time.time()

    model.train()
    train_loss, train_acc = run_epoch(model, train_iter, criterion, optimizer)

    model.eval()
    with torch.no_grad():
      valid_loss, valid_acc = run_epoch(model, valid_iter, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      torch.save(model.state_dict(), 'model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
  
  # Perform a final test evaluation
  test_loss, test_acc = run_epoch(model, test_iter, criterion)
  print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calc_accuracy(output, labels):
  rounded_preds = torch.round(torch.sigmoid(output))
  # correct = (rounded_preds == labels).float()
  # acc = correct.sum() / len(correct)
  return (rounded_preds == labels).float().mean()


def run_epoch(model, iter, criterion, optimizer=None):
  total_loss = 0.
  total_acc = 0.
  for batch in tqdm(iter):
    if model.training:
      optimizer.zero_grad()
    output = model(batch.text).squeeze(1)
    accuracy = calc_accuracy(output, batch.label)
    if model.training:
      loss.backward()
      optimizer.step()
    total_loss += loss.item()
    total_acc += accuracy
  return total_loss/len(iter), total_acc/len(iter)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', default=4, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=64, type=int,
                      help='batch size')
  parser.add_argument('--embed_dim', default=100, type=int,
                      help='embedding size')

  ARGS = parser.parse_args()
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  main()
