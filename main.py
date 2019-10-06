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
import matplotlib.pyplot as plt
import seaborn as sns
from cnn import CNN


def main():
  print('Starting process...')
  
  SEED = 111
  torch.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True

  TEXT = torchtext.data.Field(tokenize='spacy', batch_first=True)
  LABEL = torchtext.data.LabelField(dtype=torch.float)

  train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
  train_data, valid_data = train_data.split(random_state=random.seed(SEED))

  max_vocab_size = 25_000
  TEXT.build_vocab(
    train_data, max_size=max_vocab_size, vectors="glove.6B.100d",
    unk_init=torch.Tensor.normal_,
  )
  LABEL.build_vocab(train_data)

  train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
      (train_data, valid_data, test_data), batch_size=ARGS.batch_size, device=DEVICE)
  
  vocab_size = len(TEXT.vocab)
  pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
  filter_sizes = np.array(ARGS.filter_sizes.split(','), dtype=int)
  model = CNN(vocab_size, ARGS.binary_neuron, ARGS.embed_dim, ARGS.n_filters, filter_sizes,
    ARGS.output_dim, ARGS.dropout_rate, pad_idx)

  pretrained_embeddings = TEXT.vocab.vectors
  model.embedding.weight.data.copy_(pretrained_embeddings)
  UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
  model.embedding.weight.data[UNK_IDX] = torch.zeros(ARGS.embed_dim)
  model.embedding.weight.data[pad_idx] = torch.zeros(ARGS.embed_dim)

  criterion = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters())

  model.to(DEVICE)
  criterion.to(DEVICE)

  min_valid_loss = float('inf')
  for epoch in range(1, ARGS.epochs + 1):
    start_time = time.time()

    model.train()
    train_loss, train_acc, train_p, train_tn, train_fp, train_fn = run_epoch(model, train_iter, criterion, optimizer)

    model.eval()
    with torch.no_grad():
      valid_loss, valid_acc, val_p, val_tn, val_fp, val_fn = run_epoch(model, valid_iter, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < min_valid_loss:
      min_valid_loss = valid_loss
      torch.save(model.state_dict(), 'model.pt')
    
    print(
      f'Epoch: {epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n' \
      f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%\n' \
      f'\tVal. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%'
    )
  
  test_loss, test_acc, test_tp, test_tn, test_fp, test_fn  = run_epoch(model, test_iter, criterion)
  print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
  sns.heatmap(
    np.array([[test_tp, test_fp], [test_fn, test_tn]]),
    vmax=.5,linewidth=0.5, cmap="Blues", xticklabels=["Positive", "Negative"], yticklabels=["True", "False"])
  print(np.array([[test_tp, test_fp], [test_fn, test_tn]]))
  plt.show()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calc_accuracy(output, labels):
  predictions = torch.round(torch.sigmoid(output))
  return (predictions == labels).float().mean()


def calc_error_dist(output, labels):
  predictions = torch.round(torch.sigmoid(output))
  tp = (predictions == labels)[torch.where(labels == 1)].float().sum().item() / len(labels)
  tn = (predictions == labels)[torch.where(labels == 0)].float().sum().item() / len(labels)
  fp = (predictions != labels)[torch.where(labels == 1)].float().sum().item() / len(labels)
  fn = (predictions != labels)[torch.where(labels == 0)].float().sum().item() / len(labels)
  return tp, tn, fp, fn


def run_epoch(model, iter, criterion, optimizer=None):
  total_loss = 0.
  total_acc = 0.
  total_tp = 0.
  total_tn = 0.
  total_fp = 0.
  total_fn = 0.
  for batch in tqdm(iter):
    if model.training:
      optimizer.zero_grad()
    output = model(batch.text).squeeze(1)
    loss = criterion(output, batch.label)
    accuracy = calc_accuracy(output, batch.label)
    tp, tn, fp, fn = calc_error_dist(output, batch.label)
    total_tp += tp
    total_tn += tn
    total_fp += fp
    total_fn += fn

    if model.training:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
      optimizer.step()
    total_loss += loss.item()
    total_acc += accuracy
  return (
    total_loss/len(iter), total_acc/len(iter), total_tp/len(iter),
    total_tn/len(iter), total_fp/len(iter), total_fn/len(iter))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', default=4, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=64, type=int,
                      help='batch size')
  parser.add_argument('--embed_dim', default=100, type=int,
                      help='embedding size')
  parser.add_argument('--n_filters', default=100, type=int,
                      help='number of filters')
  parser.add_argument('--filter_sizes', default="2,3,4,5", type=str,
                      help='filter sizes')
  parser.add_argument('--output_dim', default=1, type=int,
                      help='number of outputs')
  parser.add_argument('--dropout_rate', default=0.5, type=float,
                      help='dropout rate')
  parser.add_argument('--binary_neuron', default="D-ST", type=str,
                      help='Type of binary neuron to use, has to be D-ST or S-ST')

  ARGS = parser.parse_args()
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  main()
