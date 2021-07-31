#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'


import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data.metrics import bleu_score


class InstructionsGeneratorMetrics(object):

  def __init__(self, vocab, criterion):
    self.all_losses = []
    self.running_losses = []
    self.all_bleu_scores = []
    self.running_bleu_scores = []
    self.all_token_accuracy = []
    self.running_token_accuracy = []
    # self.all_seq_accuracy = []
    # self.running_seq_accuracy = []
    self.vocab = vocab
    self.criterion = criterion
    self.train_info, self.valid_info = None, None

  def check_token(self, preds, targets):
    min_len = min([len(preds), len(targets)])
    max_len = max([len(preds), len(targets)])
    return np.float32(
        sum(
            np.equal(
                np.array(preds[:min_len], dtype=object),
                np.array(targets[:min_len], dtype=object))) / max_len)

  # def check_seq(self, preds, targets):
  #   min_len = min([len(preds), len(targets)])
  #   max_len = max([len(preds), len(targets)])
  #   if sum(
  #       np.equal(
  #           np.array(preds[:min_len], dtype=object),
  #           np.array(targets[:min_len], dtype=object))) == max_len:
  #     return 1
  #   return 0

  def calculate_token_accuracy(self, preds, targets):
    match_count = 0
    for i in range(len(preds)):
      match_count += self.check_token(preds, targets)
    return np.float32(match_count / len(preds))

  # def calculate_seq_accuracy(self, preds, targets):
  #   match_count = 0
  #   for i in range(len(preds)):
  #     match_count += self.check_seq(preds, targets)
  #   return np.float32(match_count / len(preds))

  def calculate_bleu_score_(self, preds, targets):
    preds_words = np.array([
        list(map(lambda x: self.vocab.idx2word[x], preds[i]))
        for i in range(preds.shape[0])
    ])
    targets_words = np.array(
        [[list(map(lambda x: self.vocab.idx2word[x], targets[i]))]
         for i in range(targets.shape[0])])
    return bleu_score(preds_words, targets_words)

  def calculate_loss_(self, predictions, targets, decode_lengths, alphas):
    predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True)[0]
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
    loss = self.criterion(predictions, targets)
    loss += 1. * ((1. - alphas.sum(dim=1))**2).mean()
    return loss

  def add(self, predictions, targets, decode_lengths, alphas):
    loss = self.calculate_loss_(predictions, targets, decode_lengths, alphas)
    self.running_losses.append(loss.item())

    tokens_preds = np.argmax(predictions.detach().cpu().numpy(), axis=2)
    tokens_targets = targets.cpu().numpy()
    self.running_bleu_scores.append(
        self.calculate_bleu_score_(tokens_preds, tokens_targets))

    self.running_token_accuracy.append(
        self.calculate_token_accuracy(tokens_preds.tolist(),
                                      targets.tolist()).mean())
    # self.running_seq_accuracy.append(
    #     self.calculate_seq_accuracy(tokens_preds.tolist(), targets.tolist()))

    return loss

  def get_mean(self, metric):
    return np.array(metric).mean()

  def flush(self, epoch, idx, train=True):

    if train:
        self.train_info = 'Epoch: %d, train loss: %.3f, train bleu: %.3f, train token acc: %.3f' \
        % (epoch
            , self.get_mean(self.running_losses)
            , self.get_mean(self.running_bleu_scores)
            , self.get_mean(self.running_token_accuracy)
            )
    else:
        self.valid_info = 'Epoch: %d, valid loss: %.3f, valid bleu: %.3f, valid token acc: %.3f' \
        % (epoch
            , self.get_mean(self.running_losses)
            , self.get_mean(self.running_bleu_scores)
            , self.get_mean(self.running_token_accuracy)
             )

    self.all_losses.append(self.get_mean(self.running_losses))
    self.running_losses = []

    self.all_bleu_scores.append(self.get_mean(self.running_bleu_scores))
    self.running_bleu_scores = []

    self.all_token_accuracy.append(self.get_mean(self.running_token_accuracy))
    self.running_token_accuracy = []

    # self.all_seq_accuracy.append(self.get_mean(self.running_seq_accuracy))
    # self.running_seq_accuracy = []