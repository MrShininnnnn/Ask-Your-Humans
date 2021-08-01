#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'


# public
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data.metrics import bleu_score
        

class InstructionsGeneratorMetrics(object):
    """for Instruction Generator"""
    def __init__(self, vocab, criterion):
        super().__init__()
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

    def flush(self, epoch, train=True):

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


class ActionMetrics(object):
    """for Imitation Learning (IL)"""
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        # step-wise
        self.running_losses, self.running_accuracy = [], []
        # epoch-wise
        self.all_losses, self.all_accuracy = [], [] 

    def get_mean(self, metric):
        return np.array(metric).mean()
    
    def calculate_loss_(self, predictions, targets):
        return self.criterion(predictions, targets)
    
    def calculate_accuracy_(self, preds, target): 
        batch_size = target.shape[0]
        _, pred = torch.max(preds, dim=-1)
        correct = pred.eq(target).sum() * 1.0
        acc = correct / batch_size
        return acc

    def add(self, predictions, targets):
        loss = self.calculate_loss_(predictions, targets)
        self.running_losses.append(loss.item())
        acc = self.calculate_accuracy_(predictions, targets).cpu()
        self.running_accuracy.append(acc)
        return loss

    def flush(self, epoch, train=True):
        avg_loss = self.get_mean(self.running_losses)
        avg_accuracy = self.get_mean(self.running_accuracy)
        if train:
            self.train_info = 'Epoch: %d, train loss: %.3f, train accuracy: %.3f' \
            % (epoch, avg_loss, avg_accuracy)
        else:
            self.valid_info = 'Epoch: %d, valid loss: %.3f, valid accuracy: %.3f' \
            % (epoch, avg_loss, avg_accuracy)

        self.all_losses.append(avg_loss)
        self.running_losses = []

        self.all_accuracy.append(avg_accuracy)
        self.running_accuracy = []