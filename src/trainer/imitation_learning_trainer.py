#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'

# public
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
# private
from ..utils.eva import InstructionsGeneratorMetrics


def calculate_accuracy(preds, target):
  batch_size = target.shape[0]
  _, pred = torch.max(preds, dim=-1)
  correct = pred.eq(target).sum() * 1.0
  acc = correct / batch_size

  return acc


def train(device,
          epoch,
          train_data_loader,
          il_model,
          lstm_model,
          optimizer,
          lstm_optimizer,
          criterion,
          lstm_criterion,
          parameters,
          lstm_parameters,
          vocab,
          use_generative_language=True,
          summary_writer=None):
  il_model.train()
  lstm_model.train()
  metrics = InstructionsGeneratorMetrics(vocab, lstm_criterion)

  t = tqdm(train_data_loader)

  all_losses = []
  all_accuracy = []

  for i, data in enumerate(t):
    grid_onehot, grid_embedding, inventory_embedding, action, goal_embedding, instructions, lengths = data

    grid_onehot = grid_onehot.to(device)
    grid_embedding = grid_embedding.to(device)
    goal_embedding = goal_embedding.to(device)
    inventory_embedding = inventory_embedding.to(device)

    instructions = instructions.to(device)
    action = action.to(device, dtype=torch.int64)
    action = action.squeeze(1)

    if use_generative_language:
      use_teacher_forcing = True if random.random() < 0.5 else False

      lstm_predictions, decode_lengths, alphas, lstm_hiddens = lstm_model(
          grid_embedding,
          grid_onehot,
          inventory_embedding,
          goal_embedding,
          instructions,
          lengths,
          use_teacher_forcing=use_teacher_forcing)
    else:
      lstm_hiddens = None

    predictions = il_model(
        grid_embedding,
        grid_onehot,
        inventory_embedding,
        goal_embedding,
        lstm_hiddens=lstm_hiddens)

    accuracy = calculate_accuracy(predictions, action)
    all_accuracy.append(accuracy.item())

    action_loss = criterion(predictions, action)
    clip_grad_norm_(parameters, max_norm=3)

    if use_generative_language:
      targets = instructions[:, 1:]
      lstm_loss = metrics.add(lstm_predictions, targets, decode_lengths, alphas)
      clip_grad_norm_(lstm_parameters, max_norm=3)

      lstm_optimizer.zero_grad()
      total_loss = action_loss + lstm_loss
    else:
      total_loss = action_loss

    optimizer.zero_grad()
    total_loss.backward()

    optimizer.step()
    if use_generative_language:
      lstm_optimizer.step()

    all_losses.append(action_loss.item())

    if use_generative_language:
      train_info = (
                    'Epoch: %d, train action loss: %.3f, train lang loss: '
                    '%.3f, train action acc: %.3f, train bleu: %.3f, train '
                    'token acc: %.3f') \
          % (epoch
              , action_loss.item()
              , lstm_loss.item()
              , accuracy
              , metrics.get_mean(metrics.running_bleu_scores)
              , metrics.get_mean(metrics.running_token_accuracy)
              )
    else:
      train_info = (
                    'Epoch: %d, train action loss: %.3f train action acc: %.3f') \
          % (epoch
              , action_loss.item()
              , accuracy
              )
    t.set_description(train_info, refresh=True)
    metrics.flush(epoch, i)

  action_loss = np.array(all_losses).mean()
  accuracy = np.array(all_accuracy).mean()

  if use_generative_language:
    lang_loss = metrics.get_mean(metrics.all_losses)
    bleu = metrics.get_mean(metrics.all_bleu_scores)
    tk_acc = metrics.get_mean(metrics.all_token_accuracy)
  else:
    lang_loss = None
    bleu = None
    tk_acc = None

  if summary_writer is not None:
    summary_writer.add_scalar('ActionLoss/train', action_loss, epoch + 1)
    summary_writer.add_scalar('Accuracy/train', accuracy, epoch + 1)

    if use_generative_language:
      summary_writer.add_scalar('LangLoss/train', lang_loss, epoch + 1)
      summary_writer.add_scalar('Bleu/train', bleu, epoch + 1)
      summary_writer.add_scalar('TokenAcc/train', tk_acc, epoch + 1)

  return action_loss, lang_loss, accuracy, bleu, tk_acc


def validate(device,
             epoch,
             val_loader,
             il_model,
             lstm_model,
             il_criterion,
             lstm_criterion,
             vocab,
             use_generative_language=True,
             summary_writer=None):
  all_losses = []
  all_accuracy = []
  metrics = InstructionsGeneratorMetrics(vocab, lstm_criterion)

  il_model.eval()
  lstm_model.eval()

  t = tqdm(val_loader)

  for idx, data in enumerate(t):
    grid_onehot, grid_embedding, inventory_embedding, action, goal_embedding, instructions, lengths = data

    grid_onehot = grid_onehot.to(device)
    grid_embedding = grid_embedding.to(device)
    goal_embedding = goal_embedding.to(device)
    inventory_embedding = inventory_embedding.to(device)

    instructions = instructions.to(device)
    action = action.to(device, dtype=torch.int64)
    action = action.squeeze(1)

    with torch.no_grad():
      if use_generative_language:
        lstm_predictions, decode_lengths, alphas, lstm_hiddens = lstm_model(
            grid_embedding,
            grid_onehot,
            inventory_embedding,
            goal_embedding,
            instructions,
            lengths,
            use_teacher_forcing=False)
        targets = instructions[:, 1:]
        lstm_loss = metrics.add(lstm_predictions, targets, decode_lengths,
                                alphas)
      else:
        lstm_hiddens = None

      predictions = il_model(
          grid_embedding,
          grid_onehot,
          inventory_embedding,
          goal_embedding,
          lstm_hiddens=lstm_hiddens)

      accuracy = calculate_accuracy(predictions, action)
      all_accuracy.append(accuracy.item())
      action_loss = il_criterion(predictions, action)

    if use_generative_language:
      train_info = (
                    'Epoch: %d, valid action loss: %.3f, valid lang loss: '
                    '%.3f, valid action acc: %.3f, valid bleu: %.3f, valid '
                    'token acc: %.3f') \
          % (epoch
              , action_loss.item()
              , lstm_loss.item()
              , accuracy
              , metrics.get_mean(metrics.running_bleu_scores)
              , metrics.get_mean(metrics.running_token_accuracy)
              )
    else:
      train_info = (
                    'Epoch: %d, valid action loss: %.3f valid action acc: %.3f') \
          % (epoch
              , action_loss.item()
              , accuracy
              )
    t.set_description(train_info, refresh=True)
    metrics.flush(epoch, idx)

    all_losses.append(action_loss.item())

  action_loss = np.array(all_losses).mean()
  accuracy = np.array(all_accuracy).mean()

  if use_generative_language:
    lang_loss = metrics.get_mean(metrics.all_losses)
    bleu = metrics.get_mean(metrics.all_bleu_scores)
    tk_acc = metrics.get_mean(metrics.all_token_accuracy)
  else:
    lang_loss = None
    bleu = None
    tk_acc = None

  if summary_writer is not None:
    summary_writer.add_scalar('ActionLoss/valid', action_loss, epoch + 1)
    summary_writer.add_scalar('Accuracy/valid', accuracy, epoch + 1)

    if use_generative_language:
      summary_writer.add_scalar('LangLoss/valid', lang_loss, epoch + 1)
      summary_writer.add_scalar('Bleu/valid', bleu, epoch + 1)
      summary_writer.add_scalar('TokenAcc/valid', tk_acc, epoch + 1)

  return action_loss, lang_loss, accuracy, bleu, tk_acc
