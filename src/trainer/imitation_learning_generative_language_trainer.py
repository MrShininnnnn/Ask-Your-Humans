#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'

# public
import random
from tqdm import tqdm, trange
import torch
import torchtext.vocab as vocabtorch
from torch.nn.utils import clip_grad_norm_
# private
from ..gamer import imitation_learning_generative_language_gamer
from ..utils.eva import ActionMetrics
from ..utils.eva import InstructionsGeneratorMetrics


def train(epoch,
          data_loader,
          lstm_model,
          lstm_optimizer,
          lstm_criterion,
          il_model,
          il_optimizer,
          il_criterion,
          vocab,
          config,
          summary_writer=None):
  # enable train mode
  il_model.train()
  lstm_model.train()
  # evaluation method
  action_metrics = ActionMetrics(il_criterion)
  lstm_metrics = InstructionsGeneratorMetrics(vocab, lstm_criterion)

  # setup pbar
  t = tqdm(data_loader)
  # start
  for data in t:
    # load batch data
    lengths = data[-1]
    data = (d.to(config.device) for d in data[:-1])
    grid_onehot, grid_embedding, inventory_embedding, action, goal_embedding, instructions = data
    action = action.squeeze(1).type(torch.int64)

    # forward
    use_teacher_forcing = True if random.random() < 0.5 else False
    lstm_predictions, alphas, lstm_hidden_state = lstm_model(
        grid_embedding,
        grid_onehot,
        inventory_embedding,
        goal_embedding,
        instructions=instructions,
        instructions_lengths=lengths,
        use_teacher_forcing=use_teacher_forcing)

    predictions = il_model(grid_embedding, grid_onehot, inventory_embedding,
                           goal_embedding, lstm_hidden_state)

    # backward
    targets = instructions[:, 1:]
    decode_lengths = [lengths - 1 for lengths in lengths]
    lstm_loss = lstm_metrics.add(lstm_predictions, targets, decode_lengths,
                                 alphas)
    clip_grad_norm_(lstm_model.parameters(), config.max_norm)

    action_loss = action_metrics.add(predictions, action)
    clip_grad_norm_(il_model.parameters(), config.max_norm)
    total_loss = lstm_loss + action_loss

    lstm_optimizer.zero_grad()
    il_optimizer.zero_grad()

    total_loss.backward()
    lstm_optimizer.step()
    il_optimizer.step()

    # evaluation
    train_info = (
                    'Epoch: %d, train action loss: %.3f, train lang loss: '
                    '%.3f, train action acc: %.3f, train bleu: %.3f, train '
                    'token acc: %.3f') \
          % (epoch
              , action_loss.item()
              , lstm_loss.item()
              , action_metrics.get_mean(action_metrics.running_accuracy)
              , lstm_metrics.get_mean(lstm_metrics.running_bleu_scores)
              , lstm_metrics.get_mean(lstm_metrics.running_token_accuracy)
              )

    action_metrics.flush(epoch)
    lstm_metrics.flush(epoch)
    t.set_description(train_info, refresh=True)

    if config.dev:
      break

  action_loss = action_metrics.get_mean(action_metrics.all_losses)
  accuracy = action_metrics.get_mean(action_metrics.all_accuracy)
  lstm_loss = lstm_metrics.get_mean(lstm_metrics.all_losses)
  bleu = lstm_metrics.get_mean(lstm_metrics.all_bleu_scores)
  tk_acc = lstm_metrics.get_mean(lstm_metrics.all_token_accuracy)

  if summary_writer is not None:
    summary_writer.add_scalar('ActionLoss/train', action_loss, epoch + 1)
    summary_writer.add_scalar('Accuracy/train', accuracy, epoch + 1)
    summary_writer.add_scalar('LSTMLoss/train', lstm_loss, epoch + 1)
    summary_writer.add_scalar('Bleu/train', bleu, epoch + 1)
    summary_writer.add_scalar('TokenAcc/train', tk_acc, epoch + 1)

  return action_loss, accuracy, lstm_loss, bleu, tk_acc


def validate(epoch,
             data_loader,
             lstm_model,
             lstm_criterion,
             il_model,
             il_criterion,
             vocab,
             config,
             summary_writer=None):
  # enable evaluation mode
  lstm_model.eval()
  il_model.eval()

  # evaluation method
  lstm_metrics = InstructionsGeneratorMetrics(vocab, lstm_criterion)
  action_metrics = ActionMetrics(il_criterion)

  # setup pbar
  t = tqdm(data_loader)

  # start
  for data in t:
    # load batch data
    lengths = data[-1]
    data = (d.to(config.device) for d in data[:-1])
    grid_onehot, grid_embedding, inventory_embedding, action, goal_embedding, instructions = data
    action = action.squeeze(1).type(torch.int64)

    with torch.no_grad():
      # forward
      lstm_predictions, alphas, lstm_hiddens = lstm_model(
          grid_embedding,
          grid_onehot,
          inventory_embedding,
          goal_embedding,
          use_teacher_forcing=False)
      predictions = il_model(grid_embedding, grid_onehot, inventory_embedding,
                             goal_embedding, lstm_hiddens)

    # evaluation
    targets = instructions[:, 1:]
    decode_lengths = [lengths - 1 for lengths in lengths]
    lstm_loss = lstm_metrics.add(lstm_predictions, targets, decode_lengths,
                                 alphas)
    action_loss = action_metrics.add(predictions, action)

    valid_info = (
                    'Epoch: %d, valid action loss: %.3f, valid lang loss: '
                    '%.3f, valid action acc: %.3f, valid bleu: %.3f, valid '
                    'token acc: %.3f') \
          % (epoch
              , action_loss.item()
              , lstm_loss.item()
              , action_metrics.get_mean(action_metrics.running_accuracy)
              , lstm_metrics.get_mean(lstm_metrics.running_bleu_scores)
              , lstm_metrics.get_mean(lstm_metrics.running_token_accuracy)
              )
    action_metrics.flush(epoch, train=False)
    lstm_metrics.flush(epoch, train=False)

    t.set_description(valid_info, refresh=True)
    if config.dev:
      break

  action_loss = action_metrics.get_mean(action_metrics.all_losses)
  accuracy = action_metrics.get_mean(action_metrics.all_accuracy)
  lstm_loss = lstm_metrics.get_mean(lstm_metrics.all_losses)
  bleu = lstm_metrics.get_mean(lstm_metrics.all_bleu_scores)
  tk_acc = lstm_metrics.get_mean(lstm_metrics.all_token_accuracy)

  if summary_writer is not None:
    summary_writer.add_scalar('ActionLoss/valid', action_loss, epoch + 1)
    summary_writer.add_scalar('Accuracy/valid', accuracy, epoch + 1)
    summary_writer.add_scalar('LSTMLoss/valid', lstm_loss, epoch + 1)
    summary_writer.add_scalar('Bleu/valid', bleu, epoch + 1)
    summary_writer.add_scalar('TokenAcc/valid', tk_acc, epoch + 1)

  return action_loss, accuracy, lstm_loss, bleu, tk_acc


def validate_game(epoch,
                  lstm_model,
                  il_model,
                  config,
                  summary_writer=None,
                  cache=None):
  # enable evaluation mode
  lstm_model.eval()
  il_model.eval()

  glove = vocabtorch.GloVe(name='840B', dim=300, cache=cache)
  results = []
  for _ in trange(15):
    with torch.no_grad():
      res = imitation_learning_generative_language_gamer.play(
          lstm_model, il_model, glove, config)
      results.append(res)

  reward = sum(results)
  if summary_writer is not None:
    summary_writer.add_scalar('reward/valid', reward, epoch + 1)

  return reward
