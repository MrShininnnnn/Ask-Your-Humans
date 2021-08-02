#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'

# public
import random
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
# private
from ..utils.eva import InstructionsGeneratorMetrics


def train(device,
          epoch,
          train_data_loader,
          model,
          optimizer,
          criterion,
          parameters,
          vocab,
          summary_writer=None):
  metrics = InstructionsGeneratorMetrics(vocab, criterion)
  model.train()
  t = tqdm(train_data_loader)
  for i, data in enumerate(t):
    grid_onehot, grid_embedding, inventory_embedding, _, goal_embedding, instructions, lengths = data

    grid_onehot = grid_onehot.to(device)
    grid_embedding = grid_embedding.to(device)
    goal_embedding = goal_embedding.to(device)
    inventory_embedding = inventory_embedding.to(device)
    instructions = instructions.to(device)

    use_teacher_forcing = True if random.random() < 0.5 else False

    predictions, alphas, _ = model(
        grid_embedding,
        grid_onehot,
        inventory_embedding,
        goal_embedding,
        instructions=instructions,
        instructions_lengths=lengths,
        use_teacher_forcing=use_teacher_forcing)

    targets = instructions[:, 1:]
    decode_lengths = [lengths - 1 for lengths in lengths]
    lang_loss = metrics.add(predictions, targets, decode_lengths, alphas)
    clip_grad_norm_(parameters, max_norm=3)

    optimizer.zero_grad()
    lang_loss.backward()
    optimizer.step()

    metrics.flush(epoch)
    t.set_description(metrics.train_info, refresh=True)

  loss = metrics.get_mean(metrics.all_losses)
  bleu = metrics.get_mean(metrics.all_bleu_scores)
  tk_acc = metrics.get_mean(metrics.all_token_accuracy)

  if summary_writer is not None:
    summary_writer.add_scalar('Loss/train', loss, epoch + 1)
    summary_writer.add_scalar('Bleu/train', bleu, epoch + 1)
    summary_writer.add_scalar('TokenAcc/train', tk_acc, epoch + 1)

  return loss, bleu, tk_acc


def validate(device,
             epoch,
             val_loader,
             model,
             criterion,
             vocab,
             summary_writer=None):
  metrics = InstructionsGeneratorMetrics(vocab, criterion)
  model.eval()
  t = tqdm(val_loader)
  for idx, data in enumerate(t):
    grid_onehot, grid_embedding, inventory_embedding, _, goal_embedding, instructions, lengths = data

    grid_onehot = grid_onehot.to(device)
    grid_embedding = grid_embedding.to(device)
    goal_embedding = goal_embedding.to(device)
    inventory_embedding = inventory_embedding.to(device)
    instructions = instructions.to(device)

    with torch.no_grad():
      predictions, alphas, _ = model(
          grid_embedding,
          grid_onehot,
          inventory_embedding,
          goal_embedding,
          use_teacher_forcing=False)

    decode_lengths = [lengths - 1 for lengths in lengths]
    targets = instructions[:, 1:]
    metrics.add(predictions, targets, decode_lengths, alphas)

    metrics.flush(epoch, train=False)
    t.set_description(metrics.valid_info, refresh=True)

  loss = metrics.get_mean(metrics.all_losses)
  bleu = metrics.get_mean(metrics.all_bleu_scores)
  tk_acc = metrics.get_mean(metrics.all_token_accuracy)

  if summary_writer is not None:
    summary_writer.add_scalar('Loss/valid', loss, epoch + 1)
    summary_writer.add_scalar('Bleu/valid', bleu, epoch + 1)
    summary_writer.add_scalar('TokenAcc/valid', tk_acc, epoch + 1)

  return loss, bleu, tk_acc
