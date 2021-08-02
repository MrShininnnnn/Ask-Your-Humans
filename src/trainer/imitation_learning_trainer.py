#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'


# public
from tqdm import tqdm, trange
import torch
import torchtext.vocab as vocabtorch
from torch.nn.utils import clip_grad_norm_
# private
from ..utils.eva import ActionMetrics
from ..gamer import imitation_learning_gamer


def train(epoch, data_loader, model, optimizer, criterion, config, summary_writer=None):
    # enable train mode
    model.train()
    # evaluation method
    metrics = ActionMetrics(criterion)
    # setup pbar
    t = tqdm(data_loader)
    # start
    for data in t:
        # load batch data
        data = (d.to(config.device) for d in data[:-1])
        grid_onehot, grid_embedding, inventory_embedding, action, goal_embedding, instructions = data
        action = action.squeeze(1).type(torch.int64)
        # forward
        predictions = model(
            grid_embedding,
            grid_onehot,
            inventory_embedding,
            goal_embedding)
        # backward
        loss = metrics.add(predictions, action)
        clip_grad_norm_(model.parameters(), config.max_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # evaluation
        metrics.flush(epoch)
        t.set_description(metrics.train_info, refresh=True)
        if config.dev:
            break
    loss = metrics.get_mean(metrics.all_losses)
    accuracy = metrics.get_mean(metrics.all_accuracy)

    if summary_writer is not None:
        summary_writer.add_scalar('Loss/train', loss, epoch + 1)
        summary_writer.add_scalar('Accuracy/train', accuracy, epoch + 1)

    return loss, accuracy


def validate(epoch, data_loader, model, criterion, config, summary_writer=None):
    # enable evaluation mode
    model.eval()
    # evaluation method
    metrics = ActionMetrics(criterion)
    # setup pbar
    t = tqdm(data_loader)
    # start
    for data in t:
        # load batch data
        data = (d.to(config.device) for d in data[:-1])
        grid_onehot, grid_embedding, inventory_embedding, action, goal_embedding, instructions = data
        action = action.squeeze(1).type(torch.int64)
        with torch.no_grad():
            # forward
            predictions = model(
                grid_embedding,
                grid_onehot,
                inventory_embedding,
                goal_embedding)
        # evaluation
        loss = metrics.add(predictions, action)
        metrics.flush(epoch, train=False)
        t.set_description(metrics.valid_info, refresh=True)
        if config.dev:
            break
    loss = metrics.get_mean(metrics.all_losses)
    accuracy = metrics.get_mean(metrics.all_accuracy)

    if summary_writer is not None:
        summary_writer.add_scalar('Loss/valid', loss, epoch + 1)
        summary_writer.add_scalar('Accuracy/valid', accuracy, epoch + 1)

    return loss, accuracy

def validate_game(epoch, model, config, summary_writer=None, cache=None):
    # enable evaluation mode
    model.eval()
    glove = vocabtorch.GloVe(name='840B', dim=300, cache=cache)
    results = []
    for i in trange(15):
        with torch.no_grad():
            res = imitation_learning_gamer.play(model, glove, config)
            results.append(res)

    reward = sum(results)
    if summary_writer is not None:
        summary_writer.add_scalar('reward/valid', reward, epoch + 1)

    return reward

