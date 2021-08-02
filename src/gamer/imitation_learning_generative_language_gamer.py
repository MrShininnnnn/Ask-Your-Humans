#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'

# public
import torch
import numpy as np
# private
from . import game_env
from ..utils import dataloader
from ..utils import helpers


def play(lstm_model, il_model, glove, config):

  count = 0
  game = game_env.generate_new_game(config)
  state = game.observe()['observation'][0]

  last_inv_size = 0

  while not game.is_over() and count < 250:

    count = count + 1
    state = game.observe()['observation'][0]
    goal = game.game.goal
    inventory = game.game.inventory

    if len(inventory) != last_inv_size:
      last_inv_size = len(inventory)

    states_embedding = torch.from_numpy(
        np.array(
            [dataloader.get_grid_embedding(state, glove,
                                           config.embedding_dim)]))
    states_onehot = torch.from_numpy(
        np.array([dataloader.one_hot_grid(state, glove, config.embedding_dim)]))
    goal = torch.from_numpy(
        dataloader.get_goal_embedding(goal, glove, config.embedding_dim))
    inventory = torch.Tensor(
        dataloader.get_inventory_embedding(inventory, glove,
                                           config.embedding_dim))

    states_onehot = states_onehot.to(config.device)
    states_embedding = states_embedding.to(config.device)
    goal = goal.to(config.device)
    inventory = inventory.to(config.device)

    inventory = inventory.view(1, config.embedding_dim)

    _, _, lstm_hiddens = lstm_model(
        states_embedding,
        states_onehot,
        inventory,
        goal,
        use_teacher_forcing=False)

    outputs = il_model(states_embedding, states_onehot, inventory, goal,
                       lstm_hiddens)

    _, indices = outputs[0].max(0)

    action = helpers.get_action_name(indices.item())
    game.act(action)

  return game.is_over()
