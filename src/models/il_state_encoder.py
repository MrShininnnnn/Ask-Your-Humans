#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'

# public
import torch
import torch.nn as nn
import torch.nn.functional as F


class ILStateEncoder(nn.Module):
  """State encoder used by imitation learning."""

  def __init__(self, config):
    super(ILStateEncoder, self).__init__()
    self.config = config
    # linear layer to encode grid embedding
    self.fc_grid_emb = nn.Linear(config.embedding_dim,
                                 config.grid_embedding_hidden_size)
    # linear layer to encode grid onehot
    self.fc_grid_onehot = nn.Linear(config.state_onehot_size,
                                    config.grid_onehot_hidden_size)
    # linear layer to encode grid
    self.fc_grid = nn.Linear(
        config.grid_embedding_hidden_size + config.grid_onehot_hidden_size,
        config.grid_hidden_size)
    # linear layer to encode goal embedding
    self.fc_goal_emb = nn.Linear(config.embedding_dim,
                                 config.goal_embedding_hidden_size)
    # linear layer to encode grid and goal
    self.fc_grid_goal = nn.Linear(
        config.state_size * config.grid_hidden_size +
        config.goal_embedding_hidden_size, config.grid_goal_hidden_size)
    # linear layer to encode inventory
    self.fc_inventory = nn.Linear(config.embedding_dim,
                                  config.inventory_hidden_size)
    # linear layer to encode everything
    self.fc_all = nn.Linear(
        config.grid_goal_hidden_size + config.inventory_hidden_size,
        config.all_hidden_size)

  def forward(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding):
    # grid_embedding - [batch_size, grid_size, grid_size, embedding_dim]
    # grid_onehot - [batch_size, grid_size, grid_size, state_size]]
    # inventory_embedding - [batch_size, embedding_dim]
    # goal_embedding - [batch_size, embedding_dim]
    x_grid_1 = F.relu(self.fc_grid_emb(grid_embedding))
    # x_grid_1 - [batch_size, state_size, grid_embedding_hidden_size]
    x_grid_1 = x_grid_1.view(-1, self.config.state_size,
                             self.config.grid_embedding_hidden_size)
    # grid onehot hidden state
    x_grid_2 = F.relu(self.fc_grid_onehot(grid_onehot))
    # x_grid_2 - [batch_size, state_size, grid_onehot_hidden_size]
    x_grid_2 = x_grid_2.view(-1, self.config.state_size,
                             self.config.grid_onehot_hidden_size)
    # concat both grid states and feed to a linear layer
    x_grid = F.relu(self.fc_grid(torch.cat((x_grid_1, x_grid_2), dim=2)))
    # x_grid - [batch_size, state_size * grid_hidden_size]
    x_grid = x_grid.view(-1,
                         self.config.state_size * self.config.grid_hidden_size)
    # goal embedding hidden state
    # x_goal - [batch_size, goal_embedding_hidden_size]
    x_goal = F.relu(self.fc_grid_emb(goal_embedding))
    # concate both grid and goal
    # x_grid_goal - [batch_size, grid_goal_hidden_size]
    x_grid_goal = F.relu(self.fc_grid_goal(torch.cat((x_grid, x_goal), dim=1)))
    # inventory hidden state
    # x_inventory - [batch_size, inventory_hidden_size]
    x_inventory = F.relu(self.fc_inventory(inventory_embedding))
    # concate everything
    # x - [batch_size, all_hidden_size]
    y = F.relu(self.fc_all(torch.cat((x_grid_goal, x_inventory), dim=1)))

    return y
