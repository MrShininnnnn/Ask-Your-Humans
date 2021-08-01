#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'


""" Encode the state of the game including grid, inventory and goals."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMStateEncoder(nn.Module):

  def __init__(self, embedding_dim, grid_onehot_size=7, encoder_dim=128):
    super(LSTMStateEncoder, self).__init__()

    self.encoder_dim = encoder_dim
    self.embedding_dim = embedding_dim

    self.fc_grid_emb = nn.Linear(embedding_dim, encoder_dim)
    self.fc_grid_onehot = nn.Linear(grid_onehot_size, encoder_dim)
    self.fc_goal = nn.Linear(embedding_dim, encoder_dim)
    self.fc_inventory = nn.Linear(embedding_dim, encoder_dim)

  def forward(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding):
    grid_emb_fc = F.relu(self.fc_grid_emb(grid_embedding))  # (64, 5, 5, 128)
    grid_onehot_fc = F.relu(self.fc_grid_onehot(grid_onehot))  # (64, 5, 5, 128)
    grid_fc = grid_emb_fc + grid_onehot_fc
    grid_fc = grid_fc.view(-1,
                           grid_fc.size(1) * grid_fc.size(2),
                           self.encoder_dim)  # (64, 25, 128)

    inventory_embedding = inventory_embedding.view(-1, 1, 300)
    inventory_fc = F.relu(
        self.fc_inventory(inventory_embedding))  # (64, 1, 128)
    goal_fc = F.relu(self.fc_goal(goal_embedding))
    goal_fc = goal_fc.unsqueeze(1)  # (64, 1, 128)
    encoder_out = torch.cat((grid_fc, inventory_fc, goal_fc),
                            dim=1)  # (batch, 25+10+1, encoder_dim)
    return encoder_out
