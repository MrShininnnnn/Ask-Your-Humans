#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'

# public
import torch
import torch.nn as nn
import torch.nn.functional as F
# private
from .il_state_encoder import ILStateEncoder


# follow the structure shown in Figure 12
class ImitationLearningModel(nn.Module):
  """Vanilla Imitation learning model, after encoding the state, use fully connected layer to predict action."""

  def __init__(self, config):
    super(ImitationLearningModel, self).__init__()
    self.config = config
    self.state_encoder = ILStateEncoder(config)
    # projection layers
    self.fc_output1 = nn.Linear(config.all_hidden_size, config.all_hidden_size)
    self.fc_output2 = nn.Linear(config.all_hidden_size, config.action_size)

  def forward(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding):
    # Encode states
    x = self.state_encoder(grid_embedding, grid_onehot, inventory_embedding,
                           goal_embedding)
    # concate everything
    # x - [batch_size, all_hidden_size]
    x = F.relu(self.fc_output1(x))
    # map hidden state to action space
    # y - [batch_size, action_size]
    y = self.fc_output2(x)

    return y
