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
class ImitationLearningGenerativeLanguageModel(nn.Module):
  """
     Imitation learning model with generative language, after encoding the
     state, concanate hidden state from generative language model, then pass it
     to the same fully connected layers as vanilla IL. LSTM model hidden state
     is passed separately (not included in this model), because we need to train
     LSTM separately.
  """

  def __init__(self, config):
    super(ImitationLearningGenerativeLanguageModel, self).__init__()
    self.config = config
    self.state_encoder = ILStateEncoder(config)
    # projection layers
    self.fc_output1 = nn.Linear(
        config.all_hidden_size + config.lstm_hidden_size,
        config.all_hidden_size)
    self.fc_output2 = nn.Linear(config.all_hidden_size, config.action_size)

  def forward(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding, lstm_hidden_state):
    # Encode states
    x = self.state_encoder(grid_embedding, grid_onehot, inventory_embedding,
                           goal_embedding)
    # Combine encoded state with lstm hidden state
    x = torch.cat((x, lstm_hidden_state), dim=1)
    # concate everything
    # x - [batch_size, all_hidden_size]
    x = F.relu(self.fc_output1(x))
    # map hidden state to action space
    # y - [batch_size, action_size]
    y = self.fc_output2(x)

    return y
