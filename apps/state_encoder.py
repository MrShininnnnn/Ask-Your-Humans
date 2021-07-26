""" Encode the state of the game including grid, inventory and goals."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):

  def __init__(self, embedding_dim, encoder_dim=128, grid_onehot_size=7):
    super(StateEncoder, self).__init__()

    self.encoder_dim = encoder_dim
    self.embedding_dim = embedding_dim

    self.fc_embed = nn.Linear(embedding_dim, encoder_dim)
    self.fc_onehot = nn.Linear(grid_onehot_size, encoder_dim)
    self.fc_inventory = nn.Linear(embedding_dim, encoder_dim)
    self.fc_goal = nn.Linear(embedding_dim, encoder_dim)

  def forward(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding):
    grid_fc = F.relu(self.fc_embed(grid_embedding)) + F.relu(
        self.fc_onehot(grid_onehot))
    grid_fc = grid_fc.view(-1, 25,
                           self.encoder_dim)  # TODO(xingnanzhou): What is 25?

    inventory_embedding = inventory_embedding.view(-1, 1, 300)
    inventory_fc = F.relu(self.fc_inventory(inventory_embedding))
    goal_fc = F.relu(self.fc_goal(goal_embedding)).unsqueeze(1)
    encoder_out = torch.cat((grid_fc, inventory_fc, goal_fc),
                            dim=1)  # (batch, 25+10+1, encoder_dim)
    return encoder_out
