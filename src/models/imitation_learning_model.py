""": IL with Generative langugage model"""

from .state_encoder import StateEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImitationLearningModel(nn.Module):
  """ IL model but supports passing in hidden layer from LSTM. """

  def __init__(self, embedding_dim, encoder_dim=128):
    super(ImitationLearningModel, self).__init__()

    self.state_encoder = StateEncoder(embedding_dim, encoder_dim=encoder_dim)

    self.fc_state_encoder = nn.Linear(27 * 128, 48)
    self.fc_state_and_hidden = nn.Linear(48 + 32, 48)
    self.fc_final = nn.Linear(48, 9)

  def forward(self,
              grid_embedding,
              grid_onehot,
              inventory_embedding,
              goal_embedding,
              lstm_hiddens=None):
    state = self.state_encoder(grid_embedding, grid_onehot, inventory_embedding,
                               goal_embedding)
    state = state.view(-1, state.size(1) * state.size(2))
    state = F.relu(self.fc_state_encoder(state))

    if lstm_hiddens is not None:
      state = torch.cat((state, lstm_hiddens), dim=1)
      state = self.fc_state_and_hidden(state)

    out = self.fc_final(state)
    return out
