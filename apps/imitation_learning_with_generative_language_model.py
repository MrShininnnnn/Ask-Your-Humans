""": IL with Generative langugage model"""

from state_encoder import StateEncoder
from instructions_generator_model import InstructionsGeneratorModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImitationLearningWithGenerativeLanguageModel(nn.Module):

  def __init__(self,
               instructions_generator_model,
               embedding_dim,
               vocab,
               encoder_dim=128):
    super(ImitationLearningWithGenerativeLanguageModel, self).__init__()

    self.state_encoder = StateEncoder(embedding_dim, encoder_dim=encoder_dim)
    self.instructions_generator = instructions_generator_model
    self.vocab = vocab

    self.fc_state_encoder = nn.Linear(27 * 128, 48)
    self.fc_state_and_hidden = nn.Linear(48 + 32, 48)
    self.fc_final = nn.Linear(48, 9)

  def forward(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding, instructions, instructions_lengths):
    state = self.state_encoder(grid_embedding, grid_onehot, inventory_embedding,
                               goal_embedding)
    self.instructions_generator.eval()
    _, hiddens = self.instructions_generator.predict(grid_embedding,
                                                     grid_onehot,
                                                     inventory_embedding,
                                                     goal_embedding, self.vocab)

    state = state.view(-1, state.size(1) * state.size(2))
    state = F.relu(self.fc_state_encoder(state))

    state_and_hidden = torch.cat((state, hiddens), dim=1)
    out = self.fc_state_and_hidden(state_and_hidden)
    out = self.fc_final(out)
    return out
