"""
  Model that takes the state encoding and generate natural langugage
  instructions.
"""

from state_encoder import StateEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstructionsGeneratorModel(nn.Module):

  def __init__(self,
               vocab_size,
               embedding_dim,
               embed_weights,
               encoder_dim=128,
               decoder_dim=32,
               max_seq_length=20,
               grid_onehot_size=7,
               dropout=0.5):
    super(InstructionsGeneratorModel, self).__init__()

    self.encoder = StateEncoder(embedding_dim, encoder_dim=encoder_dim)
    self.encoder_dim = encoder_dim
    self.decoder_dim = decoder_dim
    self.max_seq_length = max_seq_length
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim

    self.dropout = dropout
    # self.grm = nn.GRU(embedding_dim, decoder_dim)

  def forward(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding):
    encoder_out = self.encoder(grid_embedding, grid_onehot, inventory_embedding,
                               goal_embedding)
    return encoder_out
