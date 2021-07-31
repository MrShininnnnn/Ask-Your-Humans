"""
  Model that takes the state encoding and generate natural langugage
  instructions. When using with IL or RL, take the hidden states from forward
  call, and use hidden to predict actions.
"""

from attention_model import Attention
from state_encoder import StateEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstructionsDecoder(nn.Module):

  def __init__(self,
               device,
               vocab_size,
               embedding_dim,
               embed_weights,
               encoder_dim=128,
               attention_dim=128,
               decoder_dim=32,
               max_seq_length=20,
               grid_onehot_size=7,
               dropout=0.5):
    super(InstructionsDecoder, self).__init__()

    self.device = device

    # self.encoder = StateEncoder(embedding_dim, encoder_dim=encoder_dim)
    self.hidden_size = encoder_dim
    self.output_size = vocab_size
    self.dropout_p = dropout
    self.max_length = max_seq_length

    self.embedding = nn.Embedding(self.output_size, self.hidden_size)

    self.encoder_attn = nn.Linear(self.hidden_size, 128)
    self.decoder_attn = nn.Linear(self.hidden_size, 128)
    self.full_attn = nn.Linear(128, 1)

    # self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    self.dropout = nn.Dropout(p=dropout)
    self.gru = nn.LSTMCell(
        self.hidden_size, self.hidden_size, bias=True)
        # nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
    self.out = nn.Linear(self.hidden_size, self.output_size)

    self.fc_init_hidden = nn.Linear(self.hidden_size, self.hidden_size)
    self.fc_init_cell = nn.Linear(self.hidden_size, self.hidden_size)

  def init_hidden_state(self, encoder_out):
    """Creates the initial hidden and cell states for the decoder."""
    mean_encoder_out = encoder_out.mean(dim=1)
    hidden_state = self.fc_init_hidden(mean_encoder_out)
    cell_state = self.fc_init_cell(mean_encoder_out)
    return hidden_state, cell_state

  def forward(self, x, hidden, cell, encoder_outputs):
    if hidden is None:
      hidden, cell = self.init_hidden_state(encoder_outputs)  # (64, 128)

    embedded = self.embedding(x)  # (64, 1, 128)
    # embedded = self.dropout(embedded)

    encoder_attn = self.encoder_attn(encoder_outputs)  # (64, 27, 128)
    decoder_attn = self.decoder_attn(hidden)  # (64, 128)
    attn = self.full_attn(
        F.relu(encoder_attn + decoder_attn.unsqueeze(1))).squeeze(2)  # (64, 27)

    attn_weights = F.softmax(attn)
    attn_applied = (encoder_outputs * attn_weights.unsqueeze(2)).sum(
        dim=1)  # (64, 128)

    output = torch.cat((embedded[:, 0, :], attn_applied), 1)
    output = self.attn_combine(output).unsqueeze(0)

    output = F.relu(output)
    output, hidden = self.gru(output, (hidden, cell))

    output = F.log_softmax(self.out(output[0]), dim=1)
    return output, hidden, attn_weights
