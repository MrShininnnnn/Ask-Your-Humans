#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'


"""
  Model that takes the state encoding and generate natural langugage
  instructions. When using with IL or RL, take the hidden states from forward
  call, and use hidden to predict actions.
"""

from .attention_model import Attention
from .lstm_state_encoder import LSTMStateEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstructionsGeneratorModel(nn.Module):

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
    super(InstructionsGeneratorModel, self).__init__()

    self.device = device

    self.encoder = StateEncoder(embedding_dim, encoder_dim=encoder_dim)
    self.encoder_dim = encoder_dim
    self.decoder_dim = decoder_dim
    self.max_seq_length = max_seq_length
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim

    self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

    self.fc_init_hidden = nn.Linear(encoder_dim, decoder_dim)
    self.fc_init_cell = nn.Linear(encoder_dim, decoder_dim)

    # TODO(xingnanzhou): Change to embedding, need to train and save the model again
    self.target_instructions_embedding = nn.Embedding(vocab_size, embedding_dim)
    self.target_instructions_embedding.load_state_dict(
        {'weight': embed_weights})
    if self.training:
      self.target_instructions_embedding.weight.requires_grad = False

    self.fc_beta = nn.Linear(decoder_dim, encoder_dim)
    self.sigmoid = nn.Sigmoid()
    self.decode_step = nn.LSTMCell(
        embedding_dim + encoder_dim, decoder_dim, bias=True)
    self.fc = nn.Linear(decoder_dim, vocab_size)
    self.dropout = nn.Dropout(p=dropout)

  def init_hidden_state(self, encoder_out):
    """Creates the initial hidden and cell states for the decoder."""
    mean_encoder_out = encoder_out.mean(dim=1)
    hidden_state = self.fc_init_hidden(mean_encoder_out)
    cell_state = self.fc_init_cell(mean_encoder_out)
    return hidden_state, cell_state

  def forward(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding, instructions, instructions_lengths):
    encoder_out = self.encoder(grid_embedding, grid_onehot, inventory_embedding,
                               goal_embedding)

    batch_size = encoder_out.size(0)

    hidden_state, cell_state = self.init_hidden_state(
        encoder_out)  # (batch_size, max_caption_length, embed_dim)
    target_instructions_embedding = self.target_instructions_embedding(
        instructions)  # (batch_size, decoder_dim)

    # We won't decode at the <end> position, since we've finished generating as
    # soon as we generate <end>. So, decoding lengths are actual lengths - 1
    decode_lengths = [
        instruction_length - 1 for instruction_length in instructions_lengths
    ]

    predictions = torch.zeros(batch_size, max(decode_lengths),
                              self.vocab_size).to(self.device)
    alphas = torch.zeros(batch_size, max(decode_lengths),
                         encoder_out.size(1)).to(self.device)
    hiddens = hidden_state.clone()

    for t in range(max(decode_lengths)):
      batch_size_t = sum([l > t for l in decode_lengths])
      attention_weighted_encoding, alpha = self.attention(
          encoder_out[:batch_size_t], hidden_state[:batch_size_t])

      gate = self.sigmoid(self.fc_beta(hidden_state[:batch_size_t]))
      attention_weighted_encoding = gate * attention_weighted_encoding

      decode_input = torch.cat([
          target_instructions_embedding[:batch_size_t, t, :],
          attention_weighted_encoding
      ],
                               dim=1)
      hidden_state, cell_state = self.decode_step(
          decode_input,
          (hidden_state[:batch_size_t], cell_state[:batch_size_t]))
      hiddens[:batch_size_t] = hidden_state.clone()
      preds = self.fc(self.dropout(hidden_state))
      predictions[:batch_size_t, t, :] = preds
      alphas[:batch_size_t, t, :] = alpha

    return predictions, decode_lengths, alphas, hiddens

  def predict(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding, vocab):
    encoder_out = self.encoder(grid_embedding, grid_onehot, inventory_embedding,
                               goal_embedding)
    batch_size = encoder_out.size(0)

    k_prev_words = torch.LongTensor([[vocab.word2idx['<start>']]] *
                                    batch_size).to(self.device)

    complete_seqs = [[vocab.word2idx['<start>']] for i in range(batch_size)]
    incomplete_inds = [i for i in range(batch_size)]  # TODO: use np.linespace

    step = 1
    hidden_state, cell_state = self.init_hidden_state(encoder_out)
    hiddens = hidden_state.clone()

    while True:
      embeddings = self.target_instructions_embedding(k_prev_words).squeeze(1)
      attention_weighted_encoding, _ = self.attention(encoder_out, hidden_state)
      gate = self.sigmoid(self.fc_beta(hidden_state))
      attention_weighted_encoding = gate * attention_weighted_encoding

      decode_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
      hidden_state, cell_state = self.decode_step(decode_input,
                                                  (hidden_state, cell_state))
      hiddens[incomplete_inds] = hidden_state.clone()
      preds = self.fc(self.dropout(hidden_state))
      preds = F.log_softmax(preds, dim=1)

      _, indices = preds.max(dim=1)

      unfinished_indices = []
      for i in range(indices.size(0) - 1, -1, -1):
        complete_seqs[incomplete_inds[i]].append(indices.data.tolist()[i])
        if indices[i] == vocab.word2idx['<end>']:
          del incomplete_inds[i]
        else:
          unfinished_indices.append(i)

      # All sequences reach <end>
      if len(incomplete_inds) == 0:
        break

      hidden_state = hidden_state[unfinished_indices]
      cell_state = cell_state[unfinished_indices]
      encoder_out = encoder_out[unfinished_indices]
      k_prev_words = indices[unfinished_indices].unsqueeze(1)

      if step > 20:
        break

      step += 1
    return complete_seqs, hiddens
