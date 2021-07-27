

from state_encoder import StateEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class InstructionsDiscriminateModel(nn.Module):

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
               dropout=0.5,
               training=True):
    super(InstructionsDiscriminateModel, self).__init__()

    self.device = device

    self.encoder = StateEncoder(embedding_dim, encoder_dim, grid_onehot_size)
    self.encoder_dim = encoder_dim
    self.decoder_dim = decoder_dim

    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim


    self.encoding = nn.LSTM(embedding_dim, 512, num_layers=1)

    self.fc = nn.Linear(encoder_dim, 48)  # Why 48?
    self.fc2 = nn.Linear(48, 9)


  def forward(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding, instructions, lengths):

    state_encod = self.encoder(grid_embedding, grid_onehot, inventory_embedding,
                               goal_embedding)
    state_encod = self.fc(self.dropout(state_encod))

    embeddings = self.embed(instructions)
    embeddings = torch.cat((state_encod.unsqueeze(1), embeddings), 1)
    packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
    hiddens, hdn = self.encoding(packed)
    
    out = F.relu(self.fc(hiddens[0]))
    out = self.fc2(out)


    return out


class LSTMInstructionEncoder(nn.Module):
    '''
    Based on the paper, they applied similar method used here.
    https://github.com/facebookresearch/minirts/blob/master/scripts/behavior_clone/instruction_encoder.py
    TO DO: modify this method.
    '''
    def __init__(self, dict_size, emb_size, emb_dropout, out_size, padding_idx):
        super().__init__()
        self.out_dim = out_size
        self.emb = nn.Embedding(dict_size, emb_size, padding_idx=padding_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn = nn.LSTM(emb_size, out_size, batch_first=True)

    def forward(self, x, sizes):
        # assert(x.dim(), 3)
        e = self.emb(x)
        e = self.emb_dropout(e)
        h = torch.zeros(1, x.size(0), self.out_dim).to(x.device)
        c = torch.zeros(1, x.size(0), self.out_dim).to(x.device)
        hs, _ = self.rnn(e, (h, c))
        mask = (sizes > 0).long()
        indexes = (sizes - mask).unsqueeze(1).unsqueeze(2)
        indexes = indexes.expand(indexes.size(0), 1, hs.size(2))
        h = hs.gather(1, indexes).squeeze(1)
        # h: [batch, out_size]
        h = h * mask.float().unsqueeze(1)
        return h
