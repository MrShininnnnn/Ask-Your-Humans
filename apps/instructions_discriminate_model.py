from state_encoder import StateEncoder
from instruction_model import StateEncoderA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
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

    self.encoder = StateEncoderB(embedding_dim, encoder_dim, grid_onehot_size)
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    
    #<<<<<<<<<<<<<<<<<<<<<<Method1: Based on original function
    self.embed = nn.Embedding(vocab_size, embedding_dim)
    self.encoding = nn.LSTM(embedding_dim, 512, num_layers=1)
    #>>>>>>>>>>>>>>>>>>>>>>

    #<<<<<<<<<<<<<<<<<<<<<<Method2: Based on paper LanguageNet
    self.lstm_encoder = LSTMInstructionEncoder(vocab_size, embedding_dim, dropout, 48)
    #>>>>>>>>>>>>>>>>>>>>>>


    # self.fc = nn.Linear(encoder_dim, 48)  # Why 48?
    # self.fc2 = nn.Linear(48, 9)
    self.fc = nn.Linear(512, 48) 
    self.fc2 = nn.Linear(48, 9)


  def forward(self, grid_embedding, grid_onehot, inventory_embedding,
              goal_embedding, instructions, lengths):

    state_encod = self.encoder(grid_embedding, grid_onehot, inventory_embedding,
                               goal_embedding)
    #state_encod = self.fc(self.dropout(state_encod))

    ###-------------------------Method1 

    #print(state_encod.shape)
    #print(state_encod.unsqueeze(1).shape)
    embeddings = self.embed(instructions)
    #print(embeddings.shape)
    embeddings = torch.cat((state_encod.unsqueeze(1), embeddings), 1)
    #print(embeddings.shape)
    packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
    hiddens, hdn = self.encoding(packed)
    out = F.relu(self.fc(hdn[0]))
    #print(out.shape)

    ###-------------------------

    ###-------------------------Method2 
    #hidden = self.lstm_encoder(instructions, [500])
    #out = F.relu(self.fc(hidden))
    ###-------------------------

    out = self.fc2(out)
    out = out.squeeze()
    return out


class LSTMInstructionEncoder(nn.Module):
    '''
    Based on the paper, they applied similar method used here.
    https://github.com/facebookresearch/minirts/blob/master/scripts/behavior_clone/instruction_encoder.py
    TO DO: The current method is not working. Modification is needed.
    '''
    def __init__(self, dict_size, emb_size, emb_dropout, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.emb = nn.Embedding(dict_size, emb_size)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn = nn.LSTM(emb_size, out_dim, batch_first=True)

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


class StateEncoderB(nn.Module):
    '''
    Based on Figure 12 and original code in train_bc's model.
    TO DO: this method is similar to StateEncoderA except dim. Do we need to make them same?
    '''
    def __init__(self, embed_dim, encoder_dim, grid_onehot_size):
        super(StateEncoderB, self).__init__()

        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.grid_onehot_size = grid_onehot_size

        # TO DO: How the dimention is decided?

        self.fc_emb = nn.Linear(embed_dim, 150)
        self.fc_onehot = nn.Linear(7, 20)
        self.fc_goal = nn.Linear(embed_dim, 150)
        self.fc_inventory = nn.Linear(embed_dim, 50)

        self.fc_emb_hot = nn.Linear(170, 90)
        self.fc_goal_emb_hot = nn.Linear(2250+150, 512)
        self.fc_all = nn.Linear(512+50, 300)

    def forward(self, grid_embedding, grid_onehot, inventory, goal):

        inv = F.relu(self.fc_inventory(inventory))
        goal = F.relu(self.fc_goal(goal))
        emb = F.relu(self.fc_emb(grid_embedding))
        onehot = F.relu(self.fc_onehot(grid_onehot))

        emb = emb.view(-1, 25,150)
        onehot = onehot.view(-1, 25,20)
          
        emb_onehot = torch.cat((emb, onehot), dim = 2) # Why dim = 2 
        emb_onehot =F.relu(self.fc_emb_hot(emb_onehot))
        emb_onehot = emb_onehot.view(-1, 25*90)

        goal_emb_onehot = torch.cat((emb_onehot, goal), dim = 1)
        goal_emb_onehot = self.fc_goal_emb_hot(goal_emb_onehot)

        inv_goal_emb_onehot = torch.cat((goal_emb_onehot, inv), dim = 1)
        
        state_encod = F.relu(self.fc_all(inv_goal_emb_onehot))

        return state_encod   