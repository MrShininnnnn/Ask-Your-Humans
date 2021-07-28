from state_encoder import StateEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class InstructionModel(nn.Module):

    def __init__(self,
                device,
                embedding_dim,
                encoder_dim=128,
                grid_onehot_size=7,
                dropout=0.5):
        super(InstructionModel, self).__init__()

        self.device = device

        #TO DO: xingnan's StateEncoder has different dim with lei's StateEncoderA 
        #Is that wrong or 
        #self.state_encoder = StateEncoder(embedding_dim, encoder_dim, grid_onehot_size)
        self.state_encoder = StateEncoderA(embedding_dim, encoder_dim, grid_onehot_size)

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(48, 48)  # Why 48?
        self.fc2 = nn.Linear(48, 9)


    def forward(self, grid_embedding, grid_onehot, inventory, goal):

        #encode features
        state_encod = self.state_encoder(grid_embedding, grid_onehot, inventory, goal)
        #state_encod = self.fc(self.dropout(state_encod))
        #print(state_encod.shape)
        # torch.sum(state_encod, dim = 1)
        # print(state_encod.shape)
        out = F.relu(self.fc(state_encod))
        out = self.fc2(out)

        return out


class StateEncoderA(nn.Module):
    '''
    Based on Figure 12 and original code in train_bc's model.
    '''
    def __init__(self, embed_dim, encoder_dim, grid_onehot_size):
        super(StateEncoderA, self).__init__()

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
        self.fc_all = nn.Linear(512+50, 48)

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
