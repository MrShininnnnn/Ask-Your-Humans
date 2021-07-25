import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    '''
    Based on Figure 12 and original code.
    '''
    def __init__(self, embed_dim):
        super(StateEncoder, self).__init__()

        self.embed_dim = embed_dim

        # TO DO: How the dimention is decided?

        self.fc_emb = nn.Linear(embed_dim,)
        self.fc_onehot = nn.Linear()
        self.fc_goal = nn.Linear()
        self.fc_inventory = nn.Linear()

        self.fc_emb_hot = nn.Linear()
        self.fc_goal_emb_hot = nn.Linear()
        self.fc_all = nn.Linear()

    def forward(self, grid_embedding, grid_onehot, inventory, goal):

        inv = F.relu(self.fc_inventory(inventory))
        goal = F.relu(self.fc_goal(goal))
        emb = F.rele(self.fc_emb(grid_embedding))
        onehot = F.relu(self.fc_onehot(grid_onehot))
          
        emb_onehot = torch.cat((emb, onehot), dim = 2) # Why dim = 2 
        emb_onehot =self.fc_emb_hot(emb_onehot)

        goal_emb_onehot = torch.cat((goal, emb_onehot), dim = 1)
        goal_emb_onehot = self.fc_goal_emb_hot(goal_emb_onehot)

        inv_goal_emb_onehot = torch.cat((inv, goal_emb_onehot), dim = 1)
        
        state_encod = self.fc_all(inv_goal_emb_onehot)

        return state_encod



class LowLevelModel(nn.Module):

    def __init__(self, embed_dim, method = 'IL'):
        super(LowLevelModel, self).__init__()

        # TO DO: How the dimention is decided?

        self.embed_dim = embed_dim

        self.state_encoder = StateEncoder(embed_dim)

        if method == 'IL_GL':
            # TO DO: Add generative Language
            GenerativeLanguage = 0 
        elif method == 'IL_DL':
            # TO DO:
            DiscriminativeLanguage = 0

        self.fc = nn.Linear()
        self.fc2 = nn.Linear()


    def forward(self, grid_embedding, grid_onehot, inventory, goal):

            #encode features
            state_encod = self.state_encoder(grid_embedding, grid_onehot, inventory, goal)
            
            if self.method == 'IL_GL':
                # TO DO: Add generative Language
                GenerativeLanguage = 0 
            elif self.method == 'IL_DL':
                # TO DO:
                DiscriminativeLanguage = 0
            
            out = F.relu(self.fc(state_encod))
            out = self.fc2(out)

            return out




        
