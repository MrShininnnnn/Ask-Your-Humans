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

        self.state_encoder = StateEncoder(embedding_dim, encoder_dim, grid_onehot_size)

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(encoder_dim, 48)  # Why 48?
        self.fc2 = nn.Linear(48, 9)


    def forward(self, grid_embedding, grid_onehot, inventory, goal):

        #encode features
        state_encod = self.state_encoder(grid_embedding, grid_onehot, inventory, goal)
        #state_encod = self.fc(self.dropout(state_encod))
        out = F.relu(self.fc(state_encod))
        out = self.fc2(out)

        return out


        
