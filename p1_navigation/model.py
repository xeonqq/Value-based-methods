import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        base_size=32
        self.fc_bb_1 = nn.Linear(state_size, base_size*16)
        self.fc_bb_2 = nn.Linear(base_size*16, base_size*16)

        self.fc_s_head_1 = nn.Linear(base_size*16, base_size*8)
        self.fc_s_head_2 = nn.Linear(base_size*8, 1)

        self.fc_a_head_1 = nn.Linear(base_size*16, base_size*16)
        self.fc_a_head_2 = nn.Linear(base_size*16, action_size)



    def forward(self, state):
        """Build a network that maps state -> action values."""
        x=F.relu(self.fc_bb_1(state))
        base=F.relu(self.fc_bb_2(x))

        x1 = F.relu(self.fc_s_head_1(base))
        state_value = self.fc_s_head_2(x1)

        x2 = F.relu(self.fc_a_head_1(base))
        action_values = self.fc_a_head_2(x2)
        return action_values.sub_(action_values.mean()).add_(state_value)