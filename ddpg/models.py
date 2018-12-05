from torch import nn
import torch
import torch.nn.functional as F

from functools import reduce

def get_flattened_dim(shape):
    assert isinstance(shape, tuple)
    return reduce(lambda x, y: x*y, shape)

class Actor(nn.Module):
    def __init__(self, nactions=1, observation_shape=None):
        """

        :param nactions: action dimension
        """
        super(Actor, self).__init__()
        self.nactions = nactions
        self.fc1 = nn.Linear(get_flattened_dim(observation_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, nactions)

    def forward(self, obs):
        output = self.fc1(obs)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        output = F.tanh(output)
        return output


class Critic(nn.Module):
    def __init__(self, observation_shape=None):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(get_flattened_dim(observation_shape), 64)
        self.fc2 = nn.Linear(65, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs, actions):
        output = self.fc1(obs)
        # actions_1 = actions.unsqueeze()
        assert actions.size(-1) > 0
        output = torch.cat([output, actions], dim=-1)
        assert output.size(-1) == 65
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        return output
