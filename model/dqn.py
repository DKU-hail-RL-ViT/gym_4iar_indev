import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from gym_4iar.network import DQNBase, CosineEmbeddingNetwork


class DQN(BaseModel):
    def __init__(self, num_channels, in_features=4, num_actions=18):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()

        self.dqn_net = DQNBase(num_channels=num_channels)

        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)