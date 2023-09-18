from copy import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DQNBase(nn.Module):

    def __init__(self, batch_size=32, num_channels=5, num_actions=37, embedding_dim=5*9*4):   # channels x width x height
        super(DQNBase, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, 64)   # here
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, embedding_dim)

        self.embedding_dim = embedding_dim
        self.num_action = num_actions
        self.num_channels = num_channels
        self.batch_size = batch_size

    def forward(self, states):
        batch_size = states.shape[0]
        states = states.view(batch_size, -1)
        states = F.relu(self.fc1(states))
        states = F.relu(self.fc2(states))

        # Calculate embeddings of states.
        state_embedding = F.relu(self.fc3(states))

        # assert state_embedding.shape == (batch_size, self.embedding_dim)
        return state_embedding


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self.reset()
        self.sample()

    def reset(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.out_features))

    def f(self, x):
        return x.normal_().sign().mul(x.abs().sqrt())

    def sample(self):
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

    def forward(self, x):
        if self.training:
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)
