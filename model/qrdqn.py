from torch import nn

# from gym_4iar.network import DQNBase


class QRDQN(nn.Module):

    def __init__(self, num_channels, num_actions=36, embedding_dim=5*9*4, N=32):
        super(QRDQN, self).__init__()
        linear = nn.Linear

        # Feature extractor of DQN.
        self.dqn_net = DQNBase(num_channels=num_channels, num_actions=num_actions, embedding_dim=embedding_dim)

        # Quantile network.
        self.q_net = nn.Sequential(
            linear(embedding_dim, 32),
            nn.ReLU(),
            linear(32, num_actions * N),
        )

        self.N = N
        self.num_channels = 5
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        quantiles = self.q_net(
            state_embeddings).view(batch_size, self.N, self.num_actions)

        assert quantiles.shape == (batch_size, self.N, self.num_actions)
        return quantiles

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        # Calculate quantiles.
        quantiles = self(states=states, state_embeddings=state_embeddings)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q