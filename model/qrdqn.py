from torch import nn

from gym_4iar.network import DQNBase


class QRDQN(nn.Module):

    def __init__(self, num_channels, num_actions=36, embedding_dim=5*9*4, N_index=0):
        super(QRDQN, self).__init__()

        # Feature extractor of DQN.
        self.dqn_net = DQNBase(num_channels=num_channels, num_actions=num_actions, embedding_dim=embedding_dim)

        self.N = [2, 4, 8, 16, 32, 64]
        if self.N[N_index] == 2:
            k = self.N[N_index - 1]
            N_index += 1

        else:
            k = self.N[N_index - 1]
            N_index += 1

        self.k = k


        # Quantile network.
        self.q_net = nn.Sequential(
            nn.Linear(embedding_dim, 32),  # Correct the linear instantiation
            nn.ReLU(),
            nn.Linear(32, num_actions * self.k),  # Correct the linear instantiation
        )

        self.num_channels = num_channels  # Assign the correct value
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim


    def forward(self, states=None, state_embeddings=None,):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        quantiles = self.q_net(
            state_embeddings).view(batch_size, self.k, self.num_actions)
        print(self.k, "qrdqn 모델에서 forward할때 출력하는 quantile")

        assert quantiles.shape == (batch_size, self.k, self.num_actions)
        return quantiles

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None \
            else state_embeddings.shape[0]

        # Calculate quantiles.
        quantiles = self(states=states, state_embeddings=state_embeddings)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q