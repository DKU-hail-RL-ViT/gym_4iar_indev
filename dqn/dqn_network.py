import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import gym
import argparse
import numpy as np
from collections import deque
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionStateModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActionStateModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps

        num_features = state_dim[0] * state_dim[1] * state_dim[2]

        self.fc1 = nn.Linear(num_features, 100)
        self.fc2 = nn.Linear(100, self.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).view(1, -1)
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_values = self(state).detach().numpy().flatten()
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_values)

    def train(self, states, targets):
        states = torch.tensor(states, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        optimizer.zero_grad()
        predictions = self(states)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()


class ActionStateCNNModel(ActionStateModel):
    def __init__(self, state_dim, action_dim):
        super(ActionStateCNNModel, self).__init__(state_dim, action_dim)
        self.conv1 = nn.Conv2d(state_dim[0], 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.fc3 = nn.Linear(128 * 50, 100)
        self.fc4 = nn.Linear(100, 16)

    def forward(self, x):
        x = x.view(-1, *self.state_dim)
        print(self.state_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc2(x)
        return x


class Agent:
    def __init__(self, env):
        self.env = env
        self.action_dim = self.env.action_space.n

        # DNN case
        # self.state_dim = np.prod(self.env.observation_space.shape)
        # self.model = ActionStateModel(self.state_dim, self.action_dim)
        # self.target_model = ActionStateModel(self.state_dim, self.action_dim)

        print(self.env.observation_space.shape)

        # CNN case
        self.state_dim = self.env.observation_space.shape
        self.model = ActionStateCNNModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateCNNModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model(torch.tensor(states, dtype=torch.float32)).detach().numpy()
            next_q_values = self.target_model(torch.tensor(next_states, dtype=torch.float32)).max(dim=1).values
            targets[np.arange(args.batch_size), actions] = rewards + (1 - done) * next_q_values.numpy() * args.gamma
            self.model.train(states, targets)

    def train(self, max_episodes=1000, render_episodes=100):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state, _ = self.env.reset()

            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                self.buffer.put(state, action, reward * 0.01, next_state, done)
                total_reward += reward
                state = next_state
            if self.buffer.size() >= args.batch_size:
                self.replay()
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            if (ep + 1) % render_episodes == 0:
                state, _ = self.env.reset()
                while not done:
                    self.env.render()
                    action = self.model.get_action(state)
                    next_state, reward, done, _, _ = self.env.step(action)


def main():
    env = gym.make("Pong-v4")
    agent = Agent(env)
    agent.train(max_episodes=1000, render_episodes=100)


if __name__ == "__main__":
    main()
