import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import gym
import argparse
import random
from collections import deque


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
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)

        # PyTorch 텐서로 변환
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_DQN(nn.Module):
    def __init__(self, action_dim):
        super(CNN_DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 50 * 50, 100)  # Adjust the input features according to your input dimensions
        self.fc2 = nn.Linear(100, 16)
        self.fc3 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent:
    def __init__(self, env):
        self.env = env
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape

        # CNN model
        self.model = CNN_DQN(self.action_dim)
        self.target_model = CNN_DQN(self.action_dim)
        self.target_update()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.buffer = ReplayBuffer()

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        if len(self.buffer) < args.batch_size:
            return
        states, actions, rewards, next_states, done = self.buffer.sample()

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Q 값 예측
        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        # 선택된 액션에 대한 Q 값
        q_value = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + args.gamma * next_q_value * (1 - done)

        # 손실 계산 및 최적화
        loss = F.mse_loss(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, max_episodes=1000):
        for episode in range(max_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, *state.shape])
            total_reward = 0
            done = False

            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, *next_state.shape])

                self.buffer.put(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done:
                    print(f"Episode: {episode+1}, Total Reward: {total_reward}")
                    break

            if len(self.buffer) > args.batch_size:
                self.replay()

            self.target_update()

def main():
    env = gym.make("Pong-v4", render_mode='rgb_array')
    agent = Agent(env)
    agent.train(max_episodes=1000)


if __name__ == "__main__":
    main()
