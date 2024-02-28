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
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        # 아 여기서 state를 뽑을 때 잘줘야할 수도 있음
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminated = zip(*sample)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        terminated = torch.tensor( terminated, dtype=torch.float32)

        return states, actions, rewards, next_states, terminated

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
    def __init__(self, state_dim, action_dim, lr=0.005, eps_decay=0.995, eps_min=0.01):
        super(CNN_DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = 64
        self.lr = lr
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.epsilon = 1.0

        self.board_width = state_dim[1]
        self.board_height = state_dim[2]

        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.batch_size * self.board_width * self.board_height, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.forward(state)  # Todo 여기 잘 된거지 모르겠음 network부분
        return q_values.numpy()

    def get_action(self, state):

        available = np.where(state[3].flatten() == 0)[0]
        sensible_moves = available

        state = np.reshape(state, (1, ) + self.state_dim)   # state.shape (1, 5, 9, 4)
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            # return np.random.choice(self.action_dim)
            return np.random.choice(sensible_moves)
        return np.argmax(q_value)


class Agent:
    def __init__(self, env):
        self.env = env
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape

        # CNN model
        self.model = CNN_DQN(self.state_dim, self.action_dim)
        self.target_model = CNN_DQN(self.state_dim, self.action_dim)
        self.target_update()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.buffer = ReplayBuffer()

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        if len(self.buffer) < args.batch_size:
            return
        states, actions, rewards, next_states,  terminated = self.buffer.sample()

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        terminated = torch.tensor(terminated, dtype=torch.float32)

        # Q 값 예측
        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        # 선택된 액션에 대한 Q 값
        q_value = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + args.gamma * next_q_value * (1 - terminated)

        # 손실 계산 및 최적화
        loss = F.mse_loss(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, max_episodes=1000):
        for episode in range(max_episodes):
            state = self.env.state_
            total_reward = 0
            done = False

            while not done:
                action = self.model.get_action(state)   # 이걸 그냥 리턴 하면 안되고
                print(action) # get_action안에서 지워야할거 같음
                next_state, reward,  terminated, info = self.env.step(action)

                # when drawn, black defeat then, give -1
                if not reward[0]:
                    reward = (reward[0], 0)
                elif reward[0] and reward[1] == -0.5:
                    reward = (reward[0], -1)

                # dqn 버퍼에(5,9,4)를 넣는게 맞나 (9,4)를 넣는게 맞나
                self.buffer.put(state, action, reward[1], next_state,  terminated)

                state = next_state
                total_reward += reward[1]

                if terminated:
                    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
                    self.env.reset()

            if len(self.buffer) > args.batch_size:
                self.replay()

            self.target_update()
