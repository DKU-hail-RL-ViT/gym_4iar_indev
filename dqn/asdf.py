import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import random
import gym

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
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*sample))

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


class ActionStateCNNModel(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(ActionStateCNNModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps
        self.eps_min = args.eps_min
        self.eps_decay = args.eps_decay
        self.lr = lr

        # Define the CNN architecture
        self.conv1 = nn.Conv2d(in_channels=state_dim[0], out_channels=64, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding='same')
        self.fc1 = nn.Linear(in_features=128 * np.prod(state_dim[1:3]), out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32)  # Add batch dimension
        with torch.no_grad():
            q_values = self.forward(state)
        return q_values.numpy()

    def get_action(self, state):
        state = np.reshape(state, [1] + list(self.state_dim))
        self.epsilon *= self.eps_decay
        self.epsilon = max(self.epsilon, self.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def train(self, states, targets):
        states = torch.tensor(states, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        self.optimizer.zero_grad()
        outputs = self.forward(states)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()



class Agent:
    def __init__(self, env):
        self.env = env
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape

        self.model = ActionStateCNNModel(self.state_dim, self.action_dim, lr=0.005)
        self.target_model = ActionStateCNNModel(self.state_dim, self.action_dim, lr=0.005)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, batch_size, gamma):
        if self.buffer.size() < batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(current_q_values, target_q_values)

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

    def train(self, max_episodes=1000, render_episodes=100, batch_size=32, gamma=0.99):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:

                state = np.array(state[0])
                if len(state.shape) == 3:  # 상태가 이미 (210, 160, 3) 형태인 경우
                    state = state.transpose((2, 0, 1))  # (210, 160, 3) -> (3, 210, 160)
                else:  # 상태가 (210, 160) 형태인 경우
                    state = state.reshape((1, 210, 160))

                    # state = torch.FloatTensor(state[0])  # Convert to tensor
                action = self.model.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                next_state = np.array(next_state)
                if len(next_state.shape) == 3:  # 다음 상태가 이미 (210, 160, 3) 형태인 경우
                    next_state = next_state.transpose((2, 0, 1))  # (210, 160, 3) -> (3, 210, 160)
                else:  # 다음 상태가 (210, 160) 형태인 경우
                    next_state = next_state.reshape((1, 210, 160))

                self.buffer.put(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

            if self.buffer.size() >= batch_size:
                self.replay(batch_size, gamma)
            self.target_update()

            print(f'EP{ep} EpisodeReward={total_reward}')
            if (ep + 1) % render_episodes == 0:
                self.evaluate()

    def evaluate(self):
        done, total_reward = False, 0
        state = self.env.reset()
        while not done:
            state = torch.FloatTensor(state[0]) # Convert to tensor for PyTorch
            action = self.model.get_action(state, explore=False)  # Ensure explore is False for evaluation
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state
        print(f'Evaluation Reward: {total_reward}')


def main():
    env = gym.make("ALE/Pong-v5", render_mode='rgb_array')
    agent = Agent(env)
    agent.train(max_episodes=1000)


if __name__ == "__main__":
    main()