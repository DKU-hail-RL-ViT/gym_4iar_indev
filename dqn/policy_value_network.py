import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import os
import random
import numpy as np

from collections import deque

parser = argparse.ArgumentParser()

parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()



def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminated = zip(*sample)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        terminated = torch.tensor(terminated, dtype=torch.float32)

        return states, actions, rewards, next_states, terminated

    def size(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, board_width, board_height, state_dim, action_dim,
                 lr=0.005, eps_decay=0.995, eps_min=0.01):
        super(DQN, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = 64
        self.lr = lr
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.epsilon = 1.0

        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * self.board_width * self.board_height, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, env, board_width, board_height,
                 model_file=None, use_gpu=False):

        self.env = env
        self.use_gpu = use_gpu
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape
        self.board_width = board_width  # 9
        self.board_height = board_height  # 4
        self.l2_const = 1e-4  # coef of l2 penalty

        # CNN model
        if self.use_gpu:
            self.model = DQN(self.board_width, self.board_height,
                             self.state_dim, self.action_dim).cuda()
            self.target_model = DQN(self.board_width, self.board_height,
                                    self.state_dim, self.action_dim).cuda()
        else:
            self.model = DQN(self.board_width, self.board_height,
                             self.state_dim, self.action_dim)
            self.target_model = DQN(self.board_width, self.board_height,
                                    self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.l2_const)  # default 5e-3

        if model_file:
            state_dict = torch.load(model_file)
            self.model.load_state_dict(state_dict)
            # self.target_model.load_state_dict(state_dict) # TODO 여긴 주석처리를 풀어야할지 아직 고민

        self.target_update()  # TODO 여기에 target update가 있는게 맞나
        self.buffer = ReplayBuffer()

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        if len(self.buffer) < args.batch_size:
            return
        states, actions, rewards, next_states,  terminated = self.buffer.sample(args.batch_size)

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



class PolicyValueNet:
    """policy-value network """
    def __init__(self, env, board_width, board_height,
                 model_file=None, use_gpu=False):

        self.action_dim = env.action_space.n  # TODO action_dim 여기도 수정해야함
        self.state_dim = env.state_
        self.use_gpu = use_gpu
        self.board_width = board_width  # 9
        self.board_height = board_height  # 4
        self.l2_const = 1e-4  # coef of l2 penalty
        self.buffer = ReplayBuffer()

        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = DQN(board_width, board_height,
                                        env.state_, env.action_space.n).cuda()
        else:
            self.policy_value_net = DQN(board_width, board_height,
                                            env.state_, env.action_space.n)
        self.target_policy_value_net = DQN(board_width, board_height,
                                               self.state_dim, self.action_dim)
        self.target_update()
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            state_dict = torch.load(model_file)
            self.policy_value_net.load_state_dict(state_dict)


    def target_update(self):
        self.target_model.load_state_dict(self.policy_value_net.state_dict())


    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch_np = np.array(state_batch)
            state_batch = Variable(torch.FloatTensor(state_batch_np).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch_np = np.array(state_batch)
            state_batch = torch.FloatTensor(state_batch_np)
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.detach().cpu().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 5, self.board_width, self.board_height))

        log_act_probs, value = self.policy_value_net(
            Variable(torch.from_numpy(current_state)).float())

        act_probs = np.exp(log_act_probs.data.detach().numpy().flatten())
        act_probs = list(zip(legal_positions, act_probs[legal_positions]))
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch_np = np.array(state_batch)
            mcts_probs_np = np.array(mcts_probs)
            winner_batch_np = np.array(winner_batch)

            state_batch = Variable(torch.FloatTensor(state_batch_np).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs_np).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch_np).cuda())

        else:
            state_batch_np = np.array(state_batch)
            mcts_probs_np = np.array(mcts_probs)
            winner_batch_np = np.array(winner_batch)

            state_batch = Variable(torch.FloatTensor(state_batch_np))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs_np))
            winner_batch = Variable(torch.FloatTensor(winner_batch_np))


        # Q 값 예측
        q_values = self.policy_value_net(states)
        next_q_values = self.target_policy_value_net(next_states)

        # 선택된 액션에 대한 Q 값
        q_value = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + args.gamma * next_q_value * (1 - terminated)

        # 손실 계산 및 최적화
        loss = F.mse_loss(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()

        # backward and optimize
        loss.backward()
        self.optimizer.step()

        return loss.item()



    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params


    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        # Ensure that the directory exists before saving the file
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        torch.save(net_params, model_file)
