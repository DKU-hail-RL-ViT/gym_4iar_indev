import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os
import random


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def quantile_huber_loss(loss, kappa=1.0):

    loss = torch.where(loss.abs() <= kappa, 0.5 * loss.pow(2), kappa * (loss.abs() - 0.5 * kappa))

    return loss


class DQN(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(DQN, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action value layers (previous state value)
        self.act_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.act_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.act_fc2 = nn.Linear(64, self.num_actions)

    def forward(self, state_input, eps):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 2 * self.board_width * self.board_height)
        x_act = F.relu(self.act_fc1(x_act))

        x_val = self.act_fc2(x_act)
        x_act = F.log_softmax(x_val, dim=1)

        # epsilon greedy
        # if torch.rand(1) < eps:
        #     val_idx = random.randrange(x_val.numel())
        # else:  # (1 - eps) probs
        #     val_idx = x_val.argmax(dim=1)
        # x_val = x_val.view(-1)[val_idx].view(1)

        return x_act, x_val


class QRDQN(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super(QRDQN, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers (previous state value)
        self.act_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.act_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.act_fc2 = nn.Linear(64, self.num_actions)
        # action value
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input, eps):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 2 * self.board_width * self.board_height)
        x_act = F.relu(self.act_fc1(x_act))
        x_act = self.act_fc2(x_act)
        x_acts = self.val_fc2(x_act)
        x_acts = F.log_softmax(x_acts, dim=1)

        # epsilon greedy
        x_acts = torch.ones_like(x_acts) * eps
        x_acts[x_acts.argmax(dim=1)] = 1 - eps * x_acts.shape[1]

        # return action value
        x_val = F.tanh(x_acts)

        return x_acts, x_val


class AC(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super(AC, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)  # output about log probability of each action
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class QRAC(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height, N):
        super(QRAC, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.N = N

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, self.N)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        x_val = torch.mean(x_val, dim=1, keepdim=True)
        return x_act, x_val


class AAC(nn.Module):  # action value actor critic
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super(AAC, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # policy gradient layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # action value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, self.num_actions)

        # Initialize weights
        self.weights = nn.Parameter(torch.ones(board_width * board_height) / (board_width * board_height))

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy gradient layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = self.act_fc1(x_act)
        x_act = F.log_softmax(x_act, dim=1)  # output about log probability of each action

        # action policy layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)

        # Ensure weights sum to 1
        weights = F.softmax(self.weights, dim=0)
        x_val = torch.sum(weights * x_val, dim=1, keepdim=True)

        return x_act, x_val


class QRAAC(nn.Module):  # Quantile Regression action value actor critic
    """policy-value network module"""

    def __init__(self, board_width, board_height, quantiles):
        super(QRAAC, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height
        self.N = quantiles

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # policy gradient layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # action value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, self.num_actions * self.N)

        # Initialize weights
        self.weights = nn.Parameter(torch.ones(board_width * board_height) / (board_width * board_height))

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # policy gradient layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = self.act_fc1(x_act)
        x_act = F.log_softmax(x_act, dim=1)  # output about log probability of each action

        # action policy layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)
        x_val = x_val.view(-1, self.num_actions, self.N)

        # Calculate the mean of the quantile values for each action.
        x_val = x_val.mean(dim=2)

        # Ensure weights sum to 1
        weights = F.softmax(self.weights, dim=0)
        x_val = torch.sum(weights * x_val, dim=1, keepdim=True)

        return x_act, x_val


class EQRAC(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height, N, threshold=0.1):
        super(EQRAC, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.N = N
        self.threshold = threshold

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)

        if torch.cuda.is_available():  # Windows
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # Mac OS
            device = torch.device("mps")
        else:  # CPU
            device = torch.device("cpu")

        self.device = device
        self.init_state_value_layers()


    def forward(self, state_input):
        # Common layers processing
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Action value processing
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # State value processing
        x_value = self.compute_state_values(x)

        while self.adjust_N(x_value):
            self.init_state_value_layers()
            x_value = self.compute_state_values(x)  # Recompute state values with updated layers
        x_value = torch.mean(x_value, dim=1, keepdim=True)

        return x_act, x_value

    def compute_state_values(self, x):
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_value = F.tanh(self.val_fc2(x_val))
        return x_value

    def adjust_N(self, x_value):
        """Adjusts N based on the top two state values if their difference is below the threshold."""
        top_values, _ = torch.topk(x_value, 2, dim=1)  # Get top 2 values
        diff = (top_values[:, 0] - top_values[:, 1]).abs()
        print(diff)

        if diff >= self.threshold and self.N < 64:
            return False
        else:
            self.N = min(64, self.N * 2)
            return True





class PolicyValueNet:
    """policy-value network """

    def __init__(self, board_width, board_height, quantiles=None,
                 model_file=None, rl_model=None):

        if torch.cuda.is_available():            # Windows
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # Mac OS
            device = torch.device("mps")
        else:                                    # CPU
            device = torch.device("cpu")
        self.use_gpu = device

        self.board_width = board_width  # 9
        self.board_height = board_height  # 4
        self.l2_const = 1e-4  # coef of l2 penalty
        self.quantiles = quantiles
        self.rl_model = rl_model
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.kappa = 1.0
        self.N = quantiles
        self.quantile_tau = torch.FloatTensor([i / self.N for i in range(1, self.N + 1)]).to(device)


        # DQN, QRDQN, AC, AAC, QRAC, QRAAC, EQRDQN, DQRAAC
        # the policy value net module
        if rl_model == "DQN": # [TODO]
            self.policy_value_net = DQN(board_width, board_height).to(device)
        elif rl_model == "QRDQN":  # [TODO]
            self.policy_value_net = QRDQN(board_width, board_height).to(device)
        elif rl_model == "AC":
            self.policy_value_net = AC(board_width, board_height).to(device)
        elif rl_model == "AAC":
            self.policy_value_net = AAC(board_width, board_height).to(device)
        elif rl_model == "QRAC":
            self.policy_value_net = QRAC(board_width, board_height, quantiles).to(device)
        elif rl_model == "QRAAC":
            self.policy_value_net = QRAAC(board_width, board_height, quantiles).to(device)
        elif rl_model == "EQRAAC":
            self.policy_value_net = EQRAAC(board_width, board_height, quantiles).to(device)
        elif rl_model == "EQRAAC":
            self.policy_value_net = EQRAAC(board_width, board_height, quantiles).to(device)
        else:
            assert print("error")

        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)
        if model_file:
            state_dict = torch.load(model_file, map_location=device)
            self.policy_value_net.load_state_dict(state_dict)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        device = self.use_gpu
        state_batch = np.array(state_batch)
        state_batch = torch.from_numpy(state_batch).float().to(device)
        if self.rl_model == "DQN" or self.rl_model == "QRDQN":
            log_act_probs, value = self.policy_value_net(state_batch, self.epsilon_min)
        else:
            log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.cpu().detach().numpy())
        return act_probs, value.cpu().detach().numpy()

    def policy_value_fn(self, env, k=None):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        available = np.where(env.state_[3].flatten() == 0)[0]
        k = k   # [Todo] 여기 K는 나중에 EQRAC였나 거기서 비교해서 Quantile k값을 늘려준다 그거임
        current_state = np.ascontiguousarray(env.state_.reshape(-1, 5, env.state_.shape[1], env.state_.shape[2]))
        device = self.use_gpu
        current_state = torch.from_numpy(current_state).float().to(device)
        # if self.rl_model == "DQN" or self.rl_model == "QRDQN":
        #     if self.epsilon > self.epsilon_min:
        #         self.epsilon *= self.epsilon_decay
        #     log_act_probs, value = self.policy_value_net(current_state, self.epsilon)
        # else:
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.data.cpu().detach().numpy().flatten())
        act_probs = zip(available, act_probs[available])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr, quantiles=None):
        """perform a training step"""
        device = self.use_gpu
        state_batch_np = np.array(state_batch)
        mcts_probs_np = np.array(mcts_probs)
        winner_batch_np = np.array(winner_batch)
        # numpy array to tensor
        state_batch = torch.tensor(state_batch_np, dtype=torch.float).to(device)
        mcts_probs = torch.tensor(mcts_probs_np, dtype=torch.float).to(device)
        winner_batch = torch.tensor(winner_batch_np, dtype=torch.float).to(device)

        set_learning_rate(self.optimizer, lr)
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer

        if self.rl_model == "DQN":
            loss = F.mse_loss(value.view(-1), winner_batch)

        elif self.rl_model == "QRDQN":
            taus = torch.linspace(0, 1, self.quantiles.size(2) + 1)[1:]  # TODO quantile 받아서 보내기
            loss = quantile_huber_loss(value.view(-1), winner_batch, taus)

        elif self.rl_model == "QRAAC":
            loss = F.mse_loss(value.view(-1), winner_batch)
            huber_loss = torch.where(loss.abs() <= self.kappa, 0.5 * loss.pow(2),
                                     self.kappa * (loss.abs() - 0.5 * self.kappa))

            # Quantile Loss: 각 quantile 위치에 따라 Huber Loss에 가중치를 적용하여 Quantile Loss를 계산
            quantile_loss = abs(self.quantile_tau - (loss.detach() < 0).float()) * huber_loss / 1.0
            quantile_loss = quantile_loss.sum(dim=1).mean(dim=1)

            value_loss = quantile_loss.mean()
            policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
            loss = value_loss + policy_loss
        else:
            value_loss = F.mse_loss(value.view(-1), winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
            loss = value_loss + policy_loss

        # when call backward, the grad will accumulate. so zero grad before backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.rl_model == "DQN" or self.rl_model == "QRDQN":
            return loss.item()
        else:
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
            )
            return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    # def load_model(self, model_file):
    #    """ load model params from file """
    #    state_dict = torch.load(model_file)
    #    self.policy_value_net.load_state_dict(state_dict)
    #    return state_dict

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        # Ensure that the directory exists before saving the file
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        torch.save(net_params, model_file)
