import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os

epsilon = 0.1


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calculate_quantile_regression_loss(huber_loss, quantile_tau, value_loss):
    return (abs(quantile_tau - (value_loss.detach() < 0).float()) * huber_loss / 1.0).unsqueeze(-1).sum(
                dim=1).mean()


# def interpolate_quantiles(old_quantiles, new_quantile_count):
#     # 기존 quantile의 개수
#     old_quantile_count = len(old_quantiles)
#
#     # 새로운 quantile을 위한 위치 계산
#     new_quantile_positions = np.linspace(0, 1, new_quantile_count + 1)
#     old_quantile_positions = np.linspace(0, 1, old_quantile_count + 1)
#
#     # 새로운 quantile 값 계산 (보간 사용)
#     new_quantiles = np.interp(new_quantile_positions[1:-1], old_quantile_positions[1:-1], old_quantiles)
#
#     return new_quantiles


# 예시: quantile 2개에서 4개로 확장
# q4_values = interpolate_quantiles(q2_values, 4)



class DQN(nn.Module):
    """value network module"""

    def __init__(self, board_width, board_height):
        super(DQN, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action value layers
        self.act_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.act_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.act_fc2 = nn.Linear(64, self.num_actions)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action value layers
        x_val = F.relu(self.act_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.act_fc1(x_val))
        x_val = self.act_fc2(x_val)

        # action value to policy
        x_act = F.log_softmax(x_val, dim=1)

        return x_act, x_val


class QRDQN(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height, quantiles):
        super(QRDQN, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height
        self.N = quantiles

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action value layers (previous state value)
        self.act_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.act_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.act_fc2 = nn.Linear(64, self.num_actions * self.N)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action value layers
        x_val = F.relu(self.act_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.act_fc1(x_val))
        x_val = self.act_fc2(x_val)

        # action policy layers
        x_act = F.log_softmax(x_val, dim=1)

        x_val = x_val.view(-1, self.num_actions, self.N)
        x_val = x_val.mean(dim=2)

        return x_act, x_val


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
        # policy gradient layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy gradient layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class QRAC(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height, quantiles):
        super(QRAC, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.N = quantiles

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # policy gradient layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, self.N)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy gradient layers
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


class QAC(nn.Module):  # action value actor critic
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super(QAC, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # policy gradient layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # action value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, self.num_actions)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy gradient layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = self.act_fc1(x_act)
        x_act = F.log_softmax(x_act, dim=1)

        # action value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)

        return x_act, x_val


class QRQAC(nn.Module):  # Quantile Regression action value actor critic
    """policy-value network module"""

    def __init__(self, board_width, board_height, quantiles):
        super(QRQAC, self).__init__()
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
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # action value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, self.num_actions * self.N)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy gradient layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = self.act_fc1(x_act)
        x_act = F.log_softmax(x_act, dim=1)

        # action value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)
        x_val = x_val.view(-1, self.num_actions, self.N)

        # Calculate the mean of the quantile values for each action.
        x_val = x_val.mean(dim=2)

        return x_act, x_val


class EQRDQN(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height, device):
        super(EQRDQN, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height
        self.device = device
        # self.N = quantiles

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action policy layers
        self.dqn_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.dqn_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # action value layers (previous state value)
        self.act_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.act_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.act_fc2 = nn.Linear(64, self.num_actions * self.N)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action policy layers
        x_act = F.relu(self.dqn_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.dqn_fc1(x_act), dim=1)  # output about log probability of each action

        # action value layers
        x_val = F.relu(self.act_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.act_fc1(x_val))
        x_val = self.act_fc2(x_val)

        x_val = x_val.view(-1, self.num_actions, self.N)
        x_val = x_val.mean(dim=2)

        return x_act, x_val


class EQRQAC(nn.Module):  # Efficient Quantile Regression action value actor critic
    """policy-value network module"""

    def __init__(self, board_width, board_height, device):
        super(EQRQAC, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.num_actions = board_width * board_height
        self.device = device

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # policy gradient layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # action value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        # self.val_fc2 = nn.Linear(64, self.num_actions * self.N)

    def update_quantiles(self, quantiles):
        self.val_fc2 = nn.Linear(64, self.num_actions * quantiles).to(self.device)

    def forward(self, state_input, quantiles):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy gradient layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = self.act_fc1(x_act)
        x_act = F.log_softmax(x_act, dim=1)

        # action value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))

        # Update and use dynamic val_fc based on quantiles
        self.update_quantiles(quantiles)
        x_val = self.val_fc2(x_val)
        x_val = x_val.view(-1, self.num_actions, quantiles)
        print(quantiles, "network 안에서 찍은 quantile 개수")

        # sh = x_val.mean(dim=1).flatten()
        # print(sh)

        # Calculate the mean of the quantile values for each action.
        x_val = x_val.mean(dim=2).flatten()

        return x_act, x_val


class PolicyValueNet:
    """policy-value network """

    def __init__(self, board_width, board_height, quantiles=None,
                 model_file=None, rl_model=None):

        if torch.cuda.is_available():  # Windows
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # Mac OS
            self.device = torch.device("mps")
        else:  # CPU
            self.device = torch.device("cpu")

        self.board_width = board_width  # 9
        self.board_height = board_height  # 4
        self.l2_const = 1e-4  # coef of l2 penalty
        self.rl_model = rl_model
        self.gamma = 0.99
        self.kappa = 1.0
        self.N = quantiles   # [TODO] 여기서도 efficient search 때는 이렇게 initilize해주면 안될 거 같은데
        self.quantile_tau = torch.FloatTensor([i / self.N for i in range(1, self.N + 1)]).to(self.device)

        # DQN, QRDQN, AC, QAC, QRAC, QRQAC, EQRDQN, EQRQAC
        if rl_model == "DQN":
            self.policy_value_net = DQN(board_width, board_height).to(self.device)
        elif rl_model == "QRDQN":
            self.policy_value_net = QRDQN(board_width, board_height, quantiles).to(self.device)
        elif rl_model == "AC":
            self.policy_value_net = AC(board_width, board_height).to(self.device)
        elif rl_model == "QAC":
            self.policy_value_net = QAC(board_width, board_height).to(self.device)
        elif rl_model == "QRAC":
            self.policy_value_net = QRAC(board_width, board_height, quantiles).to(self.device)
        elif rl_model == "QRQAC":
            self.policy_value_net = QRQAC(board_width, board_height, quantiles).to(self.device)
        elif rl_model == "EQRDQN":
            self.policy_value_net = EQRDQN(board_width, board_height, self.device).to(self.device)
        elif rl_model == "EQRQAC":
            self.policy_value_net = EQRQAC(board_width, board_height, self.device).to(self.device)
        else:
            assert print("error")

        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)
        if model_file:
            state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
            self.policy_value_net.load_state_dict(state_dict)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        # state_batch = torch.from_numpy(state_batch).float().to(device)
        # log_act_probs, value = self.policy_value_net(state_batch)
        # act_probs = np.exp(log_act_probs.cpu().detach().numpy())
        # return act_probs, value.cpu().detach().numpy()
        state_batch = np.array(state_batch)
        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.rl_model == "EQRDQN" or self.rl_model == "EQRQAC":
                log_act_probs, value = self.policy_value_net(state_batch, self.N)
            else:
                log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = torch.exp(log_act_probs).cpu().numpy()
            value = value.cpu().numpy()

            if self.rl_model == "DQN": # TODO QRDQN도 해야할 수도
                value_ = torch.tensor(value)
                value_, _ = torch.max(value_, dim=1)
                value = value_.unsqueeze(1)

        return act_probs, value

    def policy_value_fn(self, env, k=None):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        available = np.nonzero(env.state_[3].flatten() == 0)[0]
        current_state = np.ascontiguousarray(env.state_.reshape(-1, 5, env.state_.shape[1], env.state_.shape[2]))
        current_state = torch.from_numpy(current_state).float().to(self.device)

        with torch.no_grad():
            if self.rl_model == "EQRDQN" or self.rl_model == "EQRQAC":
                self.N = 2 ** k
                print(self.N, "network 들어가기 전에 quantile 개수")
                log_act_probs, value = self.policy_value_net(current_state, self.N)

            else:
                log_act_probs, value = self.policy_value_net(current_state)
            act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()

        return available, act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        state_batch_np = np.array(state_batch)
        mcts_probs_np = np.array(mcts_probs)
        winner_batch_np = np.array(winner_batch)

        # numpy array to tensor
        state_batch = torch.tensor(state_batch_np, dtype=torch.float32, device=self.device)
        mcts_probs = torch.tensor(mcts_probs_np, dtype=torch.float32, device=self.device)
        winner_batch = torch.tensor(winner_batch_np, dtype=torch.float32, device=self.device)

        set_learning_rate(self.optimizer, lr)
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer

        if self.rl_model == "DQN":
            value_ = value.clone().detach().to(self.device).requires_grad_(True)
            value_, _ = torch.max(value_, dim=1)
            value = value_.unsqueeze(1)
            loss = F.mse_loss(value.view(-1), winner_batch)

        elif self.rl_model in ["QRDQN", "QRQAC", "EQRDQN", "EQRQAC"]:
            value_loss = F.mse_loss(value.view(-1), winner_batch)
            huber_loss = torch.where(value_loss.abs() <= self.kappa, 0.5 * value_loss.pow(2),
                                     self.kappa * (value_loss.abs() - 0.5 * self.kappa))
            quantile_regression_loss = calculate_quantile_regression_loss(huber_loss,
                                                                          self.quantile_tau,
                                                                          value_loss)
            if self.rl_model == "QRDQN" or self.rl_model == "EQRDQN":
                loss = quantile_regression_loss

            elif self.rl_model == "QRQAC" or self.rl_model == "EQRQAC":  # QRQAC
                policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
                loss = quantile_regression_loss + policy_loss

        elif self.rl_model in ["AC", "QRAC", "QAC"]:
            if self.rl_model == "QAC":
                value_ = value.clone().detach().to(self.device).requires_grad_(True)
                value_, _ = torch.max(value_, dim=1)
                value = value_.unsqueeze(1)

            value_loss = F.mse_loss(value.view(-1), winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
            loss = value_loss + policy_loss

        else:
            assert "No define"

        # when call backward, the grad will accumulate. so zero grad before backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
