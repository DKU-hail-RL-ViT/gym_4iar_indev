import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

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
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


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
        # action policy layers (previous state value)
        self.act_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.act_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.act_fc2 = nn.Linear(64, self.num_actions)

    def forward(self, state_input, sensible_moves): # [TODO] 여기 action masking 한거 input으로 넣어줘야함
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 2 * self.board_width * self.board_height)
        x_act = F.relu(self.act_fc1(x_act))
        x_act = self.act_fc2(x_act)

        x_act = F.log_softmax(x_act, dim=1)

        # masking action
        mask = torch.ones((1, 36), dtype=torch.float32)
        all_indices = set(range(mask.size(1)))
        sensible_moves_set = set(sensible_moves)
        non_sensible_moves = list(all_indices - sensible_moves_set)
        mask[:, non_sensible_moves] = -torch.inf
        x_act = torch.where(x_act == float('inf'), -torch.inf, x_act)  # [TODO] DQN에서 이렇게 해도 될지 모르겠음

        return x_act


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
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1) # output about log probability of each action

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class AAC(nn.Module):  # action value actor critic
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super(AAC, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # common layers
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # policy gradient layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # action policy layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, board_width * board_height)

    def forward(self, state_input, k):
        # sensible_moves = [0,1,2,3,4,5,6,7,9,10,13,14,19,20,25,34]
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy gradient layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)

        x_act = self.act_fc1(x_act)
        x_act = F.log_softmax(x_act, dim=1) # output about log probability of each action

        # action policy layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))

        x_val = F.tanh(self.val_fc2(x_val))

        # # # sensible_moves 인덱스에 포함되지 않은 인덱스를 찾아 -torch.inf로 설정
        # x_val *= mask
        x_val = torch.where(x_val == float('inf'), -torch.inf, x_val)

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
        self.val_fc2 = nn.Linear(64, N)

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

    def init_state_value_layers(self):
        """ Initialize or update the state value output layer based on current self.N """
        self.val_fc2 = nn.Linear(64, self.N).to(self.device)

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
                 model_file=None, rl_model="AC"):

        self.board_width = board_width  # 9
        self.board_height = board_height  # 4
        self.l2_const = 1e-4  # coef of l2 penalty
        self.quantiles = quantiles
        self.rl_model = rl_model
        self.gamma = 0.99

        if torch.cuda.is_available():            # Windows
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # Mac OS
            device = torch.device("mps")
        else:                                    # CPU
            device = torch.device("cpu")
        self.use_gpu = device

        # the policy value net module
        if rl_model == "DQN": # [TODO]
            self.policy_value_net = DQN(board_width, board_height).to(device)
        elif rl_model == "QRDQN": # [TODO]
            self.policy_value_net = QRDQN(board_width, board_height).to(device)
        elif rl_model == "AC":
            self.policy_value_net = AC(board_width, board_height).to(device)
        elif rl_model == "AAC":
            self.policy_value_net = AAC(board_width, board_height).to(device)
        elif rl_model == "QRAC":
            self.policy_value_net = AAC(board_width, board_height, quantiles).to(device)
        elif rl_model == "EQRAC":
            self.policy_value_net = EQRAC(board_width, board_height, quantiles).to(device)

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
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.cpu().detach().numpy())
        return act_probs, value.cpu().detach().numpy()

    def policy_value_fn(self, board, sensible_move, k, value=None):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        available = sensible_move
        k = k   # [Todo] 여기 K는 나중에 AAC였나 EQRAC였나 거기서 비교해서 Quantiole k값을 늘려준다 그거임
        current_state = np.ascontiguousarray(board.reshape(-1, 5, board.shape[1], board.shape[2]))
        device = self.use_gpu

        current_state = torch.from_numpy(current_state).float().to(device)
        if self.rl_model == "DQN" or self.rl_model == "QRDQN":
            # [TODO] 여기도 DQN만 만져놓은거라 더 수정되어야할것 DQN이니까 action probability만 필요해서 value뺌  QRDQN도 여기 들어가야하지 않을까
            log_act_probs = self.policy_value_net(current_state, sensible_move)
        else:
            log_act_probs, value = self.policy_value_net(current_state, sensible_move)
        act_probs = np.exp(log_act_probs.cpu().detach().numpy().flatten())
        act_probs = zip(available, act_probs[available])
        # value = value.data[0][0]
        if self.rl_model == "DQN" or self.rl_model == "QRDQN":
            return act_probs
        else:
            return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
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

        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss

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
